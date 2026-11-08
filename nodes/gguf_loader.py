# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

GGUF LLM Loader Module.

Provides a specialized, thread-safe node for loading Large Language Models
via the GGUF format using llama.cpp. Compliant with Weave V6 specifications.
"""

import gc
import os
import threading
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSlider,
    QSpinBox,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.node.threaded import ThreadedManualNode
from weave.registry import register_node
from weave.logger import get_logger
from weave.node import VerticalSizePolicy

from weave.library.sync_tool_button.widgets.sync_tool_button import SyncToolButton
from weave.widgets.sync_button import SyncButton
from weave.widgets.path_picker import PathPickerWidget
from weave.widgets.separator_line import SeparatorLine

log = get_logger("GGUFLNode")

# Shared silk port types (registered idempotently on import).
from .silk_ports import MODEL_HANDLE_TYPE  # noqa: F401

# Qt-free pool + dependency guards live in functions/ so the GraphEngine and
# headless tests can share them. The pool now runs a background
# ``llama_cpp.server`` process, which needs the [server] extra as well.
from ..functions.model_pool import (
    GGUFModelPool, LLAMA_CPP_AVAILABLE, LLAMA_SERVER_AVAILABLE,
    server_missing_deps_message,
)
from ..functions.gguf_meta import read_gguf_meta, read_gguf_meta_fallback
from ..functions.graph_engine import apply_inference_settings
from ..functions.presets import GGUFModelStore

if not LLAMA_CPP_AVAILABLE:
    log.error("llama-cpp-python NOT found. GGUF Loader will be non-functional.")
elif not LLAMA_SERVER_AVAILABLE:
    log.error(f"GGUF Loader non-functional: {server_missing_deps_message()}")

# ==============================================================================
# GGUF Loader Node (V6)
# ==============================================================================

@register_node
class GGUFLNode(ThreadedManualNode):
    """Loads GGUF models into a thread-safe pool via llama.cpp on a worker thread."""
    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    node_class:           ClassVar[str]                 = "Silk AI"
    node_subclass:        ClassVar[str]                 = "Loaders"
    node_name:            ClassVar[Optional[str]]       = "GGUF Loader"
    node_description:     ClassVar[Optional[str]]       = "Loads GGUF models via llama.cpp into an isolated KV pool."
    node_tags:            ClassVar[Optional[List[str]]] = ["llama", "llm", "gguf", "inference", "pool"]
    node_icon:            ClassVar[Optional[str]]       = "package"
    vertical_size_policy: ClassVar[VerticalSizePolicy]  = VerticalSizePolicy.FIT

    # Base api is 1; 2 adds the optional `inference_settings` inlet whose
    # dict rides out on the model wire (handle["gen_params"]). Additive:
    # older states simply leave it unwired and sample loader defaults.
    # 3 adds the vision-projector switch + file (`enable_mmproj` /
    # `mmproj_path`): detection fills the picker, the server gets
    # `--mmproj`, the handle gains `multimodal`. Additive like 2.
    # 4 adds the MTP switch + draft count (`enable_mtp` /
    # `mtp_draft_tokens`): the server gets `spec_type=draft-mtp`.
    # Additive like 2 and 3.
    # 5 registers the three companion sliders so the mirror panel syncs
    # them through their own bindings. Values ride the spinners; the
    # slider ports are display state only.
    node_state_api = 5

    @classmethod
    def migrate_state(cls, state: Dict[str, Any],
                      from_version: int) -> Dict[str, Any]:
        """api 1 → 2: the inference inlet defaults to unwired.
        api 2 → 3: projector detection defaults to off.
        api 3 → 4: MTP defaults to off.
        api 4 → 5: slider bindings are new; older states restore the
        spinners and the sliders re-seat from them."""
        return state

    # Emitted from the probe worker thread; queued back onto the GUI thread.
    _probe_ready = Signal(str, object)

    #: The pool snapshot the worker took, handed to the GUI thread by
    #: `on_evaluate_finished`. Declared because it is written on one
    #: thread and read on another, and a `hasattr` probe is not a type.
    _pending_pool_info: Optional[dict] = None

    def __init__(self, title: str = "GGUF Loader", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # 1. Add Graph Ports
        self.add_input("model_path", datatype="filepath")
        self.add_input("prompt_cache", datatype="filepath")
        # Optional sampler overrides from an Inference Settings node. They
        # ride out on the model wire (handle["gen_params"]) so whatever
        # consumes the handle -- an Agent, local or remote -- samples as
        # configured here, at the model, instead of at the run.
        self.add_input("inference_settings", datatype="dict")
        self.add_output("model_obj", datatype="model_handle")
        self.add_output("pool_info", datatype="dict")  # live pool stats

        self.model_pool: Optional[GGUFModelPool] = None
        self._model_store = GGUFModelStore()
        # Model path the per-model memory was last applied to: restores
        # run once per file so later edits are the user's own.
        self._restored_for: Optional[str] = None
        # path -> {"size_gb", "max_ctx", "layers"}; covers stat() AND header
        # parse so a cached path costs zero filesystem I/O on the GUI thread.
        self._probe_cache: Dict[str, Dict[str, Any]] = {}
        # Last model path the projector detection attempted: a cleared
        # picker stays cleared (manual text-only wins) and repeat edits
        # cost no extra directory listing.
        self._mmproj_probed_for: Optional[str] = None

        # 2. Build Layout + WidgetCore
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        # 3. Create Widgets, Add to Layout, Register
        self.path_picker = PathPickerWidget(mode="file", filters=["GGUF Models (*.gguf)", "All Files (*)"])
        self.path_picker.setToolTip("Select the .gguf model file from your file system. Weights will be shared across all pool instances.")
        form.addRow("Model Path:", self.path_picker)
        self._widget_core.register_widget(
            "model_path", self.path_picker, role=PortRole.BIDIRECTIONAL,
            datatype="filepath", default="", add_to_layout=False
        )

        self._combo_mode = QComboBox()
        self._combo_mode.addItem("Simple", userData="simple")
        self._combo_mode.addItem("Advanced", userData="advanced")
        self._combo_mode.setToolTip(
            "Simple shows only the load essentials; Advanced reveals "
            "sampler, cache and quantization tuning.")
        form.addRow("Settings Mode:", self._combo_mode)
        self._widget_core.register_widget("mode", self._combo_mode, role=PortRole.INTERNAL, datatype="str", default="simple", add_to_layout=False)

        self.spin_gpu = QSpinBox()
        self.spin_gpu.setRange(-1, 256)
        self.spin_gpu.setValue(-1)
        self.spin_gpu.setToolTip(
            "Layers offloaded to the GPU. -1 = as many as fit (fastest); "
            "0 = CPU only; N = first N layers on GPU.")
        gpu_row = QHBoxLayout()
        self.slider_gpu = QSlider(Qt.Horizontal)
        self.slider_gpu.setRange(0, self.spin_gpu.maximum())
        self.slider_gpu.setToolTip(
            "Same as the spinner: drag to reduce offloaded layers. "
            "Full-right is everything (-1, auto); type -1 to return. "
            "The top end follows the model's layer count after probing.")
        self._link_slider(self.slider_gpu, self.spin_gpu)
        gpu_row.addWidget(self.slider_gpu, stretch=1)
        gpu_row.addWidget(self.spin_gpu)
        form.addRow("GPU Layers:", gpu_row)
        self._widget_core.register_widget(
            "gpu_layers_slider", self.slider_gpu, role=PortRole.INTERNAL,
            datatype="int", default=self.slider_gpu.value(),
            add_to_layout=False,
        )
        self._widget_core.register_widget("gpu_layers", self.spin_gpu, role=PortRole.INTERNAL, datatype="int", default=-1, add_to_layout=False)

        self.spin_ctx = QSpinBox()
        self.spin_ctx.setRange(512, 1048576)
        self.spin_ctx.setValue(4096)
        self.spin_ctx.setToolTip(
            "Context window in tokens (prompt + answer). Larger holds "
            "longer chats but needs proportionally more VRAM.")
        ctx_row = QHBoxLayout()
        self.slider_ctx = QSlider(Qt.Horizontal)
        self.slider_ctx.setRange(self.spin_ctx.minimum(),
                                 self.spin_ctx.maximum())
        self.slider_ctx.setToolTip(
            "Same as the spinner: drag to shrink the context. The top end "
            "follows the model's trained context after probing.")
        self._link_slider(self.slider_ctx, self.spin_ctx)
        ctx_row.addWidget(self.slider_ctx, stretch=1)
        ctx_row.addWidget(self.spin_ctx)
        form.addRow("Context:", ctx_row)
        self._widget_core.register_widget(
            "n_ctx_slider", self.slider_ctx, role=PortRole.INTERNAL,
            datatype="int", default=self.slider_ctx.value(),
            add_to_layout=False,
        )
        self._widget_core.register_widget("n_ctx", self.spin_ctx, role=PortRole.INTERNAL, datatype="int", default=4096, add_to_layout=False)

        self.spin_pool_size = QSpinBox()
        self.spin_pool_size.setRange(1, 16)
        self.spin_pool_size.setValue(2)
        self.spin_pool_size.setToolTip(
            "Advisory only: the model now runs as a single shared server process "
            "(weights load once). Concurrent agents share it; requests queue."
        )
        form.addRow("Pool Size:", self.spin_pool_size)
        self._widget_core.register_widget("pool_size", self.spin_pool_size, role=PortRole.INTERNAL, datatype="int", default=2, add_to_layout=False)

        # Advanced Widgets
        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(-1, 128)
        self.spin_threads.setValue(-1)
        self.spin_threads.setToolTip(
            "CPU threads for prompt processing. -1 lets llama.cpp choose; "
            "pin only when sharing the box.")
        threads_row = QHBoxLayout()
        self.slider_threads = QSlider(Qt.Horizontal)
        self.slider_threads.setRange(1, max(1, os.cpu_count() or 1))
        self.slider_threads.setToolTip(
            "Same as the spinner. Full-right is everything (-1, auto); "
            "the top end is this machine's CPU count.")
        self._link_slider(self.slider_threads, self.spin_threads)
        threads_row.addWidget(self.slider_threads, stretch=1)
        threads_row.addWidget(self.spin_threads)
        form.addRow("CPU Threads:", threads_row)
        self._widget_core.register_widget(
            "n_threads_slider", self.slider_threads, role=PortRole.INTERNAL,
            datatype="int", default=self.slider_threads.value(),
            add_to_layout=False,
        )
        self._widget_core.register_widget("n_threads", self.spin_threads, role=PortRole.INTERNAL, datatype="int", default=-1, add_to_layout=False)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 2147483647)
        self.spin_seed.setValue(-1)
        self.spin_seed.setSpecialValueText("Random (-1)")
        self.spin_seed.setToolTip(
            "Random seed. -1 rolls fresh every run; set a number to replay "
            "a generation exactly.")
        form.addRow("Seed:", self.spin_seed)
        self._widget_core.register_widget("seed", self.spin_seed, role=PortRole.INTERNAL, datatype="int", default=-1, add_to_layout=False)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 8192)
        self.spin_batch.setValue(512)
        self.spin_batch.setToolTip(
            "Prompt-processing batch size. Larger prefills faster on GPU "
            "until memory runs out.")
        form.addRow("Batch Size:", self.spin_batch)
        self._widget_core.register_widget("n_batch", self.spin_batch, role=PortRole.INTERNAL, datatype="int", default=512, add_to_layout=False)

        # Embedding mode. A server started without this refuses
        # /v1/embeddings, which is the right default for a chat model:
        # embeddings are the vector half of memory (§17) and want their own,
        # much smaller, model file. Load that one in a second GGUF Loader
        # and wire it to the ToolSet's embedding_model port.
        self.chk_embedding = QCheckBox()
        self.chk_embedding.setToolTip(
            "Serve embeddings from this model instead of chat. Wire it to a "
            "Silk ToolSet's embedding_model port to give recall vector "
            "search. A chat model loaded this way is not useful for chat."
        )
        form.addRow("Embedding Model:", self.chk_embedding)
        self._widget_core.register_widget(
            "embedding", self.chk_embedding, role=PortRole.INTERNAL,
            datatype="bool", default=False, add_to_layout=False,
        )

        # Vision projector (mmproj): auto-detected beside the model when
        # on, or picked/wired explicitly. The server flag is off unless
        # both the switch and a readable file are present.
        self.chk_mmproj = QCheckBox()
        self.chk_mmproj.setToolTip(
            "Look for a vision projector (mmproj*.gguf) next to the model "
            "and load it, so the server accepts images. Off means text "
            "only, even when the file is there.")
        form.addRow("Vision Projector:", self.chk_mmproj)
        self._widget_core.register_widget(
            "enable_mmproj", self.chk_mmproj, role=PortRole.INTERNAL,
            datatype="bool", default=False, add_to_layout=False,
        )

        self.path_picker_mmproj = PathPickerWidget(mode="file", filters=["GGUF Projectors (mmproj*.gguf)", "All Files (*)"])
        self.path_picker_mmproj.setToolTip(
            "Multimodal projector file. Auto-filled from the model's "
            "folder when detection is on; pick one to override.")
        form.addRow("Projector File:", self.path_picker_mmproj)
        self._widget_core.register_widget(
            "mmproj_path", self.path_picker_mmproj,
            role=PortRole.BIDIRECTIONAL, datatype="filepath", default="",
            add_to_layout=False,
        )
        self.add_input("mmproj_path", datatype="filepath")

        # Multi-token prediction: the model's built-in draft heads draft
        # several tokens per step (~1.5-2x on easy text). Needs MTP weights
        # and a llama.cpp stack new enough to know `spec_type`; older
        # builds ignore the keys and run one token per step.
        self.chk_mtp = QCheckBox()
        self.chk_mtp.setToolTip(
            "Draft several tokens per step with the model's built-in "
            "prediction heads. Needs MTP weights and a current llama.cpp "
            "stack; silently off on older builds.")
        form.addRow("MTP:", self.chk_mtp)
        self._widget_core.register_widget(
            "enable_mtp", self.chk_mtp, role=PortRole.INTERNAL,
            datatype="bool", default=False, add_to_layout=False,
        )

        self.spin_mtp_tokens = QSpinBox()
        self.spin_mtp_tokens.setRange(1, 8)
        self.spin_mtp_tokens.setValue(2)
        self.spin_mtp_tokens.setToolTip(
            "How many tokens the MTP heads draft per step. 2 is the "
            "balanced default; more is faster on easy text, slower when "
            "drafts miss.")
        form.addRow("MTP Draft Tokens:", self.spin_mtp_tokens)
        self._widget_core.register_widget(
            "mtp_draft_tokens", self.spin_mtp_tokens, role=PortRole.INTERNAL,
            datatype="int", default=2, add_to_layout=False,
        )

        self.chk_flash = QCheckBox()
        self.chk_flash.setToolTip(
            "Flash Attention: less VRAM on long contexts for supported "
            "GPUs and quants; can change output slightly.")
        form.addRow("Flash Attention:", self.chk_flash)
        self._widget_core.register_widget("flash_attn", self.chk_flash, role=PortRole.INTERNAL, datatype="bool", default=False, add_to_layout=False)

        self.chk_mmap = QCheckBox()
        self.chk_mmap.setChecked(True)
        self.chk_mmap.setToolTip(
            "Memory-map the weights instead of loading them into RAM. "
            "Turn off only for mounts that misbehave.")
        form.addRow("Use mmap():", self.chk_mmap)
        self._widget_core.register_widget("use_mmap", self.chk_mmap, role=PortRole.INTERNAL, datatype="bool", default=True, add_to_layout=False)

        self._combo_type_k = QComboBox()
        self._combo_type_k.addItems(["f16", "bf16", "q8_0", "q4_0"])
        self._combo_type_k.setToolTip(
            "KV-cache quantization (keys). f16 is exact; q8_0 and q4_0 "
            "trade a little recall accuracy for VRAM on long contexts.")
        form.addRow("KV Cache Type K:", self._combo_type_k)
        self._widget_core.register_widget("type_k", self._combo_type_k, role=PortRole.INTERNAL, datatype="str", default="f16", add_to_layout=False)

        self._combo_type_v = QComboBox()
        self._combo_type_v.addItems(["f16", "bf16", "q8_0", "q4_0"])
        self._combo_type_v.setToolTip(
            "KV-cache quantization (values). f16 is exact; q8_0 and q4_0 "
            "trade a little recall accuracy for VRAM on long contexts.")
        form.addRow("KV Cache Type V:", self._combo_type_v)
        self._widget_core.register_widget("type_v", self._combo_type_v, role=PortRole.INTERNAL, datatype="str", default="f16", add_to_layout=False)

        self.path_picker_prompt_cache = PathPickerWidget(mode="save_file", filters=["Cache Files (*.cache)", "All Files (*)"])
        self.path_picker_prompt_cache.setToolTip(
            "File holding the tokenized prompt cache, reused across runs "
            "so repeat prompts skip setup.")
        form.addRow("Prompt Cache File:", self.path_picker_prompt_cache)
        self._widget_core.register_widget(
            "prompt_cache", self.path_picker_prompt_cache, role=PortRole.BIDIRECTIONAL,
            datatype="filepath", default="", add_to_layout=False
        )

        self.chk_prompt_cache_all = QCheckBox()
        self.chk_prompt_cache_all.setToolTip(
            "Let every server slot reuse the prompt cache, not just the "
            "first.")
        form.addRow("Prompt Cache All:", self.chk_prompt_cache_all)
        self._widget_core.register_widget("prompt_cache_all", self.chk_prompt_cache_all, role=PortRole.INTERNAL, datatype="bool", default=False, add_to_layout=False)

        self.chk_prompt_cache_ro = QCheckBox()
        self.chk_prompt_cache_ro.setToolTip(
            "Never write to the prompt cache file: reuse what is there, "
            "keep it byte-identical.")
        form.addRow("Prompt Cache RO:", self.chk_prompt_cache_ro)
        self._widget_core.register_widget("prompt_cache_ro", self.chk_prompt_cache_ro, role=PortRole.INTERNAL, datatype="bool", default=False, add_to_layout=False)

        self.chk_clear_cache = QCheckBox()
        self.chk_clear_cache.setChecked(True)
        self.chk_clear_cache.setToolTip(
            "Wipe the KV cache when a session returns to the pool, so the "
            "next run starts with no leftover context.")
        form.addRow("Clear Cache on Return:", self.chk_clear_cache)
        self._widget_core.register_widget("clear_on_return", self.chk_clear_cache, role=PortRole.INTERNAL, datatype="bool", default=True, add_to_layout=False)

        # One server holds one resident context, so two conversations
        # running at once overwrite each other's prompt cache: measured
        # at 74.8% prefix reuse alone against 0.4% together, same work in
        # 1.1 s against 4.5 s (D47 mechanism A, G15). On by default --
        # the setting exists to turn the grouping *off*, for a graph that
        # would rather have arrival-order latency than the reuse.
        self.chk_affinity = QCheckBox()
        self.chk_affinity.setChecked(True)
        self.chk_affinity.setToolTip(
            "Pin repeat agents to the same server slot so cached prefixes "
            "stay hot. Off serves in arrival order instead.")
        form.addRow("Group Requests by Agent:", self.chk_affinity)
        self._widget_core.register_widget(
            "session_affinity", self.chk_affinity, role=PortRole.INTERNAL,
            datatype="bool", default=True, add_to_layout=False)

        self.spin_cram = QSpinBox()
        self.spin_cram.setRange(0, 131072)
        self.spin_cram.setToolTip(
            "MB of system RAM the KV cache may spill into when VRAM runs "
            "out. 0 keeps everything in VRAM.")
        form.addRow("Cache RAM (MB):", self.spin_cram)
        self._widget_core.register_widget("cram", self.spin_cram, role=PortRole.INTERNAL, datatype="int", default=0, add_to_layout=False)

        form.addRow(SeparatorLine())

        # V6: Display readout for status estimation
        self._label_status = QLabel("Select a model to estimate VRAM.")
        self._label_status.setEnabled(False)
        self._label_status.setWordWrap(True)
        self._label_status.setToolTip(
            "Measured estimate for the selected file: size, context and "
            "layer count after probing.")
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget("status", self._label_status, role=PortRole.DISPLAY, datatype="str", add_to_layout=False)

        # KV Cache usage progress bar + refresh button
        kv_row = QHBoxLayout()
        self._progress_kv = QProgressBar()
        self._progress_kv.setRange(0, 100)
        self._progress_kv.setValue(0)
        self._progress_kv.setFormat("KV Cache: %p%% filled")
        self._progress_kv.setEnabled(False)
        self._progress_kv.setToolTip(
            "How full the KV cache is right now. Refresh reads it live "
            "from the running pool.")
        kv_row.addWidget(self._progress_kv, stretch=1)

        self._btn_refresh_kv = SyncToolButton(
            initial_text="Refresh", show_label=False, dimensions=24,
        )
        self._btn_refresh_kv.set_tooltip("Refresh KV cache usage from the pool.")
        self._btn_refresh_kv.clicked.connect(self._on_refresh_kv)
        kv_row.addWidget(self._btn_refresh_kv)
        form.addRow("", kv_row)
        self._widget_core.register_widget(
            "kv_progress", self._progress_kv, role=PortRole.DISPLAY,
            datatype="int", add_to_layout=False,
        )

        # V6: Action Button Routing
        self.btn_load = SyncButton(initial_text="Load GGUF Pool",
                                   initial_icon="download")
        self.btn_load.set_tooltip(
            "Load (or reload) the model into the pool with these "
            "settings. Re-runs the spawn on every press.")
        self.btn_load.clicked.connect(self.execute)
        form.addRow("", self.btn_load)
        self._widget_core.register_widget("btn_load", self.btn_load, role=PortRole.INTERNAL, add_to_layout=False)

        # V6: Declarative UI Visibility
        self._widget_core.bind_visibility(
            trigger_port="mode",
            mapping={
                "advanced": [
                    "seed", "n_threads", "n_batch", "flash_attn", "use_mmap",
                    "type_k", "type_v", "prompt_cache", "prompt_cache_all",
                    "prompt_cache_ro", "clear_on_return", "session_affinity",
                    "cram", "enable_mtp", "mtp_draft_tokens"
                ]
            }
        )

        self._widget_core.value_changed.connect(self._on_value_changed)
        self._widget_core.port_value_written.connect(self._on_port_value_written)

        # Debounce: rapid keystrokes / drag-drops / graph port writes collapse
        # into ONE probe; the probe itself runs on a daemon thread so stat()
        # and the header read never block the GUI (matters on network mounts).
        self._probe_timer = QTimer(self._widget_core)
        self._probe_timer.setSingleShot(True)
        self._probe_timer.setInterval(250)
        self._probe_timer.timeout.connect(self._start_probe)
        self._probe_ready.connect(self._apply_probe)

        # 4. Mount & Patch
        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, 'patch_proxy'):
            self._widget_core.patch_proxy()
        if hasattr(self._widget_core, 'refresh_widget_palettes'):
            self._widget_core.refresh_widget_palettes()

        self.on_ui_change()

    # ------------------------------------------------------------------
    # Model metadata probing — debounced, off the GUI thread, cached
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_slider(slider: QSlider, spin: QSpinBox) -> None:
        """Move an unregistered companion slider to its spinner's value.

        Spinners keep the out-of-range auto value (-1); the slider shows
        that as full-right (everything). Deliberately unblocked: the
        slider's own emit publishes its binding (the mirror follows),
        and the link lambda's equality guard plus Qt's no-emit-on-same
        keep the ring one pass long.
        """
        target = (slider.maximum() if spin.value() < slider.minimum()
                  else min(spin.value(), slider.maximum()))
        slider.setValue(target)

    @classmethod
    def _link_slider(cls, slider: QSlider, spin: QSpinBox) -> None:
        """Two-way bind a companion slider to its (registered) spinner.

        The spinner owns the port, the state and the mirror; the slider
        is a local handle that writes through it. Dragging publishes
        through the spinner's signal like typing does.
        """
        slider.valueChanged.connect(
            lambda v: spin.setValue(v) if spin.value() != v else None)
        spin.valueChanged.connect(
            lambda _v: cls._sync_slider(slider, spin))
        cls._sync_slider(slider, spin)

    def _maybe_detect_mmproj(self, model_path: str) -> None:
        """Fill the projector picker from the model's folder, once per file.

        Runs on the GUI thread but costs one directory listing, only when
        detection is on, the picker is empty, and this model path has not
        been tried yet -- so a cleared picker stays cleared (manual
        text-only wins) and repeat edits cost nothing. Publishes through
        the port so mirrors and downstream see the pick; the write
        converges because the picker is non-empty afterwards.
        """
        if not self._widget_core.get_port_value("enable_mmproj"):
            return
        if (self._widget_core.get_port_value("mmproj_path") or "").strip():
            return
        model_path = (model_path or "").strip()
        if not model_path or model_path == self._mmproj_probed_for:
            return
        self._mmproj_probed_for = model_path
        parent = Path(model_path).expanduser().parent
        try:
            names = sorted(p.name for p in parent.iterdir()
                           if p.is_file()
                           and p.name.lower().startswith("mmproj")
                           and p.suffix.lower() == ".gguf")
        except OSError:
            return
        if names:
            found = str(parent / names[0])
            log.info(f"Vision projector detected: {found}")
            self._widget_core.apply_port_value("mmproj_path", found,
                                               user_edit=False)

    def _restore_remembered(self, path_str: str) -> None:
        """Re-apply the JSON-remembered settings for a reconnected model.

        Runs once per file (guarded by ``_restored_for``) so later edits
        are the user's own. Values clamp to the widget maxima; the probe
        narrows those further when it lands. Sliders follow their
        spinners through the shared signals.
        """
        try:
            preset = self._model_store.lookup(path_str)
        except Exception as exc:
            log.debug(f"GGUF settings lookup failed: {exc}")
            return
        if preset is None:
            return
        def _clamp(spin: QSpinBox, value: int) -> int:
            return max(spin.minimum(), min(int(value), spin.maximum()))
        self.spin_gpu.setValue(_clamp(self.spin_gpu, preset.gpu_layers))
        self.spin_ctx.setValue(_clamp(self.spin_ctx, preset.n_ctx))
        self.spin_threads.setValue(
            _clamp(self.spin_threads, preset.n_threads))
        self.spin_batch.setValue(
            _clamp(self.spin_batch, preset.n_batch))
        self.spin_seed.setValue(_clamp(self.spin_seed, preset.seed))
        self.chk_embedding.setChecked(bool(preset.embedding))
        self.chk_flash.setChecked(bool(preset.flash_attn))
        self.chk_mmap.setChecked(bool(preset.use_mmap))
        for combo, value in ((self._combo_type_k, preset.type_k),
                             (self._combo_type_v, preset.type_v)):
            idx = combo.findText(str(value))
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self.chk_mmproj.setChecked(bool(preset.enable_mmproj))
        if preset.mmproj_path:
            self._widget_core.apply_port_value(
                "mmproj_path", preset.mmproj_path, user_edit=False)
        self.chk_mtp.setChecked(bool(preset.enable_mtp))
        self.spin_mtp_tokens.setValue(
            _clamp(self.spin_mtp_tokens, preset.mtp_draft_tokens))
        log.info(f"Restored saved settings for {Path(path_str).name}.")

    def _effective_model_path(self) -> str:
        """The model to detect settings from: the port wins, the picker
        covers mid-typing.

        The picker is a view of a BIDIRECTIONAL port, but views stall:
        restore and graph load apply values with signals held, so the
        probe and the guards must read dataflow truth (the port), not
        the widget that may not have caught up yet.
        """
        port = str(
            self._widget_core.get_port_value("model_path") or "").strip()
        if port:
            return port
        try:
            return (self.path_picker.get_path() or "").strip()
        except RuntimeError:
            return ""

    def _update_estimation(self) -> None:
        """Refresh the status label. Cheap: never touches the filesystem.

        On a cache miss it shows a pending message and (re)starts the
        debounce timer; the actual stat + header read happen in
        `_probe_worker` on a daemon thread.
        """
        path_str = self._effective_model_path()
        self._maybe_detect_mmproj(path_str)
        if path_str and path_str != self._restored_for:
            self._restored_for = path_str
            self._restore_remembered(path_str)
        if not path_str:
            self._widget_core.push_display("status", "No model selected.")
            return

        probe = self._probe_cache.get(path_str)
        if probe is None:
            self._widget_core.push_display("status", "Scanning model metadata…")
            self._probe_timer.start()
            return
        self._push_estimation(probe)

    def _push_estimation(self, probe: Dict[str, Any]) -> None:
        """Compose the VRAM estimate from cached probe values (V6 push_display)."""
        if not probe.get("exists", False):
            self._widget_core.push_display("status", "File not found.")
            return
        size_gb = probe["size_gb"]
        # A single shared server loads the weights + one KV cache exactly once
        # (LM Studio-like), so the estimate is no longer multiplied by pool size.
        kv_cache_gb = (self.spin_ctx.value() / 1024) * 0.125
        total_est = size_gb + kv_cache_gb
        msg = (
            f"Base Model: ~{size_gb:.2f} GB\n"
            f"KV Cache (ctx {self.spin_ctx.value()}): ~{kv_cache_gb:.2f} GB\n"
            f"Est. Peak VRAM/RAM: ~{total_est:.2f} GB (single shared server)"
        )
        self._widget_core.push_display("status", msg)

    @Slot()
    def _start_probe(self) -> None:
        path_str = self._effective_model_path()
        if not path_str or path_str in self._probe_cache:
            return
        threading.Thread(
            target=self._probe_worker, args=(path_str,),
            daemon=True, name="gguf-meta-probe",
        ).start()

    def _probe_worker(self, path_str: str) -> None:
        """Daemon thread: ALL filesystem I/O (exists/stat/header) lives here."""
        probe: Dict[str, Any] = {"exists": False, "size_gb": 0.0,
                                 "max_ctx": None, "layers": None}
        try:
            path = Path(path_str)
            if path.is_file():
                probe["exists"] = True
                probe["size_gb"] = path.stat().st_size / (1024 ** 3)
                (probe["max_ctx"], probe["layers"],
                 probe["tools"]) = self._read_meta(path_str)
        except Exception as exc:
            log.debug(f"GGUF probe failed for {path_str}: {exc}")
        self._probe_ready.emit(path_str, probe)

    @staticmethod
    def _read_meta(path_str: str) -> tuple:
        """Header-only parse; falls back to gguf.GGUFReader for exotic files."""
        try:
            meta = read_gguf_meta(path_str)
            return meta.context_length, meta.block_count, meta.supports_tools
        except Exception as exc:
            log.debug(f"Manual GGUF header parse failed ({exc}); trying gguf package.")
        try:
            # The fallback reads no chat template: `gguf.GGUFReader` is only
            # reached for files the manual parser rejects, and None there
            # means "unknown", which costs the fence protocol rather than a
            # refused request.
            return (*read_gguf_meta_fallback(path_str), None)
        except ImportError:
            log.warning("The 'gguf' python package is missing, so context "
                        "and layer limits are not clamped to this model's "
                        "own header; the spinbox defaults stand instead.")
        except Exception as exc:
            log.debug(f"Failed to read GGUF metadata for limit clamping: {exc}")
        return None, None, None

    @Slot(str, object)
    def _apply_probe(self, path_str: str, probe: Dict[str, Any]) -> None:
        """GUI thread: cache the result, clamp spinboxes, refresh the label."""
        if probe.get("exists", False):
            # Negative results are NOT cached so a file created later is
            # picked up by the next (debounced) change.
            self._probe_cache[path_str] = probe
            while len(self._probe_cache) > 16:
                self._probe_cache.pop(next(iter(self._probe_cache)))

        if path_str != self._effective_model_path():
            return  # stale probe: the user moved on to another file

        max_ctx, layers = probe.get("max_ctx"), probe.get("layers")
        # Block signals so setMaximum-driven value clamps don't re-trigger
        # the evaluation loop.
        was_blocked_ctx = self.spin_ctx.blockSignals(True)
        was_blocked_gpu = self.spin_gpu.blockSignals(True)
        if max_ctx is not None and max_ctx > 0:
            self.spin_ctx.setMaximum(max_ctx)
            log.debug(f"Dynamic Limit: Max Context Window updated to {max_ctx}")
        if layers is not None and layers > 0:
            # Layers + 1 accommodates the output/LM head layer for full offloading
            self.spin_gpu.setMaximum(layers + 1)
            log.debug(f"Dynamic Limit: Max GPU Layers updated to {layers + 1}")
        self.spin_ctx.blockSignals(was_blocked_ctx)
        self.spin_gpu.blockSignals(was_blocked_gpu)
        # Companion sliders track the same maxima, then re-seat on them.
        # Unblocked on purpose: a max shrink that clamps the slider must
        # reach the spinner (and its binding) through the link.
        self.slider_gpu.setMaximum(max(1, self.spin_gpu.maximum()))
        self.slider_ctx.setMaximum(max(512, self.spin_ctx.maximum()))
        self._sync_slider(self.slider_gpu, self.spin_gpu)
        self._sync_slider(self.slider_ctx, self.spin_ctx)

        self._push_estimation(probe)

    @Slot(str)
    def _on_value_changed(self, port_name: str) -> None:
        self._update_estimation()
        self.on_ui_change()

    @Slot(str, object)
    def restore_state(self, state: Dict[str, Any]) -> None:
        # Signals are held while the core applies saved values, so no
        # change hook fires: kick detection explicitly, or a restored
        # graph would sit on defaults until the first manual edit.
        super().restore_state(state)
        self._update_estimation()

    def _on_port_value_written(self, port_name: str, value: Any) -> None:
        if port_name == "model_path":
            self.path_picker.set_path(value)
        elif port_name == "prompt_cache":
            self.path_picker_prompt_cache.set_path(value)
        elif port_name == "mmproj_path":
            self.path_picker_mmproj.set_path(value)

        self._update_estimation()

    def _is_model_loaded(self) -> bool:
        if self.model_pool is not None:
            return True
        cached_val = self._get_cached_value("model_obj")
        if cached_val and isinstance(cached_val, dict):
            return "model" in cached_val or "pool" in cached_val
        return False

    def _unload_model(self) -> None:
        log.info(f"Ejecting model for {self.node_name}...")

        if self.model_pool is not None:
            self.model_pool.cleanup()
            self.model_pool = None

        cached_val = self._get_cached_value("model_obj")
        if cached_val is not None:
            if "model" in cached_val:
                model = cached_val["model"]
                if hasattr(model, 'close'):
                    try:
                        model.close()
                    except Exception:
                        pass
                del cached_val["model"]

            if "pool" in cached_val:
                del cached_val["pool"]

        gc.collect()
        log.info("Model ejected and VRAM cleared.")

        # Reset KV cache progress bar on eject.
        self._progress_kv.setEnabled(False)
        self._progress_kv.setValue(0)
        self._progress_kv.setFormat("KV Cache: 0% filled")
        self._btn_refresh_kv.setEnabled(False)
        self.emit_stream("pool_info", None)
        self.on_ui_change()
        self._update_button_idle()

    def execute(self) -> None:
        if self._is_computing:
            log.info("Execute: Cancellation requested by user.")
            self.cancel_compute()
        elif self._is_model_loaded():
            log.info("Execute: User clicked Eject. Unloading model...")
            self._unload_model()
        else:
            log.debug("Execute: Initiating GGUF load process.")
            self.btn_load.set_label("Loading (Click to Cancel)...")
            super().execute()

    def _update_button_idle(self) -> None:
        if self._is_model_loaded():
            self.btn_load.set_label("Eject Model")
        else:
            self.btn_load.set_label("Load GGUF Pool")

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        log.debug("Evaluate Finished: Resetting button state.")
        self._update_button_idle()
        # Update KV cache progress bar if we have pending pool info.
        if self._pending_pool_info is not None:
            self._progress_kv.setEnabled(True)
            self._btn_refresh_kv.setEnabled(True)
            self._refresh_kv_progress(self._pending_pool_info)
            self._pending_pool_info = None

    def _refresh_kv_progress(self, info: dict) -> None:
        """Update the KV cache progress bar from a pool snapshot."""
        pct = info.get("kv_fill_pct", 0.0)
        used = info.get("kv_used_tokens", 0)
        total = info.get("kv_total_tokens", 0)
        self._progress_kv.setFormat(
            f"KV Cache: {used}/{total} tokens ({pct:.1f}% filled)"
        )
        self._progress_kv.setValue(int(pct))
        self.emit_stream("pool_info", info)

    def _on_refresh_kv(self) -> None:
        """Manual refresh of KV cache usage from the pool."""
        if self.model_pool is not None:
            try:
                info = self.model_pool.snapshot()
                self._refresh_kv_progress(info)
            except Exception as exc:
                log.debug(f"KV refresh failed: {exc}")

    def _cleanup_after_worker(self, skip_results: bool = False) -> None:
        # See the Agent node: Weave passes `skip_results=True` on the
        # cancelled-result path, and an override without it turns that
        # into a TypeError.
        self._update_button_idle()
        super()._cleanup_after_worker(skip_results=skip_results)

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not LLAMA_CPP_AVAILABLE:
            err_msg = "llama-cpp-python is not installed. Loading aborted."
            log.error(err_msg)
            self.compute_error.emit(err_msg)
            return {"model_obj": None}

        if not LLAMA_SERVER_AVAILABLE:
            err_msg = server_missing_deps_message()
            log.error(err_msg)
            self.compute_error.emit(err_msg)
            return {"model_obj": None}

        # Worker-thread rule: read ONLY from inputs (the BIDIRECTIONAL
        # binding delivers the picker's value there), never from widgets.
        model_path = inputs.get("model_path")

        if not model_path:
            self.compute_error.emit("No model path provided.")
            return {"model_obj": None}

        try:
            model_path = str(Path(model_path).resolve())
        except Exception as e:
            self.compute_error.emit(f"Path resolution error: {e}")
            return {"model_obj": None}

        if not Path(model_path).exists():
            self.compute_error.emit(f"Invalid path: {model_path}")
            return {"model_obj": None}

        llama_kwargs = {
            "seed": inputs.get("seed", -1),
            "n_ctx": inputs.get("n_ctx", 4096),
            "n_gpu_layers": inputs.get("gpu_layers", -1),
            "n_threads": None if inputs.get("n_threads", -1) == -1 else inputs.get("n_threads"),
            "n_batch": inputs.get("n_batch", 512),
            "flash_attn": inputs.get("flash_attn", False),
            "embedding": bool(inputs.get("embedding", False)),
            "use_mmap": inputs.get("use_mmap", True),
            # The prefix measurement (D41/D47, G15) is a *read* of the
            # server's own stderr, and this is the switch that makes the
            # server write the lines it reads. Left False, `prefix_report`
            # reports "nothing measured" forever on the canvas while
            # working perfectly from a script -- which is exactly why the
            # numbers took until 2026-09-19 to exist. It costs about six
            # lines per request into a temp file the pool already keeps,
            # and it logs no prompt or completion text.
            "verbose": True,
        }

        # KV-cache quantization: the UI provides type names, llama.cpp
        # expects GGML type enums.  Only forwarded when set away from the
        # f16 default so llama-cpp-python builds without the kwargs keep
        # working (the TypeError fallback below strips them again).
        _kv_type_consts = {"f16": "GGML_TYPE_F16", "bf16": "GGML_TYPE_BF16",
                           "q8_0": "GGML_TYPE_Q8_0", "q4_0": "GGML_TYPE_Q4_0"}
        import llama_cpp as _llama_cpp_mod
        for _key in ("type_k", "type_v"):
            _name = inputs.get(_key, "f16")
            if _name == "f16":
                continue
            _const = getattr(_llama_cpp_mod, _kv_type_consts.get(_name, ""), None)
            if _const is not None:
                llama_kwargs[_key] = _const
            else:
                log.warning(
                    f"KV cache type '{_name}' not supported by this "
                    f"llama_cpp build — '{_key}' ignored."
                )


        if prompt_cache := inputs.get("prompt_cache"):
            llama_kwargs.update({
                "prompt_cache": prompt_cache,
                "prompt_cache_all": inputs.get("prompt_cache_all", False),
                "prompt_cache_ro": inputs.get("prompt_cache_ro", False)
            })

        if (cram := inputs.get("cram", 0)) > 0:
            llama_kwargs["cram"] = cram

        if inputs.get("enable_mtp"):
            # Native MTP (built-in draft heads, PR #22673): the binding
            # learns these keys eventually; until then they are ignored
            # and the server runs one token per step (see tooltip).
            n_draft = max(1, int(inputs.get("mtp_draft_tokens", 2) or 2))
            llama_kwargs["spec_type"] = "draft-mtp"
            llama_kwargs["spec_draft_n_max"] = n_draft

        mmproj = ""
        if inputs.get("enable_mmproj"):
            mmproj = str(inputs.get("mmproj_path") or "").strip()
            if mmproj and Path(mmproj).expanduser().is_file():
                llama_kwargs["mmproj"] = mmproj
            elif mmproj:
                log.warning(
                    f"Vision projector not found: {mmproj} -- text only.")

        if self.is_compute_cancelled():
            return {"model_obj": None}

        # Spawns the background llama_cpp.server and blocks until it answers
        # /v1/models. A bad model / OOM / missing extra raises here with the
        # server's own error tail, instead of silently returning a dead pool.
        try:
            self.model_pool = GGUFModelPool(
                model_path=model_path,
                n_instances=int(inputs.get("pool_size", 2)),
                clear_on_return=inputs.get("clear_on_return", True),
                session_affinity=bool(inputs.get("session_affinity", True)),
                **llama_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - surface the reason to the UI
            log.error(f"GGUF server failed to start: {exc}")
            self.compute_error.emit(f"Server failed to start: {exc}")
            return {"model_obj": None}

        # Read from the file rather than the probe cache: the cache lives on
        # the GUI thread, and this is the worker. One KV-section scan against
        # a model we are about to spend seconds loading is not worth a
        # cross-thread handshake to avoid.
        supports_tools = self._read_meta(model_path)[2]
        if supports_tools is None:
            log.debug(
                f"{Path(model_path).name} carries no chat template, so tool "
                f"support is unknown; the agent will use the fence protocol."
            )

        pool_info = self.model_pool.snapshot()
        # Update KV cache progress bar on the GUI thread via evaluate-finished.
        self._pending_pool_info = pool_info
        # `GraphEngine.supports_native_tools` reads this, and
        # `select_transport` turns it into a protocol. Absent rather than
        # False when unknown, so the handle says "not established" instead
        # of asserting something the file never claimed.
        handle: Dict[str, Any] = {"backend": "gguf", "pool": self.model_pool}
        if supports_tools:
            handle["supports_tools"] = True
        if llama_kwargs.get("mmproj"):
            handle["multimodal"] = True
        if llama_kwargs.get("spec_type") == "draft-mtp":
            handle["mtp"] = True
        gen_override = inputs.get("inference_settings")
        handle = apply_inference_settings(handle, gen_override)
        try:
            self._model_store.remember(
                model_path,
                gpu_layers=int(inputs.get("gpu_layers", -1)),
                n_ctx=int(inputs.get("n_ctx", 4096)),
                n_threads=int(inputs.get("n_threads", -1)),
                n_batch=int(inputs.get("n_batch", 512)),
                seed=int(inputs.get("seed", -1)),
                flash_attn=bool(inputs.get("flash_attn", False)),
                use_mmap=bool(inputs.get("use_mmap", True)),
                type_k=str(inputs.get("type_k", "f16")),
                type_v=str(inputs.get("type_v", "f16")),
                embedding=bool(inputs.get("embedding", False)),
                enable_mmproj=bool(inputs.get("enable_mmproj", False)),
                mmproj_path=str(inputs.get("mmproj_path", "") or ""),
                enable_mtp=bool(inputs.get("enable_mtp", False)),
                mtp_draft_tokens=int(
                    inputs.get("mtp_draft_tokens", 2) or 2),
            )
        except Exception as exc:
            log.debug(f"GGUF settings remember failed: {exc}")
        return {"model_obj": handle, "pool_info": pool_info}

    def cleanup(self) -> None:
        log.info(f"Node Cleanup: Releasing resources for {self.node_name}")
        # Detach the probe pipeline first so a still-running probe thread
        # can't queue a callback into a half-destroyed node (WV507).
        self._probe_timer.stop()
        try:
            self._probe_ready.disconnect(self._apply_probe)
        except (RuntimeError, TypeError):
            pass
        self.cancel_compute()
        self._unload_model()
        super().cleanup()
