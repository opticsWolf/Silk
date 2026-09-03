# -*- coding: utf-8 -*-
"""GGUF LLM Loader Module.

Provides a specialized, thread-safe node for loading Large Language Models
via the GGUF format using llama.cpp. Compliant with Weave V6 specifications.
"""

import gc
import threading
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtCore import QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSpinBox,
    QFrame,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.node.threaded import ThreadedManualNode
from weave.registry import register_node
from weave.logger import get_logger
from weave.node import VerticalSizePolicy
from weave.panel.mirror_contracts import MirrorContract

from weave.library.sync_tool_button.widgets.sync_tool_button import SyncToolButton
from weave.widgets.sync_button import SyncButton
from weave.widgets.path_picker import PathPickerWidget

log = get_logger("GGUFLNode")

# Shared silk port types (registered idempotently on import).
from .silk_ports import GGUF_MODEL_TYPE  # noqa: F401

# Qt-free pool + dependency guards live in functions/ so the GraphEngine and
# headless tests can share them. The pool now runs a background
# ``llama_cpp.server`` process, which needs the [server] extra as well.
from ..functions.model_pool import (
    GGUFModelPool, LLAMA_CPP_AVAILABLE, LLAMA_SERVER_AVAILABLE,
    server_missing_deps_message,
)
from ..functions.gguf_meta import read_gguf_meta, read_gguf_meta_fallback

if not LLAMA_CPP_AVAILABLE:
    log.error("llama-cpp-python NOT found. GGUF Loader will be non-functional.")
elif not LLAMA_SERVER_AVAILABLE:
    log.error(f"GGUF Loader non-functional: {server_missing_deps_message()}")

# --- V6 Structural Mirror Widget ---
class SeparatorLine(QFrame):
    """A mirror-compliant horizontal line separator for the UI."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setProperty("opaque_bg", True)

    __mirror__ = MirrorContract(
        clone=lambda src, _b: SeparatorLine(),
    )

# ==============================================================================
# GGUF Loader Node (V6)
# ==============================================================================

@register_node
class GGUFLNode(ThreadedManualNode):
    """Loads GGUF models into a thread-safe pool via llama.cpp on a worker thread."""

    node_class:           ClassVar[str]                 = "AI"
    node_subclass:        ClassVar[str]                 = "Loaders"
    node_name:            ClassVar[Optional[str]]       = "GGUF Loader"
    node_description:     ClassVar[Optional[str]]       = "Loads GGUF models via llama.cpp into an isolated KV pool."
    node_tags:            ClassVar[Optional[List[str]]] = ["llama", "llm", "gguf", "inference", "pool"]
    node_icon:            ClassVar[Optional[str]]       = "file-import"
    vertical_size_policy: ClassVar[VerticalSizePolicy]  = VerticalSizePolicy.FIT

    # Emitted from the probe worker thread; queued back onto the GUI thread.
    _probe_ready = Signal(str, object)

    def __init__(self, title: str = "GGUF Loader", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # 1. Add Graph Ports
        self.add_input("model_path", datatype="filepath")
        self.add_input("prompt_cache", datatype="filepath")
        self.add_output("model_obj", datatype="gguf_model")
        self.add_output("pool_info", datatype="dict")  # live pool stats

        self.model_pool: Optional[GGUFModelPool] = None
        # path -> {"size_gb", "max_ctx", "layers"}; covers stat() AND header
        # parse so a cached path costs zero filesystem I/O on the GUI thread.
        self._probe_cache: Dict[str, Dict[str, Any]] = {}

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
        form.addRow("Settings Mode:", self._combo_mode)
        self._widget_core.register_widget("mode", self._combo_mode, role=PortRole.INTERNAL, datatype="str", default="simple", add_to_layout=False)

        self.spin_gpu = QSpinBox()
        self.spin_gpu.setRange(-1, 256)
        self.spin_gpu.setValue(-1)
        form.addRow("GPU Layers:", self.spin_gpu)
        self._widget_core.register_widget("gpu_layers", self.spin_gpu, role=PortRole.INTERNAL, datatype="int", default=-1, add_to_layout=False)

        self.spin_ctx = QSpinBox()
        self.spin_ctx.setRange(512, 1048576)
        self.spin_ctx.setValue(4096)
        form.addRow("Context:", self.spin_ctx)
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
        form.addRow("CPU Threads:", self.spin_threads)
        self._widget_core.register_widget("n_threads", self.spin_threads, role=PortRole.INTERNAL, datatype="int", default=-1, add_to_layout=False)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 2147483647)
        self.spin_seed.setValue(-1)
        self.spin_seed.setSpecialValueText("Random (-1)")
        form.addRow("Seed:", self.spin_seed)
        self._widget_core.register_widget("seed", self.spin_seed, role=PortRole.INTERNAL, datatype="int", default=-1, add_to_layout=False)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 8192)
        self.spin_batch.setValue(512)
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

        self.chk_flash = QCheckBox()
        form.addRow("Flash Attention:", self.chk_flash)
        self._widget_core.register_widget("flash_attn", self.chk_flash, role=PortRole.INTERNAL, datatype="bool", default=False, add_to_layout=False)

        self.chk_mmap = QCheckBox()
        self.chk_mmap.setChecked(True)
        form.addRow("Use mmap():", self.chk_mmap)
        self._widget_core.register_widget("use_mmap", self.chk_mmap, role=PortRole.INTERNAL, datatype="bool", default=True, add_to_layout=False)

        self._combo_type_k = QComboBox()
        self._combo_type_k.addItems(["f16", "bf16", "q8_0", "q4_0"])
        form.addRow("KV Cache Type K:", self._combo_type_k)
        self._widget_core.register_widget("type_k", self._combo_type_k, role=PortRole.INTERNAL, datatype="str", default="f16", add_to_layout=False)

        self._combo_type_v = QComboBox()
        self._combo_type_v.addItems(["f16", "bf16", "q8_0", "q4_0"])
        form.addRow("KV Cache Type V:", self._combo_type_v)
        self._widget_core.register_widget("type_v", self._combo_type_v, role=PortRole.INTERNAL, datatype="str", default="f16", add_to_layout=False)

        self.path_picker_prompt_cache = PathPickerWidget(mode="save_file", filters=["Cache Files (*.cache)", "All Files (*)"])
        form.addRow("Prompt Cache File:", self.path_picker_prompt_cache)
        self._widget_core.register_widget(
            "prompt_cache", self.path_picker_prompt_cache, role=PortRole.BIDIRECTIONAL,
            datatype="filepath", default="", add_to_layout=False
        )

        self.chk_prompt_cache_all = QCheckBox()
        form.addRow("Prompt Cache All:", self.chk_prompt_cache_all)
        self._widget_core.register_widget("prompt_cache_all", self.chk_prompt_cache_all, role=PortRole.INTERNAL, datatype="bool", default=False, add_to_layout=False)

        self.chk_prompt_cache_ro = QCheckBox()
        form.addRow("Prompt Cache RO:", self.chk_prompt_cache_ro)
        self._widget_core.register_widget("prompt_cache_ro", self.chk_prompt_cache_ro, role=PortRole.INTERNAL, datatype="bool", default=False, add_to_layout=False)

        self.chk_clear_cache = QCheckBox()
        self.chk_clear_cache.setChecked(True)
        form.addRow("Clear Cache on Return:", self.chk_clear_cache)
        self._widget_core.register_widget("clear_on_return", self.chk_clear_cache, role=PortRole.INTERNAL, datatype="bool", default=True, add_to_layout=False)

        self.spin_cram = QSpinBox()
        self.spin_cram.setRange(0, 131072)
        form.addRow("Cache RAM (MB):", self.spin_cram)
        self._widget_core.register_widget("cram", self.spin_cram, role=PortRole.INTERNAL, datatype="int", default=0, add_to_layout=False)

        form.addRow(SeparatorLine())

        # V6: Display readout for status estimation
        self._label_status = QLabel("Select a model to estimate VRAM.")
        self._label_status.setEnabled(False)
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget("status", self._label_status, role=PortRole.DISPLAY, datatype="str", add_to_layout=False)

        # KV Cache usage progress bar + refresh button
        kv_row = QHBoxLayout()
        self._progress_kv = QProgressBar()
        self._progress_kv.setRange(0, 100)
        self._progress_kv.setValue(0)
        self._progress_kv.setFormat("KV Cache: %p%% filled")
        self._progress_kv.setEnabled(False)
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
        self.btn_load = SyncButton(initial_text="Load GGUF Pool")
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
                    "prompt_cache_ro", "clear_on_return", "cram"
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

    def _update_estimation(self) -> None:
        """Refresh the status label. Cheap: never touches the filesystem.

        On a cache miss it shows a pending message and (re)starts the
        debounce timer; the actual stat + header read happen in
        `_probe_worker` on a daemon thread.
        """
        path_str = self.path_picker.get_path()
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
        path_str = self.path_picker.get_path()
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
                probe["max_ctx"], probe["layers"] = self._read_meta(path_str)
        except Exception as exc:
            log.debug(f"GGUF probe failed for {path_str}: {exc}")
        self._probe_ready.emit(path_str, probe)

    @staticmethod
    def _read_meta(path_str: str) -> tuple:
        """Header-only parse; falls back to gguf.GGUFReader for exotic files."""
        try:
            meta = read_gguf_meta(path_str)
            return meta.context_length, meta.block_count
        except Exception as exc:
            log.debug(f"Manual GGUF header parse failed ({exc}); trying gguf package.")
        try:
            return read_gguf_meta_fallback(path_str)
        except ImportError:
            log.debug("The 'gguf' python package is missing. Skipping dynamic limits.")
        except Exception as exc:
            log.debug(f"Failed to read GGUF metadata for limit clamping: {exc}")
        return None, None

    @Slot(str, object)
    def _apply_probe(self, path_str: str, probe: Dict[str, Any]) -> None:
        """GUI thread: cache the result, clamp spinboxes, refresh the label."""
        if probe.get("exists", False):
            # Negative results are NOT cached so a file created later is
            # picked up by the next (debounced) change.
            self._probe_cache[path_str] = probe
            while len(self._probe_cache) > 16:
                self._probe_cache.pop(next(iter(self._probe_cache)))

        if path_str != self.path_picker.get_path():
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

        self._push_estimation(probe)

    @Slot(str)
    def _on_value_changed(self, port_name: str) -> None:
        self._update_estimation()
        self.on_ui_change()

    @Slot(str, object)
    def _on_port_value_written(self, port_name: str, value: Any) -> None:
        if port_name == "model_path":
            self.path_picker.set_path(value)
        elif port_name == "prompt_cache":
            self.path_picker_prompt_cache.set_path(value)

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
        if hasattr(self, "_pending_pool_info") and self._pending_pool_info is not None:
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

    def _cleanup_after_worker(self) -> None:
        self._update_button_idle()
        super()._cleanup_after_worker()

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
            "verbose": False,
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
                **llama_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - surface the reason to the UI
            log.error(f"GGUF server failed to start: {exc}")
            self.compute_error.emit(f"Server failed to start: {exc}")
            return {"model_obj": None}

        pool_info = self.model_pool.snapshot()
        # Update KV cache progress bar on the GUI thread via evaluate-finished.
        self._pending_pool_info = pool_info
        return {"model_obj": {"backend": "gguf", "pool": self.model_pool},
                "pool_info": pool_info}

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
