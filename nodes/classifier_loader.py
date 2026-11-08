# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Classifier Loader -- a GGUF loader specialised for decision models.

Same pool machinery as the other loaders (a background ``llama_cpp.server``
via :class:`functions.model_pool.GGUFModelPool`), but the model is a
decision GGUF (Laya, Julia-1, Kev, lev, OpenJev, Nimble, Clef, ...) and the
client speaks ``POST /v1/systemone`` through
:class:`functions.systemone.SystemOneClient` instead of chat completions.

Needs a server of llama.cpp ``b11361`` or newer (i.e. a recent
``llama-cpp-python[server]``); anything older answers the smoke decision
with 404/501 and the load fails loudly here instead of mid-triage.

Settings sources, in order:

1. **The GGUF file itself** (``functions/gguf_meta.py``) -- size for the
   footprint estimate and ``context_length`` to clamp the spinbox.
2. **The widgets** -- device dropdown (CPU ``n_gpu_layers=0`` / GPU ``-1``,
   CPU recommended: decision encoders are small and fast on CPU) plus the
   numeric defaults below.
3. **The per-model memory** (``functions.presets.DecisionModelStore``,
   ``~/.weave/presets/decision_models.json``) -- written on every
   successful load (smoke decision included), re-applied when the same
   file is picked again. A GGUF header carries model hyperparams, never
   server settings, so the JSON sidecar is the mechanism.

The output is the ``classifier_model`` wire:
``{"backend": "systemone", "pool": ..., "role": "decision"}`` -- the
shape the Classifier and Ensemble nodes consume via ``SystemOneClient``.

**Loaded on demand.** There is no load button to press: the node hands
out its handle as soon as a model is chosen, and the llama.cpp server
starts the first time a Classifier or Ensemble asks it a question
(``functions.model_pool.LazyGGUFPool``). The smoke decision runs right
then, before that first question gets the server, and the settings are
remembered once it passes. Unload stops the server; the next question
starts it again.
"""

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
    QSpinBox,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.node.threaded import ThreadedNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from weave.widgets.sync_button import SyncButton
from weave.widgets.path_picker import PathPickerWidget
from weave.widgets.separator_line import SeparatorLine

# Shared silk port types (registered idempotently on import).
from .silk_ports import CLASSIFIER_MODEL_TYPE  # noqa: F401

from ..functions.model_pool import (
    GGUFModelPool, LazyGGUFPool, LLAMA_CPP_AVAILABLE, LLAMA_SERVER_AVAILABLE,
    server_missing_deps_message,
)
from ..functions.gguf_meta import read_gguf_meta, read_gguf_meta_fallback
from ..functions.presets import DecisionModelStore
from ..functions.systemone import SystemOneClient

if not LLAMA_CPP_AVAILABLE:
    log = get_logger("ClassifierLoader")
    log.error("llama-cpp-python NOT found. Classifier Loader will be non-functional.")
elif not LLAMA_SERVER_AVAILABLE:
    log = get_logger("ClassifierLoader")
    log.error(f"Classifier Loader non-functional: {server_missing_deps_message()}")
else:
    log = get_logger("ClassifierLoader")

#: Decision models worth naming in the UI. Anything else loads the same
#: way -- this is a hint, not a constraint.
KNOWN_FAMILIES = ("laya", "julia", "kev", "lev", "openjev", "nimble",
                  "clef", "pplx-decider", "tev")

#: Device dropdown value -> llama.cpp GPU-layer setting.
DEVICE_GPU_LAYERS = {"cpu": 0, "gpu": -1}

#: The smoke decision asked right after the server answers /v1/models.
SMOKE_STATE = "connectivity check"
SMOKE_QUESTIONS = {
    "ok": {"type": "noul", "instructions": "Is any text present?"},
}


@register_node
class ClassifierLoaderNode(ThreadedNode):
    """Loads a decision-model GGUF and serves /v1/systemone from it."""

    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    node_name: ClassVar[Optional[str]] = "Classifier Loader"
    node_description: ClassVar[Optional[str]] = (
        "Loads a Laya-style decision model via llama.cpp and answers typed "
        "questions (choice/score/noul). CPU-first; remembers per-model settings."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "llama", "classifier", "decision", "systemone", "laya",
        "triage", "gguf",
    ]
    node_icon: ClassVar[Optional[str]] = "scale"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    #: 3: loads on demand -- an auto node with an Unload button, no Load.
    node_state_api = 3
    node_version = 2

    @classmethod
    def migrate_state(cls, state: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """api 1 → 2: the Decision Loader is the Classifier Loader, and its
        output is the ``classifier_model`` port type instead of a generic
        ``model_handle``.

        Widgets are unchanged, so saved values land as before; only wires
        off the renamed output need re-drawing.

        api 2 → 3: the node was manual (a Load button). A saved scene
        carries that flag, which would keep the node waiting for a click
        that no longer exists.
        """
        dataflow = state.get("dataflow")
        if from_version < 3 and isinstance(dataflow, dict):
            dataflow["manual_mode"] = False
        return state

    _probe_ready = Signal(str, object)
    #: The lazy pool started (a pool) or failed to (an exception), from
    #: whichever thread asked the first question.
    _pool_started = Signal(object)

    _pending_pool_info: Optional[dict] = None

    def __init__(self, title: str = "Classifier Loader", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        self.add_input("model_path", datatype="filepath")
        self.add_output("model_obj", datatype="classifier_model")
        self.add_output("pool_info", datatype="dict")

        self.model_pool: Optional[LazyGGUFPool] = None
        self._pool_key: Optional[tuple] = None
        self._probe_cache: Dict[str, Dict[str, Any]] = {}
        self._model_store = DecisionModelStore()

        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self.path_picker = PathPickerWidget(
            mode="file", filters=["GGUF Models (*.gguf)", "All Files (*)"])
        self.path_picker.setToolTip(
            "A decision-model .gguf (e.g. ggml-org/Laya-GGUF). Same file "
            "must not also be loaded as chat -- decisions and chat are "
            "different endpoints on different servers.")
        form.addRow("Model Path:", self.path_picker)
        self._widget_core.register_widget(
            "model_path", self.path_picker, role=PortRole.BIDIRECTIONAL,
            datatype="filepath", default="", add_to_layout=False)

        self._combo_device = QComboBox()
        self._combo_device.addItem("CPU", userData="cpu")
        self._combo_device.addItem("GPU", userData="gpu")
        self._combo_device.setToolTip(
            "CPU runs the decision model in RAM (n_gpu_layers=0) -- the "
            "recommended split, since encoders like Laya answer in "
            "milliseconds on CPU and leave VRAM to the chat model.")
        form.addRow("Device:", self._combo_device)
        self._widget_core.register_widget(
            "device", self._combo_device, role=PortRole.INTERNAL,
            datatype="str", default="cpu", add_to_layout=False)

        self.spin_ctx = QSpinBox()
        self.spin_ctx.setRange(256, 131072)
        self.spin_ctx.setValue(2048)
        self.spin_ctx.setToolTip(
            "Decision prompts are short (state + questions); clamped to "
            "the file's own maximum on probe.")
        form.addRow("Context:", self.spin_ctx)
        self._widget_core.register_widget(
            "n_ctx", self.spin_ctx, role=PortRole.INTERNAL,
            datatype="int", default=2048, add_to_layout=False)

        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(-1, 128)
        self.spin_threads.setValue(8)
        self.spin_threads.setSpecialValueText("Auto (-1)")
        form.addRow("CPU Threads:", self.spin_threads)
        self._widget_core.register_widget(
            "n_threads", self.spin_threads, role=PortRole.INTERNAL,
            datatype="int", default=8, add_to_layout=False)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 8192)
        self.spin_batch.setValue(512)
        form.addRow("Batch Size:", self.spin_batch)
        self._widget_core.register_widget(
            "n_batch", self.spin_batch, role=PortRole.INTERNAL,
            datatype="int", default=512, add_to_layout=False)

        self.chk_mmap = QCheckBox()
        self.chk_mmap.setChecked(True)
        form.addRow("Use mmap():", self.chk_mmap)
        self._widget_core.register_widget(
            "use_mmap", self.chk_mmap, role=PortRole.INTERNAL,
            datatype="bool", default=True, add_to_layout=False)

        form.addRow(SeparatorLine())

        self._label_status = QLabel("Select a decision model.")
        self._label_status.setEnabled(False)
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False)

        btn_row = QHBoxLayout()
        self.btn_load = SyncButton(initial_text="Unload Classifier",
                                   initial_icon="player-eject")
        self.btn_load.setToolTip(
            "The model loads when a Classifier or Ensemble first asks it a "
            "question. Unload stops its server and frees the memory; it "
            "starts again when the next question comes.")
        self.btn_load.clicked.connect(self._on_unload_clicked)
        btn_row.addWidget(self.btn_load)
        form.addRow("", btn_row)
        self._widget_core.register_widget(
            "btn_load", self.btn_load, role=PortRole.INTERNAL,
            add_to_layout=False)

        self._widget_core.value_changed.connect(self._on_value_changed)
        self._widget_core.port_value_written.connect(self._on_port_value_written)

        self._probe_timer = QTimer(self._widget_core)
        self._probe_timer.setSingleShot(True)
        self._probe_timer.setInterval(250)
        self._probe_timer.timeout.connect(self._start_probe)
        self._probe_ready.connect(self._apply_probe)
        self._pool_started.connect(self._on_pool_started)

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()
        if hasattr(self._widget_core, "refresh_widget_palettes"):
            self._widget_core.refresh_widget_palettes()

        self.on_ui_change()

    # ------------------------------------------------------------------
    # Probing + per-model memory (GUI thread, except _probe_worker)
    # ------------------------------------------------------------------

    def _update_estimation(self) -> None:
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
        if not probe.get("exists", False):
            self._widget_core.push_display("status", "File not found.")
            return
        device = str(self._combo_device.currentData() or "cpu")
        where = "RAM only" if device == "cpu" else "VRAM (full offload)"
        family = self._guess_family(probe.get("filename", ""))
        line = f"Family: {family}\n" if family else ""
        msg = (
            f"{line}Base Model: ~{probe['size_gb']:.2f} GB\n"
            f"Device: {device.upper()} ({where})\n"
            f"Endpoint: POST /v1/systemone (choice/score/noul, "
            f"0 output tokens)"
        )
        self._widget_core.push_display("status", msg)

    @staticmethod
    def _guess_family(filename: str) -> str:
        lowered = str(filename or "").lower()
        for family in KNOWN_FAMILIES:
            if family in lowered:
                return family
        return ""

    @Slot()
    def _start_probe(self) -> None:
        path_str = self.path_picker.get_path()
        if not path_str or path_str in self._probe_cache:
            return
        threading.Thread(
            target=self._probe_worker, args=(path_str,),
            daemon=True, name="decision-meta-probe",
        ).start()

    def _probe_worker(self, path_str: str) -> None:
        probe: Dict[str, Any] = {"exists": False, "size_gb": 0.0,
                                 "max_ctx": None, "filename": ""}
        try:
            path = Path(path_str)
            if path.is_file():
                probe["exists"] = True
                probe["size_gb"] = path.stat().st_size / (1024 ** 3)
                probe["filename"] = path.name
                probe["max_ctx"] = self._read_max_ctx(path_str)
        except Exception as exc:
            log.debug(f"Decision probe failed for {path_str}: {exc}")
        self._probe_ready.emit(path_str, probe)

    @staticmethod
    def _read_max_ctx(path_str: str) -> Optional[int]:
        try:
            return read_gguf_meta(path_str).context_length
        except Exception as exc:
            log.debug(f"Manual GGUF header parse failed ({exc}); "
                      f"trying gguf package.")
        try:
            return read_gguf_meta_fallback(path_str)[0]
        except ImportError:
            log.warning("The 'gguf' python package is missing, so context "
                        "is not clamped to this model's header.")
        except Exception as exc:
            log.debug(f"Failed to read GGUF metadata: {exc}")
        return None

    @Slot(str, object)
    def _apply_probe(self, path_str: str, probe: Dict[str, Any]) -> None:
        if probe.get("exists", False):
            self._probe_cache[path_str] = probe
            while len(self._probe_cache) > 16:
                self._probe_cache.pop(next(iter(self._probe_cache)))
        if path_str != self.path_picker.get_path():
            return  # stale probe
        max_ctx = probe.get("max_ctx")
        was_blocked = self.spin_ctx.blockSignals(True)
        if isinstance(max_ctx, int) and max_ctx > 0:
            self.spin_ctx.setMaximum(max_ctx)
        self.spin_ctx.blockSignals(was_blocked)
        self._restore_remembered(path_str)
        self._push_estimation(probe)

    def _restore_remembered(self, path_str: str) -> None:
        try:
            preset = self._model_store.lookup(path_str)
        except Exception as exc:
            log.debug(f"Decision settings lookup failed: {exc}")
            return
        if preset is None:
            return
        for widget in (self._combo_device, self.spin_ctx, self.spin_threads,
                       self.spin_batch, self.chk_mmap):
            widget.blockSignals(True)
        try:
            index = self._combo_device.findData(preset.device)
            if index >= 0:
                self._combo_device.setCurrentIndex(index)
            self.spin_ctx.setValue(
                max(self.spin_ctx.minimum(),
                    min(preset.n_ctx, self.spin_ctx.maximum())))
            self.spin_threads.setValue(
                max(self.spin_threads.minimum(),
                    min(preset.n_threads, self.spin_threads.maximum())))
            self.spin_batch.setValue(
                max(self.spin_batch.minimum(),
                    min(preset.n_batch, self.spin_batch.maximum())))
            self.chk_mmap.setChecked(bool(preset.use_mmap))
        finally:
            for widget in (self._combo_device, self.spin_ctx,
                           self.spin_threads, self.spin_batch, self.chk_mmap):
                widget.blockSignals(False)
        self._widget_core.push_display(
            "status",
            f"Restored saved settings for {Path(path_str).name} "
            f"({preset.device.upper()}, ctx {self.spin_ctx.value()}).")

    @Slot(str)
    def _on_value_changed(self, port_name: str) -> None:
        self._update_estimation()
        self.on_ui_change()
        if port_name not in ("status", "btn_load"):
            # The settings are the server's: a new handle, a new server.
            self.set_dirty(port_name)

    @Slot(str, object)
    def _on_port_value_written(self, port_name: str, value: Any) -> None:
        if port_name == "model_path":
            self.path_picker.set_path(value)
        self._update_estimation()

    # ------------------------------------------------------------------
    # Load / eject
    # ------------------------------------------------------------------

    def _is_model_loaded(self) -> bool:
        return self.model_pool is not None and self.model_pool.started

    def _unload_model(self) -> None:
        """Stop the server. The handle stays: its next use starts it."""
        if self.model_pool is None or not self.model_pool.started:
            return
        log.info(f"Unloading decision model for {self.node_name}...")
        self.model_pool.cleanup()
        import gc as _gc
        _gc.collect()
        self.emit_stream("pool_info", self.model_pool.snapshot())
        self.on_ui_change()
        self._update_button_idle()

    def _on_unload_clicked(self) -> None:
        self._unload_model()
        # Downstream holds the stopped server's address: re-evaluating
        # gives it the handle again, and the next question starts it.
        self.set_dirty("unloaded")

    def _start_notice(self, result: Any) -> None:
        """The lazy pool's on_start: any thread, so through a signal."""
        self._pool_started.emit(result)

    @Slot(object)
    def _on_pool_started(self, result: Any) -> None:
        if isinstance(result, BaseException):
            self._widget_core.push_display(
                "status", f"Decision server failed to start: {result}")
        else:
            self._widget_core.push_display(
                "status", f"Loaded {Path(result.model_path).name} "
                          f"at {result.base_url}.")
            self.emit_stream("pool_info", result.snapshot())
        self._update_button_idle()

    def _update_button_idle(self) -> None:
        self.btn_load.setEnabled(self._is_model_loaded())

    def _cleanup_after_worker(self, skip_results: bool = False) -> None:
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

        model_path = inputs.get("model_path")
        if not model_path:
            self.compute_error.emit("No model path provided.")
            return {"model_obj": None}
        try:
            model_path = str(Path(model_path).resolve())
        except Exception as exc:
            self.compute_error.emit(f"Path resolution error: {exc}")
            return {"model_obj": None}
        if not Path(model_path).exists():
            self.compute_error.emit(f"Invalid path: {model_path}")
            return {"model_obj": None}

        device = str(inputs.get("device") or "cpu").strip().lower()
        if device not in DEVICE_GPU_LAYERS:
            log.warning(f"Unknown device {device!r}; using CPU.")
            device = "cpu"
        threads = inputs.get("n_threads", 8)

        llama_kwargs = {
            "n_ctx": inputs.get("n_ctx", 2048),
            "n_gpu_layers": DEVICE_GPU_LAYERS[device],
            "n_threads": None if threads == -1 else threads,
            "n_batch": inputs.get("n_batch", 512),
            "use_mmap": inputs.get("use_mmap", True),
            "verbose": True,
        }

        if self.is_compute_cancelled():
            return {"model_obj": None}

        key = (model_path, tuple(sorted(llama_kwargs.items())))
        if self.model_pool is None or key != self._pool_key:
            if self.model_pool is not None:
                self.model_pool.cleanup()   # settings changed: a new server
            remembered = dict(
                device=device,
                n_ctx=int(inputs.get("n_ctx", 2048)),
                n_threads=int(inputs.get("n_threads", 8)),
                n_batch=int(inputs.get("n_batch", 512)),
                use_mmap=bool(inputs.get("use_mmap", True)),
            )
            # Nothing starts here: the server waits for its first question.
            self.model_pool = LazyGGUFPool(
                model_path, on_start=self._start_notice,
                check=lambda pool: self._smoke_check(pool, model_path, remembered),
                n_instances=1, **llama_kwargs,
            )
            self._pool_key = key

        pool_info = self.model_pool.snapshot()
        self._pending_pool_info = pool_info
        self._sync_status = (
            f"{Path(model_path).name} on {device.upper()}: "
            + ("loaded." if self.model_pool.started
               else "loads when a Classifier or Ensemble first asks it."))
        handle: Dict[str, Any] = {"backend": "systemone",
                                  "pool": self.model_pool,
                                  "role": "decision"}
        return {"model_obj": handle, "pool_info": pool_info}

    def _smoke_check(self, pool: GGUFModelPool, model_path: str,
                     remembered: Dict[str, Any]) -> None:
        """The started server, before anyone else gets it: is it a
        *decision* model on a new enough server? A chat GGUF answers 501,
        an old llama.cpp 404 -- either fails the start, with the server's
        own reason, instead of a triage answer. Settings that pass are
        remembered for the file."""
        try:
            SystemOneClient(pool.base_url, pool.model_alias).decide(
                SMOKE_STATE, dict(SMOKE_QUESTIONS))
        except Exception as exc:
            log.error(f"Decision smoke test failed: {exc}")
            raise RuntimeError(f"Not a usable decision model: {exc}") from exc
        try:
            self._model_store.remember(model_path, **remembered)
        except Exception as exc:
            log.debug(f"Could not persist decision settings: {exc}")

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._update_button_idle()
        if getattr(self, "_sync_status", ""):
            self._widget_core.push_display("status", self._sync_status)
        if self._pending_pool_info is not None:
            self.emit_stream("pool_info", self._pending_pool_info)
            self._pending_pool_info = None

    def cleanup(self) -> None:
        self._probe_timer.stop()
        try:
            self._probe_ready.disconnect(self._apply_probe)
        except (RuntimeError, TypeError):
            pass
        self.cancel_compute()
        if self.model_pool is not None:
            self.model_pool.cleanup()
            self.model_pool = None
        super().cleanup()
