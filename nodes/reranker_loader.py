# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Reranker Loader -- a GGUF reranker for the Knowledge Graph.

The Embedder Loader's twin for the second retrieval stage. A GGUF
reranker (bge-reranker-v2-m3, Qwen3-Reranker, …) runs in its own process
behind ``POST /v1/rerank`` (``functions/rerank_server.py``, which scores
pairs with llama.cpp itself), and the Knowledge Graph hands that host to
OKFgraph's external rerank lane. The other ways to rerank are embroider's
cross-encoders: Silk KG Reranker (ONNX), loaded by okf in its own
process, or the ONNX Reranker Loader, hosted by Silk like this one.

**Loaded on demand,** like the Embedder Loader: the node hands out its
``reranker`` at once and the server starts when the Knowledge Graph
serves a graph with it. Unload stops the server; the next use starts it.

The context is the pair ceiling: a query and a candidate longer than it
are cut (the document's tail goes), and okf is told the same number so it
splits long candidates the way the server will read them.
"""

import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QSpinBox,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.node.threaded import ThreadedNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from weave.widgets.sync_button import SyncButton
from weave.widgets.path_picker import PathPickerWidget
from weave.widgets.separator_line import SeparatorLine

from .silk_ports import RERANKER_MODEL_TYPE  # noqa: F401
from ..functions.gguf_meta import read_embedding_meta
from ..functions.model_pool import LLAMA_CPP_AVAILABLE, LazyGGUFPool
from ..functions.rerank_pool import RerankPool

log = get_logger("RerankerLoader")

#: Device dropdown value -> llama.cpp GPU-layer setting.
DEVICE_GPU_LAYERS = {"cpu": 0, "gpu": -1}


def rerank_family(path: str) -> str:
    """okf's family label for a reranker file: its name, made a legal
    label (lower case, no ``/``, at most 64 characters)."""
    try:
        name = read_embedding_meta(path).name or ""
    except Exception:  # noqa: BLE001 - the file name will do
        name = ""
    text = (name or Path(path).stem).lower().replace("_", "-").replace(" ", "-")
    return re.sub(r"[^a-z0-9.-]+", "-", text).strip("-.")[:64] or "gguf-reranker"


@register_node
class RerankerLoaderNode(ThreadedNode):
    """A GGUF reranker on its own /v1/rerank server, started on first use."""

    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    node_name: ClassVar[Optional[str]] = "Reranker Loader"
    node_description: ClassVar[Optional[str]] = (
        "A GGUF reranker (bge-reranker, Qwen3-Reranker) served on "
        "/v1/rerank for the Knowledge Graph; loaded on first use."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "llama", "rerank", "gguf", "knowledge",
    ]
    node_icon: ClassVar[Optional[str]] = "list-numbers"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT

    #: The lazy pool started (a pool) or failed to (an exception).
    _pool_started = Signal(object)

    node_version = 1
    node_state_api = 1

    def __init__(self, title: str = "Reranker Loader", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        self.add_input("model_path", datatype="filepath")
        self.add_output("reranker", datatype="reranker_model")

        self.model_pool: Optional[LazyGGUFPool] = None
        self._pool_key: Optional[tuple] = None
        self._sync_status = ""

        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self.path_picker = PathPickerWidget(
            mode="file", filters=["GGUF Models (*.gguf)", "All Files (*)"])
        self.path_picker.setToolTip(
            "The reranker .gguf file: a cross-encoder with a classification "
            "head (bge-reranker-v2-m3) or a Qwen3-Reranker.")
        form.addRow("Model Path:", self.path_picker)
        self._widget_core.register_widget(
            "model_path", self.path_picker, role=PortRole.BIDIRECTIONAL,
            datatype="filepath", default="", add_to_layout=False)

        self._combo_device = QComboBox()
        self._combo_device.addItem("CPU", userData="cpu")
        self._combo_device.addItem("GPU", userData="gpu")
        self._combo_device.setToolTip(
            "CPU keeps the reranker in RAM and VRAM for the chat model; GPU "
            "offloads it fully. Reranking scores ~30 pairs per search.")
        form.addRow("Device:", self._combo_device)
        self._widget_core.register_widget(
            "device", self._combo_device, role=PortRole.INTERNAL,
            datatype="str", default="cpu", add_to_layout=False)

        self.spin_ctx = QSpinBox()
        self.spin_ctx.setRange(256, 32768)
        self.spin_ctx.setValue(2048)
        self.spin_ctx.setToolTip(
            "Tokens per query + candidate pair. Longer candidates are cut, "
            "and the Knowledge Graph is told this ceiling.")
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

        self.chk_mmap = QCheckBox()
        self.chk_mmap.setChecked(True)
        form.addRow("Use mmap():", self.chk_mmap)
        self._widget_core.register_widget(
            "use_mmap", self.chk_mmap, role=PortRole.INTERNAL,
            datatype="bool", default=True, add_to_layout=False)

        form.addRow(SeparatorLine())

        self._label_status = QLabel("Select a reranker model.")
        self._label_status.setEnabled(False)
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False)

        btn_row = QHBoxLayout()
        self.btn_unload = SyncButton(initial_text="Unload Reranker",
                                     initial_icon="player-eject")
        self.btn_unload.setToolTip(
            "The reranker loads when the Knowledge Graph first uses it. "
            "Unload stops its server; it starts again when needed.")
        self.btn_unload.clicked.connect(self._on_unload_clicked)
        btn_row.addWidget(self.btn_unload)
        form.addRow("", btn_row)
        self._widget_core.register_widget(
            "btn_unload", self.btn_unload, role=PortRole.INTERNAL,
            add_to_layout=False)

        self._widget_core.value_changed.connect(self._on_value_changed)
        self._widget_core.port_value_written.connect(self._on_port_value_written)
        self._pool_started.connect(self._on_pool_started)

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()
        if hasattr(self._widget_core, "refresh_widget_palettes"):
            self._widget_core.refresh_widget_palettes()
        self._update_button_idle()
        self.on_ui_change()

    @Slot(str)
    def _on_value_changed(self, port_name: str) -> None:
        self.on_ui_change()
        if port_name not in ("status", "btn_unload"):
            self.set_dirty(port_name)

    @Slot(str, object)
    def _on_port_value_written(self, port_name: str, value: Any) -> None:
        if port_name == "model_path":
            self.path_picker.set_path(value)

    # -- load / unload --------------------------------------------------

    def _is_loaded(self) -> bool:
        return self.model_pool is not None and self.model_pool.started

    def _on_unload_clicked(self) -> None:
        if self._is_loaded():
            self.model_pool.cleanup()
            self._widget_core.push_display(
                "status", "Unloaded; starts again when the Knowledge Graph needs it.")
        self._update_button_idle()
        # Downstream holds the stopped server's address: re-evaluating
        # hands the handle out again, and its next use starts a server.
        self.set_dirty("unloaded")

    def _start_notice(self, result: Any) -> None:
        """The lazy pool's on_start: any thread, so through a signal."""
        self._pool_started.emit(result)

    @Slot(object)
    def _on_pool_started(self, result: Any) -> None:
        if isinstance(result, BaseException):
            self._widget_core.push_display(
                "status", f"Reranker server failed to start: {result}")
        else:
            self._widget_core.push_display(
                "status", f"Loaded {Path(result.model_path).name} at {result.base_url}.")
        self._update_button_idle()

    def _update_button_idle(self) -> None:
        self.btn_unload.setEnabled(self._is_loaded())

    def _cleanup_after_worker(self, skip_results: bool = False) -> None:
        self._update_button_idle()
        super()._cleanup_after_worker(skip_results=skip_results)

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not LLAMA_CPP_AVAILABLE:
            self.compute_error.emit("llama-cpp-python is not installed.")
            return {"reranker": None}
        model_path = inputs.get("model_path")
        if not model_path:
            self.compute_error.emit("No model path provided.")
            return {"reranker": None}
        path = Path(str(model_path)).expanduser()
        if not path.is_file():
            self.compute_error.emit(f"Invalid path: {model_path}")
            return {"reranker": None}
        model_path = str(path.resolve())

        device = str(inputs.get("device") or "cpu").strip().lower()
        if device not in DEVICE_GPU_LAYERS:
            device = "cpu"
        threads = inputs.get("n_threads", 8)
        settings = {
            "n_ctx": int(inputs.get("n_ctx") or 2048),
            "n_gpu_layers": DEVICE_GPU_LAYERS[device],
            "n_threads": None if threads in (-1, None) else int(threads),
            "use_mmap": bool(inputs.get("use_mmap", True)),
        }
        key = (model_path, tuple(sorted(settings.items())))
        if self.model_pool is None or key != self._pool_key:
            if self.model_pool is not None:
                self.model_pool.cleanup()   # settings changed: a new server
            self.model_pool = LazyGGUFPool(
                model_path, on_start=self._start_notice,
                pool_class=RerankPool, **settings)
            self._pool_key = key

        self._sync_status = (
            f"{path.name} on {device.upper()}: "
            + ("loaded." if self.model_pool.started
               else "loads when the Knowledge Graph uses it."))
        return {"reranker": {
            "role": "rerank", "runtime": "gguf", "pool": self.model_pool,
            "model_id": f"gguf/{path.stem}", "family": rerank_family(model_path),
            "max_tokens": settings["n_ctx"],
        }}

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._update_button_idle()
        if self._sync_status:
            self._widget_core.push_display("status", self._sync_status)

    def cleanup(self) -> None:
        self.cancel_compute()
        if self.model_pool is not None:
            self.model_pool.cleanup()
            self.model_pool = None
        super().cleanup()
