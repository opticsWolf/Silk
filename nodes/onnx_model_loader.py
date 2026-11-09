# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

ONNX Embedder Loader and ONNX Reranker Loader -- embroider models as
real, callable models.

The ONNX twins of the Embedder Loader and the Reranker Loader. An ONNX
model never runs in Weave's process (a second ONNX Runtime takes the
interpreter down), so the model is hosted by its own process
(``functions/embroider_server.py``, run by
:class:`functions.embroider_pool.EmbroiderPool`) and the handle these
nodes put on the wire speaks to it over HTTP, the way the GGUF loaders'
handles speak to llama.cpp:

* **ONNX Embedder Loader** -- ``model_obj`` (``model_handle``,
  ``backend: onnx``): ``/v1/embeddings`` at ``pool.base_url``, so
  ``functions.embeddings.embedder_for`` serves it unchanged -- a ToolSet's
  embedding model, memory recall, or any node's own code.
* **ONNX Reranker Loader** -- ``reranker`` (``reranker_model``,
  ``runtime: onnx``): ``/v1/rerank`` at the host root, which the
  Knowledge Graph hands to okf's external rerank lane, as it does the
  Reranker Loader's.

**Not the Knowledge Graph's embedder.** A graph's vectors are made by
okf itself, so its ONNX embedding model is chosen with Silk KG Embedder
(ONNX) (``nodes/onnx_loader.py``), a specification okf loads; the KG
refuses this loader's handle on its ``embedder`` port and says so.

**Loaded on demand,** like the GGUF loaders: the handle goes out at once
(a :class:`~functions.model_pool.LazyGGUFPool`), the process starts the
first time something uses the model, and Unload stops it.
"""

import re
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel

from weave.widgetcore import WidgetCore, PortRole
from weave.node.threaded import ThreadedNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from weave.widgets.sync_button import SyncButton
from weave.widgets.path_history_picker import PathHistoryPicker
from weave.widgets.separator_line import SeparatorLine

from .silk_ports import RERANKER_MODEL_TYPE  # noqa: F401
from ..functions.embroider_pool import EmbroiderPool, embroider_available
from ..functions.model_pool import LazyGGUFPool
from ..functions.okf_graph import (
    DEVICES, EMBED_PRECISIONS, KNOWN_MODELS, KNOWN_RERANK_MODELS,
    MODEL_LADDERS, RERANK_MODEL_MAX_TOKENS, RERANK_PRECISIONS, precisions_for,
)
from ..functions.port_text import text_input

log = get_logger("OnnxModelLoader")

#: The Width row's first entry: the model's own width.
NATIVE = "native"


def onnx_family(model_id: str) -> str:
    """okf's family label for an embroider model: its name, made legal."""
    text = model_id.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9.-]+", "-", text).strip("-.")[:64] or "onnx-reranker"


class _OnnxLoaderNode(ThreadedNode):
    """Model, device, precision and cache folder; hands out a lazy pool."""

    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_version = 1
    node_state_api = 1

    #: The lazy pool started (a pool) or failed to (an exception).
    _pool_started = Signal(object)

    #: Set by each subclass.
    _KIND: ClassVar[str] = ""
    _OUTPUT: ClassVar[str] = ""
    _OUTPUT_TYPE: ClassVar[str] = ""
    _MODELS: ClassVar[Sequence[str]] = ()
    _PRECISIONS: ClassVar[Sequence[str]] = ()
    _MODEL_TIP: ClassVar[str] = ""
    _NOUN: ClassVar[str] = ""

    def __init__(self, title: str, **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self.add_output(self._OUTPUT, datatype=self._OUTPUT_TYPE)

        self.model_pool: Optional[LazyGGUFPool] = None
        self._pool_key: Optional[tuple] = None
        self._sync_status = ""

        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self._model = QComboBox()
        self._model.setEditable(True)
        self._model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._model.addItems(list(self._MODELS))
        self._model.setToolTip(self._MODEL_TIP)
        form.addRow("Model:", self._model)
        self._widget_core.register_widget(
            "model_id", self._model, role=PortRole.INPUT,
            datatype="string", default=self._MODELS[0], add_to_layout=False)
        self.add_input("model_id", datatype="string")

        self._device = QComboBox()
        self._device.addItems(list(DEVICES))
        self._device.setToolTip(
            "Where the model runs. 'cuda' needs the GPU ONNX Runtime "
            "(okfgraph[gpu]); 'auto' takes the GPU when one answers.")
        form.addRow("Device:", self._device)
        self._widget_core.register_widget(
            "device", self._device, role=PortRole.INTERNAL,
            datatype="string", default="cpu", add_to_layout=False)
        self._device.setCurrentText("cpu")

        self._precision = QComboBox()
        self._precision.addItems(list(self._PRECISIONS))
        self._precision.setToolTip(
            "auto: FP32 on the CPU, FP16 on a GPU. Only the precisions the "
            "model ships work.")
        form.addRow("Precision:", self._precision)
        self._widget_core.register_widget(
            "precision", self._precision, role=PortRole.INTERNAL,
            datatype="string", default="auto", add_to_layout=False)

        self._extra_rows(form)

        self._cache = PathHistoryPicker(mode="folder")
        self._cache.setToolTip(
            "Where embroider keeps downloaded models (a Hugging Face hub "
            "cache). Empty: embroider's default.")
        form.addRow("Cache:", self._cache)
        self._widget_core.register_widget(
            "cache_dir", self._cache, role=PortRole.INTERNAL,
            datatype="dirpath", default="", add_to_layout=False)

        form.addRow(SeparatorLine())

        self._label_status = QLabel(
            f"Runs in its own process; starts when a node uses the {self._NOUN}.")
        self._label_status.setEnabled(False)
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False)

        btn_row = QHBoxLayout()
        self.btn_unload = SyncButton(initial_text=f"Unload {self._NOUN.title()}",
                                     initial_icon="player-eject")
        self.btn_unload.setToolTip(
            f"The {self._NOUN} loads when a node first uses it. Unload stops "
            "its process; it starts again when needed.")
        self.btn_unload.clicked.connect(self._on_unload_clicked)
        btn_row.addWidget(self.btn_unload)
        form.addRow("", btn_row)
        self._widget_core.register_widget(
            "btn_unload", self.btn_unload, role=PortRole.INTERNAL,
            add_to_layout=False)

        self._widget_core.value_changed.connect(self._on_value_changed)
        self._pool_started.connect(self._on_pool_started)

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()
        if hasattr(self._widget_core, "refresh_widget_palettes"):
            self._widget_core.refresh_widget_palettes()
        self._update_button_idle()
        self.on_ui_change()

    def _extra_rows(self, form: QFormLayout) -> None:
        """Rows a subclass adds below Precision."""

    @Slot(str)
    def _on_value_changed(self, port_name: str) -> None:
        self.on_ui_change()
        if port_name not in ("status", "btn_unload"):
            self.set_dirty(port_name)

    # -- load / unload --------------------------------------------------

    def _is_loaded(self) -> bool:
        return self.model_pool is not None and self.model_pool.started

    def _on_unload_clicked(self) -> None:
        if self._is_loaded():
            self.model_pool.cleanup()
            self._widget_core.push_display(
                "status", "Unloaded; starts again when a node needs it.")
        self._update_button_idle()
        # Downstream holds the stopped process's address: re-evaluating
        # hands the handle out again, and its next use starts a process.
        self.set_dirty("unloaded")

    def _start_notice(self, result: Any) -> None:
        """The lazy pool's on_start: any thread, so through a signal."""
        self._pool_started.emit(result)

    @Slot(object)
    def _on_pool_started(self, result: Any) -> None:
        if isinstance(result, BaseException):
            self._widget_core.push_display(
                "status", f"The {self._NOUN} did not start: {result}")
        else:
            self._widget_core.push_display("status", _loaded_text(result))
        self._update_button_idle()

    def _update_button_idle(self) -> None:
        self.btn_unload.setEnabled(self._is_loaded())

    def _cleanup_after_worker(self, skip_results: bool = False) -> None:
        self._update_button_idle()
        super()._cleanup_after_worker(skip_results=skip_results)

    # -- compute --------------------------------------------------------

    def _settings(self, inputs: Dict[str, Any], model_id: str) -> Dict[str, Any]:
        device = text_input(inputs, "device", "cpu")
        precision = text_input(inputs, "precision", "auto")
        return {"kind": self._KIND,
                "device": device if device in DEVICES else "cpu",
                "precision": precision if precision in self._PRECISIONS else "auto",
                "cache_dir": text_input(inputs, "cache_dir").strip()}

    def _handle(self, model_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not embroider_available():
            self.compute_error.emit(
                "embroider is not installed: install Silk's knowledge extra "
                "(okfgraph[cpu] or okfgraph[gpu]).")
            return {self._OUTPUT: None}
        model_id = text_input(inputs, "model_id").strip() or self._MODELS[0]
        settings = self._settings(inputs, model_id)
        key = (model_id, tuple(sorted(settings.items())))
        if self.model_pool is None or key != self._pool_key:
            if self.model_pool is not None:
                self.model_pool.cleanup()   # settings changed: a new process
            # Nothing starts here: the process waits for its first user.
            self.model_pool = LazyGGUFPool(
                model_id, on_start=self._start_notice,
                pool_class=EmbroiderPool, **settings)
            self._pool_key = key
        self._sync_status = (
            _loaded_text(self.model_pool) if self.model_pool.started else
            f"{model_id} on {settings['device']}: loads when a node uses it.")
        return {self._OUTPUT: self._handle(model_id, settings)}

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["cache_history"] = self._cache.history()
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        with self._widget_core.suppress_signals():
            super().restore_state(state)
        self._cache.set_history([str(p) for p in state.get("cache_history") or ()])

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._update_button_idle()
        if self._sync_status:
            self._widget_core.push_display("status", self._sync_status)

    def cleanup(self) -> None:
        try:
            self._pool_started.disconnect(self._on_pool_started)
        except (RuntimeError, TypeError):
            pass
        self.cancel_compute()
        if self.model_pool is not None:
            self.model_pool.cleanup()
            self.model_pool = None
        super().cleanup()


def _loaded_text(pool: Any) -> str:
    """One line on a running pool: model, width, precision, where."""
    info = pool.snapshot()
    parts = [str(info.get("model_path") or "")]
    if info.get("dim"):
        parts.append(f"{info['dim']}-dim")
    if info.get("precision"):
        parts.append(str(info["precision"]))
    parts.append("CUDA" if info.get("cuda") else "CPU")
    return f"Loaded {', '.join(p for p in parts if p)} at {info.get('url', '')}."


@register_node
class OnnxEmbedderLoaderNode(_OnnxLoaderNode):
    """An embroider embedding model on its own /v1/embeddings server."""

    node_name: ClassVar[Optional[str]] = "ONNX Embedder Loader"
    node_description: ClassVar[Optional[str]] = (
        "An ONNX embedding model (embroider) served on /v1/embeddings in its "
        "own process, for ToolSets, memory and your own nodes; loaded on "
        "first use. A Knowledge Graph's model is Silk KG Embedder (ONNX)."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "onnx", "embedding", "embroider", "loader",
    ]
    node_icon: ClassVar[Optional[str]] = "vector"

    _KIND = "embed"
    _OUTPUT = "model_obj"
    _OUTPUT_TYPE = "model_handle"
    _MODELS = KNOWN_MODELS
    _PRECISIONS = EMBED_PRECISIONS
    _NOUN = "embedder"
    _MODEL_TIP = (
        "The embedding model, from embroider's registry. Retrieval models "
        "embed a query and a document differently: callers say which with "
        "task (Query / Document) on /v1/embeddings; Document is the default.")

    def __init__(self, title: str = "ONNX Embedder Loader", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self._model.currentTextChanged.connect(self._sync_choices)
        self._sync_choices(self._model.currentText())

    def _extra_rows(self, form: QFormLayout) -> None:
        self._width = QComboBox()
        self._width.setToolTip(
            "The vector width. These models are Matryoshka-trained: a "
            "narrower vector is a cut of the full one, smaller and a little "
            "less exact. native: the model's own width.")
        form.addRow("Width:", self._width)
        self._widget_core.register_widget(
            "dim", self._width, role=PortRole.INTERNAL,
            datatype="string", default=NATIVE, add_to_layout=False)

    def _sync_choices(self, model_id: str) -> None:
        """Offer *model_id*'s widths and grey out precisions it lacks."""
        model_id = model_id.strip()
        current = self._width.currentText()
        widths = [NATIVE, *(str(w) for w in MODEL_LADDERS.get(model_id, ()))]
        self._width.blockSignals(True)
        self._width.clear()
        self._width.addItems(widths)
        self._width.setCurrentText(current if current in widths else NATIVE)
        self._width.blockSignals(False)
        shipped = precisions_for(model_id)
        items = self._precision.model()
        for row in range(self._precision.count()):
            item = items.item(row) if hasattr(items, "item") else None
            if item is not None:
                item.setEnabled(self._precision.itemText(row) in shipped)
        if self._precision.currentText() not in shipped:
            self._precision.setCurrentText("auto")

    def _settings(self, inputs: Dict[str, Any], model_id: str) -> Dict[str, Any]:
        settings = super()._settings(inputs, model_id)
        precision = settings["precision"]
        if precision not in precisions_for(model_id):
            raise ValueError(
                f"{model_id} ships no {precision} weights; pick one of "
                f"{', '.join(precisions_for(model_id))}.")
        width = text_input(inputs, "dim", NATIVE).strip()
        settings["dim"] = int(width) if width.isdigit() else 0
        return settings

    def _handle(self, model_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        return {"backend": "onnx", "role": "embedding", "pool": self.model_pool,
                "model_id": model_id}


@register_node
class OnnxRerankerLoaderNode(_OnnxLoaderNode):
    """An embroider cross-encoder on its own /v1/rerank server."""

    node_name: ClassVar[Optional[str]] = "ONNX Reranker Loader"
    node_description: ClassVar[Optional[str]] = (
        "An ONNX cross-encoder (embroider) served on /v1/rerank in its own "
        "process, for the Knowledge Graph and your own nodes; loaded on "
        "first use."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "onnx", "rerank", "embroider", "loader", "knowledge",
    ]
    node_icon: ClassVar[Optional[str]] = "list-numbers"

    _KIND = "rerank"
    _OUTPUT = "reranker"
    _OUTPUT_TYPE = "reranker_model"
    _MODELS = KNOWN_RERANK_MODELS
    _PRECISIONS = RERANK_PRECISIONS
    _NOUN = "reranker"
    _MODEL_TIP = (
        "The cross-encoder that scores query + document pairs, from "
        "embroider's registry.")

    def __init__(self, title: str = "ONNX Reranker Loader", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

    def _handle(self, model_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        return {"role": "rerank", "runtime": "onnx", "pool": self.model_pool,
                "model_id": f"onnx/{model_id}", "family": onnx_family(model_id),
                "max_tokens": RERANK_MODEL_MAX_TOKENS.get(model_id, 512)}
