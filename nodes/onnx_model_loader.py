# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

ONNX Loader -- an embroider model as a real, callable model.

The ONNX twin of the GGUF Loader and the Reranker Loader. An ONNX model
never runs in Weave's process (a second ONNX Runtime takes the
interpreter down), so the model is hosted by its own process
(``functions/embroider_server.py``, run by
:class:`functions.embroider_pool.EmbroiderPool`) and the handle the node
puts on the wire speaks to it over HTTP, the way the GGUF loaders'
handles speak to llama.cpp. The Role row picks what the model is:

* **Embedding** -- ``model_obj`` (``model_handle``, ``backend: onnx``):
  ``/v1/embeddings`` at ``pool.base_url``, so
  ``functions.embeddings.embedder_for`` serves it unchanged -- a ToolSet's
  embedding model, memory recall, or any node's own code.
* **Reranker** -- ``reranker`` (``reranker_model``, ``runtime: onnx``):
  ``/v1/rerank`` at the host root, which the Knowledge Graph hands to
  okf's external rerank lane, as it does the Reranker Loader's.

Both outputs stay on the node, each typed for where it may go; the one
the role does not fill carries nothing.

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

#: Role row value -> embroider host kind.
ROLE_KINDS = {"embedding": "embed", "reranker": "rerank"}
#: Role -> the models the Model row offers.
ROLE_MODELS: Dict[str, Sequence[str]] = {
    "embedding": KNOWN_MODELS, "reranker": KNOWN_RERANK_MODELS}
#: Role -> the precisions the Precision row offers.
ROLE_PRECISIONS: Dict[str, Sequence[str]] = {
    "embedding": EMBED_PRECISIONS, "reranker": RERANK_PRECISIONS}
#: Role -> what the status line calls the model.
ROLE_NOUNS = {"embedding": "embedder", "reranker": "reranker"}


def onnx_family(model_id: str) -> str:
    """okf's family label for an embroider model: its name, made legal."""
    text = model_id.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9.-]+", "-", text).strip("-.")[:64] or "onnx-reranker"


def _shipped(role: str, model_id: str) -> Sequence[str]:
    """The precisions *model_id* can be asked for in *role*."""
    if role == "embedding":
        return precisions_for(model_id)
    return RERANK_PRECISIONS


@register_node
class OnnxLoaderNode(ThreadedNode):
    """An embroider embedder or reranker on its own HTTP server."""

    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    node_name: ClassVar[Optional[str]] = "ONNX Loader"
    node_description: ClassVar[Optional[str]] = (
        "An ONNX model (embroider) in its own process: an embedder on "
        "/v1/embeddings for ToolSets, memory and your own nodes, or a "
        "cross-encoder on /v1/rerank for the Knowledge Graph. Loaded on "
        "first use. A Knowledge Graph's embedder is Silk KG Embedder (ONNX)."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "onnx", "embedding", "rerank", "embroider", "loader",
        "knowledge",
    ]
    node_icon: ClassVar[Optional[str]] = "vector"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_version = 1
    node_state_api = 1

    #: The lazy pool started (a pool) or failed to (an exception).
    _pool_started = Signal(object)

    def __init__(self, title: str = "ONNX Loader", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self.add_input("model_id", datatype="string")
        self.add_output("model_obj", datatype="model_handle")
        self.add_output("reranker", datatype="reranker_model")

        self.model_pool: Optional[LazyGGUFPool] = None
        self._pool_key: Optional[tuple] = None
        self._sync_status = ""

        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        # Widget attributes are prefixed: a bare ``_width`` or ``_model``
        # would shadow the node's own geometry and state.
        self._combo_role = QComboBox()
        self._combo_role.addItem("Embedding", userData="embedding")
        self._combo_role.addItem("Reranker", userData="reranker")
        self._combo_role.setToolTip(
            "Embedding: a vector model on model_obj. Reranker: a "
            "cross-encoder on reranker, for the Knowledge Graph.")
        form.addRow("Role:", self._combo_role)
        self._widget_core.register_widget(
            "role", self._combo_role, role=PortRole.INTERNAL,
            datatype="str", default="embedding", add_to_layout=False)

        self._combo_model = QComboBox()
        self._combo_model.setEditable(True)
        self._combo_model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._combo_model.addItems(list(KNOWN_MODELS))
        self._combo_model.setToolTip(
            "The model, from embroider's registry. Retrieval embedders embed "
            "a query and a document differently: callers say which with task "
            "(Query / Document) on /v1/embeddings; Document is the default.")
        form.addRow("Model:", self._combo_model)
        self._widget_core.register_widget(
            "model_id", self._combo_model, role=PortRole.INPUT,
            datatype="string", default=KNOWN_MODELS[0], add_to_layout=False)

        self._combo_device = QComboBox()
        self._combo_device.addItems(list(DEVICES))
        self._combo_device.setToolTip(
            "Where the model runs. 'cuda' needs the GPU ONNX Runtime "
            "(okfgraph[gpu]); 'auto' takes the GPU when one answers.")
        form.addRow("Device:", self._combo_device)
        self._widget_core.register_widget(
            "device", self._combo_device, role=PortRole.INTERNAL,
            datatype="string", default="cpu", add_to_layout=False)
        self._combo_device.setCurrentText("cpu")

        self._combo_precision = QComboBox()
        self._combo_precision.addItems(list(EMBED_PRECISIONS))
        self._combo_precision.setToolTip(
            "auto: FP32 on the CPU, FP16 on a GPU. Only the precisions the "
            "model ships work.")
        form.addRow("Precision:", self._combo_precision)
        self._widget_core.register_widget(
            "precision", self._combo_precision, role=PortRole.INTERNAL,
            datatype="string", default="auto", add_to_layout=False)

        self._combo_dim = QComboBox()
        self._combo_dim.addItem(NATIVE)
        self._combo_dim.setToolTip(
            "The vector width (embedders only). These models are "
            "Matryoshka-trained: a narrower vector is a cut of the full one, "
            "smaller and a little less exact. native: the model's own width.")
        form.addRow("Width:", self._combo_dim)
        self._widget_core.register_widget(
            "dim", self._combo_dim, role=PortRole.INTERNAL,
            datatype="string", default=NATIVE, add_to_layout=False)

        self._picker_cache = PathHistoryPicker(mode="folder")
        self._picker_cache.setToolTip(
            "Where embroider keeps downloaded models (a Hugging Face hub "
            "cache). Empty: embroider's default.")
        form.addRow("Cache:", self._picker_cache)
        self._widget_core.register_widget(
            "cache_dir", self._picker_cache, role=PortRole.INTERNAL,
            datatype="dirpath", default="", add_to_layout=False)

        form.addRow(SeparatorLine())

        self._label_status = QLabel(
            "Runs in its own process; starts when a node uses the model.")
        self._label_status.setEnabled(False)
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False)

        btn_row = QHBoxLayout()
        self.btn_unload = SyncButton(initial_text="Unload Model",
                                     initial_icon="player-eject")
        self.btn_unload.setToolTip(
            "The model loads when a node first uses it. Unload stops its "
            "process; it starts again when needed.")
        self.btn_unload.clicked.connect(self._on_unload_clicked)
        btn_row.addWidget(self.btn_unload)
        form.addRow("", btn_row)
        self._widget_core.register_widget(
            "btn_unload", self.btn_unload, role=PortRole.INTERNAL,
            add_to_layout=False)

        self._combo_role.currentIndexChanged.connect(self._on_role_changed)
        self._combo_model.currentTextChanged.connect(self._sync_choices)
        self._widget_core.value_changed.connect(self._on_value_changed)
        self._pool_started.connect(self._on_pool_started)

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()
        if hasattr(self._widget_core, "refresh_widget_palettes"):
            self._widget_core.refresh_widget_palettes()
        self._sync_choices(self._combo_model.currentText())
        self._update_button_idle()
        self.on_ui_change()

    # -- the rows follow the role and the model ---------------------------

    def _role(self) -> str:
        return str(self._combo_role.currentData() or "embedding")

    @Slot(int)
    def _on_role_changed(self, _index: int) -> None:
        """Offer the role's models; keep a model id typed by hand."""
        role = self._role()
        models = list(ROLE_MODELS[role])
        current = self._combo_model.currentText().strip()
        other = {m for r, ms in ROLE_MODELS.items() if r != role for m in ms}
        self._combo_model.blockSignals(True)
        self._combo_model.clear()
        self._combo_model.addItems(models)
        self._combo_model.setCurrentText(
            models[0] if not current or current in other else current)
        self._combo_model.blockSignals(False)
        precision = self._combo_precision.currentText()
        self._combo_precision.blockSignals(True)
        self._combo_precision.clear()
        self._combo_precision.addItems(list(ROLE_PRECISIONS[role]))
        self._combo_precision.blockSignals(False)
        self._combo_precision.setCurrentText(precision)
        self._sync_choices(self._combo_model.currentText())
        # The model row changed with signals blocked: say so ourselves.
        self.set_dirty("model_id")

    def _sync_choices(self, model_id: str) -> None:
        """Offer *model_id*'s widths and grey out precisions it lacks."""
        role = self._role()
        model_id = model_id.strip()
        current = self._combo_dim.currentText()
        ladder = MODEL_LADDERS.get(model_id, ()) if role == "embedding" else ()
        widths = [NATIVE, *(str(w) for w in ladder)]
        self._combo_dim.blockSignals(True)
        self._combo_dim.clear()
        self._combo_dim.addItems(widths)
        self._combo_dim.setCurrentText(current if current in widths else NATIVE)
        self._combo_dim.blockSignals(False)
        self._combo_dim.setEnabled(role == "embedding")
        shipped = _shipped(role, model_id)
        items = self._combo_precision.model()
        for row in range(self._combo_precision.count()):
            item = items.item(row) if hasattr(items, "item") else None
            if item is not None:
                item.setEnabled(self._combo_precision.itemText(row) in shipped)
        if self._combo_precision.currentText() not in shipped:
            self._combo_precision.setCurrentText("auto")

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
                "status", f"The model did not start: {result}")
        else:
            self._widget_core.push_display("status", _loaded_text(result))
        self._update_button_idle()

    def _update_button_idle(self) -> None:
        self.btn_unload.setEnabled(self._is_loaded())

    def _cleanup_after_worker(self, skip_results: bool = False) -> None:
        self._update_button_idle()
        super()._cleanup_after_worker(skip_results=skip_results)

    # -- compute --------------------------------------------------------

    @staticmethod
    def _settings(inputs: Dict[str, Any], role: str,
                  model_id: str) -> Dict[str, Any]:
        device = text_input(inputs, "device", "cpu")
        precision = text_input(inputs, "precision", "auto")
        if precision not in ROLE_PRECISIONS[role]:
            precision = "auto"
        if precision not in _shipped(role, model_id):
            raise ValueError(
                f"{model_id} ships no {precision} weights; pick one of "
                f"{', '.join(_shipped(role, model_id))}.")
        settings: Dict[str, Any] = {
            "kind": ROLE_KINDS[role],
            "device": device if device in DEVICES else "cpu",
            "precision": precision,
            "cache_dir": text_input(inputs, "cache_dir").strip()}
        if role == "embedding":
            width = text_input(inputs, "dim", NATIVE).strip()
            settings["dim"] = int(width) if width.isdigit() else 0
        return settings

    def _handle(self, role: str, model_id: str) -> Dict[str, Any]:
        if role == "embedding":
            return {"backend": "onnx", "role": "embedding",
                    "pool": self.model_pool, "model_id": model_id}
        return {"role": "rerank", "runtime": "onnx", "pool": self.model_pool,
                "model_id": f"onnx/{model_id}", "family": onnx_family(model_id),
                "max_tokens": RERANK_MODEL_MAX_TOKENS.get(model_id, 512)}

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        empty = {"model_obj": None, "reranker": None}
        if not embroider_available():
            self.compute_error.emit(
                "embroider is not installed: install Silk's knowledge extra "
                "(okfgraph[cpu] or okfgraph[gpu]).")
            return empty
        role = text_input(inputs, "role", "embedding")
        if role not in ROLE_KINDS:
            role = "embedding"
        model_id = text_input(inputs, "model_id").strip() or ROLE_MODELS[role][0]
        settings = self._settings(inputs, role, model_id)
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
            f"{ROLE_NOUNS[role].capitalize()} {model_id} on "
            f"{settings['device']}: loads when a node uses it.")
        output = "model_obj" if role == "embedding" else "reranker"
        return {**empty, output: self._handle(role, model_id)}

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["cache_history"] = self._picker_cache.history()
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        with self._widget_core.suppress_signals():
            super().restore_state(state)
        self._picker_cache.set_history(
            [str(p) for p in state.get("cache_history") or ()])

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._update_button_idle()
        if self._sync_status:
            self._widget_core.push_display("status", self._sync_status)

    def cleanup(self) -> None:
        try:
            self._pool_started.disconnect(self._on_pool_started)
            self._combo_role.currentIndexChanged.disconnect(self._on_role_changed)
            self._combo_model.currentTextChanged.disconnect(self._sync_choices)
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
