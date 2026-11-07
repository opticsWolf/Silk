# -*- coding: utf-8 -*-
"""Silk ONNX Embedder and ONNX Reranker nodes.

Choose an ONNX model for the Knowledge Graph node -- the embedding model
that makes a graph's vectors, or the cross-encoder that reranks its
search results -- and put the choice on a wire.

**Nothing loads here.** An ONNX model never runs in Weave's process: a
second ONNX Runtime beside one already running takes the interpreter
down. These nodes emit a *description* (the embroider model id, device,
precision and cache folder); the Knowledge Graph node passes it to
``okf serve``, which loads the model with embroider in its own process.
That is also why the wires are not ``model_handle`` -- that type is a
model live in Weave, callable here.

The model ids are embroider's registry; the lists are editable, so a
model a newer embroider gained is usable by typing its id.
"""

from typing import Any, ClassVar, Dict, List, Optional, Sequence

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel

from weave.widgetcore import WidgetCore, PortRole
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.widgets.path_history_picker import PathHistoryPicker

from .silk_ports import ONNX_EMBEDDER_TYPE, RERANKER_MODEL_TYPE  # noqa: F401
from ..functions.port_text import text_input
from ..functions.okf_graph import (
    DEVICES, EMBED_PRECISIONS, KNOWN_MODELS, KNOWN_RERANK_MODELS,
    RERANK_PRECISIONS,
)


def onnx_description(inputs: Dict[str, Any], role: str,
                     models: Sequence[str], precisions: Sequence[str]) -> Dict[str, Any]:
    """The wire value: what okf serve needs to load the model itself."""
    device = text_input(inputs, "device", "auto")
    precision = text_input(inputs, "precision", "auto")
    return {
        "runtime": "embroider",
        "role": role,
        "model_id": text_input(inputs, "model_id").strip() or models[0],
        "device": device if device in DEVICES else "auto",
        "precision": precision if precision in precisions else "auto",
        "cache_dir": text_input(inputs, "cache_dir").strip(),
    }


class _OnnxModelNode(ActiveNode):
    """Model, device, precision and cache folder; emits a description."""
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1
    node_version = 1

    #: Set by each subclass.
    _ROLE: ClassVar[str] = ""
    _OUTPUT: ClassVar[str] = ""
    _OUTPUT_TYPE: ClassVar[str] = ""
    _MODELS: ClassVar[Sequence[str]] = ()
    _PRECISIONS: ClassVar[Sequence[str]] = ()
    _MODEL_TIP: ClassVar[str] = ""
    _PRECISION_TIP: ClassVar[str] = ""

    def __init__(self, title: str, **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self.add_output(self._OUTPUT, datatype=self._OUTPUT_TYPE)

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
            datatype="string", default=self._MODELS[0], add_to_layout=False,
        )
        self.add_input("model_id", datatype="string")

        self._device = QComboBox()
        self._device.addItems(list(DEVICES))
        self._device.setToolTip(
            "Where okf runs the model. 'cuda' needs the knowledge-gpu extra; "
            "'auto' takes the GPU when one answers.")
        form.addRow("Device:", self._device)
        self._widget_core.register_widget(
            "device", self._device, role=PortRole.INPUT,
            datatype="string", default="auto", add_to_layout=False,
        )
        self.add_input("device", datatype="string")

        self._precision = QComboBox()
        self._precision.addItems(list(self._PRECISIONS))
        self._precision.setToolTip(self._PRECISION_TIP)
        form.addRow("Precision:", self._precision)
        self._widget_core.register_widget(
            "precision", self._precision, role=PortRole.INPUT,
            datatype="string", default="auto", add_to_layout=False,
        )
        self.add_input("precision", datatype="string")

        self._cache = PathHistoryPicker(mode="folder")
        self._cache.setToolTip(
            "Where embroider keeps downloaded models (a Hugging Face hub "
            "cache). Empty: embroider's default. Shared by every model the "
            "graph's server loads.")
        form.addRow("Cache:", self._cache)
        self._widget_core.register_widget(
            "cache_dir", self._cache, role=PortRole.INPUT,
            datatype="dirpath", default="", add_to_layout=False,
        )
        self.add_input("cache_dir", datatype="dirpath")

        self._label_status = QLabel(
            "Loaded by okf serve (embroider), in its own process.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        described = onnx_description(inputs, self._ROLE, self._MODELS,
                                     self._PRECISIONS)
        self._sync_status = (
            f"{described['model_id']} on {described['device']} "
            f"({described['precision']}). Loaded by okf serve, in its own "
            "process; the first use downloads it if it is not cached.")
        return {self._OUTPUT: described}

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
        if hasattr(self, "_sync_status"):
            self._widget_core.push_display("status", self._sync_status)


@register_node
class SilkOnnxEmbedderNode(_OnnxModelNode):
    """An embroider embedding model for the Knowledge Graph node."""

    node_name: ClassVar[Optional[str]] = "ONNX Embedder"
    node_description: ClassVar[Optional[str]] = (
        "Chooses an ONNX embedding model (embroider) for the Knowledge Graph "
        "node. Described, not loaded: okf serve loads it in its own process."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "onnx", "embedding", "embroider", "okfgraph", "loader",
    ]
    node_icon: ClassVar[Optional[str]] = "vector"

    _ROLE = "embedding"
    _OUTPUT = "onnx_embedder"
    _OUTPUT_TYPE = "onnx_embedder"
    _MODELS = KNOWN_MODELS
    _PRECISIONS = EMBED_PRECISIONS
    _MODEL_TIP = (
        "The embedding model. A graph keeps the model it was embedded with; "
        "wiring a different one into the Knowledge Graph node re-embeds it.")
    _PRECISION_TIP = (
        "auto: FP32 on the CPU, FP16 on a GPU. The precision is part of the "
        "graph's pin, so changing it re-embeds.")

    def __init__(self, title: str = "ONNX Embedder", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)


@register_node
class SilkOnnxRerankerNode(_OnnxModelNode):
    """An embroider cross-encoder for the Knowledge Graph node's search."""

    node_name: ClassVar[Optional[str]] = "ONNX Reranker"
    node_description: ClassVar[Optional[str]] = (
        "Chooses an ONNX reranker (embroider cross-encoder) for the Knowledge "
        "Graph node's search. Described, not loaded: okf serve loads it."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "onnx", "rerank", "embroider", "okfgraph", "loader",
    ]
    node_icon: ClassVar[Optional[str]] = "list-numbers"

    _ROLE = "rerank"
    _OUTPUT = "reranker"
    _OUTPUT_TYPE = "reranker_model"
    _MODELS = KNOWN_RERANK_MODELS
    _PRECISIONS = RERANK_PRECISIONS
    _MODEL_TIP = (
        "The cross-encoder that reorders search results. It writes no "
        "vectors, so changing it never re-embeds the graph.")
    _PRECISION_TIP = "auto: FP32 on the CPU, FP16 on a GPU; q8 is quantized."

    def __init__(self, title: str = "ONNX Reranker", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
