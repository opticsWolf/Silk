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

The Embedder also chooses the image model for image-content search. An
image model only pairs with one text model (its registry partner), so
the node greys out what the chosen text model cannot use: other image
models, and precisions the model ships no weights for.
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
    DEVICES, EMBED_PRECISIONS, IMAGE_PRECISIONS, KNOWN_MODELS,
    KNOWN_RERANK_MODELS, RERANK_PRECISIONS, VISION_MODELS, precisions_for,
    vision_partner,
)

#: The Image model row's first entry: okf's own pairing.
IMAGE_AUTO = "auto"


def onnx_description(inputs: Dict[str, Any], role: str,
                     models: Sequence[str], precisions: Sequence[str]) -> Dict[str, Any]:
    """The wire value: what okf serve needs to load the model itself."""
    device = text_input(inputs, "device", "auto")
    precision = text_input(inputs, "precision", "auto")
    model_id = text_input(inputs, "model_id").strip() or models[0]
    if role == "embedding" and precision in precisions:
        shipped = precisions_for(model_id)
        if precision not in shipped:
            raise ValueError(
                f"{model_id} ships no {precision} weights; pick one of "
                f"{', '.join(shipped)}.")
    return {
        "runtime": "embroider",
        "role": role,
        "model_id": model_id,
        "device": device if device in DEVICES else "auto",
        "precision": precision if precision in precisions else "auto",
        "cache_dir": text_input(inputs, "cache_dir").strip(),
    }


def image_description(inputs: Dict[str, Any], model_id: str) -> Dict[str, str]:
    """The image rows of an embedder description: ``""`` is okf's pairing.

    An image model's vectors only compare against its text partner, so a
    known image model wired beside another text model is refused here
    rather than by okf at the first image search.
    """
    image = text_input(inputs, "image_model_id").strip()
    if image == IMAGE_AUTO:
        image = ""
    partner = VISION_MODELS.get(image)
    if partner and partner != model_id:
        raise ValueError(
            f"{image} embeds images for {partner} graphs, not {model_id}. "
            f"Pick 'auto' or {vision_partner(model_id) or 'a text model it pairs with'}.")
    precision = text_input(inputs, "image_precision", "auto")
    return {"image_model_id": image,
            "image_precision": precision if precision in IMAGE_PRECISIONS else "auto"}


def image_text(described: Dict[str, Any]) -> str:
    """One line on how the description embeds images."""
    image = described.get("image_model_id") or vision_partner(described["model_id"])
    if not image:
        return ("No image model pairs with this text model: images are "
                "searched by their captions.")
    how = "named" if described.get("image_model_id") else "okf's pairing"
    return f"Images: {image} ({how}, {described.get('image_precision') or 'auto'})."


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

        self._extra_rows(form)

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

    def _extra_rows(self, form: QFormLayout) -> None:
        """Rows a subclass adds above Info."""

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
    node_state_api = 2

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
        "graph's pin, so changing it re-embeds. Only the precisions the "
        "model ships can be picked (int8: jina nano).")

    def __init__(self, title: str = "ONNX Embedder", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self._model.currentTextChanged.connect(self._sync_choices)
        self._sync_choices(self._model.currentText())

    @classmethod
    def migrate_state(cls, state: Dict[str, Any],
                      from_version: int) -> Dict[str, Any]:
        """api 1 → 2: the image rows are new; a scene saved before them
        gets okf's pairing, which is what it embedded images with."""
        if from_version < 2:
            widgets = state.get("widget_data")
            if isinstance(widgets, dict):
                widgets.setdefault("image_model_id", IMAGE_AUTO)
                widgets.setdefault("image_precision", "auto")
        return state

    def _extra_rows(self, form: QFormLayout) -> None:
        self._image = QComboBox()
        self._image.setEditable(True)
        self._image.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._image.addItems([IMAGE_AUTO, *VISION_MODELS])
        self._image.setToolTip(
            "The model that embeds images for image-content search. An image "
            "model's vectors only compare against its text partner, so only "
            "the partner is offered. auto: okf's pairing (none for jina "
            "small, whose images are searched by caption). The graph keeps "
            "the image model of the first image it embeds.")
        form.addRow("Images:", self._image)
        self._widget_core.register_widget(
            "image_model_id", self._image, role=PortRole.INPUT,
            datatype="string", default=IMAGE_AUTO, add_to_layout=False,
        )
        self.add_input("image_model_id", datatype="string")

        self._image_precision = QComboBox()
        self._image_precision.addItems(list(IMAGE_PRECISIONS))
        self._image_precision.setToolTip(
            "auto: FP32 on the CPU, FP16 on a GPU. No int8 image weights "
            "ship; fp16 on the CPU fails.")
        form.addRow("Image prec.:", self._image_precision)
        self._widget_core.register_widget(
            "image_precision", self._image_precision, role=PortRole.INPUT,
            datatype="string", default="auto", add_to_layout=False,
        )
        self.add_input("image_precision", datatype="string")

    def _sync_choices(self, model_id: str) -> None:
        """Grey out the precisions and image models *model_id* has not got;
        a choice that went grey falls back to auto."""
        shipped = precisions_for(model_id.strip())
        partner = vision_partner(model_id.strip())
        for combo, allowed in ((self._precision, shipped),
                               (self._image, (IMAGE_AUTO, partner))):
            items = combo.model()
            for row in range(combo.count()):
                item = items.item(row) if hasattr(items, "item") else None
                if item is not None:
                    item.setEnabled(combo.itemText(row) in allowed)
            if combo.currentText() not in allowed and (
                    combo is self._precision or combo.currentText() in VISION_MODELS):
                combo.setCurrentText("auto")

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        described = onnx_description(inputs, self._ROLE, self._MODELS,
                                     self._PRECISIONS)
        described.update(image_description(inputs, described["model_id"]))
        self._sync_status = (
            f"{described['model_id']} on {described['device']} "
            f"({described['precision']}). {image_text(described)} Loaded by "
            "okf serve, in its own process; the first use downloads it if it "
            "is not cached.")
        return {self._OUTPUT: described}


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
