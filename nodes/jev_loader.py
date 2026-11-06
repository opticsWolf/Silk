# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Jev Loader -- a hosted decision model as a graph resource (D-S8).

Jev speaks the same ``/v1/systemone`` wire shape as a local decision
model (see ``functions/systemone.py``), so it emits the same classifier
handle a Classifier Loader emits. The provider -- URL and credential
*name* -- arrives on the ``endpoint`` input from an Endpoint node; this
node only names which hosted decision model to use. The key itself never
enters the graph (D22): it travels the wire by name and resolves at
answer time.

Nothing is probed here: a smoke question would cost money. A missing key
or an unreachable endpoint fails at ask time, is recorded in the
verdict's ``tried`` list, and defers to the human -- never to an error
that takes the graph down.
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QFormLayout, QLabel, QLineEdit

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import CLASSIFIER_MODEL_TYPE  # noqa: F401
from .silk_ports import ENDPOINT_HANDLE_TYPE  # noqa: F401
from ..functions.credentials import resolve_credential
from ..functions.model_endpoint import split_endpoint
from ..functions.systemone import hosted_decision_handle

log = get_logger("SilkJevLoader")

DEFAULT_JEV_MODEL = "jev/jev-1.13.0"


@register_node
class JevLoaderNode(ActiveNode):
    """A hosted /v1/systemone endpoint (Jev) as a classifier model."""

    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    node_name: ClassVar[Optional[str]] = "Jev Loader"
    node_description: ClassVar[Optional[str]] = (
        "A hosted decision model (Jev) on a wired Endpoint: same typed "
        "questions, same answers as a local Classifier Loader; metered, "
        "with egress."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "jev", "decision", "systemone", "triage", "hosted",
    ]
    node_icon: ClassVar[Optional[str]] = "gavel"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 2
    node_version = 1

    @classmethod
    def migrate_state(cls, state: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """api 1 → 2: the provider moved to the Endpoint node.

        The ``url`` and ``key_name`` widgets are gone: the URL and the
        credential name now arrive on the ``endpoint`` input. A custom
        credential name must be re-entered on the Endpoint node (or it
        resolves from the environment as before); the model name is
        untouched.
        """
        if from_version < 2:
            values = state.get("widget_data")
            if isinstance(values, dict):
                values.pop("url", None)
                values.pop("key_name", None)
        return state

    def __init__(self, title: str = "Jev Loader", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        self.add_input("endpoint", datatype="endpoint")
        self.add_output("model_obj", datatype="classifier_model")

        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self._model = QLineEdit(DEFAULT_JEV_MODEL)
        self._model.setToolTip(
            "A pinned version: bands are calibrated per model id. "
            "'latest' is refused -- pin it (e.g. jev/jev-1.13.0).")
        form.addRow("Model:", self._model)
        self._widget_core.register_widget(
            "model", self._model, role=PortRole.INPUT,
            datatype="string", default=DEFAULT_JEV_MODEL,
            policy=debounced(400), add_to_layout=False,
        )
        self.add_input("model", datatype="string")

        self._label_status = QLabel("Not configured.")
        self._label_status.setEnabled(False)
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self._handle: Optional[Dict[str, Any]] = None
        self._last_spec: tuple = ()
        self._sync_status: str = "Not configured."

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── Main thread ───────────────────────────────────────────────────

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._widget_core.push_display("status", self._sync_status)

    # ── Worker thread ─────────────────────────────────────────────────

    @staticmethod
    def spec_from(inputs: Dict[str, Any]) -> tuple:
        """What this handle is, as the tuple a rebuild compares on."""
        endpoint = split_endpoint(inputs.get("endpoint")) or {}
        return (
            endpoint.get("provider", "custom"),
            endpoint.get("base_url", ""),
            endpoint.get("credential", ""),
            str(inputs.get("model") or DEFAULT_JEV_MODEL).strip(),
        )

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        provider, url, key_name, model = self.spec_from(inputs)
        spec = (provider, url, key_name, model)
        if self._handle is not None and spec == self._last_spec:
            return {"model_obj": self._handle}
        if not url:
            self._handle, self._last_spec = None, spec
            self._sync_status = ("Wire an Endpoint. The provider, URL and "
                                 "credential arrive on the endpoint input; "
                                 "this node only names the hosted model.")
            return {"model_obj": None}
        if not model:
            self._handle, self._last_spec = None, spec
            self._sync_status = "No model set."
            return {"model_obj": None}
        try:
            handle: Optional[Dict[str, Any]] = hosted_decision_handle(
                model, url, key_name)
        except ValueError as exc:
            log.warning(f"Jev loader misconfigured: {exc}")
            self._handle, self._last_spec = None, spec
            self._sync_status = str(exc)
            return {"model_obj": None}
        self._handle, self._last_spec = handle, spec
        if key_name and resolve_credential(key_name):
            self._sync_status = (
                f"Ready: {model} @ {url}  ·  key '{key_name}' ✓  ·  "
                f"metered, egress gated by the classifier")
        elif key_name:
            self._sync_status = (
                f"Ready: {model} @ {url}  ·  key '{key_name}' missing "
                f"(env or ~/.weave/silk/secrets.json) -- calls will defer "
                f"to the human")
        else:
            self._sync_status = (
                f"Ready: {model} @ {url}  ·  no key named -- calls will "
                f"defer to the human")
        log.info(f"Jev loader ready: {model} @ {url}")
        return {"model_obj": handle}

    def cleanup(self) -> None:
        self._handle = None
        super().cleanup()
