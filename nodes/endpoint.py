# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The provider/model split: an Endpoint node and a Model node.

The Model Endpoint is one node that owns everything: provider, URL,
credential, model, context. When several models share one provider --
an agent and its judge on the same gateway, say -- that means the
provider rows are configured twice and can disagree. These two nodes
split the choice from the settings:

* **Endpoint** owns the provider rows (preset, base URL, credential
  name, per-provider memory) and forwards them as an ``endpoint`` dict.
* **Model** takes that dict through its ``endpoint`` input and owns the
  model dropdown, the max spinner, the reduce slider and the tools box,
  then connects -- the same picker the Model Endpoint uses, so one
  provider switch re-lists every Model node wired to it.

Nothing is probed until the Model node connects: the Endpoint node is
a configuration, not a connection. A missing URL is an empty output,
not an error -- the Model node downstream already says "wire an
Endpoint" for the empty wire.
"""
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QFormLayout, QLabel

from weave.widgetcore import WidgetCore, PortRole
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .model_picker import ModelPickerMixin, ProviderConfigMixin
from .silk_ports import ENDPOINT_HANDLE_TYPE, MODEL_HANDLE_TYPE  # noqa: F401
from ..functions.credentials import resolve_credential
from ..functions.model_endpoint import (
    connect, endpoint_dict, provider_for, split_endpoint,
)

log = get_logger("SilkEndpointNodes")


@register_node
class SilkEndpointNode(ProviderConfigMixin, ActiveNode):
    """One provider's connection info, as an endpoint dict."""

    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    node_name: ClassVar[Optional[str]] = "Endpoint"
    node_description: ClassVar[Optional[str]] = (
        "A provider (preset, base URL, credential name) forwarded to "
        "Model nodes. One Endpoint feeds many Models; switching the "
        "provider here re-points them all."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "endpoint", "provider", "openai", "openrouter", "lmstudio",
        "litellm", "vllm", "ollama", "remote",
    ]
    node_icon: ClassVar[Optional[str]] = "server"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1
    node_version = 1

    def __init__(self, title: str = "Endpoint", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        self.add_output("endpoint", datatype="endpoint")

        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self.init_provider_config(form, self._widget_core)

        self._label_status = QLabel("Not configured.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self._provider.currentIndexChanged.connect(self._on_provider_changed)

        self._handle: Optional[Dict[str, str]] = None
        self._last_spec: tuple = ()
        self._sync_status: str = "Not configured."

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

        self._on_provider_changed(0)

    # ── Main thread ───────────────────────────────────────────────────

    def _on_provider_changed(self, _index: int) -> None:
        """The mixin fills the rows; the status follows on compute."""
        key = self._provider_changed_fill()
        note = provider_for(key).note
        if note:
            self._sync_status = note
            self._widget_core.push_display("status", note)

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._widget_core.push_display("status", self._sync_status)

    # ── Worker thread ─────────────────────────────────────────────────

    @staticmethod
    def spec_from(inputs: Dict[str, Any]) -> tuple:
        """What this endpoint is, as the tuple a rebuild compares on."""
        return ProviderConfigMixin.provider_spec_from(inputs)

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        provider, base_url, credential = self.spec_from(inputs)
        spec = (provider, base_url, credential)
        if self._handle is not None and spec == self._last_spec:
            return {"endpoint": self._handle}
        if not base_url:
            self._handle = None
            self._last_spec = spec
            self._sync_status = provider_for(provider).note or (
                "No URL yet. Pick a provider, or type an endpoint's base URL."
            )
            return {"endpoint": None}
        self._handle = endpoint_dict(provider, base_url, credential)
        self._last_spec = spec
        if credential and resolve_credential(credential):
            key_note = f"key '{credential}' ✓"
        elif credential:
            key_note = (f"key '{credential}' missing "
                        f"(env or ~/.weave/silk/secrets.json)")
        else:
            key_note = "no key (a local server needs none)"
        self._sync_status = (
            f"{provider_for(provider).label} → {base_url}  ·  {key_note}")
        log.info(f"Endpoint ready — {self._sync_status}")
        return {"endpoint": self._handle}

    def cleanup(self) -> None:
        self._handle = None
        super().cleanup()


@register_node
class SilkModelNode(ModelPickerMixin, ActiveNode):
    """One remote model on a wired Endpoint, as a model handle."""

    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    node_name: ClassVar[Optional[str]] = "Model"
    node_description: ClassVar[Optional[str]] = (
        "A model on an Endpoint node's provider: dropdown, max spinner "
        "and reduce slider, connected through the wired endpoint. The "
        "pure model half of the Endpoint/Model split."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "llm", "model", "openai", "remote", "inference",
    ]
    node_icon: ClassVar[Optional[str]] = "cloud"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1
    node_version = 1

    def __init__(self, title: str = "Model", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        self.add_input("endpoint", datatype="endpoint")
        self.add_output("model_obj", datatype="model_handle")

        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        #: The wired provider, as the last compute saw it. The picker
        #: reads the provider key and source from here on the GUI thread,
        #: so it is set before the picker builds (it lists on init).
        self._wire: tuple = ("custom", "", "")

        self.init_picker(form, self._widget_core)

        self._label_status = QLabel("Not connected.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        #: The last handle handed downstream, kept so an unrelated graph
        #: edit does not re-probe a working endpoint.
        self._handle: Optional[Dict[str, Any]] = None
        self._last_spec: tuple = ()
        self._sync_status: str = "Not connected."

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── the picker asks these ────────────────────────────────────────

    def _picker_provider_key(self) -> str:
        return self._wire[0]

    def _picker_source(self) -> tuple:
        return (self._wire[1], self._wire[2])

    # ── Main thread ───────────────────────────────────────────────────

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        # The wire may have moved since the picker last looked: a new
        # provider key means a different settings file, a refill and a
        # re-list. Otherwise the slider just fits the model it names.
        if self._wire[0] != self._picker.get("key"):
            self.refresh_picker()
        else:
            self.sync_picker()
        self._widget_core.push_display("status", self._sync_status)

    def cleanup(self) -> None:
        self.cleanup_picker()
        self._handle = None
        super().cleanup()

    # ── Worker thread ─────────────────────────────────────────────────

    @staticmethod
    def spec_from(inputs: Dict[str, Any]) -> tuple:
        """What this model is, as the tuple a re-probe compares on."""
        endpoint = split_endpoint(inputs.get("endpoint")) or {}
        model, maximum, length, native = \
            ModelPickerMixin.picker_spec_from(inputs)
        return (
            endpoint.get("provider", "custom"),
            endpoint.get("base_url", ""),
            endpoint.get("credential", ""),
            model, maximum, length, native,
        )

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        spec = self.spec_from(inputs)
        provider, base_url, credential, model, maximum, length, native = spec
        # Seen by the GUI thread after this evaluation: the picker's
        # settings file and listing source.
        self._wire = (provider, base_url, credential)
        if self._handle is not None and spec == self._last_spec:
            return {"model_obj": self._handle}
        if not base_url:
            self._handle = None
            self._last_spec = spec
            self._sync_status = ("Wire an Endpoint. The provider, URL and "
                                 "credential arrive on the endpoint input; "
                                 "this node only picks the model.")
            return {"model_obj": None}

        handle, status, models = connect(
            base_url, model, credential=credential,
            context_length=self.clamp_length(model, length, maximum),
            provider=provider,
            # The box is the answer, listing included: picking a model
            # ticks it the way the listing says, and an untick means
            # fences (the OpenCode nodes work the same way).
            supports_tools=native,
        )
        if handle is not None and models and model and model not in models:
            status += "  ·  not in the listing (typed by hand)"
        self._handle = handle
        self._last_spec = spec
        self._sync_status = status
        if handle is None:
            log.warning(f"Model not connected — {status}")
        else:
            log.info(f"Model ready — {status}")
        return {"model_obj": handle}
