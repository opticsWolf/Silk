# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Silk Model Endpoint -- a remote model where the loader would be (D45).

Wires anywhere the GGUF Loader wires: it emits the same `model_handle`,
so Agent, Agent Spec and the ToolBox's embedding input take it without
knowing the difference. What changes is where the model runs -- someone
else's process, possibly someone else's machine, possibly for money.

Four things this node does that a URL field alone would not:

* **It asks before it answers.** Connecting probes ``/models``, so a
  wrong URL or an unset key is a status line here rather than a failed
  agent run three nodes downstream.
* **It names the credential, never holds it** (D22). The field takes the
  name of an environment variable or of an entry in
  ``~/.weave/silk/secrets.json``. A saved graph stays shareable.
* **It remembers per provider.** The base URL, the credential name, the
  last-picked model and held-back context lengths are kept in
  ``~/.weave/silk/model_endpoints/<provider>.json`` -- switching from
  OpenRouter to a local server and back restores everything, and a
  manual max for a model the gateway does not describe survives.
* **It says when the context window is unknown.** Compaction needs a
  denominator (D25); a remote endpoint rarely advertises one, so the max
  is explicit, auto-detected when the listing says, and its absence is
  stated rather than guessed.
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QFormLayout, QLabel

from weave.widgetcore import WidgetCore, PortRole
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .model_picker import ModelPickerMixin, ProviderConfigMixin
from .silk_ports import MODEL_HANDLE_TYPE  # noqa: F401
from ..functions.endpoint_settings import EndpointSettings
from ..functions.model_endpoint import connect, provider_for
from ..functions.port_text import text_input

log = get_logger("SilkModelEndpointNode")


@register_node
class SilkModelEndpointNode(ProviderConfigMixin, ModelPickerMixin, ActiveNode):
    """One remote OpenAI-compatible model, as a model handle."""
    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    node_name: ClassVar[Optional[str]] = "Model Endpoint"
    node_description: ClassVar[Optional[str]] = (
        "Uses a remote OpenAI-compatible model (OpenRouter, LM Studio, "
        "vLLM, a LiteLLM proxy) wherever the GGUF Loader would go. "
        "Remembers the URL, key name, model and context per provider."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "llm", "openai", "openrouter", "lmstudio", "litellm",
        "vllm", "ollama", "remote", "inference",
    ]
    node_icon: ClassVar[Optional[str]] = "plug"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 2
    node_version = 2     # bump on any state-shape change (G20)

    @classmethod
    def migrate_state(cls, state: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """api 1 → 2: the model line-edit became a dropdown, and the
        context spinbox became a max spinner plus a reduce slider.

        Port names are unchanged, so saved values land on the new
        widgets: the model text is kept (the combo is editable and falls
        back to a typed name), the saved context length becomes the
        slider's value, and the spinner starts at auto-detect. Nothing
        is remapped -- except the native-tools box, which became the
        whole answer.

        Before, an unticked box deferred to the gateway's listing, so a
        saved ``supports_tools: false`` meant "ask the endpoint", not
        "fences". It is ticked here for exactly the models the
        remembered listing says call tools, so an old graph keeps the
        transport it ran on (the OpenCode nodes migrated the same way).
        A model the listing knows nothing about stays as saved: there is
        no answer left to restore.
        """
        values = state.get("widget_data")
        if isinstance(values, dict) and not values.get("supports_tools"):
            provider = str(values.get("provider") or "custom").strip() or "custom"
            model = str(values.get("model") or "").strip()
            if EndpointSettings(provider).tool_call(model) is True:
                values["supports_tools"] = True
        return state

    def __init__(self, title: str = "Model Endpoint", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        self.add_output("model_obj", datatype="model_handle")

        # ── Layout & WidgetCore ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self.init_provider_config(form, self._widget_core)

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

        self._provider.currentIndexChanged.connect(self._on_provider_changed)
        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

        self._on_provider_changed(0)

    # ── the picker asks these ────────────────────────────────────────

    def _picker_provider_key(self) -> str:
        return str(self._provider.currentData() or "custom")

    def _picker_source(self) -> tuple:
        return (self._base_url.text().strip(),
                self._credential.text().strip())

    # ── Main thread ───────────────────────────────────────────────────

    def _on_provider_changed(self, _index: int) -> None:
        """The mixin fills the rows; the listing follows the provider."""
        key = self._provider_changed_fill()
        note = provider_for(key).note
        if note:
            # Through WidgetCore, not the label: a node's panel mirror is
            # a different widget object (WV401).
            self._sync_status = note
            self._widget_core.push_display("status", note)
        self.refresh_picker()

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        # A graph restore, or a wire, writes the widgets with signals
        # blocked; the slider still has to fit the model it now names.
        self.sync_picker()
        self._widget_core.push_display("status", self._sync_status)

    def cleanup(self) -> None:
        self.cleanup_picker()
        # Nothing to close: the client is an HTTP proxy object with no
        # session and no subprocess. Dropped so a removed node does not
        # keep a resolved key alive in its header dict.
        self._handle = None
        super().cleanup()

    # ── Worker thread ─────────────────────────────────────────────────

    @staticmethod
    def spec_from(inputs: Dict[str, Any]) -> tuple:
        """What this endpoint is, as the tuple a re-probe compares on."""
        model, maximum, length, native = \
            ModelPickerMixin.picker_spec_from(inputs)
        return (
            text_input(inputs, "provider", "custom"),
            text_input(inputs, "base_url"),
            model,
            maximum,
            length,
            text_input(inputs, "credential"),
            native,
        )

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            spec = self.spec_from(inputs)
        except ValueError as exc:
            # Reported like an unreachable endpoint: never raised (below).
            self._handle = None
            self._last_spec = ()
            self._sync_status = f"Invalid input: {exc}."
            return {"model_obj": None}
        provider, base_url, model, maximum, length, credential, native = spec

        # A graph re-evaluates on every unrelated edit. Re-probing a
        # working endpoint on each of those would add a request per
        # keystroke elsewhere in the canvas — and, on a metered provider,
        # a reason to distrust the node.
        if self._handle is not None and spec == self._last_spec:
            return {"model_obj": self._handle}

        if not base_url:
            self._handle = None
            self._last_spec = spec
            self._sync_status = provider_for(provider).note or (
                "No URL yet. Pick a provider, or type an endpoint's base URL."
            )
            return {"model_obj": None}

        handle, status, models = connect(
            base_url, model, credential=credential,
            context_length=self.clamp_length(model, length, maximum),
            provider=provider,
            # The box is the answer, listing included: picking a model
            # ticks it the way the listing says, and an untick means
            # fences. Nothing here second-guesses it behind the person's
            # back (the OpenCode nodes work the same way).
            supports_tools=native,
        )
        if handle is not None and models and model and model not in models:
            status += "  ·  not in the listing (typed by hand)"
        self._handle = handle
        self._last_spec = spec
        self._sync_status = status
        if handle is None:
            # Reported, never raised: one unreachable endpoint must not
            # take the graph down, and the agent downstream already says
            # "no valid model connected" for the empty wire.
            log.warning(f"Model endpoint not connected — {status}")
        else:
            log.info(f"Model endpoint ready — {status}")
        return {"model_obj": handle}
