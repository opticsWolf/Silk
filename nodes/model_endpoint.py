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

Three things this node does that a URL field alone would not:

* **It asks before it answers.** Connecting probes ``/models``, so a
  wrong URL or an unset key is a status line here rather than a failed
  agent run three nodes downstream.
* **It names the credential, never holds it** (D22). The field takes the
  name of an environment variable or of an entry in
  ``~/.weave/silk/secrets.json``. A saved graph stays shareable.
* **It says when the context window is unknown.** Compaction needs a
  denominator (D25); a remote endpoint rarely advertises one, so the
  field is explicit and its absence is stated rather than guessed.
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QLabel, QLineEdit, QSpinBox,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import MODEL_HANDLE_TYPE  # noqa: F401
from ..functions.model_endpoint import PROVIDERS, connect, provider_for

log = get_logger("SilkModelEndpointNode")


@register_node
class SilkModelEndpointNode(ActiveNode):
    """One remote OpenAI-compatible model, as a model handle."""
    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "AI"
    node_subclass: ClassVar[str] = "Loaders"
    node_name: ClassVar[Optional[str]] = "Model Endpoint"
    node_description: ClassVar[Optional[str]] = (
        "Uses a remote OpenAI-compatible model (OpenRouter, LM Studio, "
        "vLLM, a LiteLLM proxy) wherever the GGUF Loader would go."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "llm", "openai", "openrouter", "lmstudio", "litellm",
        "vllm", "ollama", "remote", "inference",
    ]
    node_icon: ClassVar[Optional[str]] = "node"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1
    node_version = 2     # bump on any state-shape change (G20)

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

        self._provider = QComboBox()
        for preset in PROVIDERS:
            self._provider.addItem(preset.label, userData=preset.key)
        self._provider.setToolTip(
            "A preset fills the URL and the usual credential name. Both "
            "stay editable — the useful endpoint is often your own proxy."
        )
        form.addRow("Provider:", self._provider)
        self._widget_core.register_widget(
            "provider", self._provider, role=PortRole.INPUT,
            datatype="string", default=PROVIDERS[0].key, add_to_layout=False,
        )
        self.add_input("provider", datatype="string")

        self._base_url = QLineEdit("")
        self._base_url.setPlaceholderText("http://localhost:1234/v1")
        form.addRow("Base URL:", self._base_url)
        self._widget_core.register_widget(
            "base_url", self._base_url, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(500),
            add_to_layout=False,
        )
        self.add_input("base_url", datatype="string")

        self._model = QLineEdit("")
        self._model.setPlaceholderText(
            "model name — blank asks the endpoint what it serves")
        form.addRow("Model:", self._model)
        self._widget_core.register_widget(
            "model", self._model, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(500),
            add_to_layout=False,
        )
        self.add_input("model", datatype="string")

        self._credential = QLineEdit("")
        self._credential.setPlaceholderText(
            "OPENROUTER_API_KEY (a name, never a value)")
        self._credential.setToolTip(
            "The NAME of an environment variable, or of an entry in "
            "~/.weave/silk/secrets.json. The value is read at connect "
            "time and never saved in the graph (D22)."
        )
        form.addRow("Credential:", self._credential)
        self._widget_core.register_widget(
            "credential", self._credential, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(500),
            add_to_layout=False,
        )
        self.add_input("credential", datatype="string")

        self._context = QSpinBox()
        self._context.setRange(0, 4_000_000)
        self._context.setSingleStep(1024)
        self._context.setSpecialValueText("unknown")
        self._context.setToolTip(
            "The model's context window. 0 means unknown — compaction "
            "then has no denominator and stays off, which is safer than "
            "a guess that summarises too early or overflows."
        )
        form.addRow("Context length:", self._context)
        self._widget_core.register_widget(
            "context_length", self._context, role=PortRole.INPUT,
            datatype="int", default=0, add_to_layout=False,
        )
        self.add_input("context_length", datatype="int")

        self._native_tools = QCheckBox("Native tool calling")
        self._native_tools.setToolTip(
            "Send tool schemas in the request (the OpenAI `tools` field) "
            "instead of asking for them in a text fence. Off by default: "
            "a server that does not support them refuses the request, and "
            "a failed run is worse than a slightly clumsier protocol that "
            "works. Most hosted gateways support it; small local models "
            "often do not."
        )
        form.addRow("", self._native_tools)
        self._widget_core.register_widget(
            "supports_tools", self._native_tools, role=PortRole.INPUT,
            datatype="bool", default=False, add_to_layout=False,
        )
        self.add_input("supports_tools", datatype="bool")

        self._label_status = QLabel("Not connected.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self._provider.currentIndexChanged.connect(self._on_provider_changed)

        #: The last handle handed downstream, kept so an unrelated graph
        #: edit does not re-probe a working endpoint.
        self._handle: Optional[Dict[str, Any]] = None
        self._last_spec: tuple = ()
        self._sync_status: str = "Not connected."

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── Main thread ───────────────────────────────────────────────────

    def _on_provider_changed(self, _index: int) -> None:
        """Fill the URL and credential name from the preset.

        Only into *empty* fields: a preset must never overwrite something
        that was typed, because the common edit is "this preset, but my
        own host" and losing that on a stray click is infuriating.
        """
        preset = provider_for(str(self._provider.currentData() or ""))
        if preset.base_url and not self._base_url.text().strip():
            self._base_url.setText(preset.base_url)
        if preset.credential and not self._credential.text().strip():
            self._credential.setText(preset.credential)
        if preset.note:
            # Through WidgetCore, not the label: a node's panel mirror is
            # a different widget object, and a direct setText leaves it
            # showing the previous provider's note (WV401).
            self._sync_status = preset.note
            self._widget_core.push_display("status", preset.note)

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._widget_core.push_display("status", self._sync_status)

    # ── Worker thread ─────────────────────────────────────────────────

    @staticmethod
    def spec_from(inputs: Dict[str, Any]) -> tuple:
        """What this endpoint is, as the tuple a re-probe compares on."""
        return (
            str(inputs.get("provider") or "custom").strip(),
            str(inputs.get("base_url") or "").strip(),
            str(inputs.get("model") or "").strip(),
            str(inputs.get("credential") or "").strip(),
            int(inputs.get("context_length") or 0),
            bool(inputs.get("supports_tools")),
        )

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        spec = self.spec_from(inputs)

        # A graph re-evaluates on every unrelated edit. Re-probing a
        # working endpoint on each of those would add a request per
        # keystroke elsewhere in the canvas — and, on a metered provider,
        # a reason to distrust the node.
        if self._handle is not None and spec == self._last_spec:
            return {"model_obj": self._handle}

        provider, base_url, model, credential, context_length, native = spec
        if not base_url:
            self._handle = None
            self._last_spec = spec
            self._sync_status = provider_for(provider).note or (
                "No URL yet. Pick a provider, or type an endpoint's base URL."
            )
            return {"model_obj": None}

        handle, status, _models = connect(
            base_url, model, credential=credential,
            context_length=context_length, provider=provider,
            supports_tools=native,
        )
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

    def cleanup(self) -> None:
        # Nothing to close: the client is an HTTP proxy object with no
        # session and no subprocess. Dropped so a removed node does not
        # keep a resolved key alive in its header dict.
        self._handle = None
        super().cleanup()
