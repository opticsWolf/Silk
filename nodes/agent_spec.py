# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Silk Agent Spec Node — a named worker bundle for the Orchestrator.

Bundles a model + (optional) toolset + (optional) role into a single
:class:`~..functions.subagent.AgentSpec` and appends it to a **chain** of agent
specs on the ``agents`` port (the same accumulate-down-the-chain pattern the
Toolchain nodes use). Wire several Agent Spec nodes in series, then feed the
final ``agents`` list into a Silk Orchestrator's ``workers`` input: each becomes
a specialist the orchestrator can ``delegate`` to by name.

The ``name`` is how the orchestrator (and its model) addresses the worker;
``description`` is the speciality advertised via ``list_workers``, so write it as
the model should read it ("researches the web", "writes and edits prose", …).
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QFormLayout, QLabel, QLineEdit

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from weave.widgets.markdown_widget import MarkdownWidget

from .silk_ports import (  # noqa: F401
    GGUF_MODEL_TYPE,
    SILK_AGENTS_TYPE,
    SILK_ROLE_TYPE,
    SILK_TOOLSET_TYPE,
)
from ..functions.role import DEFAULT_ROLE
from ..functions.subagent import AgentSpec

log = get_logger("SilkAgentSpec")


@register_node
class SilkAgentSpecNode(ActiveNode):
    """Names a model+toolset+role bundle as a delegatable worker agent."""

    node_class: ClassVar[str] = "AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk Agent Spec"
    node_description: ClassVar[Optional[str]] = (
        "A named worker bundle (model + toolset + role) for the Orchestrator; "
        "chainable into a workers list."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "orchestration", "worker", "llm",
    ]
    node_icon: ClassVar[Optional[str]] = "robot"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1   # owns a hand-written state dict

    def __init__(self, title: str = "Silk Agent Spec", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        self.add_input("model_obj", datatype="gguf_model")
        self.add_input("toolset", datatype="silk_toolset")
        self.add_input("role", datatype="silk_role")
        # Speciality text is widget-backed but also wireable (BIDIRECTIONAL).
        self.add_input("description", datatype="string")
        # Chain input: the workers accumulated so far (optional first link).
        self.add_input("agents_in", datatype="silk_agents")
        self.add_output("agents", datatype="silk_agents")

        # ── Layout & WidgetCore ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self._edit_name = QLineEdit()
        self._edit_name.setPlaceholderText("e.g. researcher")
        form.addRow("Worker Name:", self._edit_name)
        self._widget_core.register_widget(
            "worker_name", self._edit_name, role=PortRole.INTERNAL,
            datatype="string", default="", policy=debounced(300),
            add_to_layout=False,
        )

        self._edit_desc = MarkdownWidget(mode="editor")
        self._edit_desc._text_edit.setPlaceholderText(
            "Speciality advertised to the orchestrator (what this worker is for)…"
        )
        self._edit_desc._text_edit.setMaximumHeight(70)
        form.addRow("Speciality:", self._edit_desc)
        self._widget_core.register_widget(
            "description", self._edit_desc, role=PortRole.BIDIRECTIONAL,
            datatype="string", default="", add_to_layout=False,
        )

        self._label_status = QLabel("No model connected.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        # ── Mount ──
        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── Worker thread ─────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        chain = list(inputs.get("agents_in") or [])

        model_handle = inputs.get("model_obj")
        valid_model = (
            isinstance(model_handle, dict)
            and model_handle.get("backend") == "gguf"
            and ("model" in model_handle or "pool" in model_handle)
        )
        if not valid_model:
            self._sync_status = (
                "No valid GGUF model connected — this worker is not added."
            )
            return {"agents": chain}

        name = str(inputs.get("worker_name") or "").strip()
        role = inputs.get("role") or DEFAULT_ROLE
        spec = AgentSpec(
            model_handle=model_handle,
            toolset=inputs.get("toolset"),
            role=role,
            name=name or f"worker{len(chain) + 1}",
            description=str(inputs.get("description") or "").strip(),
        )
        self._sync_status = (
            f"Worker '{spec.name}' ready "
            f"({'with toolset' if spec.toolset is not None else 'chat-only'}, "
            f"role '{getattr(role, 'id', '?')}'). {len(chain) + 1} in chain."
        )
        return {"agents": chain + [spec]}

    # ── State ─────────────────────────────────────────────────────────

    def restore_state(self, state: Dict[str, Any]) -> None:
        with self._widget_core.suppress_signals():
            super().restore_state(state)

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        if hasattr(self, "_sync_status"):
            self._widget_core.push_display("status", self._sync_status)
