# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Silk Worker Node — a named worker an Orchestrator can delegate to.

**This node does not run anything.** It describes an agent; a Silk
Orchestrator runs it. Its ``workers`` output must reach an Orchestrator's
``workers`` input, or nothing happens at all -- there is no other consumer
of this port type.

That is the whole difference from the **Silk Agent** node, which runs where
it sits and has the ``run``/``done`` pins and a ``response`` output to prove
it. A Worker has none of those, and has two things an Agent has no use for:
a ``name``, because an orchestrator's model addresses it by name rather than
by wire, and a ``description``, advertised through ``list_workers`` so the
model can pick it. Write that as the model should read it ("researches the
web", "writes and edits prose", …).

It cannot be a node that runs, either: ``delegate`` spawns workers inside a
single tool call, on a worker thread, several at once -- so a worker has to
be *data the orchestrator carries*, not a box the engine schedules.

Bundles a model + (optional) toolset + (optional) role into a single
:class:`~..functions.subagent.WorkerSpec` and appends it to a **chain** on
the ``workers`` port (the accumulate-down-the-chain pattern the Toolchain
nodes use). Wire several Worker nodes in series, then feed the final list
into the Orchestrator.

Renamed from "Silk Agent Spec" (ports ``agents_in``/``agents``, type
``silk_agents``), which read like configuration for an Agent node -- the
one thing it is not.
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
    MODEL_HANDLE_TYPE,
    SILK_WORKERS_TYPE,
    SILK_ROLE_TYPE,
    SILK_TOOLSET_TYPE,
)
from ..functions.role import DEFAULT_ROLE
from ..functions.subagent import WorkerSpec
from ..functions.usage_limits import describe_budget, parse_budget

log = get_logger("SilkWorker")


@register_node
class SilkWorkerNode(ActiveNode):
    """Names a model+toolset+role bundle an Orchestrator can delegate to."""
    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk Worker"
    node_description: ClassVar[Optional[str]] = (
        "Describes a named worker (model + toolset + role) for a Silk "
        "Orchestrator to delegate to. Does not run on its own: chain these "
        "and wire the last one's 'workers' output into an Orchestrator."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "orchestration", "worker", "llm",
    ]
    node_icon: ClassVar[Optional[str]] = "robot"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1   # owns a hand-written state dict
    node_version = 2     # renamed from SilkAgentSpecNode; ports renamed

    def __init__(self, title: str = "Silk Worker", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        self.add_input("model_obj", datatype="model_handle")
        self.add_input("toolset", datatype="silk_toolset")
        self.add_input("role", datatype="silk_role")
        # Speciality text is widget-backed but also wireable (BIDIRECTIONAL).
        self.add_input("description", datatype="string")
        # This worker's own caps, wireable like the speciality text.
        self.add_input("budget", datatype="string")
        # Chain input: the workers accumulated so far (optional first link).
        self.add_input("workers_in", datatype="silk_workers")
        self.add_output("workers", datatype="silk_workers")

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

        # This worker's own share. With an orchestrator budget it
        # nests inside it, so a greedy worker exhausts its share and
        # the rest of the fan-out keeps theirs (D26, T3).
        self._edit_budget = QLineEdit()
        self._edit_budget.setPlaceholderText(
            "empty = only the orchestrator's cap, e.g. requests=5"
        )
        self._edit_budget.setToolTip(
            "This worker's own caps, comma separated: requests, "
            "tool_calls, output_tokens, input_tokens. They narrow the "
            "orchestrator's shared budget; they can never widen it."
        )
        form.addRow("Budget:", self._edit_budget)
        self._widget_core.register_widget(
            "budget", self._edit_budget, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(300),
            add_to_layout=False,
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
        chain = list(inputs.get("workers_in") or [])

        model_handle = inputs.get("model_obj")
        if not isinstance(model_handle, dict) or not (
            model_handle.get("backend")
            and ("model" in model_handle or "pool" in model_handle)
        ):
            self._sync_status = (
                "No valid model connected — this worker is not added."
            )
            return {"workers": chain}

        try:
            budget = parse_budget(inputs.get("budget"))
        except ValueError as exc:
            # Same answer as an invalid model: the worker is not added. A
            # worker registered without the caps someone typed would be
            # delegated to and run uncapped (D26).
            self._sync_status = f"Budget not readable ({exc}) - worker not added."
            return {"workers": chain}

        name = str(inputs.get("worker_name") or "").strip()
        role = inputs.get("role") or DEFAULT_ROLE
        spec = WorkerSpec(
            model_handle=model_handle,
            toolset=inputs.get("toolset"),
            role=role,
            name=name or f"worker{len(chain) + 1}",
            description=str(inputs.get("description") or "").strip(),
            usage_limits=budget,
        )
        self._sync_status = (
            f"Worker '{spec.name}' ready "
            f"({'with toolset' if spec.toolset is not None else 'chat-only'}, "
            f"role '{getattr(role, 'id', '?')}', {describe_budget(budget)}). "
            f"{len(chain) + 1} in chain."
        )
        return {"workers": chain + [spec]}

    # ── State ─────────────────────────────────────────────────────────

    def restore_state(self, state: Dict[str, Any]) -> None:
        with self._widget_core.suppress_signals():
            super().restore_state(state)

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        if hasattr(self, "_sync_status"):
            self._widget_core.push_display("status", self._sync_status)
