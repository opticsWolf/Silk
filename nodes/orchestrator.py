# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Silk Orchestrator Node — a Silk Agent that delegates to worker agents.

An orchestrator *is* a :class:`SilkAgentNode`: same model + toolset + role + Exec
chaining. The only addition is a ``workers`` input (a chain of Agent Spec nodes)
whose specs are mounted onto the agent's toolset as ``delegate`` /
``delegate_parallel`` / ``list_workers`` tools right before the run. From then on
the ordinary agent loop drives everything — the model plans, and calls
``delegate`` when a sub-task belongs to a specialist.

Because the delegation tools are attached to the *toolset* the agent already
uses, they inherit its hooks (observable on ``tool_events``, gate-able via
``wrap_tool_execute``) and compose with any plan/task tools on the same box — the
orchestrator can keep a shared task-store plan and delegate against it. See
``functions/orchestrator.py`` for the delegation mechanics and recursion guards.

Note on the toolset's role: the delegation tools are ``risk="medium"`` in the
``orchestration`` category, so a connected Role must permit them (Allow-All, or an
explicit selection/category) — a low-risk-ceiling role will (by design) block
delegation.
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QLabel, QSpinBox

from weave.widgetcore import PortRole
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .agent import SilkAgentNode
from .silk_ports import SILK_AGENTS_TYPE  # noqa: F401
from ..functions.stream_events import EventWorker
from ..functions.orchestrator import (
    DEFAULT_MAX_DEPTH,
    attach_orchestrator_tools,
    set_orchestrator_observers,
    set_orchestrator_workers,
)

log = get_logger("SilkOrchestrator")


@register_node
class SilkOrchestratorNode(SilkAgentNode):
    """Autonomous agent that delegates sub-tasks to a roster of worker agents."""

    #: How deep delegation may nest. ``2`` lets the orchestrator call a worker
    #: that itself delegates once; a true cycle is still refused by the chain
    #: guard in ``functions/orchestrator.py``. This is the *seed* for the
    #: editable ``max_depth`` port, and the same constant the runtime
    #: defaults to -- there used to be two values for the one concept, which
    #: is all D55 is about.
    DELEGATION_MAX_DEPTH: ClassVar[int] = DEFAULT_MAX_DEPTH

    node_name: ClassVar[Optional[str]] = "Silk Orchestrator"
    node_description: ClassVar[Optional[str]] = (
        "A Silk Agent that delegates self-contained sub-tasks to a roster of "
        "worker agents (delegate / delegate_parallel), reusing the agent loop."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "orchestration", "delegate", "multi-agent", "llm",
    ]
    node_icon: ClassVar[Optional[str]] = "arrows-split"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT

    def __init__(self, title: str = "Silk Orchestrator", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        # The roster of workers this orchestrator may delegate to (a chain of
        # Agent Spec nodes). Everything else is inherited from SilkAgentNode.
        self.add_input("workers", datatype="silk_agents")

        # Delegation depth is a graph-visible decision, not a class constant
        # (D55): a fan-out that may itself fan out is the difference between
        # a bounded run and a tree, and the person wiring the graph is the
        # one who should see the number. An upstream connection overrides
        # the spin box, which is what PortRole.INPUT means here.
        self.add_input("max_depth", datatype="int")

        self.spin_depth = QSpinBox()
        self.spin_depth.setRange(1, 8)
        self.spin_depth.setValue(self.DELEGATION_MAX_DEPTH)
        self.spin_depth.setToolTip(
            "How deep delegation may nest. 1 lets this orchestrator call "
            "workers but stops a worker from sub-delegating; 2 allows one "
            "further hop. Cycles are refused at any depth."
        )
        self._widget_core.register_widget(
            "max_depth", self.spin_depth, role=PortRole.INPUT,
            datatype="int", default=self.DELEGATION_MAX_DEPTH,
            add_to_layout=False,
        )
        form = getattr(self, "_form_layout", None)
        if form is not None:
            form.addWidget(QLabel("Delegation depth:"))
            form.addWidget(self.spin_depth)

    def delegation_depth(self, inputs: Dict[str, Any]) -> int:
        """The depth this run may nest to: the port, else the spin box.

        Clamped rather than validated away -- a depth of zero would disable
        delegation on a node whose whole purpose is delegating, which is a
        confusing way to answer a typo upstream.
        """
        raw = inputs.get("max_depth", None)
        try:
            depth = int(raw) if raw is not None else self.DELEGATION_MAX_DEPTH
        except (TypeError, ValueError):
            log.warning(
                f"Orchestrator max_depth {raw!r} is not a number; using "
                f"{self.DELEGATION_MAX_DEPTH}."
            )
            depth = self.DELEGATION_MAX_DEPTH
        return max(1, min(8, depth))

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        toolset = inputs.get("toolset")
        workers = list(inputs.get("workers") or [])
        max_depth = self.delegation_depth(inputs)
        # The fan-out's shared ceiling, and the orchestrator's own: one
        # object, because a cap the orchestrator counted into separately
        # from its workers would not be a cap on the fan-out at all. A
        # worker that carries its own caps nests inside this one (D26).
        # An unreadable budget is not diagnosed here -- the agent's own
        # compute refuses the run and says why; taking None keeps that the
        # single place the message comes from.
        try:
            budget = self.take_run_budget(inputs)
        except ValueError:
            budget = None

        if workers and toolset is None:
            log.warning(
                "Orchestrator has workers but no toolset; delegation tools need a "
                "toolset to live on - connect a Silk ToolSet. Running as a plain "
                "agent."
            )
        elif workers and toolset is not None:
            # Mount (or refresh) the delegation tools on the toolset the agent is
            # about to run. Refresh-in-place keeps a re-run's roster current
            # without re-registering the tools (mirrors the task-store handle).
            if "delegate" in getattr(toolset, "tools", {}):
                set_orchestrator_workers(
                    toolset, workers, max_depth=max_depth,
                    usage_limits=budget,
                )
            else:
                attach_orchestrator_tools(
                    toolset, workers=workers, max_depth=max_depth,
                    usage_limits=budget,
                )

        return super().compute(inputs)

    # -- D54: a fan-out you can watch, and stop ------------------------------

    def _attach_run_observers(self, toolset: Any, emit_event: Any) -> None:
        """Re-emit worker events on this node's stream, and forward Stop.

        Both parameters have always existed on ``run_subagent``; the
        orchestrator passed neither, so a long fan-out showed one
        ``delegate`` call and then nothing for minutes, and Stop set the
        orchestrator's own engine flag while the workers ran to completion
        inside the fan-out (spec D54, the most severe instance of G8).

        Worker events arrive tagged with the worker's name, so a nested line
        is attributable — the ``worker`` half of the identity pair whose
        top-level half is the ``agent`` field (D60.1).
        """
        if toolset is None or "delegate" not in getattr(toolset, "tools", {}):
            return

        def _on_worker_event(worker: str, event: Any) -> None:
            if emit_event is None:
                return
            emit_event(EventWorker(
                worker=worker,
                event_type=type(event).__name__,
                digest=_event_digest(event),
            ))

        set_orchestrator_observers(
            toolset,
            on_event=_on_worker_event,
            should_stop=self.is_compute_cancelled,
        )

    def _detach_run_observers(self, toolset: Any) -> None:
        """Drop the run-scoped observers; they close over this run only."""
        if toolset is not None and "delegate" in getattr(toolset, "tools", {}):
            set_orchestrator_observers(toolset, on_event=None, should_stop=None)


def _event_digest(event: Any) -> str:
    """A short, content-light description of a worker's event.

    The stream is previews, never truth (ARCHITECTURE_REVIEW R1), and a
    worker's full deltas would drown the orchestrator's own trace — so this
    reports the shape of what happened, not the text of it.
    """
    name = getattr(event, "tool_name", "")
    if name:
        return name
    text = getattr(event, "text", None)
    if isinstance(text, str) and text:
        return f"{len(text)} chars"
    error = getattr(event, "error", None)
    if isinstance(error, str) and error:
        return error[:200]
    return ""
