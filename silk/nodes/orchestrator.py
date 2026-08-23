# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0

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

from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .agent import SilkAgentNode
from .silk_ports import SILK_AGENTS_TYPE  # noqa: F401
from ..functions.orchestrator import (
    attach_orchestrator_tools,
    set_orchestrator_workers,
)

log = get_logger("SilkOrchestrator")


@register_node
class SilkOrchestratorNode(SilkAgentNode):
    """Autonomous agent that delegates sub-tasks to a roster of worker agents."""

    #: How deep delegation may nest. ``2`` lets the orchestrator call a worker
    #: that itself delegates once; a true cycle is still refused by the chain
    #: guard in ``functions/orchestrator.py``.
    DELEGATION_MAX_DEPTH: ClassVar[int] = 2

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

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        toolset = inputs.get("toolset")
        workers = list(inputs.get("workers") or [])

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
                    toolset, workers, max_depth=self.DELEGATION_MAX_DEPTH,
                )
            else:
                attach_orchestrator_tools(
                    toolset, workers=workers, max_depth=self.DELEGATION_MAX_DEPTH,
                )

        return super().compute(inputs)
