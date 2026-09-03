# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Qt-free sub-agent runner — the reusable core of one autonomous agent turn.

``SilkAgentNode.compute`` drives an :class:`AgentLoop` over a model + toolset +
role and streams the events to Qt. That same drive logic is what an
**orchestrator** needs when it delegates a sub-task to a worker agent: run a
fresh :class:`AgentLoop` to completion and hand back the final answer. This
module is that logic with the Qt removed — a plain function returning a
:class:`SubagentResult`, usable by the node, the delegation tools
(``tools/orchestrator.py``), tests, and a future CLI.

Isolation is the point of "clean" agent-to-agent hand-off: a delegated run gets
a **fresh history** (no context leak from the caller) and its **own pool
session** (its own KV cache), and the worker's toolset is activated under its
own :class:`RoleBinding`. Because each worker carries an independent toolset
instance (``build_toolset`` rebuilds the ToolBox per agent), several sub-agents
may run concurrently without fighting over one binding.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from weave.logger import get_logger

from .agent_loop import AgentLoop, DEFAULT_MAX_ROUNDS
from .graph_engine import GraphEngine
from .role import DEFAULT_ROLE, Role, RoleBinding
from .stream_events import (
    OUTCOME_COMPLETED,
    EventError,
    EventRunResult,
    EventToolCall,
)
from .tool_calling import tool_call_instructions
from .usage_limits import nest

log = get_logger("SilkSubagent")


# ── system-prompt composition (shared with SilkAgentNode) ────────────────────

def compose_system_prompt(base: str, role: Any, toolset: Any) -> str:
    """base prompt + [ROLE] block + capability/procedure blocks + tool protocol.

    Must be called *after* role activation so every toolset-derived section
    sees the role filter (denied tools are never advertised). The single
    implementation shared by the Agent node and the sub-agent runner.
    """
    sections = [base.strip()] if base.strip() else []
    role_block = role.system_prompt_block()
    if role_block:
        sections.append(role_block)
    if toolset is not None:
        sections.append(toolset.build_system_prompt("").strip())
        sections.append(tool_call_instructions(toolset).strip())
    return "\n\n".join(s for s in sections if s)


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentSpec:
    """A runnable agent bundle: model handle + optional toolset + role.

    A worker registered with an orchestrator, or a lightweight headless agent.
    ``toolset`` is a ToolBox instance (from ``build_toolset``); ``None`` means
    pure chat (tool fences are treated as final output). ``name`` /
    ``description`` are what the orchestrator advertises to its model.
    """

    model_handle: dict[str, Any]
    toolset: Optional[Any] = None
    role: Role = field(default_factory=lambda: DEFAULT_ROLE)
    name: str = ""
    description: str = ""
    system_prompt: str = ""
    max_rounds: Optional[int] = None
    gen_params: dict[str, Any] = field(default_factory=dict)
    #: This worker's own caps. An orchestrator's shared budget does not
    #: replace them: with both, these become a ``SubBudget`` *inside* the
    #: shared cap, so one greedy worker cannot spend the fan-out's whole
    #: allowance and no worker can raise the global ceiling (D26, T3).
    usage_limits: Optional[Any] = None

    def is_runnable(self) -> tuple[bool, str]:
        """Whether the spec has a usable GGUF model handle."""
        h = self.model_handle
        if not (
            isinstance(h, dict)
            and h.get("backend") == "gguf"
            and ("model" in h or "pool" in h)
        ):
            return False, "no valid GGUF model in agent spec"
        return True, ""


@dataclass
class SubagentResult:
    """Outcome of one sub-agent run — the reply plus a small run trace."""

    text: str
    ok: bool
    error: Optional[str] = None
    rounds: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


# ── the runner ───────────────────────────────────────────────────────────────

def run_subagent(
    spec: AgentSpec,
    prompt: str,
    *,
    history: Optional[list[dict[str, Any]]] = None,
    gen_params: Optional[dict[str, Any]] = None,
    session_id: Optional[str] = None,
    usage_limits: Optional[Any] = None,
    on_event: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> SubagentResult:
    """Run one autonomous agent turn to completion; Qt-free.

    Args:
        spec: The agent to run (model + toolset + role).
        prompt: The task/user input for this turn.
        history: Prior conversation to continue; ``None`` (default) starts a
            fresh, isolated context — the norm for a delegated sub-task.
        gen_params: Generation params; merged over the spec's defaults and,
            last, over the role's ``model_settings`` gap-fill.
        session_id: Pool session key; a fresh UUID by default so a sub-agent's
            KV cache never collides with the caller's.
        on_event: Optional sink for the loop's typed events (streaming/UI).
        should_stop: Optional predicate polled each event; True requests a
            graceful stop at the next round boundary.

    Returns:
        A :class:`SubagentResult`. On a hard failure (bad model, toolset held
        by another binding, stream error with no text) ``ok`` is False and
        ``error`` is set; ``text`` still carries an ``"Error: …"`` line so a
        caller that feeds it straight back to a model gets a legible result.
    """
    runnable, why = spec.is_runnable()
    if not runnable:
        return SubagentResult(text=f"Error: {why}", ok=False, error=why)

    prompt = str(prompt or "").strip()
    if not prompt:
        return SubagentResult(
            text="Error: empty task prompt", ok=False, error="empty task prompt",
        )

    role = spec.role or DEFAULT_ROLE
    toolset = spec.toolset
    binding: Optional[RoleBinding] = None
    try:
        if toolset is not None:
            try:
                binding = RoleBinding.activate(role, toolset)
            except RuntimeError as exc:
                # The worker's toolset is already bound by another live agent.
                return SubagentResult(text=f"Error: {exc}", ok=False, error=str(exc))

        system_prompt = compose_system_prompt(spec.system_prompt or "", role, toolset)

        engine = GraphEngine(
            spec.model_handle,
            system_prompt=system_prompt,
            history=history if history is not None else [],
            usage_limits=nest(usage_limits, spec.usage_limits),
            session_id=session_id or str(uuid.uuid4()),
        )
        engine.clear_stop()

        loop = AgentLoop(
            engine,
            toolset,
            max_rounds=spec.max_rounds or role.max_rounds or DEFAULT_MAX_ROUNDS,
        )

        params: dict[str, Any] = {"temperature": 0.7}
        params.update(spec.gen_params or {})
        params.update(gen_params or {})
        if binding is not None:
            params = binding.effective_gen_params(params)

        final_text = ""
        run_error: Optional[str] = None
        # How the run ended, straight from the loop. A worker that hit
        # max_rounds or its budget produces text, and reading "there is
        # text, so it worked" is exactly how such a run used to be handed
        # back to an orchestrator as a success (G13).
        outcome = OUTCOME_COMPLETED
        rounds = 0
        tool_calls: list[dict[str, Any]] = []

        for event in loop.run(prompt, params):
            if should_stop is not None and should_stop():
                engine.request_stop()
            if on_event is not None:
                on_event(event)
            if isinstance(event, EventToolCall):
                tool_calls.append({"tool": event.tool_name, "args": event.tool_args})
            elif isinstance(event, EventError):
                run_error = event.error
            elif isinstance(event, EventRunResult):
                final_text = event.text
                outcome = event.outcome or OUTCOME_COMPLETED
                rounds = len(event.tool_calls or ())

        if outcome != OUTCOME_COMPLETED:
            # Text *and* a reason: an orchestrator deciding what to do next
            # needs the partial answer, and needs to know it is partial.
            reason = run_error or f"the run ended {outcome}"
            return SubagentResult(
                text=final_text or f"Error: {reason}", ok=False, error=reason,
                rounds=rounds, tool_calls=tool_calls,
            )
        if run_error and not final_text:
            return SubagentResult(
                text=f"Error: {run_error}", ok=False, error=run_error,
                tool_calls=tool_calls,
            )
        return SubagentResult(
            text=final_text, ok=True, rounds=rounds, tool_calls=tool_calls,
        )
    except Exception as exc:  # never propagate into the caller's tool dispatch
        log.error(f"Sub-agent run failed: {exc}", exc_info=True)
        return SubagentResult(text=f"Error: {exc}", ok=False, error=str(exc))
    finally:
        if binding is not None:
            binding.deactivate()
