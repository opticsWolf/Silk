# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0

Orchestrator delegation tools — dynamic agent-to-agent hand-off.

An **orchestrator** is just a Silk agent whose toolset includes these tools: it
plans and reasons like any agent, and when a sub-task belongs to a specialist it
calls ``delegate(worker=…, task=…)`` (or ``delegate_parallel`` for a fan-out).
The tool runs that worker as an in-process sub-agent
(:func:`~.subagent.run_subagent`) to completion and returns its answer, so no
graph request/response plumbing is needed — the whole existing loop / toolset /
role / hook / reflection stack is reused for the child run.

Integration with the plan + hook systems (deliberate, not bolted on):

* **Observable for free.** ``delegate`` is a normal tool, so it flows through the
  ToolBox's ``before/after_tool_execute`` hooks — every delegation already shows
  up on the Agent node's ``tool_events`` stream, and the returned trace names the
  worker's own tool calls.
* **Gate-able for free.** Being a normal tool, ``delegate`` also passes through
  the ``wrap_tool_execute`` middleware, so a hook can require approval before a
  delegation runs. It is registered ``risk="medium"``, so a role with
  ``max_risk="low"`` blocks delegation outright — role enforcement is the static
  gate, ``wrap_tool_execute`` the dynamic one.
* **Shared blackboard, no glue.** Coordination leans on the task store: the
  orchestrator keeps the plan with the ordinary ``plan_*`` / ``task_*`` tools, and
  a worker whose sandbox shares the orchestrator's root opens the *same* SQLite
  plan file — the store's WAL + ``Conflict`` concurrency makes claim/complete from
  a worker safe without any store-injection code here.

Live config lives on the ToolBox (like ``_task_store``): the roster, depth cap,
gen-params and shared budget are attributes the tools read at call time, so a node
re-run refreshes the roster via :func:`set_orchestrator_workers` without
re-registering tools.

Bounded recursion (Phase E): every delegation is guarded by ``max_depth`` (a child
that is itself an orchestrator sees the incremented depth) **and** by a delegation
*chain* — a worker already on the current chain is refused as a cycle. Both live
on the child toolset, not the shared ``user_session``.

Why sync: ``delegate`` / ``delegate_parallel`` are registered **synchronous** on
purpose. The ToolBox offloads sync tools via ``asyncio.to_thread``, giving the
nested :class:`AgentLoop` (which itself calls ``asyncio.run``) a clean event loop
on its own thread. An async tool would raise "asyncio.run() from a running loop".
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field, field_validator

from weave.logger import get_logger

from .messaging import AgentMessage
from .subagent import AgentSpec, run_subagent

if TYPE_CHECKING:
    from .tool_box import ToolBox
    from .tools.file_sandbox import FileToolSandbox

log = get_logger("SilkOrchestrator")

#: ToolBox attributes: live delegation config + per-run recursion state.
_WORKERS_ATTR = "_orchestrator_workers"
_DEPTH_CAP_ATTR = "_orchestrator_max_depth"
_GEN_ATTR = "_orchestrator_gen_params"
_BUDGET_ATTR = "_orchestrator_usage_limits"
_DEPTH_ATTR = "_delegation_depth"
_CHAIN_ATTR = "_delegation_chain"

#: Ceiling on a single parallel fan-out (guards against a runaway assignment list).
_MAX_PARALLEL = 8


def _non_blank(v: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError("task must be a non-empty instruction for the worker")
    return v.strip()


def _worker_map(workers: Any) -> dict[str, AgentSpec]:
    """Normalise *workers* (a list of specs or a name→spec dict) into a map.

    List entries are keyed by ``spec.name``; an unnamed spec falls back to
    ``worker{n}`` so it is still addressable.
    """
    if isinstance(workers, dict):
        return {str(k): v for k, v in workers.items()}
    out: dict[str, AgentSpec] = {}
    for i, spec in enumerate(workers or ()):
        name = (getattr(spec, "name", "") or f"worker{i + 1}").strip()
        out[name] = spec
    return out


def set_orchestrator_workers(
    toolbox: "ToolBox",
    workers: Any,
    *,
    max_depth: Optional[int] = None,
    gen_params: Optional[dict[str, Any]] = None,
    usage_limits: Optional[Any] = None,
) -> None:
    """Refresh the live roster/config on an already-wired orchestrator toolbox.

    Lets a node re-run update who the orchestrator may call without tearing down
    and re-registering the tools. Only provided fields are changed.
    """
    setattr(toolbox, _WORKERS_ATTR, _worker_map(workers))
    if max_depth is not None:
        setattr(toolbox, _DEPTH_CAP_ATTR, int(max_depth))
    if gen_params is not None:
        setattr(toolbox, _GEN_ATTR, dict(gen_params))
    if usage_limits is not None:
        setattr(toolbox, _BUDGET_ATTR, usage_limits)


# ── request / response schemas ───────────────────────────────────────────────

class ListWorkersArgs(BaseModel):
    pass


class DelegateArgs(BaseModel):
    worker: str = Field(..., description="Name of the worker agent to delegate to.")
    task: str = Field(
        ..., description="Self-contained instruction for the worker (it has no "
                         "access to your context — state everything it needs).",
    )
    context: str = Field(
        "", description="Optional extra background to prepend to the task.",
    )
    _v = field_validator("task")(_non_blank)


class Assignment(BaseModel):
    worker: str = Field(..., description="Worker agent name.")
    task: str = Field(..., description="Self-contained instruction for the worker.")
    context: str = Field("", description="Optional background prepended to the task.")
    _v = field_validator("task")(_non_blank)


class DelegateParallelArgs(BaseModel):
    assignments: list[Assignment] = Field(
        ..., min_length=1,
        description="Independent sub-tasks to run concurrently, each {worker, "
                    "task, context?}. Use only when the sub-tasks do not depend "
                    "on each other's output.",
    )


class WorkerInfo(BaseModel):
    name: str
    description: str = ""


class ListWorkersResult(BaseModel):
    ok: bool = True
    workers: list[WorkerInfo] = Field(default_factory=list)
    message: Optional[str] = None


class DelegateResult(BaseModel):
    """Structured reply. ``error`` stays null on success so reflection doesn't
    misread a normal answer as a failure."""
    ok: bool = Field(..., description="True if the worker returned an answer.")
    worker: str = Field(..., description="Worker that ran (or was requested).")
    answer: Optional[str] = Field(None, description="The worker's final response.")
    tools_used: list[str] = Field(
        default_factory=list, description="Tools the worker called, in order.",
    )
    correlation_id: Optional[str] = Field(
        None, description="Ties this reply to the delegation request (tracing).",
    )
    error: Optional[str] = Field(None, description="Failure reason, if any.")


class DelegateParallelResult(BaseModel):
    ok: bool = Field(..., description="True if every delegation returned an answer.")
    results: list[DelegateResult] = Field(default_factory=list)
    message: Optional[str] = None


# ── core delegation (shared by delegate + delegate_parallel) ──────────────────

def _run_one(
    toolbox: Any, worker: str, task: str, context: str, actor: str,
) -> DelegateResult:
    """Resolve, guard, and run one worker; never raises (packs errors in-band)."""
    roster: dict[str, AgentSpec] = getattr(toolbox, _WORKERS_ATTR, {}) or {}
    max_depth = int(getattr(toolbox, _DEPTH_CAP_ATTR, 1) or 1)
    base_gen = dict(getattr(toolbox, _GEN_ATTR, {}) or {})
    budget = getattr(toolbox, _BUDGET_ATTR, None)
    depth = int(getattr(toolbox, _DEPTH_ATTR, 0) or 0)
    chain: list[str] = list(getattr(toolbox, _CHAIN_ATTR, []) or [])

    if depth >= max_depth:
        return DelegateResult(
            ok=False, worker=worker,
            error=(f"Delegation depth limit reached ({max_depth}); this agent may "
                   "not delegate further. Do the work directly."),
        )
    if worker in chain:
        return DelegateResult(
            ok=False, worker=worker,
            error=(f"Delegation cycle detected: '{worker}' is already active in "
                   f"this chain ({' -> '.join(chain)}). Refusing to recurse."),
        )

    spec = roster.get(worker)
    if spec is None:
        known = ", ".join(roster) or "none"
        return DelegateResult(
            ok=False, worker=worker,
            error=f"Unknown worker '{worker}'. Available: {known}.",
        )

    request = AgentMessage(
        content=(f"{context.strip()}\n\n{task}".strip() if context.strip() else task),
        sender=actor or "orchestrator", recipient=worker, kind="task",
    )

    # Push depth + chain onto the child toolset so a worker-that-is-an-
    # orchestrator sees them on its own toolbox and stops in time.
    child_toolset = spec.toolset
    if child_toolset is not None:
        setattr(child_toolset, _DEPTH_ATTR, depth + 1)
        setattr(child_toolset, _CHAIN_ATTR, chain + [worker])

    result = run_subagent(
        spec, request.content,
        gen_params={**base_gen, **(spec.gen_params or {})},
        usage_limits=budget,
    )

    reply = request.reply(result.text, sender=worker,
                          kind="result" if result.ok else "error")
    return DelegateResult(
        ok=result.ok, worker=worker,
        answer=result.text if result.ok else None,
        tools_used=[c.get("tool", "") for c in result.tool_calls],
        correlation_id=reply.correlation_id,
        error=None if result.ok else (result.error or "worker failed"),
    )


# ── registration ─────────────────────────────────────────────────────────────

def attach_orchestrator_tools(
    toolbox: "ToolBox",
    sandbox: "Optional[FileToolSandbox]" = None,
    *,
    workers: Any,
    max_depth: int = 1,
    gen_params: Optional[dict[str, Any]] = None,
    usage_limits: Optional[Any] = None,
) -> None:
    """Mount ``list_workers`` + ``delegate`` + ``delegate_parallel`` on *toolbox*.

    Args:
        toolbox: The orchestrator agent's ToolBox (its toolset).
        sandbox: Unused (kept for the ``attach_*(toolbox, sandbox)`` shape).
        workers: A list of :class:`AgentSpec` (keyed by ``name``) or a
            ``{name: AgentSpec}`` map — the agents this orchestrator may call.
        max_depth: How deep delegation may nest. ``1`` lets the orchestrator call
            workers but stops a worker from sub-delegating.
        gen_params: Generation overrides applied to every worker run (a worker's
            own ``spec.gen_params`` still win per key).
        usage_limits: Optional shared budget threaded into every worker run so a
            fan-out respects one global cap (see :class:`UsageLimits`).
    """
    set_orchestrator_workers(
        toolbox, workers, max_depth=max_depth,
        gen_params=gen_params or {}, usage_limits=usage_limits,
    )

    @toolbox.register(
        name="list_workers",
        tags=("orchestration",), category="orchestration", risk="low",
        description="List the worker agents you can delegate to, with their "
                    "specialities. Call this before delegate if unsure who fits.",
        args_model=ListWorkersArgs,
        procedure=(
            "See who you can delegate to.\n"
            "- Reply JSON: workers (list of {name, description}).\n"
            "- Pick the worker whose speciality matches the sub-task, then "
            "delegate."
        ),
    )
    def _list_workers(db_pool: Any, user_session: dict) -> ListWorkersResult:
        roster: dict[str, AgentSpec] = getattr(toolbox, _WORKERS_ATTR, {}) or {}
        return ListWorkersResult(
            workers=[
                WorkerInfo(name=name, description=getattr(s, "description", "") or "")
                for name, s in roster.items()
            ],
            message=f"{len(roster)} worker(s) available.",
        )

    @toolbox.register(
        name="delegate",
        tags=("orchestration",), category="orchestration", risk="medium",
        description=(
            "Delegate a self-contained sub-task to a named worker agent and get "
            "its answer back. The worker runs in a fresh, isolated context — it "
            "sees only the task you give it, not your conversation."
        ),
        args_model=DelegateArgs,
        procedure=(
            "Hand a sub-task to a specialist worker.\n"
            "- worker: a name from list_workers.\n"
            "- task (required): a COMPLETE instruction; the worker has none of "
            "your context, so include every fact and the exact output you want.\n"
            "- context: optional extra background prepended to the task.\n"
            "- Reply JSON: ok, worker, answer, tools_used, error. Integrate the "
            "answer yourself; the worker does not see your other steps."
        ),
    )
    def _delegate(
        db_pool: Any, user_session: dict,
        worker: str, task: str, context: str = "",
    ) -> DelegateResult:
        return _run_one(toolbox, worker, task, context, _actor(user_session))

    @toolbox.register(
        name="delegate_parallel",
        tags=("orchestration",), category="orchestration", risk="medium",
        description=(
            "Delegate several INDEPENDENT sub-tasks to workers at once and get "
            "all answers back. Use only when the sub-tasks do not depend on each "
            "other's output; otherwise call delegate sequentially."
        ),
        args_model=DelegateParallelArgs,
        procedure=(
            "Fan out independent sub-tasks concurrently.\n"
            "- assignments: a list of {worker, task, context?}; each task must be "
            "self-contained.\n"
            "- Reply JSON: ok (all succeeded), results (one per assignment, in "
            "order). Runs in parallel; do NOT use for a pipeline where one task "
            "needs another's output."
        ),
    )
    def _delegate_parallel(
        db_pool: Any, user_session: dict, assignments: list,
    ) -> DelegateParallelResult:
        actor = _actor(user_session)
        items = [
            (a.get("worker", ""), a.get("task", ""), a.get("context", ""))
            for a in (assignments or [])
        ][:_MAX_PARALLEL]
        if not items:
            return DelegateParallelResult(ok=False, message="No assignments given.")

        # Each _run_one → run_subagent uses asyncio.run internally, so it must run
        # on its own thread (never the caller's loop). A thread pool gives each
        # one an isolated event loop; workers with distinct toolsets don't clash.
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            results = list(pool.map(
                lambda it: _run_one(toolbox, it[0], it[1], it[2], actor), items,
            ))
        ok = all(r.ok for r in results)
        done = sum(1 for r in results if r.ok)
        return DelegateParallelResult(
            ok=ok, results=results,
            message=f"{done}/{len(results)} delegations succeeded.",
        )


def _actor(user_session: Optional[dict]) -> str:
    us = user_session or {}
    return str(us.get("agent_id") or us.get("actor") or "orchestrator")
