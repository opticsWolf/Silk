# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Task planning & tracking tools for the silk agent.

The agent drives its own plan: set a goal, grow a task/subtree, progress it, and
— on the four consequential mutations (add / complete / rescope / revise-goal) —
state a **rationale**. Enforcement is structural: those tools take a required,
non-blank ``rationale`` field, so the model cannot change the plan without saying
why. Plain progress (``task_update``) and reads need none.

Backed by :class:`~weave.plugins.silk.functions.task_store.SqliteTaskStore` (the
store of record; §see docs/TASK_TRACKER_PLAN.md). One plan may be shared by
several agents; a genuine collision (same task, double-complete, goal race) comes
back as an informational ``conflict`` result, never a silent lost update.
"""
from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field, field_validator

# Absolute import: this module is exec'd by the ToolLoader as
# ``dynamic_tools.task_tracker`` (no parent package), so a ``..`` relative import
# would be "beyond top-level". Absolute resolution works in both load paths and
# yields the same module object as a normal import (no duplicate class identity).
from weave.plugins.silk.functions.task_store import (
    Conflict, DEFAULT_ACTOR, LedgerClosed, PlanRef, SqliteTaskStore, plan_to_json,
    render_markdown,
)

if TYPE_CHECKING:
    from ..tool_box import ToolBox
    from .file_sandbox import FileToolSandbox


# ── rationale validator (shared) ────────────────────────────────────────────

def _non_blank(v: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(
            "rationale must be a non-empty sentence explaining the change"
        )
    return v.strip()


# ── request schemas ─────────────────────────────────────────────────────────

class TaskSpec(BaseModel):
    title: str = Field(..., description="Task title.")
    parent: Optional[str] = Field(None, description="Parent task id for a subtask.")
    note: str = Field("", description="Optional progress note.")


class PlanStartArgs(BaseModel):
    goal: str = Field(..., min_length=1, description="The overall goal to pursue.")
    acceptance: list[str] = Field(
        default_factory=list,
        description="Acceptance criteria that define 'done' for the goal.",
    )
    tasks: list[TaskSpec] = Field(
        default_factory=list,
        description="Initial task decomposition (order preserved).",
    )


class PlanViewArgs(BaseModel):
    section: str = Field(
        "all", description="Which section: 'all', 'tasks', or 'goal'.",
    )


class PlanHistoryArgs(BaseModel):
    limit: int = Field(20, gt=0, le=500, description="Max recent entries.")


class TaskAddArgs(BaseModel):
    title: str = Field(..., min_length=1, description="Title of the new task.")
    parent: Optional[str] = Field(None, description="Parent task id (for a subtask).")
    note: str = Field("", description="Optional progress note.")
    rationale: str = Field(..., description="Why this work is being added.")
    _v = field_validator("rationale")(_non_blank)


class TaskUpdateArgs(BaseModel):
    id: str = Field(..., description="Task id, e.g. 't3'.")
    status: Optional[str] = Field(
        None,
        description="New status: pending | in_progress | blocked. "
                    "Use task_complete for done, task_rescope to drop.",
    )
    title: Optional[str] = Field(None, description="Corrected title (typo-level).")
    note: Optional[str] = Field(None, description="Updated progress note.")


class TaskCompleteArgs(BaseModel):
    id: str = Field(..., description="Task id to mark done.")
    rationale: str = Field(
        ..., description="Completion justification (e.g. what verifies it's done).",
    )
    _v = field_validator("rationale")(_non_blank)


class TaskRescopeArgs(BaseModel):
    id: str = Field(..., description="Task id to re-scope or drop.")
    new_title: Optional[str] = Field(None, description="New title, if re-scoping.")
    new_status: str = Field(
        "dropped",
        description="New status (usually 'dropped'; may be pending/in_progress/"
                    "blocked to re-scope in place).",
    )
    rationale: str = Field(..., description="Why the task is dropped / re-scoped.")
    _v = field_validator("rationale")(_non_blank)


class GoalReviseArgs(BaseModel):
    new_text: Optional[str] = Field(None, description="Revised goal text.")
    acceptance_add: list[str] = Field(
        default_factory=list, description="Acceptance criteria to add.",
    )
    acceptance_remove: list[str] = Field(
        default_factory=list, description="Acceptance criteria to remove.",
    )
    rationale: str = Field(..., description="Why the goal / acceptance changes.")
    _v = field_validator("rationale")(_non_blank)


class TaskClaimArgs(BaseModel):
    id: str = Field(..., description="Task id to claim (advisory soft ownership).")


# ── response schema ─────────────────────────────────────────────────────────

class PlanResult(BaseModel):
    """Structured reply. ``error`` stays null on success (so reflection doesn't
    misread it); a benign collision surfaces via ``conflict`` with ``ok=False``."""
    ok: bool = Field(..., description="True if the operation was applied.")
    revision: Optional[int] = Field(None, description="Plan revision after the op.")
    message: Optional[str] = Field(None, description="Human-readable outcome.")
    conflict: Optional[str] = Field(
        None, description="Set when another agent's state blocked the op; re-read "
                          "and decide (not an error).",
    )
    plan: Optional[dict] = Field(None, description="Full plan snapshot (plan_view).")
    markdown: Optional[str] = Field(None, description="Rendered plan (plan_view).")
    history: Optional[list] = Field(None, description="Recent ops (plan_history).")
    error: Optional[str] = Field(None, description="Execution error, if any.")
    retryable: bool = Field(
        False, description="True when the same call is expected to work "
                           "shortly (Weave is restarting); re-issue it rather "
                           "than abandoning the plan.",
    )


# ── helpers ─────────────────────────────────────────────────────────────────

def _get_store(toolbox: Any) -> "SqliteTaskStore":
    store = getattr(toolbox, "_task_store", None)
    if store is None:
        raise RuntimeError(
            "No task store configured. Call attach_task_tools(sandbox=...) first."
        )
    return store


def _actor(user_session: Optional[dict]) -> str:
    us = user_session or {}
    return str(us.get("agent_id") or us.get("actor") or us.get("user_id")
               or DEFAULT_ACTOR)


def _refuses_gracefully(fn):
    """Turn a closed ledger into a reply instead of an exception.

    A write that arrives while Weave is shutting down is not a bug in
    the call, and the agent has nowhere to put a traceback.  It is the
    same shape as a conflict — "not applied, here is why" — with the
    one addition a conflict does not need: whether asking again will
    help.  A relaunch is a few seconds and a replacement; a quit is not.
    """
    @wraps(fn)
    def guarded(*args, **kwargs) -> PlanResult:
        try:
            return fn(*args, **kwargs)
        except LedgerClosed as exc:
            return PlanResult(ok=False, message=str(exc),
                              retryable=exc.retryable)
    return guarded


def _result(outcome: Any, *, message: str) -> PlanResult:
    """Normalize a store return (Plan | Conflict | None) into a PlanResult."""
    if outcome is None:
        return PlanResult(
            ok=False, message="No plan exists yet. Call plan_start first.",
        )
    if isinstance(outcome, Conflict):
        return PlanResult(
            ok=False, revision=outcome.current_revision, conflict=outcome.reason,
            message=f"Blocked: {outcome.reason}",
        )
    return PlanResult(ok=True, revision=outcome.revision, message=message)


# ── registration ────────────────────────────────────────────────────────────

def attach_task_tools(toolbox: "ToolBox", sandbox: "FileToolSandbox",
                      plan: Any = None) -> None:
    """Mount the task planning/tracking tools.

    Without *plan* the store is rooted at the sandbox working directory and
    finds the newest plan there -- shared-plan discovery, which is how
    several agents in one root work on one plan. With a `PlanRef` from a
    Task node the plan is named outright (D23), so two unrelated plans in
    one directory cannot cross-discover by file timestamp.
    """
    ref = PlanRef.coerce(plan)
    if ref is not None and (ref.is_explicit or ref.root):
        store = ref.store()
    else:
        store = SqliteTaskStore(root=getattr(sandbox, "root_dir", "."))
    toolbox._task_store = store  # type: ignore[attr-defined]

    @toolbox.register(
        name="plan_start",
        tags=("planning",), category="planning", risk="low",
        description=(
            "Create a task plan: set the goal, acceptance criteria, and an initial "
            "task decomposition. Call once at the start; errors if a plan exists."
        ),
        args_model=PlanStartArgs,
        procedure=(
            "Start the run's plan.\n"
            "- goal: the overall objective; acceptance: what 'done' means.\n"
            "- tasks: an ordered initial decomposition (each {title, parent?, note?}).\n"
            "- You can expand/adjust later with task_add / task_update; changing "
            "course (drop/rescope/revise-goal) or completing a task requires a "
            "rationale.\n"
            "- Reply JSON: ok, revision, message."
        ),
    )
    @_refuses_gracefully
    def _plan_start(
        db_pool: Any, user_session: dict,
        goal: str, acceptance: Optional[list] = None,
        tasks: Optional[list] = None,
    ) -> PlanResult:
        try:
            plan = store.start(
                goal=goal, acceptance=acceptance or [],
                tasks=[dict(t) for t in (tasks or [])],
                actor=_actor(user_session),
            )
        except LedgerClosed:
            raise                      # the decorator has the better answer
        except ValueError as e:
            return PlanResult(ok=False, message=str(e))
        return PlanResult(
            ok=True, revision=plan.revision,
            message=f"Plan created with {len(plan.tasks)} task(s).",
        )

    @toolbox.register(
        name="plan_view",
        tags=("planning", "read"), category="planning", risk="low",
        description="View the current plan: goal, task tree, and course "
                    "corrections. Returns the JSON snapshot and rendered markdown.",
        args_model=PlanViewArgs,
        procedure=(
            "Read the current plan.\n"
            "- Reply JSON: plan (full snapshot), markdown (rendered view), "
            "revision.\n"
            "- Use this before deciding next steps or when resuming."
        ),
    )
    def _plan_view(db_pool: Any, user_session: dict, section: str = "all") -> PlanResult:
        plan = store.load()
        if plan is None:
            return PlanResult(ok=False, message="No plan exists yet.")
        return PlanResult(
            ok=True, revision=plan.revision, plan=plan_to_json(plan),
            markdown=render_markdown(plan),
        )

    @toolbox.register(
        name="plan_history",
        tags=("planning", "read"), category="planning", risk="low",
        description="List recent plan changes (newest first): op, actor, "
                    "timestamp, and rationale — the audit trail.",
        args_model=PlanHistoryArgs,
        procedure=(
            "Read the plan's change history.\n"
            "- Reply JSON: history (list of {revision, at, actor, op, rationale}).\n"
            "- Deviations (rescope / goal_revise) also carry before/after in the plan."
        ),
    )
    def _plan_history(db_pool: Any, user_session: dict, limit: int = 20) -> PlanResult:
        return PlanResult(ok=True, history=store.history(limit=limit))

    @toolbox.register(
        name="task_add",
        tags=("planning",), category="planning", risk="low",
        description="Add a task (or subtask) to the plan. Requires a rationale.",
        args_model=TaskAddArgs,
        procedure=(
            "Expand the plan with new work.\n"
            "- parent: omit for a top-level task, or give a task id for a subtask.\n"
            "- rationale (required): why this work is needed.\n"
            "- Reply JSON: ok, revision, message."
        ),
    )
    @_refuses_gracefully
    def _task_add(
        db_pool: Any, user_session: dict,
        title: str, rationale: str, parent: Optional[str] = None, note: str = "",
    ) -> PlanResult:
        out = store.add_task(
            title=title, parent=parent, note=note,
            actor=_actor(user_session), rationale=rationale,
        )
        rev = out.revision if hasattr(out, "revision") and not isinstance(out, Conflict) else None
        return _result(out, message=f"Task added (revision {rev}).")

    @toolbox.register(
        name="task_update",
        tags=("planning",), category="planning", risk="low",
        description=(
            "Update a task's progress: status (pending/in_progress/blocked), note, "
            "or a typo-level title. Refuses status='done' (use task_complete) and "
            "'dropped' (use task_rescope). No rationale needed."
        ),
        args_model=TaskUpdateArgs,
        procedure=(
            "Record plain progress on a task.\n"
            "- status: pending | in_progress | blocked only.\n"
            "- To finish a task use task_complete; to drop it use task_rescope "
            "(both need a rationale).\n"
            "- Reply JSON: ok, revision, message."
        ),
    )
    @_refuses_gracefully
    def _task_update(
        db_pool: Any, user_session: dict,
        id: str, status: Optional[str] = None, title: Optional[str] = None,
        note: Optional[str] = None,
    ) -> PlanResult:
        out = store.update_task(
            task_id=id, status=status, title=title, note=note,
            actor=_actor(user_session),
        )
        return _result(out, message="Task updated.")

    @toolbox.register(
        name="task_complete",
        tags=("planning",), category="planning", risk="low",
        description="Mark a task done. Requires a rationale (the completion "
                    "justification, e.g. what verifies it).",
        args_model=TaskCompleteArgs,
        procedure=(
            "Finish a task.\n"
            "- rationale (required): what makes it done (e.g. 'tests pass').\n"
            "- Fails if already done or dropped (returns conflict).\n"
            "- Reply JSON: ok, revision, message."
        ),
    )
    @_refuses_gracefully
    def _task_complete(
        db_pool: Any, user_session: dict, id: str, rationale: str,
    ) -> PlanResult:
        out = store.complete_task(
            task_id=id, actor=_actor(user_session), rationale=rationale,
        )
        return _result(out, message="Task completed.")

    @toolbox.register(
        name="task_rescope",
        tags=("planning",), category="planning", risk="low",
        description="Drop or re-scope an existing task (a deviation). Requires a "
                    "rationale; recorded in the course-corrections ledger.",
        args_model=TaskRescopeArgs,
        procedure=(
            "Change what a task means, or drop it.\n"
            "- new_status defaults to 'dropped'; give new_title to re-scope in place.\n"
            "- rationale (required): why the course change.\n"
            "- Recorded as a deviation (before/after + rationale).\n"
            "- Reply JSON: ok, revision, message."
        ),
    )
    @_refuses_gracefully
    def _task_rescope(
        db_pool: Any, user_session: dict,
        id: str, rationale: str, new_title: Optional[str] = None,
        new_status: str = "dropped",
    ) -> PlanResult:
        out = store.rescope_task(
            task_id=id, new_title=new_title, new_status=new_status,
            actor=_actor(user_session), rationale=rationale,
        )
        return _result(out, message="Task re-scoped.")

    @toolbox.register(
        name="goal_revise",
        tags=("planning",), category="planning", risk="low",
        description="Revise the goal or its acceptance criteria (a deviation). "
                    "Requires a rationale; the original goal is preserved.",
        args_model=GoalReviseArgs,
        procedure=(
            "Change the plan's goal or acceptance criteria.\n"
            "- new_text: revised goal; acceptance_add/remove: adjust criteria.\n"
            "- rationale (required): why the goal changes.\n"
            "- The original goal is kept; recorded as a deviation.\n"
            "- Reply JSON: ok, revision, message."
        ),
    )
    @_refuses_gracefully
    def _goal_revise(
        db_pool: Any, user_session: dict,
        rationale: str, new_text: Optional[str] = None,
        acceptance_add: Optional[list] = None,
        acceptance_remove: Optional[list] = None,
    ) -> PlanResult:
        out = store.revise_goal(
            new_text=new_text, acceptance_add=acceptance_add or [],
            acceptance_remove=acceptance_remove or [],
            actor=_actor(user_session), rationale=rationale,
        )
        return _result(out, message="Goal revised.")

    @toolbox.register(
        name="task_claim",
        tags=("planning",), category="planning", risk="low",
        description="Advisory soft-ownership: claim a task so co-working agents "
                    "divide labour. Never hard-locks; no rationale needed.",
        args_model=TaskClaimArgs,
        procedure=(
            "Claim a task (advisory) when several agents share the plan.\n"
            "- Returns a conflict if another agent already holds it.\n"
            "- Reply JSON: ok, revision, message."
        ),
    )
    @_refuses_gracefully
    def _task_claim(db_pool: Any, user_session: dict, id: str) -> PlanResult:
        out = store.claim_task(task_id=id, actor=_actor(user_session))
        return _result(out, message="Task claimed.")

