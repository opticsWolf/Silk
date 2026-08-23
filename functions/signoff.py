# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0

User sign-off gate for the task tracker — per-change-type approval policy.

A **policy** maps each change type to who may sign it:

* ``agent`` — the agent self-signs; the change applies immediately (audited with
  the agent as actor).
* ``human`` — parked for the user; only :meth:`SqliteTaskStore.sign_off` can apply
  it. Deviations (rescope / goal) are *held and applied on approval*.

Change types: ``add`` (task_add), ``complete`` (task_complete), ``complete_final``
(the completion that closes the plan — the last open task), ``rescope``
(task_rescope), ``goal`` (goal_revise). Plain progress (task_update / claim) is
never gated.

Because the gate must read plan state (which task, is it the last one) and park
the item itself, it is attached with a handle to the ToolBox — so the model can't
bypass it. It is exposed as a **configurable catalog hook** (``signoff``): the
ToolBox node's hook selector edits a :class:`SignoffConfig` via the standard
config dialog, and :func:`~.hook_catalog.attach_catalog_hooks` wires the real gate
(it has the toolbox). Named ``mode`` presets expand to a policy; "custom" uses the
per-type levels.

Turn-boundary pause: parking flips a task to ``awaiting_signoff`` (or sets the
plan's ``pending_goal``); the Agent node ends the run so control returns to the
user (see ``nodes/agent.py``).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Optional

from .hooks import HOOK_WRAP_TOOL_EXECUTE
from .task_store import DEFAULT_ACTOR, open_task_ids

if TYPE_CHECKING:
    from .tool_box import ToolBox
    from .tools.file_sandbox import FileToolSandbox

#: Approval levels: agent self-signs (applies now) or human approval required.
SIGNOFF_LEVELS = ("agent", "human")

#: Configurable change types. ``complete_final`` is a completion that closes the
#: plan (the last open task), resolved dynamically from ``task_complete``.
CHANGE_TYPES = ("add", "complete", "complete_final", "rescope", "goal")

#: Tool name → change type (``task_complete`` may resolve to ``complete_final``).
_TOOL_TYPE = {
    "task_add": "add",
    "task_complete": "complete",
    "task_rescope": "rescope",
    "goal_revise": "goal",
}

#: Named presets for the config dialog + back-compat. ``requested`` == ``auto``.
SIGNOFF_MODES = ("auto", "requested", "completions", "final", "strict")

_STOP = "Do not proceed — stop here; the user will review and approve or reject."


def _all(level: str) -> dict:
    return {t: level for t in CHANGE_TYPES}


_PRESETS: dict[str, dict] = {
    "auto": _all("agent"),
    "requested": _all("agent"),
    "completions": {**_all("agent"), "complete": "human", "complete_final": "human"},
    "final": {**_all("agent"), "complete_final": "human"},
    "strict": {**_all("agent"), "complete": "human", "complete_final": "human",
               "rescope": "human", "goal": "human"},
}


def preset_policy(mode: str) -> dict:
    """Expand a named preset into a full ``{change_type: level}`` policy."""
    return dict(_PRESETS.get(mode, _PRESETS["auto"]))


def normalize_policy(policy: Optional[dict]) -> dict:
    """A complete, validated policy (unknown types dropped, missing → agent)."""
    out = _all("agent")
    for key, level in (policy or {}).items():
        if key in CHANGE_TYPES and level in SIGNOFF_LEVELS:
            out[key] = level
    return out


def _actor(toolbox: Any) -> str:
    us = getattr(toolbox, "user_session", None) or {}
    return str(us.get("agent_id") or us.get("actor") or us.get("user_id")
               or DEFAULT_ACTOR)


def attach_signoff_gate(
    toolbox: "ToolBox", sandbox: "Optional[FileToolSandbox]" = None, *,
    policy: Optional[dict] = None, mode: str = "auto",
) -> None:
    """Register the sign-off gate. *policy* (a ``{change_type: level}`` map) wins;
    otherwise *mode* expands to a preset. A wholly-``agent`` policy is a no-op.
    Recipe-compatible via ``partial(policy=…)`` / ``partial(mode=…)``."""
    resolved = normalize_policy(policy) if policy else preset_policy(mode)
    if all(level == "agent" for level in resolved.values()):
        return  # nothing gated

    def _parked(target: str, message: str) -> str:
        return json.dumps({
            "error": None, "signoff_required": True, "task_id": target,
            "message": message,
        }, ensure_ascii=False)

    def _park(store: Any, ctype: str, args: dict, actor: str) -> str:
        if ctype in ("complete", "complete_final"):
            task_id = args.get("id")
            summary = (str(args.get("rationale") or "").strip()
                       or f"Task {task_id} is ready for review.")
            store.request_signoff(task_id=task_id, summary=summary, actor=actor)
            return _parked(task_id,
                           f"Task '{task_id}' awaits the user's sign-off. {_STOP}")
        if ctype == "rescope":
            task_id = args.get("id")
            plan = store.load()
            task = next((t for t in (plan.tasks if plan else ()) if t.id == task_id),
                        None)
            new_status = args.get("new_status", "dropped")
            action = {"kind": "rescope", "new_status": new_status,
                      "new_title": args.get("new_title"),
                      "from_title": getattr(task, "title", ""),
                      "from_status": getattr(task, "status", "in_progress")}
            summary = (str(args.get("rationale") or "").strip()
                       or f"Re-scope '{task_id}' to {new_status}.")
            store.request_signoff(task_id=task_id, summary=summary, actor=actor,
                                  action=action)
            return _parked(task_id,
                           f"Re-scoping '{task_id}' needs the user's sign-off. {_STOP}")
        # goal
        summary = str(args.get("rationale") or "").strip() or "Goal revision."
        store.request_goal_signoff(
            new_text=args.get("new_text"),
            acceptance_add=args.get("acceptance_add") or [],
            acceptance_remove=args.get("acceptance_remove") or [],
            summary=summary, actor=actor,
        )
        return _parked("goal", f"Revising the goal needs the user's sign-off. {_STOP}")

    async def gate(
        handler: Callable = None, tool_name: str = "",
        tool_args: Optional[dict] = None, **_kw: Any,
    ) -> Any:
        ctype = _TOOL_TYPE.get(tool_name)
        store = getattr(toolbox, "_task_store", None)
        if ctype is None or store is None:
            return await handler()
        args = tool_args or {}

        # Resolve a plan-closing completion to `complete_final`.
        if ctype == "complete":
            task_id = args.get("id")
            if task_id:
                plan = store.load()
                if plan is not None and open_task_ids(plan) == [task_id]:
                    ctype = "complete_final"

        if resolved.get(ctype, "agent") == "agent":
            return await handler()
        return _park(store, ctype, args, _actor(toolbox))  # human

    toolbox.hooks.register_middleware(HOOK_WRAP_TOOL_EXECUTE, gate)
