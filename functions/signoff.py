# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

User sign-off gate for the task tracker — per-change-type approval policy.

A **policy** maps each change type to who may sign it:

* ``agent`` — the agent self-signs; the change applies immediately (audited with
  the agent as actor).
* ``human`` — the change needs the user's approval before it takes effect.

Change types: ``add`` (task_add), ``complete`` (task_complete), ``complete_final``
(the completion that closes the plan — the last open task), ``rescope``
(task_rescope), ``goal`` (goal_revise). Plain progress (task_update / claim) is
never gated.

Because the gate must read plan state (which task, is it the last one) it is
attached with a handle to the ToolBox — so the model can't bypass it. It is
exposed as a **configurable catalog hook** (``signoff``): the ToolBox node's hook
selector edits a :class:`SignoffConfig` via the standard config dialog, and
:func:`~.hook_catalog.attach_catalog_hooks` wires the real gate (it has the
toolbox). Named ``mode`` presets expand to a policy; "custom" uses the per-type
levels.

**Nothing is parked (spec D31–D33).** The store used to hold the change in an
``awaiting_signoff`` row and apply it when the user signed off later, which
meant a second approval subsystem living beside the tool-approval one, and a
plan whose state depended on a decision that might never arrive. That whole
path — the status, the four columns, the pending goal, the Sign-Off node — is
deleted rather than migrated (D33: early development, ``plan-*.db`` files are
recreated, not upgraded). Audit survives where it already lived, in the
``revision`` and ``deviation`` tables.

Until the inline approval gate lands (D30, Phase 2 item 3) a ``human`` change
type is **refused**, not held: the run is told, in a result the model can act
on, that this change needs the user and cannot be made now. That is D36's
fail-closed rule, and it is the honest interim — the alternative is a change
that appears to have happened and has not.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Optional

from .hooks import HOOK_WRAP_TOOL_EXECUTE
from .task_store import open_task_ids

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

    def _refused(target: str, what: str) -> str:
        """Fail closed: state the refusal in terms the model can act on.

        Not an ``error`` -- the call was well-formed and the policy is
        working as configured -- but ``applied: false`` is unambiguous, and
        naming the change type lets the model report *what* is waiting on
        the user rather than retrying.
        """
        return json.dumps({
            "error": None,
            "applied": False,
            "approval_required": True,
            "change_type": what,
            "target": target,
            "message": (
                f"{what} on '{target}' needs the user's approval and cannot be "
                f"applied by the agent under the current sign-off policy. "
                f"{_STOP}"
            ),
        }, ensure_ascii=False)

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
        target = str(args.get("id") or "goal")
        return _refused(target, ctype)  # human

    # Essential: a Role cannot deactivate the approval gate, and a derived
    # ToolSet carries it (D11, D14, invariant I7). A gate a downstream layer
    # can drop is not a gate.
    toolbox.hooks.register_middleware(
        HOOK_WRAP_TOOL_EXECUTE, gate,
        tools=tuple(_TOOL_TYPE), essential=True,
    )
