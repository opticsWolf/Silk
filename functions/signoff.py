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

This module owns the *vocabulary* -- levels, change types, presets, and the
tool-to-change-type map. The gate that enforces it lives in
:mod:`.approval`, because D31 makes task changes and tool calls two policy
domains of **one** middleware rather than two subsystems. It is exposed as a
configurable catalog hook (``signoff``): the ToolBox node's hook selector edits
a :class:`SignoffConfig` via the standard config dialog, and
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

A ``human`` change type blocks on the run's decision seam and is applied only
if the user says yes; with no way to ask it is **refused**, never held (D36).
"""
from __future__ import annotations

from typing import Optional

#: Approval levels: agent self-signs (applies now) or human approval required.
SIGNOFF_LEVELS = ("agent", "human")

#: Configurable change types. ``complete_final`` is a completion that closes the
#: plan (the last open task), resolved dynamically from ``task_complete``.
CHANGE_TYPES = ("add", "complete", "complete_final", "rescope", "goal")

#: Tool name -> change type (``task_complete`` may resolve to
#: ``complete_final``). The four tools the task domain of the gate watches.
TOOL_CHANGE_TYPE = {
    "task_add": "add",
    "task_complete": "complete",
    "task_rescope": "rescope",
    "goal_revise": "goal",
}

#: Named presets for the config dialog + back-compat. ``requested`` == ``auto``.
SIGNOFF_MODES = ("auto", "requested", "completions", "final", "strict")


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
