# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The approval gate -- one hook, two policy domains (spec D30, D31, D37).

Silk used to have two approval subsystems: a sign-off store that parked
task changes, and a tool-approval stub that was a ``pass`` with a TODO.
The unification happens in the *gate*, not in a store. One
``wrap_tool_execute`` middleware resolves both domains:

* **task changes** -- ``{change_type: level}`` over add / complete /
  complete_final / rescope / goal (see :mod:`.signoff`);
* **tool calls** -- ``{tool_or_risk: level}`` over tool names and the
  ``risk`` metadata every tool already declares at registration.

The four plan tools are reachable from both policies, and where both have
something to say the **stricter** answer wins: naming a tool in one domain
can never be undone by the other's default.

A gated call blocks on the run's :class:`~.decision_seam.DecisionSeam`,
which emits the request into the run's own stream and waits for the human
to answer there. Approve and the held call executes *in the same run*;
deny and a structured refusal becomes that tool's result. Nothing is
parked, nothing is resumed.

**The gate is a monotonic guard (invariant I10).** ``emit_middleware``
runs handlers in registration order with the first registered outermost,
so a middleware registered ahead of the gate could answer a call the gate
never sees -- harmless for the shipped hooks, which only deny, but not
harmless as a rule. :func:`attach_approval_gate` therefore forces itself
to position zero, and re-forces it if something registers later.

**Order of consultation**, cheapest and least surprising first: a
run-scoped grant, then a durable grant, then the policy, then the human.
Grants can only *skip* the question, never create one, so consulting them
first cannot make the gate stricter than the policy says.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Optional

from weave.logger import get_logger

from .decision_seam import (
    KIND_APPROVAL,
    Decision,
    DecisionRequest,
    DecisionSeam,
    new_decision_id,
)
from .grants import SCOPE_ALWAYS, SCOPE_RUN, GrantStore, RunGrants
from .hooks import HOOK_WRAP_TOOL_EXECUTE
from .signoff import (
    CHANGE_TYPES,
    TOOL_CHANGE_TYPE,
    normalize_policy,
    preset_policy,
)
from .task_store import open_task_ids

if TYPE_CHECKING:
    from .tool_box import ToolBox
    from .tools.file_sandbox import FileToolSandbox

log = get_logger("SilkApproval")

#: Approval levels, shared with the task domain: the agent self-signs, or a
#: human must say yes.
LEVEL_AGENT = "agent"
LEVEL_HUMAN = "human"

#: Risk keys a tool policy may name, alongside literal tool names. A tool
#: name always wins over its risk band -- the specific beats the general.
RISK_KEYS = ("low", "medium", "high")

#: Ready-made tool policies for the node UI.
TOOL_PRESETS: dict[str, dict] = {
    "off": {},
    "high_risk": {"high": LEVEL_HUMAN},
    "writes": {"high": LEVEL_HUMAN, "medium": LEVEL_HUMAN},
    "everything": {k: LEVEL_HUMAN for k in RISK_KEYS},
}


def normalize_tool_policy(policy: Optional[dict]) -> dict:
    """A validated ``{tool_or_risk: level}`` map; junk dropped.

    Unlike the task policy this is *not* filled in with defaults: an
    unnamed tool is ungated, and a policy of ``{}`` gates nothing. The
    asymmetry is deliberate -- the task domain has five known change types,
    while the tool domain is open-ended and a default of "gate everything"
    would make every new tool a prompt.
    """
    out: dict[str, str] = {}
    for key, level in (policy or {}).items():
        if level in (LEVEL_AGENT, LEVEL_HUMAN) and str(key):
            out[str(key)] = level
    return out


def tool_preset_policy(name: str) -> dict:
    """Expand a named tool-policy preset."""
    return dict(TOOL_PRESETS.get(name, TOOL_PRESETS["off"]))


def _actor(toolbox: Any) -> str:
    session = getattr(toolbox, "user_session", None) or {}
    return str(session.get("agent_id") or session.get("actor")
               or session.get("user_id") or "agent")


def _refusal(*, target: str, change_type: str, text: str) -> str:
    """The one refusal shape, whichever domain and whichever failure (D36).

    ``error`` stays null: the call was well-formed and the gate is working
    as configured, so reflection must not treat it as something to retry.
    ``applied: false`` is what is unambiguous.
    """
    return json.dumps({
        "error": None,
        "applied": False,
        "approval_required": True,
        "change_type": change_type,
        "target": target,
        "message": text,
    }, ensure_ascii=False)


def attach_approval_gate(
    toolbox: "ToolBox",
    sandbox: "Optional[FileToolSandbox]" = None,
    *,
    task_policy: Optional[dict] = None,
    tool_policy: Optional[dict] = None,
    seam: Optional[DecisionSeam] = None,
    grants: Optional[GrantStore] = None,
    run_grants: Optional[RunGrants] = None,
    project_root: Optional[str] = None,
) -> Optional[Any]:
    """Register the unified approval gate; returns its :class:`HookEntry`.

    Every argument is captured **here, once**, and never re-read mid-run:
    editing a Role or a hook config while a run is in flight affects the
    next run, not the one already going (D38's policy-snapshot rule). The
    seam is likewise run-scoped -- the node makes one per run and hands it
    in.

    Returns ``None`` when nothing is gated, so the caller can tell "no gate
    was needed" from "a gate is installed".
    """
    tasks = normalize_policy(task_policy) if task_policy else {}
    tools = normalize_tool_policy(tool_policy)
    gated_tasks = {t for t in CHANGE_TYPES if tasks.get(t) == LEVEL_HUMAN}
    if not gated_tasks and not tools:
        return None

    run_scoped = run_grants if run_grants is not None else RunGrants()
    durable = grants
    root = project_root or str(getattr(sandbox, "root_dir", "") or "")
    actor = _actor(toolbox)

    # Which tools the gate must see. The task domain is the four plan
    # tools; the tool domain may name tools directly, but a *risk* key
    # applies to tools that are not known yet (a capability loaded
    # mid-run), so a risk-based policy has to stay unbound.
    named = {name for name in tools if name not in RISK_KEYS}
    by_risk = any(name in RISK_KEYS for name in tools)
    bound: tuple[str, ...] | None = (
        None if by_risk
        else tuple({*named, *(TOOL_CHANGE_TYPE if gated_tasks else ())})
    )

    def _task_change(tool_name: str, args: dict) -> Optional[str]:
        """The change type this call makes, or None if it makes none."""
        ctype = TOOL_CHANGE_TYPE.get(tool_name)
        if ctype is None:
            return None
        if ctype != "complete":
            return ctype
        # A completion that closes the plan is its own change type, and
        # resolving it needs the store, which is why the gate holds a
        # toolbox handle rather than being a free function.
        store = getattr(toolbox, "_task_store", None)
        task_id = args.get("id")
        if store is None or not task_id:
            return ctype
        plan = store.load()
        if plan is not None and open_task_ids(plan) == [task_id]:
            return "complete_final"
        return ctype

    def _level_for_tool(tool_name: str) -> str:
        """Policy level for a tool call: the name beats the risk band."""
        if tool_name in tools:
            return tools[tool_name]
        risk = str((toolbox.tools.get(tool_name) or {}).get("risk", "low"))
        return tools.get(risk, LEVEL_AGENT)

    def _prompt(tool_name: str, ctype: Optional[str], args: dict) -> str:
        if ctype is not None:
            target = args.get("id") or "the goal"
            return f"Allow the agent to {ctype.replace('_', ' ')} '{target}'?"
        return f"Allow the agent to call {tool_name}?"

    def _remember(decision: Decision, tool_name: str) -> None:
        """Turn "and don't ask again" into the grant it asked for (D10)."""
        if decision.remember == SCOPE_RUN:
            run_scoped.allow(tool_name)
        elif decision.remember == SCOPE_ALWAYS and durable is not None:
            durable.grant(root, tool_name, granted_by=decision.actor or actor,
                          note="granted from an approval prompt")

    async def gate(
        handler: Callable = None, tool_name: str = "",
        tool_args: Optional[dict] = None, **_kw: Any,
    ) -> Any:
        args = dict(tool_args or {})
        ctype = _task_change(tool_name, args) if gated_tasks else None

        # Two domains, one call: the stricter answer wins. A tool that is
        # also a task change (the four plan tools) is reachable from both
        # policies, and a policy that names it must not be silently
        # outranked by the domain it happens to be evaluated in first.
        levels = {_level_for_tool(tool_name) if tools else LEVEL_AGENT}
        if ctype is not None:
            levels.add(tasks.get(ctype, LEVEL_AGENT))
        level = LEVEL_HUMAN if LEVEL_HUMAN in levels else LEVEL_AGENT
        if level != LEVEL_HUMAN:
            return await handler()

        # Grants skip the question; they never create one.
        if run_scoped.allows(tool_name):
            return await handler()
        if durable is not None and durable.allows(root, tool_name):
            return await handler()

        target = str(args.get("id") or (tool_name if ctype is None else "goal"))
        if seam is None:
            # No seam at all is D36's first failure by another route: the
            # gate was configured but the run has nothing to ask with.
            return _refusal(
                target=target, change_type=ctype or tool_name,
                text=("This call needs the user's approval and this run has no "
                      "way to ask. A durable grant is how to allow it without "
                      "a human present."),
            )

        decision = seam.await_decision(DecisionRequest(
            decision_id=new_decision_id(),
            kind=KIND_APPROVAL,
            prompt=_prompt(tool_name, ctype, args),
            tool_name=tool_name,
            tool_args=args,
            detail={
                "risk": str((toolbox.tools.get(tool_name) or {}).get("risk",
                                                                    "low")),
                "change_type": ctype or "",
                "project": root,
            },
        ))

        if decision.approved:
            _remember(decision, tool_name)
            return await handler()

        # I10's corollary: a denial never fabricates success. The refusal
        # goes back as the tool's result and the model adapts.
        return _refusal(target=target, change_type=ctype or tool_name,
                        text=decision.refusal_text())

    entry = toolbox.hooks.register_middleware(
        HOOK_WRAP_TOOL_EXECUTE, gate,
        tools=bound if bound is not None else (),
        essential=True,   # D11/I7: a Role cannot drop the gate
    )
    # D37: forced outermost, not merely registered early. A middleware
    # may return without calling handler(), so anything ahead of the gate
    # could answer a call the gate never sees.
    toolbox.hooks.make_outermost(HOOK_WRAP_TOOL_EXECUTE, entry)
    return entry


def attach_signoff_gate(
    toolbox: "ToolBox", sandbox: "Optional[FileToolSandbox]" = None, *,
    policy: Optional[dict] = None, mode: str = "auto",
    seam: Optional[DecisionSeam] = None,
    grants: Optional[GrantStore] = None,
    run_grants: Optional[RunGrants] = None,
) -> Optional[Any]:
    """The task domain alone, by policy or by named preset.

    Kept as its own entry point because that is what the ``signoff``
    catalog hook selects and what the recipe replays through
    ``partial(policy=...)``. It is the same gate underneath -- D31 -- so a
    box that gates both domains still has exactly one middleware.
    """
    resolved = normalize_policy(policy) if policy else preset_policy(mode)
    return attach_approval_gate(
        toolbox, sandbox, task_policy=resolved, seam=seam, grants=grants,
        run_grants=run_grants,
    )
