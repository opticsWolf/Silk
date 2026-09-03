# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The floor under ``requires_approval=True`` (spec §7, D81; closes G1).

A tool is registered with ``requires_approval=True`` by whoever wrote it:
``remove_node`` and ``disconnect`` carry it (D73), and any capability may
set it on a tool of its own. Until this module existed the flag was
stored, shown in the catalogue -- and then *ignored*: the only thing that
actually asked was the policy gate in `approval.py`, which is attached
only when a Role or hook config configures one and, when it is, sees only
the tools that policy named. So a registration that said "a human must
say yes" executed unchecked in every run whose policy happened not to
mention it.

This module is the second half of the design D77 already used for the
load verbs: a gate that exists *because the tool does*, not because a
policy asked for one.

- It asks whenever the flag is set, whatever the tool policy says.
- A grant may still pre-authorise it. That is the difference from the
  load floor: `requires_approval` is a tool author saying "check with a
  human", not the process-authority boundary that no grant may cross.
- Fail-closed (D36): no seam, no call. And a flagged tool whose toolbox
  has no floor at all is refused at the execution site rather than run --
  see `ToolBox._safe_execute`, which is where G1's TODO used to sit.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from weave.logger import get_logger

from .approval import (
    _HEADLESS_ATTR,
    _SEAM_ATTR,
    _actor,
    _refusal,
    headless_refusals,
)
from .decision_seam import (
    KIND_APPROVAL,
    DecisionRequest,
    DecisionSeam,
    new_decision_id,
)
from .grants import SCOPE_ALWAYS, SCOPE_RUN, GrantStore, RunGrants
from .hooks import HOOK_WRAP_TOOL_EXECUTE
from .self_modify import ALWAYS_APPROVE

if TYPE_CHECKING:
    from .tool_box import ToolBox

log = get_logger("SilkApprovalFloor")

#: Set on a ToolBox once the floor is installed. `_safe_execute` reads it
#: to decide whether a flagged tool has anything gating it at all.
APPROVAL_FLOOR_ATTR = "_silk_approval_floor"

#: Where the floor keeps its seam, grants and project root. Held apart from
#: the entry so a host can hand the floor a grant store *after* the first
#: flagged registration already installed it.
_CONFIG_ATTR = "_silk_approval_floor_config"

#: The stock refusal for a flagged tool with no floor. Kept here so the
#: execution site and the tests say the same thing.
NO_FLOOR_TEXT = (
    "This tool is registered as needing the user's approval and this "
    "toolbox has no approval floor installed, so there is nothing that "
    "could ask. Nothing was run."
)

__all__ = [
    "APPROVAL_FLOOR_ATTR",
    "NO_FLOOR_TEXT",
    "attach_approval_floor",
    "ensure_approval_floor",
    "flagged",
]


def flagged(toolbox: Any, tool_name: str) -> bool:
    """Whether *tool_name* was registered with ``requires_approval=True``."""
    # One lookup covers both shapes: `register_tool` stores the flag in
    # its meta, and a capability writes it on the tool definition it
    # hands over (`capabilities.py`), which `ToolBox` stores as the meta.
    return bool((toolbox.tools.get(tool_name) or {}).get("requires_approval"))


def attach_approval_floor(
    toolbox: "ToolBox",
    *,
    seam: Optional[DecisionSeam] = None,
    grants: Optional[GrantStore] = None,
    run_grants: Optional[RunGrants] = None,
    project_root: str = "",
) -> Any:
    """Install the floor; returns the `HookEntry`.

    Unbound on purpose -- the middleware filter is fixed at registration
    and a flagged tool may arrive later (a capability loaded mid-run, an
    MCP toolset mounted after the first turn), so the floor sees every
    call and reads the flag at call time.

    Calling this twice **reconfigures** the one floor rather than adding a
    second: the first install usually happens at registration time, before
    the host has a grant store to hand it, and the entry is essential
    (I7), so it could not be taken out again even if stacking were wanted.
    Two floors would also mean two dialogs for one call.
    """
    config: dict[str, Any] = getattr(toolbox, _CONFIG_ATTR, None) or {}
    config.update({
        "seam": seam if seam is not None else config.get("seam"),
        "grants": grants if grants is not None else config.get("grants"),
        "run_grants": (run_grants if run_grants is not None
                       else config.get("run_grants") or RunGrants()),
        "root": project_root or config.get("root") or "",
    })
    setattr(toolbox, _CONFIG_ATTR, config)
    actor = _actor(toolbox)

    existing = getattr(toolbox, APPROVAL_FLOOR_ATTR, None)
    if existing is not None:
        return existing

    def _remember(decision: Any, tool_name: str) -> None:
        store = config.get("grants")
        if decision.remember == SCOPE_RUN:
            config["run_grants"].allow(tool_name)
        elif decision.remember == SCOPE_ALWAYS and store is not None:
            store.grant(config["root"], tool_name,
                        granted_by=decision.actor or actor,
                        note="granted from a requires_approval prompt")

    async def floor(handler: Optional[Callable] = None, tool_name: str = "",
                    tool_args: Optional[dict] = None, **_kw: Any) -> Any:
        if handler is None:  # middleware is always handed its next layer
            raise TypeError(
                "the approval floor is middleware: it wraps a handler, "
                "so one must be given"
            )
        if tool_name in ALWAYS_APPROVE or not flagged(toolbox, tool_name):
            return await handler()

        args = dict(tool_args or {})
        store = config.get("grants")
        root = config["root"]
        if config["run_grants"].allows(tool_name):
            return await handler()
        if store is not None and store.allows(root, tool_name):
            return await handler()

        bound = config.get("seam")
        asker = bound if bound is not None else getattr(toolbox, _SEAM_ATTR,
                                                        None)
        if asker is None:
            # Same shape as the policy gate's headless path: refuse, and
            # say so once rather than once per call (D53's legibility rule).
            seen = headless_refusals(toolbox) + 1
            try:
                setattr(toolbox, _HEADLESS_ATTR, seen)
            except AttributeError:
                pass
            if seen == 1:
                log.warning(
                    f"'{tool_name}' is registered as requiring approval and "
                    f"this run has no way to ask: it will be refused. A "
                    f"durable grant is how to allow it without a human."
                )
            return _refusal(
                target=str(args.get("id") or tool_name), change_type=tool_name,
                text=("This tool is registered as needing the user's approval "
                      "and this run has no way to ask. A durable grant is how "
                      "to allow it without a human present."),
            )

        decision = asker.await_decision(DecisionRequest(
            decision_id=new_decision_id(),
            kind=KIND_APPROVAL,
            prompt=f"Allow the agent to call {tool_name}?",
            tool_name=tool_name,
            tool_args=args,
            detail={
                "risk": str((toolbox.tools.get(tool_name) or {}).get("risk",
                                                                    "low")),
                "requires_approval": True,
                "project": project_root,
            },
        ))
        if decision.approved:
            _remember(decision, tool_name)
            return await handler()
        return _refusal(target=str(args.get("id") or tool_name),
                        change_type=tool_name, text=decision.refusal_text())

    entry = toolbox.hooks.register_middleware(
        HOOK_WRAP_TOOL_EXECUTE, floor,
        tools=(),          # every call: the flag is read at call time
        essential=True,    # D11/I7: a Role cannot drop it
    )
    # D37, for the same reason the policy gate and the load floor are
    # forced outermost: middleware ahead of this one may answer a call
    # without ever calling handler(), and that answer stands in for the
    # tool -- including for a tool the human was supposed to be asked about.
    toolbox.hooks.make_outermost(HOOK_WRAP_TOOL_EXECUTE, entry)
    setattr(toolbox, APPROVAL_FLOOR_ATTR, entry)
    return entry


def ensure_approval_floor(toolbox: Any) -> Any:
    """Install the floor once, on the first flagged registration.

    Called from `ToolBox.register_tool`, so declaring `requires_approval`
    is all a tool author has to do -- the flag arrives with its gate. A
    host that wants grants or a fixed seam attaches the floor itself
    beforehand; this only fills the gap when nobody did.
    """
    entry = getattr(toolbox, APPROVAL_FLOOR_ATTR, None)
    if entry is not None:
        return entry
    try:
        return attach_approval_floor(toolbox)
    except Exception as exc:      # a toolbox mid-construction, a test double
        log.debug(f"could not install the approval floor: {exc}")
        return None
