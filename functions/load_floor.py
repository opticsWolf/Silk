# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The always-approve floor for the load verbs (spec §19, D77).

`import` is an execution boundary the sandbox does not cross. Every other
gated tool in Silk is gated *by policy* -- a Role, a risk band, a hook
config, and a grant can pre-authorise it. The load verbs are the one
place where that is not enough, so this module installs a second,
unconditional gate:

- It asks **every time**, whatever the tool policy says, and whatever
  grants exist. A "don't ask again" for `load_suite` would be a durable
  grant of process authority, so `remember` is not offered and would not
  be honoured.
- The request carries **the code** -- the file listing with sizes and
  mtimes, and the diff of every file this run touched. A human approving
  "load `my_nodes`" has approved nothing.
- Fail-closed like everything else (D36): no seam, no load.

Kept apart from `approval.py` because that module implements a *policy*
gate -- it exists when a policy asks for it and it is configurable. This
one exists whenever the tools do, and is not.
"""
from __future__ import annotations

from typing import Any, Optional

from weave.logger import get_logger

from .approval import _SEAM_ATTR, _refusal
from .decision_seam import KIND_APPROVAL, DecisionRequest, new_decision_id
from .hooks import HOOK_WRAP_TOOL_EXECUTE
from .self_modify import (
    ALWAYS_APPROVE, OP_RELOAD_SUITE, approval_prompt, check_suite, find,
    lint_suite, load_request_detail, run_changes,
)

log = get_logger("SilkLoadFloor")

__all__ = ["attach_load_floor", "ALWAYS_APPROVE"]


def attach_load_floor(toolbox: Any, *, seam: Any = None,
                      lint: bool = True) -> Any:
    """Gate the load verbs unconditionally. Returns the `HookEntry`.

    *lint* exists for the tests and for a host that runs the version
    check itself; leaving it on is the D78 behaviour, where a WV521
    finding stops the load before the human is even asked -- there is no
    point asking a person to approve something the linter already knows
    breaks saved graphs.
    """

    async def floor(handler=None, tool_name: str = "",
                    tool_args: Optional[dict] = None, **_kw: Any) -> Any:
        args = dict(tool_args or {})
        name = str(args.get("name") or "").strip()
        is_reload = tool_name == OP_RELOAD_SUITE

        refusal = check_suite(name, op=tool_name)
        if refusal is not None:
            return _refusal(target=name or tool_name, change_type=tool_name,
                            text=refusal.reason)

        info = find(name) or {}
        path = info.get("path", "")

        outcome = lint_suite(path) if lint else None
        if outcome is not None and not outcome.ok:
            # D78: the linter is the code review a machine author gets,
            # so its verdict lands *before* the human is asked rather
            # than as a footnote in a dialog they are already reading.
            log.warning(f"refusing to load '{name}': {outcome.note}")
            return _refusal(target=name, change_type=tool_name,
                            text=outcome.refusal_text())

        asker = getattr(toolbox, _SEAM_ATTR, None)
        if asker is None:
            return _refusal(
                target=name, change_type=tool_name,
                text=("Loading code always needs the user's approval and "
                      "this run has no way to ask. No grant, preset or Role "
                      "can pre-authorise it -- importing runs the code with "
                      "the full authority of this process."),
            )

        changes = run_changes(toolbox)
        detail = load_request_detail(name, path, changes=changes,
                                     lint=outcome, reload=is_reload)
        decision = asker.await_decision(DecisionRequest(
            decision_id=new_decision_id(),
            kind=KIND_APPROVAL,
            prompt=approval_prompt(name, detail["changed_files"],
                                   reload=is_reload),
            tool_name=tool_name,
            tool_args=args,
            detail=detail,
        ))

        if not decision.approved:
            # Denial is cheap and normal (D77): the files stay on disk and
            # the agent can still report what it built.
            return _refusal(target=name, change_type=tool_name,
                            text=decision.refusal_text())
        return await handler()

    entry = toolbox.hooks.register_middleware(
        HOOK_WRAP_TOOL_EXECUTE, floor, tools=ALWAYS_APPROVE,
        essential=True,     # D11/I7: a Role cannot drop it
    )
    # Outermost for the same reason the policy gate is: a middleware may
    # answer without calling handler(), and anything ahead of this one
    # could load code the human never saw.
    toolbox.hooks.make_outermost(HOOK_WRAP_TOOL_EXECUTE, entry)
    return entry
