# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The missing verb: loading agent-authored code (spec §19, D75-D81).

Write is the file tools, verify is the toolchain runner, observe is
`describe_graph` -- the loop already existed except for **load**. These
four tools are that verb, registered as ordinary `ToolBox` tools so they
inherit hooks, `tool_events`, role enforcement and the approval gate for
free (the D56 property). There is no self-improvement subsystem here; the
composition *is* the feature -- an agent that writes a node can then load
it and place it (§18) without a single new mechanism in between.

The order a load goes through, and why:

1. **Is it agent-authored?** Only suites under the user plugin dir (D76).
2. **The version check** -- `weave_lint`, in the floor, before anyone is
   asked (D78).
3. **The human** -- always, with the diff (D77); the floor in
   `functions/load_floor.py` does this and no policy can switch it off.
4. **Validation in a subprocess** -- because a dry-run import *executes*
   the candidate's top-level code, and for machine-authored code a
   validation step that can segfault the session is not validation.
5. **The load itself, on the main thread** -- across the D70 seam, since
   registering classes rehydrates placeholders on the canvas.

`request_relaunch` is the fifth wheel of the same story: some changes no
reload can absorb. It queues and returns; a human confirms at a turn
boundary (D79, I12).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

# Absolute import: this module may be exec'd by the ToolLoader without a
# parent package, where a ``..`` relative import is "beyond top-level".
from weave.plugins.silk.functions.graph_author import canvas_binding
from weave.plugins.silk.functions.self_modify import (
    OP_LIST_SUITES, OP_LOAD_SUITE, OP_RELAUNCH, OP_RELOAD_SUITE, annotate,
    capability_for, check_suite, clear_quarantine, find, suites,
    user_plugin_root,
)

if TYPE_CHECKING:
    from ..tool_box import ToolBox
    from .file_sandbox import FileToolSandbox


# ── schemas ──────────────────────────────────────────────────────────────

class NoArgs(BaseModel):
    pass


class SuiteArgs(BaseModel):
    name: str = Field(..., min_length=1,
                      description="Suite name, as list_suites reports it.")


class RelaunchArgs(BaseModel):
    reason: str = Field(..., min_length=1,
                        description="Why Weave needs restarting. The user "
                                    "reads this before deciding.")


class SuiteResult(BaseModel):
    ok: bool = Field(..., description="Whether the operation happened.")
    op: str = Field("", description="Which operation this was.")
    result: dict = Field(default_factory=dict,
                         description="Operation-specific payload.")
    message: str = Field("", description="Why it was refused, if it was.")


def _refused(op: str, text: str, subject: str = "") -> SuiteResult:
    return SuiteResult(ok=False, op=op, message=text,
                       result={"subject": subject})


def attach_suite_tools(toolbox: "ToolBox", sandbox: "FileToolSandbox",
                       *, validate: bool = True) -> None:
    """Mount the load verbs and install the always-approve floor.

    The floor is installed *here*, with the tools, and not by a policy:
    D77's floor exists whenever the verbs do. Registering the tools
    without it would be registering a way to run arbitrary code without a
    human, which is the one thing this section is about not doing.
    """
    from weave.plugins.silk.functions.load_floor import attach_load_floor

    attach_load_floor(toolbox)

    def _seam() -> Any:
        """The main-thread seam this run was given, or None."""
        binding = canvas_binding(toolbox)
        return None if binding is None else binding.seam

    def _no_main_thread(op: str) -> SuiteResult:
        return _refused(op, (
            "This run has no way to reach the main thread, and loading code "
            "touches the registry and the canvas. Nothing was loaded."))

    # ── read ─────────────────────────────────────────────────────────────

    @toolbox.register(
        name=OP_LIST_SUITES,
        tags=("plugins", "read"), category="plugins", risk="low",
        description=(
            "List the node suites Weave can see: which are loaded, which "
            "you may load, and which were quarantined after failing."
        ),
        args_model=NoArgs,
        procedure=(
            "Look before you load.\n"
            "- writable=true means the suite is under your plugin root and "
            "you may load it; shipped suites are not yours to replace.\n"
            "- quarantined=true means a previous start refused it after a "
            "crash; the note is the traceback. Fix that before reloading."
        ),
    )
    def _list_suites(db_pool: Any, user_session: dict) -> SuiteResult:
        rows = annotate(suites())
        return SuiteResult(ok=True, op=OP_LIST_SUITES, result={
            "suites": rows,
            "plugin_root": str(user_plugin_root()),
            "writable": [r["name"] for r in rows if r.get("writable")],
        })

    # ── the load verbs (always approved -- see functions/load_floor.py) ──

    def _load(op: str, name: str) -> SuiteResult:
        refusal = check_suite(name, op=op)
        if refusal is not None:
            return _refused(op, refusal.reason, name)

        info = find(name) or {}
        if validate:
            # D77: validation is execution too. The dry run happens in a
            # fresh interpreter, so a segfault or a hang in generated code
            # costs one tool call instead of the session -- and the
            # traceback that comes back is the feedback the generator
            # needs anyway.
            from weave.engine.validation import validate_suite

            report = validate_suite(name)
            if not report.ok:
                return _refused(op, (
                    f"'{name}' does not import cleanly, so it was not "
                    f"loaded. The session is untouched. "
                    f"{report.note or ''}\n"
                    + "\n".join(f"{mod}: {text}" for mod, text
                                in sorted(report.tracebacks.items()))
                ).strip(), name)

        seam = _seam()
        if seam is None:
            return _no_main_thread(op)

        answer = seam.call(op, name=name,
                           capability=capability_for(name))
        if not answer.ok:
            return _refused(op, answer.failure_text(), name)

        value = dict(answer.value or {})
        if value.get("ok"):
            # It loads again, so the old failure is no longer a fact
            # about it (D81).
            clear_quarantine(name)
        value.setdefault("path", info.get("path", ""))
        return SuiteResult(ok=bool(value.get("ok", True)), op=op,
                           result=value,
                           message="" if value.get("ok", True)
                           else str(value.get("note", "")))

    @toolbox.register(
        name=OP_LOAD_SUITE,
        tags=("plugins", "write"), category="plugins", risk="high",
        requires_approval=True,
        description=(
            "Import a suite you wrote under the plugin root and register "
            "its node classes in the running session."
        ),
        args_model=SuiteArgs,
        procedure=(
            "Load code you wrote. This always asks the user, every time, "
            "and shows them your diff -- importing runs the code with the "
            "full authority of this process.\n"
            "- The suite must be a package directory under the plugin root "
            "with an __init__.py.\n"
            "- Lint it first: a WV520/WV521/WV522 finding stops the load, "
            "because it means saved graphs would not survive it.\n"
            "- If it is already loaded, use reload_suite instead."
        ),
    )
    def _load_suite(db_pool: Any, user_session: dict,
                    name: str) -> SuiteResult:
        return _load(OP_LOAD_SUITE, name)

    @toolbox.register(
        name=OP_RELOAD_SUITE,
        tags=("plugins", "write"), category="plugins", risk="high",
        requires_approval=True,
        description=(
            "Re-import a suite you have edited and swap every node on the "
            "canvas onto the new classes, keeping values and connections."
        ),
        args_model=SuiteArgs,
        procedure=(
            "Reload after editing. Same approval as load_suite, every "
            "time.\n"
            "- Failure is not partial: if the new code raises on import "
            "the session keeps running the old code.\n"
            "- The swap is one undo step, so the user can put the session "
            "back.\n"
            "- Changing a node class's state shape needs node_state_api "
            "bumped and migrate_state written, or the reload is refused."
        ),
    )
    def _reload_suite(db_pool: Any, user_session: dict,
                      name: str) -> SuiteResult:
        return _load(OP_RELOAD_SUITE, name)

    # ── the restart request (D79) ────────────────────────────────────────

    @toolbox.register(
        name=OP_RELAUNCH,
        tags=("plugins", "write"), category="plugins", risk="high",
        description=(
            "Ask for Weave to be restarted, for a change no reload can "
            "absorb. Queues the request and returns immediately."
        ),
        args_model=RelaunchArgs,
        procedure=(
            "Ask, do not expect. The run finishes normally and the user is "
            "asked at a turn boundary; you never see the other side of a "
            "restart.\n"
            "- Anything that must survive it has to be in the plan or the "
            "history ledger BEFORE you ask.\n"
            "- 'Continue after the restart' is a task with a claim, not a "
            "promise the runtime keeps."
        ),
    )
    def _request_relaunch(db_pool: Any, user_session: dict,
                          reason: str) -> SuiteResult:
        seam = _seam()
        if seam is None:
            return _no_main_thread(OP_RELAUNCH)
        answer = seam.call(OP_RELAUNCH, reason=reason)
        if not answer.ok:
            return _refused(OP_RELAUNCH, answer.failure_text())
        return SuiteResult(ok=True, op=OP_RELAUNCH,
                           result=dict(answer.value or {}),
                           message=("Queued. The user is asked once this "
                                    "run reaches a turn boundary."))


def suite_tool_names() -> tuple:
    """The four names, for a Role selector or a test."""
    return (OP_LIST_SUITES, OP_LOAD_SUITE, OP_RELOAD_SUITE, OP_RELAUNCH)
