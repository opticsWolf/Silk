# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The writer memory was missing (§17, D65).

``recall`` searches a history ledger; the ledger knows how to record runs
and turns; and until now *nothing in a running graph called it*. Memory
that only a test writes to is a search over an empty table -- the tool
answers "nothing remembered matches that" forever, which is the one
failure that looks exactly like working correctly.

This is the hook that fills it. It rides the run lifecycle Silk already
emits -- ``before_run`` for the task, ``after_model_response`` for each
answer, ``after_tool_execute`` for what was touched, ``after_run`` for the
outcome -- so remembering is not a new code path through the loop. It is
selectable in the hook catalog like any other, because whether a run is
remembered is a decision a person makes per graph, not a default that
quietly starts writing to disk.

**Identity comes from the node** (D60). A run id invented here would not
be the one on the events port, and the whole point of the ledger's keys is
that they are the *same* keys observability uses: `agent:<uuid>`,
`session:<id>`, `run:<run_id>`. So the Agent node binds this run's
identity onto the toolset the way it binds the decision seam, and this
hook reads it.

**A failing ledger never fails a run.** Memory is a side effect of doing
the work, not part of it. Every write here is guarded, and a ledger that
raises is dropped for the rest of the run after saying so once -- the same
shape as an embedder that cannot embed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from weave.logger import get_logger

from .hooks import (
    HOOK_AFTER_MODEL_RESPONSE,
    HOOK_AFTER_RUN,
    HOOK_AFTER_TOOL_EXECUTE,
    HOOK_BEFORE_RUN,
)

log = get_logger("SilkRemember")

#: Where a run's identity sits on the toolbox, mirroring the decision
#: seam's attribute. One name, set on both edges of a run.
_IDENTITY_ATTR = "_silk_run_identity"

#: Argument keys that name a file. A turn's ``files`` edges are what make
#: *which runs touched this file* a traversal rather than a log scan
#: (§17), and tools spell the path differently.
_PATH_KEYS = ("path", "file_path", "target", "source", "destination", "file")


@dataclass(frozen=True)
class RunIdentity:
    """Who is running, and as what -- the ledger's keys (D60, §17)."""

    run_id: str = ""
    agent: str = ""
    agent_id: str = ""
    session: str = ""

    def __bool__(self) -> bool:
        return bool(self.run_id)


def bind_run_identity(toolbox: Any, identity: Optional[RunIdentity]) -> None:
    """Name this run on *toolbox* (or ``None`` to unbind at run end).

    Unbinding matters as much as binding: an identity left behind would
    file the next run's turns under the last run's id, and the ledger is
    append-only -- a wrong turn is not editable afterwards, only
    superseded.
    """
    try:
        setattr(toolbox, _IDENTITY_ATTR, identity)
    except AttributeError:      # a toolbox that forbids attributes
        log.debug("could not bind the run identity to %r", toolbox)


def run_identity(toolbox: Any) -> RunIdentity:
    """This run's identity, or an empty one when nothing bound it."""
    value = getattr(toolbox, _IDENTITY_ATTR, None)
    return value if isinstance(value, RunIdentity) else RunIdentity()


def attach_remember_hook(toolbox: Any, ledger: Any, *,
                         min_chars: int = 1) -> None:
    """Record this toolbox's runs and turns into *ledger*.

    Args:
        toolbox: the box whose hooks drive the recording.
        ledger: a :class:`~.ledger.HistoryLedger` (or anything with its
            ``start_run`` / ``record_turn`` / ``finish_run`` shape).
        min_chars: turns shorter than this are not remembered. A run of
            one-word acknowledgements is noise in a search, and noise in
            a search is what makes people stop trusting it.

    Nothing is registered when *ledger* is ``None``: a hook that fires and
    does nothing is harder to notice than one that was never attached, and
    the missing-extra case (no ``macrame-db``) is exactly that.
    """
    if ledger is None:
        log.info("Remember hook not attached: no history ledger available.")
        return

    state: dict[str, Any] = {
        "index": 0, "run": "", "tools": [], "files": [], "alive": True,
    }

    def _guard(what: str, fn: Any) -> None:
        """Do a ledger write, or give up on the ledger for this run."""
        if not state["alive"]:
            return
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - memory never fails a run
            state["alive"] = False
            log.warning(
                f"History write failed while {what}: {exc}. This run will "
                f"not be remembered; the run itself is unaffected."
            )

    def _turn(role: str, text: str) -> None:
        body = str(text or "").strip()
        if len(body) < max(1, int(min_chars)) or not state["run"]:
            return
        tools = tuple(state["tools"])
        files = tuple(state["files"])
        state["tools"] = []
        state["files"] = []
        index = state["index"]
        state["index"] = index + 1
        _guard(
            f"recording a {role} turn",
            lambda: ledger.record_turn(
                state["run"], index=index, role=role, text=body,
                tools=tools, files=files,
            ),
        )

    def run_started(user_input: str = "", **_kw: Any) -> None:
        identity = run_identity(toolbox)
        if not identity:
            # No identity, no keys. Remembering under an invented id would
            # put turns in the ledger that no event stream can be joined
            # to, which is worse than not remembering them (D60).
            log.debug("Remember hook idle: this run has no bound identity.")
            state["run"] = ""
            return
        state.update(index=0, run=identity.run_id, tools=[], files=[],
                     alive=True)
        goal = str(user_input or "").strip()
        _guard("starting a run", lambda: ledger.start_run(
            identity.run_id, agent=identity.agent or "agent",
            session=identity.session, goal=goal[:200],
        ))
        _turn("user", goal)

    def answered(text: str = "", **_kw: Any) -> None:
        _turn("assistant", text)

    def tool_used(tool_name: str = "", tool_args: Optional[dict] = None,
                  **_kw: Any) -> None:
        """Remember what the next turn used, not a turn of its own.

        A tool call is part of the assistant's turn -- that is what makes
        `recall` able to answer *which run touched this file* by walking
        edges instead of scanning a log.
        """
        if tool_name and tool_name not in state["tools"]:
            state["tools"].append(str(tool_name))
        for key in _PATH_KEYS:
            value = (tool_args or {}).get(key)
            if isinstance(value, str) and value.strip():
                if value not in state["files"]:
                    state["files"].append(value)

    def run_finished(final_text: str = "", **_kw: Any) -> None:
        if not state["run"]:
            return
        # The final answer is the last assistant turn only when the loop
        # ended without one being emitted (a stop, an error). Recording it
        # unconditionally would double the last turn in every search.
        run = state["run"]
        summary = str(final_text or "").strip()[:500]
        _guard("finishing a run", lambda: ledger.finish_run(
            run, status="finished", summary=summary,
        ))
        state["run"] = ""

    for event, callback in (
        (HOOK_BEFORE_RUN, run_started),
        (HOOK_AFTER_MODEL_RESPONSE, answered),
        (HOOK_AFTER_TOOL_EXECUTE, tool_used),
        (HOOK_AFTER_RUN, run_finished),
    ):
        toolbox.hooks.register(event, callback)
