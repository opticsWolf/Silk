# -*- coding: utf-8 -*-
"""The writer memory was missing (§17, D65).

``recall`` searches a history ledger and the ledger knows how to record a
run, but until this hook existed nothing in a running graph called it: the
tool answered "nothing remembered matches that" forever, which is the one
failure that looks exactly like working correctly. These tests are about
the join between a run and its memory -- that turns arrive, under the id
the events port uses, and that a ledger having a bad day cannot take the
run down with it.
"""
from __future__ import annotations

import pytest

from silk.functions import ledger as ledger_mod
from silk.functions.hooks import (
    HOOK_AFTER_MODEL_RESPONSE,
    HOOK_AFTER_RUN,
    HOOK_AFTER_TOOL_EXECUTE,
    HOOK_BEFORE_RUN,
)
from silk.functions.remember import (
    RunIdentity, attach_remember_hook, bind_run_identity, run_identity,
)
from silk.functions.tool_box import ToolBox


class _Recorder:
    """A ledger with the three methods the hook uses, and a memory of calls."""

    def __init__(self, fail_on: str = "") -> None:
        self.runs: list[dict] = []
        self.turns: list[dict] = []
        self.finished: list[dict] = []
        self.fail_on = fail_on

    def _maybe_fail(self, what: str) -> None:
        if self.fail_on == what:
            raise RuntimeError(f"ledger is unhappy about {what}")

    def start_run(self, run_id, *, agent="", session="", goal="", **kw):
        self._maybe_fail("start")
        self.runs.append(
            {"run_id": run_id, "agent": agent, "session": session, "goal": goal}
        )

    def record_turn(self, run_id, *, index, role, text, tools=(), files=(),
                    **kw):
        self._maybe_fail("turn")
        self.turns.append({
            "run_id": run_id, "index": index, "role": role, "text": text,
            "tools": tuple(tools), "files": tuple(files),
        })

    def finish_run(self, run_id, *, status="finished", summary="", **kw):
        self._maybe_fail("finish")
        self.finished.append(
            {"run_id": run_id, "status": status, "summary": summary}
        )


def _box(ledger, *, identity=None, **kw):
    if identity is None:
        identity = RunIdentity(run_id="run-1", agent="scribe",
                               session="s1")
    box = ToolBox(db_pool=None, user_session={"agent_id": "a"})
    bind_run_identity(box, identity)
    attach_remember_hook(box, ledger, **kw)
    return box


def _a_run(box, *, task="please summarise the parser changes",
           answers=("the parser now keeps a quote stack",)):
    box.hooks.emit(HOOK_BEFORE_RUN, user_input=task, settings={})
    for text in answers:
        box.hooks.emit(HOOK_AFTER_MODEL_RESPONSE, text=text, round_index=0)
    box.hooks.emit(HOOK_AFTER_RUN, final_text=answers[-1] if answers else "",
                   rounds=len(answers), elapsed_s=0.1)


def test_a_run_is_remembered_turn_by_turn():
    led = _Recorder()
    _a_run(_box(led))

    assert led.runs == [{
        "run_id": "run-1", "agent": "scribe", "session": "s1",
        "goal": "please summarise the parser changes",
    }]
    assert [(t["index"], t["role"]) for t in led.turns] == [
        (0, "user"), (1, "assistant"),
    ], "the task and the answer, in order, under one run"
    assert led.finished[0]["run_id"] == "run-1"


def test_memory_is_filed_under_the_run_the_events_port_uses():
    """A remembered turn nothing can be joined to is a turn nobody finds."""
    led = _Recorder()
    box = _box(led, identity=RunIdentity(run_id="abc-123", agent="Coordinator",
                                         agent_id="uuid-9", session="sess"))
    _a_run(box)

    assert {t["run_id"] for t in led.turns} == {"abc-123"}
    assert led.runs[0]["agent"] == "Coordinator"
    assert run_identity(box).agent_id == "uuid-9"


def test_a_run_with_no_identity_is_not_remembered():
    """Better an unremembered run than turns under an invented id (D60)."""
    led = _Recorder()
    box = ToolBox(db_pool=None, user_session={"agent_id": "a"})
    attach_remember_hook(box, led)
    _a_run(box)

    assert led.runs == [] and led.turns == []


def test_an_unbound_identity_does_not_leak_into_the_next_run():
    led = _Recorder()
    box = _box(led)
    _a_run(box)
    bind_run_identity(box, None)
    _a_run(box, task="a second, unnamed run")

    assert len(led.runs) == 1, "the second run had no identity to file under"
    assert not run_identity(box)


def test_a_turn_carries_what_it_touched():
    """The edges that make 'which run touched this file' a traversal (§17)."""
    led = _Recorder()
    box = _box(led)
    box.hooks.emit(HOOK_BEFORE_RUN, user_input="fix the lexer", settings={})
    box.hooks.emit(HOOK_AFTER_TOOL_EXECUTE, tool_name="read_file",
                   tool_args={"path": "src/lexer.py"}, tool_result="…")
    box.hooks.emit(HOOK_AFTER_TOOL_EXECUTE, tool_name="write_file",
                   tool_args={"file_path": "src/lexer.py"}, tool_result="ok")
    box.hooks.emit(HOOK_AFTER_MODEL_RESPONSE, text="fixed the quote stack",
                   round_index=1)

    answer = led.turns[-1]
    assert answer["tools"] == ("read_file", "write_file")
    assert answer["files"] == ("src/lexer.py",), (
        "one file, however many tools named it"
    )


def test_what_a_turn_used_belongs_to_that_turn_only():
    led = _Recorder()
    box = _box(led)
    box.hooks.emit(HOOK_BEFORE_RUN, user_input="a task worth remembering",
                   settings={})
    box.hooks.emit(HOOK_AFTER_TOOL_EXECUTE, tool_name="read_file",
                   tool_args={"path": "a.py"}, tool_result="…")
    box.hooks.emit(HOOK_AFTER_MODEL_RESPONSE, text="first answer, long enough",
                   round_index=0)
    box.hooks.emit(HOOK_AFTER_MODEL_RESPONSE, text="second answer, also long",
                   round_index=1)

    assert led.turns[1]["tools"] == ("read_file",)
    assert led.turns[2]["tools"] == (), (
        "the second answer used nothing; carrying the first one's tools "
        "forward would attribute work to the wrong turn"
    )


def test_short_turns_are_not_remembered():
    """Noise in a search is what makes people stop trusting it."""
    led = _Recorder()
    box = _box(led, min_chars=20)
    _a_run(box, task="a task long enough to be worth remembering",
           answers=("ok", "and here is the real answer, at length"))

    assert [t["text"] for t in led.turns] == [
        "a task long enough to be worth remembering",
        "and here is the real answer, at length",
    ]


def test_a_broken_ledger_does_not_break_the_run():
    """Memory is a side effect of doing the work, not part of it."""
    led = _Recorder(fail_on="turn")
    box = _box(led)
    _a_run(box)   # must not raise

    assert led.runs, "the run was started before the failure"
    assert led.turns == []
    assert led.finished == [], (
        "one failure drops the ledger for the rest of the run rather than "
        "retrying a write that is going to fail the same way"
    )


def test_no_ledger_means_no_hooks():
    """A hook that fires and does nothing is harder to notice than none."""
    box = ToolBox(db_pool=None, user_session={"agent_id": "a"})
    bind_run_identity(box, RunIdentity(run_id="r"))
    attach_remember_hook(box, None)

    assert not box.hooks._hooks.get(HOOK_BEFORE_RUN)


@pytest.mark.skipif(not ledger_mod.available(),
                    reason="the ledger extra is not installed")
def test_a_remembered_run_is_findable_by_recall(tmp_path):
    """End to end: the hook writes, and the tool that reads finds it."""
    registry = ledger_mod.LedgerRegistry()
    try:
        history = ledger_mod.HistoryLedger(tmp_path, registry=registry)
        box = _box(history)
        _a_run(box, task="why does the lexer keep a quote stack",
               answers=("because nested quotes need one, at length",))

        hits = history.recall("quote stack")
        assert hits, "a run that happened is a run that can be recalled"
        assert {hit["run_id"] for hit in hits} == {"run-1"}
        history.release()
    finally:
        registry.close_all()


def test_a_compaction_is_remembered_as_an_assertion(monkeypatch):
    """The squeeze is recorded, not the deletion of what it squeezed."""
    from silk.functions.hooks import HOOK_AFTER_COMPACTION

    events = []

    class _WithCompaction(_Recorder):
        def compacted(self, run_id, *, dropped, kept, rationale="", **kw):
            events.append({"run_id": run_id, "dropped": tuple(dropped),
                           "kept": kept, "rationale": rationale})

    led = _WithCompaction()
    box = _box(led)
    box.hooks.emit(HOOK_BEFORE_RUN, user_input="a task worth remembering",
                   settings={})
    box.hooks.emit(HOOK_AFTER_MODEL_RESPONSE, text="an answer, long enough",
                   round_index=0)
    box.hooks.emit(HOOK_AFTER_COMPACTION, reason="pressure", turns_dropped=6,
                   tokens_before=8000, tokens_after=2000, summary_ref="a.md")

    assert events and events[0]["run_id"] == "run-1"
    assert events[0]["kept"] == 2, "the turns this run has remembered so far"
    assert events[0]["dropped"] == (), (
        "the loop counts history messages and this hook counts remembered "
        "turns; inventing a mapping would put a claim nothing can correct "
        "into an append-only ledger"
    )
    assert "6 history messages dropped" in events[0]["rationale"]
    assert "8000 -> 2000" in events[0]["rationale"]
