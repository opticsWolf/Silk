# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Compaction (spec §12, D24/D25/D40/D41; invariant I9).

Three claims, in descending order of how much damage getting them wrong
does.

**It cuts on round boundaries.** An assistant turn and the tool results
answering it move together or not at all. The half that bites is the
orphan: a surviving `tool` message whose call was summarized away is
accepted by the current request and corrupts the *next* one, so the bug
surfaces rounds later as a model discussing a tool it never called. The
rule is enforced twice on purpose -- the compactor plans cuts that satisfy
it, and the engine refuses cuts that do not -- so a second caller cannot
quietly reintroduce it.

**It is atomic.** The summary is produced first; only a summary that
actually arrived is swapped in. Every failure -- a summarizer that raises,
refuses, or returns whitespace -- leaves the history byte-identical, which
is exactly the pre-D24 behaviour rather than a new way to lose a run.

**It is rare.** D41: a compaction costs two full prefills, and the trigger
therefore has hysteresis, a generous keep-recent, and a minimum worth
dropping. A compactor that fires every round is not a smaller bug than one
that never fires; it is a slower run than not compacting at all.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest


from silk.functions.agent_loop import AgentLoop  # noqa: E402
from silk.functions.compaction import (  # noqa: E402
    SUMMARY_MARKER,
    Compactor,
    plan_cut,
    render_transcript,
    round_boundaries,
)
from silk.functions.graph_engine import GraphEngine  # noqa: E402
from silk.functions.usage_limits import UsageLimits  # noqa: E402
from silk.functions.stream_events import (  # noqa: E402
    EventCompaction,
    EventError,
    EventRunResult,
)


# -- doubles ----------------------------------------------------------------


def _round(n: int) -> list[dict[str, Any]]:
    """One full round: a user turn, an assistant turn, two tool results."""
    return [
        {"role": "user", "content": f"question {n}"},
        {"role": "assistant", "content": f"calling tools for {n}"},
        {"role": "tool", "name": "read_file", "content": f"result {n}a"},
        {"role": "tool", "name": "grep", "content": f"result {n}b"},
    ]


def _history(rounds: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in range(rounds):
        out.extend(_round(n))
    return out


class FakeEngine:
    """The AgentEngine surface compaction touches, and nothing else."""

    def __init__(self, history=None, *, context=1000, tokens=None):
        self.history = history if history is not None else _history(6)
        self._context = context
        self._tokens = tokens
        self.summarized: list[str] = []
        self.summary_reply = "the story so far"
        self.replaced: list[int] = []

    def append_message(self, role, content, **_stats):
        self.history.append({"role": role, "content": content})

    # the two optional operations
    def replace_history_prefix(self, count: int, summary: str, **_kw) -> int:
        if count < len(self.history) and self.history[count].get("role") == "tool":
            raise ValueError("that cut would orphan a tool result (I9)")
        self.history[:count] = [{"role": "user", "content": summary}]
        self.replaced.append(count)
        return count

    def sibling(self, *, system_prompt: str = "", history=None) -> "FakeEngine":
        parent = self

        class _Sibling(FakeEngine):
            def stream_response(self, _params):
                parent.summarized.append(self.history[-1]["content"])
                if isinstance(parent.summary_reply, Exception):
                    raise parent.summary_reply
                yield parent.summary_reply

        return _Sibling([], context=self._context, tokens=self._tokens)

    def stream_response(self, _params):      # pragma: no cover - overridden
        yield ""

    def context_length(self) -> Optional[int]:
        return self._context

    def count_prompt_tokens(self) -> int:
        if self._tokens is not None:
            return self._tokens
        return sum(len(str(e.get("content", ""))) for e in self.history) // 4


# -- where a cut may land (I9) ----------------------------------------------


def test_round_boundaries_exclude_tool_turns():
    boundaries = round_boundaries(_history(2))
    assert boundaries == [0, 1, 4, 5], "a tool result is never a cut point"


def test_an_assistant_turn_and_its_results_move_together():
    """I9. Every planned cut lands where no call is split from its results."""
    for rounds in range(2, 12):
        history = _history(rounds)
        cut = plan_cut(history, keep_recent=4, min_dropped=2)
        assert cut is not None
        assert history[cut]["role"] != "tool"
        kept = history[cut:]
        assert kept[0]["role"] != "tool", "the surviving head starts a round"


def test_a_tool_result_is_never_orphaned():
    """I9, the half that corrupts the *next* request rather than this one."""
    engine = FakeEngine(_history(6))
    Compactor(keep_recent=6, min_dropped=2).maybe_compact(engine, force=True)

    for index, entry in enumerate(engine.history):
        if entry.get("role") != "tool":
            continue
        assert any(e.get("role") == "assistant" for e in engine.history[:index]), (
            "a tool result survived without the assistant turn that called it"
        )


def test_the_engine_refuses_a_cut_that_would_orphan_a_result():
    """The rule is a property of the operation, not of one caller."""
    engine = GraphEngine({"backend": "gguf", "model": object()})
    engine.history = _history(2)
    with pytest.raises(ValueError, match="I9"):
        engine.replace_history_prefix(2, "summary")     # index 2 is a tool turn
    assert engine.history == _history(2), "a refused cut changes nothing"


def test_the_cut_snaps_down_so_it_keeps_more_than_asked_not_less():
    history = _history(3)                                # 12 turns
    cut = plan_cut(history, keep_recent=5, min_dropped=2)
    assert cut == 5, "target 7 is a tool turn; the snap keeps turns 5..11"
    assert len(history) - cut >= 5


@pytest.mark.parametrize("history", [
    [],
    _history(1),                                  # four turns, nothing to drop
    [{"role": "tool", "name": "t", "content": "orphan"}] * 6,
])
def test_a_history_with_no_worthwhile_cut_is_left_alone(history):
    assert plan_cut(list(history), keep_recent=4, min_dropped=4) is None


# -- the swap is atomic -----------------------------------------------------


def test_a_compaction_replaces_the_prefix_with_one_labelled_summary():
    engine = FakeEngine(_history(6))
    before = len(engine.history)
    event = Compactor(keep_recent=8, min_dropped=4).maybe_compact(
        engine, force=True)

    assert isinstance(event, EventCompaction)
    assert len(engine.history) == before - event.turns_dropped + 1
    head = engine.history[0]
    assert head["content"].startswith(SUMMARY_MARKER)
    assert "the story so far" in head["content"]
    assert engine.history[1:] == _history(6)[event.turns_dropped:]


def test_the_history_object_is_mutated_not_rebound():
    """The Agent node holds this list; a rebind would strand it."""
    history = _history(6)
    engine = FakeEngine(history)
    Compactor(keep_recent=8, min_dropped=4).maybe_compact(engine, force=True)
    assert engine.history is history and history[0]["content"].startswith(
        SUMMARY_MARKER)


def test_the_summary_is_produced_before_anything_is_dropped():
    engine = FakeEngine(_history(6))
    Compactor(keep_recent=8, min_dropped=4).maybe_compact(engine, force=True)

    assert len(engine.summarized) == 1
    prompt = engine.summarized[0]
    assert "question 0" in prompt, "the dropped turns are what gets summarized"
    assert "transcript ends" in prompt


@pytest.mark.parametrize("reply", [
    RuntimeError("the summarizer died"),
    "",
    "   \n  ",
])
def test_a_failed_summary_leaves_the_history_untouched(reply):
    engine = FakeEngine(_history(6))
    engine.summary_reply = reply
    original = list(engine.history)

    assert Compactor(keep_recent=8, min_dropped=4).maybe_compact(
        engine, force=True) is None
    assert engine.history == original and engine.replaced == []


def test_an_engine_that_cannot_rewrite_its_history_is_not_compacted():
    class _Appendix(FakeEngine):
        replace_history_prefix = None

    engine = _Appendix(_history(6))
    assert Compactor().maybe_compact(engine, force=True) is None


# -- the trigger, and why it is rare (D41) ----------------------------------


def test_pressure_below_the_threshold_does_not_compact():
    engine = FakeEngine(_history(6), context=8000, tokens=5000)
    compactor = Compactor()
    assert compactor.threshold(8000) == 6000, "a quarter of the window, reserved"
    assert compactor.should_compact(engine) is False
    assert compactor.maybe_compact(engine) is None


def test_pressure_above_the_threshold_compacts():
    engine = FakeEngine(_history(6), context=8000, tokens=6500)
    assert Compactor(min_dropped=4, keep_recent=8).maybe_compact(
        engine) is not None


def test_an_unknown_context_window_never_triggers_pressure():
    """No denominator, no pressure -- and no guessing (G14(c))."""
    engine = FakeEngine(_history(6), context=None, tokens=10 ** 6)
    compactor = Compactor()
    assert compactor.should_compact(engine) is False
    assert compactor.maybe_compact(engine) is None


def test_hysteresis_stops_a_run_compacting_every_round():
    engine = FakeEngine(_history(8), context=8000, tokens=6500)
    compactor = Compactor(keep_recent=8, min_dropped=4, hysteresis=0.10)
    assert compactor.maybe_compact(engine) is not None

    # The prompt is still over the threshold -- compaction rarely gets a
    # context far under it -- but it has not grown since, so a second one
    # would buy a turn or two for two more full prefills.
    engine._tokens = 6600
    assert compactor.should_compact(engine) is False
    engine._tokens = 7500
    assert compactor.should_compact(engine) is True, "real growth still counts"


def test_the_event_reports_the_prefill_cost_not_only_what_was_dropped():
    engine = FakeEngine(_history(6), context=8000)
    event = Compactor(keep_recent=8, min_dropped=4).maybe_compact(
        engine, force=True)

    assert event.tokens_before > event.tokens_after
    assert event.prefill_tokens >= event.tokens_after, (
        "the summarization request is the prefill nobody expects (D41)"
    )
    assert event.turns_dropped >= 4


def test_the_event_carries_a_reference_not_the_transcript():
    """EventCompaction is content-free (§5); the dropped text goes to a file."""
    from silk.functions.spill import SpillWriter

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        writer = SpillWriter(tmp)
        engine = FakeEngine(_history(6))
        event = Compactor(keep_recent=8, min_dropped=4,
                          writer=writer).maybe_compact(engine, force=True)

        assert event.summary_ref and event.summary_ref.endswith(".md")
        assert "question 0" in (writer.written[0]).read_text(encoding="utf-8")
        assert "question 0" not in str(event), "the event stays content-free"


def test_a_custom_summarizer_replaces_the_model_request():
    engine = FakeEngine(_history(6))
    event = Compactor(keep_recent=8, min_dropped=4,
                      summarizer=lambda text: f"{len(text)} chars, summarized"
                      ).maybe_compact(engine, force=True)

    assert event is not None and engine.summarized == [], "no model request"
    assert "summarized" in engine.history[0]["content"]


# -- rendering the dropped turns --------------------------------------------


def test_the_transcript_names_the_tool_a_result_came_from():
    text = render_transcript(_history(1))
    assert "tool result (read_file):" in text and "question 0" in text


def test_a_huge_turn_is_trimmed_before_it_reaches_the_summarizer():
    """A summarization request that itself overflows is a wasted prefill."""
    text = render_transcript(
        [{"role": "user", "content": "A" * 500 + "B" * 500}], max_turn_chars=100)
    assert len(text) < 400 and "characters omitted" in text
    assert text.startswith("user: " + "A" * 50)


# -- the engine operation ---------------------------------------------------


def test_replace_history_prefix_forgives_the_break_it_causes():
    """Compaction is I11's single legitimate prefix break."""
    engine = GraphEngine({"backend": "gguf", "model": object()},
                         system_prompt="stable")
    engine.history = _history(2)
    engine.prefix_guard.observe(engine.build_messages(), system_prompt="stable")
    engine.replace_history_prefix(4, "summary")
    assert engine.prefix_guard.observe(engine.build_messages(),
                                       system_prompt="stable") is None
    assert engine.prefix_guard.clean


@pytest.mark.parametrize("count", [0, -1, 99])
def test_replace_history_prefix_rejects_a_nonsensical_count(count):
    engine = GraphEngine({"backend": "gguf", "model": object()})
    engine.history = _history(1)
    with pytest.raises(ValueError):
        engine.replace_history_prefix(count, "summary")


def test_a_sibling_shares_the_budget_and_the_pool_session():
    """D25: the agent's own model and pool session does the summarizing."""
    engine = GraphEngine({"backend": "gguf", "model": object()},
                         system_prompt="agent", session_id="conversation-a")
    engine.append_message("user", "hi")
    sibling = engine.sibling(system_prompt="summarizer")

    assert sibling.session_id == "conversation-a"
    assert sibling.usage_limits is engine.usage_limits
    assert sibling.system_prompt == "summarizer" and sibling.history == []
    assert engine.history == [{"role": "user", "content": "hi"}], (
        "the summarization prompt never touches the run history"
    )


# -- the loop wiring --------------------------------------------------------


class LoopEngine(FakeEngine):
    """An engine whose model request can be told to fail."""

    def __init__(self, *args, replies=None, **kw):  # noqa: D107
        super().__init__(*args, **kw)
        self.replies = list(replies or ["done"])
        self.requests = 0
        self.usage_limits = UsageLimits()
        self.reflection_config = None
        self.last_stats: dict[str, Any] = {}
        self._stopped = False

    def stream_response(self, _params):
        self.requests += 1
        reply = self.replies.pop(0) if self.replies else "done"
        if isinstance(reply, Exception):
            self.last_stats = {"error": str(reply)}
            raise reply
        self.last_stats = {"finish_reason": "stop", "tokens": 1}
        yield reply

    def append_message(self, role, content, **_stats):
        self.history.append({"role": role, "content": content})

    def request_stop(self):
        self._stopped = True

    def stop_requested(self):
        return self._stopped


def _events(loop, prompt="go"):
    return list(loop.run(prompt))


def test_a_loop_without_a_compactor_behaves_exactly_as_before():
    engine = LoopEngine(_history(6), context=8000, tokens=10 ** 6)
    events = _events(AgentLoop(engine))
    assert not any(isinstance(e, EventCompaction) for e in events)
    assert engine.history[0] == _history(6)[0], "history untouched"


def test_the_pre_request_seam_shrinks_instead_of_failing_the_run():
    """D24: the seam that used to end a full run now makes room at it."""
    engine = LoopEngine(_history(8), context=8000, tokens=7000)
    events = _events(AgentLoop(engine, compactor=Compactor(keep_recent=8,
                                                           min_dropped=4)))

    compactions = [e for e in events if isinstance(e, EventCompaction)]
    assert len(compactions) == 1 and compactions[0].turns_dropped >= 4
    assert isinstance(events[-1], EventRunResult)
    assert engine.requests == 1, "the run carried on rather than failing"


def test_a_classified_overflow_compacts_once_and_retries():
    engine = LoopEngine(
        _history(8), context=1000, tokens=500,
        replies=[ValueError("Requested tokens (5000) exceed context window of 4096"),
                 "recovered"],
    )
    events = _events(AgentLoop(engine, compactor=Compactor(keep_recent=8,
                                                           min_dropped=4)))

    assert [type(e).__name__ for e in events].count("EventCompaction") == 1
    assert engine.requests == 2, "the round was retried after making room"
    assert isinstance(events[-1], EventRunResult)
    assert events[-1].text == "recovered"


def test_a_dead_server_is_not_answered_with_a_summarization_request():
    """D40: only a classified overflow may trigger compaction."""
    engine = LoopEngine(_history(8), context=1000, tokens=500,
                        replies=[ConnectionError("connection refused")])
    events = _events(AgentLoop(engine, compactor=Compactor()))

    assert not any(isinstance(e, EventCompaction) for e in events)
    assert engine.summarized == [], "no request was spent on a dead backend"
    assert any(isinstance(e, EventError) for e in events)
    assert engine.requests == 1


def test_an_overflow_that_cannot_be_compacted_still_ends_the_run():
    engine = LoopEngine(
        _history(1), context=1000, tokens=500,
        replies=[ValueError("exceeds context window of 4096"), "unreachable"],
    )
    events = _events(AgentLoop(engine, compactor=Compactor()))

    assert not any(isinstance(e, EventCompaction) for e in events)
    assert engine.requests == 1
    assert isinstance(events[-1], EventError)


def test_the_overflow_retry_happens_at_most_once_a_run():
    """A second one would spend two more prefills to be told the same thing."""
    boom = ValueError("exceeds context window of 4096")
    engine = LoopEngine(_history(12), context=1000, tokens=500,
                        replies=[boom, boom, "unreachable"])
    events = _events(AgentLoop(engine, compactor=Compactor(keep_recent=4,
                                                           min_dropped=2)))

    assert [type(e).__name__ for e in events].count("EventCompaction") == 1
    assert engine.requests == 2
    assert isinstance(events[-1], EventError)


def test_a_compactor_that_raises_does_not_kill_the_run():
    class _Broken(Compactor):
        def maybe_compact(self, *_a, **_kw):
            raise RuntimeError("the compactor is broken")

    engine = LoopEngine(_history(8), context=1000, tokens=900)
    events = _events(AgentLoop(engine, compactor=_Broken()))
    assert isinstance(events[-1], EventRunResult)


# -- carrying a run across several compactions (lessons from pi) -------------


def _summary_entry(body: str) -> dict[str, Any]:
    return {"role": "user", "content": SUMMARY_MARKER + body}


def test_a_previous_summary_is_carried_forward_whole_not_clipped():
    """The second compaction updates the first summary; it never re-summarizes
    it as one more turn, which clipped it to max_turn_chars and lost its
    middle."""
    middle = "the measured gap is 41.7 nm"
    previous = "## Task\n" + "x" * 3000 + f"\n{middle}\n" + "y" * 3000
    engine = FakeEngine([_summary_entry(previous)] + _history(6))
    Compactor(keep_recent=8, min_dropped=4).maybe_compact(engine, force=True)

    prompt = engine.summarized[0]
    assert middle in prompt, "the old summary's middle reached the summarizer"
    assert "previous summary begins" in prompt and "PRESERVE" in prompt
    transcript = prompt.split("--- transcript begins ---")[1]
    assert SUMMARY_MARKER.strip() not in transcript, (
        "the old summary is not also rendered as a turn"
    )
    assert engine.history[0]["content"].count(SUMMARY_MARKER) == 1


def test_a_first_compaction_uses_the_structured_template():
    engine = FakeEngine(_history(6))
    Compactor(keep_recent=8, min_dropped=4).maybe_compact(engine, force=True)
    prompt = engine.summarized[0]
    for section in ("## Task", "## Progress", "### Done", "## Key decisions",
                    "## Facts discovered", "## Next steps"):
        assert section in prompt
    assert "previous summary" not in prompt
    assert "Preserve exact file paths" in prompt


def test_the_focus_reaches_the_prompt():
    engine = FakeEngine(_history(6))
    Compactor(keep_recent=8, min_dropped=4,
              focus="keep every layer thickness").maybe_compact(engine, force=True)
    assert "Additional focus: keep every layer thickness" in engine.summarized[0]


def test_compact_now_skips_the_threshold_and_takes_a_one_off_focus():
    engine = FakeEngine(_history(6), context=100_000, tokens=10)
    compactor = Compactor(keep_recent=8, min_dropped=4, focus="standing")
    assert compactor.should_compact(engine) is False
    event = compactor.compact_now(engine, focus="just this once")
    assert event is not None
    assert "Additional focus: just this once" in engine.summarized[0]
    assert "standing" not in engine.summarized[0]


def test_compact_now_still_respects_the_minimum_worth_dropping():
    engine = FakeEngine(_history(1))
    assert Compactor().compact_now(engine) is None
    assert engine.summarized == []


# -- what the summarizer is shown ---------------------------------------------


def test_native_tool_calls_are_rendered_with_their_arguments():
    entries = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path": "src/stack.py"}'}},
        ]},
        {"role": "tool", "name": "read_file", "content": "def stack(): ..."},
    ]
    text = render_transcript(entries)
    assert 'read_file({"path": "src/stack.py"})' in text
    assert "assistant tool calls:" in text


def test_fenced_calls_are_lifted_out_so_clipping_cannot_split_them():
    fence = '```tool_call\n{"name": "edit_file", "arguments": {"path": "a.py"}}\n```'
    content = "A" * 3000 + "\n" + fence + "\n" + "B" * 3000
    text = render_transcript([{"role": "assistant", "content": content}],
                             max_turn_chars=200)
    assert 'edit_file({"path": "a.py"})' in text
    assert "```tool_call" not in text


def test_a_huge_call_argument_is_clipped():
    entries = [{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "write_file",
                      "arguments": '{"path": "big.txt", "content": "' + "z" * 5000 + '"}'}},
    ]}]
    text = render_transcript(entries, max_arg_chars=100)
    assert "big.txt" in text and "characters omitted" in text
    assert len(text) < 400


# -- files touched, recorded by code ------------------------------------------


def _file_round() -> list[dict[str, Any]]:
    def call(name, **args):
        import json
        return {"function": {"name": name, "arguments": json.dumps(args)}}
    return [
        {"role": "user", "content": "refactor"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("read_file", path="a.py"), call("read_file", path="b.py"),
            call("edit_file", path="b.py", old="x", new="y"),
            call("move_file", source="c.py", destination="d.py"),
        ]},
        {"role": "tool", "name": "read_file", "content": "..."},
        {"role": "tool", "name": "read_file", "content": "..."},
        {"role": "tool", "name": "edit_file", "content": "ok"},
        {"role": "tool", "name": "move_file", "content": "ok"},
    ]


def test_the_files_a_dropped_turn_touched_are_appended_to_the_summary():
    engine = FakeEngine(_file_round() + _history(3))
    Compactor(keep_recent=8, min_dropped=4).maybe_compact(engine, force=True)
    head = engine.history[0]["content"]
    assert "<read-files>\na.py\n</read-files>" in head, (
        "b.py was modified, so it is listed once, as modified"
    )
    assert "<modified-files>\nb.py\nc.py\nd.py\n</modified-files>" in head


def test_file_lists_accumulate_across_compactions():
    previous = "## Task\nold work\n\n<read-files>\nold.py\n</read-files>"
    engine = FakeEngine([_summary_entry(previous)] + _file_round() + _history(3))
    Compactor(keep_recent=8, min_dropped=4).maybe_compact(engine, force=True)
    head = engine.history[0]["content"]
    assert "old.py" in head and "a.py" in head
    assert "<read-files>" not in engine.summarized[0].split(
        "--- previous summary ends ---")[0], (
        "the lists are the code's, so the model is not asked to keep them"
    )


def test_file_lists_a_model_writes_itself_are_replaced_by_the_records():
    engine = FakeEngine(_file_round() + _history(3))
    engine.summary_reply = "notes\n\n<modified-files>\ninvented.py\n</modified-files>"
    Compactor(keep_recent=8, min_dropped=4).maybe_compact(engine, force=True)
    head = engine.history[0]["content"]
    assert "invented.py" not in head and "b.py" in head


# -- a truncated summary is no summary ----------------------------------------


@pytest.mark.parametrize("stats", [
    {"finish_reason": "length", "tokens": 512},
    {"finish_reason": None, "truncated": True},
])
def test_a_cut_off_summary_leaves_the_history_untouched(stats):
    engine = FakeEngine(_history(6))
    original = list(engine.history)
    real_sibling = engine.sibling

    def sibling(**kw):
        sib = real_sibling(**kw)
        inner = sib.stream_response

        def stream(params):
            yield from inner(params)
            sib.last_stats = dict(stats)
        sib.stream_response = stream
        return sib

    engine.sibling = sibling
    assert Compactor(keep_recent=8, min_dropped=4).maybe_compact(
        engine, force=True) is None
    assert engine.history == original


# -- the tail is sized in tokens ----------------------------------------------


def test_the_tail_is_sized_by_tokens_when_the_window_is_known():
    """Eight huge tool results would fill the window by themselves."""
    big = [{"role": "user", "content": "q"},
           {"role": "assistant", "content": "reading"}] + [
        {"role": "tool", "name": "read_file", "content": "r" * 4000}
        for _ in range(2)]
    history = big * 6                                  # 24 turns, ~12k tokens
    cut = plan_cut(history, keep_tokens=2500, min_dropped=2)
    kept = history[cut:]
    assert sum(len(e["content"]) for e in kept) // 4 <= 2500 + 2000, (
        "the tail fits the budget, give or take the snap to a round"
    )
    assert len(kept) < 8, "fewer turns than the old fixed count"


def test_small_turns_keep_more_than_eight_when_they_fit():
    history = _history(12)                             # 48 short turns
    cut = plan_cut(history, keep_tokens=40, min_dropped=2)
    assert len(history) - cut > 8


def test_the_default_compactor_uses_the_window_share():
    compactor = Compactor()
    assert compactor.keep_tokens(10_000) == 3000
    assert compactor.keep_tokens(None) is None, "unknown window: count turns"
    assert Compactor(keep_recent=8).keep_tokens(10_000) is None, (
        "an explicit turn count wins"
    )


def test_the_newest_turn_is_always_kept():
    history = _history(3) + [{"role": "user", "content": "u" * 40_000}]
    cut = plan_cut(history, keep_tokens=10, min_dropped=1)
    assert cut is not None and cut <= len(history) - 1


# -- an early length stop is an overflow --------------------------------------


class LengthEngine(LoopEngine):
    """Replies are ``(text, stats)``; plain strings finish with ``stop``."""

    def stream_response(self, params):
        self.requests += 1
        reply = self.replies.pop(0) if self.replies else "done"
        text, stats = reply if isinstance(reply, tuple) else (
            reply, {"finish_reason": "stop", "tokens": 1})
        self.last_stats = dict(stats)
        yield text


def test_a_length_stop_short_of_the_cap_compacts_and_retries():
    engine = LengthEngine(
        _history(8), context=1000, tokens=500,
        replies=[("half an ans", {"finish_reason": "length", "tokens": 3}),
                 "whole answer"],
    )
    loop = AgentLoop(engine, compactor=Compactor(keep_recent=8, min_dropped=4))
    events = list(loop.run("go", gen_params={"max_tokens": 1024}))

    assert [type(e).__name__ for e in events].count("EventCompaction") == 1
    assert engine.requests == 2
    assert events[-1].text == "whole answer"
    assert not any(e.get("content") == "half an ans" for e in engine.history), (
        "the cut-off answer is dropped, not persisted"
    )


def test_a_length_stop_with_no_cap_is_an_overflow_too():
    engine = LengthEngine(
        _history(8), context=1000, tokens=500,
        replies=[("half", {"finish_reason": "length", "tokens": 3}), "whole"],
    )
    events = _events(AgentLoop(engine, compactor=Compactor(keep_recent=8,
                                                           min_dropped=4)))
    assert engine.requests == 2 and events[-1].text == "whole"


def test_a_length_stop_at_the_cap_is_the_callers_choice_not_an_overflow():
    engine = LengthEngine(
        _history(8), context=1000, tokens=500,
        replies=[("capped answer", {"finish_reason": "length", "tokens": 64})],
    )
    loop = AgentLoop(engine, compactor=Compactor(keep_recent=8, min_dropped=4))
    events = list(loop.run("go", gen_params={"max_tokens": 64}))
    assert not any(isinstance(e, EventCompaction) for e in events)
    assert engine.requests == 1 and events[-1].text == "capped answer"


def test_a_length_stop_without_a_compactor_is_kept_as_before():
    engine = LengthEngine(
        _history(8), context=1000, tokens=500,
        replies=[("half", {"finish_reason": "length", "tokens": 3})],
    )
    events = _events(AgentLoop(engine))
    assert engine.requests == 1 and events[-1].text == "half"


# -- configured on the ToolBox: the auto_compact hook and the tool -----------


def _box(hooks=(), configs=None, tools=False):
    from silk.functions.hook_catalog import attach_catalog_hooks
    from silk.functions.tool_box import ToolBox
    from silk.functions.tools.context_tools import attach_context_tools

    box = ToolBox()
    if tools:
        attach_context_tools(box, None)
    if hooks:
        attach_catalog_hooks(box, None, names=hooks, configs=configs or {})
    return box


def test_auto_compact_is_a_default_hook_and_not_a_gate():
    from silk.functions.hook_catalog import DEFAULT_HOOKS, GATING_HOOKS

    assert "auto_compact" in DEFAULT_HOOKS and "auto_compact" not in GATING_HOOKS


def test_no_toolbox_compacts_automatically_with_the_defaults():
    from silk.functions.compaction import compactor_for

    c = compactor_for(None)
    assert c.auto and c.keep_recent_ratio == 0.30


def test_the_hook_settings_reach_the_compactor():
    from silk.functions.compaction import compactor_for

    box = _box(["auto_compact"], {"auto_compact": {
        "keep_recent_pct": 40, "reserve_pct": 20, "hysteresis_pct": 5,
        "min_dropped": 2, "focus": "keep every thickness"}})
    c = compactor_for(box)
    assert c.auto
    assert (c.keep_recent_ratio, c.reserve_ratio, c.hysteresis, c.min_dropped) == (
        0.40, 0.20, 0.05, 2)
    assert c.focus == "keep every thickness"


def test_an_invalid_hook_config_falls_back_to_the_defaults():
    from silk.functions.compaction import compactor_for

    c = compactor_for(_box(["auto_compact"], {"auto_compact": {"keep_recent_pct": 99}}))
    assert c.auto and c.keep_recent_ratio == 0.30


def test_a_toolbox_without_the_hook_never_compacts_on_its_own():
    from silk.functions.compaction import compactor_for

    c = compactor_for(_box(["timing"]))
    assert c.auto is False
    engine = FakeEngine(_history(8), context=8000, tokens=7900)
    assert c.maybe_compact(engine) is None, "no pressure trigger"
    assert c.maybe_compact(engine, reason="overflow", force=True) is None, (
        "no overflow trigger either"
    )
    c.keep_recent = 4                    # the doubles' turns are tiny
    assert c.compact_now(engine) is not None, "an explicit request still works"


def test_the_tool_records_a_request_the_loop_then_takes():
    from silk.functions.compaction import take_compaction_request

    box = _box(tools=True)
    assert take_compaction_request(box) is None
    out = box.tools["compact_context"]["executable"](focus="  the failing tests ")
    assert out.ok
    box.tools["compact_context"]["executable"](focus="")
    assert take_compaction_request(box) == "the failing tests", (
        "two calls are one compaction; the last non-empty focus wins"
    )
    assert take_compaction_request(box) is None, "taken once"


def test_the_loop_compacts_before_the_next_request_when_the_tool_asked():
    box = _box(tools=True)               # no auto_compact: only the tool
    from silk.functions.compaction import compactor_for

    engine = LoopEngine(
        _history(8), context=100_000, tokens=100,
        replies=['```tool_call\n{"name": "compact_context", '
                 '"arguments": {"focus": "keep the ids"}}\n```', "done"],
    )
    loop = AgentLoop(engine, box, compactor=compactor_for(box))
    loop.compactor.keep_recent = 4
    events = _events(loop)

    compactions = [e for e in events if isinstance(e, EventCompaction)]
    assert len(compactions) == 1
    assert "Additional focus: keep the ids" in engine.summarized[0]
    assert engine.requests == 2 and isinstance(events[-1], EventRunResult)


def test_an_old_toolbox_keeps_compacting_after_migration():
    from silk.nodes.toolbox import SilkToolBoxNode

    state = {"widget_data": {"hooks_config": {"names": ["timing"], "configs": {}}},
             "seen_tools": ["read_file"]}
    out = SilkToolBoxNode.migrate_state(state, 3)
    assert out["widget_data"]["hooks_config"]["names"] == ["timing", "auto_compact"]
    assert "compact_context" in out["seen_tools"], "offered, so not auto-ticked"
    assert SilkToolBoxNode.node_state_api == 7
