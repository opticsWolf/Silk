# -*- coding: utf-8 -*-
"""Tests for prompt-prefix reuse measurement (spec D41/D47, G15).

The point of this measurement is to tell "reuse is zero" apart from "we do
not know what reuse is", so most of these tests are about what the report
refuses to claim.
"""

from __future__ import annotations


import pytest


from silk.functions.prefix_stats import (
    LogDrain,
    PrefixMeter,
    PrefixSample,
    parse_line,
    summarize,
    summarize_log,
)

HIT = "Llama.generate: 512 prefix-match hit, remaining 8 prompt tokens to eval"
PROMPT_EVAL = (
    "llama_perf_context_print: prompt eval time =     210.11 ms /     8 tokens"
)
TOTAL = "llama_perf_context_print:       total time =    1980.44 ms /   140 tokens"


# ── parsing ───────────────────────────────────────────────────────────────


def test_parses_the_numeric_prefix_line():
    fact = parse_line(HIT)
    assert (fact.kind, fact.matched, fact.evaluated) == ("prefix", 512, 8)


def test_parses_the_bare_prefix_line_without_inventing_numbers():
    fact = parse_line("Llama.generate: prefix-match hit")
    assert fact.kind == "prefix"
    assert fact.matched is None and fact.evaluated is None


def test_parses_both_timing_prefixes():
    for line in (PROMPT_EVAL, "llama_print_timings: prompt eval time = 5.0 ms / 2 tokens"):
        fact = parse_line(line)
        assert fact.kind == "prompt_eval"
    assert parse_line(TOTAL).kind == "total"


def test_ignores_unrelated_lines():
    assert parse_line("INFO: Uvicorn running on http://127.0.0.1:8000") is None


# ── one request ───────────────────────────────────────────────────────────


def test_a_request_with_no_prefix_line_counts_as_no_reuse():
    """A prompt-eval line with no hit means the whole prompt was evaluated."""
    meter = PrefixMeter()
    sample = meter.record_lines([PROMPT_EVAL, TOTAL], session="a")
    assert sample.matched == 0 and sample.evaluated == 8
    assert sample.reuse == 0.0


def test_a_request_the_server_said_nothing_about_is_not_a_sample():
    meter = PrefixMeter()
    assert meter.record_lines(["INFO: something else"], session="a") is None
    assert meter.report().requests == 0


def test_reuse_is_matched_over_prompt_tokens():
    meter = PrefixMeter()
    sample = meter.record_lines([HIT, PROMPT_EVAL, TOTAL], session="a")
    assert sample.prompt_tokens == 520
    assert sample.reuse == pytest.approx(512 / 520)


# ── the three numbers ─────────────────────────────────────────────────────


def test_report_is_unknown_before_anything_is_measured():
    report = summarize([])
    assert report.reuse_rate is None
    assert report.contention_rate is None
    assert report.prefill_share is None
    assert "unknown" in report.describe()


def test_contention_is_the_previous_request_being_another_session():
    meter = PrefixMeter()
    for session in ("a", "a", "b", "b"):
        meter.record_lines([HIT, PROMPT_EVAL, TOTAL], session=session)
    report = meter.report()
    # Three requests have a predecessor; exactly one of them switched session.
    assert report.contention_rate == pytest.approx(1 / 3)


def test_the_first_request_has_no_predecessor_and_no_verdict():
    meter = PrefixMeter()
    sample = meter.record_lines([HIT, TOTAL], session="a")
    assert sample.contended is None


def test_prefill_share_is_prompt_eval_over_total():
    meter = PrefixMeter()
    meter.record_lines([HIT, PROMPT_EVAL, TOTAL], session="a")
    assert meter.report().prefill_share == pytest.approx(210.11 / 1980.44)


def test_prefill_share_stays_unknown_without_a_total():
    meter = PrefixMeter()
    meter.record_lines([HIT, PROMPT_EVAL], session="a")
    assert meter.report().prefill_share is None


def test_bare_hits_are_counted_but_do_not_move_the_rate():
    meter = PrefixMeter()
    meter.record_lines(["Llama.generate: prefix-match hit"], session="a")
    report = meter.report()
    assert report.bare_hits == 1
    assert report.reuse_rate is None


def test_the_sample_window_is_bounded():
    meter = PrefixMeter(max_samples=3)
    for _ in range(10):
        meter.record_lines([HIT, TOTAL], session="a")
    assert len(meter.samples()) == 3


def test_as_dict_carries_every_metric():
    keys = set(PrefixMeter().report().as_dict())
    assert {"reuse_rate", "contention_rate", "prefill_share", "requests"} <= keys


# ── reading a log ─────────────────────────────────────────────────────────


def test_the_drain_only_returns_what_is_new(tmp_path):
    path = tmp_path / "server.log"
    path.write_text("one\ntwo\n", encoding="utf-8")
    drain = LogDrain(str(path))
    assert drain.drain() == ["one", "two"]
    assert drain.drain() == []
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("three\n")
    assert drain.drain() == ["three"]


def test_a_truncated_log_is_read_from_the_start(tmp_path):
    """A relaunched server rewrites the file; a stale offset would skip it."""
    path = tmp_path / "server.log"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    drain = LogDrain(str(path))
    drain.drain()
    path.write_text("x\n", encoding="utf-8")
    assert drain.drain() == ["x"]


def test_a_missing_log_drains_to_nothing(tmp_path):
    assert LogDrain(str(tmp_path / "nope.log")).drain() == []


def test_summarize_log_reads_requests_but_not_contention(tmp_path):
    path = tmp_path / "server.log"
    path.write_text("\n".join([HIT, PROMPT_EVAL, TOTAL] * 2) + "\n", encoding="utf-8")
    report = summarize_log(path)
    assert report.requests == 2
    assert report.reuse_rate == pytest.approx(512 / 520)
    # Nothing in the file says whose request a line was.
    assert report.contention_rate is None


def test_summarize_log_of_a_missing_file_is_empty():
    assert summarize_log("no-such-file.log").requests == 0


# ── the pool seam ─────────────────────────────────────────────────────────


def test_the_engine_measures_through_the_pool_and_survives_a_pool_that_cannot():
    """`begin_request`/`end_request` are optional, and must never fail a run."""
    from silk.functions.graph_engine import GraphEngine

    engine = GraphEngine.__new__(GraphEngine)
    engine.session_id = "s1"

    calls = []

    class Measuring:
        def begin_request(self, session_id="default"):
            calls.append(("begin", session_id))

        def end_request(self, session_id="default", wall_s=None):
            calls.append(("end", session_id, wall_s))

    class Broken:
        def begin_request(self, **_):
            raise RuntimeError("boom")

        def end_request(self, **_):
            raise RuntimeError("boom")

    pool = Measuring()
    engine._begin_measured_request(pool)
    engine._end_measured_request(pool, 1.5)
    assert calls == [("begin", "s1"), ("end", "s1", 1.5)]

    # A pool without the hooks, and a pool whose hooks raise, are both fine.
    engine._begin_measured_request(object())
    engine._end_measured_request(object(), 1.0)
    engine._begin_measured_request(Broken())
    engine._end_measured_request(Broken(), 1.0)
    engine._begin_measured_request(None)
    engine._end_measured_request(None, 1.0)


def test_sample_reports_unknown_reuse_without_numbers():
    assert PrefixSample(session="a").reuse is None
    assert PrefixSample(session="a").prompt_tokens is None
