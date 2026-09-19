# -*- coding: utf-8 -*-
"""D90: what the runs cost, kept across them.

D88 made money a ceiling and D90 makes it a record. The distinction these
tests keep pinning down is the one the whole module turns on: an unpriced
run is not a free one, so it produces no line rather than a line of
zeroes, and a model that ran without a quote is flagged in the breakdown
rather than folded into the total.
"""
from __future__ import annotations

import json

import pytest

from weave.plugins.silk.functions.cost_ledger import (
    SCHEMA,
    describe_summary,
    read_lines,
    record_run,
    summarize,
)
from weave.plugins.silk.functions.graph_engine import GraphEngine
from weave.plugins.silk.functions.model_fallback import build_chain
from weave.plugins.silk.functions.usage_limits import UsageLimits


@pytest.fixture
def ledger(tmp_path):
    return tmp_path / "spend.jsonl"


def entry(model="openai:gpt-x", cost=0.01, **kw):
    base = {"model": model, "backend": "openai", "requests": 1,
            "input_tokens": 1_000, "output_tokens": 500, "cost": cost}
    base.update(kw)
    return base


# -- what gets written ------------------------------------------------------

def test_a_priced_run_writes_one_line(ledger):
    assert record_run([entry()], run_id="r1", agent="Agent",
                      elapsed_s=2.5, outcome="completed", path=ledger) is True
    lines = read_lines(ledger)
    assert len(lines) == 1
    assert lines[0]["cost"] == 0.01
    assert lines[0]["schema"] == SCHEMA
    assert lines[0]["run_id"] == "r1"
    assert lines[0]["agent"] == "Agent"
    assert lines[0]["outcome"] == "completed"
    assert lines[0]["models"][0]["input_tokens"] == 1_000


def test_an_unpriced_run_writes_nothing_at_all(ledger):
    """"Off by default", answered by the data instead of a checkbox:
    somebody who only runs local models never acquires a file."""
    assert record_run([entry(cost=None)], path=ledger) is False
    assert not ledger.exists()


def test_a_run_that_reached_no_model_writes_nothing(ledger):
    assert record_run([], path=ledger) is False
    assert record_run([entry(requests=0)], path=ledger) is False


def test_a_free_quote_is_still_a_quote(ledger):
    """Zero is a price some gateways genuinely quote, and a run at zero
    is a fact worth having -- it is not the same as nobody saying."""
    assert record_run([entry(cost=0.0)], path=ledger) is True
    assert read_lines(ledger)[0]["cost"] == 0.0


def test_lines_accumulate_rather_than_replace(ledger):
    for i in range(3):
        record_run([entry(cost=0.01)], run_id=f"r{i}", path=ledger)
    assert [r["run_id"] for r in read_lines(ledger)] == ["r0", "r1", "r2"]


def test_a_partly_priced_chain_totals_only_what_was_quoted(ledger):
    """A run that fell back from a metered gateway to a local model has a
    real, partial bill -- and the unpriced half stays visible."""
    record_run([entry("openai:gpt-x", 0.02),
                entry("gguf:qwen", None)], path=ledger)
    line = read_lines(ledger)[0]
    assert line["cost"] == 0.02
    assert [m["cost"] for m in line["models"]] == [0.02, None]


def test_a_torn_line_costs_only_itself(ledger):
    record_run([entry()], run_id="good", path=ledger)
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write('{"schema": 1, "cost": 0.5, "mo\n')   # killed mid-write
    record_run([entry()], run_id="also-good", path=ledger)
    assert [r["run_id"] for r in read_lines(ledger)] == ["good", "also-good"]


def test_a_missing_ledger_reads_as_empty(tmp_path):
    assert read_lines(tmp_path / "nope.jsonl") == []
    assert summarize(path=tmp_path / "nope.jsonl")["runs"] == 0


def test_a_write_that_fails_does_not_raise(tmp_path):
    """Bookkeeping must never fail the run it is keeping books on."""
    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory", encoding="utf-8")
    assert record_run([entry()], path=blocked / "spend.jsonl") is False


# -- what gets reported -----------------------------------------------------

def test_summarize_totals_and_breaks_down_by_model(ledger):
    record_run([entry("openai:big", 0.09)], path=ledger)
    record_run([entry("openai:big", 0.01),
                entry("openai:small", 0.001)], path=ledger)
    summary = summarize(days=None, path=ledger)
    assert summary["runs"] == 2
    assert summary["cost"] == pytest.approx(0.101)
    assert [m["model"] for m in summary["models"]] == [
        "openai:big", "openai:small",
    ]
    assert summary["models"][0]["cost"] == pytest.approx(0.10)
    assert summary["models"][0]["runs"] == 2


def test_the_breakdown_is_what_makes_the_total_actionable(ledger):
    """Nine of eleven dollars on one model is the sentence that tells you
    to put a cheaper one behind it (D89)."""
    record_run([entry("openai:expensive", 9.0)], path=ledger)
    record_run([entry("openai:cheap", 2.0)], path=ledger)
    top = summarize(days=None, path=ledger)["models"][0]
    assert top["model"] == "openai:expensive"


def test_an_unpriced_member_is_flagged_not_counted_as_zero(ledger):
    record_run([entry("gguf:local", None), entry("openai:x", 0.5)],
               path=ledger)
    models = {m["model"]: m for m in summarize(days=None, path=ledger)["models"]}
    assert models["gguf:local"]["unpriced"] is True
    assert models["gguf:local"]["cost"] == 0.0
    assert models["openai:x"]["unpriced"] is False


def test_a_window_excludes_older_runs(ledger):
    record_run([entry(cost=5.0)], path=ledger)
    # Rewrite the one line as if it were a month old.
    line = json.loads(ledger.read_text(encoding="utf-8").strip())
    line["ts"] = "2020-01-01T00:00:00+00:00"
    ledger.write_text(json.dumps(line) + "\n", encoding="utf-8")
    record_run([entry(cost=1.0)], path=ledger)
    assert summarize(days=7, path=ledger)["cost"] == pytest.approx(1.0)
    assert summarize(days=None, path=ledger)["cost"] == pytest.approx(6.0)


def test_describe_summary_never_uses_scientific_notation(ledger):
    record_run([entry(cost=0.0000075)], path=ledger)
    text = describe_summary(summarize(days=None, path=ledger))
    assert "e-" not in text
    assert "1 run(s)" in text


def test_describe_summary_says_so_when_there_is_nothing(tmp_path):
    text = describe_summary(summarize(days=7, path=tmp_path / "none.jsonl"))
    assert text == "No priced runs in the last 7 day(s)."


# -- what the engine reports ------------------------------------------------

def handle(name="m", **extra):
    h = {"backend": "openai", "model": object(), "model_alias": name}
    h.update(extra)
    return h


def priced(**extra):
    return handle(pricing={"input": 1e-6, "output": 2e-6, "source": "t"},
                  **extra)


class _Client:
    """A client whose one response is a fixed number of deltas."""

    def __init__(self, deltas=3):
        self.deltas = deltas

    def create_chat_completion(self, messages, stream=True, **kw):
        for _ in range(self.deltas):
            yield {"choices": [{"delta": {"content": "x"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def test_the_engine_reports_only_models_it_reached():
    """A chain of two that never needed its fallback must not report the
    fallback at zero -- that reads as tried and free."""
    eng = GraphEngine(build_chain(priced(name="a", model=_Client()),
                                  priced(name="b", model=_Client())))
    list(eng.stream_response({}))
    report = eng.spend_report()
    assert [e["model"] for e in report] == ["openai:a"]
    assert report[0]["requests"] == 1
    assert report[0]["output_tokens"] == 3
    assert report[0]["cost"] > 0


def test_spend_is_attributed_to_the_model_that_spent_it():
    """A run that switched models must not bill the whole thing to
    whichever one happened to finish it."""
    eng = GraphEngine(build_chain(priced(name="a", model=_Client(deltas=2)),
                                  priced(name="b", model=_Client(deltas=5))))
    list(eng.stream_response({}))
    eng.advance_model()
    list(eng.stream_response({}))
    report = {e["model"]: e for e in eng.spend_report()}
    assert report["openai:a"]["output_tokens"] == 2
    assert report["openai:b"]["output_tokens"] == 5
    assert report["openai:a"]["cost"] != report["openai:b"]["cost"]


def test_an_unpriced_model_reports_tokens_and_no_cost():
    eng = GraphEngine(handle(model=_Client()))
    list(eng.stream_response({}))
    report = eng.spend_report()[0]
    assert report["output_tokens"] == 3
    assert report["cost"] is None       # not 0.0 -- nobody quoted


def test_a_failed_request_still_reports_its_prefill():
    """D15: a failed request is still a request, and it was prefilled."""
    class _Dead:
        def create_chat_completion(self, *a, **kw):
            raise RuntimeError("model not found")

    eng = GraphEngine(priced(model=_Dead()), usage_limits=UsageLimits())
    with pytest.raises(RuntimeError):
        list(eng.stream_response({}))
    report = eng.spend_report()[0]
    assert report["requests"] == 1
    assert report["input_tokens"] > 0
    assert report["cost"] > 0
