# -*- coding: utf-8 -*-
"""The durable event sink: one JSONL file per run (T7, D85).

Compaction drops turns to keep a run under its window, and what it drops
is unrecoverable -- which is what made a durable event log a precondition
rather than a convenience. The rule that matters most here is the one the
sink adds on top of the wire vocabulary: a file outlives the widget it
was mirrored to, so it keeps sizes and shapes and never text.

The rest is operational honesty: a sink that cannot write must not break
the run, a runaway loop must not fill a disk, and the directory must not
grow forever.
"""

from __future__ import annotations

import json

import pytest


from silk.functions.event_sink import (  # noqa: E402
    RunSink,
    redact,
)


def _lines(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ── redaction ────────────────────────────────────────────────────────────

def test_free_text_becomes_a_length():
    out = redact({"type": "delta", "delta": "hello", "run_id": "r1"})
    assert "delta" not in out
    assert out["delta_chars"] == 5
    assert out["run_id"] == "r1", "identity is metadata and stays"


def test_tool_arguments_keep_their_names_and_size_only():
    out = redact({"type": "tool_call", "tool_name": "write_file",
                  "tool_args": {"path": "/tmp/x", "content": "SECRET"}})
    assert out["tool_args_keys"] == ["content", "path"]
    assert out["tool_args_chars"] > 0
    assert "tool_args" not in out
    assert "SECRET" not in json.dumps(out)
    assert out["tool_name"] == "write_file", "which tool ran is the point"


def test_redaction_does_not_touch_the_caller_s_event():
    wire = {"type": "delta", "delta": "hello"}
    redact(wire)
    assert wire["delta"] == "hello", (
        "the same event is on its way to a widget that still wants it"
    )


def test_numbers_and_enums_travel_whole():
    out = redact({"type": "run_finished", "rounds": 3, "elapsed_s": 1.5,
                  "outcome": "completed"})
    assert out["rounds"] == 3 and out["outcome"] == "completed"


# ── writing ──────────────────────────────────────────────────────────────

def test_a_run_writes_one_file_of_one_line_per_event(tmp_path):
    sink = RunSink(tmp_path)
    assert sink.write({"type": "run_start", "run_id": "abcd1234", "seq": 0})
    assert sink.write({"type": "delta", "run_id": "abcd1234", "seq": 1,
                       "delta": "hi"})
    sink.close()

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    rows = _lines(files[0])
    assert [r["type"] for r in rows] == ["run_start", "delta"]
    assert rows[1]["delta_chars"] == 2 and "delta" not in rows[1]


def test_nothing_is_written_until_there_is_an_event(tmp_path):
    """An agent that never runs must not leave a file behind."""
    RunSink(tmp_path).close()
    assert list(tmp_path.glob("*.jsonl")) == []


def test_the_run_id_is_in_the_file_name(tmp_path):
    sink = RunSink(tmp_path)
    sink.write({"type": "run_start", "run_id": "deadbeef-1111"})
    sink.close()
    assert "deadbeef" in list(tmp_path.glob("*.jsonl"))[0].name


def test_a_cap_stops_the_writing_and_says_so(tmp_path):
    sink = RunSink(tmp_path, max_lines=2)
    for n in range(5):
        sink.write({"type": "delta", "run_id": "r", "seq": n, "delta": "x"})
    sink.close()

    rows = _lines(list(tmp_path.glob("*.jsonl"))[0])
    assert [r["type"] for r in rows] == ["delta", "delta", "sink_truncated"]
    assert rows[-1]["lines"] == 2


def test_the_truncation_line_is_written_once(tmp_path):
    sink = RunSink(tmp_path, max_lines=1)
    for n in range(6):
        sink.write({"type": "delta", "run_id": "r", "seq": n})
    sink.close()
    rows = _lines(list(tmp_path.glob("*.jsonl"))[0])
    assert sum(1 for r in rows if r["type"] == "sink_truncated") == 1


def test_only_the_newest_runs_survive(tmp_path):
    for n in range(5):
        sink = RunSink(tmp_path, keep_runs=3)
        sink.write({"type": "run_start", "run_id": f"run{n}"})
        sink.close()
    names = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert len(names) == 3, names
    assert "run4" in names[-1], "the newest run is the one that is kept"


def test_a_sink_that_cannot_write_does_not_break_the_run(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("I am a file", encoding="utf-8")

    sink = RunSink(blocked)
    assert sink.write({"type": "run_start", "run_id": "r"}) is False
    assert sink.write({"type": "delta", "run_id": "r"}) is False, (
        "a failed sink stays off rather than retrying every event"
    )
    sink.close()


def test_junk_is_declined_rather_than_written(tmp_path):
    sink = RunSink(tmp_path)
    assert sink.write(None) is False       # type: ignore[arg-type]
    assert sink.write({}) is False
    assert list(tmp_path.glob("*.jsonl")) == []


def test_the_sink_is_a_context_manager(tmp_path):
    with RunSink(tmp_path) as sink:
        sink.write({"type": "run_start", "run_id": "r"})
    assert len(list(tmp_path.glob("*.jsonl"))) == 1


@pytest.mark.parametrize("field", ["prompt", "system_prompt", "result",
                                   "message", "content", "goal"])
def test_every_named_content_field_is_a_length_on_disk(tmp_path, field):
    sink = RunSink(tmp_path)
    sink.write({"type": "x", "run_id": "r", field: "the quick brown fox"})
    sink.close()
    row = _lines(list(tmp_path.glob("*.jsonl"))[0])[0]
    assert row[f"{field}_chars"] == 19 and field not in row
