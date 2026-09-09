# -*- coding: utf-8 -*-
"""The spill hook (spec D41 option A, D57).

The claim under test is narrow and load-bearing: a big result is replaced
**before it is appended**, so history stays append-only. Everything else
here defends the part that makes it usable rather than merely smaller --
the model is told where the rest went, in a path it can actually open, and
a delegation fan-out keeps its per-worker framing instead of becoming one
truncated blob.

The failure direction matters too. Every way spilling can fail -- no
sandbox, an unwritable directory, a result shape nobody anticipated --
leaves the result **inline and whole**, which is exactly the behaviour of
not having the hook at all.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel


from silk.functions.hook_catalog import (  # noqa: E402
    attach_catalog_hooks,
)
from silk.functions.hooks import (  # noqa: E402
    HOOK_WRAP_TOOL_EXECUTE,
)
from silk.functions.spill import (  # noqa: E402
    SPILL_DIR,
    SpillWriter,
    attach_spill_hook,
    preview,
    sweep,
)
from silk.functions.tool_box import ToolBox  # noqa: E402

BIG = "x" * 200 + "\n" + "middle " * 500 + "\nTHE CONCLUSION"


class _Result(BaseModel):
    ok: bool = True
    worker: str = ""
    answer: str | None = None


class _Fanout(BaseModel):
    ok: bool = True
    results: list[_Result] = []


def _box(tmp_path, payload, *, name="delegate_parallel", **kw):
    box = ToolBox(None, {"agent_id": "ag"})

    @box.register(name, "delegates", risk="medium")
    def _tool(_pool, _session, **_kw):
        return payload

    entry = attach_spill_hook(box, SimpleNamespace(root_dir=str(tmp_path)),
                              tools=(name,), **kw)
    return box, entry


def _call(box, name, args=None):
    tc = SimpleNamespace(id="c1", function=SimpleNamespace(
        name=name, arguments=json.dumps(args or {})))
    return asyncio.run(box.execute_tool_calls_async([tc]))[0]["content"]


def _spilled(tmp_path) -> list[Path]:
    return sorted((tmp_path / SPILL_DIR).glob("*.md"))


# -- the preview -----------------------------------------------------------


def test_a_preview_keeps_both_ends_and_says_where_the_rest_went():
    text = preview(BIG, "notes/big.md", head=100, tail=40)
    assert text.startswith("x" * 50)
    assert text.endswith("THE CONCLUSION"), (
        "the tail carries the conclusion, which is the half a summary loses"
    )
    assert "notes/big.md" in text and "characters omitted" in text
    assert len(text) < len(BIG)


def test_a_preview_can_drop_the_tail_entirely():
    text = preview(BIG, "p.md", head=50, tail=0)
    assert "THE CONCLUSION" not in text and "p.md" in text


# -- the writer ------------------------------------------------------------


def test_the_writer_numbers_within_a_run(tmp_path):
    writer = SpillWriter(tmp_path)
    first = writer.write("a", tool_name="delegate", label="alice")
    second = writer.write("b", tool_name="delegate", label="bob")
    assert first != second, "two spills in one batch must not collide"
    assert "alice" in first.name and "bob" in second.name
    assert writer.written == [first, second]


def test_paths_are_relative_to_the_sandbox_root(tmp_path):
    writer = SpillWriter(tmp_path)
    path = writer.write("a", tool_name="t")
    shown = writer.relative(path)
    assert not Path(shown).is_absolute() and shown.startswith(SPILL_DIR)
    assert (tmp_path / shown).is_file(), "the model must be able to open it"


def test_a_writer_without_a_root_writes_nothing():
    writer = SpillWriter(None)
    assert writer.available is False
    assert writer.write("a", tool_name="t") is None


# -- the hook --------------------------------------------------------------


def test_a_small_result_is_left_exactly_alone(tmp_path):
    box, _entry = _box(tmp_path, "short and sweet", name="delegate")
    assert _call(box, "delegate") == "short and sweet"
    assert not (tmp_path / SPILL_DIR).exists(), "nothing to spill, nothing written"


def test_a_big_string_result_is_spilled(tmp_path):
    box, _entry = _box(tmp_path, BIG, name="delegate", threshold=500,
                       head=100, tail=40)
    content = _call(box, "delegate")

    assert len(content) < len(BIG)
    files = _spilled(tmp_path)
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == BIG, "the file is complete"
    assert files[0].name in content or SPILL_DIR in content


def test_a_fan_out_spills_each_worker_and_keeps_the_framing(tmp_path):
    """D57: the per-worker framing is small and worth keeping."""
    payload = _Fanout(results=[
        _Result(worker="alice", answer=BIG),
        _Result(worker="bob", answer="brief"),
        _Result(worker="carol", answer=BIG),
    ])
    box, _entry = _box(tmp_path, payload, threshold=500, head=80, tail=30)
    body = json.loads(_call(box, "delegate_parallel"))

    assert [r["worker"] for r in body["results"]] == ["alice", "bob", "carol"]
    assert body["ok"] is True
    assert body["results"][1]["answer"] == "brief", "a short answer is untouched"
    assert len(_spilled(tmp_path)) == 2, "one file per oversized worker answer"
    for spilled in (body["results"][0], body["results"][2]):
        assert "characters omitted" in spilled["answer"]
        assert spilled["answer"].endswith("THE CONCLUSION")

    names = {p.name for p in _spilled(tmp_path)}
    assert any("alice" in n for n in names) and any("carol" in n for n in names)


def test_a_single_delegation_spills_its_answer(tmp_path):
    box, _entry = _box(tmp_path, _Result(worker="alice", answer=BIG),
                       name="delegate", threshold=500)
    body = json.loads(_call(box, "delegate"))
    assert body["worker"] == "alice" and body["ok"] is True
    assert "characters omitted" in body["answer"]


def test_a_structured_result_with_no_answer_field_spills_whole(tmp_path):
    class _Odd(BaseModel):
        blob: str

    box, _entry = _box(tmp_path, _Odd(blob=BIG), name="delegate", threshold=500)
    content = _call(box, "delegate")
    assert "characters omitted" in content and len(_spilled(tmp_path)) == 1


# -- every failure leaves the result whole ---------------------------------


def test_no_sandbox_means_no_hook_at_all():
    box = ToolBox(None, {"agent_id": "ag"})

    @box.register("delegate", "delegates", risk="medium")
    def _tool(_pool, _session, **_kw):
        return BIG

    assert attach_spill_hook(box, None) is None, (
        "spilling where the agent cannot read back is worse than not spilling"
    )
    assert box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE) == []
    assert _call(box, "delegate") == BIG


def test_an_unwritable_spill_directory_leaves_the_result_inline(tmp_path):
    (tmp_path / SPILL_DIR).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / SPILL_DIR).write_text("not a directory", encoding="utf-8")

    box, entry = _box(tmp_path, BIG, name="delegate", threshold=500)
    assert entry is not None
    assert _call(box, "delegate") == BIG, (
        "a failed write must not cost the call its result"
    )


def test_the_hook_is_bound_to_the_tools_it_names(tmp_path):
    box = ToolBox(None, {"agent_id": "ag"})

    @box.register("delegate", "delegates", risk="medium")
    def _delegate(_pool, _session, **_kw):
        return BIG

    @box.register("read_file", "reads", risk="low")
    def _read(_pool, _session, **_kw):
        return BIG

    attach_spill_hook(box, SimpleNamespace(root_dir=str(tmp_path)),
                      threshold=500, tools=("delegate",))
    assert "characters omitted" in _call(box, "delegate")
    assert _call(box, "read_file") == BIG, "an unnamed tool is not spilled"


def test_an_empty_tool_list_spills_everything(tmp_path):
    box = ToolBox(None, {"agent_id": "ag"})

    @box.register("read_file", "reads", risk="low")
    def _read(_pool, _session, **_kw):
        return BIG

    attach_spill_hook(box, SimpleNamespace(root_dir=str(tmp_path)),
                      threshold=500, tools=())
    assert "characters omitted" in _call(box, "read_file")


# -- the catalog hook ------------------------------------------------------


def test_spill_as_a_catalog_hook(tmp_path):
    box = ToolBox(None, {"agent_id": "ag"})

    @box.register("delegate_parallel", "fans out", risk="medium")
    def _tool(_pool, _session, **_kw):
        return _Fanout(results=[_Result(worker="alice", answer=BIG)])

    attach_catalog_hooks(
        box, SimpleNamespace(root_dir=str(tmp_path)), names=("spill",),
        configs={"spill": {"threshold": 500, "head": 60, "tail": 20,
                           "tools": "delegate_parallel"}},
    )
    body = json.loads(_call(box, "delegate_parallel"))
    assert "characters omitted" in body["results"][0]["answer"]
    assert len(_spilled(tmp_path)) == 1


# -- what happens to the files afterwards (§22 q4) -------------------------


def _aged(tmp_path, name, *, days, size=10):
    """A spill file with the writer's own naming shape, *days* old."""
    directory = tmp_path / SPILL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("x" * size, encoding="utf-8")
    old = time.time() - days * 86400
    os.utime(path, (old, old))
    return path


def test_sweeping_removes_what_is_older_than_the_window(tmp_path):
    stale = _aged(tmp_path, "20260101T000000-delegate-01.md", days=30)
    fresh = _aged(tmp_path, "20260101T000000-delegate-02.md", days=1)

    removed = sweep(tmp_path, retain_days=14)

    assert removed == [stale]
    assert not stale.exists() and fresh.exists()


def test_sweeping_never_touches_a_file_it_did_not_write(tmp_path):
    """The spill directory is inside the user's project, not ours alone."""
    _aged(tmp_path, "20260101T000000-delegate-01.md", days=30)
    theirs = _aged(tmp_path, "notes.md", days=400)
    also_theirs = _aged(tmp_path, "2026-01-01-meeting.md", days=400)

    sweep(tmp_path, retain_days=14)

    assert theirs.exists() and also_theirs.exists(), (
        "a name that is not the one write() produces is not ours to delete"
    )


def test_the_size_ceiling_takes_the_oldest_first(tmp_path):
    old = _aged(tmp_path, "20260101T000000-delegate-01.md", days=3, size=600)
    mid = _aged(tmp_path, "20260101T000000-delegate-02.md", days=2, size=600)
    new = _aged(tmp_path, "20260101T000000-delegate-03.md", days=1, size=600)

    removed = sweep(tmp_path, retain_days=14, retain_bytes=1300)

    assert removed == [old], "just enough to get under the ceiling"
    assert mid.exists() and new.exists()


def test_sweeping_a_directory_that_is_not_there_is_not_an_error(tmp_path):
    assert sweep(tmp_path / "nowhere") == []
    assert sweep(None) == []


def test_a_run_sweeps_on_the_way_in_not_on_the_way_out(tmp_path):
    """§22 q4: the file has to outlive the run that wrote it.

    The preview left in history names the path, and that history is
    persisted with the node -- so cleaning up at run end would turn every
    reference the model was given into a dangling one. Attaching is the
    safe moment: nothing live is holding a path yet.
    """
    stale = _aged(tmp_path, "20260101T000000-delegate-01.md", days=30)

    box, _entry = _box(tmp_path, _Fanout(results=[_Result(
        worker="alice", answer=BIG)]), threshold=500, retain_days=14)
    assert not stale.exists(), "swept when the hook attached"

    _call(box, "delegate_parallel")
    written = _spilled(tmp_path)
    assert len(written) == 1 and written[0].exists(), (
        "and this run's own file is still there when the run is over"
    )


def test_keeping_nothing_is_expressible(tmp_path):
    """``retain_days=0`` is a real answer: only this run's files survive."""
    old = _aged(tmp_path, "20260101T000000-delegate-01.md", days=0)
    assert sweep(tmp_path, retain_days=0) == [old]


def test_the_catalog_hook_carries_the_retention_settings(tmp_path):
    stale = _aged(tmp_path, "20260101T000000-delegate-01.md", days=10)
    box = ToolBox(None, {"agent_id": "ag"})

    @box.register("delegate_parallel", "fans out", risk="medium")
    def _tool(_pool, _session, **_kw):
        return _Fanout(results=[_Result(worker="alice", answer=BIG)])

    attach_catalog_hooks(
        box, SimpleNamespace(root_dir=str(tmp_path)), names=("spill",),
        configs={"spill": {"threshold": 500, "retain_days": 7,
                           "tools": "delegate_parallel"}},
    )
    assert not stale.exists(), "a preset can shorten the window"
