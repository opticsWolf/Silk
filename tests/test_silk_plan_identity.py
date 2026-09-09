# -*- coding: utf-8 -*-
"""Explicit plan identity: which plan, named rather than inferred (D23).

The failure this closes is quiet: the store picks the newest ``plan-*.db``
under a root, so two plans in one directory cross-discover and which one
an agent lands on depends on file timestamps (T4). A `PlanRef` names the
file. Discovery-by-newest stays, because it is also the mechanism by
which several agents deliberately share one plan -- so the tests pin both
behaviours, and the boundary between them.
"""
from __future__ import annotations

import time

from silk.functions.task_store import PlanRef, SqliteTaskStore
from silk.functions.tool_box import ToolBox
from silk.functions.tools.file_sandbox import FileToolSandbox
from silk.functions.tools.task_tracker import attach_task_tools


def _plan(root, goal, *, db_path=None):
    store = SqliteTaskStore(root, db_path=db_path)
    store.start(goal=goal, acceptance=["done"], tasks=[{"title": "step"}])
    return store


# ── the ambiguity this closes ────────────────────────────────────────────


def test_without_a_reference_the_newest_plan_wins(tmp_path):
    _plan(tmp_path, "first goal", db_path=tmp_path / "plan-a.db")
    time.sleep(0.01)
    _plan(tmp_path, "second goal", db_path=tmp_path / "plan-b.db")

    found = SqliteTaskStore(tmp_path).load()
    assert found is not None and found.goal.text == "second goal", (
        "the old behaviour: whichever plan file was touched last"
    )


def test_a_reference_names_one_plan_and_keeps_it(tmp_path):
    first = tmp_path / "plan-a.db"
    _plan(tmp_path, "first goal", db_path=first)
    time.sleep(0.01)
    _plan(tmp_path, "second goal", db_path=tmp_path / "plan-b.db")

    pinned = PlanRef(root=str(tmp_path), db_path=str(first)).store().load()
    assert pinned.goal.text == "first goal", (
        "a newer plan appearing beside it must not change which plan this is"
    )


def test_an_explicit_store_does_not_see_a_sibling_plan(tmp_path):
    _plan(tmp_path, "other", db_path=tmp_path / "plan-other.db")
    store = SqliteTaskStore(tmp_path, db_path=tmp_path / "plan-mine.db")
    assert store.load() is None, (
        "an explicit path that does not exist yet means 'no plan here', "
        "not 'use the neighbour'"
    )


def test_a_new_plan_lands_at_the_named_path(tmp_path):
    target = tmp_path / "plan-refactor.db"
    store = SqliteTaskStore(tmp_path, db_path=target)
    store.start(goal="rewrite the parser", acceptance=[], tasks=[])
    assert target.exists(), (
        "naming the file is what makes the plan findable again after a "
        "restart; a generated stem would not be"
    )


# ── PlanRef itself ───────────────────────────────────────────────────────


def test_a_reference_round_trips_through_plain_data():
    ref = PlanRef(root="/r", db_path="/r/plan-x.db", plan_id="abc", label="x")
    assert PlanRef.coerce(ref.to_dict()) == ref


def test_a_bare_path_coerces_to_a_root_only_reference():
    ref = PlanRef.coerce("/some/root")
    assert ref.root == "/some/root" and not ref.is_explicit


def test_nothing_coerces_to_nothing():
    assert PlanRef.coerce(None) is None
    assert PlanRef.coerce("") is None


def test_a_root_only_reference_still_finds_the_newest(tmp_path):
    _plan(tmp_path, "only goal", db_path=tmp_path / "plan-a.db")
    ref = PlanRef(root=str(tmp_path))
    assert not ref.is_explicit
    assert ref.store().load().goal.text == "only goal", (
        "several agents sharing one root is a legitimate way to share a "
        "plan, so the unnamed case must keep working"
    )


# ── scanning ─────────────────────────────────────────────────────────────


def test_scan_all_reports_every_plan_with_its_goal(tmp_path):
    _plan(tmp_path, "first goal", db_path=tmp_path / "plan-a.db")
    time.sleep(0.01)
    _plan(tmp_path, "second goal", db_path=tmp_path / "plan-b.db")

    rows = SqliteTaskStore.scan_all(tmp_path)
    assert [row["label"] for row in rows] == ["plan-b", "plan-a"], "newest first"
    assert {row["goal"] for row in rows} == {"first goal", "second goal"}
    assert all(row["tasks"] == 1 and row["open_tasks"] == 1 for row in rows), (
        "a row has to say enough to choose by, not just a filename"
    )


def test_scan_all_finds_plans_in_the_fallback_directory(tmp_path):
    nested = tmp_path / ".silk" / "plan"
    nested.mkdir(parents=True)
    _plan(tmp_path, "hidden goal", db_path=nested / "plan-h.db")
    assert [row["goal"] for row in SqliteTaskStore.scan_all(tmp_path)] == ["hidden goal"]


def test_scan_all_is_empty_on_a_directory_with_no_plans(tmp_path):
    assert SqliteTaskStore.scan_all(tmp_path) == []


def test_scan_all_reports_a_broken_file_rather_than_raising(tmp_path):
    (tmp_path / "plan-broken.db").write_text("not a database", encoding="utf-8")
    rows = SqliteTaskStore.scan_all(tmp_path)
    assert len(rows) == 1 and "error" in rows[0], (
        "one unreadable plan must not hide the readable ones"
    )


# ── the tools take the reference ─────────────────────────────────────────


def test_task_tools_use_the_referenced_plan(tmp_path):
    first = tmp_path / "plan-a.db"
    _plan(tmp_path, "first goal", db_path=first)
    time.sleep(0.01)
    _plan(tmp_path, "second goal", db_path=tmp_path / "plan-b.db")

    box = ToolBox()
    sandbox = FileToolSandbox(root_dir=str(tmp_path), allowed_paths=[str(tmp_path)])
    attach_task_tools(box, sandbox, plan=PlanRef(root=str(tmp_path),
                                                 db_path=str(first)))
    assert box._task_store.load().goal.text == "first goal"


def test_task_tools_without_a_reference_keep_discovering(tmp_path):
    _plan(tmp_path, "discovered goal", db_path=tmp_path / "plan-a.db")
    box = ToolBox()
    sandbox = FileToolSandbox(root_dir=str(tmp_path), allowed_paths=[str(tmp_path)])
    attach_task_tools(box, sandbox)
    assert box._task_store.load().goal.text == "discovered goal"


def test_a_reference_with_only_a_root_roots_the_store_there(tmp_path):
    _plan(tmp_path, "rooted goal", db_path=tmp_path / "plan-a.db")
    box = ToolBox()
    sandbox = FileToolSandbox(root_dir=str(tmp_path / "elsewhere"),
                              allowed_paths=[str(tmp_path)])
    attach_task_tools(box, sandbox, plan=PlanRef(root=str(tmp_path)))
    assert box._task_store.load().goal.text == "rooted goal", (
        "the reference decides where to look, not the sandbox"
    )
