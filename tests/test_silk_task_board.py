# -*- coding: utf-8 -*-
"""The multi-agent board projection (D58, D60(2), D60(3)).

Independent top-level agents share no event port, so the only place that
sees all of them is the store they all write to. These tests pin what the
projection has to get right for that to be usable: every plan under every
root exactly once, ``claimed_by`` surfaced per task, and an unreadable
plan reported rather than silently dropped.
"""
from __future__ import annotations

import time

from silk.functions.stream_events import EventType
from silk.functions.task_board import (
    LANES, PendingDecisions, actors_of, board, render_board, scan_roots,
)
from silk.functions.task_store import SqliteTaskStore


def _plan(root, goal, db_path, tasks=("step",)):
    store = SqliteTaskStore(root, db_path=db_path)
    store.start(goal=goal, acceptance=["ok"],
                tasks=[{"title": t} for t in tasks])
    return store


# ── scanning across roots ────────────────────────────────────────────────


def test_a_board_gathers_every_root(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir(), two.mkdir()
    _plan(one, "first", one / "plan-a.db")
    time.sleep(0.01)
    _plan(two, "second", two / "plan-b.db")

    rows = scan_roots([one, two])
    assert [r["goal"] for r in rows] == ["second", "first"], "newest first"


def test_overlapping_roots_do_not_double_a_plan(tmp_path):
    inner = tmp_path / "inner"
    inner.mkdir()
    _plan(inner, "shared", inner / "plan-a.db")

    rows = scan_roots([tmp_path, inner])
    assert len(rows) == 1, (
        "a graph may wire a project and a subdirectory of it; the same "
        "plan reached twice is one lane, not two"
    )


def test_an_unreachable_root_costs_one_lane_not_the_board(tmp_path):
    _plan(tmp_path, "real", tmp_path / "plan-a.db")
    rows = scan_roots([tmp_path, tmp_path / "does-not-exist", "", None])
    assert [r["goal"] for r in rows] == ["real"]


# ── the projection ───────────────────────────────────────────────────────


def test_claimed_by_reaches_the_board(tmp_path):
    store = _plan(tmp_path, "goal", tmp_path / "plan-a.db",
                  tasks=("write", "review"))
    plan = store.load()
    store.claim_task(task_id=plan.tasks[0].id, actor="agent-a")

    data = board(scan_roots([tmp_path]))
    lane = data["plans"][0]["lanes"]
    claimed = [t for tasks in lane.values() for t in tasks if t["actor"]]
    assert [t["actor"] for t in claimed] == ["agent-a"], (
        "the schema has carried claimed_by all along and no view showed "
        "it — that is what made 'who is doing what' unanswerable"
    )
    assert data["actors"] == ["agent-a"]


def test_tasks_are_grouped_by_lane(tmp_path):
    store = _plan(tmp_path, "goal", tmp_path / "plan-a.db",
                  tasks=("a", "b", "c"))
    plan = store.load()
    store.update_task(task_id=plan.tasks[0].id, status="in_progress")
    store.complete_task(task_id=plan.tasks[1].id, rationale="done")

    lanes = board(scan_roots([tmp_path]))["plans"][0]["lanes"]
    assert set(lanes) >= set(LANES)
    assert [t["title"] for t in lanes["in_progress"]] == ["a"]
    assert [t["title"] for t in lanes["done"]] == ["b"]
    assert [t["title"] for t in lanes["pending"]] == ["c"]


def test_the_board_counts_across_plans(tmp_path):
    _plan(tmp_path, "one", tmp_path / "plan-a.db", tasks=("x", "y"))
    _plan(tmp_path, "two", tmp_path / "plan-b.db", tasks=("z",))
    data = board(scan_roots([tmp_path]))
    assert data["plan_count"] == 2 and data["open_tasks"] == 3


def test_an_unreadable_plan_is_a_lane_with_an_error(tmp_path):
    _plan(tmp_path, "good", tmp_path / "plan-a.db")
    (tmp_path / "plan-bad.db").write_text("not a database", encoding="utf-8")

    data = board(scan_roots([tmp_path]))
    assert data["plan_count"] == 2, (
        "a board that hides the plan it could not read lies about how "
        "many plans there are"
    )
    broken = [p for p in data["plans"] if p.get("error")]
    assert len(broken) == 1 and broken[0]["lanes"]["pending"] == []


def test_an_empty_board_says_what_to_wire(tmp_path):
    assert "root_paths" in render_board(board(scan_roots([tmp_path])))


def test_rendering_names_the_agents_and_the_lanes(tmp_path):
    store = _plan(tmp_path, "ship it", tmp_path / "plan-a.db",
                  tasks=("write", "review"))
    plan = store.load()
    store.claim_task(task_id=plan.tasks[0].id, actor="agent-a")

    text = render_board(board(scan_roots([tmp_path])))
    assert "ship it" in text and "agent-a" in text and "pending" in text


def test_actors_are_first_seen_order_not_alphabetical(tmp_path):
    store = _plan(tmp_path, "goal", tmp_path / "plan-a.db",
                  tasks=("first", "second"))
    plan = store.load()
    store.claim_task(task_id=plan.tasks[0].id, actor="zeta")
    store.claim_task(task_id=plan.tasks[1].id, actor="alpha")

    assert actors_of(SqliteTaskStore(tmp_path,
                                     db_path=tmp_path / "plan-a.db").load()) == [
        "zeta", "alpha"
    ]


# ── pending decisions: counted here, answered elsewhere ──────────────────


def _request(decision_id, run_id="run-1"):
    return {"type": EventType.DECISION_REQUEST.value,
            "decision_id": decision_id, "run_id": run_id}


def _response(decision_id, run_id="run-1"):
    return {"type": EventType.DECISION_RESPONSE.value,
            "decision_id": decision_id, "run_id": run_id, "approved": True}


def test_a_request_raises_the_count_and_an_answer_lowers_it():
    pending = PendingDecisions()
    assert pending.record(_request("d1")) and pending.count == 1
    assert pending.record(_response("d1")) and pending.count == 0


def test_two_agents_waiting_count_as_two():
    pending = PendingDecisions()
    pending.record(_request("d1", "run-a"))
    pending.record(_request("d2", "run-b"))
    assert pending.count == 2 and pending.waiting() == ["d1", "d2"]


def test_the_same_request_seen_twice_counts_once():
    pending = PendingDecisions()
    pending.record(_request("d1"))
    assert not pending.record(_request("d1")), "a re-delivered preview"
    assert pending.count == 1


def test_a_finished_run_clears_what_it_never_answered():
    pending = PendingDecisions()
    pending.record(_request("d1", "run-a"))
    pending.record(_request("d2", "run-b"))
    pending.record({"type": EventType.RUN_FINISHED.value, "run_id": "run-a"})
    assert pending.waiting() == ["d2"], (
        "a timed-out or cancelled request would otherwise pin the count "
        "above zero for the life of the graph"
    )


def test_unrelated_events_are_ignored():
    pending = PendingDecisions()
    assert not pending.record({"type": EventType.TOOL_CALL.value})
    assert not pending.record("not an event")
    assert not pending.record(None)
    assert pending.count == 0
