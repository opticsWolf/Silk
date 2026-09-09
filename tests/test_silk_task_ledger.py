# -*- coding: utf-8 -*-
"""The task store, backed by the Macrame ledger (D63-D66).

Two things are being pinned here, and they pull in opposite directions.

The first is *sameness*: every caller above the store -- the task tools,
the Plan Viewer, the sign-off flow, the D58 hub -- was written against
``SqliteTaskStore`` and must not be able to tell which backend answered.
So most of this file runs the identical sequence through both and
compares what comes back.

The second is the *reason for the swap*: an append-only ledger can answer
"what did the plan look like at 14:00" as a read. The SQLite store cannot,
and no amount of parity would give it that. Those tests deliberately have
no counterpart.
"""
from __future__ import annotations

import datetime as dt
import threading
import time

import pytest

from silk.functions import ledger as ledger_mod
from silk.functions.ledger import (
    DISTRIBUTION, EDGE_CLAIMED_BY, LedgerRegistry, TaskLedger, _agent_id,
    _plan_id, _related, _task_id,
)
from silk.functions.task_store import (
    Conflict, PLAN_SCHEMA_VERSION, SqliteTaskStore,
)

pytestmark = pytest.mark.skipif(
    not ledger_mod.available(),
    reason=f"the {DISTRIBUTION} extra is not installed",
)


@pytest.fixture
def registry():
    reg = LedgerRegistry()
    yield reg
    reg.close_all()


@pytest.fixture
def ledger(registry, tmp_path):
    return TaskLedger(tmp_path, registry=registry)


@pytest.fixture
def started(ledger):
    ledger.start(goal="ship the parser",
                 acceptance=["tests pass", "docs updated"],
                 tasks=[{"title": "lex"}, {"title": "parse"},
                        {"title": "emit", "note": "last"}])
    return ledger


def _both(tmp_path, registry):
    """One ledger and one SQLite store, each on its own empty root."""
    sqlite_root = tmp_path / "sqlite"
    sqlite_root.mkdir()
    return (TaskLedger(tmp_path, registry=registry),
            SqliteTaskStore(sqlite_root))


def _shape(plan):
    """Everything a caller above the store can actually observe."""
    return {
        "revision": plan.revision,
        "goal": (plan.goal.text, plan.goal.original_text,
                 list(plan.goal.acceptance), plan.goal.revised),
        "tasks": [(t.id, t.title, t.status, t.parent, t.order, t.note,
                   t.origin, t.claimed_by, t.done_by) for t in plan.tasks],
        "deviations": [(d.kind, d.target, d.rationale)
                       for d in plan.deviations],
    }


# ── parity: the same sequence through both backends ──────────────────────


def _sequence(store):
    """A plan's whole life, using only the shared protocol."""
    store.start(goal="ship the parser", acceptance=["tests pass"],
                tasks=[{"title": "lex"}, {"title": "parse"}],
                now="2026-09-02T10:00:00Z")
    store.claim_task(task_id="t1", actor="agent-a", now="2026-09-02T10:01:00Z")
    store.update_task(task_id="t1", status="in_progress", actor="agent-a",
                      now="2026-09-02T10:02:00Z")
    store.complete_task(task_id="t1", rationale="lexer landed", actor="agent-a",
                        now="2026-09-02T10:03:00Z")
    store.add_task(title="emit", rationale="parse needs a sink",
                   actor="agent-b", now="2026-09-02T10:04:00Z")
    store.rescope_task(task_id="t2", rationale="folded into emit",
                       actor="agent-b", now="2026-09-02T10:05:00Z")
    return store.revise_goal(new_text="ship the parser and the emitter",
                             acceptance_add=["emitter round-trips"],
                             rationale="scope grew", actor="agent-b",
                             now="2026-09-02T10:06:00Z")


def test_a_plans_whole_life_reads_the_same_from_both(tmp_path, registry):
    ledger, sqlite = _both(tmp_path, registry)
    assert _shape(_sequence(ledger)) == _shape(_sequence(sqlite)), (
        "the callers above the store were written against SQLite; if the "
        "shapes diverge, D66's 'swap the backend' is not a swap"
    )


def test_the_revision_log_reads_the_same_from_both(tmp_path, registry):
    ledger, sqlite = _both(tmp_path, registry)
    _sequence(ledger)
    _sequence(sqlite)

    def rows(store):
        return [(h["revision"], h["op"], h["actor"], h["rationale"])
                for h in store.history()]

    assert rows(ledger) == rows(sqlite)
    assert rows(ledger)[0][1] == "goal_revise", "newest first, as before"


def test_both_refuse_the_same_things_the_same_way(tmp_path, registry):
    ledger, sqlite = _both(tmp_path, registry)
    for store in (ledger, sqlite):
        store.start(goal="g", tasks=[{"title": "a"}])
        store.claim_task(task_id="t1", actor="first")

    def refusal(store, call):
        outcome = call(store)
        assert isinstance(outcome, Conflict), "a refusal is a Conflict"
        return (outcome.reason, outcome.target, outcome.current_revision)

    checks = (
        lambda s: s.claim_task(task_id="t1", actor="second"),
        lambda s: s.claim_task(task_id="nope", actor="second"),
        lambda s: s.update_task(task_id="t1", status="done"),
        lambda s: s.update_task(task_id="t1", status="dropped"),
        lambda s: s.update_task(task_id="t1", status="sideways"),
        lambda s: s.update_task(task_id="t1"),
        lambda s: s.complete_task(task_id="ghost", rationale="r"),
        lambda s: s.add_task(title="x", parent="ghost", rationale="r"),
        lambda s: s.revise_goal(rationale="r"),
    )
    for check in checks:
        assert refusal(ledger, check) == refusal(sqlite, check), (
            "a tool that special-cases a refusal message must not have to "
            "ask which backend it is talking to"
        )


def test_a_refusal_leaves_no_trace_in_the_ledger(started):
    before = started.load().revision
    assert isinstance(started.claim_task(task_id="ghost", actor="a"), Conflict)
    after = started.load()
    assert after.revision == before, (
        "D64: adjudication happens before the write, so a refused decision "
        "writes nothing -- not even a bumped revision"
    )
    assert not any(h["op"] == "task_claim" for h in started.history())


def test_the_schema_version_is_the_stores(ledger):
    assert ledger.schema_version == PLAN_SCHEMA_VERSION, (
        "one plan schema, two storage engines"
    )


def test_a_second_plan_start_is_refused(started):
    with pytest.raises(ValueError, match="already exists"):
        started.start(goal="another")


def test_an_empty_ledger_has_no_plan(ledger):
    assert ledger.load() is None and ledger.history() == []


# ── what only the ledger can do ──────────────────────────────────────────


def test_the_past_is_a_read_not_an_excavation(ledger):
    """D63's whole reason: the plan at a past instant, in one call."""
    ledger.start(goal="ship", tasks=[{"title": "lex"}])
    time.sleep(1.1)                     # the ledger's clock is second-grained
    mark = dt.datetime.now(dt.timezone.utc)
    time.sleep(1.1)
    ledger.claim_task(task_id="t1", actor="agent-a")
    ledger.complete_task(task_id="t1", rationale="done", actor="agent-a")

    now, past = ledger.load(), ledger.load(as_of=mark)
    assert (now.tasks[0].status, now.tasks[0].done_by) == ("done", "agent-a")
    assert (past.tasks[0].status, past.tasks[0].claimed_by) == ("pending", None)
    assert past.revision < now.revision, (
        "and the plan header travels with the tasks, or the read would be "
        "yesterday's tasks under today's revision number"
    )


def test_a_dropped_task_is_superseded_not_deleted(started):
    started.rescope_task(task_id="t2", rationale="not needed")
    plan = started.load()
    dropped = next(t for t in plan.tasks if t.id == "t2")
    assert dropped.status == "dropped", (
        "Doctrine V: the row stays and its status is superseded; a plan "
        "that forgets what it dropped cannot explain itself"
    )
    assert any(d.target == "t2" and d.rationale == "not needed"
               for d in plan.deviations)


def test_an_actor_is_a_node_not_a_string(started):
    """§17: 'what did this agent touch' should be a traversal."""
    started.claim_task(task_id="t1", actor="agent-a")
    db = started._db()
    claimed = _related(db, _task_id(started.load().plan_id, "t1"),
                       EDGE_CLAIMED_BY)
    assert [node.id for node in claimed] == [_agent_id("agent-a")]


def test_the_revision_log_hangs_off_the_plan(started):
    db = started._db()
    plan = started.load()
    revisions = _related(db, _plan_id(plan.plan_id), "HASREVISION")
    assert len(revisions) == 1, "plan_start is revision 0 and is logged"


# ── the decision lock (D64) ──────────────────────────────────────────────


def test_only_one_agent_wins_a_contested_claim(started):
    """The lock is for decisions, not for throughput."""
    outcomes: list = []
    ready = threading.Barrier(8)

    def claim(name):
        ready.wait()
        outcomes.append(started.claim_task(task_id="t1", actor=name))

    threads = [threading.Thread(target=claim, args=(f"agent-{i}",))
               for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    winners = [o for o in outcomes if not isinstance(o, Conflict)]
    assert len(winners) == 1, (
        "read-check-assert without a lock is a race; one process means "
        "one lock is complete prevention (D64)"
    )
    assert len(outcomes) == 8
    claimed = next(t for t in started.load().tasks if t.id == "t1").claimed_by
    assert claimed and all(claimed in o.reason
                           for o in outcomes if isinstance(o, Conflict)), (
        "and every loser is told who won"
    )


def test_the_lock_is_per_ledger_file(tmp_path, registry):
    first = TaskLedger(tmp_path / "a", registry=registry)
    second = TaskLedger(tmp_path / "b", registry=registry)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert first._lock() is not second._lock(), (
        "two roots are two decision domains; serialising them together "
        "would make one agent's claim wait on an unrelated plan"
    )
    assert first._lock() is TaskLedger(tmp_path / "a",
                                       registry=registry)._lock(), (
        "but two adapters on one file share the decision"
    )


def test_a_reclaim_by_the_same_actor_is_not_a_conflict(started):
    started.claim_task(task_id="t1", actor="agent-a")
    assert not isinstance(started.claim_task(task_id="t1", actor="agent-a"),
                          Conflict), (
        "a retrying agent is not a competitor"
    )


# ── two adapters, one file ───────────────────────────────────────────────


def test_a_second_adapter_sees_the_first_ones_writes(tmp_path, registry):
    first = TaskLedger(tmp_path, registry=registry)
    first.start(goal="ship", tasks=[{"title": "lex"}])
    second = TaskLedger(tmp_path, registry=registry)
    assert second.load().goal.text == "ship", (
        "the registry hands both the one write actor (D62), so this is "
        "one plan seen twice rather than two plans"
    )
    second.claim_task(task_id="t1", actor="agent-b")
    assert first.load().tasks[0].claimed_by == "agent-b"


def test_releasing_an_adapter_leaves_the_plan_readable(tmp_path, registry):
    ledger = TaskLedger(tmp_path, registry=registry)
    ledger.start(goal="ship")
    ledger.release()
    assert TaskLedger(tmp_path, registry=registry).load().goal.text == "ship"
