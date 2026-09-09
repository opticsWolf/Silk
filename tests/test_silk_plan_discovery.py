# -*- coding: utf-8 -*-
"""Finding a plan without knowing who wrote it (T4, D63/D66).

Everything above the task store went backend-blind with D66 — the tools,
the Plan Viewer, the sign-off flow, the D58 hub all speak one protocol.
Discovery did not: a plan was a `plan-*.db`, the Task node's dropdown was
a list of those files, and that single fact is why the ledger backend
stayed opt-in per process long after the store itself was finished.

The answer here is the cheapest one available: **the extension names the
backend.** Nothing new is stored, nothing has to be kept in step, and a
row that came out of a scan can be reopened by reading its own path. What
is pinned below is that both kinds show up in one list, that each opens
through the store that wrote it, and that a root's *memory* ledger is not
offered as one of its plans.
"""

from __future__ import annotations


import pytest


from silk.functions import ledger as ledger_mod
from silk.functions import plan_discovery as pd
from silk.functions.task_store import PlanRef, SqliteTaskStore

needs_ledger = pytest.mark.skipif(
    not ledger_mod.available(),
    reason=f"the {ledger_mod.DISTRIBUTION} extra is not installed",
)


@pytest.fixture
def registry():
    reg = ledger_mod.LedgerRegistry()
    yield reg
    reg.close_all()


def _sqlite_plan(root, goal="sqlite goal"):
    store = SqliteTaskStore(root)
    store.start(goal=goal, acceptance=["a"], tasks=[{"title": "t"}])
    return store


def test_the_extension_is_the_backend():
    assert pd.backend_of("/x/plan-1.db") == pd.SQLITE
    assert pd.backend_of("/x/ledger.macrame") == pd.LEDGER
    # Case is not a distinction anyone should have to think about.
    assert pd.backend_of("/x/PLAN.MACRAME") == pd.LEDGER


def test_a_sqlite_plan_is_found_and_labelled(tmp_path):
    _sqlite_plan(tmp_path)

    rows = pd.scan_all(tmp_path)

    assert len(rows) == 1
    assert rows[0]["backend"] == pd.SQLITE
    assert rows[0]["goal"] == "sqlite goal"


def test_a_root_with_nothing_in_it_scans_empty(tmp_path):
    assert pd.scan_all(tmp_path) == []


@needs_ledger
def test_both_backends_appear_in_one_list(tmp_path, registry):
    _sqlite_plan(tmp_path)
    ledger_mod.TaskLedger(tmp_path, registry=registry).start(
        goal="ledger goal", acceptance=["a"], tasks=[{"title": "t"}])

    rows = pd.scan_all(tmp_path, registry=registry)

    assert {r["backend"] for r in rows} == {pd.SQLITE, pd.LEDGER}
    assert {r["goal"] for r in rows} == {"sqlite goal", "ledger goal"}
    # Newest first, the ordering every consumer of a scan assumes.
    assert rows == sorted(rows, key=lambda r: r["mtime"], reverse=True)


@needs_ledger
def test_the_history_ledger_is_not_offered_as_a_plan(tmp_path, registry):
    """It is the root's memory (§17) — a different thing under a right name."""
    history = ledger_mod.HistoryLedger(tmp_path, registry=registry)
    history.start_run("r1", agent="a")
    history.record_turn("r1", index=0, role="user", text="hello")

    assert pd.ledger_files(tmp_path) == []
    assert pd.scan_all(tmp_path, registry=registry) == []


@needs_ledger
def test_a_scanned_row_reopens_through_the_store_that_wrote_it(
    tmp_path, registry,
):
    ledger_mod.TaskLedger(tmp_path, registry=registry).start(
        goal="ledger goal", acceptance=["a"], tasks=[{"title": "t"}])

    row = pd.scan_all(tmp_path, registry=registry)[0]
    plan = pd.load_plan(row, registry=registry)

    assert plan is not None and plan.goal.text == "ledger goal"


@needs_ledger
def test_a_plan_ref_naming_a_ledger_file_opens_a_ledger(tmp_path, registry):
    """A reference is a file name; nothing else has to travel with it."""
    path = ledger_mod.ledger_path(tmp_path)
    ledger_mod.TaskLedger(tmp_path, db_path=path, registry=registry).start(
        goal="ledger goal", acceptance=["a"], tasks=[{"title": "t"}])

    store = PlanRef(root=str(tmp_path), db_path=str(path)).store(
        registry=registry)

    assert isinstance(store, ledger_mod.TaskLedger)
    assert store.load().goal.text == "ledger goal"


def test_a_plan_ref_naming_a_db_file_still_opens_sqlite(tmp_path):
    store = _sqlite_plan(tmp_path)
    path = store._locate_db()

    reopened = PlanRef(root=str(tmp_path), db_path=str(path)).store()

    assert isinstance(reopened, SqliteTaskStore)
    assert reopened.load().goal.text == "sqlite goal"


@needs_ledger
def test_a_named_plan_outranks_the_process_backend(tmp_path, registry,
                                                   monkeypatch):
    """Opening "the plan I was given" on some other backend is opening the
    wrong file, or the right file with the wrong reader."""
    store = _sqlite_plan(tmp_path)
    path = store._locate_db()
    monkeypatch.setenv(ledger_mod.BACKEND_ENV, ledger_mod.BACKEND_LEDGER)

    opened = ledger_mod.open_task_store(
        tmp_path, plan={"root": str(tmp_path), "db_path": str(path)},
        registry=registry)

    assert isinstance(opened, SqliteTaskStore)
    assert opened.load().goal.text == "sqlite goal"


def test_an_unreadable_ledger_is_a_row_with_an_error_not_a_missing_lane(
    tmp_path,
):
    """A board that quietly shows one lane fewer is worse than one that
    shows a lane saying what is wrong."""
    (tmp_path / "ledger.macrame").write_bytes(b"not a ledger at all")

    rows = pd.scan_all(tmp_path)

    if not ledger_mod.available():
        # Nothing can be read at all; the log line is the whole answer.
        assert rows == []
    else:
        assert len(rows) == 1 and rows[0].get("error")
        assert rows[0]["backend"] == pd.LEDGER
