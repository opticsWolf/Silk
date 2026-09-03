# -*- coding: utf-8 -*-
"""Finding plans without knowing which backend wrote them (T4, D63/D66).

Plan discovery has been file-shaped since D23: a plan is a ``plan-*.db``
under a sandbox root, the newest one wins when nobody names a file, and
the Task node's dropdown and the Task Hub's board are both lists of such
files. That is why the ledger backend is still opt-in per process even
though everything *above* the store has been backend-blind since D66 --
discovery was the one layer that still knew what SQLite looked like.

So discovery becomes backend-blind too, and by the cheapest possible
means: **the file extension names the backend.** ``plan-*.db`` is a
SQLite plan, ``ledger.macrame`` or ``plan-*.macrame`` is a ledger one.
Nothing new has to be stored, kept in step, or migrated, and a row that
came out of a scan can be reopened by looking at its own path.

`history.macrame` is deliberately not a plan: it is the same root's
*memory*, written every turn (§17), and offering it in a plan dropdown
would be offering a different thing under the right name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from weave.logger import get_logger

from .task_store import SqliteTaskStore, open_task_ids

log = get_logger("SilkPlans")

#: Backend names, as they appear on a scan row.
SQLITE = "sqlite"
LEDGER = "ledger"

#: The ledger file that is a root's memory, not its plan (§17).
HISTORY_STEM = "history"


def backend_of(path: "str | Path") -> str:
    """Which backend wrote the file at *path*, by its extension."""
    return LEDGER if str(path).lower().endswith(".macrame") else SQLITE


def ledger_files(root: "str | Path") -> list[Path]:
    """Task-ledger files under *root* -- never the history ledger."""
    base = Path(root).expanduser().resolve()
    found: list[Path] = []
    for directory in (base, base / ".silk" / "plan"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.macrame")):
            if path.stem == HISTORY_STEM:
                continue
            found.append(path)
    return found


def _ledger_rows(root: "str | Path", registry: Any = None) -> list[dict]:
    """Scan rows for the ledger plans under *root*.

    A ledger that will not open is a row carrying its error, exactly as a
    corrupt SQLite plan is: the board shows a lane that says what is
    wrong, rather than silently having one lane fewer.
    """
    from . import ledger as ledger_mod

    files = ledger_files(root)
    if not files:
        return []
    if not ledger_mod.available():
        # Say it once, and only when there is something to miss.
        log.info(
            "silk: %d ledger plan(s) under %s are not readable (%s)",
            len(files), root, ledger_mod.unavailable_reason(),
        )
        return []

    base = Path(root).expanduser().resolve()
    rows: list[dict] = []
    for path in files:
        row = {"db_path": str(path), "root": str(base), "label": path.stem,
               "plan_id": "", "goal": "", "updated_at": "", "open_tasks": 0,
               "tasks": 0, "mtime": path.stat().st_mtime, "backend": LEDGER}
        try:
            plan = ledger_mod.TaskLedger(
                base, db_path=path, registry=registry).load()
        except Exception as exc:      # noqa: BLE001 - a bad file is a row
            row["error"] = str(exc)
            rows.append(row)
            continue
        if plan is not None:
            row.update(
                plan_id=plan.plan_id,
                goal=plan.goal.text if plan.goal else "",
                updated_at=plan.updated_at,
                tasks=len(plan.tasks),
                open_tasks=len(open_task_ids(plan)),
            )
        rows.append(row)
    return rows


def scan_all(root: "str | Path", registry: Any = None) -> list[dict]:
    """Every plan under *root*, whichever backend wrote it -- newest first.

    Same row shape as ``SqliteTaskStore.scan_all`` with one field added:
    ``backend``. Callers that only pass rows back into :func:`open_plan`
    or a ``PlanRef`` never need to read it; it is there so a viewer can
    say what it is looking at.
    """
    rows = [dict(row, backend=SQLITE) for row in SqliteTaskStore.scan_all(root)]
    rows.extend(_ledger_rows(root, registry=registry))
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def open_store(root: "str | Path", db_path: "str | Path", *,
               registry: Any = None) -> Any:
    """The store for one discovered plan file, on the backend that wrote it."""
    if backend_of(db_path) == LEDGER:
        from . import ledger as ledger_mod

        if ledger_mod.available():
            return ledger_mod.TaskLedger(root, db_path=db_path,
                                         registry=registry)
        raise RuntimeError(
            f"'{db_path}' is a ledger plan and the ledger extra is not "
            f"installed ({ledger_mod.unavailable_reason()})."
        )
    return SqliteTaskStore(root, db_path=db_path)


def load_plan(row: dict, registry: Any = None) -> Optional[Any]:
    """The plan a scan row points at, or ``None`` if it will not open."""
    try:
        return open_store(row["root"], row["db_path"], registry=registry).load()
    except Exception:      # noqa: BLE001 - a bad file is an empty lane
        return None
