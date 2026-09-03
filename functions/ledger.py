# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The Macrame ledger seam: one handle per file, process-wide (spec D62).

Macrame is a bitemporal graph ledger with **one Write Actor per open
handle**. Two handles on one file is outside its contract, and the
library does not stop you -- opening the same path twice succeeds and
gives you two writers racing on one file. So the rule has to live
somewhere, and the only place it can be complete is the single place that
opens: this registry. Nodes, tools, the hub and the compactor never call
``Database.open`` themselves.

What the registry is:

- a process singleton mapping resolved path -> open handle, refcounted;
- the owner of ``close()``, which Macrame says has exactly one owner --
  plugin unload or graph close closes, never a node and never a run;
- the answer to "is this file already open here", which is what turns
  the Task Hub's file discovery (D58, D60(2)) into *discovery plus
  lookup* rather than a second open of a live ledger.

What it is not: a backend. ``macrame-db`` is an optional extra (D66).
Absent, :func:`available` is False and the caller falls back to
``SqliteTaskStore`` -- loudly, with one log line, never silently.

Edge-type note: Macrame validates edge types against ``[A-Z0-9]+``, so
the vocabulary here is ``CLAIMEDBY`` / ``SUBTASKOF`` / ``INRUN`` rather
than the underscored names §17 writes prose in.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Optional

from weave.logger import get_logger

log = get_logger("SilkLedger")

#: The optional extra that provides the backend (D66, G5).
DISTRIBUTION = "macrame-db"

#: Snapshot cadence for a Silk ledger. Silk's writes are small and its
#: processes are long-lived, so the library default is right; the field
#: exists so a test can ask for none without reaching into the registry.
DEFAULT_SNAPSHOT_ENTRIES: Optional[int] = None

_IMPORT_ERROR: Optional[BaseException] = None

try:                                    # pragma: no cover - import shape
    import macrame as _macrame
except BaseException as exc:            # noqa: BLE001 - a binary wheel can
    _macrame = None                     #   fail to load, not just be absent
    _IMPORT_ERROR = exc


def available() -> bool:
    """Whether the ledger backend can be used at all."""
    return _macrame is not None


def unavailable_reason() -> str:
    """Why it cannot, in a sentence a log line can carry."""
    if _macrame is not None:
        return ""
    detail = f" ({_IMPORT_ERROR})" if _IMPORT_ERROR else ""
    return (
        f"the {DISTRIBUTION} extra is not installed{detail}; Silk is using "
        f"the SQLite task store and keeping history in the node"
    )


class LedgerUnavailable(RuntimeError):
    """Raised when a ledger is asked for and the backend is not there."""


@dataclass
class _Entry:
    handle: Any
    refs: int
    path: str


class LedgerRegistry:
    """One open handle per ledger file, for the life of the process.

    Refcounted rather than opened-and-closed per user: several agents in
    one sandbox root share one task ledger, and closing it under the
    others because one run ended would take the write actor with it.

    Closing is deliberately *not* automatic when the count reaches zero.
    A graph that finishes a run and starts another would otherwise pay a
    full open per run, and Macrame's close writes a final snapshot --
    cheap once, wasteful in a loop. :meth:`close_all` at plugin unload or
    graph close is the owner, as D62 requires.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._open: dict[str, _Entry] = {}
        self._participant = None

    # ── the release protocol (spec D80) ───────────────────────────────

    def _register_for_shutdown(self) -> None:
        """Let the process let go of these handles before it hands off.

        One write actor per process is the whole of Macrame's concurrency
        model, so two processes on one file -- which is exactly what a
        relaunch produces if the parent still holds it -- breaks D64's
        earliest-`recorded_at` adjudication. Registering here rather than
        relying on GC is the point: interpreter teardown is not a
        guarantee, and a handle closed late loses its final snapshot.
        """
        if self._participant is not None:
            return
        try:
            from weave.engine.shutdown import (
                get_shutdown_registry, install_shutdown_handlers,
            )
        except Exception:  # noqa: BLE001 - a plugin must not need the host
            return
        install_shutdown_handlers()
        self._participant = get_shutdown_registry().register(
            "Macrame ledger handles",
            lambda timeout_s=0.0: (self.close_all(), True)[1],
            busy=lambda: (f"{len(self._open)} ledger file(s) open"
                          if self._open else None),
        )

    # ── opening ───────────────────────────────────────────────────────

    def acquire(self, path: str | os.PathLike, *,
                snapshot_every_entries: Optional[int] = DEFAULT_SNAPSHOT_ENTRIES,
                ) -> Any:
        """The shared handle for *path*, opening it once if needed.

        Raises :class:`LedgerUnavailable` when the extra is missing --
        the caller's cue to fall back (D66), not a crash to swallow.
        """
        if _macrame is None:
            raise LedgerUnavailable(unavailable_reason())

        key = self.key(path)
        with self._lock:
            entry = self._open.get(key)
            if entry is not None and not _is_closed(entry.handle):
                entry.refs += 1
                return entry.handle
            if entry is not None:
                # A handle that died under us (Macrame closes hard on some
                # errors). Reopening is right; pretending it is live is not.
                log.warning(f"Ledger handle for {key} was closed; reopening")
                self._open.pop(key, None)

            Path(key).parent.mkdir(parents=True, exist_ok=True)
            handle = _macrame.Database.open(
                key, snapshot_every_entries=snapshot_every_entries)
            self._open[key] = _Entry(handle=handle, refs=1, path=key)
            self._register_for_shutdown()
            log.debug(f"Opened ledger {key}")
            return handle

    def release(self, path: str | os.PathLike) -> int:
        """Give up one reference. The handle stays open (see the class doc)."""
        key = self.key(path)
        with self._lock:
            entry = self._open.get(key)
            if entry is None:
                return 0
            entry.refs = max(0, entry.refs - 1)
            return entry.refs

    # ── looking ───────────────────────────────────────────────────────

    def get(self, path: str | os.PathLike) -> Optional[Any]:
        """The handle for *path* if this process already has it open.

        This is the half D58's scan needs: files are found on disk, and
        a file that is already open is read through the handle that owns
        it rather than opened a second time.
        """
        with self._lock:
            entry = self._open.get(self.key(path))
        if entry is None or _is_closed(entry.handle):
            return None
        return entry.handle

    def is_open(self, path: str | os.PathLike) -> bool:
        return self.get(path) is not None

    def paths(self) -> list[str]:
        with self._lock:
            return sorted(self._open)

    def refs(self, path: str | os.PathLike) -> int:
        with self._lock:
            entry = self._open.get(self.key(path))
            return entry.refs if entry is not None else 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._open)

    # ── closing (the registry's job, nobody else's) ───────────────────

    def close(self, path: str | os.PathLike) -> bool:
        """Close one ledger regardless of refcount. Owner-only.

        Callers that merely finished with a ledger want :meth:`release`.
        This is for the owner -- graph close, plugin unload -- and it is
        blunt on purpose: a handle nobody closes loses its final
        snapshot.
        """
        key = self.key(path)
        with self._lock:
            entry = self._open.pop(key, None)
        if entry is None:
            return False
        _close(entry.handle, key)
        return True

    def close_all(self) -> int:
        """Close every ledger this process holds. Plugin unload calls this."""
        with self._lock:
            entries = list(self._open.values())
            self._open.clear()
        for entry in entries:
            _close(entry.handle, entry.path)
        return len(entries)

    # ── identity ──────────────────────────────────────────────────────

    @staticmethod
    def key(path: str | os.PathLike) -> str:
        """One canonical string per file.

        Two names for one file must be one entry, or the sole-writer rule
        is enforced against spellings rather than files -- which is no
        rule at all on a case-insensitive filesystem.
        """
        resolved = Path(path).expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:                 # pragma: no cover - unreachable path
            resolved = resolved.absolute()
        text = str(resolved)
        return os.path.normcase(text) if os.name == "nt" else text


def _is_closed(handle: Any) -> bool:
    """Whether a handle has already shut down. Defensive: the flag moved
    from method to property once, and a wrong answer here reopens a live
    ledger."""
    flag = getattr(handle, "is_closed", False)
    if callable(flag):
        try:
            return bool(flag())
        except Exception:               # noqa: BLE001
            return False
    return bool(flag)


def _close(handle: Any, key: str) -> None:
    try:
        handle.close()
        log.debug(f"Closed ledger {key}")
    except Exception as exc:            # noqa: BLE001 - closing twice, or a
        log.debug(f"Ledger {key} closed with: {exc}")   # dead actor


#: The process's registry -- the only thing that opens a ledger (D62).
REGISTRY = LedgerRegistry()


def ledger_path(root: str | os.PathLike, stem: str = "ledger") -> Path:
    """Where a sandbox root's task ledger lives.

    One ledger per sandbox root is the default placement (§17), which
    keeps T4/D58 file discovery working unchanged: a ledger is a file
    under a root like a plan is.
    """
    return Path(root).expanduser() / f"{stem}.macrame"


# ── The task-store protocol, on the ledger (D63-D66) ─────────────────────

#: Edge vocabulary. Macrame validates types against ``[A-Z0-9]+``, so the
#: underscored names §17 writes prose in are spelled solid here.
EDGE_HAS_TASK = "HASTASK"
EDGE_HAS_REVISION = "HASREVISION"
EDGE_SUBTASK_OF = "SUBTASKOF"
EDGE_CLAIMED_BY = "CLAIMEDBY"
EDGE_DONE_BY = "DONEBY"

#: The concept that says which plan this ledger's task lineage is. One
#: task ledger per sandbox root (§17 placement), so one head.
HEAD_ID = "silk:plan:head"


def _plan_id(pid: str) -> str:
    return f"plan:{pid}"


def _task_id(pid: str, tid: str) -> str:
    return f"task:{pid}:{tid}"


def _revision_id(pid: str, revision: int) -> str:
    return f"rev:{pid}:{revision}"


def _agent_id(actor: str) -> str:
    return f"agent:{actor}"


class TaskLedger:
    """``SqliteTaskStore``'s protocol, backed by the Macrame ledger (D66).

    Same methods, same ``Plan`` / ``Task`` / ``Conflict`` objects, same
    return shapes -- so the Plan Viewer, the task tools, the sign-off flow
    and the D58 hub do not know which backend they are talking to. What
    changes is underneath: a status transition supersedes rather than
    overwrites, so *what did the plan look like at 14:00* is a read rather
    than an archaeology project.

    **Writes.** Macrame's Write Actor serialises writes within the process
    already, so nothing here locks for throughput. The one lock is for
    *decisions* (D64): claim-a-task and advance-a-status-with-a-
    precondition are read-check-assert, and one process means one lock is
    complete prevention -- it is what ``BEGIN IMMEDIATE`` was doing
    cross-process for the old store. The lock is per ledger file, because
    two roots are two decision domains.

    **Deletion never happens** (Doctrine V): a dropped task is a superseded
    status, and the revision that dropped it stays readable.
    """

    #: One decision lock per ledger file, shared by every adapter on it.
    _locks: dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, root: str | os.PathLike, *,
                 db_path: str | os.PathLike | None = None,
                 direct_write: bool = True,
                 registry: Optional[LedgerRegistry] = None) -> None:
        from .task_store import PLAN_SCHEMA_VERSION  # local: keeps ledger.py
        self.schema_version = PLAN_SCHEMA_VERSION    # importable Qt-free

        self.root = Path(root).expanduser().resolve()
        self.direct_write = direct_write
        self._registry = registry if registry is not None else REGISTRY
        self._path = Path(db_path) if db_path else ledger_path(self.root)
        self._key = LedgerRegistry.key(self._path)
        self._pid: Optional[str] = None

    # ── plumbing ──────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    def _db(self) -> Any:
        return self._registry.acquire(self._path)

    def _lock(self) -> threading.RLock:
        with TaskLedger._locks_guard:
            lock = TaskLedger._locks.get(self._key)
            if lock is None:
                lock = threading.RLock()
                TaskLedger._locks[self._key] = lock
            return lock

    def release(self) -> None:
        """Give up this adapter's share of the handle. Never closes (D62)."""
        self._registry.release(self._path)

    # ── reading ───────────────────────────────────────────────────────

    def _head(self, db: Any) -> Optional[str]:
        """The plan id this ledger's task lineage points at."""
        if self._pid is not None:
            return self._pid
        node = _concept(db, HEAD_ID)
        if node is None:
            return None
        self._pid = str(_payload(node).get("plan_id") or "") or None
        return self._pid

    def load(self, as_of=None) -> "Any":
        """The plan as of *as_of* (a datetime), or as it stands now.

        The whole reason for D63: "what did the plan look like at 14:00"
        is one read against the same store, not a reconstruction from an
        audit trail. Passing *as_of* is the only difference between the
        two, and every caller that does not care sees today's behaviour.
        """
        from .task_store import Deviation, Goal, Plan, Task

        try:
            db = self._db()
        except LedgerUnavailable:
            return None
        pid = self._head(db)
        if pid is None:
            return None
        plan_node = _concept(db, _plan_id(pid), as_of)
        if plan_node is None:
            return None
        head = _payload(plan_node)
        goal = head.get("goal") or {}

        tasks = []
        for node in _related(db, _plan_id(pid), EDGE_HAS_TASK, as_of):
            body = _payload(node)
            if not body:
                continue
            tasks.append(Task(
                id=body.get("id", ""), title=node.title or "",
                status=body.get("status", "pending"), parent=body.get("parent"),
                order=int(body.get("order", 0)), note=body.get("note", ""),
                origin=body.get("origin", "added"),
                added_by=body.get("added_by", "agent"),
                claimed_by=body.get("claimed_by"), done_by=body.get("done_by"),
                created_at=body.get("created_at", ""),
                updated_at=body.get("updated_at", ""),
            ))
        tasks.sort(key=lambda t: (t.order, t.id))

        deviations = []
        for entry in self._revisions(db, pid, as_of):
            dev = entry.get("deviation")
            if not dev:
                continue
            deviations.append(Deviation(
                at=entry.get("at", ""), actor=entry.get("actor", ""),
                kind=dev.get("kind", ""), target=dev.get("target", ""),
                from_val=dev.get("from"), to_val=dev.get("to"),
                rationale=entry.get("rationale") or "",
            ))

        return Plan(
            plan_id=pid, created_at=head.get("created_at", ""),
            updated_at=head.get("updated_at", ""),
            revision=int(head.get("revision", 0)),
            goal=Goal(text=goal.get("text", ""),
                      original_text=goal.get("original_text", ""),
                      acceptance=list(goal.get("acceptance") or []),
                      revised=bool(goal.get("revised"))),
            tasks=tasks, deviations=deviations,
        )

    def _revisions(self, db: Any, pid: str, as_of=None) -> list[dict]:
        """The revision log, oldest first."""
        entries = []
        for node in _related(db, _plan_id(pid), EDGE_HAS_REVISION, as_of):
            body = _payload(node)
            if body:
                entries.append(body)
        entries.sort(key=lambda e: int(e.get("revision", 0)))
        return entries

    def history(self, limit: Optional[int] = None) -> list[dict]:
        """Recent revision-log entries, newest first -- the store's shape."""
        try:
            db = self._db()
        except LedgerUnavailable:
            return []
        pid = self._head(db)
        if pid is None:
            return []
        rows = [
            {"revision": entry.get("revision", 0), "at": entry.get("at", ""),
             "actor": entry.get("actor", ""), "op": entry.get("op", ""),
             "args": entry.get("args") or {},
             "rationale": entry.get("rationale")}
            for entry in reversed(self._revisions(db, pid))
        ]
        return rows[:limit] if limit else rows

    # ── writing ───────────────────────────────────────────────────────

    def start(self, *, goal: str, acceptance: Optional[list] = None,
              tasks: Optional[list] = None, actor: str = "agent",
              now: Optional[str] = None) -> "Any":
        """Create the plan and its initial decomposition.

        One ``write_bulk_atomic`` for the edges (D65): a plan either
        started or did not, and a half-started one would read as a plan
        with missing tasks rather than as no plan.
        """
        import secrets

        from .task_store import _now_iso

        now = now or _now_iso()
        with self._lock():
            if self.load() is not None:
                raise ValueError(
                    "A plan already exists; use task_add / task_update / … "
                    "instead of plan_start."
                )
            db = self._db()
            pid = secrets.token_hex(8)
            self._pid = pid
            specs = list(tasks or [])

            _put(db, _plan_id(pid), goal, {
                "plan_id": pid, "created_at": now, "updated_at": now,
                "revision": len(specs),
                "goal": {"text": goal, "original_text": goal,
                         "acceptance": list(acceptance or []), "revised": False},
            })
            edges = []
            for index, spec in enumerate(specs):
                tid = f"t{index + 1}"
                title = str(spec.get("title", "")).strip() or tid
                _put(db, _task_id(pid, tid), title, {
                    "id": tid, "status": "pending", "parent": spec.get("parent"),
                    "order": index, "note": str(spec.get("note", "")),
                    "origin": "initial", "added_by": actor, "claimed_by": None,
                    "done_by": None, "created_at": now, "updated_at": now,
                })
                edges.append(_edge(_plan_id(pid), _task_id(pid, tid),
                                   EDGE_HAS_TASK))
                if spec.get("parent"):
                    edges.append(_edge(_task_id(pid, tid),
                                       _task_id(pid, str(spec["parent"])),
                                       EDGE_SUBTASK_OF))
            if edges:
                db.write_bulk_atomic(edges)

            self._record(db, pid, actor=actor, now=now,
                         op="plan_start",
                         args={"goal": goal, "acceptance": list(acceptance or []),
                               "tasks": len(specs)})
            _put(db, HEAD_ID, "current plan", {"plan_id": pid})
            plan = self.load()
        return plan

    def _record(self, db: Any, pid: str, *, actor: str,
                now: str, op: str, args: dict,
                rationale: Optional[str] = None,
                deviation: Optional[dict] = None) -> None:
        """Append one revision-log entry. Never rewrites an earlier one.

        Entries are numbered by position in the log, not by the plan's
        revision -- the same thing the SQLite store's rowid was doing,
        and the number its ``history()`` has always returned.
        """
        revision = len(self._revisions(db, pid)) + 1
        rid = _revision_id(pid, revision)
        _put(db, rid, op, {
            "revision": revision, "at": now, "actor": actor, "op": op,
            "args": args, "rationale": rationale, "deviation": deviation,
        })
        db.assert_edge(_edge(_plan_id(pid), rid, EDGE_HAS_REVISION))

    def _commit(self, *, actor: str, now: str, op: str,
                rationale: Optional[str],
                work) -> "Any":
        """Read-check-assert under the decision lock (D64).

        *work(plan)* returns either a ``Conflict`` -- nothing is written --
        or ``{"op": …, "deviation": …, "writes": [callables]}``. The lock
        is what makes the precondition and the write one decision; the
        ledger's own actor is what makes the writes durable.
        """
        from .task_store import Conflict

        with self._lock():
            try:
                db = self._db()
            except LedgerUnavailable:
                return None
            plan = self.load()
            if plan is None:
                return None
            outcome = work(db, plan)
            if isinstance(outcome, Conflict):
                # Adjudication, not prevention: the near-miss is worth
                # recording, but a refused write must leave no state (D64).
                outcome.current_revision = plan.revision
                return outcome

            revision = plan.revision + 1
            head = _payload(_concept(db, _plan_id(plan.plan_id))) or {}
            head["revision"] = revision
            head["updated_at"] = now
            if outcome.get("goal") is not None:
                head["goal"] = outcome["goal"]
            _put(db, _plan_id(plan.plan_id),
                 head.get("goal", {}).get("text", ""), head)
            self._record(db, plan.plan_id, actor=actor,
                         now=now, op=op, args=outcome.get("op", {}),
                         rationale=rationale,
                         deviation=outcome.get("deviation"))
            return self.load()

    # ── mutations (the store's signatures, exactly) ───────────────────

    def add_task(self, *, title: str, parent: Optional[str] = None,
                 note: str = "", actor: str = "agent",
                 now: Optional[str] = None, rationale: str) -> "Any":
        from .task_store import Conflict, _now_iso

        now = now or _now_iso()

        def work(db, plan):
            if parent is not None and not _find(plan, parent):
                return Conflict(0, f"parent task '{parent}' does not exist",
                                parent)
            tid = _next_task_id(plan)
            order = max((t.order for t in plan.tasks), default=-1) + 1
            _put(db, _task_id(plan.plan_id, tid), title.strip() or tid, {
                "id": tid, "status": "pending", "parent": parent,
                "order": order, "note": note, "origin": "added",
                "added_by": actor, "claimed_by": None, "done_by": None,
                "created_at": now, "updated_at": now,
            })
            db.assert_edge(_edge(_plan_id(plan.plan_id),
                                 _task_id(plan.plan_id, tid), EDGE_HAS_TASK))
            if parent:
                db.assert_edge(_edge(_task_id(plan.plan_id, tid),
                                     _task_id(plan.plan_id, parent),
                                     EDGE_SUBTASK_OF))
            return {"op": {"id": tid, "title": title, "parent": parent}}

        return self._commit(actor=actor, now=now, op="task_add",
                            rationale=rationale, work=work)

    def update_task(self, *, task_id: str, status: Optional[str] = None,
                    title: Optional[str] = None, note: Optional[str] = None,
                    actor: str = "agent", now: Optional[str] = None) -> "Any":
        from .task_store import UPDATE_STATUSES, Conflict, _now_iso

        now = now or _now_iso()

        def work(db, plan):
            task = _find(plan, task_id)
            if task is None:
                return Conflict(0, f"task '{task_id}' does not exist", task_id)
            if status is not None and status not in UPDATE_STATUSES:
                if status == "done":
                    return Conflict(0, "use task_complete to mark a task done "
                                    "(it requires a rationale)", task_id)
                if status == "dropped":
                    return Conflict(0, "use task_rescope to drop a task "
                                    "(it requires a rationale)", task_id)
                return Conflict(0, f"invalid status '{status}'", task_id)
            if status is None and title is None and note is None:
                return Conflict(0, "nothing to update", task_id)
            body = _task_body(task)
            if status is not None:
                body["status"] = status
            if note is not None:
                body["note"] = note
            body["updated_at"] = now
            _put(db, _task_id(plan.plan_id, task_id),
                 title if title is not None else task.title, body)
            return {"op": {"id": task_id, "status": status, "title": title,
                           "note": note}}

        return self._commit(actor=actor, now=now, op="task_update",
                            rationale=None, work=work)

    def complete_task(self, *, task_id: str, actor: str = "agent",
                      now: Optional[str] = None, rationale: str) -> "Any":
        from .task_store import Conflict, _now_iso

        now = now or _now_iso()

        def work(db, plan):
            task = _find(plan, task_id)
            if task is None:
                return Conflict(0, f"task '{task_id}' does not exist", task_id)
            if task.status == "done":
                return Conflict(0, f"task '{task_id}' is already done "
                                f"(by {task.done_by or 'unknown'})", task_id)
            if task.status == "dropped":
                return Conflict(0, f"task '{task_id}' was dropped; cannot "
                                f"complete", task_id)
            body = _task_body(task)
            body.update(status="done", done_by=actor, updated_at=now)
            _put(db, _task_id(plan.plan_id, task_id), task.title, body)
            _ensure_agent(db, actor)
            _link(db, _task_id(plan.plan_id, task_id),
                  _agent_id(actor), EDGE_DONE_BY)
            return {"op": {"id": task_id}}

        return self._commit(actor=actor, now=now, op="task_complete",
                            rationale=rationale, work=work)

    def rescope_task(self, *, task_id: str, new_title: Optional[str] = None,
                     new_status: str = "dropped", actor: str = "agent",
                     now: Optional[str] = None, rationale: str) -> "Any":
        from .task_store import (
            DEV_DROP, DEV_RESCOPE, STATUSES, Conflict, _now_iso,
        )

        now = now or _now_iso()

        def work(db, plan):
            task = _find(plan, task_id)
            if task is None:
                return Conflict(0, f"task '{task_id}' does not exist", task_id)
            if new_status not in STATUSES:
                return Conflict(0, f"invalid status '{new_status}'", task_id)
            if task.status == "dropped" and new_status == "dropped":
                return Conflict(0, f"task '{task_id}' is already dropped",
                                task_id)
            before = {"title": task.title, "status": task.status}
            body = _task_body(task)
            body.update(status=new_status, updated_at=now)
            title = new_title if new_title is not None else task.title
            _put(db, _task_id(plan.plan_id, task_id), title, body)
            return {
                "op": {"id": task_id, "new_status": new_status,
                       "new_title": new_title},
                "deviation": {
                    "kind": DEV_DROP if new_status == "dropped" else DEV_RESCOPE,
                    "target": task_id, "from": before,
                    "to": {"title": title, "status": new_status},
                },
            }

        return self._commit(actor=actor, now=now, op="task_rescope",
                            rationale=rationale, work=work)

    def revise_goal(self, *, new_text: Optional[str] = None,
                    acceptance_add: Optional[list] = None,
                    acceptance_remove: Optional[list] = None,
                    actor: str = "agent", now: Optional[str] = None,
                    rationale: str) -> "Any":
        from .task_store import DEV_ACCEPT, DEV_GOAL, Conflict, _now_iso

        now = now or _now_iso()

        def work(db, plan):
            acceptance = list(plan.goal.acceptance)
            before = {"text": plan.goal.text, "acceptance": list(acceptance)}
            text = new_text if new_text is not None else plan.goal.text
            for crit in (acceptance_remove or []):
                if crit in acceptance:
                    acceptance.remove(crit)
            for crit in (acceptance_add or []):
                if crit not in acceptance:
                    acceptance.append(crit)
            if text == plan.goal.text and acceptance == before["acceptance"]:
                return Conflict(0, "goal_revise changed nothing", "goal")
            return {
                "op": {"new_text": new_text, "acceptance_add": acceptance_add,
                       "acceptance_remove": acceptance_remove},
                "goal": {"text": text,
                         "original_text": plan.goal.original_text,
                         "acceptance": acceptance,
                         "revised": plan.goal.revised
                         or text != plan.goal.original_text},
                "deviation": {
                    "kind": DEV_GOAL if text != plan.goal.text else DEV_ACCEPT,
                    "target": "goal", "from": before,
                    "to": {"text": text, "acceptance": acceptance},
                },
            }

        return self._commit(actor=actor, now=now, op="goal_revise",
                            rationale=rationale, work=work)

    def claim_task(self, *, task_id: str, actor: str = "agent",
                   now: Optional[str] = None) -> "Any":
        """Claim a task -- the compound decision the lock exists for (D64)."""
        from .task_store import Conflict, _now_iso

        now = now or _now_iso()

        def work(db, plan):
            task = _find(plan, task_id)
            if task is None:
                return Conflict(0, f"task '{task_id}' does not exist", task_id)
            if task.claimed_by and task.claimed_by != actor:
                # The losing claim stays in the log as an audited
                # near-miss; under Doctrine III a conflict is evidence.
                return Conflict(0, f"task '{task_id}' is claimed by "
                                f"{task.claimed_by}", task_id)
            body = _task_body(task)
            body.update(claimed_by=actor, updated_at=now)
            _put(db, _task_id(plan.plan_id, task_id), task.title, body)
            _ensure_agent(db, actor)
            _link(db, _task_id(plan.plan_id, task_id),
                  _agent_id(actor), EDGE_CLAIMED_BY)
            return {"op": {"id": task_id}}

        return self._commit(actor=actor, now=now, op="task_claim",
                            rationale=None, work=work)


# ── ledger helpers ───────────────────────────────────────────────────────

def _now_dt():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _put(db: Any, concept_id: str, title: str, payload: dict) -> None:
    """Upsert one concept. Doctrine III: this supersedes, never rewrites."""
    import json

    db.upsert_concept(_macrame.ConceptUpsert(
        id=concept_id, title=title or concept_id,
        content=json.dumps(payload, ensure_ascii=False),
        valid_from=_now_dt(),
    ))


def _ensure_agent(db: Any, actor: str) -> None:
    """An edge needs both ends to exist, so an actor is a concept too.

    Which is also what §17 wants: `agent:<name>` is a node in the graph,
    so "what did this agent touch" is a traversal rather than a scan.
    """
    if _concept(db, _agent_id(actor)) is None:
        _put(db, _agent_id(actor), actor, {"actor": actor})


def _link(db: Any, source: str, target: str, edge_type: str) -> None:
    """Assert an edge unless it is already open.

    Macrame refuses a second open interval on the same edge, and rightly:
    an edge asserted twice would be two claims of the same fact. Silk's
    idempotent cases -- an agent retrying its own claim -- want the fact,
    not a second assertion of it.
    """
    if any(node.id == target for node in _related(db, source, edge_type)):
        return
    db.assert_edge(_edge(source, target, edge_type))


def _edge(source: str, target: str, edge_type: str, weight: float = 1.0):
    return _macrame.EdgeAssertion(source=source, target=target,
                                  edge_type=edge_type, weight=weight,
                                  valid_from=_now_dt())


def _when(as_of) -> dict:
    """Traversal kwargs for a read at *as_of*, or for the present.

    Macrame refuses `as_of` without an attribute mode on purpose: the
    past's topology wearing the present's titles is a thing you can want
    and a terrible thing to get by accident. A time-travelling read here
    always wants both to be the same instant.
    """
    if as_of is None:
        return {}
    return {"as_of": as_of, "attribute_mode": _macrame.AttributeMode.AT_TIME}


def _concept(db: Any, concept_id: str, as_of=None) -> Optional[Any]:
    for node in db.traverse(concept_id, max_depth=0, **_when(as_of)):
        if node.id == concept_id:
            return node
    return None


def _related(db: Any, start: str, edge_type: str, as_of=None) -> list:
    """Neighbours one hop along *edge_type* -- the start is not one."""
    return [node for node in db.traverse(start, max_depth=1,
                                         edge_types=[edge_type],
                                         **_when(as_of))
            if node.id != start]


def _payload(node: Any) -> dict:
    import json

    try:
        body = json.loads(node.content or "{}")
    except (TypeError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def _find(plan: Any, task_id: str):
    return next((t for t in plan.tasks if t.id == task_id), None)


def _task_body(task: Any) -> dict:
    return {
        "id": task.id, "status": task.status, "parent": task.parent,
        "order": task.order, "note": task.note, "origin": task.origin,
        "added_by": task.added_by, "claimed_by": task.claimed_by,
        "done_by": task.done_by, "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


def _next_task_id(plan: Any) -> str:
    highest = 0
    for task in plan.tasks:
        if task.id.startswith("t") and task.id[1:].isdigit():
            highest = max(highest, int(task.id[1:]))
    return f"t{highest + 1}"


# ── History: turns, runs, and the search over them (D65, D66) ────────────

#: History's edge vocabulary. §17 writes these underscored in prose;
#: Macrame validates edge types against ``[A-Z0-9]+``, so they are solid.
EDGE_IN_RUN = "INRUN"          # turn -> run
EDGE_BY_AGENT = "BYAGENT"      # run  -> agent
EDGE_TOUCHED = "TOUCHED"       # run  -> file
EDGE_USED = "USED"             # run  -> tool
EDGE_IN_SESSION = "INSESSION"  # run  -> session
EDGE_SUPERSEDES = "SUPERSEDES"  # compaction event -> the run it compacted

KIND_TURN = "turn"
KIND_RUN = "run"
KIND_COMPACTION = "compaction"


def history_path(root: str | os.PathLike, stem: str = "history") -> Path:
    """Where a sandbox root's history ledger lives.

    A separate file from the task ledger on purpose: history is written
    every turn and read by search, tasks are written rarely and read by
    every viewer. One Write Actor each keeps a chatty writer from queueing
    behind a plan read, and lets a graph keep its plan while dropping its
    memory (or the reverse).
    """
    return Path(root).expanduser() / f"{stem}.macrame"


def _run_cid(run_id: str) -> str:
    return f"run:{run_id}"


def _turn_cid(run_id: str, index: int) -> str:
    return f"turn:{run_id}:{index}"


def _session_cid(session_id: str) -> str:
    return f"session:{session_id}"


def _file_cid(path: str) -> str:
    return f"file:{Path(str(path)).as_posix()}"


def _tool_cid(name: str) -> str:
    return f"tool:{name}"


def _compaction_cid(run_id: str, index: int) -> str:
    return f"compaction:{run_id}:{index}"


class HistoryLedger:
    """Turns and runs as concepts, and ``recall`` over them (§17, D66).

    This is the half of §17 that fits structurally: history *is* an
    append-only assertion log, and Silk already treats it as one (I11 --
    the prefix grows only at the tail). Writing it here rather than to
    ``self._history`` on a node buys two things the node cannot:

    * a turn outlives the node, the run, and the session, so ``recall``
      is memory across sessions rather than scrollback; and
    * compaction stops being destructive (D24/D25). The dropped rounds
      are superseded, not deleted, so *what did the model actually see
      at round 7* stays answerable -- which is what D41's measurement
      and D42's tests want.

    Writes are turn-shaped (D65): the discrete facts go in as they
    happen, and a turn's bookkeeping edges go in as **one**
    ``write_bulk_atomic`` at turn end, so a turn either happened or did
    not. That is also what keeps ``as_of`` replay clean and the T7 event
    firehose out of the ledger -- events are a log, not belief.
    """

    def __init__(self, root: str | os.PathLike, *,
                 db_path: str | os.PathLike | None = None,
                 registry: Optional[LedgerRegistry] = None,
                 embedder: Any = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self._registry = registry if registry is not None else REGISTRY
        self._path = Path(db_path) if db_path else history_path(self.root)
        #: The vector half of recall (SS17), or ``None`` for keyword only.
        #: Nothing here requires it: a turn is written the same either way
        #: and only its vector is missing, which is what every turn written
        #: before an embedding model was wired already looks like.
        self._embedder = embedder
        #: Whether this ledger has told Macrame the model's dimension yet.
        #: The width is only known once a vector exists, so registration
        #: cannot happen at construction time.
        self._model_registered = False

    @property
    def path(self) -> Path:
        return self._path

    def _db(self) -> Any:
        return self._registry.acquire(self._path)

    def release(self) -> None:
        """Give up this writer's share of the handle. Never closes (D62)."""
        self._registry.release(self._path)

    # ── writing ───────────────────────────────────────────────────────

    def start_run(self, run_id: str, *, agent: str = "", session: str = "",
                  goal: str = "", now: Optional[str] = None) -> str:
        """Assert the run concept and its identity edges (D65, D60).

        The identity plumbing D60 mandates for observability is the same
        plumbing the ledger wants for keys, so this writes both at once.
        """
        from .task_store import _now_iso

        now = now or _now_iso()
        db = self._db()
        cid = _run_cid(run_id)
        _put(db, cid, goal or run_id, {
            "kind": KIND_RUN, "run_id": run_id, "agent": agent,
            "session": session, "goal": goal, "started_at": now,
            "status": "running", "turns": 0,
        })
        edges = []
        if agent:
            _ensure_agent(db, agent)
            edges.append(_edge(cid, _agent_id(agent), EDGE_BY_AGENT))
        if session:
            _put(db, _session_cid(session), session,
                 {"kind": "session", "session_id": session})
            edges.append(_edge(cid, _session_cid(session), EDGE_IN_SESSION))
        if edges:
            db.write_bulk_atomic(edges)
        return cid

    def record_turn(self, run_id: str, *, index: int, role: str, text: str,
                    tools: Optional[Sequence[str]] = None,
                    files: Optional[Sequence[str]] = None,
                    now: Optional[str] = None) -> str:
        """One turn, plus its bookkeeping edges in one atomic write (D65).

        The turn concept goes in first because it is the discrete fact;
        the edges follow as a batch because they are bookkeeping. A
        crash between the two leaves an orphan turn, which reads as *a
        turn happened and we do not know what it touched* -- the honest
        answer, and the one that keeps the hot loop at a couple of
        ledger calls.
        """
        from .task_store import _now_iso

        now = now or _now_iso()
        db = self._db()
        cid = _turn_cid(run_id, index)
        _put(db, cid, f"{role} turn {index}", {
            "kind": KIND_TURN, "run_id": run_id, "index": int(index),
            "role": role, "text": text, "at": now,
            "tools": list(tools or []), "files": [str(f) for f in (files or [])],
        })

        self._embed(db, cid, text)

        run_node = _concept(db, _run_cid(run_id))
        edges = [_edge(cid, _run_cid(run_id), EDGE_IN_RUN)]
        for name in (tools or []):
            _put(db, _tool_cid(name), name, {"kind": "tool", "name": name})
            edges.append(_edge(_run_cid(run_id), _tool_cid(name), EDGE_USED))
        for path in (files or []):
            _put(db, _file_cid(path), str(path),
                 {"kind": "file", "path": str(path)})
            edges.append(_edge(_run_cid(run_id), _file_cid(path), EDGE_TOUCHED))

        if run_node is None:                     # a turn without a start_run
            self.start_run(run_id, now=now)      # still gets a run to hang on
        _bulk(db, edges)

        body = _payload(_concept(db, _run_cid(run_id))) or {}
        body["turns"] = max(int(body.get("turns", 0)), int(index) + 1)
        body["updated_at"] = now
        _put(db, _run_cid(run_id), body.get("goal") or run_id, body)
        return cid

    def finish_run(self, run_id: str, *, status: str = "finished",
                   summary: str = "", now: Optional[str] = None) -> None:
        """Supersede the run's status. Doctrine III: never a rewrite."""
        from .task_store import _now_iso

        now = now or _now_iso()
        db = self._db()
        node = _concept(db, _run_cid(run_id))
        if node is None:
            return
        body = _payload(node)
        body.update(status=status, summary=summary, finished_at=now)
        _put(db, _run_cid(run_id), node.title or run_id, body)

    def compacted(self, run_id: str, *, dropped: Sequence[int],
                  kept: int, rationale: str = "",
                  now: Optional[str] = None) -> str:
        """Record a compaction as a supersession event (§12, D24/D25).

        This is the point of putting history here at all: compaction is
        the one deliberate invalidation Silk allows (I11), and in the
        ledger it is an assertion about earlier turns rather than the
        destruction of them. The compacted rounds stay readable.
        """
        from .task_store import _now_iso

        now = now or _now_iso()
        db = self._db()
        index = len(self.compactions(run_id))
        cid = _compaction_cid(run_id, index)
        _put(db, cid, f"compaction {index}", {
            "kind": KIND_COMPACTION, "run_id": run_id,
            "dropped": [int(i) for i in dropped], "kept": int(kept),
            "rationale": rationale, "at": now,
        })
        edges = [_edge(cid, _run_cid(run_id), EDGE_SUPERSEDES)]
        for turn_index in dropped:
            edges.append(_edge(cid, _turn_cid(run_id, int(turn_index)),
                               EDGE_SUPERSEDES))
        _bulk(db, edges)
        return cid

    # ── reading ───────────────────────────────────────────────────────

    def run(self, run_id: str) -> Optional[dict]:
        node = _concept(self._db(), _run_cid(run_id))
        return _payload(node) if node is not None else None

    def turns(self, run_id: str, *, include_superseded: bool = True,
              as_of=None) -> list[dict]:
        """This run's turns, oldest first.

        With *include_superseded* false the compacted rounds are left
        out -- what the model would see now. With it true (the default)
        the whole conversation is there, which is the question the
        ledger exists to answer.
        """
        db = self._db()
        run_node = _concept(db, _run_cid(run_id), as_of)
        if run_node is None:
            return []
        # `IN_RUN` points turn -> run (§17's vocabulary: the turn is the
        # thing claiming membership), and traversal is directed, so the
        # walk back down runs on the deterministic ids instead.
        rows = []
        for index in range(int(_payload(run_node).get("turns", 0))):
            node = _concept(db, _turn_cid(run_id, index), as_of)
            if node is None:
                continue
            body = _payload(node)
            if body.get("kind") == KIND_TURN:
                rows.append(body)
        rows.sort(key=lambda row: int(row.get("index", 0)))
        if include_superseded:
            return rows
        gone = {i for event in self.compactions(run_id)
                for i in event.get("dropped", [])}
        return [row for row in rows if int(row.get("index", -1)) not in gone]

    def compactions(self, run_id: str) -> list[dict]:
        """Supersession events for this run, oldest first."""
        # Traversal is directed and the edge is compaction -> run (the
        # event is the thing making a claim about the run), so this walks
        # the event ids rather than the graph.
        db = self._db()
        events = []
        for index in range(_COMPACTION_SCAN_LIMIT):
            node = _concept(db, _compaction_cid(run_id, index))
            if node is None:
                break
            events.append(_payload(node))
        return events

    def touched(self, run_id: str) -> list[str]:
        """Files this run touched -- a traversal, not a log scan (§17)."""
        return sorted(_payload(node).get("path", "")
                      for node in _related(self._db(), _run_cid(run_id),
                                           EDGE_TOUCHED))

    def used(self, run_id: str) -> list[str]:
        """Tools this run used."""
        return sorted(_payload(node).get("name", "")
                      for node in _related(self._db(), _run_cid(run_id),
                                           EDGE_USED))

    # -- the vector half (§17) ------------------------------------------

    @property
    def embedder(self) -> Any:
        """The embedding model backing this ledger's memory, if any."""
        return self._embedder

    def _embed(self, db: Any, cid: str, text: str) -> bool:
        """Store *text*'s vector against *cid*. Never raises.

        A turn is a fact; its vector is an index entry. Losing the index
        entry costs a search a little recall, and losing the turn would
        cost the run its memory -- so every failure here is a warning and
        the write goes on. That also covers the ordinary case of a chat
        model wired where an embedding model was meant: the embedder
        disables itself after one attempt and this becomes a no-op.
        """
        embedder = self._embedder
        if embedder is None:
            return False
        try:
            vector = embedder.embed(text)
            if not vector:
                return False
            if not self._model_registered:
                # The width is discovered, not declared: registering at
                # the wrong one would raise on every later write.
                db.register_model(embedder.name, len(vector))
                self._model_registered = True
            db.upsert_embeddings(embedder.name, [(cid, vector)])
            return True
        except Exception as exc:  # noqa: BLE001 - the index is optional
            log.warning(f"History embedding skipped for {cid}: {exc}")
            embedder.disable(f"{type(exc).__name__}: {exc}")
            return False

    def recall(self, query: str, *, top_k: int = 10,
               kinds: Sequence[str] = (KIND_TURN,)) -> list[dict]:
        """Search remembered turns and runs -- hybrid when it can be.

        §17's plan was FTS5 first (it needs no model) and vectors when
        something in the graph produces them. With an embedder wired, the
        query is embedded and the ledger fuses keyword and vector ranks by
        RRF, which is what makes *what did we conclude about the lexer*
        findable when the words used then are not the words used now.

        Without one -- or when embedding the query fails -- this is the
        keyword search it has always been. The result shape does not
        change either way: hits, ranked, with enough identity to traverse
        onwards. ``via`` says which arm found each hit -- ``both``,
        ``vector`` or ``keyword`` -- because a hit both arms found is a
        different kind of hit from one only the vectors found, and the
        fused score alone cannot say which.

        The two paths do not share a scale (BM25 is negative and ascends,
        RRF is positive and descends), so scores are comparable *within*
        one search and not across ledgers searched differently. That is
        the same caveat `recall`'s merge across roots already carries.
        """
        text = (query or "").strip()
        if not text:
            return []
        db = self._db()
        hits = []
        for cid, score, via in self._ranked(db, text, top_k):
            node = _concept(db, cid)
            if node is None:
                continue
            body = _payload(node)
            if kinds and body.get("kind") not in kinds:
                continue
            hits.append({
                "id": cid, "kind": body.get("kind", ""),
                "run_id": body.get("run_id", ""),
                "index": body.get("index"), "role": body.get("role", ""),
                "at": body.get("at", ""), "title": node.title or "",
                "text": body.get("text", body.get("goal", "")),
                "score": float(score), "via": via,
            })
            if len(hits) >= top_k:
                break
        return hits

    def _ranked(self, db: Any, text: str, top_k: int) -> list[tuple]:
        """``(concept_id, score, via)`` for a query, hybrid where possible.

        Kinds are filtered by the caller, so this over-fetches: a query
        whose best matches are all runs must still be able to return
        turns.

        A hybrid search that fails falls back to keyword rather than
        failing the search. Memory that stops working because its
        *optional* half broke would be worse than memory that got a
        little less clever.
        """
        width = max(1, int(top_k) * 3)
        vector = None
        if self._embedder is not None:
            try:
                vector = self._embedder.embed(text)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Query embedding failed: {exc}")
        if vector:
            try:
                if not self._model_registered:
                    # A ledger that only reads -- another root's memory,
                    # or this one in a later session -- never wrote a
                    # vector, so registration happens here too. It is
                    # idempotent at the same width and raises at another,
                    # which is the honest answer to a changed model.
                    db.register_model(self._embedder.name, len(vector))
                    self._model_registered = True
                rows = db.hybrid_search(
                    self._embedder.name, text, vector, top_k=width,
                )
                # A hit both arms found is a different kind of hit from one
                # only the vectors found, and the fused score cannot say
                # which -- so the answer carries which arm saw it.
                return [(hit.concept_id, hit.score, _via(hit)) for hit in rows]
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Hybrid search failed, using keywords: {exc}")
        return [(cid, score, "keyword")
                for cid, score in db.keyword_search(text, top_k=width)]


def _via(hit: Any) -> str:
    """Which arm of the hybrid search found this hit."""
    vector = getattr(hit, "vector_rank", None) is not None
    keyword = getattr(hit, "keyword_rank", None) is not None
    if vector and keyword:
        return "both"
    return "vector" if vector else "keyword"


#: A run with more compaction events than this is not a run any more.
_COMPACTION_SCAN_LIMIT = 512


def _bulk(db: Any, edges: list) -> None:
    """Write *edges* atomically, skipping the ones already open.

    Re-asserting an open edge is refused by Macrame (rightly -- it would
    be two claims of one fact), and a turn that used a tool the run used
    last turn is exactly that case.
    """
    fresh = []
    for edge in edges:
        if any(node.id == edge.target
               for node in _related(db, edge.source, edge.edge_type)):
            continue
        fresh.append(edge)
    if fresh:
        db.write_bulk_atomic(fresh)


# ── choosing a backend, loudly (D66) ─────────────────────────────────────

#: Which task backend to use: ``ledger`` or ``sqlite``. The default is
#: ``sqlite`` for now -- not because the ledger is unfinished, but because
#: plan *discovery* is still file-shaped: T4's `PlanRef`, `scan_all` and
#: the D58 hub all look for ``plan-*.db``. Flipping the default is a
#: separate, discovery-shaped change; until then the ledger is opt-in per
#: process and everything above the store is already backend-blind (which
#: is what `tests/test_silk_task_ledger.py` pins).
BACKEND_ENV = "SILK_TASK_BACKEND"
BACKEND_LEDGER = "ledger"
BACKEND_SQLITE = "sqlite"
DEFAULT_BACKEND = BACKEND_SQLITE


def requested_backend(default: str = DEFAULT_BACKEND) -> str:
    """What the environment asks for, normalised."""
    asked = str(os.environ.get(BACKEND_ENV, "") or "").strip().lower()
    return asked if asked in (BACKEND_LEDGER, BACKEND_SQLITE) else default


def open_task_store(root: str | os.PathLike, *, plan: Any = None,
                    backend: Optional[str] = None,
                    registry: Optional[LedgerRegistry] = None) -> Any:
    """The task store for *root*, on whichever backend is asked for.

    Both backends answer the same protocol (D66), so the caller gets one
    object and never asks which it is. If the ledger is asked for and the
    extra is missing, that is **one log line and the SQLite store** --
    the graph degrades to today's behaviour, never silently and never by
    crashing a run that was only ever going to write a task list.
    """
    from .task_store import PlanRef, SqliteTaskStore

    ref = PlanRef.coerce(plan)
    want = (backend or requested_backend()).lower()
    if want == BACKEND_LEDGER:
        if available():
            return TaskLedger(root, registry=registry)
        log.warning(
            "silk: task backend '%s' requested but unavailable (%s); "
            "falling back to the SQLite task store",
            BACKEND_LEDGER, unavailable_reason(),
        )
    if ref is not None and (ref.is_explicit or ref.root):
        return ref.store()
    return SqliteTaskStore(root=root)


def open_history(root: str | os.PathLike, *,
                 registry: Optional[LedgerRegistry] = None
                 ) -> Optional["HistoryLedger"]:
    """This root's history ledger, or ``None`` with one log line.

    History has no SQLite fallback to degrade to -- before D66 it lived
    on the node and died with it -- so the honest answer when the extra
    is absent is "no memory", said once, rather than an empty search
    result that reads like "nothing happened".
    """
    if not available():
        log.info("silk: no history ledger (%s); recall is unavailable",
                 unavailable_reason())
        return None
    return HistoryLedger(root, registry=registry)
