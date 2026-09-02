# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Task planning & tracking store (Qt-free, stdlib only).

The silk agent drives this itself during a run: it sets a goal, decomposes it
into a living task/subtask tree, and — whenever it changes course — records a
rationale. SQLite is the **store of record** (atomic, concurrent, full
timestamped history via an append-only ``revision`` log over materialized
``plan``/``task`` tables). One pure renderer turns the current plan into a
``(json dict, markdown str)`` pair — "the output the viewer gets" — used by the
``plan_view`` tool, the Plan Viewer node's ports, and the optional direct-write
of ``<stem>.md`` / ``<stem>.json`` next to the DB.

Design (see docs/TASK_TRACKER_PLAN.md):
- **The op is the diff.** Every mutation is a semantic op; we log the op stream,
  never infer a text diff.
- **Rationale on the four consequential mutations** — add / complete / rescope /
  revise-goal — is required by the tool layer; the store records it.
- **Multiple agents may share one plan.** ``BEGIN IMMEDIATE`` + WAL serialize
  writers; per-op precondition checks turn a genuine collision (same task, or a
  double-complete) into a :class:`Conflict` the caller surfaces, never a silent
  lost update.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

# ── Vocabulary ──────────────────────────────────────────────────────────────

#: Task lifecycle states.
STATUSES: frozenset[str] = frozenset(
    {"pending", "in_progress", "blocked", "done", "dropped"}
)

#: States a plain ``task_update`` may set (``done`` → task_complete, ``dropped``
#: → task_rescope, each of which requires a rationale).
UPDATE_STATUSES: frozenset[str] = frozenset({"pending", "in_progress", "blocked"})

MODES = ("read", "read_write", "blocked")  # (unused here; kept for symmetry)

#: Emoji *shortcodes* (plain ASCII in the markdown; mordant expands them on
#: render, so the on-disk .md stays greppable/diffable).
_STATUS_GLYPH: dict[str, str] = {
    "pending": ":white_large_square:",
    "in_progress": ":hourglass_flowing_sand:",
    "blocked": ":no_entry:",
    "done": ":white_check_mark:",
    "dropped": ":wastebasket:",
}

DEFAULT_ACTOR = "agent"

# Op kinds recorded in the revision log.
OP_START = "plan_start"
OP_ADD = "task_add"
OP_UPDATE = "task_update"
OP_COMPLETE = "task_complete"
OP_RESCOPE = "task_rescope"
OP_GOAL = "goal_revise"
OP_CLAIM = "task_claim"

# Deviation kinds (the "course corrections" ledger).
DEV_RESCOPE = "rescope_task"
DEV_DROP = "drop_task"
DEV_GOAL = "revise_goal"
DEV_ACCEPT = "change_acceptance"

_BUSY_TIMEOUT_MS = 5000
_WRITE_RETRIES = 6


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Goal:
    text: str
    original_text: str
    acceptance: list[str] = field(default_factory=list)
    revised: bool = False


@dataclass
class Task:
    id: str
    title: str
    status: str
    parent: Optional[str]
    order: int
    note: str = ""
    origin: str = "added"          # initial | added
    added_by: str = DEFAULT_ACTOR
    claimed_by: Optional[str] = None
    done_by: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Deviation:
    at: str
    actor: str
    kind: str
    target: str
    from_val: Any
    to_val: Any
    rationale: str


@dataclass
class Plan:
    plan_id: str
    created_at: str
    updated_at: str
    revision: int
    goal: Goal
    tasks: list[Task] = field(default_factory=list)
    deviations: list[Deviation] = field(default_factory=list)


@dataclass
class Conflict:
    """A genuine collision (same task / double-complete / goal race). The caller
    surfaces this to the model; retrying the identical op will not help."""
    current_revision: int
    reason: str
    target: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


log = logging.getLogger("SilkTaskStore")   # stdlib only, as the module says


class LedgerClosed(ValueError):
    """Raised when a write is attempted after the ledger was released.

    ``retryable`` is the difference between the two shutdowns.  A quit
    is the end and the caller should stop; a relaunch is a gap of a few
    seconds with a replacement on the other side, and an agent that
    reads *that* as a hard failure abandons work it could simply have
    asked for again.  Nothing was written either way — that part is the
    same, and is the part the message leads with.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class _LedgerGate:
    """The process's one writer, and the shutdown that closes it.

    ``BEGIN IMMEDIATE`` + WAL already serialize writers *between*
    processes.  What that does not give is a moment when this process
    can say "no more writes from here, and the one in flight has
    finished" — which is exactly what a relaunch needs before it spawns
    a replacement, since the replacement is a second writer and the two
    would otherwise overlap.

    Process-wide rather than per store, because "one writer per process"
    is a property of the process: stores are cheap, short-lived objects
    created per tool call, and registering each would leave the registry
    holding every one of them for the life of the run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._participant = None
        self._writing = 0
        self._retryable = False

    def _register(self) -> None:
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
            "task ledger", self.release, busy=self.busy,
        )

    @contextmanager
    def writing(self) -> Iterator[None]:
        """Hold the gate for one write.  Raises once shutdown has run."""
        self._register()
        if self._closed:
            raise self._closed_error()
        with self._lock:
            if self._closed:
                raise self._closed_error()
            self._writing += 1
            try:
                yield
            finally:
                self._writing -= 1

    def _closed_error(self) -> LedgerClosed:
        """The refusal, in the words the caller can act on."""
        if self._retryable:
            return LedgerClosed(
                "The plan was not modified: Weave is restarting. The same "
                "call will work once the replacement is up.",
                retryable=True,
            )
        return LedgerClosed(
            "The plan was not modified: Weave is shutting down."
        )

    def busy(self) -> Optional[str]:
        """A description while a write is in flight, else None.

        The relaunch sequence asks before it releases anything, so a
        transaction is not interrupted by a restart that could just as
        well have waited a second and asked again.
        """
        return "a plan write is in flight" if self._writing else None

    def release(self, timeout: float = 5.0) -> bool:
        """Refuse further writes and wait for the one in flight.

        Returns whether the ledger went quiet.  There is nothing
        forceful to escalate to: killing a write mid-transaction is what
        WAL recovery is for, and the honest answer to "it did not
        finish" is to say so rather than to make it worse.
        """
        self._retryable = self._reason_is_relaunch()
        self._closed = True
        acquired = self._lock.acquire(timeout=max(0.0, timeout))
        if acquired:
            self._lock.release()
        else:
            log.warning("A plan write was still in flight at shutdown")
        return acquired

    @staticmethod
    def _reason_is_relaunch() -> bool:
        """Whether a replacement is coming, as far as the host will say."""
        try:
            from weave.engine.shutdown import RELAUNCH, shutdown_reason
        except Exception:  # noqa: BLE001 - a plugin must not need the host
            return False
        return shutdown_reason() == RELAUNCH

    def reopen(self) -> None:
        """A gate with nothing in flight and nobody shutting down.

        For tests, and for a child after a fork: the child inherits the
        parent's ``_closed`` — a fork *during* a shutdown would hand it
        a ledger that refuses writes for no reason of its own — and a
        lock the parent may have been holding, which in the child is
        held by a thread that does not exist.  Both are replaced rather
        than reasoned about.
        """
        self._lock = threading.Lock()
        self._closed = False
        self._retryable = False
        self._writing = 0
        self._participant = None


#: The gate every store writes through.
LEDGER = _LedgerGate()

if hasattr(os, "register_at_fork"):        # POSIX only; a no-op on Windows
    os.register_at_fork(after_in_child=LEDGER.reopen)


# ── Rendering (the one renderer: plan → json / markdown) ─────────────────────

def plan_to_json(plan: Plan) -> dict:
    """The canonical JSON snapshot — the ``plan_json`` output / ``<stem>.json``."""
    return {
        "plan_id": plan.plan_id,
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
        "revision": plan.revision,
        "goal": {
            "text": plan.goal.text,
            "original_text": plan.goal.original_text,
            "acceptance": list(plan.goal.acceptance),
            "revised": plan.goal.revised,
        },
        "tasks": [
            {
                "id": t.id, "title": t.title, "status": t.status,
                "parent": t.parent, "order": t.order, "note": t.note,
                "origin": t.origin, "added_by": t.added_by,
                "claimed_by": t.claimed_by, "done_by": t.done_by,
                "created_at": t.created_at, "updated_at": t.updated_at,
            }
            for t in plan.tasks
        ],
        "deviations": [
            {
                "at": d.at, "actor": d.actor, "kind": d.kind,
                "target": d.target, "from": d.from_val, "to": d.to_val,
                "rationale": d.rationale,
            }
            for d in plan.deviations
        ],
    }


def plan_from_json(data: dict) -> Plan:
    """Reconstruct a :class:`Plan` from a ``plan_to_json`` snapshot — so a viewer
    can render a plan it received on a port without opening the DB."""
    g = data.get("goal", {}) or {}
    goal = Goal(
        text=g.get("text", ""), original_text=g.get("original_text", g.get("text", "")),
        acceptance=list(g.get("acceptance", []) or []), revised=bool(g.get("revised")),
    )
    tasks = [
        Task(
            id=t.get("id", ""), title=t.get("title", ""), status=t.get("status", "pending"),
            parent=t.get("parent"), order=int(t.get("order", 0)), note=t.get("note", "") or "",
            origin=t.get("origin", "added"), added_by=t.get("added_by", DEFAULT_ACTOR),
            claimed_by=t.get("claimed_by"), done_by=t.get("done_by"),
            created_at=t.get("created_at", ""), updated_at=t.get("updated_at", ""),
        )
        for t in (data.get("tasks", []) or [])
    ]
    deviations = [
        Deviation(
            at=d.get("at", ""), actor=d.get("actor", DEFAULT_ACTOR), kind=d.get("kind", ""),
            target=d.get("target", ""), from_val=d.get("from"), to_val=d.get("to"),
            rationale=d.get("rationale", ""),
        )
        for d in (data.get("deviations", []) or [])
    ]
    return Plan(
        plan_id=data.get("plan_id", ""), created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""), revision=int(data.get("revision", 0)),
        goal=goal, tasks=tasks, deviations=deviations,
    )


#: Statuses that still count as open work (used to resolve a plan-closing
#: completion -- the ``complete_final`` change type of the approval policy).
_OPEN_STATUSES: frozenset[str] = frozenset({"pending", "in_progress", "blocked"})


def open_task_ids(plan: Plan) -> list[str]:
    """Ids of tasks that are still open (not done, not dropped)."""
    return [t.id for t in plan.tasks if t.status in _OPEN_STATUSES]


def plan_changed_event(
    store: "SqliteTaskStore", last_revision: Optional[int],
) -> Optional[dict]:
    """A ``plan_summary`` event with the current snapshot **iff** the plan
    advanced past *last_revision*, else ``None``.

    The Agent node calls this after each tool to push live updates to a Plan
    Viewer: dedup-by-revision means only genuine mutations emit (reads never bump
    the revision), so an unchanged plan never re-streams.
    """
    try:
        plan = store.load()
    except Exception:  # noqa: BLE001 - a store hiccup must not break the run
        return None
    if plan is None or plan.revision == last_revision:
        return None
    return {"event": "plan_summary", "revision": plan.revision,
            "plan": plan_to_json(plan)}


def _children(tasks: list[Task], parent: Optional[str]) -> list[Task]:
    kids = [t for t in tasks if t.parent == parent]
    kids.sort(key=lambda t: (t.order, t.id))
    return kids


def render_markdown(plan: Plan) -> str:
    """The glanceable view — GFM tuned for mordant (task-list checkboxes, emoji
    shortcodes, a Course-corrections section). Pure ASCII; still plain markdown."""
    g = plan.goal
    lines: list[str] = [f"# Plan: {g.text}", ""]
    lines.append(f"**Goal:** {g.text}")
    if g.revised and g.original_text != g.text:
        lines.append(f"> _Original goal:_ {g.original_text}")
    lines.append("")
    if g.acceptance:
        lines.append("**Acceptance:**")
        for crit in g.acceptance:
            lines.append(f"- [ ] {crit}")
        lines.append("")

    lines.append("## Tasks")
    if not plan.tasks:
        lines.append("_No tasks yet._")
    else:
        def emit(task: Task, depth: int) -> None:
            box = "[x]" if task.status in ("done", "dropped") else "[ ]"
            glyph = _STATUS_GLYPH.get(task.status, "")
            title = f"~~{task.title}~~" if task.status == "dropped" else task.title
            owner = f" _(@{task.claimed_by})_" if task.claimed_by else ""
            note = f" -- {task.note}" if task.note else ""
            indent = "  " * depth
            lines.append(
                f"{indent}- {box} {glyph} {title}{owner} "
                f"<!-- {task.id} {task.status} -->{note}"
            )
            for kid in _children(plan.tasks, task.id):
                emit(kid, depth + 1)

        for root in _children(plan.tasks, None):
            emit(root, 0)
    lines.append("")

    if plan.deviations:
        lines.append("## Course corrections")
        for d in plan.deviations:
            frm = _short(d.from_val)
            to = _short(d.to_val)
            lines.append(
                f"- **{d.kind}** `{d.target}` ({d.actor}): "
                f"{frm} -> {to} -- _{d.rationale}_"
            )
        lines.append("")

    lines.append(
        f"<sub>revision {plan.revision} - updated {plan.updated_at}</sub>"
    )
    return "\n".join(lines) + "\n"


def _short(val: Any) -> str:
    if val is None:
        return "(none)"
    s = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
    return s if len(s) <= 80 else s[:77] + "..."


# ── The store ───────────────────────────────────────────────────────────────

class SqliteTaskStore:
    """One plan's store of record, anchored at a working-directory *root*.

    The DB + projections live at ``<dir>/<stem>.db|.md|.json`` where *dir* is the
    root if writable, else ``<root>/.silk/plan/`` (fallback). A ``.silk``-level
    pointer is not needed: agents locate the active plan by newest ``plan-*.db``
    in the candidate dirs, which is what lets several agents share one plan.
    """

    def __init__(self, root: str | os.PathLike, *, direct_write: bool = True) -> None:
        self.root = Path(root).resolve()
        self.direct_write = direct_write
        self._db_path: Optional[Path] = None
        self._pid: Optional[str] = None

    # -- location ---------------------------------------------------------

    def _candidate_dirs(self) -> list[Path]:
        return [self.root, self.root / ".silk" / "plan"]

    @staticmethod
    def _writable_dir(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        return os.access(path, os.W_OK)

    def _target_dir(self) -> Path:
        """Where a *new* plan's files go: root if writable, else .silk/plan/."""
        if self._writable_dir(self.root):
            return self.root
        fallback = self.root / ".silk" / "plan"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _locate_db(self) -> Optional[Path]:
        """Newest ``plan-*.db`` across the candidate dirs (shared-plan discovery)."""
        if self._db_path is not None and self._db_path.exists():
            return self._db_path
        found: list[Path] = []
        for d in self._candidate_dirs():
            if d.is_dir():
                found.extend(d.glob("plan-*.db"))
        if not found:
            return None
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        self._db_path = found[0]
        return self._db_path

    @staticmethod
    def _new_stem() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"plan-{ts}-{secrets.token_hex(3)}"

    # -- connection -------------------------------------------------------

    def _connect(self, path: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(path), timeout=_BUSY_TIMEOUT_MS / 1000,
                              isolation_level=None)  # explicit BEGIN IMMEDIATE
        con.execute("PRAGMA journal_mode=WAL")
        con.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    @staticmethod
    def _schema(con: sqlite3.Connection) -> None:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS plan (
                plan_id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT,
                goal_text TEXT, goal_original TEXT, goal_acceptance TEXT,
                revised INTEGER, revision INTEGER
            );
            CREATE TABLE IF NOT EXISTS task (
                plan_id TEXT, id TEXT, title TEXT, status TEXT, parent TEXT,
                ord INTEGER, note TEXT, origin TEXT, added_by TEXT,
                claimed_by TEXT, done_by TEXT, created_at TEXT, updated_at TEXT,
                PRIMARY KEY (plan_id, id)
            );
            CREATE TABLE IF NOT EXISTS revision (
                id INTEGER PRIMARY KEY AUTOINCREMENT, plan_id TEXT, at TEXT,
                actor TEXT, op_kind TEXT, op_json TEXT, rationale TEXT
            );
            CREATE TABLE IF NOT EXISTS deviation (
                revision_id INTEGER PRIMARY KEY, plan_id TEXT, kind TEXT,
                target TEXT, from_json TEXT, to_json TEXT, actor TEXT,
                rationale TEXT, at TEXT
            );
            """
        )

    # -- load -------------------------------------------------------------

    def load(self) -> Optional[Plan]:
        """Return the current plan, or ``None`` if no plan exists yet."""
        path = self._locate_db()
        if path is None:
            return None
        con = self._connect(path)
        try:
            return self._load_con(con)
        finally:
            con.close()

    def _load_con(self, con: sqlite3.Connection) -> Optional[Plan]:
        row = con.execute(
            "SELECT plan_id, created_at, updated_at, goal_text, goal_original, "
            "goal_acceptance, revised, revision FROM plan LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        (pid, created, updated, gtext, goriginal, gaccept, revised,
         revision) = row
        self._pid = pid
        goal = Goal(
            text=gtext, original_text=goriginal,
            acceptance=json.loads(gaccept or "[]"), revised=bool(revised),
        )
        tasks = [
            Task(
                id=r[0], title=r[1], status=r[2], parent=r[3], order=r[4],
                note=r[5] or "", origin=r[6] or "added",
                added_by=r[7] or DEFAULT_ACTOR, claimed_by=r[8], done_by=r[9],
                created_at=r[10] or "", updated_at=r[11] or "",
            )
            for r in con.execute(
                "SELECT id, title, status, parent, ord, note, origin, added_by, "
                "claimed_by, done_by, created_at, updated_at FROM task "
                "WHERE plan_id=? ORDER BY ord, id", (pid,)
            )
        ]
        deviations = [
            Deviation(
                at=r[7], actor=r[5], kind=r[0], target=r[1],
                from_val=json.loads(r[2]) if r[2] is not None else None,
                to_val=json.loads(r[3]) if r[3] is not None else None,
                rationale=r[6] or "",
            )
            for r in con.execute(
                "SELECT kind, target, from_json, to_json, revision_id, actor, "
                "rationale, at FROM deviation WHERE plan_id=? ORDER BY revision_id",
                (pid,),
            )
        ]
        return Plan(
            plan_id=pid, created_at=created, updated_at=updated,
            revision=revision, goal=goal, tasks=tasks, deviations=deviations,
        )

    # -- history ----------------------------------------------------------

    def history(self, limit: Optional[int] = None) -> list[dict]:
        """Recent revision-log entries (newest first): op + actor + rationale."""
        path = self._locate_db()
        if path is None:
            return []
        con = self._connect(path)
        try:
            pid_row = con.execute("SELECT plan_id FROM plan LIMIT 1").fetchone()
            if pid_row is None:
                return []
            self._pid = pid_row[0]
            q = ("SELECT id, at, actor, op_kind, op_json, rationale FROM revision "
                 "WHERE plan_id=? ORDER BY id DESC")
            if limit:
                q += f" LIMIT {int(limit)}"
            return [
                {
                    "revision": r[0], "at": r[1], "actor": r[2],
                    "op": r[3], "args": json.loads(r[4] or "{}"),
                    "rationale": r[5],
                }
                for r in con.execute(q, (self._pid,))
            ]
        finally:
            con.close()

    # -- creation ---------------------------------------------------------

    def start(
        self, *, goal: str, acceptance: Optional[list[str]] = None,
        tasks: Optional[list[dict]] = None, actor: str = DEFAULT_ACTOR,
        now: Optional[str] = None,
    ) -> Plan:
        """Create the plan + initial decomposition. Raises if one already exists."""
        if self._locate_db() is not None and self.load() is not None:
            raise ValueError(
                "A plan already exists; use task_add / task_update / … instead of "
                "plan_start."
            )
        now = now or _now_iso()
        acceptance = list(acceptance or [])
        stem = self._new_stem()
        self._db_path = self._target_dir() / f"{stem}.db"
        pid = secrets.token_hex(8)
        self._pid = pid

        with LEDGER.writing():
            self._start_locked(goal, acceptance, tasks, actor, now, pid)
            plan = self.load()
            assert plan is not None
            self._write_projections(plan)
        return plan

    def _start_locked(self, goal, acceptance, tasks, actor, now, pid) -> None:
        """The write half of :meth:`start`, run while holding the gate."""
        assert self._db_path is not None
        con = self._connect(self._db_path)
        try:
            self._schema(con)
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "INSERT INTO plan (plan_id, created_at, updated_at, goal_text, "
                "goal_original, goal_acceptance, revised, revision) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (pid, now, now, goal, goal, json.dumps(acceptance), 0, 0),
            )
            rev = 0
            for i, spec in enumerate(tasks or []):
                rev += 1
                tid = f"t{i + 1}"
                con.execute(
                    "INSERT INTO task (plan_id, id, title, status, parent, ord, "
                    "note, origin, added_by, claimed_by, done_by, created_at, "
                    "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, tid, str(spec.get("title", "")).strip() or tid,
                     "pending", spec.get("parent"), i, str(spec.get("note", "")),
                     "initial", actor, None, None, now, now),
                )
            con.execute("UPDATE plan SET revision=? WHERE plan_id=?", (rev if (tasks) else 0, pid))
            con.execute(
                "INSERT INTO revision (plan_id, at, actor, op_kind, op_json, rationale) "
                "VALUES (?,?,?,?,?,?)",
                (pid, now, actor, OP_START,
                 json.dumps({"goal": goal, "acceptance": acceptance,
                             "tasks": len(tasks or [])}), None),
            )
            con.execute("COMMIT")
        finally:
            con.close()

    # -- commit helper ----------------------------------------------------

    def _commit(
        self, *, actor: str, now: str, op_kind: str, rationale: Optional[str],
        work: Callable[[sqlite3.Connection], "dict | Conflict"],
    ) -> "Plan | Conflict | None":
        """Run *work* inside one ``BEGIN IMMEDIATE`` txn, then append the revision
        row (+ optional deviation), bump the revision, and re-render projections.

        *work(con)* performs the precondition check and table mutation and returns
        either ``{"op": <dict>, "deviation": <dict|None>}`` or a :class:`Conflict`
        (which rolls the txn back). Returns ``None`` if no plan exists.
        """
        path = self._locate_db()
        if path is None:
            return None
        with LEDGER.writing():
            outcome = self._commit_locked(
                path=path, actor=actor, now=now, op_kind=op_kind,
                rationale=rationale, work=work,
            )
            if outcome is not None:
                return outcome

            # Inside the gate: the projections beside the DB are part of
            # the write, and a shutdown that stopped between the commit
            # and the render would leave the .md disagreeing with the
            # store of record.
            plan = self.load()
            if plan is not None:
                self._write_projections(plan)
            return plan

    def _commit_locked(
        self, *, path: Path, actor: str, now: str, op_kind: str,
        rationale: Optional[str],
        work: Callable[[sqlite3.Connection], "dict | Conflict"],
    ) -> "Conflict | None":
        """The transaction half of :meth:`_commit`, holding the gate.

        Returns a ``Conflict`` when the transaction rolled back, and
        ``None`` both when it committed and when there was no plan to
        commit against -- the caller reloads either way, and a store
        with no plan reloads as ``None``, which is the answer that case
        wants.
        """
        for attempt in range(_WRITE_RETRIES):
            con = self._connect(path)
            try:
                con.execute("BEGIN IMMEDIATE")
                row = con.execute(
                    "SELECT plan_id, revision FROM plan LIMIT 1"
                ).fetchone()
                if row is None:
                    con.execute("ROLLBACK")
                    return None
                self._pid, cur_rev = row   # bootstrap pid for a shared-plan store
                outcome = work(con)
                if isinstance(outcome, Conflict):
                    con.execute("ROLLBACK")
                    outcome.current_revision = cur_rev
                    return outcome

                new_rev = cur_rev + 1
                con.execute(
                    "INSERT INTO revision (plan_id, at, actor, op_kind, op_json, "
                    "rationale) VALUES (?,?,?,?,?,?)",
                    (self._pid, now, actor, op_kind,
                     json.dumps(outcome.get("op", {})), rationale),
                )
                dev = outcome.get("deviation")
                if dev is not None:
                    rev_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
                    con.execute(
                        "INSERT INTO deviation (revision_id, plan_id, kind, target, "
                        "from_json, to_json, actor, rationale, at) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (rev_id, self._pid, dev["kind"], dev["target"],
                         json.dumps(dev.get("from")), json.dumps(dev.get("to")),
                         actor, rationale, now),
                    )
                con.execute(
                    "UPDATE plan SET revision=?, updated_at=? WHERE plan_id=?",
                    (new_rev, now, self._pid),
                )
                con.execute("COMMIT")
                break
            except sqlite3.OperationalError as exc:
                try:
                    con.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
                con.close()
                if "locked" in str(exc).lower() and attempt < _WRITE_RETRIES - 1:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise
            finally:
                try:
                    con.close()
                except sqlite3.ProgrammingError:
                    pass
        return None

    # -- mutations --------------------------------------------------------

    def _next_task_id(self, con: sqlite3.Connection) -> str:
        n = 0
        for (tid,) in con.execute("SELECT id FROM task WHERE plan_id=?", (self._pid,)):
            if tid and tid[0] == "t" and tid[1:].isdigit():
                n = max(n, int(tid[1:]))
        return f"t{n + 1}"

    def add_task(
        self, *, title: str, parent: Optional[str] = None, note: str = "",
        actor: str = DEFAULT_ACTOR, now: Optional[str] = None,
        rationale: str,
    ) -> "Plan | Conflict | None":
        now = now or _now_iso()

        def work(con: sqlite3.Connection) -> "dict | Conflict":
            if parent is not None and not _task_row(con, self._pid, parent):
                return Conflict(0, f"parent task '{parent}' does not exist", parent)
            tid = self._next_task_id(con)
            ordv = con.execute(
                "SELECT COALESCE(MAX(ord), -1) + 1 FROM task WHERE plan_id=?",
                (self._pid,),
            ).fetchone()[0]
            con.execute(
                "INSERT INTO task (plan_id, id, title, status, parent, ord, "
                "note, origin, added_by, claimed_by, done_by, created_at, "
                "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self._pid, tid, title.strip() or tid, "pending", parent, ordv,
                 note, "added", actor, None, None, now, now),
            )
            return {"op": {"id": tid, "title": title, "parent": parent}}

        return self._commit(actor=actor, now=now, op_kind=OP_ADD,
                            rationale=rationale, work=work)

    def update_task(
        self, *, task_id: str, status: Optional[str] = None,
        title: Optional[str] = None, note: Optional[str] = None,
        actor: str = DEFAULT_ACTOR, now: Optional[str] = None,
    ) -> "Plan | Conflict | None":
        now = now or _now_iso()

        def work(con: sqlite3.Connection) -> "dict | Conflict":
            row = _task_row(con, self._pid, task_id)
            if not row:
                return Conflict(0, f"task '{task_id}' does not exist", task_id)
            if status is not None and status not in UPDATE_STATUSES:
                if status == "done":
                    return Conflict(0, "use task_complete to mark a task done "
                                    "(it requires a rationale)", task_id)
                if status == "dropped":
                    return Conflict(0, "use task_rescope to drop a task "
                                    "(it requires a rationale)", task_id)
                return Conflict(0, f"invalid status '{status}'", task_id)
            sets, params = [], []
            if status is not None:
                sets.append("status=?"); params.append(status)
            if title is not None:
                sets.append("title=?"); params.append(title)
            if note is not None:
                sets.append("note=?"); params.append(note)
            if not sets:
                return Conflict(0, "nothing to update", task_id)
            sets.append("updated_at=?"); params.append(now)
            params.extend([self._pid, task_id])
            con.execute(
                f"UPDATE task SET {', '.join(sets)} WHERE plan_id=? AND id=?", params
            )
            return {"op": {"id": task_id, "status": status, "title": title,
                           "note": note}}

        return self._commit(actor=actor, now=now, op_kind=OP_UPDATE,
                            rationale=None, work=work)

    def complete_task(
        self, *, task_id: str, actor: str = DEFAULT_ACTOR,
        now: Optional[str] = None, rationale: str,
    ) -> "Plan | Conflict | None":
        now = now or _now_iso()

        def work(con: sqlite3.Connection) -> "dict | Conflict":
            row = _task_row(con, self._pid, task_id)
            if not row:
                return Conflict(0, f"task '{task_id}' does not exist", task_id)
            if row["status"] == "done":
                return Conflict(0, f"task '{task_id}' is already done "
                                f"(by {row['done_by'] or 'unknown'})", task_id)
            if row["status"] == "dropped":
                return Conflict(0, f"task '{task_id}' was dropped; cannot complete",
                                task_id)
            con.execute(
                "UPDATE task SET status='done', done_by=?, updated_at=? "
                "WHERE plan_id=? AND id=?", (actor, now, self._pid, task_id),
            )
            return {"op": {"id": task_id}}

        return self._commit(actor=actor, now=now, op_kind=OP_COMPLETE,
                            rationale=rationale, work=work)

    def rescope_task(
        self, *, task_id: str, new_title: Optional[str] = None,
        new_status: str = "dropped", actor: str = DEFAULT_ACTOR,
        now: Optional[str] = None, rationale: str,
    ) -> "Plan | Conflict | None":
        now = now or _now_iso()

        def work(con: sqlite3.Connection) -> "dict | Conflict":
            row = _task_row(con, self._pid, task_id)
            if not row:
                return Conflict(0, f"task '{task_id}' does not exist", task_id)
            if new_status not in STATUSES:
                return Conflict(0, f"invalid status '{new_status}'", task_id)
            if row["status"] == "dropped" and new_status == "dropped":
                return Conflict(0, f"task '{task_id}' is already dropped", task_id)
            before = {"title": row["title"], "status": row["status"]}
            sets, params = ["status=?", "updated_at=?"], [new_status, now]
            if new_title is not None:
                sets.insert(0, "title=?"); params.insert(0, new_title)
            params.extend([self._pid, task_id])
            con.execute(
                f"UPDATE task SET {', '.join(sets)} WHERE plan_id=? AND id=?", params
            )
            after = {"title": new_title if new_title is not None else row["title"],
                     "status": new_status}
            kind = DEV_DROP if new_status == "dropped" else DEV_RESCOPE
            return {
                "op": {"id": task_id, "new_status": new_status,
                       "new_title": new_title},
                "deviation": {"kind": kind, "target": task_id,
                              "from": before, "to": after},
            }

        return self._commit(actor=actor, now=now, op_kind=OP_RESCOPE,
                            rationale=rationale, work=work)

    def revise_goal(
        self, *, new_text: Optional[str] = None,
        acceptance_add: Optional[list[str]] = None,
        acceptance_remove: Optional[list[str]] = None,
        actor: str = DEFAULT_ACTOR, now: Optional[str] = None, rationale: str,
    ) -> "Plan | Conflict | None":
        now = now or _now_iso()

        def work(con: sqlite3.Connection) -> "dict | Conflict":
            r = con.execute(
                "SELECT goal_text, goal_original, goal_acceptance, revised "
                "FROM plan LIMIT 1"
            ).fetchone()
            gtext, goriginal, gaccept_json, revised = r
            acceptance = json.loads(gaccept_json or "[]")
            before = {"text": gtext, "acceptance": list(acceptance)}
            new_goal = new_text if new_text is not None else gtext
            for crit in (acceptance_remove or []):
                if crit in acceptance:
                    acceptance.remove(crit)
            for crit in (acceptance_add or []):
                if crit not in acceptance:
                    acceptance.append(crit)
            if new_goal == gtext and acceptance == before["acceptance"]:
                return Conflict(0, "goal_revise changed nothing", "goal")
            revised_flag = 1 if (new_goal != goriginal) else revised
            con.execute(
                "UPDATE plan SET goal_text=?, goal_acceptance=?, revised=? "
                "WHERE plan_id=?",
                (new_goal, json.dumps(acceptance), revised_flag, self._pid),
            )
            after = {"text": new_goal, "acceptance": acceptance}
            kind = DEV_GOAL if new_goal != gtext else DEV_ACCEPT
            return {
                "op": {"new_text": new_text, "acceptance_add": acceptance_add,
                       "acceptance_remove": acceptance_remove},
                "deviation": {"kind": kind, "target": "goal",
                              "from": before, "to": after},
            }

        return self._commit(actor=actor, now=now, op_kind=OP_GOAL,
                            rationale=rationale, work=work)

    def claim_task(
        self, *, task_id: str, actor: str = DEFAULT_ACTOR,
        now: Optional[str] = None,
    ) -> "Plan | Conflict | None":
        now = now or _now_iso()

        def work(con: sqlite3.Connection) -> "dict | Conflict":
            row = _task_row(con, self._pid, task_id)
            if not row:
                return Conflict(0, f"task '{task_id}' does not exist", task_id)
            if row["claimed_by"] and row["claimed_by"] != actor:
                return Conflict(0, f"task '{task_id}' is claimed by "
                                f"{row['claimed_by']}", task_id)
            con.execute(
                "UPDATE task SET claimed_by=?, updated_at=? WHERE plan_id=? AND id=?",
                (actor, now, self._pid, task_id),
            )
            return {"op": {"id": task_id}}

        return self._commit(actor=actor, now=now, op_kind=OP_CLAIM,
                            rationale=None, work=work)

    # -- projections ------------------------------------------------------

    def _write_projections(self, plan: Plan) -> None:
        """Best-effort md/json next to the DB. Failures are swallowed — the DB
        is the source of truth; the file regenerates on the next apply."""
        if not self.direct_write or self._db_path is None:
            return
        stem_path = self._db_path.with_suffix("")
        try:
            stem_path.with_suffix(".json").write_text(
                json.dumps(plan_to_json(plan), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            stem_path.with_suffix(".md").write_text(
                render_markdown(plan), encoding="utf-8",
            )
        except OSError:
            pass  # projection is disposable; never let it corrupt state


def _task_row(con: sqlite3.Connection, pid: Optional[str], task_id: str):
    r = con.execute(
        "SELECT id, title, status, parent, ord, note, origin, added_by, "
        "claimed_by, done_by FROM task WHERE plan_id=? AND id=?", (pid, task_id),
    ).fetchone()
    if r is None:
        return None
    keys = ("id", "title", "status", "parent", "ord", "note", "origin",
            "added_by", "claimed_by", "done_by")
    return dict(zip(keys, r, strict=True))
