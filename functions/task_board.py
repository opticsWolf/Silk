# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The multi-agent progress projection behind the Task Hub node (spec D58).

N independent top-level agents share no event port and never will, so
there is nowhere to watch them all at once -- except the place they all
write to. Every agent's plan is a ``plan-*.db`` under a sandbox root, and
the ToolBox node already aggregates the graph's roots onto one wire. That
makes the projection a database read: scan the roots, group each plan's
tasks by status, and put ``claimed_by`` on the task that carries it.

``claimed_by`` is the point. The store has recorded it since the schema
was written and no view has ever shown it, which is why "who is doing
what" has been unanswerable in a graph running four agents.

Everything here is pure: rows in, dict/markdown out, no Qt, no store
writes. The hub node is the display; this is what it displays.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .stream_events import EventType
from .task_store import Plan, SqliteTaskStore, open_task_ids

#: Lanes, in board order. ``dropped`` is deliberately last and usually
#: empty -- a rescoped task is history, not work.
LANES: tuple[str, ...] = ("blocked", "in_progress", "pending", "done", "dropped")

#: Lane -> heading glyph shortcode (mordant expands these on render, so the
#: markdown stays greppable -- same convention as the plan renderer).
_LANE_GLYPH: dict[str, str] = {
    "blocked": ":no_entry:",
    "in_progress": ":hourglass_flowing_sand:",
    "pending": ":white_large_square:",
    "done": ":white_check_mark:",
    "dropped": ":wastebasket:",
}

#: What a task with no ``claimed_by`` is shown as.
UNCLAIMED = "unclaimed"


# ── Scanning ─────────────────────────────────────────────────────────────

def scan_roots(roots: Iterable[Any]) -> list[dict]:
    """Every plan under every root, newest first, de-duplicated by file.

    Two roots may overlap (a graph wiring both a project and a subdirectory
    of it), and the same plan reached twice must be one lane, not two --
    which is why identity is the resolved ``db_path`` and not the root it
    was found under.
    """
    seen: set[str] = set()
    rows: list[dict] = []
    for root in roots or ():
        text = str(root or "").strip()
        if not text:
            continue
        try:
            found = SqliteTaskStore.scan_all(text)
        except OSError:
            # An unreachable root is one missing lane, not a dead board.
            continue
        for row in found:
            if row["db_path"] in seen:
                continue
            seen.add(row["db_path"])
            rows.append(row)
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def load_plan(row: dict) -> Optional[Plan]:
    """The plan a scan row points at, or ``None`` if it will not open."""
    try:
        return SqliteTaskStore(row["root"], db_path=row["db_path"]).load()
    except Exception:      # noqa: BLE001 - a bad file is an empty lane
        return None


# ── Projection ───────────────────────────────────────────────────────────

def lane_of(plan: Plan) -> dict[str, list[dict]]:
    """Tasks grouped by status, each carrying who claimed it."""
    lanes: dict[str, list[dict]] = {name: [] for name in LANES}
    for task in sorted(plan.tasks, key=lambda t: t.order):
        lanes.setdefault(task.status, []).append({
            "id": task.id,
            "title": task.title,
            "status": task.status,
            "actor": task.claimed_by or task.done_by or "",
            "note": task.note,
        })
    return lanes


def actors_of(plan: Plan) -> list[str]:
    """Which agents appear on this plan, in first-seen order.

    A plan with two actors is the multi-agent case working; a plan with
    one is an agent working alone. Both are worth seeing at a glance,
    which is why this is a list and not a count.
    """
    order: list[str] = []
    for task in sorted(plan.tasks, key=lambda t: t.order):
        for who in (task.claimed_by, task.done_by):
            if who and who not in order:
                order.append(who)
    return order


def board(rows: Iterable[dict]) -> dict:
    """The whole projection as plain data (the node's ``plans_json``).

    A row that will not open still becomes a lane -- with its error and no
    tasks. A board that hides the plan it could not read is a board that
    lies about how many plans there are.
    """
    plans: list[dict] = []
    for row in rows or ():
        plan = load_plan(row)
        entry: dict[str, Any] = {
            "db_path": row.get("db_path", ""),
            "root": row.get("root", ""),
            "label": row.get("label", ""),
            "plan_id": row.get("plan_id", ""),
            "goal": row.get("goal", ""),
            "updated_at": row.get("updated_at", ""),
            "tasks": row.get("tasks", 0),
            "open_tasks": row.get("open_tasks", 0),
            "actors": [],
            "lanes": {name: [] for name in LANES},
        }
        if "error" in row:
            entry["error"] = row["error"]
        if plan is None:
            entry.setdefault("error", "the plan file did not open")
        else:
            entry.update(revision=plan.revision, actors=actors_of(plan),
                         lanes=lane_of(plan),
                         open_tasks=len(open_task_ids(plan)),
                         tasks=len(plan.tasks))
        plans.append(entry)

    return {
        "plans": plans,
        "plan_count": len(plans),
        "open_tasks": sum(p["open_tasks"] for p in plans),
        "actors": sorted({a for p in plans for a in p["actors"]}),
    }


# ── Rendering ────────────────────────────────────────────────────────────

def render_board(data: dict) -> str:
    """Markdown for the board. One section per plan, one list per lane."""
    plans = data.get("plans") or []
    if not plans:
        return (
            "*No plans under these roots yet.*\n\n"
            "Wire `Silk ToolBox.root_paths` here; plans appear as agents "
            "create them."
        )

    out: list[str] = []
    actors = data.get("actors") or []
    out.append(
        f"**{len(plans)} plan(s)** · **{data.get('open_tasks', 0)} open** · "
        f"{len(actors)} agent(s): {', '.join(actors) if actors else '—'}"
    )
    for plan in plans:
        out.append("")
        out.append(f"### {plan['label']} — {plan.get('goal') or '*no goal yet*'}")
        if plan.get("error"):
            out.append(f"> :warning: {plan['error']}")
            continue
        badge = ", ".join(plan.get("actors") or []) or UNCLAIMED
        out.append(
            f"`{plan.get('open_tasks', 0)}/{plan.get('tasks', 0)} open` · "
            f"rev {plan.get('revision', 0)} · {badge}"
        )
        for lane in LANES:
            tasks = plan["lanes"].get(lane) or []
            if not tasks:
                continue
            out.append("")
            out.append(f"**{_LANE_GLYPH.get(lane, '')} {lane}** ({len(tasks)})")
            for task in tasks:
                who = f" — *{task['actor']}*" if task["actor"] else ""
                out.append(f"- {task['title']}{who}")
    return "\n".join(out)


# ── Pending decisions ────────────────────────────────────────────────────

class PendingDecisions:
    """How many agents are blocked waiting on a human, right now.

    D58 lets the hub **count** mid-run requests; D59 is emphatic that only
    the asking node -- or its dock mirror -- may answer one. So this holds
    correlation ids and nothing that could resolve them.

    A request that is never answered (timeout, cancelled run, closed
    graph) would otherwise pin the count above zero forever, so a run that
    finishes clears whatever it still had outstanding.
    """

    def __init__(self) -> None:
        self._open: dict[str, str] = {}     # correlation id -> run id

    def __len__(self) -> int:
        return len(self._open)

    @property
    def count(self) -> int:
        return len(self._open)

    def clear(self) -> None:
        self._open.clear()

    def record(self, event: Any) -> bool:
        """Fold one event in. Returns True when the count changed."""
        if not isinstance(event, dict):
            return False
        kind = event.get("type")
        before = len(self._open)

        if kind == EventType.DECISION_REQUEST.value:
            key = _correlation(event)
            if key is not None:
                self._open[key] = str(event.get("run_id") or "")
        elif kind == EventType.DECISION_RESPONSE.value:
            key = _correlation(event)
            if key is not None:
                self._open.pop(key, None)
        elif kind in (EventType.RUN_FINISHED.value, EventType.RUN_RESULT.value):
            run = str(event.get("run_id") or "")
            if run:
                for key in [k for k, r in self._open.items() if r == run]:
                    self._open.pop(key, None)
        return len(self._open) != before

    def waiting(self) -> list[str]:
        """The open correlation ids -- a directory, not a handle."""
        return sorted(self._open)


def _correlation(event: dict) -> Optional[str]:
    """The id that pairs a request with its answer.

    Falls back to the run id: an event without correlation still stands
    for one blocked agent, and dropping it would undercount the very thing
    the port exists to report.
    """
    for field_name in ("decision_id", "correlation_id", "request_id"):
        value = event.get(field_name)
        if value:
            return str(value)
    run = event.get("run_id")
    return str(run) if run else None
