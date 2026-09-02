# -*- coding: utf-8 -*-
"""Formatting and counting for the Agent node's ``events`` stream.

Qt-free so the Hook Monitor node's logic is testable headless. Wire events
carry ``type`` (a member of :class:`EventType`), ``ts``, ``run_id`` and
``seq`` (monotonic per run — the dedup key for re-evaluations), plus the
agent identity and the event's own fields.

One vocabulary, so one formatter (spec D2): what used to be three streams
with three shapes is a single table keyed by type, and an unknown type
renders as its own name rather than being silently dropped.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any, Optional

from .stream_events import EventType

#: The types the monitor counts as "a tool was called". Kept as data so the
#: summary line does not have to guess from strings at three call sites.
_CALL = EventType.TOOL_CALL.value
_DENIED = EventType.TOOL_DENIED.value
_FINISHED = EventType.RUN_FINISHED.value


def _args_digest(args: Any, limit: int = 80) -> str:
    try:
        text = json.dumps(args or {}, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(args)
    return text[:limit] + "…" if len(text) > limit else text


def format_event(event: dict[str, Any]) -> str:
    """One human-readable log line for a wire event."""
    kind = str(event.get("type", "?"))
    stamp = time.strftime("%H:%M:%S", time.localtime(event.get("ts", time.time())))
    tool = event.get("tool_name", "")

    if kind == EventType.RUN_START.value:
        window = event.get("context_length")
        room = f" — context {window}" if window else ""
        body = f"▶ run started{room}"
    elif kind == EventType.RUN_FINISHED.value:
        rounds = event.get("rounds", "?")
        elapsed = event.get("elapsed_s")
        timing = f", {elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
        body = f"■ run finished — {rounds} round(s){timing}"
    elif kind == EventType.RUN_RESULT.value:
        body = f"✓ {event.get('outcome', 'completed')}"
    elif kind == EventType.MODEL_REQUEST.value:
        body = f"· model round {event.get('round', '?')}…"
    elif kind == EventType.MODEL_RESPONSE.value:
        body = (
            f"· response round {event.get('round', '?')} "
            f"({event.get('chars', 0)} chars)"
        )
    elif kind == EventType.TOOL_CALL.value:
        body = f"→ {tool} {_args_digest(event.get('tool_args'))}"
    elif kind == EventType.TOOL_RESULT.value:
        mark = "⚠" if event.get("error") else "←"
        body = f"{mark} {tool} ({event.get('chars', 0)} chars)"
    elif kind == EventType.TOOL_DENIED.value:
        body = f"✗ {tool} DENIED by role"
    elif kind == EventType.PLAN.value:
        body = f"◆ plan revision {event.get('revision', '?')}"
    elif kind == EventType.COMPACTION.value:
        body = (
            f"⇲ compacted {event.get('turns_dropped', 0)} turn(s): "
            f"{event.get('tokens_before', '?')} → {event.get('tokens_after', '?')}"
        )
    elif kind == EventType.DECISION_REQUEST.value:
        body = f"? {event.get('kind', 'approval')} needed — {event.get('prompt', '')}"
    elif kind == EventType.DECISION_RESPONSE.value:
        verdict = "approved" if event.get("approved") else "denied"
        body = f"! {event.get('kind', 'approval')} {verdict}"
    elif kind == EventType.ERROR.value:
        body = f"✗ {event.get('context', 'error')}: {event.get('error', '')}"
    elif kind == EventType.USAGE_LIMIT.value:
        body = (
            f"⊘ {event.get('limit_type', 'limit')} reached "
            f"({event.get('current_value', '?')}/{event.get('limit_value', '?')})"
        )
    elif kind == EventType.REFLECTION.value:
        body = (
            f"↺ retry {event.get('retry_count', 0) + 1}/"
            f"{event.get('max_retries', '?')} — {event.get('error_type', '')}"
        )
    elif kind == EventType.WORKER.value:
        body = (
            f"⇢ {event.get('worker', '?')}: {event.get('event_type', '')} "
            f"{event.get('digest', '')}".rstrip()
        )
    elif kind == EventType.CHAT_TURN.value:
        body = f"💬 turn ({len(event.get('ai') or '')} chars)"
    else:
        body = kind

    who = event.get("agent")
    prefix = f"[{stamp}]" if not who else f"[{stamp}] {who} ·"
    return f"{prefix} {body}"


class EventCounter:
    """Per-type and per-tool counters over an ``events`` stream."""

    def __init__(self) -> None:
        self.kinds: Counter[str] = Counter()
        self.tools: Counter[str] = Counter()

    def record(self, event: dict[str, Any]) -> None:
        kind = str(event.get("type", "?"))
        self.kinds[kind] += 1
        if kind == _CALL and event.get("tool_name"):
            self.tools[str(event["tool_name"])] += 1

    def clear(self) -> None:
        self.kinds.clear()
        self.tools.clear()

    def summary(self) -> str:
        calls = self.kinds.get(_CALL, 0)
        denied = self.kinds.get(_DENIED, 0)
        runs = self.kinds.get(_FINISHED, 0)
        per_tool = ", ".join(f"{n}×{c}" for n, c in self.tools.most_common()) or "—"
        text = f"runs: {runs} · tool calls: {calls}"
        if denied:
            text += f" · denied: {denied}"
        return f"{text}\n{per_tool}"

    def as_dict(self) -> dict[str, Any]:
        return {"kinds": dict(self.kinds), "tools": dict(self.tools)}


def event_key(event: dict[str, Any]) -> Optional[tuple[str, int]]:
    """Dedup key for graph re-evaluations; None if the event carries none."""
    run_id = event.get("run_id")
    seq = event.get("seq")
    if run_id is None or seq is None:
        return None
    return (str(run_id), int(seq))
