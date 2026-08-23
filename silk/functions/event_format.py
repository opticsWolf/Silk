# -*- coding: utf-8 -*-
"""Formatting and counting for the Agent node's ``tool_events`` stream.

Qt-free so the Hook Monitor node's logic is testable headless. Event
dicts carry at least ``event`` (kind), ``ts``, ``run_id`` and ``seq``
(monotonic per run — the dedup key for re-evaluations), plus
kind-specific fields.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any, Optional


def format_event(event: dict[str, Any]) -> str:
    """One human-readable log line for a tool_events dict."""
    kind = str(event.get("event", "?"))
    stamp = time.strftime("%H:%M:%S", time.localtime(event.get("ts", time.time())))
    tool = event.get("tool", "")

    if kind == "run_started":
        body = "▶ run started"
    elif kind == "run_finished":
        rounds = event.get("rounds", "?")
        elapsed = event.get("elapsed_s")
        timing = f", {elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
        body = f"■ run finished — {rounds} round(s){timing}"
    elif kind == "model_request":
        body = f"· model round {event.get('round', '?')}…"
    elif kind == "model_response":
        body = (
            f"· response round {event.get('round', '?')} "
            f"({event.get('chars', 0)} chars)"
        )
    elif kind == "tool_call":
        try:
            args = json.dumps(event.get("args") or {}, ensure_ascii=False)
        except (TypeError, ValueError):
            args = str(event.get("args"))
        if len(args) > 80:
            args = args[:80] + "…"
        body = f"→ {tool} {args}"
    elif kind == "tool_result":
        body = f"← {tool} ({event.get('chars', 0)} chars)"
    elif kind == "tool_denied":
        body = f"✗ {tool} DENIED by role"
    else:
        body = kind
    return f"[{stamp}] {body}"


class EventCounter:
    """Per-kind and per-tool counters over a tool_events stream."""

    def __init__(self) -> None:
        self.kinds: Counter[str] = Counter()
        self.tools: Counter[str] = Counter()

    def record(self, event: dict[str, Any]) -> None:
        kind = str(event.get("event", "?"))
        self.kinds[kind] += 1
        if kind == "tool_call" and event.get("tool"):
            self.tools[str(event["tool"])] += 1

    def clear(self) -> None:
        self.kinds.clear()
        self.tools.clear()

    def summary(self) -> str:
        calls = self.kinds.get("tool_call", 0)
        denied = self.kinds.get("tool_denied", 0)
        runs = self.kinds.get("run_finished", 0)
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
