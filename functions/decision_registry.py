# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Who is waiting on a human, right now (spec D59).

With N agents, the answering widgets built into each Agent node (D48) are
scattered across N node bodies, and finding the one that is blocked means
hunting the canvas. Centralising that must not reintroduce the answerer
node D51 rejected -- and does not have to, because the thing being
centralised is a *directory*, not a channel.

This registry is that directory. A run-scoped seam registers when it asks
and unregisters when the answer, timeout or cancel lands; the Decision
Inbox dock reads it and shows one row per waiting agent. It holds **weak**
references to the asking node, so seam lifetime (D49) is unchanged: the
registry can never be the reason a node or a run stays alive.

What it deliberately does not hold is anything that resolves a decision.
An entry knows which node asked, and answering goes through *that node's*
own surface -- which is exactly what a mirrored button does. The hub node
(D58) may count these; only the asking node's UI, or a mirror of it, may
answer one (I12).
"""
from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Optional

#: The kind of decision, when the request does not say.
DEFAULT_KIND = "approval"


@dataclass(frozen=True)
class DecisionEntry:
    """One agent, waiting. The node reference is weak, by design."""

    decision_id: str
    run_id: str
    kind: str
    prompt: str
    agent: str
    tool_name: str
    _node_ref: Any = None

    @property
    def node(self) -> Optional[Any]:
        """The node that asked, or ``None`` once it is gone."""
        if self._node_ref is None:
            return None
        return self._node_ref()

    @property
    def alive(self) -> bool:
        return self._node_ref is None or self._node_ref() is not None

    def title(self) -> str:
        return f"{self.agent or 'agent'} · {self.kind}"

    def detail(self) -> str:
        tool = f" ({self.tool_name})" if self.tool_name else ""
        return f"{self.prompt}{tool}"


class DecisionRegistry:
    """Session-scoped directory of open decision requests.

    Session-scoped and not global-per-process in spirit: a graph is one
    session, and the dock that reads this belongs to the window showing
    that graph. The module-level instance below is the one the Agent node
    and the dock share; tests build their own.

    Thread-safe, because registration happens on whichever thread the
    seam asked from while the dock reads on the main thread.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, DecisionEntry] = {}
        self._listeners: list[Callable[[], None]] = []

    # ── membership ────────────────────────────────────────────────────

    def register(self, request: Any, *, node: Any = None,
                 agent: str = "", run_id: str = "") -> Optional[DecisionEntry]:
        """Record one open request. Returns the entry, or ``None``.

        A request with no id cannot be unregistered later, and an entry
        that can never leave is worse than one that never arrives -- the
        dock would offer a button for a decision nobody is waiting on.
        """
        fields = _fields(request)
        decision_id = fields.get("decision_id") or ""
        if not decision_id:
            return None

        entry = DecisionEntry(
            decision_id=str(decision_id),
            run_id=str(fields.get("run_id") or run_id or ""),
            kind=str(fields.get("kind") or DEFAULT_KIND),
            prompt=str(fields.get("prompt") or "Approve?"),
            agent=str(agent or fields.get("agent") or ""),
            tool_name=str(fields.get("tool_name") or ""),
            _node_ref=weakref.ref(node) if node is not None else None,
        )
        with self._lock:
            self._entries[entry.decision_id] = entry
        self._notify()
        return entry

    def unregister(self, decision_id: str) -> bool:
        """Drop one request -- answered, timed out or cancelled."""
        with self._lock:
            gone = self._entries.pop(str(decision_id), None) is not None
        if gone:
            self._notify()
        return gone

    def clear_run(self, run_id: str) -> int:
        """Drop everything a run left behind when it ended.

        A run that is stopped mid-question never answers it, and a row for
        an agent that is no longer running is a button that does nothing.
        """
        run_id = str(run_id)
        with self._lock:
            doomed = [key for key, entry in self._entries.items()
                      if entry.run_id == run_id]
            for key in doomed:
                self._entries.pop(key, None)
        if doomed:
            self._notify()
        return len(doomed)

    def clear(self) -> None:
        with self._lock:
            had = bool(self._entries)
            self._entries.clear()
        if had:
            self._notify()

    # ── reading ───────────────────────────────────────────────────────

    def entries(self) -> list[DecisionEntry]:
        """Open requests whose asking node still exists.

        Pruning here rather than on a timer is what keeps the weak
        reference honest: a deleted node's row disappears the next time
        anyone looks, without the registry having to watch for it.
        """
        with self._lock:
            dead = [key for key, entry in self._entries.items()
                    if not entry.alive]
            for key in dead:
                self._entries.pop(key, None)
            live = list(self._entries.values())
        if dead:
            self._notify()
        return live

    def get(self, decision_id: str) -> Optional[DecisionEntry]:
        with self._lock:
            return self._entries.get(str(decision_id))

    def __len__(self) -> int:
        return len(self.entries())

    # ── change notification ───────────────────────────────────────────

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call *listener* whenever the set changes. Returns an unsubscribe."""
        with self._lock:
            self._listeners.append(listener)

        def _off() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)
        return _off

    def _notify(self) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener()
            except Exception:      # noqa: BLE001 - a bad listener is not a bug
                pass               #                in the registry


def _fields(request: Any) -> dict:
    """Read a request whether it arrives as a dict or a DecisionRequest."""
    if isinstance(request, dict):
        return request
    return {
        name: getattr(request, name, None)
        for name in ("decision_id", "run_id", "kind", "prompt", "tool_name",
                     "agent")
    }


#: The session's registry -- what the Agent node writes to and the Decision
#: Inbox dock reads. One per process because one app window shows one
#: canvas; tests construct their own rather than sharing this.
REGISTRY = DecisionRegistry()
