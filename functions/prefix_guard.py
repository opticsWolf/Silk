# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Invariant I11: the model-visible prefix grows only at the tail.

Between two requests in one run, the earlier request's message sequence
must be a **prefix** of the later one. The exception is compaction, which
is the single deliberate invalidation and says so.

*Why this is worth a guard rather than a comment.* The rule is invisible
at the call site: nothing in ``build_messages`` breaks if a section starts
rendering the time of day, if a tool schema is added mid-run, or if a hook
edits an already-sent message in place. What breaks is the backend's KV
cache -- llama.cpp keeps the longest common prefix between the new prompt
and the last one it evaluated and re-evaluates only the suffix, so a
change at position *k* costs a re-prefill of everything after *k*. The
failure is silent, it is expensive in proportion to how far back it
happened, and it looks exactly like the model being slow (D41).

So the guard records what each request looked like and reports where the
next one stopped agreeing. It reports; it does not repair. A prefix break
is a bug in whatever produced the message list, and hiding it behind an
automatic fix would leave the cost in place and remove the signal.

Three kinds, because they have three different causes:

``system``
    The system prompt changed. Almost always volatile content -- a
    timestamp, a plan summary, a tool list rendered into prose -- and the
    most expensive kind, since it invalidates everything.
``tools``
    The advertised tool schemas changed. Deferred capability loading does
    this by design; it is legal, and worth seeing, because the cost lands
    on the next request rather than the one that loaded the capability.
``history``
    An already-sent message was edited or dropped. The one kind that is
    never intentional outside compaction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from weave.logger import get_logger

log = get_logger("SilkPrefix")

KIND_SYSTEM = "system"
KIND_TOOLS = "tools"
KIND_HISTORY = "history"

#: Why a break was allowed rather than reported.
REASON_COMPACTION = "compaction"


@dataclass(frozen=True)
class PrefixBreak:
    """One place where a request stopped extending its predecessor."""

    kind: str
    #: Which request this was, counting from zero within the run.
    request_index: int
    #: For ``history``, the message index that changed; -1 otherwise.
    position: int = -1
    detail: str = ""

    def __str__(self) -> str:      # pragma: no cover - log text only
        where = f" at message {self.position}" if self.position >= 0 else ""
        return (f"prefix break ({self.kind}) on request {self.request_index}"
                f"{where}: {self.detail}")


def _fingerprint(message: Any) -> str:
    """A stable, comparable rendering of one message.

    ``json.dumps(sort_keys=True)`` rather than the object itself: two
    messages that differ only in key order render the same prompt, and a
    guard that called that a break would cry wolf on a dict rebuild.
    """
    try:
        return json.dumps(message, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(message)


@dataclass
class PrefixGuard:
    """Watches one run's requests and records where the prefix broke.

    Cheap enough to leave on: one JSON rendering per message per request,
    against a request that is about to cost orders of magnitude more.
    """

    #: Every break seen this run, in order.
    breaks: list[PrefixBreak] = field(default_factory=list)
    #: Requests observed so far.
    requests: int = 0

    _system: Optional[str] = None
    _tools: Optional[str] = None
    _messages: list[str] = field(default_factory=list)
    _forgiven: Optional[str] = None

    # -- the deliberate invalidation ---------------------------------------

    def note_compaction(self) -> None:
        """Declare the next break intentional (D24/D41).

        Compaction rewrites the head of the context on purpose. Saying so
        keeps the one legitimate break out of the report, which is what
        makes the rest of the report worth reading.
        """
        self._forgiven = REASON_COMPACTION

    # -- observation --------------------------------------------------------

    def observe(
        self,
        messages: Sequence[Any],
        *,
        system_prompt: str = "",
        tools: Any = None,
    ) -> Optional[PrefixBreak]:
        """Record one request; return the break it introduced, if any.

        Only the *first* break of a request is reported. Once the prefix
        has diverged, everything after it is re-evaluated regardless, so
        the later differences are consequences rather than causes.
        """
        rendered = [_fingerprint(m) for m in messages]
        tool_sig = _fingerprint(tools) if tools else ""
        index = self.requests
        self.requests += 1

        first = self._system is None
        previous, previous_tools, previous_messages = (
            self._system, self._tools, self._messages)
        self._system, self._tools, self._messages = (
            system_prompt, tool_sig, rendered)

        if first:
            self._forgiven = None
            return None

        forgiven, self._forgiven = self._forgiven, None
        found = self._compare(index, system_prompt, previous, tool_sig,
                              previous_tools, rendered, previous_messages)
        if found is None or forgiven is not None:
            return None
        self.breaks.append(found)
        # A warning, not an exception: the request is still correct, it is
        # merely about to cost far more than it should, and a run that died
        # over a cache miss would be the worse trade.
        log.warning(str(found))
        return found

    def _compare(
        self, index: int, system: str, previous_system: Optional[str],
        tools: str, previous_tools: Optional[str],
        messages: list[str], previous_messages: list[str],
    ) -> Optional[PrefixBreak]:
        if system != previous_system:
            return PrefixBreak(
                KIND_SYSTEM, index,
                detail="the system prompt did not render byte-identically; "
                       "the whole context is re-prefilled",
            )
        if tools != previous_tools:
            return PrefixBreak(
                KIND_TOOLS, index,
                detail="the advertised tool schemas changed between requests",
            )
        if len(messages) < len(previous_messages):
            return PrefixBreak(
                KIND_HISTORY, index, position=len(messages),
                detail=f"the history shrank from {len(previous_messages)} to "
                       f"{len(messages)} messages",
            )
        # strict=False on purpose: the new list is the longer one by the
        # check above, and its extra tail is exactly what growth looks like.
        for position, (now, before) in enumerate(
            zip(messages, previous_messages, strict=False)
        ):
            if now != before:
                return PrefixBreak(
                    KIND_HISTORY, index, position=position,
                    detail="an already-sent message was rewritten",
                )
        return None

    # -- reporting ----------------------------------------------------------

    @property
    def clean(self) -> bool:
        """Whether every request so far extended its predecessor."""
        return not self.breaks

    def report(self) -> str:
        """One line per break, for a log or a test failure message."""
        if self.clean:
            return f"{self.requests} request(s), prefix intact"
        return "\n".join(str(b) for b in self.breaks)
