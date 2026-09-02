# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The waiter both seams are made of (spec D49, D70).

A tool runs on a worker thread. Two things it may need live somewhere
else: a **human's answer** (D30/D48 -- the approval gate) and a **main
thread** (D70 -- the canvas, which Qt says may only be touched there).
Both are the same shape: emit a request, block the worker, have someone
else commit an outcome, wake up and act on it. D49 was specified as a
general waiter for exactly this reason, and D70 is its second user.

So the mechanism lives here once and the two seams differ only in what a
request and an outcome *are*, and in who resolves them: a person, or the
main-thread event loop.

**The ordering rule** is the whole point of concentrating this:

    Write the outcome under the lock, then set the event; the waiter
    re-reads under the lock before acting.

Without it, Stop, a timeout and a real answer race into one "something
woke me" and the seam cannot say which happened -- which is what turns
D42's race catalog into flaky tests instead of deterministic ones.

**Every failure path fails closed** (D36). No resolver at all (a headless
evaluation, a subagent with no UI, a graph with no canvas), a resolver
destroyed mid-request, a transport that raises, a timeout: all four wake
the waiter with a *named cause*, and the seam's owner turns that name
into a refusal. Nothing here can produce consent, or a mutation, that
nobody performed.
"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Callable, Optional, TypeVar

from weave.logger import get_logger

log = get_logger("SilkSeam")

#: Why a waiter woke. Only :data:`CAUSE_ANSWERED` means somebody actually
#: did the thing; the rest are D36's failure paths, kept distinct because
#: "denied by the user" and "nobody was there to ask" are different facts
#: even when they have the same effect.
CAUSE_ANSWERED = "answered"
CAUSE_CANCELLED = "cancelled"
CAUSE_TIMEOUT = "timeout"
CAUSE_NO_ANSWERER = "no_answerer"
CAUSE_TRANSPORT = "transport_error"

Outcome = TypeVar("Outcome")


class DriveGate:
    """Park the seam at named checkpoints so a race can be driven in order.

    Pi's test-mode gate (spec D42). A blocked worker thread, another
    thread resolving, and Stop and the timeout racing that resolution:
    five races, ten orderings, and no reliable way to exercise them except
    by holding one side still. Invariant fixtures do not test races.

    Checkpoints: ``ask`` (request recorded, before it is emitted), ``wait``
    (emitted, before blocking), ``resolve`` (inside resolve, under the
    lock, before the write) and ``wake`` (the event fired, before
    re-reading). A checkpoint nobody armed is a no-op, so production pays
    one dict lookup.
    """

    def __init__(self, *checkpoints: str) -> None:
        self._gates: dict[str, threading.Event] = {
            name: threading.Event() for name in checkpoints
        }
        self._arrived: dict[str, threading.Event] = {
            name: threading.Event() for name in checkpoints
        }

    def wait_at(self, name: str) -> None:
        """Called by the seam; blocks if a test armed this checkpoint."""
        gate = self._gates.get(name)
        if gate is None:
            return
        self._arrived[name].set()
        gate.wait()

    def arrived_at(self, name: str, timeout: float = 2.0) -> bool:
        """Wait until the seam reaches *name*. The test's synchronisation."""
        event = self._arrived.get(name)
        return bool(event and event.wait(timeout))

    def release(self, name: str) -> None:
        """Let the seam past *name*."""
        gate = self._gates.get(name)
        if gate is not None:
            gate.set()

    def release_all(self) -> None:
        for gate in self._gates.values():
            gate.set()


class BlockingSeam:
    """Emit a request, block the worker, wake on a committed outcome.

    *deliver* is how a request reaches whoever resolves it: a callable
    taking the request object. ``None`` means there is nobody -- and every
    request then fails immediately with :data:`CAUSE_NO_ANSWERER` rather
    than blocking, because waiting out a timeout for a resolver that does
    not exist spends five minutes to reach the same answer.

    Subclasses (and callers) decide what a request and an outcome are.
    This object owns only correlation, the ordering rule, cancellation and
    the timeout -- the four parts that are each easy to get wrong, and
    easy to get wrong *differently* at every call site.
    """

    def __init__(
        self,
        deliver: Optional[Callable[[Any], Any]] = None,
        *,
        timeout_s: float,
        drive: Optional[DriveGate] = None,
    ) -> None:
        self._deliver = deliver
        self._timeout_s = float(timeout_s)
        self._drive = drive
        self._lock = threading.Lock()
        #: id -> outcome, for answers that have landed.
        self._answers: dict[str, Any] = {}
        #: id -> request, for what is currently outstanding.
        self._open: dict[str, Any] = {}
        #: id -> the event its waiter sleeps on. One event per request
        #: rather than one broadcast: a shared event would wake every
        #: waiter for somebody else's answer, and re-arming it correctly
        #: under contention is exactly the kind of thing this object
        #: exists to not have five copies of.
        self._waiters: dict[str, threading.Event] = {}
        #: Set once by cancel(); every waiter, current and future, fails.
        self._cancelled: Optional[str] = None
        self._closed = False

    # -- properties -------------------------------------------------------

    @property
    def can_ask(self) -> bool:
        """Whether this run has a resolver at all (D36's first failure)."""
        return self._deliver is not None and not self._closed

    def attach(self, deliver: Optional[Callable[[Any], Any]]) -> None:
        """Name the resolver after construction.

        The Qt resolver needs the seam (to answer it) and the seam needs
        the resolver (to reach it), so one of the two has to be told
        second. Doing it through a method rather than by writing the
        attribute keeps "who resolves this seam" a stated part of the
        contract, and keeps the fail-closed reading of `None` in one place.
        """
        self._deliver = deliver

    def outstanding(self) -> list:
        """Requests still waiting, for a UI that renders after the ask."""
        with self._lock:
            return list(self._open.values())

    # -- the worker side --------------------------------------------------

    def submit(
        self,
        request_id: str,
        request: Any,
        *,
        failed: Callable[[str, str], Outcome],
        timeout: Optional[float] = None,
    ) -> Outcome:
        """Ask, block, and return what happened. **Worker thread.**

        *failed(cause, reason)* builds the outcome for each of D36's four
        failure paths, so this returns rather than raises on every one of
        them: the caller is producing a tool result, and an exception
        there would become a crash where a refusal was meant.
        """
        deadline = self._timeout_s if timeout is None else float(timeout)

        with self._lock:
            if self._cancelled is not None:
                return failed(CAUSE_CANCELLED, self._cancelled)
            if self._deliver is None or self._closed:
                return failed(CAUSE_NO_ANSWERER, "")
            self._open[request_id] = request
            woken = threading.Event()
            self._waiters[request_id] = woken

        self.park("ask")

        try:
            self._deliver(request)
        except Exception as exc:  # noqa: BLE001 -- the transport is arbitrary
            log.warning(f"seam request could not be delivered: {exc}")
            self.forget(request_id)
            return failed(CAUSE_TRANSPORT, str(exc))

        self.park("wait")

        fired = woken.wait(deadline if deadline > 0 else 0.0)
        self.park("wake")

        # Re-read under the lock. The event only says *something* happened;
        # what happened is the state the writer committed before setting
        # it, and that is what makes an answer, Stop and a timeout three
        # distinguishable wakeups rather than one ambiguous one.
        with self._lock:
            answer = self._answers.pop(request_id, None)
            cancelled = self._cancelled
        self.forget(request_id)

        if answer is not None:
            return answer
        if cancelled is not None:
            return failed(CAUSE_CANCELLED, cancelled)
        if not fired:
            return failed(CAUSE_TIMEOUT, "")
        # Woken with nothing committed: the writer's state is gone or the
        # seam closed under us. Fail closed like everything else here.
        return failed(CAUSE_NO_ANSWERER, "")

    # -- the resolver side ------------------------------------------------

    def commit(self, request_id: str, outcome: Any) -> bool:
        """Deliver an outcome. Idempotent per id.

        Returns False for a second outcome on an id that is already
        resolved (or was never asked) -- D42's fifth race. A no-op
        reporting "already resolved" is the only safe answer: the first
        outcome may already have executed something.
        """
        self.park("resolve")
        with self._lock:
            if self._cancelled is not None:
                return False
            if request_id not in self._open:
                return False
            if request_id in self._answers:
                return False
            # Write the outcome under the lock ...
            self._answers[request_id] = outcome
            waiter = self._waiters.get(request_id)
        # ... then wake. Never the other way round.
        if waiter is not None:
            waiter.set()
        return True

    def cancel(self, reason: str = "stopped") -> None:
        """Stop every waiter, now and later.

        Stop must call this *directly* rather than relying on a consumer
        loop: that loop is inside a single ``next()`` while the worker
        blocks and is not polling anything (D38, G8).

        The reason is recorded before the wake, so no waiter can observe a
        wakeup without being able to say why it happened.
        """
        with self._lock:
            if self._cancelled is None:
                self._cancelled = reason or "stopped"
            waiters = list(self._waiters.values())
        for waiter in waiters:
            waiter.set()

    def close(self, reason: str = "the run ended") -> None:
        """End of run: no further requests are served, only failed.

        Covers D36's second failure -- the widget destroyed, or the graph
        closed, while a request is outstanding.
        """
        with self._lock:
            self._closed = True
            if self._cancelled is None:
                self._cancelled = reason
            waiters = list(self._waiters.values())
        for waiter in waiters:
            waiter.set()

    # -- internals, shared with subclasses --------------------------------

    def forget(self, request_id: str) -> None:
        with self._lock:
            self._open.pop(request_id, None)
            self._waiters.pop(request_id, None)

    def park(self, checkpoint: str) -> None:
        if self._drive is not None:
            self._drive.wait_at(checkpoint)


def new_request_id() -> str:
    """A fresh correlation id. Short enough to read in a log line."""
    return uuid.uuid4().hex[:12]
