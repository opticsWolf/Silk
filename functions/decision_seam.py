# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

One run-scoped decision seam -- the single place a run blocks on a human
(spec D30, D38, D48, D49).

The gate does not park a call and end the run. It emits a request on the
run's event stream, blocks the tool call on the worker thread, and waits
for a decision delivered back through this object. Approve and the held
call executes in the same run; deny and a refusal becomes that tool's
result. There is no pending-action store, no resume run, no exec handoff.

**Why this is one object.** Four hard parts -- correlation, cancel
ordering, timeout, and the policy snapshot -- are each easy to get wrong
and each easy to get wrong *differently* at every call site. Concentrating
them means the closed method set below is the complete crash-site catalog:

===================  ==========================================  =========
Called by            Method                                      Thread
===================  ==========================================  =========
gate (blocked)       :meth:`DecisionSeam.await_decision`         worker
UI widget            :meth:`DecisionSeam.resolve`                main
Stop handler         :meth:`DecisionSeam.cancel`                 main
===================  ==========================================  =========

**The ordering rule**, which is what makes the four wake causes
*distinguishable* rather than one ambiguous wakeup:

    Write the outcome under the lock, then set the event; the waiter
    re-reads under the lock before acting.

Without it, Stop, a timeout and a real approval race into a single
"something woke me" and the seam cannot say which happened -- which is
also what turns the race catalog (D42) into flaky tests instead of
deterministic ones.

**Every failure path denies (D36).** No answerer at all (a headless graph
evaluation, a subagent with no UI), an answerer destroyed mid-request, a
transport that raises, a timeout: all four produce the same structured
refusal the model sees for an explicit rejection. An absent grant store
grants nothing; an absent answerer approves nothing. Every degradation in
this subsystem points the same way.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from weave.logger import get_logger

log = get_logger("SilkDecision")

#: Why a waiter woke. ``answered`` is the only one a human produced; the
#: rest are the four failure paths of D36, kept distinct because "denied by
#: the user" and "nobody was there to ask" are different facts even though
#: they have the same effect.
CAUSE_ANSWERED = "answered"
CAUSE_CANCELLED = "cancelled"
CAUSE_TIMEOUT = "timeout"
CAUSE_NO_ANSWERER = "no_answerer"
CAUSE_TRANSPORT = "transport_error"

#: The questions one seam serves (D50). All three are the same block, the
#: same correlation id, the same timeout and the same fail-closed rule --
#: only the response payload differs, which is why a second
#: human-in-the-loop question must not build a second waiter.
KIND_APPROVAL = "approval"        # approve/deny a held tool call
KIND_ACKNOWLEDGE = "acknowledge"  # a continue/abort checkpoint (e.g. compaction)
KIND_RELEASE = "release"          # resume a step the human paused

KINDS = (KIND_APPROVAL, KIND_ACKNOWLEDGE, KIND_RELEASE)

#: How long a request waits before denying, when the caller names no
#: timeout. A blocked gate holds the worker thread *and* the exclusive
#: RoleBinding on the toolset, so no other Agent node can use that toolset
#: meanwhile -- the wait must be bounded, and generously rather than
#: tightly, because the cost of expiry is a denial the human did not intend.
DEFAULT_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class DecisionRequest:
    """What is being asked. Emitted as an event; carried back by id."""

    decision_id: str
    kind: str = KIND_APPROVAL
    prompt: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    #: Free-form context for the UI (risk, the diff, the reason it is gated).
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """The answer, or the reason there is none.

    ``approved`` is False for every cause but :data:`CAUSE_ANSWERED` with an
    approval -- there is no path through this object that produces consent
    nobody gave.
    """

    decision_id: str
    kind: str = KIND_APPROVAL
    approved: bool = False
    reason: str = ""
    actor: str = ""
    cause: str = CAUSE_ANSWERED
    #: Set when the answerer said "and don't ask again" -- the gate turns
    #: this into a run-scoped or durable grant (D10); the seam only carries it.
    remember: str = ""

    @property
    def answered(self) -> bool:
        """Whether a human actually decided this."""
        return self.cause == CAUSE_ANSWERED

    def refusal_text(self) -> str:
        """A sentence for the model, saying which of the five happened."""
        if self.approved:
            return "Approved."
        if self.cause == CAUSE_ANSWERED:
            return f"Denied by {self.actor or 'the user'}." + (
                f" Reason: {self.reason}" if self.reason else ""
            )
        return {
            CAUSE_CANCELLED: "Denied: the run was stopped while waiting for "
                             "approval.",
            CAUSE_TIMEOUT: "Denied: no answer arrived before the approval "
                           "request timed out.",
            CAUSE_NO_ANSWERER: "Denied: this run has no interface to ask for "
                               "approval. A durable grant is the way to allow "
                               "this without a human present.",
            CAUSE_TRANSPORT: "Denied: the approval request could not be "
                             "delivered.",
        }.get(self.cause, "Denied.")


class DriveGate:
    """Park the seam at named checkpoints so a race can be driven in order.

    Pi's test-mode gate, narrowed to this seam (spec D42). D30 is Silk's
    first real concurrency surface -- a parked worker thread, a Qt thread
    resolving the decision, and Stop and the timeout racing that resolution
    -- and invariant fixtures do not test races. Five races, ten orderings,
    and no reliable way to exercise them except by holding one side still.

    Checkpoints: ``ask`` (request recorded, before it is emitted), ``wait``
    (emitted, before blocking), ``resolve`` (inside resolve, under the lock,
    before the write) and ``wake`` (the event fired, before re-reading).
    A checkpoint nobody armed is a no-op, so production pays one dict lookup.
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


class DecisionSeam:
    """The run-scoped block. One per run, created by the node, closed over
    by the gate.

    *ask* is how a request reaches a human: a callable taking a
    :class:`DecisionRequest`. ``None`` means there is nobody to ask -- a
    headless evaluation, a subagent -- and every request then denies with
    :data:`CAUSE_NO_ANSWERER` *without blocking*, because waiting out a
    timeout for an answerer that does not exist wastes five minutes to
    reach the same answer.
    """

    def __init__(
        self,
        ask: Optional[Callable[[DecisionRequest], Any]] = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        drive: Optional[DriveGate] = None,
    ) -> None:
        self._ask = ask
        self._timeout_s = float(timeout_s)
        self._drive = drive
        self._lock = threading.Lock()
        #: decision_id -> Decision, for answers that have landed.
        self._answers: dict[str, Decision] = {}
        #: decision_id -> DecisionRequest, for what is currently outstanding.
        self._open: dict[str, DecisionRequest] = {}
        #: decision_id -> the event its waiter sleeps on. One event per
        #: request rather than one broadcast: a shared event would wake
        #: every waiter for somebody else's answer, and re-arming it
        #: correctly under contention is exactly the kind of thing this
        #: object exists to not have five copies of.
        self._waiters: dict[str, threading.Event] = {}
        #: Set once by cancel(); every waiter, current and future, denies.
        self._cancelled: Optional[str] = None
        self._closed = False

    # -- properties -------------------------------------------------------

    @property
    def can_ask(self) -> bool:
        """Whether this run has an answerer at all (D36's first failure)."""
        return self._ask is not None and not self._closed

    def outstanding(self) -> list[DecisionRequest]:
        """Requests still waiting, for a UI that renders after the ask."""
        with self._lock:
            return list(self._open.values())

    # -- the worker side --------------------------------------------------

    def await_decision(
        self,
        request: DecisionRequest,
        timeout: Optional[float] = None,
    ) -> Decision:
        """Ask, block, and return what happened. **Worker thread.**

        Returns rather than raises on every failure path: the caller is a
        tool gate whose job is to produce a tool result, and an exception
        there would become a crash where a refusal was meant (D36).
        """
        deadline = self._timeout_s if timeout is None else float(timeout)

        with self._lock:
            if self._cancelled is not None:
                return self._denied(request, CAUSE_CANCELLED, self._cancelled)
            if self._ask is None or self._closed:
                return self._denied(request, CAUSE_NO_ANSWERER)
            self._open[request.decision_id] = request
            woken = threading.Event()
            self._waiters[request.decision_id] = woken

        self._park("ask")

        try:
            self._ask(request)
        except Exception as exc:  # noqa: BLE001 -- the transport is arbitrary
            log.warning(f"approval request could not be delivered: {exc}")
            self._forget(request.decision_id)
            return self._denied(request, CAUSE_TRANSPORT, str(exc))

        self._park("wait")

        fired = woken.wait(deadline if deadline > 0 else 0.0)
        self._park("wake")

        # Re-read under the lock. The event only says *something* happened;
        # what happened is the state the writer committed before setting it,
        # and that is what makes approve, deny, Stop and timeout four
        # distinguishable wakeups rather than one ambiguous one.
        with self._lock:
            answer = self._answers.pop(request.decision_id, None)
            cancelled = self._cancelled
        self._forget(request.decision_id)

        if answer is not None:
            return answer
        if cancelled is not None:
            return self._denied(request, CAUSE_CANCELLED, cancelled)
        if not fired:
            return self._denied(request, CAUSE_TIMEOUT)
        # Woken with nothing committed: the writer's state is gone or the
        # seam closed under us. Fail closed like everything else here.
        return self._denied(request, CAUSE_NO_ANSWERER)

    # -- the answerer side ------------------------------------------------

    def resolve(self, decision: Decision) -> bool:
        """Deliver an answer. **Main thread.** Idempotent per decision id.

        Returns False for a second decision on an id that is already
        resolved (or was never asked) -- D42's fifth race. A no-op reporting
        "already resolved" is the only safe answer: the first decision may
        already have executed a tool.
        """
        self._park("resolve")
        with self._lock:
            if self._cancelled is not None:
                return False
            if decision.decision_id not in self._open:
                return False
            if decision.decision_id in self._answers:
                return False
            # Write the outcome under the lock ...
            self._answers[decision.decision_id] = decision
            waiter = self._waiters.get(decision.decision_id)
        # ... then wake. Never the other way round.
        if waiter is not None:
            waiter.set()
        return True

    def approve(self, decision_id: str, *, actor: str = "user",
                remember: str = "", kind: str = KIND_APPROVAL) -> bool:
        """Convenience for the common answer."""
        return self.resolve(Decision(
            decision_id=decision_id, kind=kind, approved=True, actor=actor,
            remember=remember,
        ))

    def deny(self, decision_id: str, *, actor: str = "user",
             reason: str = "", kind: str = KIND_APPROVAL) -> bool:
        return self.resolve(Decision(
            decision_id=decision_id, kind=kind, approved=False, actor=actor,
            reason=reason,
        ))

    def cancel(self, reason: str = "stopped") -> None:
        """Stop every waiter, now and later. **Main thread.**

        Stop must call this *directly* rather than relying on the consumer
        loop: that loop is inside a single ``next()`` while the gate blocks
        and is not polling anything (D38, G8).

        The reason is recorded before the wake, so no waiter can observe a
        wakeup without being able to say why it happened.
        """
        with self._lock:
            if self._cancelled is None:
                self._cancelled = reason or "stopped"
            waiters = list(self._waiters.values())
        for waiter in waiters:
            waiter.set()

    def close(self) -> None:
        """End of run: no further requests are asked, only denied.

        Covers D36's second failure -- the widget destroyed, or the graph
        closed, while a request is outstanding.
        """
        with self._lock:
            self._closed = True
            if self._cancelled is None:
                self._cancelled = "the run ended"
            waiters = list(self._waiters.values())
        for waiter in waiters:
            waiter.set()

    # -- internals --------------------------------------------------------

    def _forget(self, decision_id: str) -> None:
        with self._lock:
            self._open.pop(decision_id, None)
            self._waiters.pop(decision_id, None)

    def _park(self, checkpoint: str) -> None:
        if self._drive is not None:
            self._drive.wait_at(checkpoint)

    @staticmethod
    def _denied(
        request: DecisionRequest, cause: str, reason: str = "",
    ) -> Decision:
        return Decision(
            decision_id=request.decision_id, kind=request.kind,
            approved=False, cause=cause, reason=reason,
        )


def new_decision_id() -> str:
    """A fresh correlation id. Short enough to read in a log line."""
    return uuid.uuid4().hex[:12]
