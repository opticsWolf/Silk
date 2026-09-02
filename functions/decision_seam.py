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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from weave.logger import get_logger

# The waiter itself lives in `blocking_seam.py` -- one mechanism, two
# resolvers (a human here, the Qt main thread in `main_thread_call.py`),
# which is what D49 was specified as a *general* waiter for (D70).
from .blocking_seam import (
    CAUSE_ANSWERED, CAUSE_CANCELLED, CAUSE_NO_ANSWERER, CAUSE_TIMEOUT,
    CAUSE_TRANSPORT, BlockingSeam, DriveGate, new_request_id,
)

log = get_logger("SilkDecision")

__all__ = [
    "CAUSE_ANSWERED", "CAUSE_CANCELLED", "CAUSE_NO_ANSWERER",
    "CAUSE_TIMEOUT", "CAUSE_TRANSPORT", "DEFAULT_TIMEOUT_S", "KINDS",
    "KIND_ACKNOWLEDGE", "KIND_APPROVAL", "KIND_RELEASE", "Decision",
    "DecisionRequest", "DecisionSeam", "DriveGate", "new_decision_id",
]

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


class DecisionSeam(BlockingSeam):
    """The run-scoped block on a *human*. One per run, created by the node,
    closed over by the gate.

    The waiting is :class:`BlockingSeam`'s (D49, D70); what this class adds
    is the vocabulary of the question -- a :class:`DecisionRequest` in, a
    :class:`Decision` out, and the rule that no path through it produces
    consent nobody gave.

    *ask* is how a request reaches a human. ``None`` means there is nobody
    to ask -- a headless evaluation, a subagent -- and every request then
    denies with :data:`CAUSE_NO_ANSWERER` *without blocking*.
    """

    def __init__(
        self,
        ask: Optional[Callable[[DecisionRequest], Any]] = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        drive: Optional[DriveGate] = None,
    ) -> None:
        super().__init__(ask, timeout_s=timeout_s, drive=drive)

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
        return self.submit(
            request.decision_id, request, timeout=timeout,
            failed=lambda cause, reason: self._denied(request, cause, reason),
        )

    # -- the answerer side ------------------------------------------------

    def resolve(self, decision: Decision) -> bool:
        """Deliver an answer. **Main thread.** Idempotent per decision id."""
        return self.commit(decision.decision_id, decision)

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

    # -- internals --------------------------------------------------------

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
    return new_request_id()
