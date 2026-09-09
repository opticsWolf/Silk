# -*- coding: utf-8 -*-
"""The run-scoped decision seam and its race catalog (spec D30/D38/D42/D49).

Two halves. The first pins the seam's plain behaviour: correlation, the
five wake causes, idempotence, and the rule that every failure denies. The
second is D42's catalog -- five races, driven in both orders through the
DriveGate, because a parked worker thread racing a Qt thread is exactly
the situation invariant fixtures cannot test and a sleep-based test cannot
test *reliably*.

The property the drive gate enforces is the one that makes the catalog
worth having: **zero effects while parked**. A held tool call that ran
anyway would pass every assertion about the decision and still be a
disaster.
"""

from __future__ import annotations

import threading
import time

import pytest


from silk.functions.decision_seam import (
    CAUSE_ANSWERED,
    CAUSE_CANCELLED,
    CAUSE_NO_ANSWERER,
    CAUSE_TIMEOUT,
    CAUSE_TRANSPORT,
    KIND_ACKNOWLEDGE,
    Decision,
    DecisionRequest,
    DecisionSeam,
    DriveGate,
    new_decision_id,
)

#: Long enough that a scheduling hiccup cannot expire it; the tests that
#: care about expiry set their own.
PATIENT = 5.0
#: Short enough that a timeout test does not slow the suite.
IMPATIENT = 0.15


def _request(**kw) -> DecisionRequest:
    kw.setdefault("decision_id", new_decision_id())
    kw.setdefault("tool_name", "write_file")
    kw.setdefault("prompt", "Write to config.toml?")
    return DecisionRequest(**kw)


class _Asker:
    """Records what was asked, on which thread."""

    def __init__(self, raises: bool = False) -> None:
        self.seen: list[DecisionRequest] = []
        self.raises = raises

    def __call__(self, request: DecisionRequest) -> None:
        if self.raises:
            raise RuntimeError("the widget is gone")
        self.seen.append(request)


def _in_thread(fn, *args, **kw) -> tuple[threading.Thread, list]:
    out: list = []

    def body() -> None:
        out.append(fn(*args, **kw))

    thread = threading.Thread(target=body, daemon=True)
    thread.start()
    return thread, out


# -- the plain behaviour ---------------------------------------------------


def test_an_approval_comes_back_by_id():
    asker = _Asker()
    seam = DecisionSeam(asker, timeout_s=PATIENT)
    request = _request()

    thread, out = _in_thread(seam.await_decision, request)
    for _ in range(200):                       # wait for the ask to land
        if asker.seen:
            break
        time.sleep(0.005)
    assert seam.approve(request.decision_id, actor="frank") is True
    thread.join(PATIENT)

    decision = out[0]
    assert decision.decision_id == request.decision_id
    assert decision.approved and decision.answered
    assert decision.actor == "frank"
    assert asker.seen == [request], "the request reached the answerer verbatim"


def test_a_denial_carries_its_reason():
    seam = DecisionSeam(_Asker(), timeout_s=PATIENT)
    request = _request()
    thread, out = _in_thread(seam.await_decision, request)
    time.sleep(0.02)
    seam.deny(request.decision_id, actor="frank", reason="wrong directory")
    thread.join(PATIENT)

    assert out[0].approved is False and out[0].cause == CAUSE_ANSWERED
    assert "wrong directory" in out[0].refusal_text()


def test_no_answerer_denies_without_waiting():
    """D36's first failure: a headless run, or a subagent with no UI."""
    seam = DecisionSeam(None, timeout_s=60.0)
    assert seam.can_ask is False

    started = time.perf_counter()
    decision = seam.await_decision(_request())
    assert time.perf_counter() - started < 1.0, "it must not wait out a timeout"
    assert decision.approved is False and decision.cause == CAUSE_NO_ANSWERER
    assert "durable grant" in decision.refusal_text()


def test_a_transport_that_raises_denies():
    seam = DecisionSeam(_Asker(raises=True), timeout_s=PATIENT)
    decision = seam.await_decision(_request())
    assert decision.approved is False and decision.cause == CAUSE_TRANSPORT
    assert not seam.outstanding(), "a failed ask leaves nothing outstanding"


def test_a_timeout_denies():
    seam = DecisionSeam(_Asker(), timeout_s=IMPATIENT)
    decision = seam.await_decision(_request())
    assert decision.approved is False and decision.cause == CAUSE_TIMEOUT


def test_a_closed_seam_denies_every_later_request():
    """D36's second failure: the widget destroyed mid-run."""
    seam = DecisionSeam(_Asker(), timeout_s=PATIENT)
    seam.close()
    decision = seam.await_decision(_request())
    assert decision.approved is False
    assert decision.cause in (CAUSE_CANCELLED, CAUSE_NO_ANSWERER)


def test_one_seam_serves_the_other_question_kinds():
    """D50: acknowledge is the same block, not a second waiter."""
    seam = DecisionSeam(_Asker(), timeout_s=PATIENT)
    request = _request(kind=KIND_ACKNOWLEDGE, tool_name="",
                       prompt="About to compact 6 turns. Continue?")
    thread, out = _in_thread(seam.await_decision, request)
    time.sleep(0.02)
    seam.approve(request.decision_id, kind=KIND_ACKNOWLEDGE)
    thread.join(PATIENT)
    assert out[0].kind == KIND_ACKNOWLEDGE and out[0].approved


def test_resolving_an_unknown_id_is_refused():
    seam = DecisionSeam(_Asker(), timeout_s=PATIENT)
    assert seam.approve("never-asked") is False


def test_outstanding_lists_what_is_waiting():
    seam = DecisionSeam(_Asker(), timeout_s=PATIENT)
    request = _request()
    thread, _out = _in_thread(seam.await_decision, request)
    time.sleep(0.05)
    assert [r.decision_id for r in seam.outstanding()] == [request.decision_id]
    seam.deny(request.decision_id)
    thread.join(PATIENT)
    assert seam.outstanding() == []


# -- D42's race catalog ----------------------------------------------------
#
# Each race is driven in both orders by parking one side at a checkpoint.


def _blocked(*, timeout_s: float = PATIENT):
    """A waiter parked at ``wait`` -- asked, not yet asleep.

    This is the window every race lives in: the request has left, so the
    answerer can act, and the waiter has not read anything yet, so whatever
    is committed first is what it will see.
    """
    checkpoint = "wait"
    drive = DriveGate(checkpoint)
    seam = DecisionSeam(_Asker(), timeout_s=timeout_s, drive=drive)
    request = _request()
    thread, out = _in_thread(seam.await_decision, request)
    assert drive.arrived_at(checkpoint), f"the seam never reached {checkpoint}"
    return seam, request, drive, thread, out


def test_race_approve_then_stop():
    """The approval is committed first; Stop must not overwrite the answer."""
    seam, request, drive, thread, out = _blocked()
    seam.approve(request.decision_id, actor="frank")
    seam.cancel("user pressed Stop")
    drive.release_all()
    thread.join(PATIENT)
    assert out[0].approved is True, (
        "an answer committed before the cancel is the answer"
    )


def test_race_stop_then_approve():
    """Stop wins, and the late approval is refused rather than applied."""
    seam, request, drive, thread, out = _blocked()
    seam.cancel("user pressed Stop")
    assert seam.approve(request.decision_id) is False, (
        "a cancelled seam must not accept an approval"
    )
    drive.release_all()
    thread.join(PATIENT)
    assert out[0].approved is False and out[0].cause == CAUSE_CANCELLED
    assert "stopped" in out[0].refusal_text()


def test_race_approve_then_timeout():
    """The wait expired, but an answer had already been committed."""
    seam, request, drive, thread, out = _blocked(timeout_s=IMPATIENT)
    seam.approve(request.decision_id, actor="frank")
    time.sleep(IMPATIENT * 2)          # the deadline passes while parked
    drive.release_all()
    thread.join(PATIENT)
    assert out[0].approved is True, (
        "an expiry cannot retract a decision that was already made"
    )


def test_race_timeout_then_approve():
    seam = DecisionSeam(_Asker(), timeout_s=IMPATIENT)
    request = _request()
    decision = seam.await_decision(request)
    assert decision.cause == CAUSE_TIMEOUT
    assert seam.approve(request.decision_id) is False, (
        "the request is gone; a late approval has nothing to approve"
    )


def test_race_deny_then_stop():
    seam, request, drive, thread, out = _blocked()
    seam.deny(request.decision_id, reason="no")
    seam.cancel("stopped")
    drive.release_all()
    thread.join(PATIENT)
    assert out[0].approved is False and out[0].cause == CAUSE_ANSWERED, (
        "a real denial is a different fact from a cancellation, and stays one"
    )


def test_race_stop_then_deny():
    seam, request, drive, thread, out = _blocked()
    seam.cancel("stopped")
    assert seam.deny(request.decision_id) is False
    drive.release_all()
    thread.join(PATIENT)
    assert out[0].cause == CAUSE_CANCELLED


def test_race_decision_arrives_after_the_timeout_expired():
    """Answered late is not answered: the tool call is long refused."""
    seam = DecisionSeam(_Asker(), timeout_s=IMPATIENT)
    request = _request()
    thread, out = _in_thread(seam.await_decision, request)
    thread.join(PATIENT)
    assert out[0].cause == CAUSE_TIMEOUT
    assert seam.approve(request.decision_id) is False


def test_race_second_decision_for_a_resolved_id():
    """D42's fifth race: the first decision may already have run a tool."""
    seam, request, drive, thread, out = _blocked()
    assert seam.deny(request.decision_id, reason="first") is True
    assert seam.approve(request.decision_id) is False, "already resolved"
    drive.release_all()
    thread.join(PATIENT)
    assert out[0].approved is False and "first" in out[0].reason


def test_a_resolve_parked_mid_write_still_orders_correctly():
    """The ordering rule itself: outcome under the lock, *then* the wake."""
    drive = DriveGate("resolve")
    seam = DecisionSeam(_Asker(), timeout_s=PATIENT, drive=drive)
    request = _request()
    waiter, out = _in_thread(seam.await_decision, request)
    time.sleep(0.05)

    resolver, _r = _in_thread(seam.approve, request.decision_id)
    assert drive.arrived_at("resolve")
    # The resolver is parked before it writes anything, so the waiter is
    # still asleep: no wake without a committed outcome.
    assert waiter.is_alive()
    drive.release_all()
    resolver.join(PATIENT)
    waiter.join(PATIENT)
    assert out[0].approved is True


def test_zero_effects_while_parked():
    """The property that makes the catalog worth writing."""
    drive = DriveGate("wait")
    seam = DecisionSeam(_Asker(), timeout_s=PATIENT, drive=drive)
    effects: list[str] = []

    def held_call() -> str:
        decision = seam.await_decision(_request())
        if decision.approved:
            effects.append("wrote the file")
            return "ok"
        return decision.refusal_text()

    thread, out = _in_thread(held_call)
    assert drive.arrived_at("wait")
    time.sleep(0.05)
    assert effects == [], "the held call ran while the gate was still blocked"

    seam.cancel("stopped")
    drive.release_all()
    thread.join(PATIENT)
    assert effects == [] and "stopped" in out[0]


@pytest.mark.parametrize("cause", [
    CAUSE_CANCELLED, CAUSE_TIMEOUT, CAUSE_NO_ANSWERER, CAUSE_TRANSPORT,
])
def test_every_non_answer_cause_reads_as_a_denial(cause):
    decision = Decision(decision_id="d", cause=cause)
    assert decision.approved is False and not decision.answered
    assert decision.refusal_text().startswith("Denied")
