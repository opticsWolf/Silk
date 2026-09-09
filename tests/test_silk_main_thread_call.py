# -*- coding: utf-8 -*-
"""The main-thread seam, and the waiter both seams are made of (D49, D70).

D49 was specified as a *general* waiter rather than an approval-specific
one, and D70 is the claim being cashed: the canvas seam is the same
object with a different resolver -- the event loop instead of a person.
These tests pin that it really is the same object (one implementation of
the ordering rule, not two), and that the D36 failure paths behave
identically when the missing party is a thread rather than a human.
"""
from __future__ import annotations

import threading

import pytest

from silk.functions.blocking_seam import (
    CAUSE_CANCELLED, CAUSE_NO_ANSWERER, CAUSE_TIMEOUT, CAUSE_TRANSPORT,
    BlockingSeam, DriveGate,
)
from silk.functions.decision_seam import DecisionSeam
from silk.functions.main_thread_call import (
    CallRequest, CallResult, MainThreadCall,
)

PATIENT = 5.0
IMPATIENT = 0.05


class _Resolver:
    """Stands in for the Qt main thread: answers on another thread."""

    def __init__(self, seam, handler=None, raises: bool = False) -> None:
        self.seam = seam
        self.handler = handler or (lambda request: {"did": request.op})
        self.raises = raises
        self.seen: list = []
        self.threads: list = []

    def deliver(self, request) -> None:
        self.seen.append(request)
        if self.raises:
            raise RuntimeError("the canvas is gone")
        thread = threading.Thread(
            target=lambda: self.seam.serve(request, self.handler))
        self.threads.append(thread)
        thread.start()

    def join(self) -> None:
        for thread in self.threads:
            thread.join(timeout=PATIENT)


def _seam(**kwargs) -> tuple:
    seam = MainThreadCall(timeout_s=kwargs.pop("timeout_s", PATIENT),
                          drive=kwargs.pop("drive", None))
    resolver = _Resolver(seam, **kwargs)
    seam.attach(resolver.deliver)
    return seam, resolver


# ── one waiter, two seams (D70) ──────────────────────────────────────────


def test_both_seams_are_the_same_waiter():
    assert issubclass(DecisionSeam, BlockingSeam)
    assert issubclass(MainThreadCall, BlockingSeam)
    for method in ("submit", "commit", "cancel", "close"):
        assert getattr(DecisionSeam, method) is getattr(MainThreadCall, method), (
            "D49 was specified as a general waiter so its second user "
            "would not be a second implementation of the ordering rule"
        )


def test_the_ordering_rule_lives_in_one_place():
    """The event is set after the outcome is committed, not before."""
    drive = DriveGate("resolve", "wake")
    seam, resolver = _seam(drive=drive)

    answers: list = []
    worker = threading.Thread(
        target=lambda: answers.append(seam.call("place_node")))
    worker.start()

    assert drive.arrived_at("resolve"), "the resolver reached the commit"
    # The waiter cannot have woken yet: nothing is committed.
    assert not answers
    drive.release("resolve")
    assert drive.arrived_at("wake")
    drive.release_all()
    worker.join(timeout=PATIENT)
    resolver.join()

    assert answers and answers[0].ok, (
        "and what it re-reads under the lock is what the resolver wrote"
    )


# ── the happy path ───────────────────────────────────────────────────────


def test_a_call_returns_what_the_main_thread_did():
    seam, resolver = _seam(handler=lambda request: {"id": "n1"})
    answer = seam.call("place_node", class_name="TextNode")
    resolver.join()

    assert (answer.ok, answer.performed) == (True, True)
    assert answer.value == {"id": "n1"}
    assert resolver.seen[0].args == {"class_name": "TextNode"}


def test_a_handler_may_refuse_without_failing():
    seam, resolver = _seam(handler=lambda r: {"ok": False, "error": "no port"})
    answer = seam.call("connect")
    resolver.join()

    assert answer.ok is False and answer.error == "no port"
    assert answer.performed, (
        "the main thread ran and said no -- a different fact from never "
        "having reached it, and the model is told which"
    )
    assert answer.failure_text() == "no port"


def test_each_call_gets_its_own_correlation():
    seam, resolver = _seam()
    first, second = seam.call("a"), seam.call("b")
    resolver.join()
    assert first.call_id != second.call_id


# ── D36: every failure path refuses, none of them hangs ──────────────────


def test_no_canvas_refuses_immediately():
    seam = MainThreadCall(None, timeout_s=600.0)
    answer = seam.call("place_node")
    assert (answer.ok, answer.cause) == (False, CAUSE_NO_ANSWERER)
    assert "no canvas" in answer.failure_text()
    assert not seam.can_ask, (
        "and waiting out a ten-minute timeout for a resolver that does "
        "not exist spends the time to reach the same answer"
    )


def test_a_transport_that_raises_refuses():
    seam, _resolver = _seam(raises=True)
    answer = seam.call("place_node")
    assert (answer.ok, answer.cause) == (False, CAUSE_TRANSPORT)
    assert "could not reach" in answer.failure_text()


def test_a_silent_main_thread_times_out():
    seam = MainThreadCall(lambda request: None, timeout_s=IMPATIENT)
    answer = seam.call("place_node")
    assert (answer.ok, answer.cause) == (False, CAUSE_TIMEOUT)
    assert "no change was made" in answer.failure_text(), (
        "a timeout must read as 'nothing happened', because nothing did"
    )


def test_a_handler_that_raises_becomes_a_refusal_not_a_crash():
    def _boom(request):
        raise ValueError("scene deleted")

    seam, resolver = _seam(handler=_boom)
    answer = seam.call("remove_node")
    resolver.join()

    assert answer.ok is False and "scene deleted" in answer.error, (
        "this runs inside a Qt slot: an escaping exception would take out "
        "the event loop and leave the worker to time out for no reason"
    )


def test_stop_wakes_a_waiting_call():
    drive = DriveGate("wait")
    seam = MainThreadCall(lambda request: None, timeout_s=600.0, drive=drive)

    answers: list = []
    worker = threading.Thread(
        target=lambda: answers.append(seam.call("place_node")))
    worker.start()
    assert drive.arrived_at("wait")
    seam.cancel("stopped")
    drive.release_all()
    worker.join(timeout=PATIENT)

    assert answers[0].cause == CAUSE_CANCELLED
    assert "run was stopped" in answers[0].failure_text()


def test_a_closed_seam_refuses_later_calls():
    seam, _resolver = _seam()
    seam.close()
    assert seam.call("place_node").cause == CAUSE_CANCELLED
    assert not seam.can_ask


def test_a_second_outcome_is_refused():
    """D42's fifth race: the first outcome may already have run."""
    seam, resolver = _seam()
    answer = seam.call("place_node")
    resolver.join()
    assert seam.complete(answer.call_id, ok=True, value={"late": True}) is False


def test_an_unknown_id_cannot_be_completed():
    seam, _resolver = _seam()
    assert seam.complete("never-asked", ok=True) is False


# ── the result object promises nothing it did not do ─────────────────────


@pytest.mark.parametrize("cause", [CAUSE_CANCELLED, CAUSE_TIMEOUT,
                                   CAUSE_NO_ANSWERER, CAUSE_TRANSPORT])
def test_no_failure_cause_reports_success(cause):
    result = CallResult(call_id="x", op="place_node", cause=cause)
    assert not result.ok and not result.performed
    assert result.failure_text().startswith("Refused")


def test_a_request_carries_its_arguments():
    request = CallRequest(call_id="x", op="connect", args={"src_id": "a"})
    assert request.args["src_id"] == "a" and request.op == "connect"
