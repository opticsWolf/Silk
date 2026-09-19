# -*- coding: utf-8 -*-
"""D47 mechanism A: the queue is grouped by conversation, not by arrival.

The measurement (G15, 2026-09-19) is what chose this: one server holds one
resident context, so alternating two conversations drops prefix reuse from
74.8% to 0.4% and makes the same work take five times as long. These tests
pin the scheduling arithmetic that fixes it, and -- just as important --
the two things that keep it from becoming a liability: a hold window that
stops holding when it stops paying, and a streak cap so grouping cannot
starve anybody.

The clock is injected, so nothing here sleeps.
"""
from __future__ import annotations

import threading

import pytest

from weave.plugins.silk.functions.session_affinity import (
    SessionAffinityGate,
    describe_gate,
)


class Clock:
    """A hand-wound monotonic clock."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return Clock()


def gate(clock, **kw):
    kw.setdefault("hold_s", 0.25)
    return SessionAffinityGate(clock=clock, **kw)


def run(g, session):
    g.acquire(session)
    g.release(session)


# -- grouping ---------------------------------------------------------------

def test_the_same_conversation_walks_straight_back_in(clock):
    g = gate(clock)
    for _ in range(4):
        run(g, "A")
    report = g.report()
    assert report["grants"] == 4
    # The first grant had nobody to follow; the other three followed
    # themselves.
    assert report["grouped"] == 3
    assert report["grouping_rate"] == pytest.approx(0.75)
    assert report["avg_wait_s"] == 0.0


def test_a_waiting_stranger_is_let_in_once_the_window_lapses(clock):
    g = gate(clock)
    run(g, "A")
    clock.advance(1.0)           # well past the hold
    run(g, "B")
    assert g.report()["holds_expired"] == 1
    assert g.report()["grouped"] == 0


def test_a_return_inside_the_window_is_what_the_window_is_for(clock):
    g = gate(clock)
    run(g, "A")
    clock.advance(0.1)           # inside the 0.25 s hold
    g.acquire("A")
    assert g.report()["holds_honoured"] == 1
    g.release("A")


def test_a_stranger_waits_out_the_window_rather_than_jumping_in(clock):
    """The whole mechanism in one assertion: B is queued and A is not,
    and B still does not get the slot until A's claim lapses."""
    g = gate(clock)
    run(g, "A")

    entered = threading.Event()

    def other():
        g.acquire("B")
        entered.set()
        g.release("B")

    thread = threading.Thread(target=other, daemon=True)
    thread.start()
    assert not entered.wait(0.15), "B took the slot while A still held it"
    clock.advance(1.0)
    with g._cv:                  # nudge the timed waiter to re-check
        g._cv.notify_all()
    assert entered.wait(2.0), "B never got in after the window lapsed"
    thread.join(timeout=2.0)


def test_a_queued_owner_goes_before_a_queued_stranger(clock):
    """Two waiters, one slot. Affinity decides, not arrival order."""
    g = gate(clock)
    g.acquire("A")               # A is running
    order: list[str] = []
    started = threading.Barrier(3)

    def contend(name):
        started.wait(timeout=5)
        g.acquire(name)
        order.append(name)
        g.release(name)

    threads = [threading.Thread(target=contend, args=(n,), daemon=True)
               for n in ("B", "A")]
    for thread in threads:
        thread.start()
    started.wait(timeout=5)
    # Let both queue up before the slot frees.
    while True:
        with g._cv:
            if g._waiting.get("A") and g._waiting.get("B"):
                break
    g.release("A")
    # A's queued round takes the slot; B is still waiting out the claim.
    while not order:
        pass
    assert order == ["A"]
    # Now let the claim lapse. (The clock is hand-wound, so B's timed wait
    # would otherwise re-check the same frozen instant forever.)
    clock.advance(1.0)
    with g._cv:
        g._cv.notify_all()
    for thread in threads:
        thread.join(timeout=5)
    assert order == ["A", "B"]


# -- the leash --------------------------------------------------------------

def test_three_unused_windows_suppress_the_hold(clock):
    """A hold that never pays must stop being paid for."""
    g = gate(clock)
    for _ in range(3):
        run(g, "A")
        clock.advance(1.0)
        run(g, "B")
        clock.advance(1.0)
    assert g.report()["holds_expired"] >= 3
    assert g.report()["hold_suppressed"] is True


def test_a_suppressed_gate_still_notices_a_return_and_recovers(clock):
    """Suppression is cheap to reverse on purpose: the window keeps being
    computed, so an arrival that *would* have been in time restores it."""
    g = gate(clock)
    for _ in range(3):
        run(g, "A")
        clock.advance(1.0)
        run(g, "B")
        clock.advance(1.0)
    assert g.report()["hold_suppressed"] is True

    run(g, "A")
    clock.advance(0.05)          # back inside the window
    run(g, "A")
    assert g.report()["hold_suppressed"] is False


def test_a_suppressed_hold_does_not_make_anybody_wait(clock):
    g = gate(clock)
    g._misses = 99               # as if the hold had never once paid
    run(g, "A")
    g.acquire("B")               # would block if the window were enforced
    g.release("B")


# -- fairness ---------------------------------------------------------------

def test_a_long_turn_eventually_yields_to_somebody_waiting(clock):
    """A's cost, per D47, is fairness -- bounded rather than argued away."""
    g = gate(clock, max_streak=3)
    for _ in range(3):
        run(g, "A")
    g.acquire("A")               # A is running its fourth round
    order: list[str] = []

    def other():
        g.acquire("B")
        order.append("B")
        g.release("B")

    thread = threading.Thread(target=other, daemon=True)
    thread.start()
    while True:
        with g._cv:
            if g._waiting.get("B"):
                break
    g.release("A")
    thread.join(timeout=5)
    assert order == ["B"], "B was starved past the streak cap"
    assert g.report()["fairness_yields"] == 1


def test_the_streak_cap_does_not_fire_when_nobody_is_waiting(clock):
    g = gate(clock, max_streak=2)
    for _ in range(6):
        run(g, "A")
    assert g.report()["fairness_yields"] == 0
    assert g.report()["grouped"] == 5


# -- lifecycle --------------------------------------------------------------

def test_forgetting_a_session_drops_its_claim(clock):
    g = gate(clock)
    run(g, "A")
    g.forget("A")
    g.acquire("B")               # no window left to wait out
    g.release("B")
    assert g.report()["holds_expired"] == 0


def test_forgetting_the_session_that_is_running_changes_nothing(clock):
    g = gate(clock)
    g.acquire("A")
    g.forget("A")
    assert g._busy is True
    g.release("A")


def test_release_all_lets_a_waiter_out_of_a_dying_server(clock):
    g = gate(clock)
    g.acquire("A")
    out = threading.Event()

    def other():
        g.acquire("B")
        out.set()

    thread = threading.Thread(target=other, daemon=True)
    thread.start()
    assert not out.wait(0.1)
    g.release_all()
    assert out.wait(2.0)
    thread.join(timeout=2.0)


def test_disabled_is_a_straight_pass_through(clock):
    g = SessionAffinityGate(enabled=False, clock=clock)
    g.acquire("A")
    g.acquire("B")               # would deadlock if it gated
    g.release("A")
    g.release("B")
    assert g.report()["grants"] == 2
    assert g.report()["enabled"] is False


def test_reset_clears_the_counters_but_not_the_claim(clock):
    g = gate(clock)
    run(g, "A")
    g.reset_stats()
    assert g.report()["grants"] == 0
    assert g.report()["grouping_rate"] is None
    assert g._affinity == "A"


# -- what it says -----------------------------------------------------------

def test_describe_says_nothing_rather_than_zero_before_any_request(clock):
    """The same rule the prefix meter follows: 0% and 'nobody looked' lead
    to opposite decisions."""
    text = describe_gate(gate(clock).report())
    assert "no requests yet" in text
    assert "0%" not in text


def test_describe_reports_the_grouping_rate_and_the_price_of_it(clock):
    g = gate(clock)
    for _ in range(4):
        run(g, "A")
    text = describe_gate(g.report())
    assert "grouped 75%" in text
    assert "ms avg" in text


def test_describe_says_when_it_is_off(clock):
    text = describe_gate(SessionAffinityGate(enabled=False).report())
    assert "off" in text
    assert "arrival order" in text


def test_describe_admits_a_suppressed_hold(clock):
    g = gate(clock)
    for _ in range(3):
        run(g, "A")
        clock.advance(1.0)
        run(g, "B")
        clock.advance(1.0)
    assert "hold suppressed" in describe_gate(g.report())


# -- the window is capped by what it protects -------------------------------

def test_a_cheap_prefix_does_not_buy_an_expensive_wait(clock):
    """A hold is a bet that a re-prefill costs more than the wait. The
    pool measures the stake, so the bet is never larger than it."""
    g = SessionAffinityGate(clock=clock, hold_s=0.25,
                            prefill_cost_s=lambda: 0.04)
    run(g, "A")
    assert g.report()["enforced_hold_s"] == pytest.approx(0.04)
    clock.advance(0.05)          # past the 40 ms stake, inside the 250 ms
    g.acquire("B")               # would block on the full window
    g.release("B")


def test_an_expensive_prefix_gets_the_whole_window(clock):
    g = SessionAffinityGate(clock=clock, hold_s=0.25,
                            prefill_cost_s=lambda: 7.0)
    run(g, "A")
    assert g.report()["enforced_hold_s"] == pytest.approx(0.25)


def test_an_unmeasured_prefix_backs_the_bet(clock):
    """None is not zero: before anything is measured the window stands."""
    g = SessionAffinityGate(clock=clock, hold_s=0.25,
                            prefill_cost_s=lambda: None)
    run(g, "A")
    assert g.report()["enforced_hold_s"] == pytest.approx(0.25)


def test_a_meter_that_raises_is_a_meter_with_nothing_to_say(clock):
    def broken():
        raise RuntimeError("no log")

    g = SessionAffinityGate(clock=clock, hold_s=0.25, prefill_cost_s=broken)
    run(g, "A")
    assert g.report()["enforced_hold_s"] == pytest.approx(0.25)


def test_the_leash_still_wins_over_a_costly_prefix(clock):
    """Suppression is about holds that never pay, whatever they protect."""
    g = SessionAffinityGate(clock=clock, hold_s=0.25,
                            prefill_cost_s=lambda: 7.0)
    for _ in range(3):
        run(g, "A")
        clock.advance(1.0)
        run(g, "B")
        clock.advance(1.0)
    assert g.report()["hold_suppressed"] is True
    assert g.report()["enforced_hold_s"] == 0.0
