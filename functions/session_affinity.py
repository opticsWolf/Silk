# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Session-affine admission to the shared server (D47 mechanism A; G15).

One `llama_cpp.server` holds **one resident context**. Whichever prompt
ran last is the only one the KV cache can match against, so two
conversations taking turns overwrite each other and each request
re-prefills everything past the shared system prompt. Measured on
2026-09-19 (`docs/prefix_reuse_measurement.md`), at a 1600-token prompt:

===============  ==========  ==========  ===============
Shape            Reuse       Prefill     Wall, same work
===============  ==========  ==========  ===============
one session      74.8%       33.1%       1.1 s
two at once      **0.4%**    **74.1%**   **4.5 s**
===============  ==========  ==========  ===============

That is D47's rule landing on clause 2 -- reuse near zero, contention at
100% -- and so on mechanism **A: do not interleave**. Run one
conversation's round to completion, then the other's.

**This costs no throughput to give away.** Per D43 the server already
serialises every request through ``llama_outer_lock``; a fan-out never
had model-level concurrency. All this changes is the *order* of a queue
that already exists, from arrival order to session-grouped order.

**Grouping needs a moment of anticipation, and that is the whole
design.** When a round finishes, the next request from the same
conversation has not been built yet -- the agent is parsing the answer or
running a tool. A queue that admits whoever is waiting therefore always
admits the *other* conversation, which is exactly the alternation being
fixed. So a release opens a short **hold window** in which the outgoing
session keeps its claim.

**A hold that never pays is a hold that stops.** The window is worth
paying only when the conversation comes back inside it; a turn that has
genuinely ended would pay it on every switch forever. So the window is
kept on a short leash: three consecutive windows that elapse unused
suppress it, and the gate keeps *watching* for an arrival that would have
been in time, which restores it. Suppression therefore costs nothing to
reverse, and the gate converges on whichever behaviour the graph actually
has.

**The window is also capped by what it is protecting.** A hold is a bet
that a re-prefill costs more than the wait, and the pool already measures
the stake: ``prefix_report()['full_prefill_ms']`` is what re-evaluating
one prompt from scratch costs on this server. A 40 ms prefix is not worth
a 250 ms wait, so the window never exceeds it -- which is why a graph of
short prompts pays almost nothing for a mechanism it does not need, and
one at 4000 tokens (7 s of prefill) pays the full quarter second.

**Fairness is A's one real cost** (D47) and is bounded here rather than
argued away: after ``max_streak`` consecutive grants, a conversation with
someone else waiting gives up its claim for one slot.

Qt-free and pool-free on purpose: this is scheduling arithmetic, and it
is tested as such.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from weave.logger import get_logger

__all__ = ["DEFAULT_HOLD_S", "DEFAULT_MAX_STREAK", "SessionAffinityGate"]

log = get_logger("SilkAffinity")

#: How long an outgoing conversation keeps its claim on the next slot.
#: Sized against what a lost prefix costs, not against a round: the
#: measurement puts a contended round at +0.5 s (1600-token prompts) to
#: +1.7 s (4000-token), so a quarter second is cheap when it lands and
#: bounded when it does not -- and three misses stop it landing at all.
DEFAULT_HOLD_S = 0.25

#: Consecutive grants to one conversation before it yields a slot to
#: somebody who is waiting. Grouping is the point, so this is generous;
#: it exists so a long agent turn cannot starve a fan-out outright.
DEFAULT_MAX_STREAK = 8

#: Consecutive hold windows that elapse unused before the window is
#: suppressed. Three, not one: a single late round is ordinary.
_MISSES_BEFORE_SUPPRESSION = 3


class SessionAffinityGate:
    """Serialises requests, preferring the conversation that last ran.

    Thread-safe and re-entrant-safe in the only way that matters here:
    every :meth:`acquire` is matched by exactly one :meth:`release`, and
    the pool calls the pair from ``checkout``/``checkin``, which the
    engine already runs inside a ``finally``.
    """

    def __init__(
        self,
        enabled: bool = True,
        hold_s: float = DEFAULT_HOLD_S,
        max_streak: int = DEFAULT_MAX_STREAK,
        clock: Any = time.monotonic,
        prefill_cost_s: Any = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.hold_s = max(0.0, float(hold_s))
        self.max_streak = max(1, int(max_streak))
        self._clock = clock
        #: Callable returning what one lost prefix costs, in seconds, or
        #: None while nothing has been measured. Optional: a gate with no
        #: meter behind it simply uses the full window.
        self._prefill_cost_s = prefill_cost_s

        self._cv = threading.Condition()
        self._busy = False
        #: Whoever is running right now, or None.
        self._holder: Optional[str] = None
        #: Whoever holds the claim on the next slot.
        self._affinity: Optional[str] = None
        #: When that claim lapses. Always maintained, even while the
        #: window is suppressed, because a suppressed window still has to
        #: notice the arrival that would have been in time.
        self._hold_until = 0.0
        #: The part of the window actually enforced on other conversations.
        #: Separate from ``_hold_until`` because the leash has to keep
        #: watching the full window even while enforcing none of it.
        self._enforced_until = 0.0
        self._streak = 0
        self._misses = 0
        self._waiting: Dict[str, int] = {}

        # What the gate did, so the Pool Monitor can say whether it is
        # earning its latency rather than only that it is switched on.
        self._grants = 0
        self._grouped = 0        # granted to the session that last ran
        self._waited_s = 0.0     # time callers spent queued
        self._held = 0           # windows that a return arrival honoured
        self._expired = 0        # windows that elapsed unused
        self._yields = 0         # slots given up to the streak cap

    # -- admission ---------------------------------------------------------

    def acquire(self, session_id: str = "default") -> None:
        """Block until *session_id* may talk to the server."""
        if not self.enabled:
            with self._cv:
                self._grants += 1
            return
        session = str(session_id or "default")
        started = self._clock()
        with self._cv:
            self._waiting[session] = self._waiting.get(session, 0) + 1
            try:
                while not self._may_enter(session):
                    self._cv.wait(self._wait_timeout(session))
            finally:
                remaining = self._waiting.get(session, 1) - 1
                if remaining > 0:
                    self._waiting[session] = remaining
                else:
                    self._waiting.pop(session, None)
            self._admit(session, self._clock() - started)

    def _may_enter(self, session: str) -> bool:
        """Whether *session* can take the slot right now."""
        if self._busy:
            return False
        if self._affinity is None or session == self._affinity:
            return True
        # Somebody else holds the claim.
        if self._streak >= self.max_streak:
            return True                     # fairness: they have had enough
        if self._waiting.get(self._affinity):
            return False                    # they are queued; they go first
        return self._clock() >= self._enforced_until

    def _wait_timeout(self, session: str) -> Optional[float]:
        """Wake this waiter when the hold window lapses, not before.

        Without it a non-affinity waiter would sleep until the next
        ``notify``, and the event it is waiting for -- a deadline passing
        -- does not notify anything.
        """
        if self._busy or self._affinity is None or session == self._affinity:
            return None
        remaining = self._enforced_until - self._clock()
        return remaining if remaining > 0 else None

    def _admit(self, session: str, waited_s: float) -> None:
        """Record the grant. Called with the condition held."""
        now = self._clock()
        returning = session == self._affinity
        if self._affinity is not None and self._hold_until:
            if returning and now < self._hold_until:
                # Came back inside the window: the wait paid for itself,
                # whether or not anybody actually waited for it.
                self._held += 1
                self._misses = 0
            elif not returning and now >= self._hold_until:
                self._expired += 1
                self._misses += 1
                if self._misses == _MISSES_BEFORE_SUPPRESSION:
                    log.debug(
                        f"Affinity hold suppressed after {self._misses} "
                        "unused windows; still watching for a return."
                    )
        if returning:
            self._streak += 1
        else:
            if (self._affinity is not None
                    and self._streak >= self.max_streak):
                self._yields += 1
            self._streak = 1
        self._busy = True
        self._holder = session
        self._affinity = session
        self._hold_until = 0.0
        self._enforced_until = 0.0
        self._grants += 1
        if returning:
            self._grouped += 1
        self._waited_s += max(0.0, waited_s)

    def release(self, session_id: str = "default") -> None:
        """Hand the server back and open the hold window."""
        with self._cv:
            if not self.enabled:
                self._busy = False
                self._cv.notify_all()
                return
            self._busy = False
            self._holder = None
            # Maintained even when suppressed: an arrival inside it is
            # what restores the window, so it has to be observable.
            now = self._clock()
            self._hold_until = now + self.hold_s
            self._enforced_until = now + self._effective_hold()
            self._cv.notify_all()

    def release_all(self) -> None:
        """Let every waiter through; the server is going away.

        Shutdown must not wait on a turn nobody will ever release, so the
        gate stops gating rather than trying to drain in order.
        """
        with self._cv:
            self.enabled = False
            self._busy = False
            self._holder = None
            self._affinity = None
            self._hold_until = 0.0
            self._enforced_until = 0.0
            self._cv.notify_all()

    def _effective_hold(self) -> float:
        """The window actually enforced, after the misses leash."""
        if self._misses >= _MISSES_BEFORE_SUPPRESSION:
            return 0.0
        cost = self._prefill_cost()
        if cost is None:
            return self.hold_s      # nothing measured yet; back the bet
        return min(self.hold_s, max(0.0, cost))

    def _prefill_cost(self) -> Optional[float]:
        """Seconds one lost prefix costs, or None if nobody has measured.

        Never allowed to fail the gate: a meter that raises is a meter
        that has nothing to say, which is the same as not having one.
        """
        source = self._prefill_cost_s
        if not callable(source):
            return None
        try:
            value = source()
        except Exception:  # noqa: BLE001
            return None
        return float(value) if value is not None else None

    # -- what it did -------------------------------------------------------

    def report(self) -> dict:
        """What the gate has done, in terms the rule can be re-checked on.

        ``grouping_rate`` is the number that matters: the share of
        requests that followed one from the same conversation, which is
        what reuse can survive at all. Reported as ``None`` before there
        is anything to divide, never as zero -- the same rule the prefix
        meter follows, and for the same reason.
        """
        with self._cv:
            grants = self._grants
            return {
                "enabled": self.enabled,
                "hold_s": self.hold_s,
                "enforced_hold_s": round(self._effective_hold(), 4),
                "max_streak": self.max_streak,
                "grants": grants,
                "grouped": self._grouped,
                "grouping_rate": (self._grouped / grants) if grants else None,
                "avg_wait_s": (self._waited_s / grants) if grants else None,
                "holds_honoured": self._held,
                "holds_expired": self._expired,
                "hold_suppressed": self._misses >= _MISSES_BEFORE_SUPPRESSION,
                "fairness_yields": self._yields,
            }

    def reset_stats(self) -> None:
        with self._cv:
            self._grants = 0
            self._grouped = 0
            self._waited_s = 0.0
            self._held = 0
            self._expired = 0
            self._yields = 0

    def forget(self, session_id: str) -> None:
        """Drop a conversation's claim; called when a session is released.

        Without it a finished agent keeps the next slot warm for a
        quarter second on a server nobody is going to ask it about.
        """
        session = str(session_id or "default")
        with self._cv:
            if self._affinity == session and self._holder != session:
                self._affinity = None
                self._hold_until = 0.0
                self._enforced_until = 0.0
                self._streak = 0
                self._cv.notify_all()


def describe_gate(report: dict) -> str:
    """The gate's report as a person reads it, for the Pool Monitor."""
    if not report.get("enabled"):
        return "Session affinity: off — requests run in arrival order."
    grants = report.get("grants") or 0
    if not grants:
        return "Session affinity: on — no requests yet."
    rate = report.get("grouping_rate")
    wait = report.get("avg_wait_s") or 0.0
    parts = [
        f"grouped {rate * 100:.0f}%" if rate is not None else "grouped —",
        f"wait {wait * 1000:.0f} ms avg",
        f"{report.get('holds_honoured', 0)} held / "
        f"{report.get('holds_expired', 0)} expired",
    ]
    if report.get("hold_suppressed"):
        parts.append("hold suppressed")
    if report.get("fairness_yields"):
        parts.append(f"{report['fairness_yields']} yield(s)")
    return "Session affinity:  " + "  ·  ".join(parts)
