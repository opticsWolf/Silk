# -*- coding: utf-8 -*-
"""Session accounting on the shared server pool (spec D46/D47).

Two bugs that reading D47 exposed, and one property that keeps them fixed.

`checkout` runs once per *request*, not once per conversation, so counting
checkouts and calling the result `bound_sessions` measured requests-ever
and only ever grew -- the Pool Monitor showed that growth as bound
conversations. And `Clear Context` released a session by reaching into a
`_session_instances` dict this pool does not have, an AttributeError
swallowed by a broad `except`, so the release never happened at all.

The fix is a set of session ids and a public `release_session`. The
property worth pinning is that both operations are **idempotent**: a
conversation is bound or it is not, and neither a second request nor a
second Clear Context can move the number anywhere it should not go.

The pool is not constructed here -- doing so spawns a `llama_cpp.server`
subprocess -- so the accounting is exercised on an instance built without
`__init__`, which is exactly the surface under test and nothing else.
"""

from __future__ import annotations

import threading


from silk.functions.model_pool import GGUFModelPool  # noqa: E402


def _pool() -> GGUFModelPool:
    """A pool with only its session bookkeeping wired up."""
    pool = object.__new__(GGUFModelPool)
    pool._lock = threading.RLock()
    pool._bound_sessions = set()
    pool._client = object()
    pool._in_flight = 0
    pool._peak_in_flight = 0
    pool._queued_requests = 0
    pool._queue_reported = False
    return pool


def test_a_fresh_pool_has_no_bound_sessions():
    assert _pool().bound_sessions == 0


def test_many_requests_from_one_session_bind_it_once():
    """The bug: checkout runs per request, so counting checkouts only grew."""
    pool = _pool()
    for _ in range(10):
        assert pool.checkout("conversation-a") is pool._client
    assert pool.bound_sessions == 1


def test_two_conversations_are_two_bound_sessions():
    pool = _pool()
    pool.checkout("conversation-a")
    pool.checkout("conversation-b")
    assert pool.bound_sessions == 2


def test_an_ordinary_checkin_does_not_unbind():
    """A run ends every round with one, and the conversation is still live."""
    pool = _pool()
    pool.checkout("conversation-a")
    pool.checkin(pool._client, session_id="conversation-a")
    assert pool.bound_sessions == 1


def test_a_releasing_checkin_unbinds():
    pool = _pool()
    pool.checkout("conversation-a")
    pool.checkin(pool._client, session_id="conversation-a",
                 release_session=True)
    assert pool.bound_sessions == 0


def test_release_session_reports_whether_it_knew_the_session():
    pool = _pool()
    pool.checkout("conversation-a")
    assert pool.release_session("conversation-a") is True
    assert pool.release_session("conversation-a") is False, "already gone"
    assert pool.release_session("never-seen") is False
    assert pool.bound_sessions == 0, "a double release cannot go negative"


def test_releasing_one_conversation_leaves_the_others():
    pool = _pool()
    for name in ("a", "b", "c"):
        pool.checkout(name)
    pool.release_session("b")
    assert pool.bound_sessions == 2
    assert pool._bound_sessions == {"a", "c"}


def test_a_released_session_rebinds_on_its_next_request():
    """Clear Context is not an eviction from the pool; it is a fresh start."""
    pool = _pool()
    pool.checkout("conversation-a")
    pool.release_session("conversation-a")
    pool.checkout("conversation-a")
    assert pool.bound_sessions == 1


def test_the_snapshot_reports_distinct_conversations():
    pool = _pool()
    pool._model_path = "/models/some-model.gguf"
    pool._max_instances = 1
    pool._clear_on_return = False
    pool.prefix_report = lambda: {}
    for _ in range(5):
        pool.checkout("conversation-a")
    pool.checkout("conversation-b")

    assert pool.snapshot()["bound_sessions"] == 2, (
        "the Pool Monitor reads this; it must not be a request counter"
    )


# -- the queue behind one server (§22 q1c) --------------------------------
#
# D43 means every agent's request goes through one server, one at a time.
# A fan-out of eight is therefore correct and sequential -- and looks like
# a hang, which is the failure D53 named and q1d already answered once for
# the approval gate. Same answer here: say it once, then count.


def _flight_pool():
    pool = _pool()
    pool._model_path = "/models/some-model.gguf"
    return pool


def test_a_lone_request_never_queues():
    pool = _flight_pool()
    pool._flight_begin()
    pool._flight_end()
    report = pool.serialization_report()
    assert report["queued_requests"] == 0 and not report["serialising"]
    assert report["peak_in_flight"] == 1


def test_a_fan_out_is_counted_not_hidden():
    pool = _flight_pool()
    for _ in range(4):          # four workers arrive before any finishes
        pool._flight_begin()
    report = pool.serialization_report()
    assert report["in_flight"] == 4
    assert report["peak_in_flight"] == 4
    assert report["queued_requests"] == 3, "three of the four waited"
    assert report["serialising"]


def test_the_queue_drains_but_the_count_stays():
    """What happened is still worth knowing after it stops happening."""
    pool = _flight_pool()
    for _ in range(3):
        pool._flight_begin()
    for _ in range(3):
        pool._flight_end()
    report = pool.serialization_report()
    assert report["in_flight"] == 0
    assert report["peak_in_flight"] == 3 and report["queued_requests"] == 2


def test_it_is_logged_once_and_then_only_counted(caplog):
    pool = _flight_pool()
    with caplog.at_level("INFO"):
        for _ in range(5):
            pool._flight_begin()
    lines = [r for r in caplog.records if "serialising" in r.getMessage()]
    assert len(lines) == 1, (
        "once per request is noise; never is a correct fan-out that looks "
        "hung (D53, and the shape q1d settled)"
    )
    assert pool.serialization_report()["queued_requests"] == 4


def test_the_monitor_says_nothing_until_something_waits():
    from silk.nodes.pool_monitor import queue_note

    quiet = {"serialization": {"in_flight": 1, "peak_in_flight": 1,
                               "queued_requests": 0, "serialising": False}}
    assert queue_note(quiet) == "", (
        "a line that is always there is one nobody reads"
    )
    assert queue_note({}) == "", "an older pool without the field says nothing"

    busy = {"serialization": {"in_flight": 3, "peak_in_flight": 8,
                              "queued_requests": 12, "serialising": True}}
    note = queue_note(busy)
    assert "3 in flight" in note and "peak 8" in note and "12 waited" in note


# ── the measurement, made visible (G15, D47) ─────────────────────────────

def test_the_monitor_says_nobody_has_looked_yet():
    """0% and "unmeasured" lead to opposite decisions, so they must differ."""
    from silk.nodes.pool_monitor import prefix_note

    assert "no requests measured" in prefix_note({})
    assert "no requests measured" in prefix_note({"prefix_reuse": {"requests": 0}})


def test_the_monitor_renders_the_three_numbers():
    from silk.nodes.pool_monitor import prefix_note

    note = prefix_note({"prefix_reuse": {
        "requests": 12, "measured_requests": 11, "reuse_rate": 0.932,
        "contention_rate": 0.25, "prefill_share": 0.114,
    }})
    assert "reuse 93.2%" in note
    assert "contention 25.0%" in note
    assert "prefill 11.4%" in note
    assert "11/12 measured" in note, (
        "how much of the window was actually readable is part of the answer"
    )


def test_a_metric_the_log_never_stated_is_unknown_not_zero():
    from silk.nodes.pool_monitor import prefix_note

    note = prefix_note({"prefix_reuse": {
        "requests": 3, "measured_requests": 3, "reuse_rate": 0.5,
        "contention_rate": None, "prefill_share": None,
    }})
    assert "contention unknown" in note and "prefill unknown" in note
