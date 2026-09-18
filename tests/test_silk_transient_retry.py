# -*- coding: utf-8 -*-
"""Asking the same backend again, when the error says to (D15, D40).

`classify_model_error` has always separated a 429 or a 503 from a bad
request, but the loop only ever acted on the overflow verdict, so a rate
limit ended a run that would have succeeded a second later. What is
pinned here is that retrying happens, that it stays bounded, and above
all the two cases where it must *not* happen.
"""

from __future__ import annotations

import time

from silk.functions.agent_loop import AgentLoop
from silk.functions.stream_events import EventDelta, EventError


class _Engine:
    """Fails the first ``fail_times`` attempts, then answers."""

    def __init__(self, error, fail_times=1, deltas=("ok",), emit_before=0):
        self.error = error
        self.fail_times = fail_times
        self.deltas = deltas
        self.emit_before = emit_before
        self.attempts = 0
        self.history = []
        self.last_stats = {}
        self.usage_limits = _NoLimits()
        self.system_prompt = ""
        self._stop = False

    def stream_response(self, gen_params):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            # Some failures arrive mid-stream, after tokens are out.
            for i in range(self.emit_before):
                yield f"partial{i}"
            raise RuntimeError(self.error)
        yield from self.deltas

    def append_message(self, role, content, **kw):
        self.history.append((role, content))

    def request_stop(self):
        self._stop = True

    def stop_requested(self):
        return self._stop

    def count_prompt_tokens(self):
        return 1

    def context_length(self):
        return None


class _NoLimits:
    def check_request(self):
        pass

    def check_input_tokens(self, n):
        pass

    def reserve_request(self):
        pass

    def snapshot(self):
        return {}


def _run(engine, **kw):
    loop = AgentLoop(engine, toolbox=None, **kw)
    return list(loop.run("go"))


def test_a_rate_limit_is_asked_again_rather_than_ending_the_run(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    engine = _Engine("429 Too Many Requests", fail_times=1,
                     deltas=("the ", "answer"))
    events = _run(engine, transient_retries=3)

    assert engine.attempts == 2, "it asked again"
    text = "".join(e.delta for e in events if isinstance(e, EventDelta))
    assert text == "the answer", "and the run produced its answer"
    errors = [e for e in events if isinstance(e, EventError)]
    assert len(errors) == 1, (
        "the failure is still reported once -- retrying changes what happens "
        "next, not what the consumer is told"
    )


def test_a_retry_does_not_spend_a_reasoning_round(monkeypatch):
    """`max_rounds` bounds the model's thinking; a 503 is not a thought."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    engine = _Engine("503 Service Unavailable", fail_times=2, deltas=("hi",))
    _run(engine, max_rounds=1, transient_retries=3)

    assert engine.attempts == 3, (
        "three attempts inside a single round: had retries spent rounds, the "
        "run would have died on a one-round budget"
    )


def test_retries_are_bounded_and_then_it_is_down(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    engine = _Engine("502 Bad Gateway", fail_times=99)
    events = _run(engine, transient_retries=2)

    assert engine.attempts == 3, "the first try plus two retries"
    assert any(isinstance(e, EventError) for e in events)


def test_a_terminal_error_is_not_retried(monkeypatch):
    """The default for an unrecognised message is terminal, deliberately."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    engine = _Engine("invalid model 'nope'", fail_times=99)
    _run(engine, transient_retries=3)

    assert engine.attempts == 1, (
        "a bad request fails the same way however many times it is sent"
    )


def test_a_failure_after_tokens_are_out_is_not_retried(monkeypatch):
    """The consumer has already rendered them; a retry would replay them.

    This is the condition that makes the whole thing safe. The deltas are
    yielded as they arrive, so by the time a mid-stream 503 lands, the
    caller has drawn that text. Asking again would produce a second
    answer on top of a partial one.
    """
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    engine = _Engine("503 Service Unavailable", fail_times=1,
                     deltas=("second",), emit_before=2)
    events = _run(engine, transient_retries=3)

    assert engine.attempts == 1, "no retry once anything was emitted"
    text = "".join(e.delta for e in events if isinstance(e, EventDelta))
    assert text == "partial0partial1", (
        "the run keeps the fragment it already showed rather than doubling it"
    )


def test_a_stop_during_the_backoff_is_not_held_for_it():
    """The longest wait is several seconds; a stop must not queue behind it."""
    engine = _Engine("429 rate limit", fail_times=99)
    engine.request_stop()

    started = time.time()
    _run(engine, transient_retries=3)

    assert time.time() - started < 1.0, "it noticed the stop inside the wait"
    assert engine.attempts == 1


def test_the_backoff_grows_and_is_jittered(monkeypatch):
    """Lockstep retries from a fan-out rebuild the spike that caused them."""
    waits: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: waits.append(s))
    engine = _Engine("429 rate limit", fail_times=99)
    loop = AgentLoop(engine, toolbox=None, transient_retries=3)

    delays = []
    for attempt in range(3):
        waits.clear()
        before = time.time()
        loop._retry_transient(
            EventError(error="429", kind="retryable"), attempt, emitted=0,
        )
        delays.append(time.time() - before)

    assert delays[1] > delays[0] and delays[2] > delays[1], "1s, 2s, 4s"
    assert delays[0] >= 1.0 and delays[2] < 6.0, "with jitter on top, capped"
