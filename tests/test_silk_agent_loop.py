# -*- coding: utf-8 -*-
"""AgentLoop tests with a fake engine + fake toolbox (Qt-free, no model).

Covers: plain completion, tool round, reflection retry, non-retryable
role_denied results, usage-limit stop, and the max_rounds cap — the
coverage gap the silk architecture review flagged as highest-risk.
"""

from __future__ import annotations

import json
import sys


from silk.functions.agent_loop import AgentLoop
from silk.functions.reflection import ReflectionConfig
from silk.functions.stream_events import (
    EventDelta,
    EventError,
    EventReflection,
    EventRunResult,
    EventStart,
    EventToolCall,
    EventToolResult,
    EventUsageLimit,
)
from silk.functions.usage_limits import UsageLimits

TOOL_FENCE = '```tool_call\n{"name": "echo", "arguments": {"text": "hi"}}\n```'


class FakeEngine:
    """Scripted AgentEngine: yields canned responses, records history."""

    def __init__(self, responses: list[str], usage_limits: UsageLimits | None = None):
        self._responses = list(responses)
        self.usage_limits = usage_limits or UsageLimits()
        self.reflection_config = ReflectionConfig(max_retries=2)
        self.history: list[dict] = []
        self.last_stats: dict = {}
        self._stopped = False

    def stream_response(self, gen_params):
        self.usage_limits.check_request()
        self.usage_limits.record_request()
        text = self._responses.pop(0) if self._responses else "(exhausted)"
        self.last_stats = {"tokens": len(text.split()), "input_tokens": 1,
                           "tps": 1.0, "finish_reason": "stop"}
        for word in text.split(" "):
            yield word + " "

    def append_message(self, role, content, **stats):
        self.history.append({"role": role, "content": content})

    def count_prompt_tokens(self):
        return 1

    def request_stop(self):
        self._stopped = True

    def stop_requested(self):
        return self._stopped


class FakeToolbox:
    """Returns scripted results per call name; counts executions."""

    def __init__(self, results: dict[str, str]):
        self.tools = {name: {} for name in results}
        self._results = results
        self.executed: list[str] = []

    async def execute_tool_calls_async(self, tool_calls):
        out = []
        for tc in tool_calls:
            self.executed.append(tc.function.name)
            body = self._results.get(
                tc.function.name,
                json.dumps({"error": f"Tool '{tc.function.name}' is not registered."}),
            )
            out.append({"tool_call_id": tc.id, "name": tc.function.name, "content": body})
        return out


def run_loop(engine, toolbox=None, **kwargs):
    loop = AgentLoop(engine, toolbox, **kwargs)
    return list(loop.run("do the thing", {"max_tokens": 32}))


def events_of(events, cls):
    return [e for e in events if isinstance(e, cls)]


# ── plain completion ────────────────────────────────────────────────────


def test_plain_completion_yields_start_deltas_and_run_result():
    engine = FakeEngine(["hello world"])
    events = run_loop(engine)
    assert isinstance(events[0], EventStart)
    assert events_of(events, EventDelta), "deltas must stream"
    run_results = events_of(events, EventRunResult)
    assert len(run_results) == 1
    assert "hello" in run_results[0].text
    # user + assistant persisted
    roles = [h["role"] for h in engine.history]
    assert roles == ["user", "assistant"]


def test_the_start_event_carries_the_prompt_the_engine_holds():
    """§22 q3: EventStart.system_prompt stops being permanently None."""
    engine = FakeEngine(["hi"])
    engine.system_prompt = "You are terse."
    start = run_loop(engine)[0]
    assert start.system_prompt == "You are terse."


def test_an_engine_without_a_system_prompt_says_none():
    """It is optional on the protocol, like context_length.

    An engine that assembles instructions somewhere else is still an
    engine, and an empty prompt is None rather than "" so a consumer
    cannot show a run an empty instruction block it never had.
    """
    assert run_loop(FakeEngine(["hi"]))[0].system_prompt is None

    engine = FakeEngine(["hi"])
    engine.system_prompt = ""
    assert run_loop(engine)[0].system_prompt is None


# ── tool round ───────────────────────────────────────────────────────────


def test_tool_round_executes_and_feeds_back():
    engine = FakeEngine([TOOL_FENCE, "final answer"])
    toolbox = FakeToolbox({"echo": "echo says hi"})
    events = run_loop(engine, toolbox)

    assert [e.tool_name for e in events_of(events, EventToolCall)] == ["echo"]
    tool_results = events_of(events, EventToolResult)
    assert len(tool_results) == 1 and not tool_results[0].error
    assert toolbox.executed == ["echo"]

    roles = [h["role"] for h in engine.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    run_result = events_of(events, EventRunResult)[0]
    assert run_result.text.strip() == "final answer"
    assert run_result.tool_calls == [{"name": "echo"}]


def test_tool_fence_without_toolbox_ends_run():
    engine = FakeEngine([TOOL_FENCE])
    events = run_loop(engine, toolbox=None)
    assert events_of(events, EventRunResult), "run must still complete"
    assert not events_of(events, EventToolCall)


# ── reflection ───────────────────────────────────────────────────────────


def test_retryable_tool_error_triggers_reflection():
    error_body = json.dumps({"error": "Validation error: bad args"})
    engine = FakeEngine([TOOL_FENCE, "final answer"])
    toolbox = FakeToolbox({"echo": error_body})
    events = run_loop(engine, toolbox)

    reflections = events_of(events, EventReflection)
    assert len(reflections) == 1
    # The errored result is fed back in the normal slot (the model sees the
    # actual tool output AND the UI gets an EventToolResult), then the retry
    # nudge follows in a template-safe `user` turn — never mid-conversation
    # `system`, which local chat templates mishandle.
    tool_results = events_of(events, EventToolResult)
    assert len(tool_results) == 1 and tool_results[0].error
    roles = [h["role"] for h in engine.history]
    assert roles == ["user", "assistant", "tool", "user", "assistant"]
    assert not any(h["role"] == "system" for h in engine.history)
    assert events_of(events, EventRunResult)[0].text.strip() == "final answer"


def test_parse_tool_error_ignores_null_or_empty_error_field():
    """A payload carrying ``error: null`` / ``error: ""`` is a success, not a
    failure — only a truthy error counts (regression: read_file-style output
    with ``error: null`` was misread as a failure)."""
    from silk.functions.reflection import parse_tool_error

    assert parse_tool_error(json.dumps({"content": "ok", "error": None})) == (False, "")
    assert parse_tool_error(json.dumps({"error": ""})) == (False, "")
    is_err, msg = parse_tool_error(json.dumps({"error": "boom"}))
    assert is_err and "boom" in msg


def test_successful_tool_with_null_error_field_triggers_no_reflection():
    """The read_file scenario: a successful call returns
    ``{...,"error":null}``. The loop must feed it back as a normal result and
    never fire a spurious "schema mismatch" retry."""
    ok_body = json.dumps({"path": "a.txt", "content": "5 facts...", "error": None})
    engine = FakeEngine([TOOL_FENCE, "analysis done"])
    toolbox = FakeToolbox({"echo": ok_body})
    events = run_loop(engine, toolbox)

    assert not events_of(events, EventReflection), \
        "a null error field must not trigger reflection"
    tool_results = events_of(events, EventToolResult)
    assert len(tool_results) == 1 and not tool_results[0].error
    roles = [h["role"] for h in engine.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert events_of(events, EventRunResult)[0].text.strip() == "analysis done"


def test_fanout_errors_emit_one_reflection_and_feed_all_results():
    """A round where several calls error must feed every result back (so the
    model sees them and the UI records them) but emit only ONE reflection
    nudge — a fan-out round can't burn the whole retry budget or spam the
    transcript with a nudge per call."""
    two_calls = (
        '```tool_call\n{"name": "a", "arguments": {}}\n```\n'
        '```tool_call\n{"name": "b", "arguments": {}}\n```'
    )
    err = json.dumps({"error": "bad args"})
    engine = FakeEngine([two_calls, "final answer"])
    toolbox = FakeToolbox({"a": err, "b": err})
    events = run_loop(engine, toolbox)

    # Both errored results are fed back as tool turns and surfaced as events.
    tool_turns = [h for h in engine.history if h["role"] == "tool"]
    assert len(tool_turns) == 2
    assert len(events_of(events, EventToolResult)) == 2
    # ...but only one reflection nudge for the whole round.
    assert len(events_of(events, EventReflection)) == 1
    nudges = [h for h in engine.history
              if h["role"] == "user" and "did not match" in h["content"]]
    assert len(nudges) == 1
    assert events_of(events, EventRunResult)[0].text.strip() == "final answer"


def test_plaintext_tool_error_does_not_crash_reflection():
    """A non-JSON ``Error: …`` result is retryable but carries no
    correct_schema; the reflection path must not raise (regression: the retry
    block referenced an undefined ``json`` name and NameError'd on this path)."""
    engine = FakeEngine([TOOL_FENCE, "recovered"])
    toolbox = FakeToolbox({"echo": "Error: disk offline"})
    events = run_loop(engine, toolbox)  # must not raise

    assert len(events_of(events, EventReflection)) == 1
    assert events_of(events, EventRunResult)[0].text.strip() == "recovered"


def test_role_denied_error_is_not_retried():
    denied = json.dumps({"error": "Tool 'echo' is not available to the active role.",
                         "error_type": "role_denied"})
    engine = FakeEngine([TOOL_FENCE, "adapted answer"])
    toolbox = FakeToolbox({"echo": denied})
    events = run_loop(engine, toolbox)

    assert not events_of(events, EventReflection), "role_denied must not burn retries"
    tool_results = events_of(events, EventToolResult)
    assert tool_results and tool_results[0].error
    # The denial is fed back as a tool turn so the model can adapt.
    assert any(h["role"] == "tool" for h in engine.history)
    assert events_of(events, EventRunResult)[0].text.strip() == "adapted answer"


# ── usage limits & bounds ────────────────────────────────────────────────


def test_request_limit_stops_the_loop():
    engine = FakeEngine([TOOL_FENCE, "never reached"],
                        usage_limits=UsageLimits(request_limit=1))
    toolbox = FakeToolbox({"echo": "ok"})
    events = run_loop(engine, toolbox)

    limits = events_of(events, EventUsageLimit)
    assert limits and limits[0].limit_type == "request"
    errors = events_of(events, EventError)
    assert errors and errors[-1].context == "usage_limits"

    # A run stopped by its budget still reports how it ended (G7/G13).
    # It used to end with no RunResult at all, so `usage_limited` -- an
    # outcome declared since G13 -- was never once set, and a consumer had
    # to infer the ending from "an error and no text".
    results = events_of(events, EventRunResult)
    assert results and results[0].outcome == "usage_limited"
    assert results[0].finish_reason == "usage_limit"


def test_the_event_says_which_cap_was_hit():
    """Both gates used to share one `try` and both said "request" (G7)."""
    budget = UsageLimits(input_tokens_limit=1)
    budget.record_input_tokens(1)        # the allowance is already spent
    engine = FakeEngine(["never reached"], usage_limits=budget)
    events = run_loop(engine, FakeToolbox({"echo": "ok"}))

    limits = events_of(events, EventUsageLimit)
    assert limits and limits[0].limit_type == "input_tokens", (
        "the message text used to be the only way to tell the two apart"
    )
    assert limits[0].scope == "own"


def test_the_event_says_whose_cap_it_was():
    """With nested budgets both can stop a request (D26); which is which
    is the difference between *this worker asked too much* and *the
    fan-out is spent*."""
    from silk.functions.usage_limits import nest

    shared = UsageLimits(request_limit=1)
    shared.record_request()          # someone else already spent it
    engine = FakeEngine(["never reached"],
                        usage_limits=nest(shared, UsageLimits(request_limit=9)))
    events = run_loop(engine, FakeToolbox({"echo": "ok"}))

    limits = events_of(events, EventUsageLimit)
    assert limits and limits[0].scope == "shared"


def test_max_rounds_cap():
    engine = FakeEngine([TOOL_FENCE] * 10)
    toolbox = FakeToolbox({"echo": "ok"})
    events = run_loop(engine, toolbox, max_rounds=3)

    assert len(events_of(events, EventToolCall)) == 3
    errors = events_of(events, EventError)
    assert any("max_rounds" in e.error for e in errors)
    # A capped run still emits a final RunResult with what it has.
    assert events_of(events, EventRunResult)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
