# -*- coding: utf-8 -*-
"""How a run ended, and how big the room was.

Two Phase 1 items that are one line each in the spec and were load-bearing
gaps in the code. G13: a run that exhausted ``max_rounds`` still produced a
final assistant turn, so every consumer keying off "is there text" read it
as a clean finish. G14(c): the context window is known at model load and
stopped there, so the loop had no denominator for any pressure decision.
"""

from __future__ import annotations



from silk.functions.agent_loop import AgentLoop
from silk.functions.graph_engine import GraphEngine
from silk.functions.reflection import ReflectionConfig
from silk.functions.stream_events import (
    OUTCOME_COMPLETED,
    OUTCOME_ERROR,
    OUTCOME_STOPPED,
    EventRunResult,
    EventStart,
)
from silk.functions.usage_limits import UsageLimits

FENCE = '```tool_call\n{"name": "echo", "arguments": {"text": "hi"}}\n```'


class _Engine:
    def __init__(self, responses, context_length=None):
        self._responses = list(responses)
        self._context_length = context_length
        self.usage_limits = UsageLimits()
        self.reflection_config = ReflectionConfig(max_retries=2)
        self.history: list[dict] = []
        self.last_stats: dict = {}
        self._stopped = False

    def stream_response(self, gen_params):
        text = self._responses.pop(0) if self._responses else FENCE
        self.last_stats = {"tokens": 1, "input_tokens": 1, "tps": 1.0,
                           "finish_reason": "stop"}
        yield text

    def append_message(self, role, content, **stats):
        self.history.append({"role": role, "content": content})

    def count_prompt_tokens(self):
        return 1

    def request_stop(self):
        self._stopped = True

    def stop_requested(self):
        return self._stopped

    def context_length(self):
        return self._context_length


class _Toolbox:
    tools = {"echo": {}}

    async def execute_tool_calls_async(self, tool_calls):
        return [{"tool_call_id": c.id, "name": c.function.name, "content": "ok"}
                for c in tool_calls]


def _run(engine, toolbox=None, **kw):
    return list(AgentLoop(engine, toolbox, **kw).run("go"))


# -- outcome ---------------------------------------------------------------

def test_a_plain_answer_reports_completed():
    events = _run(_Engine(["here you go"]))
    result = [e for e in events if isinstance(e, EventRunResult)][0]
    assert result.outcome == OUTCOME_COMPLETED


def test_running_out_of_rounds_is_not_a_clean_finish():
    """The G13 case: text *and* failure. The outcome is what tells them apart."""
    engine = _Engine([FENCE, FENCE])          # never stops calling tools
    events = _run(engine, _Toolbox(), max_rounds=2)
    result = [e for e in events if isinstance(e, EventRunResult)][0]
    assert result.text, "there is a final assistant turn — that was the trap"
    assert result.outcome == OUTCOME_ERROR


def test_a_stopped_run_says_so():
    engine = _Engine([FENCE, "done"])
    engine.request_stop()
    events = _run(engine, _Toolbox())
    result = [e for e in events if isinstance(e, EventRunResult)][0]
    assert result.outcome == OUTCOME_STOPPED


# -- context length --------------------------------------------------------

def test_the_context_window_reaches_the_first_event():
    events = _run(_Engine(["hi"], context_length=8192))
    start = [e for e in events if isinstance(e, EventStart)][0]
    assert start.context_length == 8192


def test_an_engine_that_does_not_know_reports_none_rather_than_a_guess():
    class _Mute(_Engine):
        context_length = None                  # not callable: no such method

    events = _run(_Mute(["hi"]))
    start = [e for e in events if isinstance(e, EventStart)][0]
    assert start.context_length is None


def test_the_engine_reads_the_window_off_the_handle_or_the_pool():
    class _Pool:
        context_length = 4096

    class _Model:
        def create_chat_completion(self, messages, stream=False, **kw):
            return iter(())

        def tokenize(self, b):
            return [0]

    assert GraphEngine({"backend": "gguf", "model": _Model(),
                        "context_length": 2048}).context_length() == 2048
    assert GraphEngine({"backend": "gguf", "pool": _Pool()}).context_length() == 4096
    assert GraphEngine({"backend": "gguf", "model": _Model()}).context_length() is None
