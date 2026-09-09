# -*- coding: utf-8 -*-
"""Model-request failures: classification, and the truncation that hides.

Two spec decisions meet here. D40 says a stream failure must be classified
before anything reacts to it, because ``context="stream_response"`` covers
both "the prompt no longer fits" (which compaction answers) and "the server
died" (which it must not). D43 says the shared llama.cpp server truncates an
in-flight stream when a second agent asks, and that the truncation arrives
looking exactly like a clean finish — so the loop has to notice the one
thing that is missing rather than the one thing that is wrong.
"""

from __future__ import annotations



from silk.functions.agent_loop import AgentLoop
from silk.functions.graph_engine import GraphEngine
from silk.functions.model_errors import (
    OVERFLOW,
    RETRYABLE,
    TERMINAL,
    TRUNCATED,
    classify_model_error,
)
from silk.functions.reflection import ReflectionConfig
from silk.functions.stream_events import EventError, EventRunResult
from silk.functions.usage_limits import UsageLimits


# -- the classifier --------------------------------------------------------

def test_an_overflow_is_told_from_a_dead_server():
    assert classify_model_error(
        "Requested tokens (5000) exceed context window of 4096").kind == OVERFLOW
    assert classify_model_error("llama_decode: n_ctx exceeded").kind == OVERFLOW
    assert classify_model_error(
        "Failed to reach local Llama server: connection refused").kind == TERMINAL


def test_a_transport_hiccup_is_retryable_and_an_unknown_error_is_not():
    assert classify_model_error("HTTP 503: temporarily unavailable").kind == RETRYABLE
    assert classify_model_error("read timed out").kind == RETRYABLE
    # The default has to be terminal: an unrecognised message retried
    # forever is worse than one surfaced once.
    assert classify_model_error("something nobody has seen before").kind == TERMINAL
    assert classify_model_error("").kind == TERMINAL


def test_truncation_is_its_own_kind_and_never_retryable():
    verdict = classify_model_error("", truncated=True)
    assert verdict.kind == TRUNCATED
    assert verdict.is_terminal and not verdict.is_retryable


# -- the engine notices the missing finish_reason --------------------------

class _Stream:
    """A minimal ``create_chat_completion`` returning scripted chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    def create_chat_completion(self, messages, stream=False, **kwargs):
        return iter(self._chunks)

    def tokenize(self, text):
        return [0]


def _chunk(text=None, finish=None):
    return {"choices": [{"delta": {"content": text} if text else {},
                         "finish_reason": finish}]}


def test_a_clean_stream_reports_its_finish_reason():
    engine = GraphEngine({"backend": "gguf",
                          "model": _Stream([_chunk("hi"), _chunk(finish="stop")])})
    assert "".join(engine.stream_response({})) == "hi"
    assert engine.last_stats["finish_reason"] == "stop"
    assert engine.last_stats["truncated"] is False


def test_a_stream_that_stops_without_a_reason_is_marked_truncated():
    engine = GraphEngine({"backend": "gguf", "model": _Stream([_chunk("half an ")])})
    assert "".join(engine.stream_response({})) == "half an "
    # The value that used to hide this was the default "stop".
    assert engine.last_stats["finish_reason"] is None
    assert engine.last_stats["truncated"] is True


# -- and the loop refuses to reason over half an answer --------------------

class _EngineWithStats:
    def __init__(self, text, stats):
        self._text, self._stats = text, stats
        self.usage_limits = UsageLimits()
        self.reflection_config = ReflectionConfig()
        self.history: list[dict] = []
        self.last_stats: dict = {}

    def stream_response(self, gen_params):
        yield self._text
        self.last_stats = dict(self._stats)

    def append_message(self, role, content, **stats):
        self.history.append({"role": role, "content": content})

    def count_prompt_tokens(self):
        return 1

    def request_stop(self):
        pass

    def stop_requested(self):
        return False


def test_the_loop_fails_the_round_on_a_truncated_stream():
    engine = _EngineWithStats("half an ans", {"finish_reason": None, "truncated": True})
    events = list(AgentLoop(engine).run("go"))
    errors = [e for e in events if isinstance(e, EventError)]
    assert errors and errors[0].kind == TRUNCATED
    assert errors[0].context == "stream_response"
    # No run result: a truncated turn is not a finished one.
    assert not [e for e in events if isinstance(e, EventRunResult)]


def test_the_loop_carries_the_classification_of_a_raised_failure():
    class _Boom(_EngineWithStats):
        def stream_response(self, gen_params):
            raise RuntimeError("Requested tokens exceed context window")
            yield  # pragma: no cover - generator marker

    events = list(AgentLoop(_Boom("", {})).run("go"))
    errors = [e for e in events if isinstance(e, EventError)]
    assert errors and errors[0].kind == OVERFLOW
