# -*- coding: utf-8 -*-
"""Classify a failed model request before anything upstream reacts to it.

Every model-request failure arrives at the AgentLoop as the same thing: an
exception out of ``stream_response``, turned into
``EventError(context="stream_response")``. That single shape covers a dead
server, a network blip, and a prompt that no longer fits the context — and
those want three different answers. Compaction (spec D24) reacts to the
third; answering the first by spending a summarization request against a
corpse is the failure D40 exists to prevent.

So the rule is: classify, then react. Three orthogonal facts, each cheap:

``kind``
    ``overflow`` — the prompt (or prompt + completion) exceeds the
    context window. The only kind compaction may act on.
``retryable``
    the same request might succeed if issued again (a transport hiccup, a
    busy server, a 5xx).
``terminal``
    nothing about repeating the request helps — the server is gone, the
    model is missing, the request was malformed.

This is the model-side sibling of :func:`~.reflection.is_retryable_tool_error`,
which does the same job for tool results. Both are string matchers over
backend prose, because that is what the backends give us; both keep the
matching in one place so a new backend's wording is a one-line change here
rather than a scattered ``if "context" in msg`` in the loop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "OVERFLOW", "RETRYABLE", "TERMINAL", "TRUNCATED",
    "ModelErrorClass", "classify_model_error", "is_context_overflow",
]

#: The prompt does not fit. Compaction's trigger, and nothing else's.
OVERFLOW = "overflow"
#: Worth issuing again as-is.
RETRYABLE = "retryable"
#: Repeating the request cannot help.
TERMINAL = "terminal"
#: The stream ended mid-response without a terminal ``finish_reason``.
#: Its own kind because the request *succeeded* — the answer was cut off
#: (spec D43), which is a correctness problem, not a transport one.
TRUNCATED = "truncated"

#: Phrases that mean "too many tokens". llama.cpp, the llama-cpp-python
#: server, vLLM and the OpenAI-compatible providers each word it
#: differently; all of them say one of these.
_OVERFLOW_PATTERNS = (
    r"exceed\w*\s+context",
    r"context\s+(?:window|length|size)\s+(?:exceeded|overflow)",
    r"requested\s+tokens?\s+\(\d+\)\s+exceed",
    r"maximum\s+context\s+length",
    r"n_ctx",
    r"kv\s+cache\s+(?:is\s+)?full",
    r"prompt\s+is\s+too\s+long",
    r"too\s+many\s+tokens",
    r"input\s+is\s+too\s+long",
)

#: Phrases that mean "ask again". Deliberately narrow: the default for an
#: unrecognised error is terminal, so a new wording costs one wasted
#: request at most, never an infinite retry.
_RETRYABLE_PATTERNS = (
    r"\b(?:429|500|502|503|504)\b",
    r"timed?\s*out",
    r"temporarily\s+unavailable",
    r"rate\s*limit",
    r"connection\s+reset",
    r"connection\s+aborted",
    r"server\s+is\s+busy",
    r"try\s+again",
    r"overloaded",
)


@dataclass(frozen=True)
class ModelErrorClass:
    """What a model-request failure was, in the three terms upstream needs.

    ``kind`` is the single answer; the two booleans are the questions
    callers actually ask, so they never have to compare against the
    constants themselves.
    """

    kind: str
    message: str

    @property
    def is_overflow(self) -> bool:
        return self.kind == OVERFLOW

    @property
    def is_retryable(self) -> bool:
        return self.kind == RETRYABLE

    @property
    def is_terminal(self) -> bool:
        return self.kind in (TERMINAL, TRUNCATED)


def is_context_overflow(message: str) -> bool:
    """Whether *message* is a backend saying the prompt does not fit."""
    text = (message or "").lower()
    return any(re.search(p, text) for p in _OVERFLOW_PATTERNS)


def classify_model_error(
    error: object,
    *,
    truncated: bool = False,
) -> ModelErrorClass:
    """Classify one model-request failure.

    Args:
        error: the exception, or the message text, that ended the request.
        truncated: set when the stream ended without a terminal
            ``finish_reason`` — the D43 case, where nothing raised at all.

    Returns:
        A :class:`ModelErrorClass`. Unrecognised messages are ``terminal``:
        the safe default is to surface the failure rather than to spend
        another request guessing.
    """
    message = str(error or "").strip()
    if truncated:
        return ModelErrorClass(TRUNCATED, message or "the response stream ended early")
    if is_context_overflow(message):
        return ModelErrorClass(OVERFLOW, message)
    lowered = message.lower()
    if any(re.search(p, lowered) for p in _RETRYABLE_PATTERNS):
        return ModelErrorClass(RETRYABLE, message)
    return ModelErrorClass(TERMINAL, message)
