# -*- coding: utf-8 -*-
"""Invariant I11: the model-visible prefix grows only at the tail.

The guard is a detector, so the tests are mostly about *not* crying wolf.
A run that appends is the overwhelmingly common case and must stay silent;
the three ways a prefix actually breaks must each be named, because they
have three different causes and three different fixes. And compaction --
the one deliberate invalidation -- must be forgiven exactly once, not
forever, or the guard stops reporting the moment a run compacts.
"""

from __future__ import annotations


import pytest


from silk.functions.prefix_guard import (  # noqa: E402
    KIND_HISTORY,
    KIND_SYSTEM,
    KIND_TOOLS,
    PrefixGuard,
)

SYSTEM = "You are a careful agent."
TOOLS = [{"type": "function", "function": {"name": "read_file"}}]


def _turn(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _run(guard: PrefixGuard, messages: list, **kw):
    return guard.observe(messages, system_prompt=SYSTEM, **kw)


# -- the common case: appending ---------------------------------------------


def test_a_run_that_only_appends_never_reports():
    guard = PrefixGuard()
    history = [_turn("system", SYSTEM), _turn("user", "do the thing")]
    assert _run(guard, history) is None, "the first request has no predecessor"

    for reply in ("thinking", "tool output", "done"):
        history = [*history, _turn("assistant", reply)]
        assert _run(guard, history) is None

    assert guard.clean and guard.requests == 4
    assert "prefix intact" in guard.report()


def test_an_identical_request_is_not_a_break():
    """A retry re-sends the same messages; nothing was invalidated."""
    guard = PrefixGuard()
    history = [_turn("user", "hi")]
    _run(guard, history)
    assert _run(guard, list(history)) is None


def test_key_order_is_not_a_break():
    guard = PrefixGuard()
    _run(guard, [{"role": "user", "content": "hi"}])
    assert _run(guard, [{"content": "hi", "role": "user"}]) is None, (
        "two dicts that render the same prompt are the same prefix"
    )


# -- the three ways it breaks -----------------------------------------------


def test_a_volatile_system_prompt_is_the_expensive_kind():
    guard = PrefixGuard()
    history = [_turn("user", "hi")]
    guard.observe(history, system_prompt="You are an agent. It is 10:04.")
    broken = guard.observe(history, system_prompt="You are an agent. It is 10:05.")

    assert broken is not None and broken.kind == KIND_SYSTEM
    assert broken.request_index == 1
    assert "byte-identically" in broken.detail


def test_changing_the_advertised_tools_is_a_break():
    """Deferred capability loading does this by design; it still costs."""
    guard = PrefixGuard()
    history = [_turn("user", "hi")]
    _run(guard, history, tools=TOOLS)
    broken = _run(guard, history, tools=[*TOOLS, {"function": {"name": "write"}}])

    assert broken is not None and broken.kind == KIND_TOOLS


def test_rewriting_an_already_sent_message_is_a_break():
    guard = PrefixGuard()
    _run(guard, [_turn("user", "hi"), _turn("assistant", "hello")])
    broken = _run(guard, [_turn("user", "hi"), _turn("assistant", "HELLO"),
                          _turn("user", "again")])

    assert broken is not None and broken.kind == KIND_HISTORY
    assert broken.position == 1, "the guard names where it stopped agreeing"


def test_a_shrinking_history_is_a_break():
    guard = PrefixGuard()
    _run(guard, [_turn("user", "a"), _turn("assistant", "b"),
                 _turn("user", "c")])
    broken = _run(guard, [_turn("user", "a")])

    assert broken is not None and broken.kind == KIND_HISTORY
    assert "shrank from 3 to 1" in broken.detail


def test_only_the_first_break_of_a_request_is_reported():
    """Everything after a divergence is a consequence, not a cause."""
    guard = PrefixGuard()
    guard.observe([_turn("user", "a"), _turn("assistant", "b")],
                  system_prompt="one", tools=TOOLS)
    broken = guard.observe([_turn("user", "A"), _turn("assistant", "B")],
                           system_prompt="two", tools=[])

    assert broken is not None and broken.kind == KIND_SYSTEM
    assert len(guard.breaks) == 1


# -- the one deliberate invalidation ----------------------------------------


def test_compaction_is_forgiven():
    guard = PrefixGuard()
    _run(guard, [_turn("user", "a"), _turn("assistant", "b"),
                 _turn("user", "c")])

    guard.note_compaction()
    assert _run(guard, [_turn("assistant", "summary of a-c")]) is None
    assert guard.clean, "compaction is the single legitimate break"


def test_compaction_forgives_exactly_once():
    """Otherwise the guard goes quiet for the rest of a run that compacts."""
    guard = PrefixGuard()
    _run(guard, [_turn("user", "a"), _turn("assistant", "b")])
    guard.note_compaction()
    _run(guard, [_turn("assistant", "summary")])

    broken = _run(guard, [_turn("assistant", "rewritten summary")])
    assert broken is not None and broken.kind == KIND_HISTORY
    assert not guard.clean


def test_a_clean_run_after_a_forgiven_compaction_stays_clean():
    guard = PrefixGuard()
    _run(guard, [_turn("user", "a"), _turn("assistant", "b")])
    guard.note_compaction()
    base = [_turn("assistant", "summary")]
    _run(guard, base)
    _run(guard, [*base, _turn("user", "next")])
    assert guard.clean


# -- reporting ---------------------------------------------------------------


def test_the_report_names_every_break():
    guard = PrefixGuard()
    _run(guard, [_turn("user", "a")], tools=TOOLS)
    _run(guard, [_turn("user", "a")], tools=[])
    _run(guard, [_turn("user", "b")], tools=[])

    lines = guard.report().splitlines()
    assert len(lines) == 2
    assert KIND_TOOLS in lines[0] and KIND_HISTORY in lines[1]
    assert "message 0" in lines[1]


@pytest.mark.parametrize("payload", [
    object(),                       # not JSON-serialisable
    {"role": "user", "content": None},
])
def test_an_unserialisable_message_does_not_raise(payload):
    guard = PrefixGuard()
    assert _run(guard, [payload]) is None
    assert _run(guard, [payload]) is None, "the fallback must be stable too"


# -- the engine wires it in --------------------------------------------------


def test_the_engine_carries_a_guard():
    from silk.functions.graph_engine import GraphEngine

    engine = GraphEngine({"backend": "gguf", "model": object()},
                         system_prompt=SYSTEM)
    assert isinstance(engine.prefix_guard, PrefixGuard)
    assert engine.prefix_guard.clean

    # The engine's own message builder is the thing under the rule: two
    # builds of the same state must agree, or nothing else can.
    engine.append_message("user", "hi")
    first = engine.build_messages()
    assert engine.prefix_guard.observe(first, system_prompt=SYSTEM) is None
    engine.append_message("assistant", "hello")
    second = engine.build_messages()
    assert engine.prefix_guard.observe(second, system_prompt=SYSTEM) is None
    assert second[:len(first)] == first, (
        "build_messages must extend, not rewrite"
    )
