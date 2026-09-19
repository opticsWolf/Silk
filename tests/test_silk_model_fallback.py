# -*- coding: utf-8 -*-
"""D89: a model handle that names its successor, and a loop that walks it.

Three layers, tested separately because they fail separately:

* the chain arithmetic (`functions/model_fallback.py`) -- flattening,
  deduplication, and the three intersection rules;
* the engine's walk (`GraphEngine.advance_model`) -- which model is
  current, and what that changes;
* the loop's decision -- *when* a switch happens, which is the part that
  has to agree with the D87 retry boundary rather than duplicate it.
"""
from __future__ import annotations

import pytest

from weave.plugins.silk.functions.agent_loop import AgentLoop
from weave.plugins.silk.functions.graph_engine import GraphEngine
from weave.plugins.silk.functions.model_fallback import (
    FALLBACK_KEY,
    build_chain,
    chain_can_price,
    chain_context_length,
    chain_of,
    chain_supports_tools,
    describe_chain,
    model_label,
)
from weave.plugins.silk.functions.stream_events import (
    EventDelta,
    EventError,
    EventModelSwitch,
    EventRunResult,
)
from weave.plugins.silk.functions.usage_limits import UsageLimits


# -- helpers ---------------------------------------------------------------

def handle(backend="openai", name="m", **extra):
    """A minimal handle the ``model_handle`` validator would accept."""
    h = {"backend": backend, "model": object(), "model_alias": name}
    h.update(extra)
    return h


def priced(**extra):
    return handle(pricing={"input": 1e-6, "output": 2e-6, "source": "t"},
                  **extra)


# -- the chain arithmetic ---------------------------------------------------

def test_a_plain_handle_is_a_one_element_chain():
    """The general case is the only case: no branch for "has no fallback"."""
    h = handle()
    assert chain_of(h) == [h] or chain_of(h)[0]["backend"] == "openai"
    assert len(chain_of(h)) == 1


def test_a_non_handle_flattens_to_nothing():
    for bad in (None, {}, "gpt", {"backend": "openai"}, 7):
        assert chain_of(bad) == []


def test_build_chain_puts_the_rest_under_fallbacks():
    a, b = handle(name="a"), handle(name="b")
    chained = build_chain(a, b)
    assert chained is not None
    assert chained["model_alias"] == "a"
    assert [h["model_alias"] for h in chained[FALLBACK_KEY]] == ["b"]
    # And it is still a usable handle in its own right.
    assert chained["backend"] and "model" in chained


def test_build_chain_returns_none_when_nothing_is_usable():
    """An unconnected endpoint must fail at its own node, not at the agent."""
    assert build_chain(None, None) is None
    assert build_chain(None, {}) is None


def test_build_chain_passes_a_lone_survivor_through():
    chained = build_chain(None, handle(name="only"))
    assert chained is not None
    assert chained["model_alias"] == "only"
    assert FALLBACK_KEY not in chained


def test_chains_compose_when_the_nodes_are_wired_in_series():
    a, b, c = handle(name="a"), handle(name="b"), handle(name="c")
    chained = build_chain(build_chain(a, b), c)
    assert [h["model_alias"] for h in chain_of(chained)] == ["a", "b", "c"]


def test_the_same_model_is_never_its_own_fallback():
    """Falling back to the model that just failed cannot help."""
    a = handle(name="a")
    same_client = dict(a)      # a second node, one client
    chained = build_chain(a, same_client)
    assert len(chain_of(chained)) == 1


def test_the_head_never_carries_a_stale_chain():
    """Flattening strips ``fallbacks`` so the walk cannot recurse."""
    chained = build_chain(handle(name="a"), handle(name="b"))
    assert all(FALLBACK_KEY not in h for h in chain_of(chained))


# -- the three intersection rules ------------------------------------------

def test_native_tools_need_every_member():
    both = [handle(supports_tools=True), handle(supports_tools=True)]
    mixed = [handle(supports_tools=True), handle()]
    assert chain_supports_tools(both) is True
    assert chain_supports_tools(mixed) is False
    assert chain_supports_tools([]) is False


def test_the_context_window_is_the_smallest_known_one():
    assert chain_context_length(
        [handle(context_length=32_000), handle(context_length=8_192)]
    ) == 8_192


def test_an_unknown_window_is_skipped_not_treated_as_zero():
    assert chain_context_length(
        [handle(context_length=8_192), handle()]
    ) == 8_192
    assert chain_context_length([handle(), handle()]) is None


def test_a_pool_reports_its_own_window():
    class _Pool:
        context_length = 4_096

    h = {"backend": "gguf", "pool": _Pool()}
    assert chain_context_length([h]) == 4_096


def test_one_unpriced_member_makes_the_chain_unpriced():
    assert chain_can_price([priced(), priced()]) is True
    assert chain_can_price([priced(), handle()]) is False


def test_describe_chain_reads_in_walk_order():
    text = describe_chain(chain_of(build_chain(
        handle("openai", "gpt-x"), handle("gguf", "qwen"),
    )))
    assert text == "openai:gpt-x → gguf:qwen"


def test_model_label_keeps_only_the_leaf_of_a_path():
    assert model_label({"backend": "gguf", "model": 1,
                        "model_path": "C:\\models\\q.gguf"}) == "gguf:q.gguf"
    assert model_label({"backend": "openai", "model": 1}) == "openai"
    assert model_label(None) == "<no model>"


# -- the engine's walk ------------------------------------------------------

def test_the_engine_starts_on_the_primary():
    eng = GraphEngine(build_chain(handle(name="a"), handle(name="b")))
    assert eng.chain_position() == (1, 2)
    assert eng.has_fallback() is True
    assert eng.model_description() == "openai:a"


def test_advance_moves_to_the_next_model_and_stops_at_the_end():
    eng = GraphEngine(build_chain(handle(name="a"), handle(name="b")))
    assert eng.advance_model() == "openai:b"
    assert eng.chain_position() == (2, 2)
    assert eng.has_fallback() is False
    assert eng.advance_model() is None


def test_advance_repoints_the_price():
    """The point of a cheap model behind an expensive one."""
    eng = GraphEngine(build_chain(
        priced(name="paid"),
        handle(name="local", pricing={"input": 0.0, "output": 0.0}),
    ))
    assert eng.price_description() == "$1.00/$2.00 per 1M tokens"
    eng.advance_model()
    assert eng.price_description() == "free"


def test_a_plain_handle_has_no_fallback_and_behaves_as_before():
    eng = GraphEngine(handle(supports_tools=True, context_length=99))
    assert eng.has_fallback() is False
    assert eng.advance_model() is None
    assert eng.supports_native_tools() is True
    assert eng.context_length() == 99
    assert eng.model_description() == "openai:m"


def test_the_engine_answers_capabilities_for_the_whole_chain():
    eng = GraphEngine(build_chain(
        priced(supports_tools=True, context_length=32_000),
        handle(context_length=8_192),
    ))
    assert eng.supports_native_tools() is False   # one member cannot
    assert eng.context_length() == 8_192          # the smaller window
    assert eng.can_price() is False               # one member unpriced


def test_a_sibling_inherits_the_whole_chain():
    """D25's summarizer must not be pinned to whichever model is current."""
    eng = GraphEngine(build_chain(handle(name="a"), handle(name="b")))
    eng.advance_model()
    assert eng.sibling().chain_position() == (1, 2)


# -- the loop's decision ----------------------------------------------------

class _Engine:
    """An engine whose per-attempt behaviour the test scripts."""

    def __init__(self, script, chain=("a", "b")):
        self.script = list(script)
        self.chain = list(chain)
        self.index = 0
        self.history: list = []
        self.usage_limits = UsageLimits()
        self.last_stats: dict = {}
        self.attempts: list[str] = []

    # -- the chain
    def chain_position(self):
        return (self.index + 1, len(self.chain))

    def model_description(self):
        return self.chain[self.index]

    def advance_model(self):
        if self.index + 1 >= len(self.chain):
            return None
        self.index += 1
        return self.chain[self.index]

    # -- the engine
    def stream_response(self, _params):
        step = self.script.pop(0) if self.script else "ok"
        self.attempts.append(f"{self.chain[self.index]}:{step}")
        if isinstance(step, Exception):
            raise step
        if step == "partial":
            yield "half "
            raise RuntimeError("Internal Server Error 500")
        self.last_stats = {"tokens": 1, "finish_reason": "stop",
                           "input_tokens": 1, "tps": 1.0}
        yield "done"

    def append_message(self, role, content, **kw):
        self.history.append((role, content))

    def count_prompt_tokens(self):
        return 1

    def context_length(self):
        return None

    def stop_requested(self):
        return False

    def request_stop(self):
        pass

    def clear_stop(self):
        pass

    def build_messages(self):
        return []


def run(engine, **kw):
    loop = AgentLoop(engine, toolbox=None, transient_retries=0, **kw)
    return list(loop.run("hi"))


def test_a_terminal_failure_falls_back_instead_of_ending_the_run():
    engine = _Engine([RuntimeError("model not found"), "ok"])
    events = run(engine)
    switches = [e for e in events if isinstance(e, EventModelSwitch)]
    assert len(switches) == 1
    assert (switches[0].from_model, switches[0].to_model) == ("a", "b")
    assert switches[0].kind == "terminal"
    result = [e for e in events if isinstance(e, EventRunResult)][-1]
    assert result.text == "done"
    assert engine.attempts == ["a:model not found", "b:ok"]


def test_the_failure_is_still_reported_before_the_switch():
    """A fallback that hides the primary's death is a fallback nobody
    can debug. The error event is yielded either way."""
    events = run(_Engine([RuntimeError("model not found"), "ok"]))
    kinds = [type(e).__name__ for e in events]
    assert kinds.index("EventError") < kinds.index("EventModelSwitch")


def test_the_chain_is_walked_to_its_end_then_the_run_ends():
    engine = _Engine(
        [RuntimeError("gone"), RuntimeError("gone"), RuntimeError("gone")],
        chain=("a", "b"),
    )
    events = run(engine)
    assert len([e for e in events if isinstance(e, EventModelSwitch)]) == 1
    assert engine.attempts == ["a:gone", "b:gone"]
    assert not [e for e in events if isinstance(e, EventRunResult)]


def test_retries_are_spent_on_the_primary_before_the_switch():
    """D87 first, D89 second: a 429 passes, a dead server does not."""
    engine = _Engine([
        RuntimeError("429 rate limit"), RuntimeError("429 rate limit"), "ok",
    ])
    loop = AgentLoop(engine, toolbox=None, transient_retries=1)
    loop._retry_transient = (                     # no real backoff in a test
        lambda ev, attempt, emitted: ev.kind == "retryable" and attempt < 1
        and not emitted
    )
    events = list(loop.run("hi"))
    assert engine.attempts == [
        "a:429 rate limit", "a:429 rate limit", "b:ok",
    ]
    assert len([e for e in events if isinstance(e, EventModelSwitch)]) == 1


def test_the_fallback_gets_a_retry_budget_of_its_own():
    engine = _Engine(
        [RuntimeError("gone"), RuntimeError("429 busy"), "ok"],
        chain=("a", "b"),
    )
    loop = AgentLoop(engine, toolbox=None, transient_retries=1)
    loop._retry_transient = (
        lambda ev, attempt, emitted: ev.kind == "retryable" and attempt < 1
        and not emitted
    )
    list(loop.run("hi"))
    assert engine.attempts == ["a:gone", "b:429 busy", "b:ok"]


def test_a_failure_after_deltas_does_not_switch():
    """Two voices must not be spliced into one turn -- the same boundary
    the D87 retry path draws."""
    engine = _Engine(["partial"])
    events = run(engine)
    assert [e for e in events if isinstance(e, EventDelta)]
    assert not [e for e in events if isinstance(e, EventModelSwitch)]
    assert engine.attempts == ["a:partial"]


def test_an_overflow_is_compactions_not_the_chains():
    """The next model's window is no larger -- the chain reports the
    smallest one -- so a switch would be a second way to fail."""
    engine = _Engine([RuntimeError("maximum context length exceeded")])
    events = run(engine)
    assert not [e for e in events if isinstance(e, EventModelSwitch)]
    assert engine.attempts == ["a:maximum context length exceeded"]
    assert [e for e in events if isinstance(e, EventError)][0].kind == "overflow"


def test_an_engine_without_a_chain_is_untouched():
    """Every other engine in the codebase lacks ``advance_model``."""
    class _Plain(_Engine):
        advance_model = None      # not callable -> no chain to walk

    events = run(_Plain([RuntimeError("model not found")]))
    assert not [e for e in events if isinstance(e, EventModelSwitch)]


def test_a_switch_does_not_spend_a_round():
    """`max_rounds` bounds the model's reasoning steps; being handed a
    dead server is not one of them."""
    engine = _Engine([RuntimeError("gone"), "ok"])
    loop = AgentLoop(engine, toolbox=None, max_rounds=1,
                     transient_retries=0)
    events = list(loop.run("hi"))
    assert [e for e in events if isinstance(e, EventRunResult)][-1].text == "done"


@pytest.mark.parametrize("cap_reachable", [True, False])
def test_a_cost_cap_binds_only_when_every_member_quotes(cap_reachable):
    second = priced(name="b") if cap_reachable else handle(name="b")
    eng = GraphEngine(
        build_chain(priced(name="a"), second),
        usage_limits=UsageLimits(cost_limit=1.0),
    )
    loop = AgentLoop(eng, toolbox=None)
    refusal = loop._unpriceable_cost_cap()
    assert (refusal is None) is cap_reachable
