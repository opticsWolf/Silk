# -*- coding: utf-8 -*-
"""What a run costs, and what happens when nobody will say (D88).

Cost is the one cap whose denominator lives outside Silk: tokens and
requests are counted here, but money is what someone else charges. So
the tests split in two -- the arithmetic, and the refusal to pretend.
"""

from __future__ import annotations

import pytest

from silk.functions.agent_loop import AgentLoop
from silk.functions.pricing import (
    ModelPrice, cost_of, price_from_handle, price_from_spec,
)
from silk.functions.stream_events import EventError, EventRunResult
from silk.functions.usage_limits import (
    SubBudget, UsageLimitExceeded, UsageLimits, describe_budget, format_cost,
    parse_budget,
)


# ── reading a price off the wire ─────────────────────────────────────────

def test_a_quoted_price_is_read_as_the_provider_writes_it():
    """OpenRouter quotes per-token decimal strings, not per-million floats."""
    price = price_from_spec(
        {"pricing": {"prompt": "0.000000075", "completion": "0.0000005"}},
        source="openrouter",
    )
    assert price is not None
    assert price.per_million() == pytest.approx((0.075, 0.5))
    assert price.source == "openrouter"


@pytest.mark.parametrize("spec", [
    {}, {"pricing": None}, {"pricing": {}},
    {"pricing": {"prompt": "free"}},
    {"pricing": {"prompt": "0.1"}},            # completion missing
    {"pricing": {"prompt": "-1", "completion": "1"}},
])
def test_an_unreadable_quote_is_no_price_rather_than_zero(spec):
    """A malformed quote must not become a confident claim that it is free."""
    assert price_from_spec(spec) is None


def test_a_quoted_zero_is_a_price_and_says_so():
    price = price_from_spec({"pricing": {"prompt": "0", "completion": "0"}})
    assert price is not None and price.is_free()
    assert price.describe() == "free"


def test_unknown_cost_is_none_not_zero():
    """Zero claims the run was free; None says nobody would say."""
    assert cost_of(None, 1000, 1000) is None
    assert cost_of(ModelPrice(1e-6, 2e-6), 1000, 1000) == pytest.approx(0.003)


def test_a_handle_carries_its_price_back():
    handle = {"backend": "openai", "pricing": {
        "input": 1e-6, "output": 2e-6, "currency": "USD", "source": "openrouter"}}
    price = price_from_handle(handle)
    assert price is not None and price.output_per_token == 2e-6
    assert price_from_handle({"backend": "gguf"}) is None


# ── the cap ──────────────────────────────────────────────────────────────

def test_cost_is_claimed_atomically_like_every_other_cap():
    limits = UsageLimits(cost_limit=1.0)
    limits.reserve_cost(0.6)
    with pytest.raises(UsageLimitExceeded, match="cost_limit"):
        limits.reserve_cost(0.6)
    assert limits.snapshot()["_cost_used"] == pytest.approx(0.6), (
        "a refused claim is not charged"
    )


def test_a_worker_cannot_spend_past_the_shared_cost_cap():
    """D26/T3: the worker's own cap does not raise the orchestrator's."""
    shared = UsageLimits(cost_limit=1.0)
    worker = SubBudget(cost_limit=10.0, parent=shared)
    worker.reserve_cost(0.9)
    with pytest.raises(UsageLimitExceeded) as caught:
        worker.reserve_cost(0.5)
    assert caught.value.scope == "shared"
    assert worker.snapshot()["_cost_used"] == pytest.approx(0.9), (
        "and it is refunded its own claim when the shared one refuses"
    )


@pytest.mark.parametrize("text, expected", [
    ("cost=0.50", 0.5), ("cost=$2", 2.0), ("spend=50c", 0.5),
    ("budget=1.25", 1.25), ("requests=5, cost=0.10", 0.10),
])
def test_a_money_cap_is_written_the_way_money_is(text, expected):
    limits = parse_budget(text)
    assert limits is not None
    assert limits.cost_limit == pytest.approx(expected)


def test_a_money_cap_is_parsed_apart_from_the_token_caps():
    """`0.50` is a good budget and a bad token count, so they differ."""
    with pytest.raises(ValueError):
        parse_budget("output=0.5")
    with pytest.raises(ValueError):
        parse_budget("cost=0")
    assert parse_budget("requests=5").cost_limit is None


def test_small_amounts_are_readable_rather_than_scientific():
    """Per-token prices run to six decimals; `1e-05` helps nobody."""
    assert "e-" not in format_cost(0.0000075), "never scientific"
    assert format_cost(0.0000075).startswith("$0.00000"), (
        "six decimals: enough to show a sub-cent spend is sub-cent, and a "
        "spend total is not a per-token price"
    )
    assert format_cost(0.001) == "$0.001", "no trailing zeros to read past"
    assert format_cost(2) == "$2.00"
    assert "cost $0.50" in describe_budget(UsageLimits(cost_limit=0.5))


# ── the refusal ──────────────────────────────────────────────────────────

class _Engine:
    def __init__(self, priced, limits):
        self._priced = priced
        self.usage_limits = limits
        self.last_stats = {}
        self.history = []
        self.system_prompt = ""
        self.ran = False

    def can_price(self):
        return self._priced

    def stream_response(self, gen_params):
        self.ran = True
        yield "hello"

    def append_message(self, role, content, **kw):
        self.history.append((role, content))

    def count_prompt_tokens(self):
        return 1

    def context_length(self):
        return None

    def request_stop(self):
        pass

    def stop_requested(self):
        return False


def test_a_cost_cap_over_an_unpriced_model_refuses_before_spending():
    """A ceiling that cannot bind reads like one that is not being neared.

    The person would discover the difference from the bill, which is the
    worst possible place. One edit -- drop the cap, or point at an
    endpoint that quotes -- is cheaper than that.
    """
    engine = _Engine(priced=False, limits=UsageLimits(cost_limit=5.0))
    events = list(AgentLoop(engine, toolbox=None).run("go"))

    assert not engine.ran, "refused before the first request, so before any spend"
    errors = [e for e in events if isinstance(e, EventError)]
    assert errors and "does not quote a price" in errors[0].error
    assert "local model has no price" in errors[0].error, "and says what to do"
    result = [e for e in events if isinstance(e, EventRunResult)][-1]
    assert result.finish_reason == "usage_limit", (
        "a refused run still reports an outcome (G13) rather than just stopping"
    )


def test_an_unpriced_model_runs_fine_without_a_cost_cap():
    """Unpriced is the normal case -- every local model is."""
    engine = _Engine(priced=False, limits=UsageLimits())
    list(AgentLoop(engine, toolbox=None).run("go"))
    assert engine.ran


def test_a_priced_model_runs_under_its_cap():
    engine = _Engine(priced=True, limits=UsageLimits(cost_limit=5.0))
    list(AgentLoop(engine, toolbox=None).run("go"))
    assert engine.ran
