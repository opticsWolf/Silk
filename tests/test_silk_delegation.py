# -*- coding: utf-8 -*-
"""What a fan-out tells the model, and what it does to a shared budget.

Spec D52 catalogues four ways ``delegate_parallel`` was silently wrong with
more than one assignment; D53 rules that it runs sequentially until there is
more than one backend to be concurrent across; D54 wires the two hooks
``run_subagent`` already accepted and the orchestrator never passed. These
tests pin the reported behaviour -- what the model is told -- because every
one of the four defects was invisible from the model's side.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest
from types import SimpleNamespace


from silk.functions.orchestrator import (
    _MAX_PARALLEL,
    attach_orchestrator_tools,
    set_orchestrator_observers,
)
from silk.functions.role import DEFAULT_ROLE
from silk.functions.subagent import AgentSpec
from silk.functions.tool_box import ToolBox
from silk.functions.usage_limits import (
    SubBudget, UsageLimitExceeded, UsageLimits, describe_budget, nest,
    parse_budget,
)


class _Model:
    def __init__(self, responses, on_request=None):
        self._responses = list(responses)
        self._on_request = on_request

    def create_chat_completion(self, messages, stream=False, **kw):
        if self._on_request is not None:
            self._on_request()
        text = self._responses.pop(0) if self._responses else "(exhausted)"

        def gen():
            yield {"choices": [{"delta": {"content": text}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        return gen()

    def tokenize(self, b):
        return [0] * max(1, len(b) // 4)


def _worker(name, responses, on_request=None):
    return AgentSpec(
        model_handle={"backend": "gguf", "model": _Model(responses, on_request)},
        toolset=None, role=DEFAULT_ROLE, name=name,
    )


def _orchestrator(workers, **kw):
    tb = ToolBox(db_pool=None, user_session={"agent_id": "orchestrator"})
    attach_orchestrator_tools(tb, workers=workers, **kw)
    return tb


def _call(tb, name, args):
    tc = SimpleNamespace(
        id="c1", function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )
    out = asyncio.run(tb.execute_tool_calls_async([tc]))
    return json.loads(out[0]["content"])


def _assign(*names):
    return {"assignments": [{"worker": n, "task": f"do {n}"} for n in names]}


# -- D52.3: refuse, never trim --------------------------------------------

def test_a_fan_out_over_the_cap_is_refused_rather_than_silently_trimmed():
    names = [f"w{i}" for i in range(_MAX_PARALLEL + 4)]
    tb = _orchestrator([_worker(n, ["done"]) for n in names])
    r = _call(tb, "delegate_parallel", _assign(*names))
    assert not r["ok"]
    assert not r["results"], "nothing ran"
    assert str(len(names)) in r["message"] and str(_MAX_PARALLEL) in r["message"]


# -- D52.1: a same-worker fan-out is an error the model can act on --------

def test_the_same_worker_twice_in_one_fan_out_is_refused_with_the_remedy():
    tb = _orchestrator([_worker("w", ["a", "b"]), _worker("other", ["c"])])
    r = _call(tb, "delegate_parallel", _assign("w", "other", "w"))
    assert not r["ok"] and not r["results"]
    assert "w" in r["message"] and "sequentially" in r["message"].lower()


# -- D53: sequential, and it says so by running in order -------------------

def test_assignments_run_one_at_a_time_in_the_order_given():
    order: list[str] = []
    live: list[str] = []
    overlapped: list[bool] = []

    def watcher(name):
        def _hit():
            live.append(name)
            overlapped.append(len(live) > 1)
            order.append(name)
            live.remove(name)
        return _hit

    tb = _orchestrator([_worker(n, ["done"], watcher(n)) for n in ("a", "b", "c")])
    r = _call(tb, "delegate_parallel", _assign("a", "b", "c"))
    assert r["ok"] and order == ["a", "b", "c"]
    assert not any(overlapped), "no two workers were in the model at once"


# -- D54: the two hooks that already existed ------------------------------

def test_worker_events_reach_the_orchestrator_tagged_with_the_worker():
    tb = _orchestrator([_worker("a", ["hello"]), _worker("b", ["there"])])
    seen: list[str] = []
    set_orchestrator_observers(tb, on_event=lambda worker, ev: seen.append(worker))
    _call(tb, "delegate_parallel", _assign("a", "b"))
    assert set(seen) == {"a", "b"}, "a fan-out is observable while it runs"


def test_stop_reaches_the_fan_out_instead_of_running_every_worker():
    ran: list[str] = []
    tb = _orchestrator([
        _worker(n, ["done"], (lambda n=n: ran.append(n))) for n in ("a", "b", "c")
    ])
    # Stop after the first worker has been into the model.
    set_orchestrator_observers(tb, should_stop=lambda: bool(ran))
    r = _call(tb, "delegate_parallel", _assign("a", "b", "c"))
    assert ran == ["a"], "the remaining workers never started"
    assert not r["ok"]
    assert [x["ok"] for x in r["results"]] == [True, False, False]
    assert "Stopped" in r["results"][1]["error"]


# -- D52.4: one budget, several claimants ---------------------------------

def test_a_shared_budget_cannot_be_overrun_by_concurrent_claims():
    """check-then-record is a race; reserve is not.

    Twelve threads racing for a budget of four. With the old
    check-then-record pair several could pass the same check before any of
    them recorded; a reservation makes exactly four succeed, every time.
    """
    budget = UsageLimits(request_limit=4)
    barrier = threading.Barrier(12)
    granted: list[int] = []
    lock = threading.Lock()

    def claim():
        barrier.wait()
        try:
            budget.reserve_request()
        except UsageLimitExceeded:
            return
        with lock:
            granted.append(1)

    threads = [threading.Thread(target=claim) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) == 4
    assert budget.snapshot()["_request_count"] == 4


# -- D26 / T3: a share of the shared budget -------------------------------
#
# One global cap answers "the fan-out may not cost more than this" and
# nothing else: a greedy worker spends the whole allowance and every other
# worker starts getting USAGE_LIMIT events. A sub-budget is the worker's
# own ceiling *inside* the shared one.


def test_a_worker_cannot_starve_the_fan_out():
    """The failure T3 named, prevented."""
    shared = UsageLimits(request_limit=10)
    greedy = nest(shared, UsageLimits(request_limit=4))
    patient = nest(shared, UsageLimits(request_limit=4))

    for _ in range(4):
        greedy.reserve_request()
    with pytest.raises(UsageLimitExceeded):
        greedy.reserve_request()

    for _ in range(4):
        patient.reserve_request()   # the second worker still has its share
    assert shared.snapshot()["_request_count"] == 8


def test_the_global_cap_still_binds_every_worker():
    """A sub-budget narrows; it can never raise the ceiling."""
    shared = UsageLimits(request_limit=3)
    worker = nest(shared, UsageLimits(request_limit=100))

    for _ in range(3):
        worker.reserve_request()
    with pytest.raises(UsageLimitExceeded) as caught:
        worker.reserve_request()
    assert "request_limit of 3" in str(caught.value), (
        "the worker hears about whichever cap it actually hit"
    )


def test_a_refused_parent_does_not_charge_the_worker():
    """No worker is billed for a request it was not allowed to make."""
    shared = UsageLimits(tool_calls_limit=1)
    worker = nest(shared, UsageLimits(tool_calls_limit=10))

    worker.reserve_tool_calls()
    with pytest.raises(UsageLimitExceeded):
        worker.reserve_tool_calls()

    assert worker.snapshot()["_tool_call_count"] == 1, (
        "the refused claim was refunded; a worker whose own counter drifts "
        "up on refusals loses its share to failures it never made"
    )
    assert shared.snapshot()["_tool_call_count"] == 1


def test_both_halves_are_optional():
    """A fan-out with only a global cap behaves as it always did."""
    shared = UsageLimits(request_limit=2)
    assert nest(shared, None) is shared
    own = UsageLimits(request_limit=2)
    assert nest(None, own) is own
    assert nest(None, None) is None


def test_nesting_is_idempotent():
    """Re-entering a run must not stack a sub-budget on a sub-budget."""
    shared = UsageLimits(request_limit=5)
    once = nest(shared, UsageLimits(request_limit=2))
    assert nest(shared, once) is once


def test_a_check_consults_both_ceilings():
    shared = UsageLimits(output_tokens_limit=100)
    worker = nest(shared, UsageLimits(output_tokens_limit=1000))
    shared.record_output_tokens(100)
    with pytest.raises(UsageLimitExceeded):
        worker.check_output_tokens(1)


def test_the_snapshot_shows_what_is_left_of_both():
    shared = UsageLimits(request_limit=5)
    worker = nest(shared, UsageLimits(request_limit=2))
    worker.reserve_request()
    body = worker.snapshot()
    assert body["_request_count"] == 1
    assert body["shared"]["_request_count"] == 1, (
        "a fan-out that looks fine per worker and is out of shared budget "
        "is exactly the situation worth being able to read"
    )


def test_concurrent_workers_cannot_overrun_the_shared_cap():
    """D52.4's race, through the nested path this time."""
    shared = UsageLimits(request_limit=4)
    workers = [nest(shared, UsageLimits(request_limit=3)) for _ in range(6)]
    granted, start = [], threading.Event()

    def claim(budget):
        start.wait()
        try:
            budget.reserve_request()
            granted.append(1)
        except UsageLimitExceeded:
            pass

    threads = [threading.Thread(target=claim, args=(b,))
               for b in workers for _ in range(2)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join()

    assert len(granted) == 4 and shared.snapshot()["_request_count"] == 4


def test_a_worker_spec_budget_becomes_a_sub_budget_of_the_shared_one():
    """The wiring: run_subagent nests rather than choosing (D26)."""
    from silk.functions.subagent import AgentSpec

    spec = AgentSpec(model_handle={}, usage_limits=UsageLimits(request_limit=2))
    shared = UsageLimits(request_limit=9)
    effective = nest(shared, spec.usage_limits)

    assert isinstance(effective, SubBudget) and effective.parent is shared
    assert effective.request_limit == 2


# -- D26: the surface -- text a person types, on a node --------------------
#
# The nesting mechanism landed before anything built a budget, so in a
# running graph there were no caps at all. parse_budget is the whole of the
# translation from a node's field to a UsageLimits.


def test_a_typed_budget_becomes_caps():
    budget = parse_budget("requests=20, tool_calls=50, output=8k")
    assert budget.request_limit == 20
    assert budget.tool_calls_limit == 50
    assert budget.output_tokens_limit == 8000, "'8k' is what people write"
    assert budget.input_tokens_limit is None, "unnamed means uncapped"


def test_an_empty_field_is_no_budget():
    """What every run did before there was a field, and most still want."""
    assert parse_budget("") is None
    assert parse_budget(None) is None
    assert parse_budget("   \n ") is None


def test_the_short_names_are_the_ones_people_type():
    budget = parse_budget("turns=3; tools=4, input=2m")
    assert budget.request_limit == 3
    assert budget.tool_calls_limit == 4
    assert budget.input_tokens_limit == 2_000_000


def test_a_misspelled_limit_is_refused_not_ignored():
    """A budget that quietly means 'unlimited' is not a budget (D77's shape)."""
    with pytest.raises(ValueError) as caught:
        parse_budget("req=5")
    assert "'req' is not a limit name" in str(caught.value)
    assert "requests" in str(caught.value), "it says what may be written"


def test_a_budget_of_zero_is_refused():
    with pytest.raises(ValueError) as caught:
        parse_budget("requests=0")
    assert "would refuse the run's first request" in str(caught.value)
    with pytest.raises(ValueError):
        parse_budget("requests=-4")


def test_unreadable_shapes_are_refused():
    for text in ("requests", "requests=x", "20"):
        with pytest.raises(ValueError):
            parse_budget(text)


def test_the_same_limit_twice_with_two_answers_is_refused():
    assert parse_budget("requests=5, turns=5").request_limit == 5, (
        "two names for one cap agreeing is not a conflict"
    )
    with pytest.raises(ValueError):
        parse_budget("requests=5, turns=6")


def test_a_budget_describes_itself_for_a_status_line():
    assert describe_budget(None) == "no budget"
    assert describe_budget(UsageLimits()) == "no budget"
    assert describe_budget(parse_budget("requests=20, output=8k")) == (
        "requests 20, output 8,000"
    )


def test_a_typed_worker_budget_nests_under_a_typed_shared_one():
    """The two fields meet exactly where the two objects do."""
    shared = parse_budget("requests=10")
    own = parse_budget("requests=3")
    worker = nest(shared, own)
    for _ in range(3):
        worker.reserve_request()
    with pytest.raises(UsageLimitExceeded):
        worker.reserve_request()
    assert shared.snapshot()["_request_count"] == 3
