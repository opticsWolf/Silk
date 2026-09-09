# -*- coding: utf-8 -*-
"""Stops inside a tool batch (G8).

`stop_requested()` was read between rounds and at token boundaries, so a
batch already dispatched ran to the end whatever the user did -- for a
sequential batch of long calls, the stop was invisible until every one of
them had finished.

What this pins is the honest half: a call already running keeps running
(its only bound is the registration `timeout`), and **nothing further
starts**. Every skipped call still comes back with a result, because the
next request's message list is malformed without one. The other half of
G8 -- the deliberate block, where the approval gate waits on a human --
is cancelled directly through the seam (D38/D49) and is pinned in
`test_silk_decision_seam.py`.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace


from silk.functions.tool_box import ToolBox  # noqa: E402


def _calls(*names):
    return [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(
        name=n, arguments="{}")) for i, n in enumerate(names)]


def _run(box, calls):
    return asyncio.run(box.execute_tool_calls_async(calls))


def _box(sequential=False):
    """A toolbox whose tools record the order they ran in."""
    box = ToolBox(None, {"agent_id": "ag"})
    ran: list[str] = []

    for name in ("one", "two", "three"):
        def _make(tool_name):
            def _tool(_pool, _session, **_kw):
                ran.append(tool_name)
                return f"{tool_name} ok"
            return _tool
        box.register(name, f"the {name} tool", sequential=sequential)(_make(name))

    return box, ran


def test_without_a_bound_stop_nothing_changes():
    box, ran = _box()
    results = _run(box, _calls("one", "two"))
    assert ran == ["one", "two"]
    assert [r["content"] for r in results] == ["one ok", "two ok"]


def test_a_stopped_run_starts_no_call_in_the_batch():
    box, ran = _box()
    box.bind_stop(lambda: True)

    results = _run(box, _calls("one", "two"))
    assert ran == [], "the stop arrived before the batch was dispatched"
    assert all("Not run" in r["content"] for r in results)


def test_every_skipped_call_still_comes_back():
    """A missing result is a malformed message list, not a saved call."""
    box, _ran = _box()
    box.bind_stop(lambda: True)

    calls = _calls("one", "two", "three")
    results = _run(box, calls)
    assert [r["tool_call_id"] for r in results] == [c.id for c in calls]
    assert [r["name"] for r in results] == ["one", "two", "three"]


def test_a_skipped_call_is_not_an_error_the_model_should_retry():
    box, _ran = _box()
    box.bind_stop(lambda: True)

    content = _run(box, _calls("one"))[0]["content"]
    try:
        payload = json.loads(content)
    except ValueError:
        return          # plain text: nothing that reflection reads as an error
    assert not payload.get("error"), (
        "a stop is the user's decision, not something to nudge a retry for"
    )


def test_a_stop_mid_batch_stops_the_rest_of_a_sequential_queue():
    """The case G8 names: the queue is what there is still time to spare."""
    box, ran = _box(sequential=True)
    stopped = {"now": False}
    box.bind_stop(lambda: stopped["now"])

    # The first tool is what stops the run -- the way a user's Stop lands
    # while a long call is in flight.
    def _stopper(_pool, _session, **_kw):
        ran.append("one")
        stopped["now"] = True
        return "one ok"

    box.tools["one"]["executable"] = lambda **kw: _stopper(None, None, **kw)

    results = _run(box, _calls("one", "two", "three"))
    assert ran == ["one"], "two and three never started"
    assert results[0]["content"] == "one ok"
    assert all("Not run" in r["content"] for r in results[1:])


def test_a_broken_predicate_is_not_a_stop():
    box, ran = _box()

    def _boom():
        raise RuntimeError("no")

    box.bind_stop(_boom)
    _run(box, _calls("one"))
    assert ran == ["one"], "a predicate that broke must not cancel the run"


def test_unbinding_restores_the_batch():
    box, ran = _box()
    box.bind_stop(lambda: True)
    box.bind_stop(None)

    _run(box, _calls("one"))
    assert ran == ["one"], (
        "a ToolBox outlives a run, so a finished run's stop must not "
        "reach the next one's calls"
    )


# -- the loop is what binds it --------------------------------------------


def test_the_loop_binds_the_stop_for_the_run_and_unbinds_after():
    """The flag belongs to the run; the toolbox outlives it."""
    from test_silk_agent_loop import FakeEngine, run_loop

    seen: list = []

    class _Recording(ToolBox):
        def bind_stop(self, predicate):
            seen.append(predicate)
            super().bind_stop(predicate)

    box = _Recording(None, {"agent_id": "ag"})
    engine = FakeEngine(["nothing to call here"])
    run_loop(engine, box)

    assert len(seen) == 2 and seen[1] is None, "bound for the run, then cleared"
    assert callable(seen[0]) and seen[0]() is False
    assert box._should_stop is None
