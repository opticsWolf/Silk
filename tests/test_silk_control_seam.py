# -*- coding: utf-8 -*-
"""An agent builds, evaluates and reads a headless graph through Weave's
control API alone (Weave manifest P3 gate).

The graph tools are the ones a canvas run gets; only the seam differs:
`ControlSeam` turns each op into `weave-control 1.0` calls on a headless
Weave -- in this process (`weave.control.LocalControl`) or run as a child
(`weave.control.ControlClient.spawn`) -- with no canvas and no Qt on this
side. The rules above the seam hold unchanged: the whitelist, the run
scope, the self-modification guard, approval of the destructive verbs.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

weave_rs = pytest.importorskip("weave_rs")
if not hasattr(weave_rs, "ControlServer"):
    pytest.skip("Weave without the control API (P3)", allow_module_level=True)

from weave.control import ControlClient, LocalControl, headless_document  # noqa: E402

from silk.functions.approval import bind_run_seam  # noqa: E402
from silk.functions.control_seam import ControlSeam, control_binding  # noqa: E402
from silk.functions.decision_seam import DecisionSeam  # noqa: E402
from silk.functions.graph_author import (  # noqa: E402
    OP_CONNECT, OP_DESCRIBE, OP_EVALUATE, OP_LIST, OP_PLACE, OP_READ, OP_REMOVE,
    OP_SET_VALUE, OP_SETTINGS, OPS, bind_canvas,
)
from silk.functions.tool_box import ToolBox  # noqa: E402
from silk.functions.tools.graph_authoring import attach_graph_tools  # noqa: E402

WHITELIST = ("MultiFloatOutputNode", "PythonArithmeticNode")


def _approving_seam():
    holder: list = [None]

    def answer(request):
        holder[0].approve(request.decision_id, actor="frank", remember="")

    holder[0] = DecisionSeam(answer, timeout_s=5.0)
    return holder[0]


def _box(control, agent_uid=""):
    toolbox = ToolBox()
    attach_graph_tools(toolbox, sandbox=None, whitelist=WHITELIST)
    bind_canvas(toolbox, control_binding(control, agent_uid=agent_uid))
    bind_run_seam(toolbox, _approving_seam())
    return toolbox


def call(box, name, **args):
    """Invoke a tool the way the model does -- through dispatch."""
    request = SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    out = asyncio.run(box.execute_tool_calls_async([request]))
    return json.loads(out[0]["content"])


@pytest.fixture
def local():
    return LocalControl(headless_document(["weave.library.python_math"]))


def _build(box):
    """Source (1.0) -> arithmetic (+ B): what every gate test builds."""
    listed = call(box, OP_LIST)
    assert listed["ok"] and {n["class_name"] for n in listed["result"]["nodes"]} == set(WHITELIST)
    arith = next(n for n in listed["result"]["nodes"] if n["class_name"] == "PythonArithmeticNode")
    assert [p["name"] for p in arith["inputs"]] == ["A", "B"], "ports come from the declaration"
    src = call(box, OP_PLACE, class_name="MultiFloatOutputNode", x=0, y=0)
    add = call(box, OP_PLACE, class_name="PythonArithmeticNode", x=300, y=0, title="Plus B")
    assert src["ok"] and add["ok"], (src, add)
    src_id, add_id = src["result"]["id"], add["result"]["id"]
    assert add["result"]["title"] == "Plus B"
    return src_id, add_id


def test_an_agent_builds_evaluates_and_reads_a_headless_graph(local):
    box = _box(local)
    assert set(OPS) <= set(box.tools), "the eight, and evaluate and read"
    src_id, add_id = _build(box)

    settings = call(box, OP_SETTINGS, id=add_id)["result"]["settings"]
    by_name = {s["name"]: s for s in settings}
    assert by_name["B"]["settable"] and by_name["B"]["datatype"] == "float"
    assert not by_name["op"]["settable"], "INTERNAL is not a setting"
    set_b = call(box, OP_SET_VALUE, id=add_id, port="B", value=4)
    assert set_b["ok"] and set_b["result"]["value"] == 4.0, set_b

    edge = call(box, OP_CONNECT, src_id=src_id, src_port="float_0", dst_id=add_id, dst_port="A")
    assert edge["ok"], edge
    graph = call(box, OP_DESCRIBE)["result"]
    assert graph["edges"] == [[src_id, "float_0", add_id, "A"]]
    assert next(n for n in graph["nodes"] if n["id"] == add_id)["inputs"][0]["connected"]

    evaluated = call(box, OP_EVALUATE)
    assert evaluated["ok"] and evaluated["result"]["errors"] == [], evaluated
    read = call(box, OP_READ, id=add_id, port="result")
    assert read["ok"] and read["result"]["value"] == 5.0, read

    # Every call was one undo step on the Weave that serves it (D72).
    state = local.call("history.state")
    assert state["undo_depth"] == 4 and state["undo_label"] == "Connect"


def test_the_rules_above_the_seam_still_hold(local):
    box = _box(local)
    off = call(box, OP_PLACE, class_name="TypeAdapterNode")
    assert not off["ok"] and "TypeAdapterNode" in off["message"], "the whitelist (D71)"
    user = local.call("node.add", kind="PythonArithmeticNode")["uid"]
    removed = call(box, OP_REMOVE, id=user)
    assert not removed["ok"], "a node the user placed is not this run's to remove"
    retuned = call(box, OP_SET_VALUE, id=user, port="B", value=2.0)
    assert not retuned["ok"]
    src_id, add_id = _build(box)
    wired = call(box, OP_CONNECT, src_id=src_id, src_port="float_0", dst_id=add_id, dst_port="A")
    assert wired["ok"]
    again = call(box, OP_CONNECT, src_id=src_id, src_port="float_1", dst_id=add_id, dst_port="A")
    assert not again["ok"] and "already connected" in again["message"]
    bad = call(box, OP_SET_VALUE, id=add_id, port="B", value="lots")
    assert not bad["ok"] and "float" in bad["message"]
    gone = call(box, OP_REMOVE, id=src_id)
    assert gone["ok"] and gone["result"]["edges_removed"] == 1, gone


def test_the_agents_own_path_is_never_evaluated(local):
    # A graph the agent runs in: the "agent" feeds nothing here, but its
    # upstream (the source) is protected from evaluation.
    src = local.call("node.add", kind="MultiFloatOutputNode")["uid"]
    agent = local.call("node.add", kind="PythonArithmeticNode")["uid"]
    local.call("wire.connect", src=src, output="float_0", dst=agent, input="A")
    box = _box(local, agent_uid=agent)
    refused = call(box, OP_EVALUATE, ids=[src])
    assert not refused["ok"] and "upstream" in refused["message"]
    everything = call(box, OP_EVALUATE)
    assert everything["ok"]
    assert agent not in everything["result"]["evaluated"] and src not in everything["result"]["evaluated"]
    read = call(box, OP_READ, id=agent, port="result")
    assert read["ok"], "its outputs are read as they are"


def test_a_refusal_and_a_dead_transport_are_results_not_exceptions(local):
    seam = ControlSeam(local)
    missing = seam.call(OP_READ, id="nope", port="result")
    assert not missing.ok and missing.performed and "nope" in missing.error

    class Broken:
        def call(self, method, **params):
            raise BrokenPipeError("the server went away")

    dead = ControlSeam(Broken()).call(OP_DESCRIBE)
    assert not dead.ok and not dead.performed and "went away" in dead.error
    assert "could not reach" in dead.failure_text()


def test_the_gate_holds_against_a_weave_in_another_process():
    with ControlClient.spawn(["weave.library.python_math"]) as client:
        box = _box(client)
        src_id, add_id = _build(box)
        assert call(box, OP_SET_VALUE, id=add_id, port="B", value=0.5)["ok"]
        assert call(box, OP_CONNECT, src_id=src_id, src_port="float_0", dst_id=add_id, dst_port="A")["ok"]
        assert call(box, OP_EVALUATE)["ok"]
        read = call(box, OP_READ, id=add_id, port="result")
        assert read["ok"] and read["result"]["value"] == 1.5, read
