# -*- coding: utf-8 -*-
"""The agent places nodes -- and the three rules that let it (§18, D69-D73).

This is the first Silk tool family whose effect is on Weave itself, so
what is pinned here is mostly what it *cannot* do: place a class nobody
whitelisted, remove something the user made, or touch the graph that is
running it. The canvas half is exercised through a fake resolver rather
than a live scene -- the seam is the boundary, and it is the same one the
real `CanvasAuthor` sits behind.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from silk.functions.graph_author import (
    CanvasBinding, OP_CONNECT, OP_DESCRIBE, OP_DISCONNECT, OP_LIST, OP_PLACE,
    OP_REMOVE, OP_SET_VALUE, OP_SETTINGS, OPS, RunScope, Whitelist,
    bind_canvas, canvas_binding, check_self_modification, check_settable,
    coerce_value, protected_nodes, upstream_of,
)
from silk.functions.approval import bind_run_seam
from silk.functions.decision_seam import DecisionSeam
from silk.functions.main_thread_call import MainThreadCall
from silk.functions.tool_box import ToolBox
from silk.functions.tools.graph_authoring import attach_graph_tools


# ── a fake canvas, on the calling thread ─────────────────────────────────

class FakeCanvas:
    """What the real `CanvasAuthor` does, minus Qt.

    Records every op so a test can ask what actually reached the canvas,
    which is the difference between "refused" and "did nothing visible".
    """

    def __init__(self, nodes=None, edges=None) -> None:
        self.nodes = list(nodes or [])
        self.edges = [list(e) for e in (edges or [])]
        self.calls: list = []
        self.next_id = "new-1"
        self.agent_id = "agent"
        self.settings = [{"name": "text", "role": "INPUT",
                          "datatype": "string", "value": "", "default": "",
                          "settable": True, "why_not": "",
                          "connected": False, "description": ""}]

    def deliver(self, request) -> None:
        self.seam.serve(request, self.perform)

    def perform(self, request) -> dict:
        self.calls.append((request.op, dict(request.args)))
        if request.op == OP_LIST:
            return {"ok": True, "value": {
                "nodes": [{"class_name": n} for n in request.args["allowed"]],
                "total": len(request.args["allowed"])}}
        if request.op == OP_DESCRIBE:
            return {"ok": True, "value": {"nodes": self.nodes,
                                          "edges": self.edges,
                                          "agent_id": self.agent_id}}
        if request.op == OP_PLACE:
            return {"ok": True, "value": {"id": self.next_id,
                                          "class_name": request.args["class_name"]}}
        if request.op == OP_CONNECT:
            edge = [request.args["src_id"], request.args["src_port"],
                    request.args["dst_id"], request.args["dst_port"]]
            self.edges.append(edge)
            return {"ok": True, "value": {"edge": edge}}
        if request.op == OP_DISCONNECT:
            return {"ok": True, "value": {"edge": list(request.args.values())}}
        if request.op == OP_REMOVE:
            return {"ok": True, "value": {"id": request.args["id"]}}
        if request.op == OP_SETTINGS:
            return {"ok": True, "value": {"id": request.args["id"],
                                          "settings": self.settings}}
        if request.op == OP_SET_VALUE:
            return {"ok": True, "value": {"id": request.args["id"],
                                          "port": request.args["port"],
                                          "value": request.args["value"]}}
        return {"ok": False, "error": f"unknown op {request.op}"}


@pytest.fixture
def canvas():
    return FakeCanvas(edges=[["box", "toolbox", "set", "toolbox"],
                             ["set", "toolset", "agent", "toolset"]])


@pytest.fixture
def box(canvas):
    """A ToolBox with the graph tools mounted and a canvas bound."""
    toolbox = ToolBox()
    attach_graph_tools(toolbox, sandbox=None,
                       whitelist=("TextNode", "SilkAgentNode"))
    seam = MainThreadCall(canvas.deliver, timeout_s=5.0)
    canvas.seam = seam
    bind_canvas(toolbox, CanvasBinding(seam=seam, scope=RunScope(),
                                       agent_uid="agent"))
    # `remove_node` and `disconnect` are registered `requires_approval=True`
    # (D73) and the floor under that flag (D81) asks whatever the policy
    # says, so a run with no decision seam refuses both. These tests are
    # about the *scope* rules, so the human here always says yes.
    bind_run_seam(toolbox, _approving_seam())
    return toolbox


def _approving_seam():
    holder: list = [None]

    def answer(request):
        holder[0].approve(request.decision_id, actor="frank", remember="")

    holder[0] = DecisionSeam(answer, timeout_s=5.0)
    return holder[0]


def call(box, name, **args):
    """Invoke a tool the way the model does -- through dispatch."""
    request = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )
    out = asyncio.run(box.execute_tool_calls_async([request]))
    return json.loads(out[0]["content"])


# ── the whitelist is the grant (D71) ─────────────────────────────────────


def test_the_eight_tools_are_registered(box):
    assert set(OPS) <= set(box.tools), "D69's six and q9's two"


def test_an_empty_whitelist_refuses_every_placement(canvas):
    toolbox = ToolBox()
    attach_graph_tools(toolbox, sandbox=None, whitelist=())
    seam = MainThreadCall(canvas.deliver, timeout_s=5.0)
    canvas.seam = seam
    bind_canvas(toolbox, CanvasBinding(seam=seam, scope=RunScope(),
                                       agent_uid="agent"))

    result = call(toolbox, OP_PLACE, class_name="TextNode")
    assert not result["ok"] and "whitelisted" in result["message"]
    assert canvas.calls == [], (
        "default-deny means the request never reaches the canvas: the "
        "safe state is the one you get by doing nothing (D71, I6)"
    )


def test_a_class_off_the_list_is_refused_by_name(box, canvas):
    result = call(box, OP_PLACE, class_name="ShellNode")
    assert not result["ok"]
    assert "ShellNode" in result["message"] and "TextNode" in result["message"], (
        "the refusal says what *is* allowed, so the model can pick again"
    )
    assert canvas.calls == []


def test_a_whitelisted_class_is_placed(box, canvas):
    result = call(box, OP_PLACE, class_name="TextNode", x=40, y=10)
    assert result["ok"] and result["result"]["id"] == "new-1"
    assert canvas.calls[-1][1]["position"] == [40.0, 10.0]


def test_the_whitelist_narrows_and_never_widens():
    allowed = Whitelist(("A", "B"))
    assert set(allowed.narrowed(("B", "C"))) == {"B"}, (
        "a ToolSet or Role may remove entries, never add (I6)"
    )
    assert set(allowed.narrowed(())) == set()


def test_a_whitelisted_class_that_vanished_is_reported():
    assert Whitelist(("A", "Gone")).missing(("A", "B")) == ["Gone"], (
        "surfaced at the node, not as a refusal an agent hits halfway "
        "through building something (D71)"
    )


# ── reads (D69) ──────────────────────────────────────────────────────────


def test_the_agent_can_read_what_it_may_place(box):
    result = call(box, OP_LIST)
    assert result["ok"]
    assert {n["class_name"] for n in result["result"]["nodes"]} == {
        "TextNode", "SilkAgentNode"}


def test_the_agent_can_read_the_graph(box, canvas):
    result = call(box, OP_DESCRIBE)
    assert result["ok"] and result["result"]["edges"] == canvas.edges
    assert result["result"]["agent_id"] == "agent", (
        "placement is relative: an agent that cannot see the graph cannot "
        "place a node beside one, or know which ports are free"
    )


# ── destructive calls are scoped to the run (D73) ────────────────────────


def test_a_node_this_run_did_not_place_cannot_be_removed(box, canvas):
    result = call(box, OP_REMOVE, id="user-node")
    assert not result["ok"] and "not created by this run" in result["message"]
    assert not any(op == OP_REMOVE for op, _ in canvas.calls), (
        "an agent may clean up after itself; it may not prune the user's "
        "graph"
    )


def test_what_this_run_placed_it_may_remove(box, canvas):
    placed = call(box, OP_PLACE, class_name="TextNode")["result"]["id"]
    assert call(box, OP_REMOVE, id=placed)["ok"]
    assert not call(box, OP_REMOVE, id=placed)["ok"], (
        "and once removed it is no longer this run's to remove again"
    )


def test_an_edge_this_run_did_not_make_cannot_be_cut(box, canvas):
    result = call(box, OP_DISCONNECT, src_id="box", src_port="toolbox",
                  dst_id="set", dst_port="toolbox")
    assert not result["ok"] and "not created by this run" in result["message"]


def test_what_this_run_connected_it_may_disconnect(box):
    call(box, OP_PLACE, class_name="TextNode")
    assert call(box, OP_CONNECT, src_id="new-1", src_port="text",
                dst_id="other", dst_port="text")["ok"]
    assert call(box, OP_DISCONNECT, src_id="new-1", src_port="text",
                dst_id="other", dst_port="text")["ok"]


def test_removing_a_node_forgets_its_edges():
    scope = RunScope()
    scope.placed("n1")
    scope.connected(("n1", "out", "n2", "in"))
    scope.forget_node("n1")
    assert not scope.owns_edge(("n1", "out", "n2", "in")), (
        "the edges went with the node; claiming them afterwards would be "
        "claiming ownership of something that is gone"
    )


# ── the self-modification guard (D73) ────────────────────────────────────


def test_the_agent_may_not_edit_itself(box, canvas):
    result = call(box, OP_CONNECT, src_id="new-1", src_port="out",
                  dst_id="agent", dst_port="toolset")
    assert not result["ok"] and "own execution path" in result["message"]
    assert not any(op == OP_CONNECT for op, _ in canvas.calls)


def test_the_agent_may_not_edit_what_feeds_it(box, canvas):
    """The whole chain, not one hop: box -> set -> agent."""
    result = call(box, OP_CONNECT, src_id="new-1", src_port="out",
                  dst_id="box", dst_port="anything")
    assert not result["ok"] and "upstream of this agent" in result["message"]


def test_an_unrelated_node_is_not_protected(box):
    call(box, OP_PLACE, class_name="TextNode")
    assert call(box, OP_CONNECT, src_id="new-1", src_port="out",
                dst_id="sibling", dst_port="in")["ok"]


def test_the_guard_walks_the_whole_chain():
    edges = [("a", "o", "b", "i"), ("b", "o", "c", "i"), ("c", "o", "me", "i"),
             ("x", "o", "y", "i")]
    assert upstream_of("me", edges) == {"a", "b", "c"}
    assert protected_nodes("me", edges) == {"me", "a", "b", "c"}
    assert check_self_modification("connect", "me", edges, "y") is None


def test_a_cycle_in_the_walk_terminates():
    edges = [("a", "o", "b", "i"), ("b", "o", "a", "i"), ("a", "o", "me", "i")]
    assert upstream_of("me", edges) == {"a", "b"}


def test_without_an_agent_id_nothing_is_protected():
    """A graph that cannot say which node is the agent protects nothing --
    so the binding always carries one, and this is the honest reading."""
    assert protected_nodes("", [("a", "o", "b", "i")]) == set()


# ── no canvas: refuse, never hang, never half-build (D36, D70) ───────────


def test_without_a_bound_canvas_every_tool_refuses():
    toolbox = ToolBox()
    attach_graph_tools(toolbox, sandbox=None, whitelist=("TextNode",))
    assert canvas_binding(toolbox) is None

    for op, args in ((OP_LIST, {}), (OP_DESCRIBE, {}),
                     (OP_PLACE, {"class_name": "TextNode"}),
                     (OP_CONNECT, {"src_id": "a", "src_port": "o",
                                   "dst_id": "b", "dst_port": "i"})):
        result = call(toolbox, op, **args)
        assert not result["ok"], f"{op} must refuse without a canvas"
        assert "Nothing was changed" in result["message"] or "cannot edit" in \
            result["message"]


def test_unbinding_takes_the_canvas_away(box, canvas):
    assert call(box, OP_DESCRIBE)["ok"]
    bind_canvas(box, None)
    assert not call(box, OP_DESCRIBE)["ok"], (
        "a seam left bound after its run points at a canvas nobody is "
        "driving, and the run scope would let the next run delete this "
        "one's nodes"
    )


# ── configuring a placed node (§22 q9) ───────────────────────────────────


def test_a_node_this_run_placed_can_be_configured(box, canvas):
    placed = call(box, OP_PLACE, class_name="TextNode")["result"]["id"]
    result = call(box, OP_SET_VALUE, id=placed, port="text", value="hello")
    assert result["ok"] and result["result"]["value"] == "hello"
    assert canvas.calls[-1][0] == OP_SET_VALUE


def test_a_node_the_user_placed_is_not_retuned(box, canvas):
    """The same boundary as removal, said in the verb that was tried."""
    result = call(box, OP_SET_VALUE, id="theirs", port="text", value="x")
    assert result["ok"] is False
    assert "not yours to change" in result["message"]
    assert all(op != OP_SET_VALUE for op, _ in canvas.calls), (
        "and nothing reached the canvas"
    )


def test_the_agent_may_not_configure_its_own_execution_path(box):
    """D73 covers values too: a node does not retune what is running it.

    The scope check would already refuse a node this run did not place,
    so the guard is exercised where it is the *only* thing standing in
    the way -- a run that somehow owns the agent's own node.
    """
    canvas_binding(box).scope.placed("agent")
    result = call(box, OP_SET_VALUE, id="agent", port="text", value="x")
    assert result["ok"] is False and "execution path" in result["message"]

    canvas_binding(box).scope.placed("box")     # upstream of the agent
    upstream = call(box, OP_SET_VALUE, id="box", port="text", value="x")
    assert upstream["ok"] is False
    assert "upstream" in upstream["message"], (
        "the whole chain, not one hop: the ToolBox that feeds the "
        "ToolSet that feeds the agent is exactly what it must not retune"
    )


def test_settings_are_readable_for_any_node(box, canvas):
    """Reading is not editing: an agent may look at what it did not place."""
    result = call(box, OP_SETTINGS, id="theirs")
    assert result["ok"] and result["result"]["settings"]


def test_a_headless_run_refuses_both_verbs(canvas):
    toolbox = ToolBox()
    attach_graph_tools(toolbox, sandbox=None, whitelist=("TextNode",))
    for op in (OP_SETTINGS, OP_SET_VALUE):
        result = call(toolbox, op, id="n", port="text", value="x")
        assert result["ok"] is False and "no canvas" in result["message"]


# -- what counts as a setting ---------------------------------------------


def _port(**kw):
    row = {"name": "text", "role": "INPUT", "datatype": "string",
           "connected": False, "exists": True}
    row.update(kw)
    return row


def test_a_value_port_is_settable():
    assert check_settable(_port(), "n1") is None


def test_a_display_or_internal_widget_is_not_a_setting():
    """INTERNAL is where the permission switches live (a ToolBox's
    'Plugin authoring'), so writing it would be an agent granting itself
    authority by ticking a box."""
    for role in ("DISPLAY", "INTERNAL", "OUTPUT"):
        refusal = check_settable(_port(role=role), "n1")
        assert refusal is not None and "cannot be written" in refusal.reason


def test_an_object_port_needs_a_connection():
    for datatype in ("gguf_model", "silk_toolset", "file_permissions",
                     "dirpath_list", "dict", "list"):
        refusal = check_settable(_port(datatype=datatype), "n1")
        assert refusal is not None and "connection" in refusal.reason, (
            "the authority-bearing ports are unwritable by construction, "
            "not by a rule someone has to remember"
        )


def test_a_wired_port_is_not_overwritten_behind_the_wire():
    refusal = check_settable(_port(connected=True), "n1")
    assert refusal is not None and "already fed by a connection" in refusal.reason


def test_a_port_that_does_not_exist_says_where_to_look():
    refusal = check_settable(_port(exists=False, name="nope"), "n1")
    assert refusal is not None and OP_SETTINGS in refusal.reason


# -- the value itself ------------------------------------------------------


@pytest.mark.parametrize("datatype,given,expected", [
    ("string", "x", "x"),
    ("string", 17, "17"),
    ("int", 3, 3),
    ("int", "4", 4),
    ("int", 5.0, 5),
    ("float", 1, 1.0),
    ("bool", True, True),
])
def test_a_value_arrives_as_the_widget_s_type(datatype, given, expected):
    value, refusal = coerce_value(datatype, given)
    assert refusal is None and value == expected and type(value) is type(expected)


@pytest.mark.parametrize("datatype,given", [
    ("int", 1.7),          # a silent truncation is a value nobody chose
    ("int", "seven"),
    ("int", True),
    ("bool", "yes"),
    ("bool", 1),
    ("float", "many"),
    ("string", False),
])
def test_a_value_the_widget_cannot_hold_is_refused_with_the_type(datatype,
                                                                 given):
    value, refusal = coerce_value(datatype, given)
    assert value is None and refusal is not None
    assert datatype in refusal.reason


def test_an_unsettable_type_never_coerces():
    value, refusal = coerce_value("gguf_model", "some/path")
    assert value is None and refusal is not None


def test_the_destructive_verbs_are_refused_when_nobody_can_be_asked():
    """D73 says the gate covers them; D81 is what makes that true.

    `remove_node` and `disconnect` are registered `requires_approval=True`,
    and until the floor existed that flag was read by nothing unless a tool
    policy happened to name them -- so in a graph with no approval hook
    configured they ran unasked. The refusal comes *before* the canvas
    check, which is the right order: approval is outermost.
    """
    toolbox = ToolBox()
    attach_graph_tools(toolbox, sandbox=None, whitelist=("TextNode",))

    for op, args in ((OP_REMOVE, {"id": "a"}),
                     (OP_DISCONNECT, {"src_id": "a", "src_port": "o",
                                      "dst_id": "b", "dst_port": "i"})):
        result = call(toolbox, op, **args)
        assert result["applied"] is False and result["approval_required"]
        assert "no way to ask" in result["message"]
