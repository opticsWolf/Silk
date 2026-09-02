# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Graph authoring: the agent places nodes and connects them (spec §18, D69).

This is the first Silk tool family whose effect is on *Weave itself*
rather than on files, a model, or a task store, and that changes what has
to be true before it runs. Three things carry the weight:

* **A whitelist that starts empty** (D71) -- the user ticks node classes
  on the ToolBox node, and nothing inside a run can widen that.
* **The main-thread seam** (D70) -- the canvas belongs to the Qt thread,
  the tool runs on a worker, and every failure to reach the main thread
  refuses rather than hanging or half-building.
* **One undoable command per call** (D72) -- so the human undoes the
  agent's edit with the gesture they undo their own.

Placement without inspection is blind, so two of the six are read-only:
an agent that cannot see the graph cannot place a node *relative* to it,
reuse what is already there, or know which ports are free.

Deliberately not here (v1): setting widget values, moving or resizing
nodes, saving or loading graph files, placing classes off the whitelist,
and anything touching another graph. Setting widget values is the obvious
next request and is left out on purpose -- it is how a placed node becomes
*configured*, which is every widget type at once and wants its own
decision.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

# Absolute import: this module is exec'd by the ToolLoader as
# ``dynamic_tools.graph_authoring`` (no parent package), so a ``..``
# relative import would be "beyond top-level".
from weave.plugins.silk.functions.graph_author import (
    OP_CONNECT, OP_DESCRIBE, OP_DISCONNECT, OP_LIST, OP_PLACE, OP_REMOVE,
    Refusal, RunScope, Whitelist, canvas_binding, check_self_modification,
)

if TYPE_CHECKING:
    from ..tool_box import ToolBox
    from .file_sandbox import FileToolSandbox


# ── schemas ──────────────────────────────────────────────────────────────

class NoArgs(BaseModel):
    pass


class PlaceNodeArgs(BaseModel):
    class_name: str = Field(
        ..., min_length=1,
        description="Class name of the node to place (from "
                    "list_placeable_nodes).",
    )
    x: float = Field(0.0, description="Canvas x position.")
    y: float = Field(0.0, description="Canvas y position.")
    title: str = Field("", description="Optional title for the new node.")


class ConnectArgs(BaseModel):
    src_id: str = Field(..., description="Id of the source node.")
    src_port: str = Field(..., description="Output port name on the source.")
    dst_id: str = Field(..., description="Id of the destination node.")
    dst_port: str = Field(..., description="Input port name on the destination.")


class DisconnectArgs(ConnectArgs):
    pass


class RemoveNodeArgs(BaseModel):
    id: str = Field(..., description="Id of the node to remove.")


class GraphResult(BaseModel):
    ok: bool = Field(..., description="Whether the operation happened.")
    op: str = Field("", description="Which operation this was.")
    result: dict = Field(default_factory=dict,
                         description="Operation-specific payload.")
    message: str = Field("", description="Why it was refused, if it was.")


# ── helpers ──────────────────────────────────────────────────────────────

def _refused(refusal: Refusal) -> GraphResult:
    return GraphResult(ok=False, op=refusal.op, message=refusal.reason,
                       result={"subject": refusal.subject})


def _no_canvas(op: str) -> GraphResult:
    return GraphResult(ok=False, op=op, message=(
        "This run cannot edit the graph: there is no canvas bound to it "
        "(a headless evaluation, a subagent, or a graph that closed). "
        "Nothing was changed."))


def attach_graph_tools(toolbox: "ToolBox", sandbox: "FileToolSandbox",
                       whitelist: Any = ()) -> None:
    """Mount the six graph-authoring tools against *whitelist*.

    The whitelist is fixed when the ToolBox is built -- it travels in the
    recipe, so a derived ToolSet replays it and may narrow it, never widen
    it (I6, D71). The *canvas* is bound per run by the Agent node, so the
    same box works in a headless evaluation: the tools are there, and they
    refuse.
    """
    allowed = whitelist if isinstance(whitelist, Whitelist) else Whitelist(
        whitelist or ())
    toolbox._graph_whitelist = allowed  # type: ignore[attr-defined]

    empty_note = (
        "\n- NOTE: no node classes are whitelisted, so placement will be "
        "refused. Ask the user to tick classes on the ToolBox node."
        if not len(allowed) else ""
    )

    def _bound() -> tuple:
        binding = canvas_binding(toolbox)
        if binding is None or binding.seam is None:
            return None, None, ""
        return binding.seam, binding.scope, binding.agent_uid

    def _graph_edges(seam: Any) -> list:
        """The current edges, for the self-modification walk (D73)."""
        answer = seam.call(OP_DESCRIBE)
        if not answer.ok:
            return []
        return list((answer.value or {}).get("edges") or [])

    # ── reads ────────────────────────────────────────────────────────────

    @toolbox.register(
        name=OP_LIST,
        tags=("graph", "read"), category="graph", risk="low",
        description=(
            "List the node classes you are allowed to place on the canvas, "
            "with their descriptions and ports. Placement is refused for "
            "anything not on this list."
        ),
        args_model=NoArgs,
        procedure=(
            "Read the placement whitelist.\n"
            "- Returns {class_name, display_name, description, category, "
            "tags, inputs[], outputs[]} per class.\n"
            "- Port entries carry the datatype: two ports connect only if "
            "the type system allows it, so read them before connecting."
            + empty_note
        ),
    )
    def _list_placeable(db_pool: Any, user_session: dict) -> GraphResult:
        seam, _scope, _agent = _bound()
        if seam is None:
            return _no_canvas(OP_LIST)
        answer = seam.call(OP_LIST, allowed=sorted(allowed.names))
        if not answer.ok:
            return GraphResult(ok=False, op=OP_LIST,
                               message=answer.failure_text())
        return GraphResult(ok=True, op=OP_LIST, result=answer.value or {})

    @toolbox.register(
        name=OP_DESCRIBE,
        tags=("graph", "read"), category="graph", risk="low",
        description=(
            "Describe the current graph: every node (id, class, title, "
            "position, ports and which are connected) and every edge as "
            "(src_id, src_port, dst_id, dst_port)."
        ),
        args_model=NoArgs,
        procedure=(
            "Read the canvas before changing it.\n"
            "- Place nodes *relative* to what is there; reuse a node "
            "rather than duplicating it.\n"
            "- 'connected' on an input port means it is taken.\n"
            "- 'agent_id' is you: neither you nor anything upstream of "
            "you may be edited."
        ),
    )
    def _describe_graph(db_pool: Any, user_session: dict) -> GraphResult:
        seam, _scope, _agent = _bound()
        if seam is None:
            return _no_canvas(OP_DESCRIBE)
        answer = seam.call(OP_DESCRIBE)
        if not answer.ok:
            return GraphResult(ok=False, op=OP_DESCRIBE,
                               message=answer.failure_text())
        return GraphResult(ok=True, op=OP_DESCRIBE, result=answer.value or {})

    # ── placement ────────────────────────────────────────────────────────

    @toolbox.register(
        name=OP_PLACE,
        tags=("graph", "write"), category="graph", risk="medium",
        description=(
            "Place a whitelisted node on the canvas at a position. Returns "
            "the new node's id, which you need in order to connect it."
        ),
        args_model=PlaceNodeArgs,
        procedure=(
            "Add one node to the graph.\n"
            "- class_name must come from list_placeable_nodes.\n"
            "- Check describe_graph first so the position does not land on "
            "top of an existing node.\n"
            "- One call is one undo step for the user."
            + empty_note
        ),
    )
    def _place_node(db_pool: Any, user_session: dict, class_name: str,
                    x: float = 0.0, y: float = 0.0,
                    title: str = "") -> GraphResult:
        refusal = allowed.check(class_name)
        if refusal is not None:
            return _refused(refusal)
        seam, scope, _agent = _bound()
        if seam is None:
            return _no_canvas(OP_PLACE)
        answer = seam.call(OP_PLACE, class_name=class_name,
                           position=[float(x), float(y)], title=title)
        if not answer.ok:
            return GraphResult(ok=False, op=OP_PLACE,
                               message=answer.failure_text())
        value = answer.value or {}
        if scope is not None and value.get("id"):
            scope.placed(value["id"])
        return GraphResult(ok=True, op=OP_PLACE, result=value)

    @toolbox.register(
        name=OP_CONNECT,
        tags=("graph", "write"), category="graph", risk="medium",
        description=(
            "Connect one node's output port to another node's input port. "
            "Refused with a reason if the port types do not connect."
        ),
        args_model=ConnectArgs,
        procedure=(
            "Add one edge.\n"
            "- Port names and datatypes come from describe_graph / "
            "list_placeable_nodes.\n"
            "- A refusal names the two datatypes: pick a different port "
            "or an adapter node rather than retrying the same edge.\n"
            "- You may not connect anything to yourself or to a node "
            "upstream of you."
        ),
    )
    def _connect(db_pool: Any, user_session: dict, src_id: str, src_port: str,
                 dst_id: str, dst_port: str) -> GraphResult:
        seam, scope, agent_uid = _bound()
        if seam is None:
            return _no_canvas(OP_CONNECT)
        refusal = check_self_modification(
            OP_CONNECT, agent_uid, _graph_edges(seam), dst_id)
        if refusal is not None:
            return _refused(refusal)
        answer = seam.call(OP_CONNECT, src_id=src_id, src_port=src_port,
                           dst_id=dst_id, dst_port=dst_port)
        if not answer.ok:
            return GraphResult(ok=False, op=OP_CONNECT,
                               message=answer.failure_text())
        if scope is not None:
            scope.connected((src_id, src_port, dst_id, dst_port))
        return GraphResult(ok=True, op=OP_CONNECT, result=answer.value or {})

    # ── taking apart what this run built ─────────────────────────────────

    @toolbox.register(
        name=OP_DISCONNECT,
        tags=("graph", "write"), category="graph", risk="high",
        requires_approval=True,
        description=(
            "Remove an edge **that this run created**. Edges the user made "
            "are not yours to remove."
        ),
        args_model=DisconnectArgs,
        procedure=(
            "Undo one of your own connections.\n"
            "- Only edges created earlier in this run can be removed.\n"
            "- The user is asked to approve before it happens."
        ),
    )
    def _disconnect(db_pool: Any, user_session: dict, src_id: str,
                    src_port: str, dst_id: str, dst_port: str) -> GraphResult:
        seam, scope, agent_uid = _bound()
        if seam is None:
            return _no_canvas(OP_DISCONNECT)
        edge = (src_id, src_port, dst_id, dst_port)
        refusal = (scope or RunScope()).check_edge(edge)
        if refusal is not None:
            return _refused(refusal)
        refusal = check_self_modification(
            OP_DISCONNECT, agent_uid, _graph_edges(seam), src_id, dst_id)
        if refusal is not None:
            return _refused(refusal)
        answer = seam.call(OP_DISCONNECT, src_id=src_id, src_port=src_port,
                           dst_id=dst_id, dst_port=dst_port)
        if not answer.ok:
            return GraphResult(ok=False, op=OP_DISCONNECT,
                               message=answer.failure_text())
        if scope is not None:
            scope.forget_edge(edge)
        return GraphResult(ok=True, op=OP_DISCONNECT, result=answer.value or {})

    @toolbox.register(
        name=OP_REMOVE,
        tags=("graph", "write"), category="graph", risk="high",
        requires_approval=True,
        description=(
            "Remove a node **that this run placed**. Nodes the user placed "
            "are not yours to remove."
        ),
        args_model=RemoveNodeArgs,
        procedure=(
            "Undo one of your own placements.\n"
            "- Only nodes placed earlier in this run can be removed.\n"
            "- Its connections go with it, in the same undo step.\n"
            "- The user is asked to approve before it happens."
        ),
    )
    def _remove_node(db_pool: Any, user_session: dict, id: str) -> GraphResult:  # noqa: A002
        seam, scope, agent_uid = _bound()
        if seam is None:
            return _no_canvas(OP_REMOVE)
        refusal = (scope or RunScope()).check_node(id)
        if refusal is not None:
            return _refused(refusal)
        refusal = check_self_modification(
            OP_REMOVE, agent_uid, _graph_edges(seam), id)
        if refusal is not None:
            return _refused(refusal)
        answer = seam.call(OP_REMOVE, id=id)
        if not answer.ok:
            return GraphResult(ok=False, op=OP_REMOVE,
                               message=answer.failure_text())
        if scope is not None:
            scope.forget_node(id)
        return GraphResult(ok=True, op=OP_REMOVE, result=answer.value or {})
