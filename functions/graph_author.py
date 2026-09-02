# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The rules an agent's graph edits obey (spec §18, D69, D71, D73).

Qt-free on purpose: everything here is a decision *about* a graph edit --
is this class allowed, did this run create that node, would this rewire
the agent's own inputs -- and none of it needs a canvas to answer. The
canvas half (undo commands on the main thread, D70/D72) lives in the node
layer; this module is what it consults, and what the tests can exercise
without a scene.

Three rules, in the order they are checked:

1. **Default-deny by class** (D71). The whitelist starts empty, so an
   agent that was given the tools and nothing else can do nothing with
   them. There is no "allow all": selecting everything is possible and
   must be a deliberate act. Same reasoning as I6 -- the safe state is
   the one you get by doing nothing.
2. **Destructive calls are scoped to the run** (D73). `remove_node` and
   `disconnect` touch only what *this* agent placed in *this* run. An
   agent may clean up after itself; it may not prune the user's graph.
3. **No self-modification** (D73). Nothing may touch the Agent node, its
   ToolBox / ToolSet / Role / model chain, or anything upstream of it.
   The graph that is running the agent is not material the agent edits
   mid-run: the evaluation model gives no coherent meaning to rewiring a
   node's own inputs while it sits inside `compute()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

#: The six tools of D69, named once so the Role's selector, the tests and
#: the tool registrations cannot drift apart.
OP_LIST = "list_placeable_nodes"
OP_DESCRIBE = "describe_graph"
OP_PLACE = "place_node"
OP_CONNECT = "connect"
OP_DISCONNECT = "disconnect"
OP_REMOVE = "remove_node"

READ_OPS = (OP_LIST, OP_DESCRIBE)
WRITE_OPS = (OP_PLACE, OP_CONNECT)
DESTRUCTIVE_OPS = (OP_DISCONNECT, OP_REMOVE)
OPS = READ_OPS + WRITE_OPS + DESTRUCTIVE_OPS


@dataclass(frozen=True)
class Refusal:
    """Why an edit will not happen. Carries a reason the model can act on.

    A refusal is a *tool result*, never an exception: "you may not place
    that class" is information the agent can use (ask the user to
    whitelist it, pick another node), and a traceback is not.
    """

    reason: str
    op: str = ""
    subject: str = ""

    def as_dict(self) -> dict:
        return {"ok": False, "op": self.op, "subject": self.subject,
                "error": self.reason}


class Whitelist:
    """Which node classes this agent may place. Empty means none (D71).

    Entries are **class names**, resolved against `NODE_REGISTRY` when the
    ToolBox evaluates, so a whitelisted class that is no longer registered
    is reported in the node rather than at agent run time.

    It narrows like everything else (I6): :meth:`narrowed` may remove
    entries and never add, so a ToolSet or Role downstream of a ToolBox
    cannot widen what the box allowed.
    """

    def __init__(self, names: Iterable[str] = ()) -> None:
        self._names = frozenset(str(n).strip() for n in names if str(n).strip())

    def __contains__(self, name: object) -> bool:
        return str(name) in self._names

    def __len__(self) -> int:
        return len(self._names)

    def __iter__(self):
        return iter(sorted(self._names))

    @property
    def names(self) -> frozenset[str]:
        return self._names

    def narrowed(self, names: Iterable[str]) -> "Whitelist":
        """The intersection -- a narrowing, never a widening (I6)."""
        return Whitelist(self._names & frozenset(
            str(n).strip() for n in names if str(n).strip()))

    def missing(self, registered: Iterable[str]) -> list[str]:
        """Whitelisted classes that are not registered any more.

        Surfaced by the ToolBox node so a renamed or unloaded plugin shows
        up as a visible problem in the graph, not as a refusal an agent
        hits halfway through building something.
        """
        return sorted(self._names - frozenset(registered))

    def check(self, class_name: str) -> Optional[Refusal]:
        if not self._names:
            return Refusal(
                "No node classes are whitelisted for placement. The user "
                "must tick them on the ToolBox node; there is no way to "
                "grant this from inside the run.",
                op=OP_PLACE, subject=class_name,
            )
        if class_name not in self._names:
            return Refusal(
                f"'{class_name}' is not on the placement whitelist. "
                f"Allowed: {', '.join(sorted(self._names))}.",
                op=OP_PLACE, subject=class_name,
            )
        return None


@dataclass
class RunScope:
    """What this run built, and therefore what it may take apart (D73).

    The record is per run and per agent, not per graph: two agents
    building side by side each undo only their own work, and neither can
    reach a node the user placed.
    """

    nodes: set = field(default_factory=set)
    edges: set = field(default_factory=set)

    def placed(self, node_uid: str) -> None:
        self.nodes.add(str(node_uid))

    def connected(self, edge: tuple) -> None:
        self.edges.add(tuple(str(part) for part in edge))

    def forget_node(self, node_uid: str) -> None:
        self.nodes.discard(str(node_uid))
        self.edges = {e for e in self.edges
                      if e[0] != str(node_uid) and e[2] != str(node_uid)}

    def forget_edge(self, edge: tuple) -> None:
        self.edges.discard(tuple(str(part) for part in edge))

    def owns_node(self, node_uid: str) -> bool:
        return str(node_uid) in self.nodes

    def owns_edge(self, edge: tuple) -> bool:
        return tuple(str(part) for part in edge) in self.edges

    def check_node(self, node_uid: str) -> Optional[Refusal]:
        if not self.owns_node(node_uid):
            return Refusal(
                f"Node '{node_uid}' was not created by this run, so it "
                f"cannot be removed. You may only take apart what you "
                f"built here.",
                op=OP_REMOVE, subject=str(node_uid),
            )
        return None

    def check_edge(self, edge: tuple) -> Optional[Refusal]:
        if not self.owns_edge(edge):
            return Refusal(
                "That connection was not created by this run, so it cannot "
                "be removed. You may only take apart what you built here.",
                op=OP_DISCONNECT, subject=" -> ".join(str(p) for p in edge),
            )
        return None


def upstream_of(node_uid: str, edges: Iterable[tuple]) -> set[str]:
    """Every node that feeds *node_uid*, transitively.

    A plain reverse walk over ``(src_uid, src_port, dst_uid, dst_port)``
    tuples -- the same shape the undo commands speak, and the same shape
    `describe_graph` returns, so the guard reasons about exactly what the
    model was shown.
    """
    incoming: dict[str, set[str]] = {}
    for src_uid, _src_port, dst_uid, _dst_port in edges:
        incoming.setdefault(str(dst_uid), set()).add(str(src_uid))

    seen: set[str] = set()
    frontier = [str(node_uid)]
    while frontier:
        current = frontier.pop()
        for parent in incoming.get(current, ()):  # noqa: SIM118
            if parent not in seen:
                seen.add(parent)
                frontier.append(parent)
    return seen


def protected_nodes(agent_uid: str, edges: Iterable[tuple]) -> set[str]:
    """The agent's own execution path: itself, and everything upstream.

    "Upstream" is the whole chain, not one hop, because the ToolBox that
    feeds the ToolSet that feeds the Role that feeds the agent is exactly
    the sequence an agent would otherwise be able to rewire while running
    on it.
    """
    if not agent_uid:
        return set()
    return {str(agent_uid)} | upstream_of(agent_uid, edges)


def check_self_modification(op: str, agent_uid: str, edges: Iterable[tuple],
                            *targets: str) -> Optional[Refusal]:
    """Refuse any edit touching the graph that is running this agent (D73).

    Cheap: one reverse walk from the Agent node at request time. The
    objection is the same as D51's -- a node does not get to rewrite the
    thing evaluating it mid-evaluation, and there is no coherent meaning
    to give the result if it did.
    """
    guarded = protected_nodes(agent_uid, edges)
    for target in targets:
        if target and str(target) in guarded:
            which = ("this agent" if str(target) == str(agent_uid)
                     else "a node upstream of this agent")
            return Refusal(
                f"Refused: '{target}' is {which}. An agent may not modify "
                f"its own execution path -- the graph running it is not "
                f"material it edits mid-run.",
                op=op, subject=str(target),
            )
    return None


def describe_class(node_cls: Any) -> dict:
    """One whitelist entry as the model needs to read it (D69).

    Everything here is metadata that already exists for the node UI --
    `node_name`, `node_description`, `node_tags`, port `datatype` and
    `port_description`. Nothing had to be authored for the model, which is
    the point: a plugin that documents itself for humans is placeable.
    """
    from weave.registry.metadata import get_description, get_display_name, get_tags

    entry = {
        "class_name": getattr(node_cls, "__name__", ""),
        "display_name": get_display_name(node_cls),
        "description": (get_description(node_cls) or "").strip(),
        "category": str(getattr(node_cls, "node_category", "") or ""),
        "tags": list(get_tags(node_cls) or ()),
        "inputs": [],
        "outputs": [],
    }
    for side in ("inputs", "outputs"):
        for spec in _port_specs(node_cls, side):
            entry[side].append(spec)
    return entry


def _port_specs(node_cls: Any, side: str) -> list[dict]:
    """Port metadata from a class, without instantiating it if possible.

    Weave declares ports in `__init__`, so a class-level declaration is
    not always there; when it is not, the caller (which is on the main
    thread and may build widgets) passes an instance instead.
    """
    declared = getattr(node_cls, f"declared_{side}", None)
    if not declared:
        return []
    specs = []
    for port in declared:
        if isinstance(port, dict):
            specs.append({"name": port.get("name", ""),
                          "datatype": port.get("datatype", ""),
                          "description": port.get("description", "")})
        else:
            specs.append({"name": getattr(port, "name", ""),
                          "datatype": getattr(port, "datatype", ""),
                          "description": getattr(port, "port_description", "")})
    return specs


def describe_instance(node: Any) -> dict:
    """One live node as `describe_graph` returns it (D69).

    Position and connectedness are in here because placement is
    *relative*: an agent that cannot see where things are cannot put a
    node beside one, and an agent that cannot see which ports are taken
    will try to connect to them.
    """
    pos = node.pos() if hasattr(node, "pos") else None
    return {
        "id": str(getattr(node, "unique_id", "")),
        "class_name": type(node).__name__,
        "title": str(getattr(node, "title", "") or ""),
        "position": [float(pos.x()), float(pos.y())] if pos is not None else None,
        "inputs": [_live_port(p) for p in getattr(node, "inputs", []) or []],
        "outputs": [_live_port(p) for p in getattr(node, "outputs", []) or []],
    }


def _live_port(port: Any) -> dict:
    return {
        "name": str(getattr(port, "name", "")),
        "datatype": str(getattr(port, "datatype", "")),
        "description": str(getattr(port, "port_description", "") or ""),
        "connected": bool(getattr(port, "connected_traces", None)),
    }


# ── binding the run's canvas seam (the D48/D51 pattern) ──────────────────

#: Where a run's canvas seam and scope live on the ToolBox. The same
#: shape the approval gate uses for the decision seam: the box is built
#: once by the ToolBox node, the seam exists only for the length of a
#: run, and the tools read it at call time.
_SEAM_ATTR = "_silk_canvas_seam"
_SCOPE_ATTR = "_silk_canvas_scope"


@dataclass
class CanvasBinding:
    """What a run lends the graph tools: a way to the canvas, and a memory."""

    seam: Any = None
    scope: RunScope = field(default_factory=RunScope)
    agent_uid: str = ""


def bind_canvas(toolbox: Any, binding: Optional[CanvasBinding]) -> None:
    """Point the graph tools at this run's canvas (or at nothing).

    The Agent node calls this on both edges of a run. Unbinding matters
    as much as binding: a seam left behind after its run points at a
    canvas nobody is driving, and the run scope of a finished run would
    let the *next* run delete the previous one's nodes.
    """
    try:
        setattr(toolbox, _SEAM_ATTR, binding)
        setattr(toolbox, _SCOPE_ATTR, None if binding is None else binding.scope)
    except AttributeError:      # a toolbox that forbids attributes
        pass


def canvas_binding(toolbox: Any) -> Optional[CanvasBinding]:
    return getattr(toolbox, _SEAM_ATTR, None)
