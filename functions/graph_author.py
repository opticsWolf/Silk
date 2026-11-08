# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

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
2. **Editing a node needs access to it** (D73, widened). Each node has an
   agent access level -- none, connect, configure or full -- that the user
   sets from its right-click menu (**Agent access**). Left at its default, a
   node the agent placed is the placing agent's in full, in this run and
   every later one, and a node the user placed is nobody's. Reading a node
   and wiring *from* its outputs need nothing; wiring into or out of its
   inputs needs connect, `set_node_value` needs configure, `remove_node`
   needs full.
3. **No self-modification** (D73). Nothing may touch the Agent node, its
   ToolBox / ToolSet / Role / model chain, or anything upstream of it.
   The graph that is running the agent is not material the agent edits
   mid-run: the evaluation model gives no coherent meaning to rewiring a
   node's own inputs while it sits inside `compute()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

#: The tools of D69 (six) and §22 q9 (two more), named once so the Role's
#: selector, the tests and the tool registrations cannot drift apart.
OP_LIST = "list_placeable_nodes"
OP_DESCRIBE = "describe_graph"
OP_PLACE = "place_node"
OP_CONNECT = "connect"
OP_DISCONNECT = "disconnect"
OP_REMOVE = "remove_node"

#: The two of §22 q9: a placed node that cannot be configured is a
#: skeleton a human has to finish.
OP_SETTINGS = "list_node_settings"
OP_SET_VALUE = "set_node_value"

#: Running the graph and reading what it computed (Weave's control API,
#: P3): on a canvas a result shows in a widget, but an agent working a
#: headless graph has nothing to look at but the values.
OP_EVALUATE = "evaluate_graph"
OP_READ = "read_output"

READ_OPS = (OP_LIST, OP_DESCRIBE, OP_SETTINGS, OP_READ)
WRITE_OPS = (OP_PLACE, OP_CONNECT, OP_SET_VALUE)
DESTRUCTIVE_OPS = (OP_DISCONNECT, OP_REMOVE)
RUN_OPS = (OP_EVALUATE,)
OPS = READ_OPS + WRITE_OPS + DESTRUCTIVE_OPS + RUN_OPS

#: Widget roles a value may be written to on any node. DISPLAY is computed
#: output and never settable. INTERNAL is node-local state: it is where
#: the permission-bearing switches live (a ToolBox's "Plugin authoring",
#: for one), and also where most of Weave's value nodes keep the very text
#: box they exist for -- String Input, Text Box Input, the Int / Float /
#: Bool inputs. So INTERNAL is settable on a node that grants no authority
#: (:func:`grants_authority`) and on no other.
SETTABLE_ROLES = ("INPUT", "BIDIRECTIONAL")
ROLE_INTERNAL = "INTERNAL"

#: Output datatypes that carry authority: a node emitting one of these
#: hands a downstream node tools, permissions, roots, a model or a plan.
#: Its node-local switches are how that authority is set, so an agent that
#: could tick them could grant itself what it was not given -- place a
#: ToolBox, enable Plugin authoring, wire it to an Agent of its own. Read
#: off the node's own ports, so a new value node is configurable with no
#: extra code and a new authority node is locked as soon as it says what it
#: emits. ``agent_message`` is not one: a message is content, not a grant,
#: and the only node that emits one (Silk Agent) is locked as a Silk node.
AUTHORITY_TYPES = frozenset({
    "silk_toolbox", "silk_toolset", "silk_role", "silk_classifier",
    "silk_plan", "silk_workers", "model_handle",
    "file_permissions", "dirpath_list", "mcp_servers", "toolchains",
    "okf_graph",
})

#: This plugin's own package, however it was imported (``weave.plugins.silk``
#: in the app, ``silk`` in its tests). Every Silk node configures an agent,
#: its tools or its view of them, so none of their INTERNAL state is the
#: agent's to set, whatever it happens to output.
_SILK_PACKAGE = __name__.rsplit(".", 2)[0]


def grants_authority(output_types: Iterable[str], module: str = "") -> bool:
    """Whether a node's node-local state is authority the agent must not set.

    True for a Silk node and for any node that emits an
    :data:`AUTHORITY_TYPES` value. A caller that cannot tell (no module,
    no ports) should pass nothing and treat the row as authority-bearing:
    :func:`check_settable` fails closed on a row without the flag.
    """
    module = str(module or "")
    if module == _SILK_PACKAGE or module.startswith(_SILK_PACKAGE + "."):
        return True
    return any(str(t).lower() in AUTHORITY_TYPES for t in output_types)

#: Datatypes a *value* can carry. Everything else on a node -- a model, a
#: toolset, a permissions object, a list of roots -- is supplied by a
#: connection, and the refusal says so rather than guessing at a
#: conversion. Narrow on purpose: it keeps the authority-bearing ports
#: (`file_permissions`, `dirpath_list`) unwritable by construction rather
#: than by a rule somebody has to remember.
SETTABLE_TYPES = {
    "string": str, "str": str, "text": str, "filepath": str, "dirpath": str,
    "choice": str, "enum": str,
    "int": int, "integer": int,
    "float": float, "number": float,
    "bool": bool, "boolean": bool,
}


#: Agent access to one node, weakest first; each level includes the ones
#: before it. Reading a node and wiring from its outputs need none.
ACCESS_NONE = "none"
ACCESS_CONNECT = "connect"
ACCESS_CONFIGURE = "configure"
ACCESS_FULL = "full"
ACCESS_LEVELS = (ACCESS_NONE, ACCESS_CONNECT, ACCESS_CONFIGURE, ACCESS_FULL)

#: What each edit needs on the node it edits (the destination, for a wire).
ACCESS_NEEDED = {
    OP_CONNECT: ACCESS_CONNECT,
    OP_DISCONNECT: ACCESS_CONNECT,
    OP_SET_VALUE: ACCESS_CONFIGURE,
    OP_REMOVE: ACCESS_FULL,
}

_ACCESS_WORDS = {
    ACCESS_CONNECT: "wire into its inputs",
    ACCESS_CONFIGURE: "change its settings",
    ACCESS_FULL: "remove it",
}

#: The key Silk's marks live under in a node's plugin data
#: (``Node.plugin_data``): ``owner`` is the uid of the Agent node that placed
#: it, ``access`` a level the user chose. Both are saved with the graph.
PLUGIN_KEY = "silk"
MARK_OWNER = "owner"
MARK_ACCESS = "access"


def access_marks(node: Any) -> dict:
    """Silk's marks on a live node: ``{"owner": ..., "access": ...}``, or {}."""
    reader = getattr(node, "plugin_data", None)
    if not callable(reader):
        return {}
    try:
        marks = reader(PLUGIN_KEY)
    except Exception:  # noqa: BLE001 - a node mid-teardown
        return {}
    return dict(marks) if isinstance(marks, dict) else {}


def default_access(marks: dict, agent_uid: str) -> str:
    """The level a node has when the user has not chosen one: full for the
    agent that placed it, none for everyone else."""
    owner = str(marks.get(MARK_OWNER) or "")
    return ACCESS_FULL if owner and owner == str(agent_uid) else ACCESS_NONE


def effective_access(marks: dict, agent_uid: str) -> str:
    """What *agent_uid* may do to a node with *marks*. A level the user
    chose applies to every agent, the placing one included."""
    chosen = str(marks.get(MARK_ACCESS) or "")
    if chosen in ACCESS_LEVELS:
        return chosen
    return default_access(marks, agent_uid)


def check_access(op: str, node_uid: str, level: str) -> Optional[Refusal]:
    """Refuse *op* on a node the agent has *level* access to, if too low."""
    need = ACCESS_NEEDED.get(op)
    if need is None:
        return None
    have = level if level in ACCESS_LEVELS else ACCESS_NONE
    if ACCESS_LEVELS.index(have) >= ACCESS_LEVELS.index(need):
        return None
    return Refusal(
        f"You may not {_ACCESS_WORDS[need]}: node '{node_uid}' gives you "
        f"'{have}' access and this needs '{need}'. Nodes you placed are "
        f"yours; any other node's access is set by the user (right-click "
        f"the node, Agent access). Place your own node, or ask the user.",
        op=op, subject=str(node_uid))


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
    """What this run built: the fallback for a seam that keeps no marks.

    A canvas marks the nodes an agent places (:func:`access_marks`), so
    there the marks decide and outlive the run. A seam whose nodes carry
    no marks (the control API) has only this record, so there a node is
    the agent's in full for the run that placed it and nobody's after.
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


def node_access(nodes: Iterable[dict], scope: Optional[RunScope],
                node_uid: str) -> str:
    """The agent's access to *node_uid*, from what `describe_graph` said.

    A seam that knows the marks reports ``access`` on each node and that
    is the answer, even for a node this run placed (the user may have
    taken it back). Otherwise the run's own record: placed here is full,
    anything else none.
    """
    for entry in nodes:
        if str(entry.get("id", "")) == str(node_uid) and "access" in entry:
            return str(entry["access"])
    owned = scope is not None and scope.owns_node(node_uid)
    return ACCESS_FULL if owned else ACCESS_NONE


def check_settable(port: dict, node_uid: str) -> Optional[Refusal]:
    """Whether one port of a live node may be written (§22 q9).

    Four ways a port is not a setting, and each gets its own sentence
    because "no" without a reason sends a model round the same loop:
    it does not exist, it is not a value port, it carries something only
    a wire can supply, or a wire is already supplying it.

    An INTERNAL widget is a value port only when the row says
    ``authority=False`` (see :func:`grants_authority`); a row without the
    flag is treated as authority-bearing, so a resolver that forgets to
    say fails closed.
    """
    name = str(port.get("name", ""))
    if not port.get("exists", True):
        return Refusal(
            f"Node '{node_uid}' has no setting called '{name}'. Call "
            f"{OP_SETTINGS} to see what it does have.",
            op=OP_SET_VALUE, subject=name)

    role = str(port.get("role", "")).upper()
    if role == ROLE_INTERNAL and port.get("authority", True):
        return Refusal(
            f"'{name}' is internal state on a node that grants authority "
            f"(tools, permissions, roots, a model or a plan). Setting it "
            f"would be granting that authority, so it cannot be written; "
            f"ask the user to set it.",
            op=OP_SET_VALUE, subject=name)
    if role not in SETTABLE_ROLES and role != ROLE_INTERNAL:
        return Refusal(
            f"'{name}' is a {role.lower() or 'display'} widget, not a "
            f"setting: it shows what the node computed or holds state the "
            f"node manages itself. It cannot be written.",
            op=OP_SET_VALUE, subject=name)

    datatype = str(port.get("datatype", "")).lower()
    if datatype not in SETTABLE_TYPES:
        return Refusal(
            f"'{name}' carries '{datatype}', which comes from a "
            f"connection rather than a typed-in value. Connect a node "
            f"that produces it.",
            op=OP_SET_VALUE, subject=name)

    if port.get("connected"):
        return Refusal(
            f"'{name}' is already fed by a connection, and the upstream "
            f"value wins over the widget. Disconnect it first, or change "
            f"the node that feeds it.",
            op=OP_SET_VALUE, subject=name)
    return None


def coerce_value(datatype: str, value: Any) -> tuple:
    """(value, refusal) -- the model's JSON as the widget's type.

    Deliberately strict in one direction only: `1` is a fine float and
    `"3"` is a fine int, but `"yes"` is not a bool and `1.7` is not an
    int. A silent truncation is a value the agent believes it set and the
    user never chose.
    """
    want = SETTABLE_TYPES.get(str(datatype).lower())
    if want is None:
        return None, Refusal(f"'{datatype}' is not a settable type.",
                             op=OP_SET_VALUE, subject=str(datatype))
    try:
        if want is bool:
            if isinstance(value, bool):
                return value, None
            raise TypeError("not a bool")
        if want is int:
            if isinstance(value, bool):
                raise TypeError("a bool is not a number")
            if isinstance(value, float) and value != int(value):
                raise TypeError("not a whole number")
            return int(value), None
        if want is float:
            if isinstance(value, bool):
                raise TypeError("a bool is not a number")
            return float(value), None
        if isinstance(value, bool):
            raise TypeError("a bool is not text")
        return str(value), None
    except (TypeError, ValueError) as exc:
        return None, Refusal(
            f"Cannot use {value!r} as a {datatype}: {exc}. Send a "
            f"{want.__name__} instead.",
            op=OP_SET_VALUE, subject=str(datatype))


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

    entry: dict[str, Any] = {
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


def node_title(node: Any) -> str:
    """What the canvas *shows* for this node.

    A Weave node has no `title` attribute: the constructor takes one, the
    header renders `node.name` when there is one and its own text
    otherwise (`weave/node/components.py`), and `NodeTitleCommand` writes
    both. Reading `node.title` -- which is what graph authoring did --
    returned an attribute nobody sets, so every described node was
    titleless unless something had assigned that attribute by hand.
    """
    name = getattr(node, "name", None)
    if callable(name):
        try:
            name = name()
        except Exception:  # noqa: BLE001 - a node mid-teardown
            name = None
    if name:
        return str(name)
    header = getattr(node, "header", None)
    label = getattr(header, "_title", None)
    for reader in ("toolTip", "toPlainText"):
        fn = getattr(label, reader, None)
        if callable(fn):
            try:
                text = fn()
            except Exception:  # noqa: BLE001
                text = ""
            if text:
                return str(text)
    return str(getattr(node, "title", "") or "")


def set_node_title(node: Any, title: str) -> bool:
    """Rename a node the way the canvas does, or report that it did not.

    Same order as `NodeTitleCommand._apply_title`: the model first
    (`set_name`, else `name`), then the header text, then the layout that
    reads it. Called *before* the placement snapshot is captured, so undo
    restores the title with the node rather than losing it.
    """
    applied = False
    setter = getattr(node, "set_name", None)
    if callable(setter):
        try:
            setter(title)
            applied = True
        except Exception:  # noqa: BLE001
            applied = False
    if not applied and hasattr(node, "name"):
        try:
            node.name = title
            applied = True
        except (AttributeError, TypeError):
            applied = False   # read-only, or a method by that name

    label = getattr(getattr(node, "header", None), "_title", None)
    if label is not None:
        try:
            label.setToolTip(title)
            label.setPlainText(title)
            header = node.header
            recalc = getattr(header, "_recalculate_layout", None)
            if callable(recalc):
                recalc()
            header.update()
            applied = True
        except Exception:  # noqa: BLE001 - a node not yet on a scene
            pass
    return applied


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
        "title": node_title(node),
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
