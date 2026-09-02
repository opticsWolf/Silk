# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The main-thread half of graph authoring (spec §18, D70, D72).

The tools run on the agent's worker thread and may not touch the scene;
this object lives on the main thread and does the touching. Between them
is :class:`~..functions.main_thread_call.MainThreadCall` -- D49's waiter
with the event loop as its resolver instead of a person.

**Every mutation is one undoable command** (D72). Placements and
connections go through the canvas's own `AddNodeCommand`,
`AddConnectionCommand`, `RemoveNodesCommand` and
`RemoveConnectionsCommand`, pushed onto the canvas's `UndoManager`, never
through raw scene manipulation. That is the primary safety property: the
human undoes the agent's edit with the gesture they undo their own, and
the agent's work shows up in the place a user looks to see what happened.
An agent whose edits could not be undone would be a different and much
more dangerous tool.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QObject, Qt, Signal

from weave.logger import get_logger

from ..functions.graph_author import (
    OP_CONNECT, OP_DESCRIBE, OP_DISCONNECT, OP_LIST, OP_PLACE, OP_REMOVE,
    describe_class, describe_instance,
)
from ..functions.main_thread_call import CallRequest, MainThreadCall

log = get_logger("SilkGraphCanvas")


class CanvasAuthor(QObject):
    """Performs graph edits on the main thread, on behalf of a worker.

    Constructed by the Agent node (main thread), so its queued slot runs
    on the main thread no matter which thread emitted the signal. The
    seam it serves is the agent's; when the run ends the node closes the
    seam and every outstanding call refuses rather than hanging.
    """

    #: Worker -> main. Queued explicitly: the default would be a direct
    #: call when a worker happens to be the main thread, and the seam's
    #: ordering rule is easier to reason about with one path.
    _requested = Signal(object)

    def __init__(self, node: Any, seam: MainThreadCall) -> None:
        super().__init__()
        self._node = node
        self._seam = seam
        self._requested.connect(self._serve, Qt.ConnectionType.QueuedConnection)

    # -- the worker side --------------------------------------------------

    def deliver(self, request: CallRequest) -> None:
        """Hand a request to the main thread. **Worker thread.**"""
        self._requested.emit(request)

    # -- the main-thread side ---------------------------------------------

    def _serve(self, request: CallRequest) -> None:
        self._seam.serve(request, self._perform)

    def _perform(self, request: CallRequest) -> Any:
        canvas = self._canvas()
        if canvas is None:
            return {"ok": False, "error": (
                "There is no canvas for this run (headless evaluation or a "
                "closed graph), so the graph cannot be edited.")}
        handler = {
            OP_LIST: self._list,
            OP_DESCRIBE: self._describe,
            OP_PLACE: self._place,
            OP_CONNECT: self._connect,
            OP_DISCONNECT: self._disconnect,
            OP_REMOVE: self._remove,
        }.get(request.op)
        if handler is None:
            return {"ok": False, "error": f"unknown operation '{request.op}'"}
        return handler(canvas, request.args)

    def _canvas(self) -> Optional[Any]:
        scene = None
        try:
            scene = self._node.scene()
        except RuntimeError:      # the node's C++ side is gone
            return None
        return scene if scene is not None and hasattr(scene, "undo_manager") \
            else scene

    # -- reads ------------------------------------------------------------

    def _list(self, canvas: Any, args: dict) -> dict:
        """The whitelist, rendered as the model needs it (D69)."""
        from weave.registry import NODE_REGISTRY

        allowed = [str(n) for n in (args.get("allowed") or ())]
        entries = []
        for name in allowed:
            cls = NODE_REGISTRY.get_node_class(name)
            if cls is None:
                continue
            entry = describe_class(cls)
            if not entry["inputs"] and not entry["outputs"]:
                # Weave declares ports in __init__, so ask an instance --
                # cheap here, and this is the thread where building one
                # (widgets and all) is legal.
                entry = self._describe_by_instance(cls, entry)
            entries.append(entry)
        return {"ok": True, "value": {"nodes": entries,
                                      "total": len(entries)}}

    @staticmethod
    def _describe_by_instance(cls: Any, entry: dict) -> dict:
        try:
            probe = cls()
        except Exception as exc:  # noqa: BLE001 - a node that will not build
            entry["error"] = f"cannot inspect ports: {type(exc).__name__}: {exc}"
            return entry
        try:
            live = describe_instance(probe)
            entry["inputs"] = [
                {k: p[k] for k in ("name", "datatype", "description")}
                for p in live["inputs"]]
            entry["outputs"] = [
                {k: p[k] for k in ("name", "datatype", "description")}
                for p in live["outputs"]]
        finally:
            probe.deleteLater()
        return entry

    def _describe(self, canvas: Any, args: dict) -> dict:
        """Nodes and edges, in the tuple shape the undo commands speak."""
        nodes, edges = [], []
        for item in canvas.items():
            if not hasattr(item, "unique_id"):
                continue
            nodes.append(describe_instance(item))
            for port in getattr(item, "outputs", []) or []:
                for trace in getattr(port, "connected_traces", None) or ():
                    target = getattr(trace, "target", None)
                    dst_node = getattr(target, "node", None)
                    if target is None or dst_node is None:
                        continue
                    edges.append([str(item.unique_id), str(port.name),
                                  str(getattr(dst_node, "unique_id", "")),
                                  str(getattr(target, "name", ""))])
        return {"ok": True, "value": {"nodes": nodes, "edges": edges,
                                      "agent_id": str(getattr(
                                          self._node, "unique_id", ""))}}

    # -- mutations, each one undoable command ------------------------------

    def _place(self, canvas: Any, args: dict) -> dict:
        from weave.canvas.undo_commands import (
            AddNodeCommand, capture_node_snapshot,
        )
        from weave.canvas.undo_manager import default_registry_map
        from weave.registry import NODE_REGISTRY

        class_name = str(args.get("class_name") or "")
        node_cls = NODE_REGISTRY.get_node_class(class_name)
        if node_cls is None:
            return {"ok": False,
                    "error": f"'{class_name}' is not a registered node class"}
        position = args.get("position") or [0.0, 0.0]
        try:
            node = node_cls()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False,
                    "error": f"{class_name} could not be created: {exc}"}
        title = str(args.get("title") or "").strip()
        if title:
            try:
                node.title = title
            except Exception:  # noqa: BLE001 - a node that fixes its title
                pass
        canvas.add_node(node, (float(position[0]), float(position[1])))

        uid, cls_name, state, npos = capture_node_snapshot(node)
        canvas.undo_manager.push(
            AddNodeCommand(cls_name, state, uid, npos, default_registry_map()))
        return {"ok": True, "value": {"id": str(uid), "class_name": cls_name,
                                      "title": str(getattr(node, "title", "")),
                                      "position": list(npos)}}

    def _connect(self, canvas: Any, args: dict) -> dict:
        from weave.canvas.undo_commands import AddConnectionCommand
        from weave.node.port_utils import ConnectionFactory, PortUtils

        src, src_port_name = self._port(canvas, args.get("src_id"),
                                        args.get("src_port"), "outputs")
        dst, dst_port_name = self._port(canvas, args.get("dst_id"),
                                        args.get("dst_port"), "inputs")
        if isinstance(src, str):
            return {"ok": False, "error": src}
        if isinstance(dst, str):
            return {"ok": False, "error": dst}

        # The port type system does the hard part: an illegal connection
        # is refused with a reason the model can act on, rather than
        # producing a broken graph (D69).
        if not PortUtils.are_compatible(src, dst):
            return {"ok": False, "error": (
                f"'{src.datatype}' does not connect to a '{dst.datatype}' "
                f"input, or that input is already connected.")}
        if ConnectionFactory.create(canvas, src, dst, validate=True,
                                    trigger_compute=True) is None:
            return {"ok": False,
                    "error": "the connection could not be created"}

        edge = (str(args.get("src_id")), src_port_name,
                str(args.get("dst_id")), dst_port_name)
        canvas.undo_manager.push(AddConnectionCommand(edge))
        return {"ok": True, "value": {"edge": list(edge)}}

    def _disconnect(self, canvas: Any, args: dict) -> dict:
        from weave.canvas.undo_commands import RemoveConnectionsCommand

        edge = (str(args.get("src_id")), str(args.get("src_port")),
                str(args.get("dst_id")), str(args.get("dst_port")))
        command = RemoveConnectionsCommand([edge])
        command.redo(canvas)
        canvas.undo_manager.push(command)
        return {"ok": True, "value": {"edge": list(edge)}}

    def _remove(self, canvas: Any, args: dict) -> dict:
        from weave.canvas.undo_commands import (
            RemoveNodesCommand, capture_node_connections, capture_node_snapshot,
        )
        from weave.canvas.undo_manager import default_registry_map

        node = self._node_by_id(canvas, str(args.get("id") or ""))
        if node is None:
            return {"ok": False,
                    "error": f"no node with id '{args.get('id')}' on the canvas"}
        snapshot = [capture_node_snapshot(node)]
        # The canvas's own capture is used rather than a walk of our own:
        # undo has to put back exactly what removal took away, and that is
        # the function removal everywhere else in Weave uses.
        edges = list(capture_node_connections(canvas, node))
        command = RemoveNodesCommand(snapshot, edges, default_registry_map())
        command.redo(canvas)
        canvas.undo_manager.push(command)
        return {"ok": True, "value": {"id": str(args.get("id")),
                                      "edges_removed": len(edges)}}

    # -- lookups ----------------------------------------------------------

    @staticmethod
    def _node_by_id(canvas: Any, node_uid: str) -> Optional[Any]:
        for item in canvas.items():
            if str(getattr(item, "unique_id", "")) == node_uid:
                return item
        return None

    @staticmethod
    def _traces_of(node: Any) -> list:
        traces = []
        for side in ("inputs", "outputs"):
            for port in getattr(node, side, []) or []:
                for trace in getattr(port, "connected_traces", None) or ():
                    if trace not in traces:
                        traces.append(trace)
        return traces

    def _port(self, canvas: Any, node_uid: Any, port_name: Any,
              side: str) -> tuple:
        node = self._node_by_id(canvas, str(node_uid or ""))
        if node is None:
            return f"no node with id '{node_uid}' on the canvas", ""
        name = str(port_name or "")
        for port in getattr(node, side, []) or []:
            if str(getattr(port, "name", "")) == name:
                return port, name
        available = ", ".join(str(getattr(p, "name", ""))
                              for p in getattr(node, side, []) or [])
        return (f"'{node_uid}' has no {side[:-1]} port '{name}' "
                f"(it has: {available or 'none'})"), ""
