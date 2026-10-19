# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The graph tools' seam over Weave's control API (``weave-control 1.0``).

The graph-authoring tools speak to a *seam*: ``seam.call(op, **args)``
returns a :class:`~.main_thread_call.CallResult` and never raises. On a
canvas that seam is :class:`~.main_thread_call.MainThreadCall`, served by
the Qt-side ``CanvasAuthor``. :class:`ControlSeam` is the same seam over
Weave's control API instead -- a headless Weave in this process
(``weave.control.LocalControl``), one run as a child or reached over TCP
(``weave.control.ControlClient``), or a canvas serving it -- so an agent
builds, configures, evaluates and reads a graph with no canvas and no Qt
on this side.

What stays Silk's is everything above the seam: the whitelist (D71), the
run scope, the self-modification guard (D73), approval of the destructive
verbs (D81). The seam turns each op into control-API calls and the
answers back into the shapes ``CanvasAuthor`` returns, so a tool cannot
tell which one it is talking to. Every edit is one call, and so one undo
step on the Weave that serves it (D72).

Two ops have no canvas twin before this: ``evaluate_graph`` (run the
graph, or some of its nodes) and ``read_output`` (an output port's value)
-- on a canvas a result shows in a widget, headless there is nothing to
look at but the value.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from .blocking_seam import CAUSE_ANSWERED, CAUSE_TRANSPORT, new_request_id
from .graph_author import (
    OP_CONNECT, OP_DESCRIBE, OP_DISCONNECT, OP_EVALUATE, OP_LIST, OP_PLACE,
    OP_READ, OP_REMOVE, OP_SET_VALUE, OP_SETTINGS, check_settable, coerce_value,
)
from .main_thread_call import CallResult

#: The roles node.describe reports, as the canvas's widget roles name them.
_ROLES = {"INTERNAL": "INTERNAL", "BIDIRECTIONAL": "BIDIRECTIONAL", "INPUT": "INPUT",
          "OUTPUT": "OUTPUT", "DISPLAY": "DISPLAY", "ACTION": "ACTION"}


class ControlSeam:
    """A seam whose resolver is a Weave serving the control API.

    *control* is anything with ``call(method, **params)`` that returns the
    method's result and raises on a refusal (an exception with ``code`` and
    ``message``, ``weave.control.ControlError``): ``LocalControl``,
    ``ControlClient``, ``CanvasControl``. *agent_uid* is the node running
    the agent in that graph, if it is one (a headless graph usually is not).
    """

    def __init__(self, control: Any, *, agent_uid: str = "") -> None:
        self.control = control
        self.agent_uid = agent_uid
        self._kinds: Optional[Dict[str, dict]] = None

    # -- the seam ---------------------------------------------------------

    def call(self, op: str, *, timeout: Optional[float] = None,
             detail: Optional[dict] = None, **args: Any) -> CallResult:
        """Do *op* and return what happened; never raises."""
        call_id = new_request_id()
        handler: Optional[Callable[[dict], Any]] = {
            OP_LIST: self._list, OP_DESCRIBE: self._describe, OP_PLACE: self._place,
            OP_CONNECT: self._connect, OP_DISCONNECT: self._disconnect,
            OP_REMOVE: self._remove, OP_SETTINGS: self._settings,
            OP_SET_VALUE: self._set_value, OP_EVALUATE: self._evaluate, OP_READ: self._read,
        }.get(op)
        if handler is None:
            return CallResult(call_id=call_id, op=op, ok=False, cause=CAUSE_ANSWERED,
                              error=f"unknown operation '{op}'")
        try:
            out = handler(dict(args))
        except _Refused as refused:
            return CallResult(call_id=call_id, op=op, ok=False, cause=CAUSE_ANSWERED,
                              error=str(refused))
        except Exception as exc:  # noqa: BLE001 - a refusal or a broken transport
            code = getattr(exc, "code", None)
            message = getattr(exc, "message", None) or str(exc)
            if isinstance(code, int):
                return CallResult(call_id=call_id, op=op, ok=False, cause=CAUSE_ANSWERED,
                                  error=f"Refused: {message}")
            return CallResult(call_id=call_id, op=op, ok=False, cause=CAUSE_TRANSPORT,
                              error=f"{type(exc).__name__}: {exc}")
        return CallResult(call_id=call_id, op=op, ok=True, value=out)

    # -- helpers ----------------------------------------------------------

    def _api(self, method: str, **params: Any) -> Any:
        return self.control.call(method, **params)

    def kinds(self, refresh: bool = False) -> Dict[str, dict]:
        """Every kind the Weave can build, by name (its declarations)."""
        if self._kinds is None or refresh:
            self._kinds = {d["kind"]: d for d in self._api("graph.kinds").get("kinds", [])}
        return self._kinds

    def _node(self, uid: str) -> dict:
        if not uid:
            raise _Refused("No node id was given.")
        try:
            return self._api("node.describe", uid=uid)
        except Exception as exc:  # noqa: BLE001
            if isinstance(getattr(exc, "code", None), int):
                raise _Refused(f"no node with id '{uid}' in the graph") from None
            raise

    @staticmethod
    def _port(p: dict) -> dict:
        return {"name": p.get("name", ""), "datatype": p.get("datatype", ""),
                "description": p.get("description", ""), "connected": bool(p.get("connected"))}

    def _entry(self, decl: dict) -> dict:
        meta = decl.get("meta") or {}
        category = "/".join(c for c in (meta.get("class"), meta.get("subclass")) if c)
        return {
            "class_name": decl["kind"],
            "display_name": meta.get("name") or decl["kind"],
            "description": (meta.get("description") or "").strip(),
            "category": category,
            "tags": list(meta.get("tags") or ()),
            "inputs": [{k: p.get(k, "") for k in ("name", "datatype", "description")}
                       for p in decl.get("inputs", [])],
            "outputs": [{k: p.get(k, "") for k in ("name", "datatype", "description")}
                        for p in decl.get("outputs", [])],
        }

    def _rows(self, node: dict) -> List[dict]:
        """The node's settings as ``CanvasAuthor._bindings`` lists them."""
        decl = self.kinds().get(node.get("kind", ""), {})
        controls = {c.get("name"): c for c in decl.get("controls", []) if c.get("name")}
        connected = {p["name"] for p in node.get("inputs", []) if p.get("connected")}
        rows = []
        for p in node.get("params", []):
            control = controls.get(p["name"], {})
            rows.append({
                "name": p["name"],
                "role": _ROLES.get(str(p.get("role", "")).upper(), str(p.get("role", ""))),
                "datatype": control.get("datatype") or _datatype_of(p.get("value")),
                "description": control.get("description") or "",
                "connected": p["name"] in connected,
                "value": p.get("value"),
                "default": p.get("default"),
                "exists": True,
            })
        return sorted(rows, key=lambda r: r["name"])

    # -- the ops ----------------------------------------------------------

    def _list(self, args: dict) -> dict:
        kinds = self.kinds(refresh=True)
        entries = [self._entry(kinds[name]) for name in (args.get("allowed") or ())
                   if name in kinds]
        return {"nodes": entries, "total": len(entries)}

    def _describe(self, args: dict) -> dict:
        graph = self._api("graph.describe")
        nodes = [{
            "id": n["uid"], "class_name": n["kind"], "title": n.get("title", ""),
            "position": list(n.get("pos") or (0.0, 0.0)),
            "inputs": [self._port(p) for p in n.get("inputs", [])],
            "outputs": [self._port(p) for p in n.get("outputs", [])],
        } for n in graph.get("nodes", [])]
        return {"nodes": nodes, "edges": [list(w) for w in graph.get("wires", [])],
                "agent_id": self.agent_uid}

    def _place(self, args: dict) -> dict:
        class_name = str(args.get("class_name") or "")
        if class_name not in self.kinds():
            raise _Refused(f"'{class_name}' is not a node kind this Weave can build")
        position = [float(v) for v in (args.get("position") or (0.0, 0.0))]
        params: Dict[str, Any] = {"kind": class_name, "pos": position}
        title = str(args.get("title") or "").strip()
        if title:
            params["title"] = title
        uid = self._api("node.add", **params)["uid"]
        node = self._node(uid)
        return {"id": uid, "class_name": node.get("kind", class_name),
                "title": node.get("title", title), "position": list(node.get("pos") or position)}

    def _connect(self, args: dict) -> dict:
        edge = [str(args.get(k) or "") for k in ("src_id", "src_port", "dst_id", "dst_port")]
        dst = self._node(edge[2])
        port = next((p for p in dst.get("inputs", []) if p["name"] == edge[3]), None)
        if port is not None and port.get("connected"):
            raise _Refused(f"'{edge[3]}' on '{edge[2]}' is already connected; disconnect it first.")
        try:
            self._api("wire.connect", src=edge[0], output=edge[1], dst=edge[2], input=edge[3],
                      replace=False)
        except Exception as exc:  # noqa: BLE001
            if isinstance(getattr(exc, "code", None), int):
                raise _Refused(f"the connection was refused: {getattr(exc, 'message', exc)}") from None
            raise
        return {"edge": edge}

    def _disconnect(self, args: dict) -> dict:
        edge = [str(args.get(k) or "") for k in ("src_id", "src_port", "dst_id", "dst_port")]
        wires = [list(w) for w in self._api("graph.describe").get("wires", [])]
        if edge not in wires:
            raise _Refused(f"there is no edge {edge} in the graph")
        self._api("wire.disconnect", dst=edge[2], input=edge[3])
        return {"edge": edge}

    def _remove(self, args: dict) -> dict:
        uid = str(args.get("id") or "")
        self._node(uid)
        wires = self._api("graph.describe").get("wires", [])
        touching = [w for w in wires if uid in (w[0], w[2])]
        self._api("node.remove", uid=uid)
        return {"id": uid, "edges_removed": len(touching)}

    def _settings(self, args: dict) -> dict:
        uid = str(args.get("id") or "")
        node = self._node(uid)
        rows = []
        for row in self._rows(node):
            refusal = check_settable(row, uid)
            rows.append({**{k: row[k] for k in ("name", "role", "datatype", "description", "connected")},
                         "value": row["value"], "default": row["default"],
                         "settable": refusal is None, "why_not": "" if refusal is None else refusal.reason})
        return {"id": uid, "class_name": node.get("kind", ""), "title": node.get("title", ""),
                "settings": rows}

    def _set_value(self, args: dict) -> dict:
        uid, name = str(args.get("id") or ""), str(args.get("port") or "")
        rows = {r["name"]: r for r in self._rows(self._node(uid))}
        row = rows.get(name, {"name": name, "exists": False})
        refusal = check_settable(row, uid)
        if refusal is not None:
            available = ", ".join(r["name"] for r in rows.values() if check_settable(r, uid) is None)
            raise _Refused(f"{refusal.reason} Settable here: {available or 'nothing'}.")
        value, refusal = coerce_value(row["datatype"], args.get("value"))
        if refusal is not None:
            raise _Refused(refusal.reason)
        before = row["value"]
        after = self._api("param.set", uid=uid, name=name, value=value).get("value", value)
        return {"id": uid, "port": name, "requested": value, "value": after, "previous": before,
                "note": ("" if after == value else
                         "the node adjusted the value; 'value' is what it holds now")}

    def _evaluate(self, args: dict) -> dict:
        ids = [str(i) for i in (args.get("ids") or ())]
        exclude = {str(i) for i in (args.get("exclude") or ())}
        if exclude and not ids:
            ids = [n["uid"] for n in self._api("graph.describe").get("nodes", [])
                   if n["uid"] not in exclude]
        out = self._api("graph.evaluate", **({"roots": ids} if ids else {}))
        return {"evaluated": ids or "all", "errors": out.get("errors", []) if isinstance(out, dict) else []}

    def _read(self, args: dict) -> dict:
        uid, port = str(args.get("id") or ""), str(args.get("port") or "")
        node = self._node(uid)
        if not any(p["name"] == port for p in node.get("outputs", [])):
            names = ", ".join(p["name"] for p in node.get("outputs", [])) or "none"
            raise _Refused(f"'{uid}' has no output '{port}' (its outputs: {names}).")
        method = "node.request" if args.get("evaluate", True) else "node.cached"
        out = self._api(method, uid=uid, port=port)
        return {"id": uid, "port": port, "value": out.get("value"), "state": out.get("state"),
                "error": out.get("error")}


class _Refused(Exception):
    """A refusal with a sentence for the model (raised inside an op)."""


def _datatype_of(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    return "any"


def control_binding(control: Any, *, agent_uid: str = "", scope: Any = None) -> Any:
    """A :class:`~.graph_author.CanvasBinding` whose seam is the control API."""
    from .graph_author import CanvasBinding, RunScope
    return CanvasBinding(seam=ControlSeam(control, agent_uid=agent_uid),
                         scope=scope if scope is not None else RunScope(), agent_uid=agent_uid)


__all__: Iterable[str] = ("ControlSeam", "control_binding")
