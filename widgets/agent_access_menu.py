# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

**Agent access** in a node's right-click menu: how much an agent may edit
that node (``graph_author`` rule 2).

The levels are the user's to set and the agent's to read: the submenu
writes Silk's marks into the node's plugin data, which is saved with the
graph, and the graph tools read them through ``describe_graph``. Each
change is one undo step, as any other edit to a node is.
"""
from __future__ import annotations

from typing import Any, Iterable, List

from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu

from ..functions.graph_author import (
    ACCESS_CONFIGURE, ACCESS_CONNECT, ACCESS_FULL, ACCESS_NONE, MARK_ACCESS,
    MARK_OWNER, PLUGIN_KEY, access_marks, node_title,
)

#: The key this entry is registered under in Weave's node menu.
MENU_KEY = "silk.agent_access"

#: The choices, in menu order. "" is the default: no level chosen.
CHOICES = (
    ("", None),
    (ACCESS_NONE, "No access: read it, wire from its outputs"),
    (ACCESS_CONNECT, "Connect: also wire into its inputs"),
    (ACCESS_CONFIGURE, "Configure: also change its settings"),
    (ACCESS_FULL, "Full: also remove it"),
)


def _editable(nodes: Iterable[Any]) -> List[Any]:
    """The nodes a level can be set on (a group has no access of its own)."""
    try:
        from weave.group.node import is_group_node
    except ImportError:  # pragma: no cover - a Weave without groups
        def is_group_node(_node: Any) -> bool:
            return False
    return [n for n in nodes if hasattr(n, "set_plugin_data")
            and hasattr(n, "unique_id") and not is_group_node(n)]


def _owner_line(nodes: List[Any], canvas: Any) -> str:
    owners = {str(access_marks(n).get(MARK_OWNER) or "") for n in nodes}
    if len(owners) != 1:
        return "Placed by you and by agents"
    owner = owners.pop()
    if not owner:
        return "Placed by you" if len(nodes) == 1 else "All placed by you"
    for item in getattr(canvas, "items", lambda: [])():
        if str(getattr(item, "unique_id", "")) == owner:
            return f"Placed by agent “{node_title(item)}”"
    return "Placed by an agent no longer on the canvas"


def _default_label(nodes: List[Any]) -> str:
    owned = [bool(access_marks(n).get(MARK_OWNER)) for n in nodes]
    if all(owned):
        return "Default: full for the agent that placed it, none for others"
    if not any(owned):
        return "Default: no access"
    return "Default: full for the placing agent, else none"


def set_access(nodes: Iterable[Any], level: str, canvas: Any) -> int:
    """Choose *level* ("" for the default) on *nodes*, as one undo step.

    Returns how many nodes changed.
    """
    from weave.canvas.undo_commands import NodePropertyCommand, get_node_uid

    changes = []
    for node in _editable(nodes):
        before = access_marks(node)
        after = dict(before)
        if level:
            after[MARK_ACCESS] = level
        else:
            after.pop(MARK_ACCESS, None)
        if after == before:
            continue
        node.set_plugin_data((PLUGIN_KEY, after))
        changes.append((get_node_uid(node), (PLUGIN_KEY, before or None),
                        (PLUGIN_KEY, after or None)))
    manager = getattr(canvas, "undo_manager", None)
    if changes and manager is not None:
        manager.push(NodePropertyCommand(changes, "set_plugin_data", "Agent access"))
    return len(changes)


def build_agent_access_menu(menu: QMenu, nodes: Iterable[Any], canvas: Any) -> None:
    """Weave node-menu entry: the **Agent access** submenu."""
    nodes = _editable(nodes)
    if not nodes:
        return
    sub = menu.addMenu("Agent access")
    sub.setToolTipsVisible(True)
    who = QAction(_owner_line(nodes, canvas), sub)
    who.setEnabled(False)
    sub.addAction(who)
    sub.addSeparator()

    chosen = {str(access_marks(n).get(MARK_ACCESS) or "") for n in nodes}
    current = chosen.pop() if len(chosen) == 1 else None
    group = QActionGroup(sub)
    group.setExclusive(True)
    for level, label in CHOICES:
        action = QAction(label or _default_label(nodes), sub)
        action.setCheckable(True)
        action.setChecked(level == current)
        action.triggered.connect(
            lambda _checked=False, lv=level: set_access(nodes, lv, canvas))
        group.addAction(action)
        sub.addAction(action)
        if not level:
            sub.addSeparator()


def install() -> bool:
    """Put the submenu into Weave's node menu. False on a Weave without
    the hook (the levels still apply; only the menu to change them is
    missing)."""
    try:
        from weave.canvas.menu import register_node_menu_extension
    except ImportError:
        return False
    register_node_menu_extension(MENU_KEY, build_agent_access_menu)
    return True
