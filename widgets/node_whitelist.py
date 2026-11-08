# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Which node classes an agent may place (spec §18, D71).

A checkable tree over `NODE_REGISTRY` in the levels the canvas's own
*Browse Nodes* menu shows -- category, then the nodes placed directly in
it, then its subcategories -- so a class is found where the user already
knows to look for it. Nodes are ticked or not; a level row is ticked,
unticked or partly ticked, and ticking it ticks every node beneath it.
The widget value is a plain list of class names, so it travels in the
ToolBox recipe, saves with the graph, and fits in a preset. Unlike grants
(D35) it carries no secret and no filesystem authority: it says which
*kinds of node* may appear on the canvas, and nothing else.

**The default is empty and there is no "allow all".** *Select All* in
the right-click menu ticks every class the tree shows, one name each --
a deliberate act, and one that does not reach a class registered later.
The safe state is the one you get by doing nothing (I6). A whitelisted
class that is no longer registered stays in the value and is listed
under its own heading, rather than becoming a refusal the agent hits
halfway through building something; unticking it there drops it.

The tree re-reads the registry when it changes (`NodeRegistry.generation`
plus its listener), so a hot-loaded plugin's nodes become tickable
without rebuilding the graph.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import (
    QHeaderView, QLabel, QLineEdit, QMenu, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from weave.widgets.composite_value import CompositeValueWidget
from weave.widgetcore import menus as widget_menus

#: A node row's class name; ``None`` on a level row.
_NAME_ROLE = Qt.ItemDataRole.UserRole
#: A level row's path (``"Math"``, ``"Math/Trig"``), which keeps its
#: expansion across a rebuild.
_LEVEL_ROLE = Qt.ItemDataRole.UserRole + 1

EMPTY_HINT = "Nothing ticked — the agent may not place any node."
MISSING_HINT = "not registered any more: {}"
MISSING_LEVEL = "Not registered"

_Leaf = Tuple[str, str, Any]                         # (name, label, class)
_Level = Tuple[str, List[_Leaf], List[Tuple[str, List[_Leaf]]]]


class NodeWhitelistWidget(CompositeValueWidget):
    """Checkable tree of registered node classes. Value: list of names."""

    #: What the dialog around the tree says above it.
    intro = ("Node classes this ToolBox's agents may place on the canvas, "
             "in the levels of Browse Nodes. Nothing ticked means none — "
             "there is no 'allow all'. A ToolSet or Role downstream may "
             "narrow this list, never widen it.")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checked: set[str] = set()
        self._reload_pending = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter nodes…")
        self._filter.setClearButtonEnabled(True)
        self._filter.setToolTip(
            "Show only the nodes whose name matches. Ticking a level and "
            "the right-click selections act on the nodes shown."
        )
        layout.addWidget(self._filter)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Node", "Ticked"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setMinimumHeight(220)
        header = self._tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._tree, 1)

        self._status = QLabel(EMPTY_HINT)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        self._filter.textChanged.connect(self._apply_filter)
        self.reload()
        self._subscribe()

    # ── staying current across a hot load (D74) ──────────────────────

    def _subscribe(self) -> None:
        """Re-read the registry when it changes.

        Weave's hot-load work gives `NODE_REGISTRY` a generation counter
        and a listener; without this, a plugin loaded into the running
        session is placeable by the *tools* (they resolve live) and
        invisible in the *widget*, so the user cannot tick it without
        rebuilding the graph.
        """
        try:
            from weave.registry import NODE_REGISTRY
        except ImportError:      # pragma: no cover - Weave without a registry
            return
        NODE_REGISTRY.add_listener(self._on_registry_changed)
        self.destroyed.connect(lambda *_: self._unsubscribe())

    def _unsubscribe(self) -> None:
        try:
            from weave.registry import NODE_REGISTRY

            NODE_REGISTRY.remove_listener(self._on_registry_changed)
        except Exception:  # noqa: BLE001 - teardown, and nothing depends on it
            pass

    def _on_registry_changed(self, event: str, cls: Any = None) -> None:
        """Coalesce a burst into one rebuild.

        Loading a suite registers every class in it, one event each; a
        tree rebuilt per class would rebuild N times and lose the user's
        expansion state N times.
        """
        if self._reload_pending:
            return
        self._reload_pending = True
        QTimer.singleShot(0, self._reload_now)

    def _reload_now(self) -> None:
        self._reload_pending = False
        try:
            self.reload()
        except RuntimeError:      # the C++ side went away first
            self._unsubscribe()

    # ── the registry side ────────────────────────────────────────────

    def reload(self) -> None:
        """Rebuild the tree from the registry, keeping what was ticked.

        Ticks survive a reload even when their class is momentarily
        absent: a plugin being hot-reloaded must not silently clear a
        user's choices. So does which levels were open.
        """
        expanded = {item.data(0, _LEVEL_ROLE) for item in self._level_items()
                    if item.isExpanded()}
        icons = _node_icons()
        blocked = self._tree.blockSignals(True)
        try:
            self._tree.clear()
            for category, direct, subs in self._registry_levels():
                cat_item = self._add_level(self._tree, category, category)
                cat_classes = [c for *_n, c in direct] + [
                    c for _s, leaves in subs for *_n, c in leaves]
                _set_icon(cat_item, icons, cat_classes,
                          ("for_menu_class", "for_menu"))
                for leaf in direct:
                    self._add_leaf(cat_item, leaf, icons)
                for sub, leaves in subs:
                    sub_item = self._add_level(cat_item, sub, f"{category}/{sub}")
                    _set_icon(sub_item, icons, [c for *_n, c in leaves],
                              ("for_menu_subclass", "for_menu_class", "for_menu"))
                    for leaf in leaves:
                        self._add_leaf(sub_item, leaf, icons)
            gone = self.missing()
            if gone:
                level = self._add_level(self._tree, MISSING_LEVEL, "\0missing")
                level.setToolTip(0, "Ticked classes the registry does not "
                                    "have now. Untick one to drop it.")
                for name in gone:
                    self._add_leaf(level, (name, name, None), None)
                expanded.add("\0missing")
            for item in self._level_items():
                item.setExpanded(item.data(0, _LEVEL_ROLE) in expanded)
            self._sync_levels()
        finally:
            self._tree.blockSignals(blocked)
        self._apply_filter(self._filter.text())
        self._refresh_status()

    def _add_level(self, parent, text: str, path: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [text])
        item.setData(0, _NAME_ROLE, None)
        item.setData(0, _LEVEL_ROLE, path)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, Qt.CheckState.Unchecked)
        item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight
                              | Qt.AlignmentFlag.AlignVCenter)
        return item

    def _add_leaf(self, parent: QTreeWidgetItem, leaf: _Leaf, icons) -> None:
        name, label, cls = leaf
        item = QTreeWidgetItem(parent, [label])
        item.setData(0, _NAME_ROLE, name)
        item.setToolTip(0, name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, Qt.CheckState.Checked if name in self._checked
                           else Qt.CheckState.Unchecked)
        if icons is not None and cls is not None:
            _set_icon(item, icons, [cls], ("for_menu",))

    @staticmethod
    def _registry_levels() -> List[_Level]:
        """The registry in Browse Nodes' order (`canvas.menu`): categories
        by name; in each, the nodes placed directly in it, then its
        subcategories by name; nodes by display name."""
        try:
            from weave.registry import NODE_REGISTRY
        except ImportError:      # pragma: no cover - Weave without a registry
            return []
        tree = NODE_REGISTRY.get_tree() or {}

        def leaves(classes) -> List[_Leaf]:
            found = [(c.__name__, NODE_REGISTRY.get_node_display_name(c), c)
                     for c in classes if c is not None]
            return sorted(found, key=lambda leaf: leaf[1])

        levels: List[_Level] = []
        for category, sub_dict in sorted(tree.items(), key=lambda kv: str(kv[0])):
            if not isinstance(sub_dict, dict):
                continue
            direct = leaves(sub_dict.get(None, ()))
            subs = [(str(sub), leaves(classes))
                    for sub, classes in sorted(
                        ((k, v) for k, v in sub_dict.items() if k is not None),
                        key=lambda kv: str(kv[0]))]
            subs = [(sub, found) for sub, found in subs if found]
            if direct or subs:
                levels.append((str(category), direct, subs))
        return levels

    def registered_names(self) -> set[str]:
        return {name for _cat, direct, subs in self._registry_levels()
                for name, _label, _cls in direct + [
                    leaf for _s, found in subs for leaf in found]}

    def missing(self) -> list[str]:
        """Ticked classes the registry does not have (D71)."""
        return sorted(self._checked - self.registered_names())

    # ── walking the tree ─────────────────────────────────────────────

    def _items(self, root: Optional[QTreeWidgetItem] = None
               ) -> Iterator[QTreeWidgetItem]:
        """Every row under *root* (the whole tree for ``None``), depth first."""
        stack = ([root] if root is not None else
                 [self._tree.topLevelItem(i)
                  for i in reversed(range(self._tree.topLevelItemCount()))])
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in reversed(range(item.childCount())))

    def _level_items(self) -> Iterator[QTreeWidgetItem]:
        return (i for i in self._items() if i.data(0, _NAME_ROLE) is None)

    def _leaves(self, root: Optional[QTreeWidgetItem] = None, *,
                shown_only: bool = False) -> Iterator[QTreeWidgetItem]:
        for item in self._items(root):
            if item.data(0, _NAME_ROLE) is None:
                continue
            if shown_only and item.isHidden():
                continue
            yield item

    def _sync_levels(self) -> None:
        """Each level's tick and count from the nodes beneath it."""
        def visit(item: QTreeWidgetItem) -> Tuple[int, int]:
            if item.data(0, _NAME_ROLE) is not None:
                return (int(item.checkState(0) == Qt.CheckState.Checked), 1)
            ticked = total = 0
            for i in range(item.childCount()):
                t, n = visit(item.child(i))
                ticked, total = ticked + t, total + n
            item.setCheckState(0, Qt.CheckState.Unchecked if ticked == 0
                               else Qt.CheckState.Checked if ticked == total
                               else Qt.CheckState.PartiallyChecked)
            item.setText(1, f"{ticked} / {total}")
            return ticked, total

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i))

    # ── the value side ───────────────────────────────────────────────

    def _read(self) -> Any:
        return sorted(self._checked)

    def _write(self, value: Any) -> None:
        self._checked = {str(n).strip() for n in (value or ())
                         if str(n).strip()}
        self.reload()

    def _internal_signal_sources(self) -> Iterable:
        return (self._tree,)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        if item.data(0, _NAME_ROLE) is None:
            # A level row: tick or untick every node shown beneath it.
            # From partly ticked, a click ticks the rest.
            target = item.checkState(0) != Qt.CheckState.Unchecked
            self._tick(lambda _ticked: target, item)
        else:
            self._tick(None)

    def _tick(self, rule: Optional[Callable[[bool], bool]],
              root: Optional[QTreeWidgetItem] = None) -> None:
        """Apply *rule* (old tick -> new tick) to the nodes shown under
        *root*, then re-read the value from the tree. ``None`` only
        re-reads: a node row the user clicked has its tick already."""
        blocked = self._tree.blockSignals(True)
        try:
            if rule is not None:
                for leaf in self._leaves(root, shown_only=True):
                    ticked = leaf.checkState(0) == Qt.CheckState.Checked
                    leaf.setCheckState(0, Qt.CheckState.Checked if rule(ticked)
                                       else Qt.CheckState.Unchecked)
            self._sync_levels()
        finally:
            self._tree.blockSignals(blocked)
        checked = {str(leaf.data(0, _NAME_ROLE)) for leaf in self._leaves()
                   if leaf.checkState(0) == Qt.CheckState.Checked}
        if checked == self._checked:
            return
        self._checked = checked
        self._refresh_status()
        self._notify_value_changed()

    # ── bulk selection ───────────────────────────────────────────────

    def check_all(self, root: Optional[QTreeWidgetItem] = None) -> None:
        self._tick(lambda _ticked: True, root)

    def check_none(self, root: Optional[QTreeWidgetItem] = None) -> None:
        self._tick(lambda _ticked: False, root)

    def invert_checked(self, root: Optional[QTreeWidgetItem] = None) -> None:
        self._tick(lambda ticked: not ticked, root)

    def _show_context_menu(self, position) -> None:
        item = self._tree.itemAt(position)
        if item is not None and item.data(0, _NAME_ROLE) is not None:
            item = item.parent()              # a node row: its level
        menu = self._create_menu(item)
        # Open at the physical cursor: inside a QGraphicsProxyWidget,
        # mapToGlobal ignores the view transform and drifts with zoom.
        menu.exec(QCursor.pos())

    def _create_menu(self, level: Optional[QTreeWidgetItem] = None) -> QMenu:
        """Select All / None / Invert over the nodes shown, and the same
        for the level under the cursor."""
        # Parentless: a QMenu parented to a proxy-embedded widget gets
        # repositioned through the scene transform (zoom drift).
        menu = QMenu()

        def add(text: str, icon: str, slot) -> None:
            action = QAction(text, menu)
            if ic := widget_menus._icon(icon):
                action.setIcon(ic)
            action.triggered.connect(lambda _=False: slot())
            menu.addAction(action)

        add("Select All", "select-all", self.check_all)
        add("Select None", "deselect", self.check_none)
        add("Invert Selection", "arrows-shuffle", self.invert_checked)

        if level is not None:
            name = level.text(0)
            menu.addSeparator()
            add(f"Select All in “{name}”", "select-all",
                lambda: self.check_all(level))
            add(f"Select None in “{name}”", "deselect",
                lambda: self.check_none(level))
            add(f"Invert in “{name}”", "arrows-shuffle",
                lambda: self.invert_checked(level))

        menu.addSeparator()
        add("Expand All", "chevron-down", self._tree.expandAll)
        add("Collapse All", "chevron-up", self._tree.collapseAll)
        return menu

    # ── the filter ───────────────────────────────────────────────────

    def _apply_filter(self, text: str) -> None:
        """Hide the nodes that do not match *text*, and the levels left
        with nothing shown; open what is left while a filter is on."""
        needle = text.strip().casefold()

        def visit(item: QTreeWidgetItem) -> bool:
            name = item.data(0, _NAME_ROLE)
            if name is not None:
                shown = (not needle or needle in item.text(0).casefold()
                         or needle in str(name).casefold())
            else:
                shown = False
                for i in range(item.childCount()):
                    shown = visit(item.child(i)) or shown
                if needle and shown:
                    item.setExpanded(True)
            item.setHidden(not shown)
            return shown

        for i in range(self._tree.topLevelItemCount()):
            visit(self._tree.topLevelItem(i))

    def _refresh_status(self) -> None:
        if not self._checked:
            self._status.setText(EMPTY_HINT)
            return
        text = f"{len(self._checked)} class(es) placeable."
        gone = self.missing()
        if gone:
            text += "  " + MISSING_HINT.format(", ".join(gone))
        self._status.setText(text)


def _node_icons():
    """Browse Nodes' icon provider, or ``None`` -- icons are optional."""
    try:
        from weave.node.icons import get_node_icon_provider
        return get_node_icon_provider()
    except Exception:  # noqa: BLE001 - a tree without icons still works
        return None


def _set_icon(item: QTreeWidgetItem, icons, classes, lookups) -> None:
    """The first icon any of *classes* has, trying *lookups* in turn --
    the menu's rule, so a level shows the icon its submenu does."""
    if icons is None:
        return
    for lookup in lookups:
        getter = getattr(icons, lookup, None)
        if getter is None:
            continue
        for cls in classes:
            try:
                ic = getter(cls)
            except Exception:  # noqa: BLE001 - one bad class costs one icon
                ic = None
            if ic:
                item.setIcon(0, ic)
                return
