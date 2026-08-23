# -*- coding: utf-8 -*-
"""Tool catalog widgets for the silk node suite.

:class:`ToolTreeWidget`
    Category-grouped tree of a tool catalog. In *checkable* mode each
    tool row carries a checkbox and category rows act as tri-state
    quick-selectors (checking a category checks every tool in it); the
    widget value is the sorted list of checked tool names. In read-only
    mode it is a plain browsable overview.

:class:`ToolDetailWidget`
    Structured key/value preview of a single catalog entry (category,
    risk, tags, description, parameter schema). Driven by the tree's
    ``tool_focused`` signal.

Catalog entries are the plain dicts produced by
``functions.toolset_build.tool_catalog``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHeaderView,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from weave.widgets.composite_value import CompositeValueWidget
from weave.widgetcore import menus as widget_menus
from weave.panel.mirror_contracts import MirrorContract

_NAME_ROLE = Qt.ItemDataRole.UserRole
_ENTRY_ROLE = Qt.ItemDataRole.UserRole + 1


class ToolTreeWidget(CompositeValueWidget):
    """Category-grouped tool tree; value = sorted checked tool names."""

    #: Emits the focused catalog entry dict, or None when focus clears.
    tool_focused = Signal(object)
    #: Display-state notify for panel mirroring (catalog rows).
    catalog_changed = Signal()

    def __init__(self, parent=None, checkable: bool = True) -> None:
        super().__init__(parent)
        self._checkable = checkable
        self._catalog: List[Dict[str, Any]] = []
        # Checked names are kept even while absent from the catalog, so a
        # transient upstream disconnect (empty catalog) never wipes the
        # user's selection; names simply re-appear ticked on reconnect.
        self._checked: set[str] = set()
        self._available: set[str] = set()

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Tool", "Risk"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setMinimumHeight(140)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.layout().addWidget(self._tree)

        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.currentItemChanged.connect(self._on_current_changed)

        # Own the right-click: checkable mode shows the bulk-selection
        # menu; read-only mode swallows the event (no widget menu, and —
        # via the registered builder returning None — no node menu either).
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)

    # -- context menu ----------------------------------------------------

    def _show_context_menu(self, _position) -> None:
        menu = self._create_menu()
        if menu is not None:
            # Open at the physical cursor: inside a QGraphicsProxyWidget,
            # mapToGlobal ignores the view transform and drifts with zoom.
            from PySide6.QtGui import QCursor
            menu.exec(QCursor.pos())

    def _create_menu(self) -> Optional[QMenu]:
        """Bulk-selection menu (checkable mode only; None = no menu)."""
        if not self._checkable:
            return None
        # Parentless: a QMenu parented to a proxy-embedded widget gets
        # repositioned through the scene transform (zoom drift).
        menu = QMenu()

        select_all = QAction("Select All", menu)
        if ic := widget_menus._icon("select-all"):
            select_all.setIcon(ic)
        select_all.triggered.connect(self.check_all)
        menu.addAction(select_all)

        deselect_all = QAction("Deselect All", menu)
        if ic := widget_menus._icon("deselect"):
            deselect_all.setIcon(ic)
        deselect_all.triggered.connect(self.check_none)
        menu.addAction(deselect_all)

        menu.addSeparator()

        invert = QAction("Invert Selection", menu)
        if ic := widget_menus._icon("arrows-shuffle"):
            invert.setIcon(ic)
        invert.triggered.connect(self.invert_checked)
        menu.addAction(invert)

        return menu

    # -- population ------------------------------------------------------

    def set_catalog(self, catalog: List[Dict[str, Any]]) -> None:
        """Rebuild the tree from *catalog*, preserving surviving checks."""
        new_catalog = list(catalog or [])

        # Prevent rebuilding the tree if the catalog hasn't changed.
        # Rebuilding on every eval destroys the current item and breaks
        # checkbox state updates when clicking rapidly.
        if self._catalog == new_catalog:
            return

        self._catalog = new_catalog
        self._available = {entry["name"] for entry in self._catalog}

        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for entry in self._catalog:
            by_category.setdefault(entry.get("category", "uncategorized"), []).append(entry)

        self._tree.blockSignals(True)
        self._tree.clear()
        for category in sorted(by_category):
            entries = by_category[category]
            parent = QTreeWidgetItem(self._tree)
            parent.setText(0, f"{category}  ({len(entries)})")
            parent.setData(0, _NAME_ROLE, None)
            parent.setFirstColumnSpanned(True)
            if self._checkable:
                parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            for entry in entries:
                child = QTreeWidgetItem(parent)
                child.setText(0, entry["name"])
                child.setText(1, entry.get("risk", "low"))
                child.setToolTip(0, entry.get("description", ""))
                child.setData(0, _NAME_ROLE, entry["name"])
                child.setData(0, _ENTRY_ROLE, entry)
                if self._checkable:
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child.setCheckState(
                        0,
                        Qt.CheckState.Checked
                        if entry["name"] in self._checked
                        else Qt.CheckState.Unchecked,
                    )
            if self._checkable:
                self._sync_category_state(parent)
            parent.setExpanded(True)
        self._tree.blockSignals(False)
        # No value notification here: the checked set is untouched by a
        # catalog rebuild (set_catalog runs on every upstream evaluation;
        # notifying would re-dirty the node each time). The display-state
        # signal keeps panel mirrors in sync instead.
        self.catalog_changed.emit()

    def set_category_filter(self, category: Optional[str]) -> None:
        """Show only *category* (falsy = show all)."""
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            label = item.text(0).rsplit("  (", 1)[0]
            item.setHidden(bool(category) and label != category)

    def categories(self) -> List[str]:
        return sorted({e.get("category", "uncategorized") for e in self._catalog})

    def catalog(self) -> List[Dict[str, Any]]:
        return list(self._catalog)

    #: camelCase writer alias — the mirror engine resolves display
    #: properties as reader ``catalog()`` / writer ``setCatalog()``.
    def setCatalog(self, catalog: List[Dict[str, Any]]) -> None:  # noqa: N802
        self.set_catalog(catalog)

    # -- interaction -----------------------------------------------------

    def _on_current_changed(
        self, current: Optional[QTreeWidgetItem], _previous
    ) -> None:
        entry = current.data(0, _ENTRY_ROLE) if current is not None else None
        self.tool_focused.emit(entry)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if not self._checkable or column != 0:
            return
        self._tree.blockSignals(True)
        try:
            if item.data(0, _NAME_ROLE) is None:
                # Category row: quick-select every tool beneath it.
                state = item.checkState(0)
                target = (
                    Qt.CheckState.Checked
                    if state != Qt.CheckState.Unchecked
                    else Qt.CheckState.Unchecked
                )
                item.setCheckState(0, target)
                for i in range(item.childCount()):
                    item.child(i).setCheckState(0, target)
            else:
                self._sync_category_state(item.parent())
        finally:
            self._tree.blockSignals(False)
        self._recollect_checked()
        self._notify_value_changed()

    def _sync_category_state(self, parent: Optional[QTreeWidgetItem]) -> None:
        if parent is None:
            return
        checked = sum(
            parent.child(i).checkState(0) == Qt.CheckState.Checked
            for i in range(parent.childCount())
        )
        if checked == 0:
            parent.setCheckState(0, Qt.CheckState.Unchecked)
        elif checked == parent.childCount():
            parent.setCheckState(0, Qt.CheckState.Checked)
        else:
            parent.setCheckState(0, Qt.CheckState.PartiallyChecked)

    def _recollect_checked(self) -> None:
        # Names outside the current catalog keep their remembered state;
        # only the visible portion is re-read from the tree.
        self._checked -= self._available
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) == Qt.CheckState.Checked:
                    self._checked.add(child.data(0, _NAME_ROLE))

    # -- bulk helpers ----------------------------------------------------

    def check_all(self) -> None:
        self._write([e["name"] for e in self._catalog])
        self._notify_value_changed()

    def check_none(self) -> None:
        self._write([])
        self._notify_value_changed()

    def invert_checked(self) -> None:
        self._write([n for n in self._available if n not in self._checked])
        self._notify_value_changed()

    def focus_tool(self, name: Optional[str]) -> None:
        """Select the row of tool *name* (None clears the selection).

        Used by the mirror action proxy: focusing a tool in a panel
        mirror re-focuses it here, so the source emits ``tool_focused``
        and the node-wired detail view (and its mirror) follow.
        """
        if not name:
            self._tree.clearSelection()
            self._tree.setCurrentItem(None)
            return
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.data(0, _NAME_ROLE) == name:
                    self._tree.setCurrentItem(child)
                    return

    # -- CompositeValueWidget contract -----------------------------------

    def _read(self) -> List[str]:
        return sorted(self._checked)

    def _write(self, value: Any) -> None:
        if not isinstance(value, (list, tuple, set)):
            value = []
        self._checked = {str(v) for v in value}
        self._tree.blockSignals(True)
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if self._checkable:
                    child.setCheckState(
                        0,
                        Qt.CheckState.Checked
                        if child.data(0, _NAME_ROLE) in self._checked
                        else Qt.CheckState.Unchecked,
                    )
            if self._checkable:
                self._sync_category_state(parent)
        self._tree.blockSignals(False)

    def _internal_signal_sources(self) -> List[QObject]:
        return [self._tree]

    __mirror__ = MirrorContract(
        clone=lambda src, _b: ToolTreeWidget(checkable=src._checkable),
        # Catalog rows are display state outside the value — live-bound
        # into panel mirrors via reader catalog() / writer setCatalog().
        display_properties=(("catalog", "catalog_changed"),),
        # Focusing a tool in a mirror re-focuses it on the source, so the
        # node-wired detail preview (and its own mirror) follow along.
        action_signal="tool_focused",
        action_invoker=lambda src, args: src.focus_tool(
            args[0].get("name") if args and isinstance(args[0], dict) else None
        ),
    )


class ToolDetailWidget(CompositeValueWidget):
    """Structured key/value preview of one tool catalog entry.

    A CompositeValueWidget (with a no-op value) so the canvas resolves
    it as a binding — the registered menu builder returns ``None``,
    which suppresses both the widget menu and the node menu over this
    purely informational view.
    """

    #: Display-state notify for panel mirroring (the shown entry).
    shown_tool_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._shown: Optional[Dict[str, Any]] = None
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Field", "Value"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setWordWrap(True)
        self._tree.setMinimumHeight(110)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        # Swallow local right-clicks: no menu on a read-only detail view.
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layout().addWidget(self._tree)
        self.show_tool(None)

    # -- CompositeValueWidget contract (informational — no value) --------

    def _read(self) -> str:
        return ""

    def _write(self, value: Any) -> None:
        pass

    def _internal_signal_sources(self) -> List[QObject]:
        return []

    __mirror__ = MirrorContract(
        clone=lambda src, _b: ToolDetailWidget(),
        # The shown entry is display state — live-bound into mirrors via
        # reader shownTool() / writer setShownTool().
        display_properties=(("shownTool", "shown_tool_changed"),),
    )

    # -- display-state accessors (panel mirroring) -----------------------

    def shownTool(self) -> Optional[Dict[str, Any]]:  # noqa: N802
        return self._shown

    def setShownTool(self, entry: Optional[Dict[str, Any]]) -> None:  # noqa: N802
        self.show_tool(entry)

    # -- content ---------------------------------------------------------

    def show_tool(self, entry: Optional[Dict[str, Any]]) -> None:
        self._shown = dict(entry) if isinstance(entry, dict) else None
        self.shown_tool_changed.emit()
        self._tree.clear()
        if not entry:
            placeholder = QTreeWidgetItem(self._tree)
            placeholder.setText(0, "—")
            placeholder.setText(1, "Select a tool to inspect it.")
            return

        def row(field: str, value: str, parent=None) -> QTreeWidgetItem:
            item = QTreeWidgetItem(parent or self._tree)
            item.setText(0, field)
            item.setText(1, value)
            item.setToolTip(1, value)
            return item

        row("Name", entry.get("name", ""))
        row("Category", entry.get("category", ""))
        row("Risk", entry.get("risk", "low"))
        row("Tags", ", ".join(entry.get("tags") or ()) or "—")
        row("Description", entry.get("description", "") or "—")

        parameters = (entry.get("parameters") or {}).get("properties") or {}
        required = set((entry.get("parameters") or {}).get("required") or ())
        params_root = row("Parameters", str(len(parameters)) if parameters else "none")
        for param_name, schema in parameters.items():
            kind = schema.get("type", "any")
            suffix = "required" if param_name in required else "optional"
            detail = schema.get("description", "")
            child = row(param_name, f"{kind} — {suffix}", parent=params_root)
            if detail:
                child.setToolTip(1, f"{kind} — {suffix}\n{detail}")
        params_root.setExpanded(True)

# ── Canvas Context Menu Registration ──
#
# ToolTreeWidget: checkable mode gets the bulk-selection menu; read-only
# overview mode returns None — which suppresses the node menu too (the
# canvas shows nothing for a binding whose builder yields None).
# ToolDetailWidget: always suppressed (purely informational).


def _build_tool_tree_menu(widget: ToolTreeWidget, binding, canvas):
    return widget._create_menu()


def _build_tool_detail_menu(widget: ToolDetailWidget, binding, canvas):
    return None


widget_menus.register(ToolTreeWidget, _build_tool_tree_menu)
widget_menus.register(ToolDetailWidget, _build_tool_detail_menu)
