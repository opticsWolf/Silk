# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Tool catalog widgets for the silk node suite.

:class:`ToolTreeWidget`
    Category-grouped tree of a tool catalog. In *checkable* mode each
    tool row carries a checkbox and category rows act as tri-state
    quick-selectors (checking a category checks every tool in it); the
    widget value is the sorted list of checked tool names. In read-only
    mode it is a plain browsable overview.

:class:`CategoryCombo`
    The tree's category filter: "All categories" plus one entry per
    category the tree offers, refilled whenever its catalog changes.

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
from PySide6.QtGui import QAction, QBrush, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QHeaderView,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from weave.widgets.composite_value import CompositeValueWidget
from weave.widgetcore import menus as widget_menus
from weave.panel.mirror_contracts import MirrorContract

from ..functions.role import RISK_ORDER

#: The category filter's entry that shows every category.
ALL_CATEGORIES = "All categories"

_NAME_ROLE = Qt.ItemDataRole.UserRole
_ENTRY_ROLE = Qt.ItemDataRole.UserRole + 1


class ToolTreeWidget(CompositeValueWidget):
    """Category-grouped tool tree; value = sorted checked tool names."""

    #: Emits the focused catalog entry dict, or None when focus clears.
    tool_focused = Signal(object)
    #: Display-state notify for panel mirroring (catalog rows).
    catalog_changed = Signal()
    #: A configurable category row was asked to open its settings.
    config_requested = Signal(str)

    def __init__(self, parent=None, checkable: bool = True) -> None:
        super().__init__(parent)
        self._checkable = checkable
        self._catalog: List[Dict[str, Any]] = []
        # {category: menu label} for the rows that own configuration.
        # Some tools are governed by settings that are not preferences --
        # graph authoring's whitelist *is* its grant -- and those belong
        # beside the tools they govern, not in a field elsewhere.
        self._configurable: Dict[str, str] = {}
        # Checked names are kept even while absent from the catalog, so a
        # transient upstream disconnect (empty catalog) never wipes the
        # user's selection; names simply re-appear ticked on reconnect.
        self._checked: set[str] = set()
        self._available: set[str] = set()
        # Locks, display only: the user's own ticks stay underneath and
        # come back when a lock is lifted. Allow All shows every tool
        # ticked; a risk ceiling (a role's Max Risk) shows the tools
        # above it unticked. The ceiling wins: it refuses a tool even
        # under Allow All.
        self._all_granted = False
        self._risk_ceiling: Optional[str] = None

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
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)

        # Own the right-click: checkable mode shows the bulk-selection
        # menu; read-only mode swallows the event (no widget menu, and —
        # via the registered builder returning None — no node menu either).
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)

    # -- configurable categories -----------------------------------------

    def set_configurable(self, categories: Dict[str, str]) -> None:
        """Mark *categories* (``{name: label}``) as carrying settings."""
        self._configurable = dict(categories or {})
        self._refresh_category_markers()

    @staticmethod
    def _category_of(item: QTreeWidgetItem) -> str:
        """The category name of a top-level row, without its count."""
        return item.text(0).rsplit("  (", 1)[0]

    def _refresh_category_markers(self) -> None:
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            configurable = self._category_of(item) in self._configurable
            if configurable:
                if ic := widget_menus._icon("settings"):
                    item.setIcon(0, ic)
                item.setToolTip(0, "Double-click to configure this group.")
            else:
                item.setToolTip(0, "")

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column) -> None:
        if item.data(0, _NAME_ROLE) is not None:
            return                      # a tool row, not a category
        category = self._category_of(item)
        if category in self._configurable:
            self.config_requested.emit(category)

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

        item = self._tree.currentItem()
        if item is not None and item.data(0, _NAME_ROLE) is None:
            category = self._category_of(item)
            label = self._configurable.get(category)
            if label:
                menu.addSeparator()
                configure = QAction(label, menu)
                if ic := widget_menus._icon("settings"):
                    configure.setIcon(ic)
                configure.triggered.connect(
                    lambda _=False, c=category: self.config_requested.emit(c)
                )
                menu.addAction(configure)

        menu.addSeparator()

        invert = QAction("Invert Selection", menu)
        if ic := widget_menus._icon("arrows-shuffle"):
            invert.setIcon(ic)
        invert.triggered.connect(self.invert_checked)
        menu.addAction(invert)

        # Under Allow All the ticks are not what counts; editing them
        # unseen would only surprise when it is lifted.
        for action in (select_all, deselect_all, invert):
            action.setEnabled(not self._all_granted)

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
                child.setData(0, _NAME_ROLE, entry["name"])
                child.setData(0, _ENTRY_ROLE, entry)
            self._paint_category(parent)
            parent.setExpanded(True)
        self._refresh_category_markers()
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

    def tool_names(self) -> List[str]:
        """Every tool the tree is currently offering, ticked or not."""
        return [str(e["name"]) for e in self._catalog]

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
                # Category row: tick every tool beneath it the user can
                # tick, or untick them if they all are. Decided from the
                # tools, not the row's new state: with locked tools in it
                # the row is part-ticked, and Qt turns a part-ticked row
                # ticked again on every click -- it could never untick.
                free = [item.child(i) for i in range(item.childCount())
                        if not self._locked(item.child(i))]
                target = (
                    Qt.CheckState.Unchecked
                    if free and all(c.checkState(0) == Qt.CheckState.Checked
                                    for c in free)
                    else Qt.CheckState.Checked
                )
                for child in free:
                    child.setCheckState(0, target)
                self._sync_category_state(item)
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
        # Names outside the current catalog keep their remembered state,
        # and so do locked rows (they show the lock, not the tick); only
        # the rows the user can tick are re-read from the tree.
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if self._locked(child):
                    continue
                name = child.data(0, _NAME_ROLE)
                if child.checkState(0) == Qt.CheckState.Checked:
                    self._checked.add(name)
                else:
                    self._checked.discard(name)

    # -- locks -----------------------------------------------------------

    def _lock_reason(self, entry: Optional[Dict[str, Any]]) -> str:
        """Why a tool's tick is not the user's to give, or ``""``."""
        if not entry:
            return ""
        ceiling = self._risk_ceiling
        risk = entry.get("risk", "low")
        if ceiling and RISK_ORDER.get(risk, 0) > RISK_ORDER.get(ceiling, 0):
            return (f"Above the Max Risk ceiling ({ceiling}): never "
                    f"granted, ticked or not.")
        if self._all_granted:
            return "Allow All grants every tool."
        return ""

    def _locked(self, child: QTreeWidgetItem) -> bool:
        return bool(self._lock_reason(child.data(0, _ENTRY_ROLE)))

    def _paint_tool(self, child: QTreeWidgetItem) -> None:
        """A tool row's tick, lock and tooltip."""
        entry = child.data(0, _ENTRY_ROLE) or {}
        reason = self._lock_reason(entry)
        description = entry.get("description", "")
        child.setToolTip(
            0, f"{reason}\n\n{description}".strip() if reason else description)
        if not self._checkable:
            return
        flags = child.flags() | Qt.ItemFlag.ItemIsUserCheckable
        if reason:
            flags &= ~Qt.ItemFlag.ItemIsUserCheckable
        child.setFlags(flags)
        if reason.startswith("Above"):
            shown = False
        elif reason:
            shown = True
        else:
            shown = child.data(0, _NAME_ROLE) in self._checked
        child.setCheckState(
            0, Qt.CheckState.Checked if shown else Qt.CheckState.Unchecked)
        # Greyed, not disabled: a locked tool can still be selected to
        # read its details.
        for column in range(child.columnCount()):
            child.setData(column, Qt.ItemDataRole.ForegroundRole,
                          self._locked_brush() if reason else None)

    def _locked_brush(self) -> QBrush:
        return QBrush(self._tree.palette().color(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text))

    def _paint_category(self, parent: QTreeWidgetItem) -> None:
        for j in range(parent.childCount()):
            self._paint_tool(parent.child(j))
        if not self._checkable:
            return
        free = any(not self._locked(parent.child(j))
                   for j in range(parent.childCount()))
        flags = parent.flags() | Qt.ItemFlag.ItemIsUserCheckable
        if not free:
            flags &= ~Qt.ItemFlag.ItemIsUserCheckable
        parent.setFlags(flags)
        self._sync_category_state(parent)

    # -- bulk helpers ----------------------------------------------------

    def set_all_granted(self, granted: bool) -> None:
        """Show every tool ticked and locked, or lift that.

        The value (the user's own ticks) is untouched either way. The
        rows stay selectable (details) and the category gears usable.
        """
        self._all_granted = bool(granted)
        self._paint_checks()

    def all_granted(self) -> bool:
        return self._all_granted

    def set_risk_ceiling(self, ceiling: Optional[str]) -> None:
        """Show tools above *ceiling* (a ``RISK_ORDER`` level, or None
        for none) unticked and locked; the user's ticks stay underneath.

        A role's Max Risk: those tools are refused whatever is ticked,
        so the tree says so instead of showing them granted.
        """
        self._risk_ceiling = ceiling or None
        self._paint_checks()

    def risk_ceiling(self) -> Optional[str]:
        return self._risk_ceiling

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
        self._paint_checks()

    def _paint_checks(self) -> None:
        self._tree.blockSignals(True)
        for i in range(self._tree.topLevelItemCount()):
            self._paint_category(self._tree.topLevelItem(i))
        self._tree.blockSignals(False)

    def _internal_signal_sources(self) -> List[QObject]:
        return [self._tree]

    __mirror__ = MirrorContract(
        clone=lambda src, _b: _cloned_tree(src),
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


class CategoryCombo(QComboBox):
    """Filters a :class:`ToolTreeWidget` to one category.

    Its entries follow the tree's catalog; a pick that is still offered
    after a refill stays picked. Built bare (a panel mirror builds it
    so) it is a plain combo until :meth:`bind` gives it a tree.
    """

    def __init__(self, tree: Optional[ToolTreeWidget] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._tree: Optional[ToolTreeWidget] = None
        self.addItem(ALL_CATEGORIES)
        self.setToolTip("Filter the tool tree to one category.")
        self.currentTextChanged.connect(self._apply)
        if tree is not None:
            self.bind(tree)

    def bind(self, tree: ToolTreeWidget) -> None:
        self._tree = tree
        tree.catalog_changed.connect(self.refill)
        self.refill()

    def refill(self) -> None:
        """One entry per category the tree offers; the pick kept."""
        if self._tree is None:
            return
        current = self.currentText()
        self.blockSignals(True)
        self.clear()
        self.addItem(ALL_CATEGORIES)
        self.addItems(self._tree.categories())
        index = self.findText(current)
        self.setCurrentIndex(index if index >= 0 else 0)
        self.blockSignals(False)
        # A rebuilt tree shows every category again.
        self._apply(self.currentText())

    def _apply(self, text: str) -> None:
        if self._tree is not None:
            self._tree.set_category_filter(
                "" if text == ALL_CATEGORIES else text)


def _cloned_tree(source: ToolTreeWidget) -> ToolTreeWidget:
    """A mirror of *source*, configurable rows and all."""
    clone = ToolTreeWidget(checkable=source._checkable)
    clone.set_configurable(dict(source._configurable))
    return clone


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
