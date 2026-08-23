# -*- coding: utf-8 -*-
"""Checkable toolchain list for the Toolchain node.

Each row is one configured toolchain: checkbox (enabled) + kind +
executable path. The path cell is edited by **double-clicking** it; the
right-click menu removes entries and offers the standard checkbox-list
bulk actions. Value = ``[{"kind", "executable", "enabled"}, ...]`` —
plain data; probing happens in the node's compute.

The same kind may appear multiple times (e.g. two different venvs);
tool-name collisions are resolved at attach time with numbered suffixes
(``run_python``, ``run_python_2``, …) — see
``functions.tools.toolchains.attach_toolchain_tools``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QObject
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from weave.widgets.composite_value import CompositeValueWidget
from weave.widgetcore import menus as widget_menus
from weave.panel.mirror_contracts import MirrorContract

_KIND_ROLE = Qt.ItemDataRole.UserRole


class ToolchainListWidget(CompositeValueWidget):
    """Checkable list of configured toolchains; value = entry dicts."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Toolchain", "Executable"])
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.setMinimumHeight(90)
        self._tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        # Double-click edits the executable cell only.
        self._tree.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self._tree.setToolTip(
            "Configured toolchains. Checkbox enables/disables; double-click "
            "the executable path to change it; right-click to remove."
        )
        layout.addWidget(self._tree)

        self._tree.itemChanged.connect(self._notify_value_changed)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)

    # -- entries ---------------------------------------------------------

    def entries(self) -> List[Dict[str, Any]]:
        result = []
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            result.append({
                "kind": item.data(0, _KIND_ROLE),
                "executable": item.text(1).strip(),
                "enabled": item.checkState(0) == Qt.CheckState.Checked,
            })
        return result

    def add_entry(self, kind: str, executable: str = "") -> bool:
        """Append a toolchain entry; exact (kind, executable) duplicates
        are skipped. Returns True when added."""
        kind = str(kind).strip()
        executable = str(executable).strip()
        for entry in self.entries():
            if entry["kind"] == kind and entry["executable"] == executable:
                return False
        self._append_item(kind, executable, enabled=True)
        self._notify_value_changed()
        return True

    def _append_item(self, kind: str, executable: str, enabled: bool) -> None:
        self._tree.blockSignals(True)
        item = QTreeWidgetItem(self._tree)
        item.setText(0, kind)
        item.setData(0, _KIND_ROLE, kind)
        item.setText(1, executable or "(auto: PATH)")
        item.setToolTip(1, "Double-click to edit the executable path.")
        item.setFlags(
            item.flags()
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsEditable
        )
        item.setCheckState(
            0, Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked
        )
        self._tree.blockSignals(False)

    # -- context menu ----------------------------------------------------

    def _show_context_menu(self, _position) -> None:
        # Open at the physical cursor: inside a QGraphicsProxyWidget,
        # mapToGlobal ignores the view transform and drifts with zoom.
        from PySide6.QtGui import QCursor
        menu = self._create_menu()
        menu.exec(QCursor.pos())

    def _create_menu(self) -> QMenu:
        # Parentless: a QMenu parented to a proxy-embedded widget gets
        # repositioned through the scene transform (zoom drift).
        menu = QMenu()

        remove = QAction("Remove Selected", menu)
        if ic := widget_menus._icon("minus"):
            remove.setIcon(ic)
        remove.setEnabled(self._tree.currentItem() is not None)
        remove.triggered.connect(self._remove_current)
        menu.addAction(remove)

        menu.addSeparator()

        enable_all = QAction("Enable All", menu)
        if ic := widget_menus._icon("select-all"):
            enable_all.setIcon(ic)
        enable_all.triggered.connect(lambda: self._bulk_set(True))
        menu.addAction(enable_all)

        disable_all = QAction("Disable All", menu)
        if ic := widget_menus._icon("deselect"):
            disable_all.setIcon(ic)
        disable_all.triggered.connect(lambda: self._bulk_set(False))
        menu.addAction(disable_all)

        return menu

    def _remove_current(self) -> None:
        index = self._tree.indexOfTopLevelItem(self._tree.currentItem())
        if index >= 0:
            self._tree.takeTopLevelItem(index)
            self._notify_value_changed()

    def _bulk_set(self, enabled: bool) -> None:
        self._tree.blockSignals(True)
        for i in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(i).setCheckState(
                0, Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked
            )
        self._tree.blockSignals(False)
        self._notify_value_changed()

    # -- CompositeValueWidget contract -----------------------------------

    def _read(self) -> List[Dict[str, Any]]:
        entries = self.entries()
        # "(auto: PATH)" is display sugar, not a path.
        for entry in entries:
            if entry["executable"] == "(auto: PATH)":
                entry["executable"] = ""
        return entries

    def _write(self, value: Any) -> None:
        if not isinstance(value, (list, tuple)):
            value = []
        self._tree.blockSignals(True)
        self._tree.clear()
        self._tree.blockSignals(False)
        for entry in value:
            if isinstance(entry, dict) and entry.get("kind"):
                self._append_item(
                    str(entry["kind"]),
                    str(entry.get("executable") or ""),
                    bool(entry.get("enabled", True)),
                )

    def _internal_signal_sources(self) -> List[QObject]:
        return [self._tree]

    __mirror__ = MirrorContract(
        clone=lambda src, _b: ToolchainListWidget()
    )


# ── Canvas Context Menu Registration ──

def _build_toolchain_list_menu(widget: ToolchainListWidget, binding, canvas) -> Optional[QMenu]:
    return widget._create_menu()


widget_menus.register(ToolchainListWidget, _build_toolchain_list_menu)
