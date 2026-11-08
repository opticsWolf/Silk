# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Hook catalog selector for the silk nodes.

A compact checkbox list over the named hook catalog
(:mod:`..functions.hook_catalog`). The widget value is
``{"names": [...], "configs": {name: {...}}}`` — checked hook names plus
per-hook config values (edited via a pydantic-generated dialog on
double-click). Pure data, preset-safe; callables are only materialised
at build/activation time by the consuming node.

Transparency: :meth:`set_inherited` marks hooks that are already active
one layer up (infrastructure hooks from the ToolBox, shown in the Role
node). The marker is state-aware — an *unchecked* inherited hook reads
"already on ToolBox", a *checked* one flips to an explicit stacking
notice ("runs twice"), so double-booking is a visible choice, never a
surprise.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QMenu, QVBoxLayout

from weave.widgets.composite_value import CompositeValueWidget
from weave.widgetcore import menus as widget_menus
from weave.panel.mirror_contracts import MirrorContract

from ..functions.hook_catalog import HOOK_CATALOG, resolve_config
from .config_dialog import PydanticConfigDialog

_NAME_ROLE = Qt.ItemDataRole.UserRole


class HookSelectWidget(CompositeValueWidget):
    """Checkbox list of catalog hooks with per-hook config editing."""

    #: Display-state notify for panel mirroring (inherited marking).
    inherited_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._inherited: set[str] = set()
        self._configs: Dict[str, dict] = {}
        #: Role-layer rules (set_role_layer): hooks this layer cannot use,
        #: and hooks whose tick here tightens/adjusts the ToolBox's.
        self._unavailable: set[str] = set()
        self._overlays: set[str] = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget()
        self._list.setToolTip(
            "Lifecycle hooks from the catalog. Presets store only names "
            "and config values — behavior always comes from the "
            "installed catalog. Double-click a ⚙ hook to configure it."
        )
        # No height cap: the list takes whatever height its row gets (a
        # capped list sits centred in a taller widget and still scrolls,
        # its rows being taller than a guess at them). A few rows at least.
        self._list.setMinimumHeight(70)
        layout.addWidget(self._list)

        for name in sorted(HOOK_CATALOG):
            item = QListWidgetItem(name)
            item.setData(_NAME_ROLE, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            # Configurable hooks carry the settings icon (widget_icons).
            if HOOK_CATALOG[name].config_model is not None:
                if ic := widget_menus._icon("settings"):
                    item.setIcon(ic)
            self._list.addItem(item)
            self._refresh_item(item)

        self._list.itemChanged.connect(self._on_item_changed)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)

    # -- display -----------------------------------------------------------

    def _refresh_item(self, item: QListWidgetItem) -> None:
        """Render name + config marker + inheritance/stacking notice."""
        name = item.data(_NAME_ROLE)
        spec = HOOK_CATALOG.get(name)
        if spec is None:
            return
        checked = item.checkState() == Qt.CheckState.Checked
        inherited = name in self._inherited
        unavailable = name in self._unavailable
        overlay = name in self._overlays

        text = name
        if unavailable:
            text += "  — ToolBox only"
        elif overlay and checked:
            text += ("  — tightens the ToolBox's for this role"
                     if name != "auto_compact"
                     else "  — this role's settings replace the ToolBox's")
        elif inherited and checked:
            text += "  — ×2: ToolBox + this role (runs twice)"
        elif inherited:
            text += "  — already on ToolBox"

        tooltip = spec.description
        if unavailable:
            tooltip += (
                "\n\nOnly the ToolBox can wire this hook: it works on the "
                "box's folder, memory or disk, not on who uses it. Tick it "
                "on the ToolBox."
            )
        elif overlay:
            tooltip += (
                "\n\nOn a Role this layers on the ToolBox's: "
                + ("the gate only ever asks more, never less."
                   if name != "auto_compact"
                   else "its settings replace the box's for this role's runs.")
            )
        if spec.config_model is not None:
            config = resolve_config(spec, self._configs.get(name))
            tooltip += "\n\nDouble-click to configure. Current settings:"
            for key, value in config.model_dump().items():
                tooltip += f"\n  {key} = {value!r}"
        if inherited and checked:
            tooltip += (
                "\n\nThis hook is active on the ToolBox AND selected for "
                "this role: two independent instances run per tool call "
                "(e.g. double log lines). Untick it here if one is enough."
            )
        elif inherited:
            tooltip += (
                "\n\nAlready active as an infrastructure hook on the "
                "connected ToolBox/ToolSet. Checking it here adds a "
                "second, independent instance."
            )

        font = item.font()
        font.setItalic((inherited and not checked) or unavailable)
        font.setBold(inherited and checked and not overlay)
        # Text mutation fires itemChanged — suppress while rendering.
        self._list.blockSignals(True)
        flags = item.flags()
        if unavailable:
            flags &= ~Qt.ItemFlag.ItemIsEnabled
        else:
            flags |= Qt.ItemFlag.ItemIsEnabled
        item.setFlags(flags)
        item.setText(text)
        item.setToolTip(tooltip)
        item.setFont(font)
        self._list.blockSignals(False)

    def _refresh_all(self) -> None:
        for i in range(self._list.count()):
            self._refresh_item(self._list.item(i))

    def set_inherited(self, names: Iterable[str]) -> None:
        """Mark hooks already active a layer up (infrastructure).

        Display-only: check state and value are untouched — checking a
        marked hook deliberately stacks a second, independent instance,
        and the row then says exactly that.
        """
        inherited = {str(n) for n in (names or ())}
        if inherited == self._inherited:
            return
        self._inherited = inherited
        self._refresh_all()
        self.inherited_changed.emit()

    def set_role_layer(self, unavailable: Iterable[str],
                       overlays: Iterable[str]) -> None:
        """Mark this list as a Role's: *unavailable* hooks are greyed out
        and cannot be ticked here; *overlays* say they layer on the
        ToolBox's rather than run a second time. Display-only."""
        self._unavailable = {str(n) for n in unavailable}
        self._overlays = {str(n) for n in overlays}
        self._refresh_all()

    # -- display-state accessors (panel mirroring) -------------------------

    def inheritedNames(self) -> List[str]:  # noqa: N802
        return sorted(self._inherited)

    def setInheritedNames(self, names: Iterable[str]) -> None:  # noqa: N802
        self.set_inherited(names)

    # -- context menu ------------------------------------------------------

    def _show_context_menu(self, _position) -> None:
        # Open at the physical cursor: inside a QGraphicsProxyWidget,
        # mapToGlobal ignores the view transform and drifts with zoom.
        from PySide6.QtGui import QCursor
        menu = self._create_menu()
        menu.exec(QCursor.pos())

    def _create_menu(self) -> QMenu:
        """Standard checkbox-list bulk actions.

        Parentless: a QMenu parented to a proxy-embedded widget gets
        repositioned through the scene transform (zoom drift).
        """
        menu = QMenu()

        check_all = QAction("Check All", menu)
        if ic := widget_menus._icon("select-all"):
            check_all.setIcon(ic)
        check_all.triggered.connect(lambda: self._bulk_set("all"))
        menu.addAction(check_all)

        uncheck_all = QAction("Uncheck All", menu)
        if ic := widget_menus._icon("deselect"):
            uncheck_all.setIcon(ic)
        uncheck_all.triggered.connect(lambda: self._bulk_set("none"))
        menu.addAction(uncheck_all)

        menu.addSeparator()

        invert = QAction("Invert Selection", menu)
        if ic := widget_menus._icon("arrows-shuffle"):
            invert.setIcon(ic)
        invert.triggered.connect(lambda: self._bulk_set("invert"))
        menu.addAction(invert)

        return menu

    def _bulk_set(self, mode: str) -> None:
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            item = self._list.item(i)
            checked = item.checkState() == Qt.CheckState.Checked
            target = {"all": True, "none": False, "invert": not checked}[mode]
            item.setCheckState(
                Qt.CheckState.Checked if target else Qt.CheckState.Unchecked
            )
        self._list.blockSignals(False)
        self._refresh_all()
        self._notify_value_changed()

    # -- interaction -------------------------------------------------------

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        # Check-state flips change the stacking notice text too.
        self._refresh_item(item)
        self._notify_value_changed()

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        name = item.data(_NAME_ROLE)
        spec = HOOK_CATALOG.get(name)
        if spec is None or spec.config_model is None:
            return
        # Parent None: inside a QGraphicsProxyWidget a widget parent
        # would spawn a ghost window (same fix as PathPickerWidget).
        dialog = PydanticConfigDialog(
            spec.config_model,
            values=self._configs.get(name),
            title=f"Configure hook: {name}",
            parent=None,
        )
        if dialog.exec() and dialog.result_values is not None:
            self._configs[name] = dialog.result_values
            self._refresh_item(item)
            self._notify_value_changed()

    # -- CompositeValueWidget contract -----------------------------------

    def _read(self) -> Dict[str, Any]:
        names = sorted(
            self._list.item(i).data(_NAME_ROLE)
            for i in range(self._list.count())
            if self._list.item(i).checkState() == Qt.CheckState.Checked
        )
        return {"names": names, "configs": dict(self._configs)}

    def _write(self, value: Any) -> None:
        # Accept the dict shape; tolerate a legacy plain name list.
        if isinstance(value, dict):
            names = value.get("names") or []
            self._configs = {
                str(k): dict(v)
                for k, v in (value.get("configs") or {}).items()
                if isinstance(v, dict)
            }
        elif isinstance(value, (list, tuple, set)):
            names = list(value)
            self._configs = {}
        else:
            names, self._configs = [], {}
        wanted = {str(v) for v in names}
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setCheckState(
                Qt.CheckState.Checked if item.data(_NAME_ROLE) in wanted
                else Qt.CheckState.Unchecked
            )
        self._refresh_all()

    def _internal_signal_sources(self) -> List[QObject]:
        return [self._list]

    __mirror__ = MirrorContract(
        clone=lambda src, _b: HookSelectWidget(),
        # The inherited marking ("already on ToolBox" / "×2 runs twice")
        # is display state — live-bound into panel mirrors.
        display_properties=(("inheritedNames", "inherited_changed"),),
    )


# ── Canvas Context Menu Registration ──

def _build_hook_select_menu(widget: HookSelectWidget, binding, canvas) -> Optional[QMenu]:
    return widget._create_menu()


widget_menus.register(HookSelectWidget, _build_hook_select_menu)
