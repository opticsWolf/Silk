# -*- coding: utf-8 -*-
"""Which node classes an agent may place (spec §18, D71).

A checkable tree over `NODE_REGISTRY`, grouped by category -- the same
shape the tool tree already uses, because it is the same question asked
about a different registry. The widget value is a plain list of class
names, so it travels in the ToolBox recipe, saves with the graph, and
fits in a preset. Unlike grants (D35) it carries no secret and no
filesystem authority: it says which *kinds of node* may appear on the
canvas, and nothing else.

**The default is empty and there is no "allow all".** Ticking every box
is possible and must be a deliberate act; the safe state is the one you
get by doing nothing (I6). A whitelisted class that is no longer
registered stays in the value and is reported here, rather than becoming
a refusal the agent hits halfway through building something.

The tree re-reads the registry when it changes (`NodeRegistry.generation`
plus its listener), so a hot-loaded plugin's nodes become tickable
without rebuilding the graph.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from weave.widgets.composite_value import CompositeValueWidget

_NAME_ROLE = Qt.ItemDataRole.UserRole

EMPTY_HINT = "Nothing ticked — the agent may not place any node."
MISSING_HINT = "not registered any more: {}"


class NodeWhitelistWidget(CompositeValueWidget):
    """Checkable tree of registered node classes. Value: list of names."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checked: set[str] = set()
        self._reload_pending = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setMaximumHeight(220)
        self._tree.setToolTip(
            "Node classes this ToolBox's agents may place on the canvas. "
            "Empty means none — there is no 'allow all'. A ToolSet or Role "
            "downstream may narrow this list, never widen it."
        )
        layout.addWidget(self._tree)

        self._status = QLabel(EMPTY_HINT)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._tree.itemChanged.connect(self._on_item_changed)
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
        user's choices.
        """
        blocked = self._tree.blockSignals(True)
        try:
            self._tree.clear()
            for category, classes in sorted(self._registry_tree().items()):
                parent = QTreeWidgetItem(self._tree, [category or "Uncategorised"])
                parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                for name, label in sorted(classes):
                    item = QTreeWidgetItem(parent, [label])
                    item.setData(0, _NAME_ROLE, name)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.CheckState.Checked
                                       if name in self._checked
                                       else Qt.CheckState.Unchecked)
                parent.setExpanded(False)
        finally:
            self._tree.blockSignals(blocked)
        self._refresh_status()

    @staticmethod
    def _registry_tree() -> dict:
        """``{category: [(class_name, display_name), …]}`` from the registry."""
        try:
            from weave.registry import NODE_REGISTRY
            from weave.registry.metadata import get_display_name
        except ImportError:      # pragma: no cover - Weave without a registry
            return {}
        grouped: dict[str, list] = {}
        for cls in NODE_REGISTRY.get_all_nodes():
            category = str(getattr(cls, "node_category", "") or "")
            grouped.setdefault(category, []).append(
                (cls.__name__, get_display_name(cls) or cls.__name__))
        return grouped

    def registered_names(self) -> set[str]:
        return {name for classes in self._registry_tree().values()
                for name, _label in classes}

    def missing(self) -> list[str]:
        """Ticked classes the registry does not have (D71)."""
        return sorted(self._checked - self.registered_names())

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
        name = item.data(0, _NAME_ROLE)
        if not name:
            return
        if item.checkState(0) == Qt.CheckState.Checked:
            self._checked.add(str(name))
        else:
            self._checked.discard(str(name))
        self._refresh_status()
        self._notify_value_changed()

    def _refresh_status(self) -> None:
        if not self._checked:
            self._status.setText(EMPTY_HINT)
            return
        text = f"{len(self._checked)} class(es) placeable."
        gone = self.missing()
        if gone:
            text += "  " + MISSING_HINT.format(", ".join(gone))
        self._status.setText(text)
