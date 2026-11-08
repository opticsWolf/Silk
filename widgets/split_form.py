# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

A node body in groups the user can resize.

The tool nodes (ToolBox, Role, ToolSet, MCP Aggregator) are a few
groups of rows -- settings, a tool tree, the selected tool's details --
and which one wants the room depends on what is being done: picking
tools wants a tall tree, reading a tool wants a tall detail. Each group
is a form of its own, stacked in a vertical :class:`WeaveSplitter`, so
the gutter between two groups is a handle.

The splitter restacks: a node kept short and dragged wide enough for
the groups to stand side by side at their preferred widths turns them
side by side, and back when it is narrowed or made tall again.

A group's room goes to the widgets marked with :meth:`SplitForm.fill`
(a tree, an editor, a detail view): a form gives a row extra height
only when its widget asks to grow, and a pane whose rows all keep their
height would leave the space the user dragged it to empty.

:class:`SplitFormState` keeps the pane sizes with the node: WidgetCore
saves the widgets' values, not where a splitter stood.
"""

from typing import Any, Dict, List, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from weave.widgets.weave_splitter import WeaveSplitter

#: The key the pane sizes are saved under in the node's state.
STATE_KEY = "splitter_sizes"


class SplitForm:
    """*count* forms in a vertical splitter; ``layout`` is what the
    node's WidgetCore lays out (the splitter, edge to edge)."""

    def __init__(self, count: int, stretch: Optional[Sequence[int]] = None) -> None:
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(0)
        self.splitter = WeaveSplitter(Qt.Orientation.Vertical, restack=True)
        # A group dragged to nothing is a group nobody can find again.
        self.splitter.setChildrenCollapsible(False)
        self.forms: List[QFormLayout] = []
        for index in range(count):
            pane = QWidget()
            form = QFormLayout(pane)
            form.setContentsMargins(0, 2, 0, 2)
            form.setSpacing(4)
            self.splitter.addWidget(pane)
            self.splitter.setStretchFactor(
                index, int(stretch[index]) if stretch else 1)
            self.forms.append(form)
        self.layout.addWidget(self.splitter)

    @staticmethod
    def add_above(form: QFormLayout, label: str, widget: QWidget) -> QLabel:
        """A row with *label* over *widget*, the widget full width.

        For the tall widgets (hooks, details): beside them, a label
        takes a column of width they could use, and sits mid-height.
        """
        caption = QLabel(label)
        form.addRow(caption)
        form.addRow(widget)
        return caption

    @staticmethod
    def fill(*widgets: QWidget) -> None:
        """Let *widgets* take their group's spare height."""
        for widget in widgets:
            policy = widget.sizePolicy()
            policy.setVerticalPolicy(QSizePolicy.Policy.Expanding)
            widget.setSizePolicy(policy)

    def sizes(self) -> List[int]:
        """The pane heights, stacked -- also while restacked side by side."""
        return self.splitter.preferred_sizes()

    def restore(self, sizes: Any) -> None:
        """Put the panes back where the user left them; anything that is
        not one positive size per pane is ignored (the defaults stand)."""
        if (isinstance(sizes, list) and len(sizes) == len(self.forms)
                and all(isinstance(s, int) and s >= 0 for s in sizes)
                and sum(sizes) > 0):
            self.splitter.setSizes(sizes)


class SplitFormState:
    """Mixin: saves and restores ``self._split``'s pane sizes.

    Listed before the node base class, so it wraps the node's own
    ``get_state`` / ``restore_state`` (which call ``super()``).
    """

    _split: SplitForm

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()  # type: ignore[misc]
        state[STATE_KEY] = self._split.sizes()
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)  # type: ignore[misc]
        self._split.restore(state.get(STATE_KEY))
