# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The Decision Inbox dock — one place to see every blocked agent (spec D59).

With N agents, the answering widgets built into each Agent node (D48) are
scattered across N node bodies. This dock lists the ones that are waiting
and offers the same four answers the node offers, for each.

**It is a dock, and that is the whole point.** D51 rejected an approval
*node* because a node cannot answer a question asked from inside
``compute()``: inputs are gathered before compute runs, so there is no
inbound channel while the run blocks. A dock has no such problem — it is
main-thread UI like the log pane, with no wires, no graph channel and no
rendezvous handle. Invariant I12 is the short form: a decision surface may
be a node only if the decision happens at a turn boundary; this one does
not, so it is not a node.

Clicking Deny here calls the asking node's own ``_answer_decision`` — the
same method its own button calls, resolving through the same run-scoped
seam. The dock owns no seam, holds no run state, and if it is closed
mid-question nothing changes: the node's own surface is still there.

The container the Agent node uses for its prompt is a composite widget
that NodePanel's clone strategies do not cover, so the rows are built
here and forward the action, which is exactly what
``mirror_contracts.wire_action_proxy`` does for a mirrored button.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from weave.logger import get_logger

from ..functions.decision_registry import REGISTRY, DecisionEntry, DecisionRegistry
from ..functions.grants import SCOPE_ALWAYS, SCOPE_ONCE, SCOPE_RUN

log = get_logger("SilkDecisionInbox")

#: The answers, in the node's order: deny first, because the safe answer
#: should never be the one that takes an extra look.
ANSWERS: tuple[tuple[str, bool, str], ...] = (
    ("Deny", False, SCOPE_ONCE),
    ("Allow once", True, SCOPE_ONCE),
    ("Allow this run", True, SCOPE_RUN),
    ("Always allow", True, SCOPE_ALWAYS),
)

EMPTY_TEXT = "No agent is waiting on you."


class DecisionInboxDock(QDockWidget):
    """Every waiting agent, in one panel."""

    #: Registry changes arrive on whichever thread the run was on; this
    #: signal is the hop back to the main thread, where widgets live.
    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None, *,
                 registry: Optional[DecisionRegistry] = None) -> None:
        super().__init__("Decision Inbox", parent)
        self.setObjectName("SilkDecisionInbox")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        self._registry = registry if registry is not None else REGISTRY

        body = QWidget()
        self._layout = QVBoxLayout(body)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(6)
        self._layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        self.setWidget(scroll)

        self._empty = QLabel(EMPTY_TEXT)
        self._empty.setWordWrap(True)
        self._layout.insertWidget(0, self._empty)

        self._rows: list[QWidget] = []
        self.changed.connect(self.refresh, Qt.ConnectionType.QueuedConnection)
        self._unsubscribe = self._registry.subscribe(self.changed.emit)
        self.refresh()

    # ── Lifetime ──────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:      # noqa: N802 - Qt override
        """Closing the dock unsubscribes; the node's own surface remains.

        Nothing about a pending decision belongs to this dock, so closing
        it while an agent waits must not strand the run -- it just takes
        away the shortcut.
        """
        self.detach()
        super().closeEvent(event)

    def detach(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    # ── Rendering ─────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild the rows from the registry. **Main thread.**"""
        for row in self._rows:
            self._layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()

        entries = self._registry.entries()
        self._empty.setText(
            EMPTY_TEXT if not entries
            else f"{len(entries)} agent(s) waiting on you."
        )
        for index, entry in enumerate(entries):
            row = self._build_row(entry)
            self._layout.insertWidget(index + 1, row)
            self._rows.append(row)

    def _build_row(self, entry: DecisionEntry) -> QWidget:
        row = QFrame()
        row.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(row)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel(entry.title())
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        prompt = QLabel(entry.detail())
        prompt.setWordWrap(True)
        layout.addWidget(prompt)

        buttons = QHBoxLayout()
        for label, approved, remember in ANSWERS:
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, e=entry, a=approved, r=remember:
                    self.answer(e, a, r)
            )
            buttons.addWidget(button)

        reveal = QPushButton("Show node")
        reveal.clicked.connect(lambda _checked=False, e=entry: self.reveal(e))
        buttons.addWidget(reveal)
        layout.addLayout(buttons)
        return row

    # ── Acting ────────────────────────────────────────────────────────

    def answer(self, entry: DecisionEntry, approved: bool,
               remember: str) -> bool:
        """Answer through the asking node, never around it (D59).

        Returns whether the answer was delivered. A node that has gone --
        deleted, or its run already over -- leaves a row that this call
        clears instead of pretending to resolve something.
        """
        node = entry.node
        if node is None or not hasattr(node, "_answer_decision"):
            self._registry.unregister(entry.decision_id)
            return False
        try:
            node._answer_decision(approved, remember)
        except RuntimeError:
            # The node's Qt object died between the click and the call.
            self._registry.unregister(entry.decision_id)
            return False
        return True

    def reveal(self, entry: DecisionEntry) -> bool:
        """Select the asking node on the canvas -- the row's other half.

        The inbox answers the question; the canvas is where the answer's
        context is, and a graph of ten agents makes "which one is this?"
        a real question.
        """
        node = entry.node
        if node is None:
            return False
        try:
            scene = node.scene()
            if scene is not None:
                scene.clearSelection()
            node.setSelected(True)
            for view in (scene.views() if scene is not None else ()):
                view.centerOn(node)
        except (AttributeError, RuntimeError):
            return False
        return True

    # ── Convenience ───────────────────────────────────────────────────

    @classmethod
    def attach(cls, main_window: Any, *,
               area: Qt.DockWidgetArea = Qt.DockWidgetArea.RightDockWidgetArea,
               registry: Optional[DecisionRegistry] = None,
               ) -> "DecisionInboxDock":
        """Create the dock and add it to *main_window*.

        Silk has no plugin-side hook into the host's window, so the host
        (or a user's startup script) calls this. Kept here rather than in
        the app so the dock ships with the thing it serves.
        """
        dock = cls(main_window, registry=registry)
        main_window.addDockWidget(area, dock)
        return dock
