# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The decision strip -- a waiting agent's question, shown where you type.

The Prompt node is where a person is looking while an agent runs, so a
question that agent asks belongs under the box they type into rather than
only on the Agent node or in the Decision Inbox. The strip shows the
questions of the agents its owner reaches (the Prompt's ``chat`` wire)
and answers them the way the inbox does: through the asking node's own
``_answer_decision``, never a channel of its own (D51, D59). It is a
mirror, so I12 holds -- the decision is still answered by the asking
node's surface.

Hidden while nobody it serves is waiting.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
)

from ..functions.decision_registry import REGISTRY, DecisionEntry, DecisionRegistry
from .decision_inbox import ANSWERS, REASON_PLACEHOLDER


class DecisionStrip(QFrame):
    """The first waiting question among the nodes *scope* returns."""

    #: Registry changes arrive on whichever thread asked; this is the hop.
    changed = Signal()

    def __init__(self, scope: Callable[[], Iterable[Any]], *,
                 registry: Optional[DecisionRegistry] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._scope = scope
        self._registry = registry if registry is not None else REGISTRY
        self._entry: Optional[DecisionEntry] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(3)
        self._title = QLabel()
        self._title.setStyleSheet("font-weight: bold;")
        self._prompt = QLabel()
        self._prompt.setWordWrap(True)
        self._advice = QLabel()
        self._advice.setWordWrap(True)
        self._advice.setStyleSheet("font-style: italic;")
        self._reason = QLineEdit()
        self._reason.setPlaceholderText(REASON_PLACEHOLDER)
        for widget in (self._title, self._prompt, self._advice, self._reason):
            layout.addWidget(widget)

        buttons = QHBoxLayout()
        for label, approved, remember in ANSWERS:
            button = QPushButton(label)
            button.clicked.connect(
                lambda _c=False, a=approved, r=remember: self.answer(a, r))
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.changed.connect(self.refresh, Qt.ConnectionType.QueuedConnection)
        self._unsubscribe: Optional[Callable[[], None]] = \
            self._registry.subscribe(self.changed.emit)
        self.refresh()

    def entries(self) -> list[DecisionEntry]:
        """The waiting questions of the nodes this strip serves."""
        try:
            served = {id(node) for node in self._scope()}
        except RuntimeError:          # the owner's Qt object is gone
            return []
        return [e for e in self._registry.entries()
                if e.node is not None and id(e.node) in served]

    def refresh(self) -> None:
        """Show the first waiting question, or hide. **Main thread.**"""
        entries = self.entries()
        entry = entries[0] if entries else None
        if entry is None or self._entry is None \
                or entry.decision_id != self._entry.decision_id:
            self._reason.clear()
        self._entry = entry
        if entry is None:
            self.hide()
            return
        more = f"  (1 of {len(entries)})" if len(entries) > 1 else ""
        self._title.setText(f"{entry.title()} is waiting on you{more}")
        self._prompt.setText(entry.detail())
        self._advice.setText(entry.advice)
        self._advice.setVisible(bool(entry.advice))
        self.show()

    def answer(self, approved: bool, remember: str) -> bool:
        """Answer the shown question through its node. False if it is gone."""
        entry = self._entry
        node = entry.node if entry is not None else None
        if entry is None:
            return False
        if node is None or not hasattr(node, "_answer_decision"):
            self._registry.unregister(entry.decision_id)
            return False
        try:
            node._answer_decision(approved, remember, self._reason.text().strip())
        except RuntimeError:
            self._registry.unregister(entry.decision_id)
            return False
        return True

    def detach(self) -> None:
        """Stop listening. Idempotent; the owner calls it on cleanup."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
