# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Where a user sees and withdraws what they granted (spec §7, §22 q1).

D10 gave "don't ask again" a durable form and D35 settled where it
lives -- `~/.weave/silk/grants.json`, outside the graph, because a grant
is **authority and not configuration**: it must not travel in a saved
file, a preset, or a shared graph. That left the other half open, and it
is the half that matters to the person: an allowance you cannot find is
an allowance you cannot take back.

So: a dock, grouped by project, one row per grant, with Revoke on each
row and on each project. Three properties it keeps:

- **It shows what is on disk, not what a run remembers.** Every refresh
  re-reads the file, so a grant another window made is visible here and a
  revocation here is seen by the next gated call in every window (the
  gate consults the store, not a cached set).
- **Revocation is deletion.** There are no deny records, so nothing here
  can create authority or a permanent refusal -- only remove an
  allowance. The safe direction is the only direction this surface goes.
- **Run-scoped grants are not shown**, because they are not here to be
  shown: they live in a gate closure and die with the run. Listing them
  would invite the user to "revoke" something that is already gone.

A dock rather than a node, like the Decision Inbox and for the same
reason (D51/I12): this is a surface over process-wide state, with no
inputs, no outputs and nothing a graph could wire to it.
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
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from weave.logger import get_logger

from ..functions.grants import Grant, GrantStore

log = get_logger("SilkGrantManager")

EMPTY_TEXT = ("Nothing is granted. Every gated tool call will ask.")

NO_PROJECT = "(no project root)"

#: What the row says about a grant nobody described.
DEFAULT_NOTE = "granted from an approval prompt"


def _stamp(grant: Grant) -> str:
    """When it was granted, in the user's local time."""
    import time

    try:
        return time.strftime("%Y-%m-%d %H:%M",
                             time.localtime(float(grant.granted_at)))
    except (OSError, OverflowError, ValueError):
        return "unknown"


class GrantManagerDock(QDockWidget):
    """List durable grants and take them back.

    Owns no run state and no seam: it reads a file and deletes from it.
    Closing it mid-anything changes nothing.
    """

    #: Emitted after a revocation, with how many grants went. Lets a host
    #: put a line in its log without this widget knowing what a log is.
    revoked = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None, *,
                 store: Optional[GrantStore] = None) -> None:
        super().__init__("Granted Permissions", parent)
        self.setObjectName("SilkGrantManagerDock")
        self._store = store if store is not None else GrantStore()
        self._rows: list[QWidget] = []

        body = QWidget(self)
        self._layout = QVBoxLayout(body)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(6)

        self._path = QLabel(str(self._store.path), body)
        self._path.setWordWrap(True)
        self._path.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._layout.addWidget(self._path)

        self._empty = QLabel(EMPTY_TEXT, body)
        self._empty.setWordWrap(True)
        self._layout.addWidget(self._empty)

        self._entries = QVBoxLayout()
        self._entries.setSpacing(4)
        self._layout.addLayout(self._entries)
        self._layout.addStretch(1)

        area = QScrollArea(self)
        area.setWidgetResizable(True)
        area.setWidget(body)
        self.setWidget(area)

        self.refresh()

    # ── contents ──────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Re-read the file and rebuild. Cheap, and always from disk."""
        self._store.reload()
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows = []

        projects = self._store.projects()
        self._empty.setVisible(not projects)
        for project in projects:
            grants = self._store.for_project(project or None)
            if not grants:
                continue
            self._add(self._project_header(project, len(grants)))
            for grant in grants:
                self._add(self._grant_row(grant))

    def _add(self, widget: QWidget) -> None:
        self._entries.addWidget(widget)
        self._rows.append(widget)

    def _project_header(self, project: str, count: int) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        row = QHBoxLayout(frame)
        row.setContentsMargins(6, 4, 6, 4)

        label = QLabel(f"<b>{project or NO_PROJECT}</b> — {count} granted")
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Preferred)
        row.addWidget(label)

        button = QPushButton("Revoke all")
        button.setToolTip("Withdraw every grant in this project. The next "
                          "gated call there asks again.")
        button.clicked.connect(lambda: self.revoke_project(project))
        row.addWidget(button)
        return frame

    def _grant_row(self, grant: Grant) -> QWidget:
        frame = QFrame()
        row = QHBoxLayout(frame)
        row.setContentsMargins(18, 2, 6, 2)

        who = f" by {grant.granted_by}" if grant.granted_by else ""
        label = QLabel(f"<b>{grant.tool_name}</b> — {_stamp(grant)}{who}")
        label.setToolTip(grant.note or DEFAULT_NOTE)
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Preferred)
        row.addWidget(label)

        button = QPushButton("Revoke")
        button.clicked.connect(
            lambda: self.revoke(grant.project, grant.tool_name))
        row.addWidget(button)
        return frame

    # ── the two verbs ─────────────────────────────────────────────────

    def revoke(self, project: str, tool_name: str) -> bool:
        """Withdraw one grant. Returns whether there was one to withdraw."""
        gone = self._store.revoke(project or None, tool_name)
        if gone:
            log.info(f"Revoked '{tool_name}' for {project or NO_PROJECT}")
            self.revoked.emit(1)
        self.refresh()
        return gone

    def revoke_project(self, project: str) -> int:
        """Withdraw every grant in one project; returns how many."""
        count = self._store.revoke_project(project or None)
        if count:
            log.info(f"Revoked {count} grant(s) for {project or NO_PROJECT}")
            self.revoked.emit(count)
        self.refresh()
        return count

    # ── convenience ───────────────────────────────────────────────────

    @classmethod
    def attach(cls, main_window: Any, *,
               area: Qt.DockWidgetArea = Qt.DockWidgetArea.RightDockWidgetArea,
               store: Optional[GrantStore] = None) -> "GrantManagerDock":
        """Create the dock and add it to *main_window*.

        Silk has no plugin-side hook into the host's window, so the host
        (or a user's startup script) calls this -- the same arrangement
        the Decision Inbox uses.
        """
        dock = cls(main_window, store=store)
        main_window.addDockWidget(area, dock)
        return dock
