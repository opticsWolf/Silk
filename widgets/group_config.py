# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The dialogs behind a tool tree's category gears.

Some tools are governed by a setting that is a grant, not a preference
-- graph authoring's whitelist, the knowledge graph's access level --
and the setting lives behind a gear on its category row, beside the
tools it governs (``ToolTreeWidget.set_configurable``). Each opens in a
dialog of its own, built once per node and reused: modal on purpose,
since a grant edited in a panel that can drift out of view is a grant
nobody re-reads.
"""

from typing import Dict, Optional, Tuple

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget,
)


class ConfigDialogs:
    """One dialog per key, holding a node's own (registered) widget."""

    def __init__(self) -> None:
        self._dialogs: Dict[str, QDialog] = {}

    def open(self, key: str, widget: QWidget, title: str, *,
             min_size: Optional[Tuple[int, int]] = None,
             size: Optional[Tuple[int, int]] = None) -> QDialog:
        """Show the dialog for *key*, building it around *widget* first.

        The widget's ``intro`` attribute, if any, is shown above it.
        """
        dialog = self._dialogs.get(key)
        if dialog is None:
            dialog = QDialog()
            dialog.setWindowTitle(title)
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(8)
            intro = getattr(widget, "intro", "")
            if intro:
                label = QLabel(intro)
                label.setWordWrap(True)
                layout.addWidget(label)
            layout.addWidget(widget, 1)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(dialog.reject)
            buttons.accepted.connect(dialog.accept)
            layout.addWidget(buttons)
            if min_size:
                dialog.setMinimumSize(*min_size)
            else:
                dialog.setMinimumWidth(380)
            if size:
                dialog.resize(*size)
            self._dialogs[key] = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def get(self, key: str) -> Optional[QDialog]:
        return self._dialogs.get(key)
