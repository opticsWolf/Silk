# -*- coding: utf-8 -*-
"""Preset quick-select bar: dropdown + add/remove tool buttons.

Wraps a :class:`~..functions.presets.PresetStore`. The host node supplies
two callables:

``collect() -> dict``
    Current node state as a dict matching the store's pydantic model
    (without ``name`` — the bar injects it from the dialog).

``apply(preset) -> None``
    Push a validated preset model back into the node's widgets.

Add opens a name dialog (with overwrite confirmation if the name exists);
Remove asks for confirmation. Both are plain modal dialogs parented to
None — inside a QGraphicsProxyWidget a widget parent would spawn ghost
windows (same fix as PathPickerWidget).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QWidget,
)

from weave.library.sync_tool_button.widgets.sync_tool_button import SyncToolButton
from weave.logger import get_logger

from ..functions.presets import PresetStore

log = get_logger("SilkPresetBar")

_PLACEHOLDER = "— presets —"


class PresetBarWidget(QWidget):
    """Dropdown of stored presets with +/- management buttons."""

    def __init__(
        self,
        store: PresetStore,
        collect: Callable[[], dict],
        apply: Callable[[Any], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._collect = collect
        self._apply = apply

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._combo = QComboBox()
        self._combo.setToolTip("Load a stored preset.")
        layout.addWidget(self._combo, stretch=1)

        self._btn_add = SyncToolButton(
            initial_text="Add", show_label=False, dimensions=24
        )
        self._btn_add.set_tooltip("Save current settings as a preset…")
        layout.addWidget(self._btn_add)

        self._btn_remove = SyncToolButton(
            initial_text="Remove", show_label=False, dimensions=24
        )
        self._btn_remove.set_tooltip("Delete the selected preset…")
        layout.addWidget(self._btn_remove)

        self._btn_add.clicked.connect(self._on_add)
        self._btn_remove.clicked.connect(self._on_remove)
        self._combo.activated.connect(self._on_selected)

        self.refresh()

    # -- combo management ------------------------------------------------

    def refresh(self, select: Optional[str] = None) -> None:
        """Rebuild the dropdown from the store; optionally select a name."""
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem(_PLACEHOLDER)
        for name in self._store.names():
            self._combo.addItem(name)
        if select:
            index = self._combo.findText(select)
            if index >= 0:
                self._combo.setCurrentIndex(index)
        self._combo.blockSignals(False)
        self._btn_remove.setEnabled(bool(self._store.names()))

    def current_name(self) -> Optional[str]:
        text = self._combo.currentText()
        return text if text and text != _PLACEHOLDER else None

    # -- actions ---------------------------------------------------------

    def _on_selected(self, _index: int) -> None:
        name = self.current_name()
        if name is None:
            return
        preset = self._store.get(name)
        if preset is None:  # stale entry (file edited externally)
            self._store.reload()
            self.refresh()
            return
        try:
            self._apply(preset)
        except Exception as exc:
            log.error(f"Applying preset '{name}' failed: {exc}", exc_info=True)

    def _on_add(self) -> None:
        suggested = self.current_name() or ""
        name, accepted = QInputDialog.getText(
            None,
            "Save Preset",
            "Preset name:",
            QLineEdit.EchoMode.Normal,
            suggested,
        )
        name = name.strip()
        if not accepted or not name:
            return
        if self._store.get(name) is not None:
            answer = QMessageBox.question(
                None,
                "Overwrite Preset",
                f"A preset named '{name}' already exists.\nOverwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            payload = dict(self._collect())
            payload["name"] = name
            self._store.upsert(self._store.model.model_validate(payload))
        except Exception as exc:
            log.error(f"Saving preset '{name}' failed: {exc}", exc_info=True)
            QMessageBox.warning(None, "Save Preset", f"Could not save preset:\n{exc}")
            return
        self.refresh(select=name)

    def _on_remove(self) -> None:
        name = self.current_name()
        if name is None:
            return
        answer = QMessageBox.question(
            None,
            "Remove Preset",
            f"Delete the preset '{name}'?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._store.remove(name)
        self.refresh()

    def cleanup(self) -> None:
        self._btn_add.cleanup()
        self._btn_remove.cleanup()
