# -*- coding: utf-8 -*-
"""PydanticConfigDialog — a small form dialog generated from a pydantic model.

Used by the hook selector to edit per-hook configs (and reusable for any
"selection is data, behavior is code" surface). Supported field types:
``bool`` (checkbox), ``int``/``float`` (spinboxes), ``str`` (line edit),
``list[str]`` (one entry per line — regex-safe, unlike commas), and
``Literal[...]`` / ``Enum`` (dropdown).

The dialog validates on accept through the model itself; validation
errors are shown inline and keep the dialog open.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, get_args, get_origin

from pydantic import BaseModel, ValidationError
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
)


class PydanticConfigDialog(QDialog):
    """Edit a dict of values against a pydantic model's fields."""

    def __init__(
        self,
        model_cls: type[BaseModel],
        values: Optional[dict[str, Any]] = None,
        title: str = "Configure",
        parent=None,
    ) -> None:
        # Parent None by callers inside a QGraphicsProxyWidget (ghost-
        # window fix); accepted here for standalone use.
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(380)
        self._model_cls = model_cls
        self._editors: dict[str, Any] = {}
        self.result_values: Optional[dict[str, Any]] = None

        current = model_cls.model_validate(dict(values or {})) \
            if values else model_cls()

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        for name, field in model_cls.model_fields.items():
            value = getattr(current, name)
            editor = self._make_editor(field.annotation, value)
            if editor is None:
                continue  # unsupported type — leave at model default
            if field.description:
                editor.setToolTip(field.description)
            form.addRow(f"{name}:", editor)
            self._editors[name] = editor

        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #d32f2f;")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # -- editors ---------------------------------------------------------

    @staticmethod
    def _make_editor(annotation: Any, value: Any):
        # Literal["a","b"] / Enum → dropdown of the allowed values.
        if get_origin(annotation) is Literal:
            combo = QComboBox()
            choices = [str(a) for a in get_args(annotation)]
            combo.addItems(choices)
            idx = combo.findText(str(value))
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            return combo
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            combo = QComboBox()
            for member in annotation:
                combo.addItem(str(member.value), member.value)
            cur = value.value if isinstance(value, Enum) else value
            idx = combo.findData(cur)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            return combo
        if annotation is bool:
            box = QCheckBox()
            box.setChecked(bool(value))
            return box
        if annotation is int:
            spin = QSpinBox()
            spin.setRange(-1_000_000_000, 1_000_000_000)
            spin.setValue(int(value or 0))
            return spin
        if annotation is float:
            spin = QDoubleSpinBox()
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(3)
            spin.setValue(float(value or 0.0))
            return spin
        if annotation is str:
            edit = QLineEdit(str(value or ""))
            return edit
        if get_origin(annotation) is list and get_args(annotation) == (str,):
            edit = QPlainTextEdit()
            edit.setPlainText("\n".join(str(v) for v in (value or [])))
            edit.setPlaceholderText("One entry per line…")
            edit.setMaximumHeight(96)
            return edit
        return None

    @staticmethod
    def _read_editor(editor: Any) -> Any:
        if isinstance(editor, QComboBox):
            data = editor.currentData()
            return data if data is not None else editor.currentText()
        if isinstance(editor, QCheckBox):
            return editor.isChecked()
        if isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            return editor.value()
        if isinstance(editor, QPlainTextEdit):
            return [line for line in editor.toPlainText().splitlines() if line.strip()]
        if isinstance(editor, QLineEdit):
            return editor.text()
        return None

    # -- accept ----------------------------------------------------------

    def _on_accept(self) -> None:
        raw = {name: self._read_editor(ed) for name, ed in self._editors.items()}
        try:
            validated = self._model_cls.model_validate(raw)
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(p) for p in first["loc"])
            self._error_label.setText(f"{location}: {first['msg']}")
            self._error_label.setVisible(True)
            return
        self.result_values = validated.model_dump()
        self.accept()
