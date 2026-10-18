# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The Classifier node's thresholds table (C7).

One row per **model id** -- ``jev/jev-1.13.0``, ``local/judge`` -- because the
bands are calibrated per model: a new Jev version or a different judge model
starts uncalibrated, and an uncalibrated model never skips a question. The
value is ``{model_id: BandsConfig-as-dict}``; rows that do not validate are
dropped from the value and marked, so a typo cannot quietly loosen a band.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ValidationError
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from weave.panel.mirror_contracts import MirrorContract
from weave.widgets.composite_value import CompositeValueWidget

from ..functions.classifier import BandsConfig

#: (column header, BandsConfig field, tooltip)
COLUMNS = (
    ("Model", None, "Versioned model id the bands were calibrated for."),
    ("Reversible ≥", "reversible", "Skip only if P(reversible) is at least this."),
    ("In scope ≥", "in_scope", "Skip only if P(in scope of the task) is at least this."),
    ("Consistent ≥", "consistent", "Skip only if P(consistent with prior decisions) is at least this."),
    ("Max risk", "max_risk", "Highest risk score (0 none … 4 severe) that may skip."),
    ("Min conf.", "min_confidence", "Confidence the risk and intent answers need."),
    ("Intents", "allow_intents", "Intents that may skip, comma-separated."),
)

_BAD = QBrush(QColor(200, 60, 60, 60))


def _row_bands(cells: List[str]) -> BandsConfig:
    """Validate one row's cells (everything after the model id)."""
    raw: Dict[str, Any] = {}
    for (_header, field, _tip), text in zip(COLUMNS[1:], cells, strict=False):
        text = (text or "").strip()
        if field == "allow_intents":
            raw[field] = [t.strip() for t in text.split(",") if t.strip()]
        elif text:
            raw[field] = float(text)
    return BandsConfig.model_validate(raw)


class BandsTableWidget(CompositeValueWidget):
    """Editable ``model id -> bands`` table; value = ``{model: bands-dict}``."""

    #: Rows added, removed or edited (display + value notify).
    bands_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in COLUMNS])
        for i, (_h, _f, tip) in enumerate(COLUMNS):
            self._table.horizontalHeaderItem(i).setToolTip(tip)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.setMinimumHeight(96)
        self.layout().addWidget(self._table)

        buttons = QHBoxLayout()
        self._btn_add = QPushButton("Add model")
        self._btn_remove = QPushButton("Remove")
        buttons.addWidget(self._btn_add)
        buttons.addWidget(self._btn_remove)
        buttons.addStretch(1)
        self.layout().addLayout(buttons)

        self._hint = QLabel("No calibrated model: the classifier scores, but never skips "
                            "a question.")
        self._hint.setWordWrap(True)
        self.layout().addWidget(self._hint)

        self._btn_add.clicked.connect(lambda: self.add_model(""))
        self._btn_remove.clicked.connect(self._remove_selected)
        self._table.itemChanged.connect(self._on_edit)
        self._invalid: set[int] = set()

    # -- editing ------------------------------------------------------------
    def add_model(self, model: str, bands: Optional[BandsConfig] = None) -> int:
        bands = bands or BandsConfig()
        row = self._table.rowCount()
        self._table.blockSignals(True)
        self._table.insertRow(row)
        self._set_row(row, model, bands)
        self._table.blockSignals(False)
        self._changed()
        return row

    def _set_row(self, row: int, model: str, bands: BandsConfig) -> None:
        values = [model] + [
            ", ".join(bands.allow_intents) if f == "allow_intents" else f"{getattr(bands, f):g}"
            for _h, f, _t in COLUMNS[1:]
        ]
        for col, text in enumerate(values):
            self._table.setItem(row, col, QTableWidgetItem(text))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        if not rows and self._table.rowCount():
            rows = [self._table.rowCount() - 1]
        for r in rows:
            self._table.removeRow(r)
        self._changed()

    def _on_edit(self, _item: QTableWidgetItem) -> None:
        self._changed()

    def _changed(self) -> None:
        self._validate_rows()
        n = len(self._read())
        self._hint.setText(
            "No calibrated model: the classifier scores, but never skips a question."
            if n == 0 else
            f"{n} calibrated model(s). Any other model answers but never skips.")
        self.bands_changed.emit()

    def _validate_rows(self) -> None:
        self._invalid.clear()
        self._table.blockSignals(True)
        for r in range(self._table.rowCount()):
            ok = True
            model = self._cell(r, 0)
            if not model or model.endswith("latest"):
                ok = False
            else:
                try:
                    _row_bands([self._cell(r, c) for c in range(1, len(COLUMNS))])
                except (ValueError, ValidationError):
                    ok = False
            if not ok:
                self._invalid.add(r)
            for c in range(len(COLUMNS)):
                item = self._table.item(r, c)
                if item is not None:
                    item.setBackground(_BAD if not ok else QBrush())
        self._table.blockSignals(False)

    def _cell(self, row: int, col: int) -> str:
        item = self._table.item(row, col)
        return item.text().strip() if item is not None else ""

    def invalid_rows(self) -> List[int]:
        return sorted(self._invalid)

    # -- CompositeValueWidget contract --------------------------------------
    def _read(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for r in range(self._table.rowCount()):
            model = self._cell(r, 0)
            if not model or model.endswith("latest"):
                continue
            try:
                bands = _row_bands([self._cell(r, c) for c in range(1, len(COLUMNS))])
            except (ValueError, ValidationError):
                continue                      # an invalid row never loosens anything
            out[model] = bands.model_dump()
        return out

    def _write(self, value: Any) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for model, raw in sorted(dict(value or {}).items()):
            try:
                bands = raw if isinstance(raw, BandsConfig) else BandsConfig.model_validate(raw)
            except ValidationError:
                continue
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._set_row(row, str(model), bands)
        self._table.blockSignals(False)
        self._changed()

    def _internal_signal_sources(self) -> List[QObject]:
        return [self._table]

    # -- convenience for the node --------------------------------------------
    def bands(self) -> Dict[str, Dict[str, Any]]:
        return self._read()

    def set_bands(self, value: Any) -> None:
        self._write(value)

    __mirror__ = MirrorContract(
        clone=lambda src, _b: BandsTableWidget(),
    )
