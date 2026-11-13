# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Silk document nodes -- documents to Markdown with bobine.

* **Converter Settings** puts bobine's options on a ``bobine_settings``
  wire: the route PDF pages take (text layer only, + formula OCR, + layout
  and OCR where needed, or everywhere), the device and precision the
  models run at, the table backend and the image and heading options. Its
  Check button reports which models are cached and whether an ONNX
  Runtime is there to run them.
* **Document Converter** converts a file, or every file in a folder, to
  Markdown on disk and on its outputs, with page progress and Cancel.
* **Excel Converter** reads a workbook: Markdown tables, the sheet names,
  a CSV per sheet and the cell JSON.
* **JATS Converter** reads a JATS XML article (PubMed Central, arXiv
  JATS, publisher XML): Markdown with LaTeX math, its title and counts.

bobine runs in a child process per call (``functions/bobine_worker.py``):
``import bobine`` loads ONNX Runtime, and a second one in Weave's process
can take the interpreter down. Without bobine installed the nodes say so
and output nothing.
"""

import threading
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QSpinBox,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import STREAM_ONLY
from weave.node.base import ActiveNode
from weave.node.threaded import ThreadedManualNode, ThreadedNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger
from weave.widgets.path_history_picker import PathHistoryPicker
from weave.widgets.sync_button import SyncButton

from .silk_ports import BOBINE_SETTINGS_TYPE  # noqa: F401
from ..functions import bobine_convert as bc
from ..functions.port_text import text_input

log = get_logger("SilkDocuments")

_DOC_FILTERS = [
    "Documents (*.pdf *.docx *.doc *.xlsx *.xls *.pptx *.ppt *.xml *.nxml "
    "*.txt *.md *.html)", "All Files (*)"]


def _form() -> QFormLayout:
    form = QFormLayout()
    form.setContentsMargins(5, 5, 5, 5)
    form.setSpacing(4)
    return form


def _missing_status() -> str:
    return f"bobine is not installed: {bc.INSTALL_HINT}."


class _PickerHistory:
    """Saves the recent-paths lists of a node's history pickers."""

    _pickers: Dict[str, PathHistoryPicker]

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()  # type: ignore[misc]
        state["path_history"] = {k: p.history() for k, p in self._pickers.items()}
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        with self._widget_core.suppress_signals():  # type: ignore[attr-defined]
            super().restore_state(state)  # type: ignore[misc]
        saved = state.get("path_history") or {}
        for key, picker in self._pickers.items():
            picker.set_history([str(p) for p in saved.get(key) or ()])


# ── Converter Settings ───────────────────────────────────────────────────

@register_node
class SilkConverterSettingsNode(_PickerHistory, ActiveNode):
    """bobine's converter options, put on a wire."""

    _widget_core: WidgetCore
    _models_checked = Signal(object)    # report dict | Exception

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Documents"
    node_name: ClassVar[Optional[str]] = "Converter Settings"
    node_description: ClassVar[Optional[str]] = (
        "bobine's options for the document converters: PDF route, device, "
        "precision, tables, images. Check reports the model cache.")
    node_tags: ClassVar[Optional[List[str]]] = ["silk", "bobine", "pdf", "markdown", "settings"]
    node_icon: ClassVar[Optional[str]] = "tool"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 2   # 2: device/precision are the stack words
    node_version = 2

    def __init__(self, title: str = "Converter Settings", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self.add_output("settings", datatype="bobine_settings")
        self._checking = False

        form = _form()
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        def combo(key: str, label: str, choices: tuple, tip: str) -> QComboBox:
            box = QComboBox()
            box.addItems(list(choices))
            box.setToolTip(tip)
            form.addRow(label, box)
            self._widget_core.register_widget(
                key, box, role=PortRole.INPUT, datatype="string",
                default=bc.DEFAULT_SETTINGS[key], add_to_layout=False)
            self.add_input(key, datatype="string")
            return box

        def check(key: str, label: str, tip: str) -> None:
            box = QCheckBox(label)
            box.setChecked(bool(bc.DEFAULT_SETTINGS[key]))
            box.setToolTip(tip)
            form.addRow("", box)
            self._widget_core.register_widget(
                key, box, role=PortRole.INTERNAL, datatype="bool",
                default=bc.DEFAULT_SETTINGS[key], add_to_layout=False)

        def spin(key: str, label: str, tip: str) -> None:
            box = QSpinBox()
            box.setRange(72, 600)
            box.setSingleStep(50)
            box.setValue(int(bc.DEFAULT_SETTINGS[key]))
            box.setToolTip(tip)
            form.addRow(label, box)
            self._widget_core.register_widget(
                key, box, role=PortRole.INTERNAL, datatype="int",
                default=bc.DEFAULT_SETTINGS[key], add_to_layout=False)

        combo("routing", "PDF route:", bc.ROUTES,
              "never: the text layer only, fastest, no models. surgical: + "
              "formula OCR into LaTeX. auto: + layout and OCR on the pages "
              "that need it (scans, tables). always: every page through "
              "layout and OCR, slowest. Office, JATS and text files ignore it.")
        combo("device", "Device:", bc.DEVICES,
              "Where the models run. auto: each pass keeps its measured "
              "device (CUDA where it is faster). cpu: every pass on the CPU. "
              "cuda: every pass on CUDA, with CPU fallback; needs an "
              "onnxruntime-gpu install. The same words as the ONNX Loader.")
        combo("precision", "Precision:", bc.PRECISIONS,
              "The stack vocabulary: auto (bobine's measured choice per "
              "pass), fp32 (full weights), fp16 (half weights on CUDA), "
              "int8 (4x smaller, ~1.9x faster formula decode on the CPU; "
              "some formulas decode differently). The same words as the "
              "ONNX Loader.")
        combo("table_backend", "Tables:", bc.TABLE_BACKENDS,
              "slanet: per-table crops (default). potatr: whole-page table "
              "structure. auto: potatr where a page has words, else slanet.")
        combo("image_mode", "Images:", bc.IMAGE_MODES,
              "link: figures saved beside the Markdown and linked. embed: "
              "inline base64 (capped per image).")
        check("extract_images", "Extract images",
              "Save the figures; off keeps the Markdown text-only.")
        check("detect_headings", "Detect headings",
              "Turn large or bold lines into Markdown headings.")
        spin("formula_dpi", "Formula DPI:",
             "Render resolution for formula crops; higher reads small "
             "sub/superscripts better, slower.")
        spin("render_dpi", "Page DPI:",
             "Render resolution for pages sent through layout and OCR.")

        self._ocr_lang = QLineEdit("en")
        self._ocr_lang.setToolTip("OCR language code for scanned pages (en, de, …).")
        form.addRow("OCR language:", self._ocr_lang)
        self._widget_core.register_widget(
            "ocr_lang", self._ocr_lang, role=PortRole.INTERNAL,
            datatype="str", default="en", add_to_layout=False)

        self._cache = PathHistoryPicker(mode="folder")
        self._cache.setToolTip(
            "Where bobine keeps its models (a Hugging Face hub cache). "
            "Empty: the default hub cache.")
        form.addRow("Model cache:", self._cache)
        self._widget_core.register_widget(
            "cache_dir", self._cache, role=PortRole.INPUT,
            datatype="dirpath", default="", add_to_layout=False)
        self.add_input("cache_dir", datatype="dirpath")
        self._pickers = {"cache_dir": self._cache}

        row = QHBoxLayout()
        self._btn_check = SyncButton(initial_text="Check models", initial_icon="refresh")
        self._btn_check.setToolTip(
            "Which of bobine's models are downloaded, and whether an ONNX "
            "Runtime is installed to run them. Downloads nothing.")
        self._btn_check.clicked.connect(self._on_check)
        row.addWidget(self._btn_check)
        row.addStretch(1)
        form.addRow("", row)
        self._widget_core.register_widget(
            "btn_check", self._btn_check, role=PortRole.INTERNAL, add_to_layout=False)

        self._label_status = QLabel("")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False)

        self._models_checked.connect(self._on_models_checked)
        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    @classmethod
    def migrate_state(cls, state: Dict[str, Any],
                      from_version: int) -> Dict[str, Any]:
        """api 1 → 2: the rows speak the stack words.

        A scene saved before: ``gpu`` became ``cuda``, and the separate
        formula-weights row folded into precision (``int8`` when it was
        picked and precision was still fp32). The quantization key is
        gone; the other dropped values were defaults.
        """
        if from_version < 2:
            widgets = state.get("widget_data")
            if isinstance(widgets, dict):
                if str(widgets.get("device") or "").strip().lower() == "gpu":
                    widgets["device"] = "cuda"
                if (str(widgets.get("quantization") or "").strip().lower() == "int8"
                        and str(widgets.get("precision") or "fp32").strip().lower()
                        in ("", "fp32")):
                    widgets["precision"] = "int8"
                widgets.pop("quantization", None)
        return state

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        settings = bc.clean_settings({k: inputs.get(k) for k in bc.DEFAULT_SETTINGS
                                      if k in inputs})
        self._settings = settings
        if not bc.BOBINE_AVAILABLE:
            self._sync_status = _missing_status()
        elif settings["routing"] != "never" and not bc.ort_installed():
            self._sync_status = (
                f"{bc.describe(settings)}. No onnxruntime installed: PDFs fall "
                "back to the text layer (pip install onnxruntime).")
        else:
            self._sync_status = f"{bc.describe(settings)}."
        return {"settings": settings}

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        if hasattr(self, "_sync_status"):
            self._widget_core.push_display("status", self._sync_status)

    # -- model check: a child process, off the UI thread -----------------

    def _on_check(self) -> None:
        if self._checking:
            return
        if not bc.BOBINE_AVAILABLE:
            self._widget_core.push_display("status", _missing_status())
            return
        self._checking = True
        self._widget_core.set_port_enabled("btn_check", False)
        self._widget_core.push_display("status", "Checking bobine's model cache…")
        settings = getattr(self, "_settings", None)

        def work() -> None:
            try:
                report: Any = bc.model_report(settings)
            except Exception as exc:  # noqa: BLE001 - shown in the label
                report = exc
            try:
                self._models_checked.emit(report)
            except RuntimeError:
                pass    # the node was deleted meanwhile

        threading.Thread(target=work, daemon=True, name="bobine-models").start()

    @Slot(object)
    def _on_models_checked(self, report: Any) -> None:
        self._checking = False
        try:
            self._widget_core.set_port_enabled("btn_check", True)
        except RuntimeError:
            return
        self._widget_core.push_display("status", model_text(report))


def model_text(report: Any) -> str:
    """The Check button's answer, for the status label."""
    if isinstance(report, BaseException):
        return f"Model check failed: {report}"
    rows = report.get("models") or []
    have = [r for r in rows if r.get("cached")]
    size = sum(int(r.get("disk_usage_bytes") or 0) for r in have) / 1024 ** 3
    lines = [f"{len(have)} of {len(rows)} models cached ({size:.2f} GB)."]
    missing = [str(r.get("model_id")) for r in rows if not r.get("cached")]
    if missing:
        lines.append("Downloaded on first use: " + ", ".join(missing) + ".")
    lines.append("ONNX Runtime: " + (report.get("ort") or
                                     "none -- PDFs keep to the text layer."))
    return "\n".join(lines)


# ── Document Converter ───────────────────────────────────────────────────

@register_node
class SilkDocumentConverterNode(_PickerHistory, ThreadedManualNode):
    """A file, or a folder of files, to Markdown with bobine."""

    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Documents"
    node_name: ClassVar[Optional[str]] = "Document Converter"
    node_description: ClassVar[Optional[str]] = (
        "Converts a PDF, Office, JATS or text file -- or a folder of them -- "
        "to Markdown with bobine, with page progress and Cancel.")
    node_tags: ClassVar[Optional[List[str]]] = ["silk", "bobine", "pdf", "markdown", "convert"]
    node_icon: ClassVar[Optional[str]] = "markdown"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1
    node_version = 1

    def __init__(self, title: str = "Document Converter", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self.add_input("settings", datatype="bobine_settings")
        self.add_output("markdown", datatype="string")
        self.add_output("md_path", datatype="filepath")
        self.add_output("output_dir", datatype="dirpath")
        self.add_output("report", datatype="dict")
        self._sync_status = ""

        form = _form()
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self._source = PathHistoryPicker(mode="file", filters=_DOC_FILTERS)
        self._source.setToolTip("The document to convert.")
        form.addRow("File:", self._source)
        self._widget_core.register_widget(
            "source", self._source, role=PortRole.INPUT,
            datatype="filepath", default="", add_to_layout=False)
        self.add_input("source", datatype="filepath")

        self._folder = PathHistoryPicker(mode="folder")
        self._folder.setToolTip(
            "Or a folder: every file bobine reads in it. Set, it wins over File.")
        form.addRow("Folder:", self._folder)
        self._widget_core.register_widget(
            "folder", self._folder, role=PortRole.INPUT,
            datatype="dirpath", default="", add_to_layout=False)
        self.add_input("folder", datatype="dirpath")

        self._recursive = QCheckBox("Include subfolders")
        form.addRow("", self._recursive)
        self._widget_core.register_widget(
            "recursive", self._recursive, role=PortRole.INTERNAL,
            datatype="bool", default=False, add_to_layout=False)

        self._out = PathHistoryPicker(mode="folder")
        self._out.setToolTip(
            "Where the Markdown goes: <output>/<file stem>/<file stem>.md with "
            "its images. Empty: a 'markdown' folder beside the source.")
        form.addRow("Output:", self._out)
        self._widget_core.register_widget(
            "out_dir", self._out, role=PortRole.INPUT,
            datatype="dirpath", default="", add_to_layout=False)
        self.add_input("out_dir", datatype="dirpath")
        self._pickers = {"source": self._source, "folder": self._folder,
                         "out_dir": self._out}

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        form.addRow(self._progress)
        self._widget_core.register_widget(
            "progress", self._progress, role=PortRole.DISPLAY, datatype="float",
            add_to_layout=False, policy=STREAM_ONLY)

        row = QHBoxLayout()
        self._btn_run = SyncButton(initial_text="Convert", initial_icon="player-play")
        self._btn_run.clicked.connect(self.execute)
        row.addWidget(self._btn_run)
        self._btn_cancel = SyncButton(initial_text="Cancel", initial_icon="cancel")
        self._btn_cancel.setToolTip("Stop the conversion; files already done stay.")
        self._btn_cancel.clicked.connect(self.cancel_compute)
        row.addWidget(self._btn_cancel)
        row.addStretch(1)
        form.addRow("", row)
        for name, btn in (("btn_run", self._btn_run), ("btn_cancel", self._btn_cancel)):
            self._widget_core.register_widget(
                name, btn, role=PortRole.INTERNAL, add_to_layout=False)

        self._label_status = QLabel(
            "Pick a file or folder." if bc.BOBINE_AVAILABLE else _missing_status())
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False)

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()
        self.compute_progress.connect(self._progress.setValue)
        self.compute_started.connect(lambda: self._on_run_state("Converting…"))
        self.compute_cancelled.connect(lambda: self._on_run_state("Cancelled."))
        self.compute_error.connect(lambda msg: self._on_run_state(f"Failed: {msg}"))
        self._refresh_buttons()

    def _on_run_state(self, text: str) -> None:
        try:
            self._widget_core.push_display("status", text)
        except RuntimeError:
            return
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        running = bool(self._is_computing)
        try:
            self._widget_core.set_port_enabled("btn_run", not running and bc.BOBINE_AVAILABLE)
            self._widget_core.set_port_enabled("btn_cancel", running)
        except RuntimeError:
            pass

    def _cleanup_after_worker(self, skip_results: bool = False) -> None:
        super()._cleanup_after_worker(skip_results=skip_results)
        self._refresh_buttons()

    @staticmethod
    def plan(inputs: Dict[str, Any]) -> tuple[List[Path], Path]:
        """The files to convert and the folder they go to."""
        folder = text_input(inputs, "folder").strip()
        source = text_input(inputs, "source").strip()
        if folder:
            root = Path(folder).expanduser()
            if not root.is_dir():
                raise RuntimeError(f"No folder at {folder}.")
            files = bc.files_in(str(root), bool(inputs.get("recursive")))
            if not files:
                raise RuntimeError(f"No document bobine reads in {folder}.")
            base = root
        elif source:
            path = Path(source).expanduser()
            if not path.is_file():
                raise RuntimeError(f"No file at {source}.")
            files, base = [path], path.parent
        else:
            raise RuntimeError("Pick a file or a folder to convert.")
        out = text_input(inputs, "out_dir").strip()
        return files, (Path(out).expanduser() if out else base / "markdown")

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not bc.BOBINE_AVAILABLE:
            raise RuntimeError(_missing_status())
        files, out = self.plan(inputs)
        settings = inputs.get("settings")
        done: List[Dict[str, Any]] = []
        failed: List[Dict[str, str]] = []
        for i, path in enumerate(files):
            def on_page(page: int, total: int, i: int = i) -> None:
                share = (page / total) if total else 0.0
                self.report_progress(int((i + share) * 100 / len(files)))

            try:
                done.append(bc.convert_file(
                    str(path), str(out / path.stem), settings,
                    should_continue=lambda: not self.is_compute_cancelled(),
                    on_page=on_page))
            except bc.ConversionCancelled:
                break
            except RuntimeError as exc:
                if len(files) == 1:
                    raise
                failed.append({"source": str(path), "error": str(exc)})
            self.report_progress(int((i + 1) * 100 / len(files)))
        if len(files) == 1 and done:
            markdown = done[0]["markdown"]
        else:
            markdown = "\n\n".join(
                f"<!-- {Path(d['source']).name} -->\n{d['markdown']}" for d in done)
        pages = sum(int(d.get("pages") or 0) for d in done)
        self._sync_status = (
            f"Converted {len(done)} of {len(files)} file(s), {pages} page(s), "
            f"into {out}." + (f" {len(failed)} failed: {failed[0]['error']}"
                              if failed else ""))
        return {"markdown": markdown,
                "md_path": done[0]["md_path"] if done else None,
                "output_dir": str(out),
                "report": {"files": done and [
                    {k: v for k, v in d.items() if k != "markdown"} for d in done],
                    "failed": failed, "settings": bc.clean_settings(settings)}}

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._refresh_buttons()
        if self._sync_status:
            self._widget_core.push_display("status", self._sync_status)

    def cleanup(self) -> None:
        self.cancel_compute()
        super().cleanup()


# ── Excel / JATS ─────────────────────────────────────────────────────────

class _SingleFileNode(_PickerHistory, ThreadedNode):
    """One file in, converted when the path changes."""

    _widget_core: WidgetCore
    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Documents"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1
    node_version = 1

    _FILTERS: ClassVar[List[str]] = []
    _SOURCE_TIP: ClassVar[str] = ""

    def __init__(self, title: str, **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self._sync_status = ""
        self._form = _form()
        self._widget_core = WidgetCore(layout=self._form)
        self._widget_core.set_node(self)

        self._source = PathHistoryPicker(mode="file", filters=self._FILTERS)
        self._source.setToolTip(self._SOURCE_TIP)
        self._form.addRow("File:", self._source)
        self._widget_core.register_widget(
            "source", self._source, role=PortRole.INPUT,
            datatype="filepath", default="", add_to_layout=False)
        self.add_input("source", datatype="filepath")
        self._pickers = {"source": self._source}

    def _finish_form(self) -> None:
        self._label_status = QLabel(
            "Pick a file." if bc.BOBINE_AVAILABLE else _missing_status())
        self._label_status.setWordWrap(True)
        self._form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False)
        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()
        self.compute_error.connect(
            lambda msg: self._widget_core.push_display("status", f"Failed: {msg}"))

    def _source_path(self, inputs: Dict[str, Any], suffixes: frozenset) -> Optional[Path]:
        source = text_input(inputs, "source").strip()
        if not source:
            self._sync_status = "Pick a file."
            return None
        if not bc.BOBINE_AVAILABLE:
            raise RuntimeError(_missing_status())
        path = Path(source).expanduser()
        if not path.is_file():
            raise RuntimeError(f"No file at {source}.")
        if path.suffix.lower() not in suffixes:
            raise RuntimeError(
                f"{path.name}: expected {', '.join(sorted(suffixes))}.")
        return path

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        if self._sync_status:
            self._widget_core.push_display("status", self._sync_status)

    def cleanup(self) -> None:
        self.cancel_compute()
        super().cleanup()


@register_node
class SilkExcelConverterNode(_SingleFileNode):
    """A workbook as Markdown tables, CSV per sheet and JSON."""

    node_name: ClassVar[Optional[str]] = "Excel Converter"
    node_description: ClassVar[Optional[str]] = (
        "Reads a workbook (xlsx, xls, xlsb, ods) with bobine: Markdown "
        "tables, sheet names, a CSV per sheet and the cell JSON.")
    node_tags: ClassVar[Optional[List[str]]] = ["silk", "bobine", "excel", "csv", "markdown"]
    node_icon: ClassVar[Optional[str]] = "table"
    _FILTERS: ClassVar[List[str]] = [
        "Workbooks (*.xlsx *.xlsm *.xls *.xlsb *.ods)", "All Files (*)"]
    _SOURCE_TIP: ClassVar[str] = "The workbook to read."

    def __init__(self, title: str = "Excel Converter", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self.add_output("markdown", datatype="string")
        self.add_output("sheets", datatype="list")
        self.add_output("csv_by_sheet", datatype="dict")
        self.add_output("json", datatype="string")
        self._finish_form()

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        path = self._source_path(inputs, bc.EXCEL_SUFFIXES)
        if path is None:
            return {"markdown": None, "sheets": None, "csv_by_sheet": None, "json": None}
        doc = bc.convert_excel(
            str(path), should_continue=lambda: not self.is_compute_cancelled())
        self._sync_status = (f"{path.name}: {len(doc['sheets'])} sheet(s) -- "
                             + ", ".join(doc["sheets"][:6])
                             + ("…" if len(doc["sheets"]) > 6 else "") + ".")
        return {"markdown": doc["markdown"], "sheets": doc["sheets"],
                "csv_by_sheet": doc["csv_by_sheet"], "json": doc["json"]}


@register_node
class SilkJatsConverterNode(_SingleFileNode):
    """A JATS XML article as Markdown, with its title and counts."""

    node_name: ClassVar[Optional[str]] = "JATS Converter"
    node_description: ClassVar[Optional[str]] = (
        "Reads a JATS XML article (PubMed Central, arXiv, publisher XML) "
        "with bobine: Markdown with LaTeX math, title, figure/table counts.")
    node_tags: ClassVar[Optional[List[str]]] = ["silk", "bobine", "jats", "xml", "paper"]
    node_icon: ClassVar[Optional[str]] = "article"
    _FILTERS: ClassVar[List[str]] = ["JATS XML (*.xml *.nxml)", "All Files (*)"]
    _SOURCE_TIP: ClassVar[str] = "The article's JATS XML (.xml / .nxml)."

    def __init__(self, title: str = "JATS Converter", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self.add_input("settings", datatype="bobine_settings")
        self.add_output("markdown", datatype="string")
        self.add_output("md_path", datatype="filepath")
        self.add_output("title", datatype="string")
        self.add_output("info", datatype="dict")

        self._out = PathHistoryPicker(mode="folder")
        self._out.setToolTip(
            "Where the Markdown is written. Empty: a 'markdown' folder "
            "beside the article.")
        self._form.addRow("Output:", self._out)
        self._widget_core.register_widget(
            "out_dir", self._out, role=PortRole.INPUT,
            datatype="dirpath", default="", add_to_layout=False)
        self.add_input("out_dir", datatype="dirpath")
        self._pickers["out_dir"] = self._out
        self._finish_form()

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        path = self._source_path(inputs, bc.JATS_SUFFIXES)
        if path is None:
            return {"markdown": None, "md_path": None, "title": None, "info": None}
        out = text_input(inputs, "out_dir").strip()
        target = Path(out).expanduser() if out else path.parent / "markdown"
        doc = bc.convert_jats(
            str(path), str(target), inputs.get("settings"),
            should_continue=lambda: not self.is_compute_cancelled())
        info = {k: doc[k] for k in ("formulas", "figures", "tables")}
        self._sync_status = (
            f"{doc['title'] or path.name}: {info['formulas']} formula(s), "
            f"{info['figures']} figure(s), {info['tables']} table(s).")
        return {"markdown": doc["markdown"], "md_path": doc["md_path"],
                "title": doc["title"], "info": info}
