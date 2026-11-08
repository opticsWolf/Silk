# -*- coding: utf-8 -*-
"""Documents to Markdown with bobine, for the agent tool and the nodes.

bobine (``pip install bobine``; Silk's ``documents`` extra) turns PDF,
Office (DOCX/XLSX/PPTX and the legacy formats), JATS XML and text files
into Markdown, with formula OCR and layout models for the PDF pages that
need them. One call does every format -- ``bobine.ingest_document``
routes by file type -- so the agent tool and the Document Converter node
share :func:`convert_file`; the Excel and JATS nodes add what only those
formats have (sheets as CSV and JSON; a paper's title and counts).

**bobine runs in a child process** (``functions/bobine_worker.py``), one
per call. ``import bobine`` loads ONNX Runtime when one is installed, and
a second ONNX Runtime in Weave's process can take the interpreter down --
the reason okf and the rerank server run apart too. A child also makes
cancel exact: stopping a conversion kills it, mid-page or not. Nothing
here imports bobine; availability is whether the package can be found.

**Settings travel as data.** The Converter Settings node puts a plain
dict on its wire (``bobine_settings`` port), never a live
``ConverterConfig``: a dict saves with the scene, shows in a tooltip and
compares by value, and the worker builds the config. A key left out is
bobine's own default.

**The fast paths need nothing else.** Office, JATS and text files, and
PDFs on the ``never`` route, run in bobine's Rust core. The model routes
(``auto``, ``surgical``, ``always``) need an ONNX Runtime library, which
bobine finds in a pip-installed ``onnxruntime``; Silk's extra leaves that
to the environment, because okfgraph pins its own onnxruntime and two
pins cannot share one environment.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

#: Whether bobine is installed (found, not imported: see the module doc).
BOBINE_AVAILABLE = importlib.util.find_spec("bobine") is not None
BOBINE_ERROR: Optional[str] = None if BOBINE_AVAILABLE else "bobine is not installed"

#: Remedy shown wherever bobine is missing.
INSTALL_HINT = "pip install bobine (Silk's `documents` extra)"

#: The script each call runs, by path: the child needs bobine, not Silk.
WORKER_SCRIPT = str(Path(__file__).with_name("bobine_worker.py"))

#: Routes, cheapest first. ``never`` is the text layer only; ``surgical``
#: adds formula OCR; ``auto`` sends flagged pages through layout + OCR;
#: ``always`` sends every page.
ROUTES = ("never", "surgical", "auto", "always")
DEVICES = ("cpu", "gpu")
PRECISIONS = ("fp32", "fp16")
QUANTIZATIONS = ("fp32", "int8")
TABLE_BACKENDS = ("slanet", "potatr", "auto")
IMAGE_MODES = ("link", "embed")

#: What a fresh Converter Settings node puts on its wire (bobine's own
#: defaults, spelled out so the node and the tool agree).
DEFAULT_SETTINGS: Dict[str, Any] = {
    "routing": "auto", "device": "cpu", "precision": "fp32",
    "quantization": "fp32", "table_backend": "slanet",
    "extract_images": True, "image_mode": "link", "detect_headings": True,
    "ocr_lang": "en", "formula_dpi": 200, "render_dpi": 300, "cache_dir": "",
}

#: File types bobine reads, by suffix (lower case, with the dot).
SUFFIXES = frozenset({
    ".pdf", ".docx", ".doc", ".xlsx", ".xlsm", ".xls", ".xlsb", ".ods",
    ".pptx", ".ppt", ".xml", ".nxml", ".txt", ".md", ".markdown", ".rst",
    ".html", ".htm", ".csv", ".json",
})
EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls", ".xlsb", ".ods"})
JATS_SUFFIXES = frozenset({".xml", ".nxml"})


class ConversionCancelled(RuntimeError):
    """The caller stopped the conversion; the child was killed."""


def ort_installed() -> bool:
    """Whether bobine's model routes will find an ONNX Runtime (the
    ``onnxruntime`` and ``onnxruntime-gpu`` packages both install it)."""
    return importlib.util.find_spec("onnxruntime") is not None


def clean_settings(settings: Any) -> Dict[str, Any]:
    """*settings* over the defaults, each key checked; junk keys dropped."""
    out = dict(DEFAULT_SETTINGS)
    if not isinstance(settings, dict):
        return out
    for key, choices in (("routing", ROUTES), ("device", DEVICES),
                         ("precision", PRECISIONS), ("quantization", QUANTIZATIONS),
                         ("table_backend", TABLE_BACKENDS), ("image_mode", IMAGE_MODES)):
        value = str(settings.get(key) or "").strip().lower()
        if value in choices:
            out[key] = value
    for key in ("extract_images", "detect_headings"):
        if key in settings:
            out[key] = bool(settings[key])
    for key, low, high in (("formula_dpi", 72, 600), ("render_dpi", 72, 600)):
        try:
            out[key] = min(high, max(low, int(settings.get(key, out[key]))))
        except (TypeError, ValueError):
            pass
    lang = str(settings.get("ocr_lang") or "").strip()
    if lang:
        out["ocr_lang"] = lang
    out["cache_dir"] = str(settings.get("cache_dir") or "").strip()
    return out


def describe(settings: Any) -> str:
    """One line for a tooltip or a status label."""
    s = clean_settings(settings)
    models = "" if s["routing"] == "never" else (
        f", {s['device'].upper()} {s['precision']}"
        + (" int8 formulas" if s["quantization"] == "int8" and s["device"] == "cpu" else ""))
    return f"route {s['routing']}{models}, tables {s['table_backend']}"


def run_worker(request: Dict[str, Any], *,
               should_continue: Optional[Callable[[], bool]] = None,
               on_page: Optional[Callable[[int, int], None]] = None,
               timeout: Optional[float] = None) -> Dict[str, Any]:
    """Run one request in a bobine child; its result, or a RuntimeError.

    *should_continue* is polled while the child works; False kills it and
    raises :class:`ConversionCancelled`. *on_page* gets ``(page, total)``
    as the child reports pages.
    """
    if not BOBINE_AVAILABLE:
        raise RuntimeError(f"{BOBINE_ERROR}: {INSTALL_HINT}")
    process = subprocess.Popen(
        [sys.executable, WORKER_SCRIPT], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace")
    stopped = threading.Event()
    finished = threading.Event()

    def watch() -> None:
        # Polls the caller's cancel flag (and the deadline) while the child
        # works; either one kills it.
        waited = 0.0
        while not finished.wait(0.2):
            waited += 0.2
            if ((should_continue is not None and not should_continue())
                    or (timeout is not None and waited > timeout)):
                stopped.set()
                process.kill()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    final: Optional[Dict[str, Any]] = None
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(request))
        process.stdin.close()
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "page" in message and on_page is not None:
                on_page(int(message["page"]), int(message.get("total") or 0))
            elif "ok" in message:
                final = message
        process.wait()
    finally:
        finished.set()
        if process.poll() is None:
            process.kill()
            process.wait()
        stderr = process.stderr.read() if process.stderr else ""
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
    if stopped.is_set():
        raise ConversionCancelled("conversion stopped")
    if final is None:
        tail = (stderr or "").strip()[-1500:]
        raise RuntimeError(f"bobine exited without a result (code "
                           f"{process.returncode}){': ' + tail if tail else ''}")
    if not final.get("ok"):
        raise RuntimeError(str(final.get("error") or "bobine failed"))
    return dict(final.get("result") or {})


def model_report(settings: Any = None) -> Dict[str, Any]:
    """``{models: [...], ort: path}``: bobine's model cache, and the ONNX
    Runtime its model routes would use (empty when none)."""
    return run_worker({"op": "models",
                       "cache_dir": clean_settings(settings)["cache_dir"]})


def convert_file(path: str, output_dir: str, settings: Any = None, *,
                 should_continue: Optional[Callable[[], bool]] = None,
                 on_page: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """Convert one file into *output_dir*; what came out, as plain data.

    ``{source, md_path, markdown, pages, images, image_dir, data_files}``.
    """
    return run_worker({"op": "convert", "path": str(path),
                       "output_dir": str(output_dir),
                       "settings": clean_settings(settings)},
                      should_continue=should_continue, on_page=on_page)


def files_in(folder: str, recursive: bool = False) -> List[Path]:
    """The files under *folder* bobine reads, sorted."""
    root = Path(folder)
    found = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in found if p.is_file() and p.suffix.lower() in SUFFIXES)


def convert_excel(path: str, *, should_continue: Optional[Callable[[], bool]] = None
                  ) -> Dict[str, Any]:
    """``{markdown, sheets, csv_by_sheet, json}`` for a workbook."""
    return run_worker({"op": "excel", "path": str(path)},
                      should_continue=should_continue)


def convert_jats(path: str, output_dir: str, settings: Any = None, *,
                 should_continue: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """``{md_path, markdown, title, formulas, figures, tables}`` for a
    JATS XML article (PubMed Central, arXiv-JATS, publisher XML)."""
    return run_worker({"op": "jats", "path": str(path), "output_dir": str(output_dir),
                       "settings": clean_settings(settings)},
                      should_continue=should_continue)
