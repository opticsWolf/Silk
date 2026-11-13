# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

bobine in a child process: one request on stdin, JSON lines on stdout.

Run by :mod:`functions.bobine_convert`, never imported into Weave's
process: ``import bobine`` loads ONNX Runtime when one is installed, and
a second ONNX Runtime in Weave's process can take the interpreter down
(the reason okf and the rerank server run apart too). It imports nothing
from Silk, so it runs as a plain script.

The request is ``{"op": ..., ...}``; settings arrive already cleaned by
``bobine_convert.clean_settings``. Progress lines are ``{"page": i,
"total": n}``; the last line is ``{"ok": true, "result": ...}`` or
``{"ok": false, "error": "..."}``.

Ops: ``convert`` (``path``, ``output_dir``, ``settings``) -- any file
bobine reads; ``excel`` (``path``) -- a workbook's Markdown, sheets as
CSV, and JSON; ``jats`` (``path``, ``output_dir``, ``settings``) -- a JATS
article with its title and counts; ``models`` (``cache_dir``) -- the model
cache report, plus whether an ONNX Runtime was found.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict



def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def config_kwargs(bob: Any, s: Dict[str, Any]) -> Dict[str, Any]:
    """``ConverterConfig`` keyword arguments from cleaned settings.

    ``device`` and ``precision`` travel as the stack vocabulary bobine 0.9
    takes (``auto|cpu|cuda``, ``auto|fp32|fp16|int8``); the enum-valued
    rows become the enum objects, because a string cannot be one. Kept
    separate from :func:`make_config` so the warm host translates its
    settings with the same code.
    """
    return {
        "routing_mode": getattr(bob.RoutingMode,
                                s.get("routing", "auto").capitalize()),
        "device": s.get("device", "auto"),
        "precision": s.get("precision", "auto"),
        "table_backend": getattr(bob.TableBackend,
                                 s.get("table_backend", "slanet").capitalize()),
        "image_mode": (bob.ImageMode.EmbedBase64 if s.get("image_mode") == "embed"
                       else bob.ImageMode.Link),
        "extract_images": bool(s.get("extract_images", True)),
        "detect_headings": bool(s.get("detect_headings", True)),
        "ocr_lang": str(s.get("ocr_lang") or "en"),
        "formula_dpi": int(s.get("formula_dpi", 200)),
        "render_dpi": int(s.get("render_dpi", 300)),
    }


def make_config(bob: Any, s: Dict[str, Any]) -> Any:
    """A ``ConverterConfig`` from cleaned settings."""
    return bob.ConverterConfig(**config_kwargs(bob, s))


def run(request: Dict[str, Any]) -> Dict[str, Any]:
    import bobine as bob

    op = request.get("op")
    settings = request.get("settings") or {}
    cache = str(settings.get("cache_dir") or request.get("cache_dir") or "") or None
    if op == "convert":
        out = Path(request["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        doc = bob.ingest_document(
            str(request["path"]), str(out), config=make_config(bob, settings),
            on_page=lambda i, n: _emit({"page": int(i), "total": int(n)}),
            cache_dir=cache)
        return {"source": str(request["path"]), "md_path": doc.md_path,
                "markdown": doc.md_text, "pages": doc.page_count,
                "images": doc.image_count, "image_dir": doc.image_dir,
                "data_files": list(doc.data_files)}
    if op == "excel":
        doc = bob.convert_excel(str(request["path"]))
        return {"markdown": doc.markdown, "sheets": list(doc.sheet_names),
                "csv_by_sheet": dict(doc.csv_by_sheet), "json": doc.json}
    if op == "jats":
        out = Path(request["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        doc = bob.convert_jats(str(request["path"]), str(out), make_config(bob, settings))
        md_path = doc.md_path
        return {"md_path": md_path,
                "markdown": Path(md_path).read_text(encoding="utf-8") if md_path else "",
                "title": doc.title, "formulas": doc.n_formulas,
                "figures": doc.n_figures, "tables": doc.n_tables}
    if op == "models":
        return {"models": list(bob.model_status(cache)),
                "ort": bob.ORT_DYLIB_PATH or ""}
    raise ValueError(f"unknown op {op!r}")


def main() -> None:
    try:
        request = json.loads(sys.stdin.read() or "{}")
        _emit({"ok": True, "result": run(request)})
    except Exception as exc:  # noqa: BLE001 -- reported to the parent
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
