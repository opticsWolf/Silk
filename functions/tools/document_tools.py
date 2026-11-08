# -*- coding: utf-8 -*-
"""Document tools: a sandbox file to Markdown with bobine.

``convert_document`` reads a PDF, Office file (DOCX/XLSX/PPTX and the
legacy formats), JATS XML article or text file inside the sandbox and
writes ``<output_dir>/<stem>/<stem>.md`` beside its images and, for
workbooks, a CSV per sheet -- so the file tools can read, search and cite
the result -- and returns the Markdown, truncated to ``max_chars``.

The input obeys the sandbox's read grant like the file tools; the output
needs its write grant, which (as for ``web_download``) is the tick on a
write tool, not this one: converting a file is a request to read it.

``extract_document`` (the web group) also reads local documents, as text
through gossamer. This is the heavier, layout-aware path: formula OCR into
LaTeX and table structure on PDF pages, with the models bobine caches.
Without bobine installed the group probes to empty and the tree does not
list it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from weave.logger import get_logger

from .. import bobine_convert

log = get_logger("SilkDocumentTools")


class ConvertDocumentArgs(BaseModel):
    path: str = Field(..., min_length=1, description=(
        "The document, relative to the sandbox root (PDF, DOCX/XLSX/PPTX, "
        "DOC/XLS/PPT, JATS .xml/.nxml, text)."))
    output_dir: str = Field("converted", description=(
        "Folder for the result, relative to the sandbox root; the Markdown "
        "lands in <output_dir>/<file stem>/."))
    route: Literal["never", "surgical", "auto", "always"] = Field("auto", description=(
        "PDF pages only. never: text layer, fastest. surgical: + formula OCR "
        "into LaTeX. auto: + layout/OCR on pages that need it (scans, "
        "tables). always: every page through layout/OCR, slowest."))
    max_chars: int = Field(20000, ge=500, le=200000, description=(
        "Return at most this much Markdown; the file on disk is complete."))


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def attach_document_tools(toolbox: Any, sandbox: Any) -> None:
    """Register ``convert_document``. Recipe-shaped like every group."""
    if not bobine_convert.BOBINE_AVAILABLE:
        log.info(
            "document tools unavailable (%s); %s",
            bobine_convert.BOBINE_ERROR, bobine_convert.INSTALL_HINT,
        )
        return

    @toolbox.register(
        name="convert_document",
        description=(
            "Convert a document in the sandbox to Markdown with bobine "
            "(PDF with formulas as LaTeX and real tables, Office files, "
            "JATS articles). Writes <output_dir>/<stem>/<stem>.md plus "
            "images and per-sheet CSVs, and returns the Markdown. Slower "
            "than extract_document; use it when layout, math or tables "
            "matter, or the result should stay on disk."
        ),
        args_model=ConvertDocumentArgs,
        tags=("file", "document", "bobine"), category="file", risk="medium",
    )
    def convert_document(_db: Any, _session: Any, **args: Any) -> str:
        try:
            parsed = ConvertDocumentArgs(**args)
        except ValidationError as exc:
            return f"error: bad convert request: {exc}"
        try:
            sandbox.check_read()
            source = sandbox.resolve_path(parsed.path)
            if not source.is_file():
                raise ValueError(f"no file at {parsed.path!r} in the sandbox")
            sandbox.check_extension(source)
            if source.suffix.lower() not in bobine_convert.SUFFIXES:
                raise ValueError(f"bobine does not read {source.suffix or 'extensionless'} files")
        except (PermissionError, ValueError, OSError) as exc:
            return f"error: convert refused: {exc}"
        try:
            sandbox.check_write()
            target = sandbox.resolve_path(str(Path(parsed.output_dir) / source.stem))
            sandbox.assert_writable(target)
        except (PermissionError, ValueError) as exc:
            return (
                f"error: convert refused: {exc}\nremedy: the result is "
                "written to disk -- tick a write tool (e.g. write_file) to "
                "grant write access, then retry"
            )
        route, note = parsed.route, ""
        if route != "never" and not bobine_convert.ort_installed():
            # The model routes need an ONNX Runtime; the text layer does not.
            route, note = "never", " (no ONNX Runtime installed: text layer only)"
        try:
            with sandbox.lock_paths(target):
                result = bobine_convert.convert_file(
                    str(source), str(target), {"routing": route})
        except Exception as exc:  # noqa: BLE001 -- reported to the model
            return f"error: conversion failed: {type(exc).__name__}: {exc}"
        root = Path(sandbox.root_dir)
        markdown = result["markdown"] or ""
        head = [f"converted {parsed.path} -> {_relative(Path(result['md_path']), root)}"
                f" (route {route}{note}; {result['pages']} page(s), "
                f"{result['images']} image(s))"]
        if result["data_files"]:
            head.append("data: " + ", ".join(
                _relative(Path(p), root) for p in result["data_files"]))
        if len(markdown) > parsed.max_chars:
            markdown = (markdown[: parsed.max_chars]
                        + f"\n\n[truncated: {len(result['markdown'])} chars in the file]")
        return "\n".join(head) + "\n\n" + markdown
