# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

bobine in Silk: the convert_document tool and the document nodes.

bobine runs in a child process per call, so these run the real worker;
they skip where bobine is not installed. The files are built here: a
text file, a JATS article and a one-sheet workbook (a minimal xlsx, so
no spreadsheet writer is needed).
"""
import asyncio
import json
import sys
import zipfile
from types import SimpleNamespace as NS

import pytest

from silk.functions import bobine_convert as bc

needs_bobine = pytest.mark.skipif(not bc.BOBINE_AVAILABLE, reason="bobine not installed")

JATS = (
    '<?xml version="1.0"?>\n<article><front><article-meta><title-group>'
    "<article-title>A Test</article-title></title-group></article-meta>"
    "</front><body><sec><title>Intro</title><p>Hello <bold>world</bold>."
    "</p></sec></body></article>")

_XLSX = {
    "[Content_Types].xml": (
        '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.'
        'openxmlformats.org/package/2006/content-types"><Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/><Override '
        'PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-'
        'officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/'
        'worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-'
        'officedocument.spreadsheetml.worksheet+xml"/></Types>'),
    "_rels/.rels": (
        '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://'
        'schemas.openxmlformats.org/package/2006/relationships"><Relationship '
        'Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'),
    "xl/workbook.xml": (
        '<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.'
        'openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet '
        'name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>'),
    "xl/_rels/workbook.xml.rels": (
        '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://'
        'schemas.openxmlformats.org/package/2006/relationships"><Relationship '
        'Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>'),
    "xl/worksheets/sheet1.xml": (
        '<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.'
        'openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c '
        'r="A1" t="inlineStr"><is><t>a</t></is></c><c r="B1" t="inlineStr"><is>'
        '<t>b</t></is></c></row><row r="2"><c r="A2"><v>1</v></c><c r="B2"><v>2'
        "</v></c></row></sheetData></worksheet>"),
}


def write_xlsx(path):
    with zipfile.ZipFile(path, "w") as z:
        for name, text in _XLSX.items():
            z.writestr(name, text)
    return path


@pytest.fixture()
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ── settings ────────────────────────────────────────────────────────────

def test_settings_are_checked_against_the_defaults():
    s = bc.clean_settings({"routing": "SURGICAL", "device": "tpu", "formula_dpi": 9000,
                           "extract_images": 0, "junk": 1})
    assert s["routing"] == "surgical" and s["device"] == "auto"
    assert s["formula_dpi"] == 600 and s["extract_images"] is False
    assert "junk" not in s
    assert bc.clean_settings(None) == bc.DEFAULT_SETTINGS
    assert bc.describe({"routing": "never"}) == "route never, tables slanet"
    assert bc.describe({}) == ("route auto, device auto, precision auto, "
                              "tables slanet")


def test_nothing_here_imports_bobine():
    """bobine loads ONNX Runtime on import; Weave's process must not."""
    assert "bobine" not in sys.modules or not bc.BOBINE_AVAILABLE


# ── the worker ──────────────────────────────────────────────────────────

@needs_bobine
def test_a_text_file_and_a_jats_article_convert(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("# Title\n\nSome text.\n", encoding="utf-8")
    out = bc.convert_file(str(note), str(tmp_path / "out"), {"routing": "never"})
    assert "Some text." in out["markdown"] and out["md_path"].endswith("note.md")

    paper = tmp_path / "paper.xml"
    paper.write_text(JATS, encoding="utf-8")
    jats = bc.convert_jats(str(paper), str(tmp_path / "jats"))
    assert jats["title"] == "A Test" and "Hello **world**." in jats["markdown"]


@needs_bobine
def test_a_workbook_gives_tables_csv_and_json(tmp_path):
    doc = bc.convert_excel(str(write_xlsx(tmp_path / "book.xlsx")))
    assert doc["sheets"] == ["Data"]
    assert "| a | b |" in doc["markdown"]
    assert doc["csv_by_sheet"]["Data"].splitlines() == ["a,b", "1,2"]
    assert json.loads(doc["json"])["sheets"][0]["headers"] == ["a", "b"]


@needs_bobine
def test_a_failure_comes_back_as_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="unknown op"):
        bc.run_worker({"op": "nothing"})


def test_cancel_kills_the_child(tmp_path, monkeypatch):
    slow = tmp_path / "slow.py"
    slow.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    monkeypatch.setattr(bc, "WORKER_SCRIPT", str(slow))
    monkeypatch.setattr(bc, "BOBINE_AVAILABLE", True)
    with pytest.raises(bc.ConversionCancelled):
        bc.run_worker({"op": "convert"}, should_continue=lambda: False)


# ── the tool ────────────────────────────────────────────────────────────

def _box(tmp_path, **sandbox_kwargs):
    from silk.functions.tool_box import ToolBox
    from silk.functions.tools.document_tools import attach_document_tools
    from silk.functions.tools.file_sandbox import FileToolSandbox
    box = ToolBox()
    attach_document_tools(box, FileToolSandbox(root_dir=str(tmp_path), **sandbox_kwargs))
    return box


def _call(box, name, **args):
    tc = NS(id="c1", function=NS(name=name, arguments=json.dumps(args)))
    return asyncio.run(box.execute_tool_calls_async([tc]))[0]["content"]


def test_without_bobine_the_group_is_empty(tmp_path, monkeypatch):
    from silk.functions.toolset_build import tool_catalog
    monkeypatch.setattr(bc, "BOBINE_AVAILABLE", False)
    assert tool_catalog(_box(tmp_path)) == []


@needs_bobine
def test_the_tool_converts_into_the_sandbox(tmp_path):
    (tmp_path / "paper.xml").write_text(JATS, encoding="utf-8")
    text = _call(_box(tmp_path, write_enabled=True), "convert_document",
                 path="paper.xml", route="never")
    assert text.startswith("converted paper.xml -> converted/paper/paper.md")
    assert "Hello **world**." in text
    assert (tmp_path / "converted" / "paper" / "paper.md").is_file()


@needs_bobine
def test_the_tool_keeps_to_the_sandbox_and_its_grants(tmp_path):
    (tmp_path / "paper.xml").write_text(JATS, encoding="utf-8")
    read_only = _call(_box(tmp_path, write_enabled=False), "convert_document",
                      path="paper.xml")
    assert "tick a write tool" in read_only
    assert not (tmp_path / "converted").exists()
    outside = _call(_box(tmp_path, write_enabled=True), "convert_document",
                    path="../elsewhere.pdf")
    assert outside.startswith("error: convert refused")


def test_the_documents_group_is_offered():
    from silk.functions.tool_groups import GROUP_ATTACHERS, tools_of
    assert "documents" in GROUP_ATTACHERS
    if bc.BOBINE_AVAILABLE:
        assert tools_of("documents") == frozenset({"convert_document"})


# ── the nodes ───────────────────────────────────────────────────────────

def test_the_settings_node_puts_data_on_its_wire(qt_app):
    from silk.nodes.document_nodes import SilkConverterSettingsNode
    from silk.nodes.silk_ports import BOBINE_SETTINGS_TYPE
    node = SilkConverterSettingsNode()
    try:
        out = node.compute({"routing": "never", "device": "gpu", "formula_dpi": 300})
        assert out["settings"]["routing"] == "never"
        assert out["settings"]["device"] == "gpu"
        assert BOBINE_SETTINGS_TYPE.validator(out["settings"])
    finally:
        node.cleanup()


def test_the_model_report_reads_plainly():
    from silk.nodes.document_nodes import model_text
    text = model_text({"models": [
        {"model_id": "a", "cached": True, "disk_usage_bytes": 2 * 1024 ** 3},
        {"model_id": "b", "cached": False}], "ort": ""})
    assert text.splitlines() == [
        "1 of 2 models cached (2.00 GB).", "Downloaded on first use: b.",
        "ONNX Runtime: none -- PDFs keep to the text layer."]


@needs_bobine
def test_the_converter_takes_a_folder(qt_app, tmp_path):
    from silk.nodes.document_nodes import SilkDocumentConverterNode
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "paper.xml").write_text(JATS, encoding="utf-8")
    (docs / "note.txt").write_text("plain words\n", encoding="utf-8")
    (docs / "skip.bin").write_bytes(b"\0")
    node = SilkDocumentConverterNode()
    try:
        out = node.compute({"folder": str(docs), "settings": {"routing": "never"}})
    finally:
        node.cleanup()
    assert out["output_dir"] == str(docs / "markdown")
    assert [f["source"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            for f in out["report"]["files"]] == ["note.txt", "paper.xml"]
    assert "<!-- paper.xml -->" in out["markdown"] and "plain words" in out["markdown"]


def test_the_converter_says_what_is_missing(qt_app, tmp_path):
    from silk.nodes.document_nodes import SilkDocumentConverterNode
    with pytest.raises(RuntimeError, match="Pick a file"):
        SilkDocumentConverterNode.plan({})
    with pytest.raises(RuntimeError, match="No file"):
        SilkDocumentConverterNode.plan({"source": str(tmp_path / "gone.pdf")})


@needs_bobine
def test_the_excel_and_jats_nodes(qt_app, tmp_path):
    from silk.nodes.document_nodes import SilkExcelConverterNode, SilkJatsConverterNode
    excel = SilkExcelConverterNode()
    jats = SilkJatsConverterNode()
    try:
        book = excel.compute({"source": str(write_xlsx(tmp_path / "book.xlsx"))})
        assert book["sheets"] == ["Data"] and "a,b" in book["csv_by_sheet"]["Data"]
        assert excel.compute({"source": ""})["markdown"] is None
        with pytest.raises(RuntimeError, match="expected"):
            excel.compute({"source": str(write_xlsx(tmp_path / "book.zip"))})
        paper = tmp_path / "paper.xml"
        paper.write_text(JATS, encoding="utf-8")
        article = jats.compute({"source": str(paper)})
        assert article["title"] == "A Test"
        assert article["md_path"].startswith(str(tmp_path / "markdown"))
    finally:
        excel.cleanup()
        jats.cleanup()


def test_a_saved_settings_scene_migrates_to_the_stack_words(qt_app):
    from silk.nodes.document_nodes import SilkConverterSettingsNode
    state = SilkConverterSettingsNode.migrate_state(
        {"widget_data": {"device": "gpu", "precision": "fp32",
                         "quantization": "int8"}}, 1)
    widgets = state["widget_data"]
    assert widgets["device"] == "cuda"
    assert widgets["precision"] == "int8"
    assert "quantization" not in widgets, "the row folded into precision"
    kept = SilkConverterSettingsNode.migrate_state(
        {"widget_data": {"device": "cpu", "precision": "fp16",
                         "quantization": "int8"}}, 1)
    assert kept["widget_data"]["precision"] == "fp16", "an explicit pick wins"
