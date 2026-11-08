# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The public-web pack: skill-distilled descriptions, budgets, soft-dep.

Qt-free throughout: the pack is functions-level (pydantic + stdlib) and
the backend import is lazy. Runs everywhere; the live-registration test
needs gossamer importable and skips otherwise.
"""
import asyncio
import json
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from silk.functions.tools import web_tools


def test_missing_backend_registers_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(web_tools, "_WebResearcherToolbox", None)
    monkeypatch.setattr(web_tools, "_GOSSAMER_ERROR", "nope")
    from silk.functions.tool_box import ToolBox
    from silk.functions.tools.file_sandbox import FileToolSandbox
    from silk.functions.toolset_build import tool_catalog
    box = ToolBox()
    web_tools.attach_web_tools(box, FileToolSandbox(root_dir=str(tmp_path)))
    assert tool_catalog(box) == []


def test_args_caps_are_skill_budgets():
    with pytest.raises(ValidationError):
        web_tools.WebSearchArgs(query="")
    with pytest.raises(ValidationError):
        web_tools.WebSearchArgs(query="x", max_results=99)
    with pytest.raises(ValidationError):
        web_tools.WebSearchArgs(query="x", depth=9)
    with pytest.raises(ValidationError):
        web_tools.CheckArgs(urls=[])
    with pytest.raises(ValidationError):
        web_tools.ExtractArgs(source="x.pdf", tables_as="yaml")
    with pytest.raises(ValidationError):
        web_tools.ResearchArgs(query="x", category="astrology")
    # OpenAlex-only knobs accepted at parse time; the backend rejects
    # them for other providers (the skill documents the rejection).
    args = web_tools.ResearchArgs(query="x", title="y", author="z")
    assert args.title == "y"
    # gossamer 0.11: a scholarly merge across up to four providers.
    merged = web_tools.ResearchArgs(query="x", providers=["openalex", "arxiv"])
    assert merged.model_dump(exclude_none=True)["providers"] == ["openalex", "arxiv"]
    with pytest.raises(ValidationError):
        web_tools.ResearchArgs(query="x", providers=["a", "b", "c", "d", "e"])
    with pytest.raises(ValidationError):
        web_tools.DownloadArgs(source="http://x/y.pdf", filename="")
    for bad in ("", ".", "..", "sub/dir.pdf", "/abs.pdf", "a\\b.pdf"):
        with pytest.raises(ValueError):
            web_tools._safe_filename(bad)
    assert web_tools._safe_filename("paper final.pdf") == "paper final.pdf"


def _fake_backend(monkeypatch, calls):
    class FakeBox:
        def execute_tool(self, name, args):
            calls.append((name, args))
            return json.dumps({"output_path": args.get("output_path")})
    monkeypatch.setattr(web_tools, "_WebResearcherToolbox", object)
    monkeypatch.setattr(web_tools, "_TOOLBOX", FakeBox())


def _box_with_sandbox(tmp_path, monkeypatch, calls, **sandbox_kwargs):
    from silk.functions.tool_box import ToolBox
    from silk.functions.tools.file_sandbox import FileToolSandbox
    _fake_backend(monkeypatch, calls)
    box = ToolBox()
    web_tools.attach_web_tools(
        box, FileToolSandbox(root_dir=str(tmp_path), **sandbox_kwargs))
    return box


def _call(box, name, **args):
    tc = NS(id="c1", function=NS(name=name, arguments=json.dumps(args)))
    return asyncio.run(box.execute_tool_calls_async([tc]))[0]["content"]


def test_download_lands_in_sandbox_downloads(tmp_path, monkeypatch):
    calls = []
    box = _box_with_sandbox(tmp_path, monkeypatch, calls)
    out = _call(box, "web_download", source="http://x/y.pdf",
                filename="paper.pdf", expected_format="pdf")
    name, sent = calls[-1]
    assert name == "download_file"
    target = tmp_path / "downloads" / "paper.pdf"
    assert sent["output_path"] == str(target)
    assert sent["expected_format"] == "pdf"
    assert target.parent.is_dir()
    assert out.startswith("saved: downloads/paper.pdf")


def test_download_refuses_escapes_and_readonly(tmp_path, monkeypatch):
    calls = []
    box = _box_with_sandbox(tmp_path, monkeypatch, calls)
    out = _call(box, "web_download", source="http://x/y.pdf",
                filename="../evil.pdf")
    assert out.startswith("error: bad download request:")
    assert calls == []
    frozen = _box_with_sandbox(tmp_path, monkeypatch, calls,
                               write_enabled=False)
    out = _call(frozen, "web_download", source="http://x/y.pdf",
                filename="paper.pdf")
    assert "remedy:" in out and "write" in out.lower()
    assert calls == []


def test_extract_reads_local_files_only_through_the_sandbox(tmp_path, monkeypatch):
    calls = []
    root = tmp_path / "root"
    root.mkdir()
    (root / "paper.pdf").write_bytes(b"%PDF-1.4")
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"%PDF-1.4")
    box = _box_with_sandbox(root, monkeypatch, calls)

    for remote in ("https://x/y.pdf", "10.1371/journal.pone.0000001",
                   "doi:10.1000/xyz"):
        _call(box, "extract_document", source=remote)
        assert calls[-1] == ("extract_document", {
            "source": remote, "tables_as": "json"})

    _call(box, "extract_document", source="paper.pdf")
    assert calls[-1][1]["source"] == str((root / "paper.pdf").resolve())

    sent = len(calls)
    for escape in (str(outside), "../secret.pdf",
                   str(tmp_path / "10.1234" / "a.pdf"), "file:///etc/passwd",
                   "missing.pdf"):
        out = _call(box, "extract_document", source=escape)
        assert out.startswith("error: extract refused:"), (escape, out)
        assert "remedy:" in out
    assert len(calls) == sent, "nothing outside the sandbox reached gossamer"

    blind = _box_with_sandbox(root, monkeypatch, calls, read_enabled=False)
    out = _call(blind, "extract_document", source="paper.pdf")
    assert out.startswith("error: extract refused:") and len(calls) == sent


def test_run_without_backend_is_remedy_not_traceback(monkeypatch):
    monkeypatch.setattr(web_tools, "_TOOLBOX", None)
    monkeypatch.setattr(web_tools, "_WebResearcherToolbox", None)
    out = web_tools._run("web_search", {"query": "x"})
    assert out.startswith("error:")
    assert "remedy:" in out


@pytest.mark.skipif(
    web_tools._WebResearcherToolbox is None,
    reason="gossamer-web not importable here",
)
def test_registration_carries_the_skill():
    from silk.functions.tool_groups import group_catalog
    entries = {e["name"]: e for e in group_catalog().get("web", ())}
    assert set(entries) == set(web_tools.WEB_TOOLS)
    for entry in entries.values():
        assert entry["description"].strip()
        assert entry["category"] == "web"
    risks = {name: entries[name]["risk"] for name in web_tools.WEB_TOOLS}
    assert risks.pop("web_download") == "medium"
    assert set(risks.values()) == {"low"}
    blob = " ".join(e["description"] for e in entries.values())
    # Routing, budgets, cache and failure handling -- the skill's four
    # load-bearing sections -- must reach the model verbatim enough.
    for phrase in (
        "research_by_category",  # search points at routed research
        "check_sources",  # candidates get probed before fetching
        "off-machine",  # the egress the tick grants
        "cache",  # cached reads vs live probes
        "exact variable",  # keyed failures: surface, don't retry
    ):
        assert phrase in blob, phrase
