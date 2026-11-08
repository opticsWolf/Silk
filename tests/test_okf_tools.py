# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The knowledge graph as tools, and the round before a compaction.

Two claims.

**The tools are OKFgraph's ops, nothing more.** Each one forwards its
validated arguments to the served graph and hands the result back as JSON;
an op error comes back as text the model can act on (code, message, the
remedy), never as an exception that ends the run. ``okf_ingest`` is always
a *thoughts* ingest, and does not echo the Markdown it wrote.

**The agent is asked to save its findings before a compaction -- only if it
could.** The round happens when the box names a pre-compaction tool, that
tool is in this run's box, and the active role permits it; a ToolSet that
dropped the tool or a role that may not write gets a plain compaction.
Asked once per compaction, and the compaction follows the next round.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS
from typing import Any

from silk.functions.agent_loop import AgentLoop
from silk.functions.compaction import (
    PRE_COMPACTION_ATTR,
    Compactor,
    pre_compaction_prompt,
)
from silk.functions.okf_graph import OkfError, OkfSpec, mismatch_text
from silk.functions.stream_events import EventCompaction, EventRunResult
from silk.functions.tool_box import ToolBox
from silk.functions.tools.okf_tools import (
    OKF_INGEST_TOOL,
    OKF_LEVEL_ATTR,
    OKF_TOOLS,
    PRE_COMPACTION_PROMPT,
    attach_okf_tools,
    narrow_okf_level,
)

from test_silk_compaction import LoopEngine, _history


class FakeGraph:
    """The OkfGraph surface the tools use: ``call`` and ``spec``."""

    def __init__(self, replies: dict[str, Any] | None = None) -> None:
        self.spec = OkfSpec(db_path="kb.db")
        self.calls: list[tuple[str, dict]] = []
        self.replies = replies or {}

    def call(self, op: str, **kwargs: Any) -> Any:
        self.calls.append((op, kwargs))
        reply = self.replies.get(op, {"op": op})
        if isinstance(reply, Exception):
            raise reply
        return reply


def _box(graph: FakeGraph | None = None, level: str = "write") -> ToolBox:
    box = ToolBox()
    attach_okf_tools(box, None, graph or FakeGraph(), level=level)
    return box


def _call(box: ToolBox, name: str, **args: Any) -> str:
    tc = NS(id="c1", function=NS(name=name, arguments=json.dumps(args)))
    return asyncio.run(box.execute_tool_calls_async([tc]))[0]["content"]


# -- the tools ----------------------------------------------------------------


READERS = {"okf_search", "okf_read", "okf_traverse"}
CURATORS = {"okf_update", "okf_supersede", "okf_delete"}


def _okf(box: ToolBox) -> set[str]:
    """The okf tools in *box*, without the box's own meta tools."""
    return set(box.tools) & set(OKF_TOOLS)


def test_each_level_registers_its_tools_and_those_below():
    assert _okf(_box(level="read")) == READERS
    assert _okf(_box(level="write")) == READERS | {OKF_INGEST_TOOL}
    assert _okf(_box(level="curate")) == set(OKF_TOOLS)
    assert _okf(_box(level="nonsense")) == _okf(_box()), "unknown → write"
    risks = {name: _box(level="curate").tools[name]["risk"] for name in OKF_TOOLS}
    assert {risks[n] for n in READERS} == {"low"}
    assert risks["okf_delete"] == "high"
    assert getattr(_box(level="read"), OKF_LEVEL_ATTR) == "read"


def test_the_catalog_probe_lists_every_tool_and_records_no_grant():
    box = ToolBox()
    attach_okf_tools(box, None, catalog_only=True)
    assert _okf(box) == set(OKF_TOOLS)
    assert getattr(box, OKF_LEVEL_ATTR, None) is None
    assert getattr(box, PRE_COMPACTION_ATTR, None) is None


def test_no_graph_attaches_nothing():
    box = ToolBox()
    attach_okf_tools(box, None, None, level="curate")
    assert not set(box.tools) & set(OKF_TOOLS)


def test_a_read_grant_has_no_round_before_compaction():
    assert pre_compaction_prompt(_box(level="read")) is None


def test_narrowing_only_ever_lowers_the_level():
    box = _box(level="curate")
    assert narrow_okf_level(box, "read") == "read"
    assert _okf(box) == READERS
    assert narrow_okf_level(box, "curate") == "read", "a toolset cannot raise it"
    assert _okf(box) == READERS
    capped = _box(level="write")
    assert narrow_okf_level(capped, "curate") == "write"
    assert not set(capped.tools) & CURATORS
    assert narrow_okf_level(ToolBox(), "read") == "", "no graph, nothing to narrow"


def test_curating_ops_forward_to_the_graph():
    graph = FakeGraph()
    box = _box(graph, level="curate")
    _call(box, "okf_supersede", old_id="a", new_id="b")
    assert graph.calls[-1] == ("supersede", {"old_id": "a", "new_id": "b"})
    _call(box, "okf_delete", concept_id="a")
    op, kwargs = graph.calls[-1]
    assert op == "delete" and kwargs["concept_id"] == "a"


def test_search_forwards_its_arguments_to_the_op():
    graph = FakeGraph({"search": [{"id": "a", "title": "A"}]})
    out = _call(_box(graph), "okf_search", query="line endings", limit=3, rank="hub")
    assert json.loads(out) == [{"id": "a", "title": "A"}]
    op, kwargs = graph.calls[-1]
    assert op == "search"
    assert kwargs["query"] == "line endings" and kwargs["limit"] == 3
    assert kwargs["rank"] == "hub"


def test_ingest_is_a_thoughts_ingest_and_drops_the_markdown():
    graph = FakeGraph({"ingest": {"concept_id": "thoughts/t/1", "markdown": "x" * 5000}})
    out = _call(_box(graph), "okf_ingest", thoughts="CRLF breaks the lexer.",
                topic="lexer", type="evidence", related=["lexer"])
    assert json.loads(out) == {"concept_id": "thoughts/t/1"}
    op, kwargs = graph.calls[-1]
    assert op == "ingest" and kwargs["kind"] == "thoughts"
    assert kwargs["topic"] == "lexer" and kwargs["related"] == ["lexer"]
    assert kwargs["type"] == "evidence"


def test_ingest_takes_only_the_shared_kinds():
    graph = FakeGraph()
    _call(_box(graph), "okf_ingest", thoughts="x", topic="t", type="finding")
    assert not graph.calls, "'finding' is not one of dogma / rule / evidence / decision"


def test_an_op_error_is_a_result_the_model_can_act_on():
    graph = FakeGraph({"read": OkfError("UNKNOWN_CONCEPT", "no such concept",
                                        remedy="search first to find IDs")})
    out = _call(_box(graph), "okf_read", concept_id="nope")
    assert out.startswith("error: UNKNOWN_CONCEPT: no such concept")
    assert "remedy: search first" in out


def test_a_huge_result_is_truncated():
    graph = FakeGraph({"search": ["x" * 100] * 1000})
    out = _call(_box(graph), "okf_search", query="q")
    assert len(out) < 21_000 and "truncated" in out


# -- whether the round before a compaction happens ----------------------------


def test_the_pack_marks_ingest_as_the_pre_compaction_tool():
    box = _box()
    assert getattr(box, PRE_COMPACTION_ATTR)[0] == OKF_INGEST_TOOL
    assert pre_compaction_prompt(box) == PRE_COMPACTION_PROMPT


def test_no_prompt_without_the_pack_or_without_a_box():
    assert pre_compaction_prompt(ToolBox()) is None
    assert pre_compaction_prompt(None) is None


def test_no_prompt_when_a_toolset_dropped_the_tool():
    box = _box()
    box.unregister(OKF_INGEST_TOOL)      # what prune_to_selection does
    assert pre_compaction_prompt(box) is None


def test_no_prompt_when_the_role_does_not_permit_the_tool():
    box = _box()
    box.set_role_filter(lambda name, _meta: name != OKF_INGEST_TOOL)
    assert pre_compaction_prompt(box) is None
    box.set_role_filter(lambda _name, _meta: True)
    assert pre_compaction_prompt(box) == PRE_COMPACTION_PROMPT


# -- the round, in the loop ---------------------------------------------------


def _nudged(engine: LoopEngine) -> list[dict]:
    return [e for e in engine.history if e.get("content") == PRE_COMPACTION_PROMPT]


def test_pressure_asks_for_the_round_then_compacts_the_next_one():
    graph = FakeGraph()
    engine = LoopEngine(
        _history(8), context=8000, tokens=7000,
        replies=['```tool_call\n{"name": "okf_ingest", "arguments": '
                 '{"thoughts": "CRLF breaks the lexer.", "topic": "lexer"}}\n```',
                 "done"],
    )
    seen: list[str] = []
    original = engine.stream_response

    def spy(params):
        seen.append(engine.history[-1]["content"])
        return original(params)

    engine.stream_response = spy
    loop = AgentLoop(engine, _box(graph), compactor=Compactor(keep_recent=8, min_dropped=4))
    events = list(loop.run("go"))

    assert seen[0] == PRE_COMPACTION_PROMPT, "asked before the first request"
    assert graph.calls and graph.calls[0][0] == "ingest", "and it saved"
    compactions = [e for e in events if isinstance(e, EventCompaction)]
    assert len(compactions) == 1, "compacted once, after the round"
    assert events.index(compactions[0]) > 0
    assert engine.requests == 2 and isinstance(events[-1], EventRunResult)


def test_a_role_that_may_not_ingest_just_compacts():
    engine = LoopEngine(_history(8), context=8000, tokens=7000)
    box = _box()
    box.set_role_filter(lambda name, _meta: name != OKF_INGEST_TOOL)
    events = list(AgentLoop(engine, box, compactor=Compactor(
        keep_recent=8, min_dropped=4)).run("go"))
    assert not _nudged(engine)
    assert sum(isinstance(e, EventCompaction) for e in events) == 1
    assert engine.requests == 1


def test_no_pressure_no_round():
    engine = LoopEngine(_history(2), context=100_000, tokens=100)
    list(AgentLoop(engine, _box(), compactor=Compactor()).run("go"))
    assert not _nudged(engine)


def test_a_compaction_request_waits_for_the_round():
    from silk.functions.compaction import COMPACTION_REQUESTS_ATTR

    engine = LoopEngine(
        _history(8), context=100_000, tokens=100,
        replies=['```tool_call\n{"name": "okf_search", "arguments": {"query": "q"}}\n```',
                 "done"],
    )
    box = _box()
    setattr(box, COMPACTION_REQUESTS_ATTR, ["keep the ids"])
    loop = AgentLoop(engine, box, compactor=Compactor(auto=False, keep_recent=4))
    events = list(loop.run("go"))

    assert len(_nudged(engine)) == 1
    compactions = [e for e in events if isinstance(e, EventCompaction)]
    assert len(compactions) == 1, "the held request compacted the next round"
    assert "keep the ids" in engine.summarized[0]


# -- pins ---------------------------------------------------------------------


def test_mismatch_text_names_each_broken_pin_against_the_expected_value():
    pins = {
        "empty": False, "mismatch": ["model_id", "precision"],
        "pinned": {"model_id": "small", "precision": "fp32"},
        "configured": {"model_id": "nano", "precision": "auto",
                       "expected_precision": "fp16"},
    }
    text = mismatch_text(pins)
    assert "model_id: graph has small, settings give nano" in text
    assert "precision: graph has fp32, settings give fp16" in text


def test_an_empty_graph_never_mismatches():
    assert mismatch_text({"empty": True, "mismatch": ["model_id"]}) == ""
    assert mismatch_text({"empty": False, "mismatch": []}) == ""
