# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

What happens before a compaction: off, ask, or require.

The automatic triggers go by the ``auto_compact`` hook's ``ingest_before``,
a ``compact_context`` request by the context tools' own setting. ``off``
compacts at once; ``ask`` gives the agent one round with the prompt;
``require`` also forces the round's request to call ``okf_ingest`` where
the engine can force a tool, and falls back to asking where it cannot.
"""

from __future__ import annotations

from silk.functions.agent_loop import AgentLoop
from silk.functions.compaction import (
    COMPACTION_REQUESTS_ATTR,
    INGEST_ASK,
    INGEST_OFF,
    INGEST_REQUIRE,
    Compactor,
    compactor_for,
    request_ingest_mode,
)
from silk.functions.hook_catalog import attach_catalog_hooks
from silk.functions.stream_events import EventCompaction
from silk.functions.tool_box import ToolBox
from silk.functions.tools.context_tools import attach_context_tools
from silk.functions.tools.okf_tools import OKF_INGEST_TOOL, PRE_COMPACTION_PROMPT

from test_okf_tools import FakeGraph, _box
from test_silk_compaction import LoopEngine, _history


class ForcingEngine(LoopEngine):
    """A LoopEngine that can force a tool, and records what it forced."""

    def __init__(self, *args, can_force=True, **kw):  # noqa: D107
        super().__init__(*args, **kw)
        self.can_force = can_force
        self.forced: list[str] = []

    def force_tool_once(self, name):
        if self.can_force:
            self.forced.append(name)
        return self.can_force


def _nudged(engine) -> int:
    return sum(e.get("content") == PRE_COMPACTION_PROMPT for e in engine.history)


def _pressure(mode: str, **kw):
    engine = ForcingEngine(_history(8), context=8000, tokens=7000, **kw)
    compactor = Compactor(keep_recent=8, min_dropped=4, ingest_before=mode)
    events = list(AgentLoop(engine, _box(FakeGraph()), compactor=compactor).run("go"))
    return engine, events


def test_off_compacts_without_a_round():
    engine, events = _pressure(INGEST_OFF)
    assert _nudged(engine) == 0 and not engine.forced
    assert sum(isinstance(e, EventCompaction) for e in events) == 1
    assert engine.requests == 1


def test_ask_asks_and_forces_nothing():
    engine, _ = _pressure(INGEST_ASK)
    assert _nudged(engine) == 1 and not engine.forced


INGEST_CALL = ('```tool_call\n{"name": "okf_ingest", "arguments": '
               '{"thoughts": "CRLF breaks the lexer.", "topic": "lexer"}}\n```')


def test_require_forces_the_ingest_tool():
    engine, events = _pressure(INGEST_REQUIRE, replies=[INGEST_CALL, "done"])
    assert _nudged(engine) == 1
    assert engine.forced == [OKF_INGEST_TOOL]
    assert sum(isinstance(e, EventCompaction) for e in events) == 1, (
        "the round's call is answered, then the compaction runs")


def test_require_asks_where_the_engine_cannot_force():
    engine, _ = _pressure(INGEST_REQUIRE, can_force=False)
    assert _nudged(engine) == 1 and not engine.forced


def test_a_request_goes_by_the_context_tools_setting():
    engine = ForcingEngine(_history(8), context=100_000, tokens=100)
    box = _box()
    attach_context_tools(box, ingest_before=INGEST_OFF)
    setattr(box, COMPACTION_REQUESTS_ATTR, ["keep the ids"])
    # The hook's setting says require; the request's says off.
    loop = AgentLoop(engine, box, compactor=Compactor(
        auto=False, keep_recent=4, ingest_before=INGEST_REQUIRE))
    events = list(loop.run("go"))
    assert _nudged(engine) == 0 and not engine.forced
    assert sum(isinstance(e, EventCompaction) for e in events) == 1


def test_settings_reach_the_compactor_and_the_box():
    box = ToolBox()
    attach_catalog_hooks(box, None, ["auto_compact"],
                         {"auto_compact": {"ingest_before": "require"}})
    assert compactor_for(box).ingest_before == INGEST_REQUIRE
    assert Compactor().ingest_before == INGEST_ASK, "the default is the old round"

    attach_context_tools(box, ingest_before="nonsense")
    assert request_ingest_mode(box) == INGEST_ASK
    attach_context_tools(box, ingest_before="require")
    assert request_ingest_mode(box) == INGEST_REQUIRE
    assert request_ingest_mode(None) == INGEST_ASK


def test_the_engine_forces_only_a_tool_it_advertises():
    from silk.functions.graph_engine import GraphEngine

    engine = GraphEngine.__new__(GraphEngine)
    engine._forced_tool = None
    engine._native_tools_enabled = False
    engine._tool_schemas = []
    assert engine.force_tool_once(OKF_INGEST_TOOL) is False, "fences cannot force"
    engine.enable_native_tools([{"type": "function",
                                 "function": {"name": OKF_INGEST_TOOL}}])
    assert engine.force_tool_once("okf_delete") is False
    assert engine.force_tool_once(OKF_INGEST_TOOL) is True
    assert engine._forced_tool == OKF_INGEST_TOOL
