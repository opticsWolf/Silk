# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The decision log, the knowledge sink, and the hooks that use them.

* The approval gate writes every answered question to the decision log, and
  the next triage of that tool reads the earlier answers back.
* A wired Knowledge Graph puts an :class:`OkfKnowledge` on the box: rules and
  dogmas come back for triage, findings go in as their kind.
* ``findings_gate`` files a reasoned user decision into the graph.
* ``classify_ingest`` types an untyped ``okf_ingest`` note, and overrides the
  agent's own type only when the classifier is very sure.
* The four kinds are the same everywhere.
"""

from __future__ import annotations

import threading

from silk.functions.classifier_binding import bind_run_classifier
from silk.functions.classifier_hooks import FINDINGS_QUESTIONS, INGEST_QUESTIONS
from silk.functions.decision_log import DECISION_LOG_ATTR, DecisionLog, decision_log_of
from silk.functions.decision_seam import DecisionSeam
from silk.functions.hook_catalog import attach_catalog_hooks
from silk.functions.knowledge_kinds import KNOWLEDGE_KINDS
from silk.functions.okf_knowledge import KNOWLEDGE_ATTR, PROPOSED_TAG, OkfKnowledge
from silk.functions.remember import RunIdentity, bind_run_identity
from silk.functions.tools.okf_tools import OkfIngestArgs

from test_classifier_wiring import approval_answers, call, handle, run, toolbox
from test_okf_tools import FakeGraph, _box


def _choice(kind: str, conf: float) -> dict:
    return {"type": "choice", "choice": kind, "confidence": conf,
            "probabilities": {kind: conf}}


# ── the four kinds ─────────────────────────────────────────────────────────

def test_the_kinds_are_shared():
    kinds = set(KNOWLEDGE_KINDS)
    assert kinds == {"dogma", "rule", "evidence", "decision"}
    assert set(FINDINGS_QUESTIONS["category"]["criteria"]) == kinds | {"none"}
    assert set(INGEST_QUESTIONS["kind"]["criteria"]) == kinds | {"none"}
    ingest_type = OkfIngestArgs.model_json_schema()["properties"]["type"]
    assert set(ingest_type["anyOf"][0]["enum"]) == kinds


# ── the decision log ───────────────────────────────────────────────────────

def test_the_gate_logs_answers_and_triage_reads_them_back():
    tb = toolbox(tool_policy={"high": "triage"})
    seam = DecisionSeam(timeout_s=5)
    seam.attach(lambda req: threading.Timer(0.02, lambda: seam.deny(
        req.decision_id, reason="never delete fixtures")).start())
    h = handle(lambda body: approval_answers(rev=0.1))
    bind_run_identity(tb, RunIdentity(run_id="r1"))

    run(tb, "delete_file", {"path": "fixtures/a.json"}, h=h, seam=seam)
    log = getattr(tb, DECISION_LOG_ATTR)
    [entry] = log.for_run("r1")
    assert entry["tool"] == "delete_file" and entry["outcome"] == "denied"
    assert entry["reason"] == "never delete fixtures"

    run(tb, "delete_file", {"path": "fixtures/b.json"}, h=h, seam=seam)
    prior = h.calls[-1]["state"]["prior_decisions"]
    assert prior and prior[0]["reason"] == "never delete fixtures"


def test_the_log_persists_and_reads_back(tmp_path):
    path = tmp_path / "d.jsonl"
    DecisionLog(path).record(decision_id="d1", tool="t", prompt="p", outcome="approved",
                             project="proj")
    again = DecisionLog(path)
    assert [e["id"] for e in again.similar("t", "proj")] == ["d1"]
    assert again.similar("t", "elsewhere") == []


def test_a_memory_log_gets_its_file_later(tmp_path):
    tb = toolbox()
    log = decision_log_of(tb)
    assert log.path is None
    assert decision_log_of(tb, tmp_path / "d.jsonl") is log and log.path is not None


# ── the knowledge sink ─────────────────────────────────────────────────────

def test_a_wired_graph_puts_a_knowledge_sink_on_the_box():
    assert getattr(_box(level="write"), KNOWLEDGE_ATTR).can_write
    assert not getattr(_box(level="read"), KNOWLEDGE_ATTR).can_write


def test_rules_come_from_rules_and_dogmas_and_skip_superseded():
    graph = FakeGraph()
    graph.call = lambda op, **kw: (graph.calls.append((op, kw)) or {
        "rule": [{"id": "r1", "title": "Old rule", "score": 0.9, "tags": ["superseded"]},
                 {"id": "r2", "title": "Run tests first", "score": 0.5}],
        "dogma": [{"id": "d1", "title": "No telemetry", "score": 0.7, "body": "Never phone home."}],
    }[kw["concept_type"]])
    k = OkfKnowledge(graph)
    assert k.rules_for("http_get") == ["No telemetry", "Run tests first"]
    assert k.nearest_rule("send usage stats")["id"] == "d1"
    assert {kw["concept_type"] for _op, kw in graph.calls} == {"rule", "dogma"}


def test_findings_go_in_as_their_kind_and_proposals_are_tagged():
    graph = FakeGraph({"ingest": {"concept_id": "new"}})
    k = OkfKnowledge(graph)
    note = {"title": "t", "body": "Fixtures are generated.", "category": "evidence",
            "relations": {"supersedes": "old"}}
    assert k.write_finding(note) == "new"
    (op, kw), (op2, kw2) = graph.calls
    assert op == "ingest" and kw["type"] == "evidence" and PROPOSED_TAG not in kw["tags"]
    assert op2 == "supersede" and kw2 == {"old_id": "old", "new_id": "new"}

    graph.calls.clear()
    k.offer_finding(note)
    [(op, kw)] = graph.calls
    assert PROPOSED_TAG in kw["tags"], "a proposal never supersedes anything"

    graph.calls.clear()
    assert OkfKnowledge(graph, can_write=False).write_finding(note) is None
    assert k.write_finding({**note, "category": "none"}) is None
    assert not graph.calls


# ── findings_gate ──────────────────────────────────────────────────────────

def test_findings_gate_files_a_reasoned_decision():
    graph = FakeGraph({"search": [], "ingest": {"concept_id": "c"}})
    tb = _box(graph)
    attach_catalog_hooks(tb, None, names=["findings_gate"])
    bind_run_identity(tb, RunIdentity(run_id="r1"))
    decision_log_of(tb).record(decision_id="d1", tool="delete_file", prompt="Delete x?",
                               outcome="denied", reason="Fixtures are generated; edit "
                               "the generator instead.", run_id="r1")
    decision_log_of(tb).record(decision_id="d2", tool="t", prompt="?", outcome="approved",
                               run_id="r1")       # no reason: not a finding
    h = handle(lambda body: {"category": _choice("evidence", 0.9),
                             "durable": {"type": "noul", "noul": 0.9},
                             "grounded": {"type": "noul", "noul": 0.9},
                             "relation": _choice("unrelated", 0.9)})
    bind_run_classifier(tb, h.for_run("r1"))
    tb.hooks.emit("after_run", final_text="", rounds=1, elapsed_s=0.1)

    ingests = [kw for op, kw in graph.calls if op == "ingest"]
    assert len(ingests) == 1 and ingests[0]["type"] == "evidence"
    assert len(h.calls) == 1


# ── classify_ingest ────────────────────────────────────────────────────────

def _ingest_box(kind: str, conf: float, mode: str = "live"):
    graph = FakeGraph({"ingest": {"concept_id": "c"}})
    tb = _box(graph)
    attach_catalog_hooks(tb, None, names=["classify_ingest"])
    h = handle(lambda body: {"kind": _choice(kind, conf)}, mode=mode)
    bind_run_classifier(tb, h.for_run("r1"))
    return tb, graph


def _ingested(tb, graph, **args) -> dict:
    call(tb, "okf_ingest", {"thoughts": "Always run the linter.", "topic": "t", **args})
    return [kw for op, kw in graph.calls if op == "ingest"][-1]


def test_an_untyped_note_is_typed():
    tb, graph = _ingest_box("rule", 0.8)
    kw = _ingested(tb, graph)
    assert kw["type"] == "rule" and "classified" in kw["tags"]


def test_an_unsure_classifier_leaves_it_untyped():
    tb, graph = _ingest_box("rule", 0.4)
    assert _ingested(tb, graph).get("type") is None


def test_the_agents_type_stands_unless_the_classifier_is_very_sure():
    tb, graph = _ingest_box("rule", 0.8)
    assert _ingested(tb, graph, type="evidence")["type"] == "evidence"
    tb, graph = _ingest_box("rule", 0.95)
    assert _ingested(tb, graph, type="evidence")["type"] == "rule"


def test_shadow_mode_only_records():
    tb, graph = _ingest_box("rule", 0.95, mode="shadow")
    assert _ingested(tb, graph).get("type") is None
