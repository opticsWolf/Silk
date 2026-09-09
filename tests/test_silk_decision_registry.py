# -*- coding: utf-8 -*-
"""The decision directory behind the inbox dock (D59).

The registry exists so a graph of N agents can answer "who needs me"
without hunting the canvas. What it must never become is a way to answer
*for* an agent (D51, I12), and what it must never do is keep a run or a
node alive (D49) — so those two are what these tests pin hardest.
"""
from __future__ import annotations

import gc

from silk.functions.decision_registry import (
    DecisionRegistry, _fields,
)


class _Node:
    """Stand-in for an Agent node: something weakly referenceable."""

    def __init__(self, title="agent-a"):
        self.title = title
        self.answered = []

    def _answer_decision(self, approved, remember):
        self.answered.append((approved, remember))


def _request(decision_id="d1", run_id="run-a", prompt="Run pytest?"):
    return {"decision_id": decision_id, "run_id": run_id, "kind": "approval",
            "prompt": prompt, "tool_name": "run_command"}


# ── membership ───────────────────────────────────────────────────────────


def test_a_request_becomes_a_row():
    reg = DecisionRegistry()
    node = _Node()
    entry = reg.register(_request(), node=node, agent="agent-a")
    assert entry is not None and len(reg) == 1
    assert entry.agent == "agent-a" and "pytest" in entry.detail()


def test_an_answer_removes_the_row():
    reg = DecisionRegistry()
    node = _Node()
    reg.register(_request(), node=node)
    assert reg.unregister("d1") and len(reg) == 0
    assert not reg.unregister("d1"), "removing twice is not an event"


def test_a_request_without_an_id_is_refused():
    reg = DecisionRegistry()
    node = _Node()
    assert reg.register({"prompt": "no id"}, node=node) is None, (
        "an entry that can never be unregistered would leave a button "
        "for a decision nobody is waiting on"
    )
    assert len(reg) == 0


def test_a_finished_run_takes_its_questions_with_it():
    reg = DecisionRegistry()
    node = _Node()
    reg.register(_request("d1", "run-a"), node=node)
    reg.register(_request("d2", "run-b"), node=node)
    assert reg.clear_run("run-a") == 1
    assert [e.decision_id for e in reg.entries()] == ["d2"], (
        "a stopped run never answers, and its row is a dead button"
    )


def test_a_dataclass_request_reads_the_same_as_a_dict():
    class Request:
        decision_id, run_id, kind = "d9", "run-z", "acknowledge"
        prompt, tool_name = "Proceed?", "write_file"

    reg = DecisionRegistry()
    node = _Node()
    entry = reg.register(Request(), node=node)
    assert entry.kind == "acknowledge" and entry.run_id == "run-z"
    assert _fields(Request())["prompt"] == "Proceed?"


# ── it is a directory, not an owner ──────────────────────────────────────


def test_the_registry_does_not_keep_a_node_alive():
    reg = DecisionRegistry()
    node = _Node()
    reg.register(_request(), node=node)

    del node
    gc.collect()
    assert reg.entries() == [], (
        "a registry that outlived its nodes would change seam lifetime, "
        "which D59 says it must not"
    )


def test_a_dead_node_row_disappears_when_anyone_looks():
    reg = DecisionRegistry()
    node = _Node()
    reg.register(_request(), node=node)
    assert len(reg.entries()) == 1
    del node
    gc.collect()
    assert reg.get("d1") is None or not reg.get("d1").alive
    assert len(reg) == 0


def test_the_registry_holds_nothing_that_resolves():
    reg = DecisionRegistry()
    node = _Node()
    entry = reg.register(_request(), node=node)
    for name in list(vars(entry)) + dir(reg):
        assert not any(word in name.lower()
                       for word in ("approve", "deny", "resolve", "seam")), (
            f"{name} would let the directory answer; only the asking "
            f"node's surface may (D59, I12)"
        )


# ── change notification ──────────────────────────────────────────────────


def test_listeners_hear_about_changes():
    reg = DecisionRegistry()
    beats = []
    node = _Node()
    off = reg.subscribe(lambda: beats.append(1))
    reg.register(_request(), node=node)
    reg.unregister("d1")
    assert len(beats) == 2
    off()
    reg.register(_request("d2"), node=node)
    assert len(beats) == 2, "unsubscribing is honoured"


def test_a_broken_listener_does_not_break_the_registry():
    reg = DecisionRegistry()

    def explode():
        raise ValueError("a dock that was closed badly")

    node = _Node()
    reg.subscribe(explode)
    assert reg.register(_request(), node=node) is not None, (
        "the run is blocked on this call; a bad listener must not be the "
        "reason the question never gets registered"
    )
