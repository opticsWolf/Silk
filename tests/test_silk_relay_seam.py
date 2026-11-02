# -*- coding: utf-8 -*-
"""A delegated worker asks the human its parent asks (D48 revised).

Before the relay, a worker had no seam and every gated call it made was
refused with ``no_answerer``. These tests pin the relay: the request reaches
the parent run's seam labelled with the chain, the parent's answer decides,
and the worker's toolset gets its own seam back afterwards.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

from pydantic import BaseModel

from silk.functions import approval, orchestrator
from silk.functions.approval import _SEAM_ATTR
from silk.functions.decision_seam import (
    CAUSE_TIMEOUT, DecisionRequest, DecisionSeam, RelaySeam,
)
from silk.functions.orchestrator import attach_orchestrator_tools
from silk.functions.role import DEFAULT_ROLE
from silk.functions.subagent import SubagentResult, WorkerSpec
from silk.functions.tool_box import ToolBox


class _Path(BaseModel):
    path: str


def _gated_box() -> ToolBox:
    tb = ToolBox(db_pool=None, user_session={"agent_id": "writer"})

    @tb.register("delete_file", "delete a file", args_model=_Path, risk="high")
    def _delete(db_pool, user_session, path):
        return f"deleted {path}"

    approval.attach_approval_gate(tb, tool_policy={"high": "human"})
    return tb


def _call(tb, name, args):
    tc = SimpleNamespace(id="c1", function=SimpleNamespace(
        name=name, arguments=json.dumps(args)))
    return asyncio.run(tb.execute_tool_calls_async([tc]))[0]["content"]


def _answering_seam(answer, asked):
    seam = DecisionSeam(timeout_s=5)

    def deliver(req):
        asked.append(req)
        threading.Timer(0.02, lambda: answer(seam, req)).start()

    seam.attach(deliver)
    return seam


def test_a_relayed_request_is_labelled_with_the_chain():
    asked = []
    parent = _answering_seam(lambda s, r: s.approve(r.decision_id), asked)
    relay = RelaySeam(parent, ["lead", "writer"])
    d = relay.await_decision(DecisionRequest(
        decision_id="d1", prompt="delete x", detail={"risk": "high"}))
    assert d.approved
    assert asked[0].prompt == "[lead > writer] delete x"
    assert asked[0].detail == {"risk": "high", "delegation_chain": ["lead", "writer"]}


def test_a_relay_of_a_relay_asks_the_top_seam_once():
    top = DecisionSeam(timeout_s=5)
    inner = RelaySeam(RelaySeam(top, ["a", "b"]), ["a", "b", "c"])
    assert inner.parent is top and inner.chain == ["a", "b", "c"]


def test_a_worker_gated_call_runs_when_the_parent_human_approves():
    asked = []
    parent = _answering_seam(lambda s, r: s.approve(r.decision_id), asked)
    worker = _gated_box()
    with approval.run_seam(worker, RelaySeam(parent, ["lead", "writer"])):
        out = _call(worker, "delete_file", {"path": "x.txt"})
    assert out == "deleted x.txt"
    assert asked[0].prompt.startswith("[lead > writer] ")
    assert asked[0].tool_name == "delete_file"


def test_the_parent_human_denial_reason_reaches_the_worker():
    asked = []
    parent = _answering_seam(
        lambda s, r: s.deny(r.decision_id, reason="use the trash folder"), asked)
    worker = _gated_box()
    with approval.run_seam(worker, RelaySeam(parent, ["lead", "writer"])):
        out = _call(worker, "delete_file", {"path": "x.txt"})
    assert "use the trash folder" in out and "deleted" not in out


def test_the_parent_seam_timeout_still_denies():
    parent = DecisionSeam(timeout_s=0.1)
    parent.attach(lambda req: None)              # nobody answers
    d = RelaySeam(parent, ["w"]).await_decision(DecisionRequest(decision_id="d2"))
    assert not d.approved and d.cause == CAUSE_TIMEOUT


def _delegating(monkeypatch, child, seen):
    def fake_run(spec, content, **kw):
        seen.append(getattr(spec.toolset, _SEAM_ATTR, None))
        return SubagentResult(text="ok", ok=True)

    monkeypatch.setattr(orchestrator, "run_subagent", fake_run)
    spec = WorkerSpec(model_handle={"backend": "gguf", "model": None},
                      toolset=child, role=DEFAULT_ROLE, name="writer")
    tb = ToolBox(db_pool=None, user_session={"agent_id": "lead"})
    attach_orchestrator_tools(tb, workers=[spec])
    return tb


def test_delegation_binds_a_relay_and_restores_the_worker_seam(monkeypatch):
    child, seen = _gated_box(), []
    tb = _delegating(monkeypatch, child, seen)
    parent = DecisionSeam(timeout_s=5)
    with approval.run_seam(tb, parent):
        _call(tb, "delegate", {"worker": "writer", "task": "tidy up"})
    assert isinstance(seen[0], RelaySeam)
    assert seen[0].parent is parent and seen[0].chain == ["writer"]
    assert getattr(child, _SEAM_ATTR, None) is None


def test_no_parent_seam_means_no_relay(monkeypatch):
    child, seen = _gated_box(), []
    tb = _delegating(monkeypatch, child, seen)
    _call(tb, "delegate", {"worker": "writer", "task": "tidy up"})
    assert seen == [None]


def test_a_worker_with_its_own_seam_keeps_it(monkeypatch):
    child, seen = _gated_box(), []
    own = DecisionSeam(timeout_s=5)
    approval.bind_run_seam(child, own)
    tb = _delegating(monkeypatch, child, seen)
    with approval.run_seam(tb, DecisionSeam(timeout_s=5)):
        _call(tb, "delegate", {"worker": "writer", "task": "tidy up"})
    assert seen == [own] and getattr(child, _SEAM_ATTR) is own

