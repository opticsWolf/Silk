"""Silk classifier port end to end (D-S8).

Runs Silk's own modules (gate, ToolBox, hooks, RoleBinding, orchestrator,
Classifier node, thresholds table) against the host Weave. Qt widgets run
offscreen on the real Weave node layer.
"""
import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from pydantic import BaseModel

from silk.functions import approval, orchestrator
from silk.functions.approval import attach_approval_gate, run_seam
from silk.functions.classifier import (
    BackendConfig, BandsConfig, ClassifierHandle, ClassifierSpec, build_classifier)
from silk.functions.classifier_binding import (
    begin_classifier_run, bind_run_classifier, end_classifier_run, run_classifier,
    triaged_calls)
from silk.functions.decision_seam import DecisionSeam
from silk.functions.hook_catalog import (
    HOOK_CATALOG, SignoffConfig, ToolApprovalConfig, attach_catalog_hooks, build_hooks,
    signoff_policy_from_config, tool_policy_from_config)
from silk.functions.hooks import HookRegistry, ToolboxBound, register_hook_map
from silk.functions.role import Role, RoleBinding
from silk.functions.signoff import normalize_policy
from silk.functions.subagent import WorkerSpec
from silk.functions.task_store import Goal, Plan, Task
from silk.functions.tool_box import ToolBox
from silk.functions.toolset_build import tool_catalog, with_port_tools

JEV = "jev/jev-1.13.0"
SILK = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ── helpers ────────────────────────────────────────────────────────────────

def approval_answers(rev=0.97, scope=0.95, cons=0.9, risk=0.3, rconf=0.9, intent="routine",
                     iconf=0.9):
    return {"reversible": {"type": "noul", "noul": rev},
            "in_scope": {"type": "noul", "noul": scope},
            "consistent": {"type": "noul", "noul": cons},
            "risk": {"type": "score", "score": risk, "confidence": rconf,
                     "legend": {}, "probabilities": {}},
            "intent": {"type": "choice", "choice": intent, "confidence": iconf,
                       "probabilities": {}}}


def handle(reply, *, mode="live", calibrated=True, judge=None):
    spec = ClassifierSpec(mode=mode, egress_allowed=True,
                          backends=[BackendConfig(kind="jev", model=JEV)],
                          bands={JEV: BandsConfig()} if calibrated else {})
    calls = []

    def http(url, headers, body, timeout):
        calls.append(body)
        return {"model": "jev-1.13.0", "answers": reply(body),
                "usage": {"input_tokens": 1200}}
    h = ClassifierHandle(spec, build_classifier(spec, secrets=lambda n: "k", http=http), judge)
    h.calls = calls
    return h


class PathArgs(BaseModel):
    path: str


def toolbox(tool_policy=None, task_policy=None):
    tb = ToolBox()

    @tb.register("write_file", "write a file", args_model=PathArgs, risk="medium")
    def _w(db_pool, user_session, path):
        return f"wrote {path}"

    @tb.register("delete_file", "delete a file", args_model=PathArgs, risk="high")
    def _d(db_pool, user_session, path):
        return f"deleted {path}"

    attach_approval_gate(tb, tool_policy=tool_policy, task_policy=task_policy)
    return tb


def call(tb, name, args, cid="c1"):
    tc = NS(id=cid, function=NS(name=name, arguments=json.dumps(args)))
    return asyncio.run(tb.execute_tool_calls_async([tc]))[0]["content"]


def human(approve=True):
    asked = []
    seam = DecisionSeam(timeout_s=5)

    def deliver(req):
        asked.append(req)
        threading.Timer(0.02, lambda: (seam.approve(req.decision_id) if approve
                                       else seam.deny(req.decision_id))).start()
    seam.attach(deliver)
    return seam, asked


def run(tb, name, args, *, h=None, seam=None):
    rc = h.for_run("r1") if h is not None else None
    bind_run_classifier(tb, rc)
    with run_seam(tb, seam):
        out = call(tb, name, args)
    return out


# ── the triage level in the gate ───────────────────────────────────────────

def test_triage_without_a_classifier_asks_the_human():
    tb = toolbox({"medium": "triage"})
    seam, asked = human()
    assert run(tb, "write_file", {"path": "a"}, seam=seam) == "wrote a"
    assert len(asked) == 1 and asked[0].detail["level"] == "human"
    assert "priority" not in asked[0].detail                # nothing wired, nothing scored


def test_live_in_band_runs_without_asking_and_is_counted():
    tb = toolbox({"medium": "triage"})
    seam, asked = human()
    h = handle(lambda b: approval_answers())
    assert run(tb, "write_file", {"path": "a"}, h=h, seam=seam) == "wrote a"
    assert asked == []
    [entry] = triaged_calls(tb)
    assert entry["tool"] == "write_file" and entry["model"] == JEV
    state = h.calls[0]["state"]
    assert state["action"]["tool"] == "write_file" and "transcript" not in state


def test_shadow_scores_and_still_asks():
    tb = toolbox({"medium": "triage"})
    seam, asked = human()
    run(tb, "write_file", {"path": "a"}, h=handle(lambda b: approval_answers(), mode="shadow"),
        seam=seam)
    [req] = asked
    assert req.detail["triage"]["shadow_route"] == "allow" and req.detail["level"] == "triage"
    assert triaged_calls(tb) == []


def test_out_of_band_or_uncalibrated_asks_with_the_reason():
    tb = toolbox({"medium": "triage"})
    seam, asked = human()
    run(tb, "write_file", {"path": "a"}, h=handle(lambda b: approval_answers(rev=0.4)),
        seam=seam)
    run(tb, "write_file", {"path": "a"}, h=handle(lambda b: approval_answers(),
                                                 calibrated=False), seam=seam)
    assert "reversible 0.40" in asked[0].detail["why"]
    assert "no calibrated bands" in asked[1].detail["why"]


def test_human_level_is_never_skipped_but_gets_a_suggestion():
    tb = toolbox({"high": "human", "medium": "triage"})
    seam, asked = human()
    run(tb, "delete_file", {"path": "a"}, h=handle(lambda b: approval_answers(cons=0.05)),
        seam=seam)
    [req] = asked
    assert req.detail["level"] == "human" and req.detail["suggestion"] == "deny"


def test_naming_a_tool_triage_never_lowers_a_human_band():
    cfg = ToolApprovalConfig(preset="high_risk", triage_tools="delete_file,write_file")
    policy = tool_policy_from_config(cfg)
    assert policy == {"high": "human", "delete_file": "triage", "write_file": "triage"}
    tb = toolbox(policy)
    seam, asked = human()
    h = handle(lambda b: approval_answers())
    assert run(tb, "write_file", {"path": "a"}, h=h, seam=seam) == "wrote a"   # medium -> triage
    run(tb, "delete_file", {"path": "a"}, h=h, seam=seam)                    # high band wins
    assert [r.tool_name for r in asked] == ["delete_file"]


def test_a_worker_with_no_seam_can_pass_triage_but_nothing_else():
    tb = toolbox({"medium": "triage", "high": "human"})
    h = handle(lambda b: approval_answers())
    assert run(tb, "write_file", {"path": "a"}, h=h, seam=None) == "wrote a"
    refused = json.loads(run(tb, "delete_file", {"path": "a"}, h=h, seam=None))
    assert refused["applied"] is False and "no way to ask" in refused["message"]


def test_closing_the_plan_also_needs_the_completion_check():
    class Store:
        def load(self):
            return Plan(plan_id="p", created_at="", updated_at="", revision=1,
                        goal=Goal(text="Ship v2", original_text="Ship v2",
                                  acceptance=["tests pass", "docs updated"]),
                        tasks=[Task(id="t1", title="last", status="in_progress", parent=None, order=0)])

    class CompleteArgs(BaseModel):
        id: str
        rationale: str

    tb = ToolBox()

    @tb.register("task_complete", "complete", args_model=CompleteArgs)
    def _c(db_pool, user_session, id, rationale):
        return f"completed {id}"
    tb._task_store = Store()
    attach_approval_gate(tb, task_policy={"complete_final": "triage"})

    def reply(body):
        qs = body["questions"]
        if "c0" in qs:                                   # the completion check
            return {"c0": {"type": "noul", "noul": 0.95}, "c1": {"type": "noul", "noul": 0.1}}
        return approval_answers()
    seam, asked = human()
    out = run(tb, "task_complete", {"id": "t1", "rationale": "pytest green"},
              h=handle(reply), seam=seam)
    assert out == "completed t1"                         # the human approved
    [req] = asked
    assert req.detail["completion"]["verdict"] == "reject"
    assert req.detail["completion"]["unmet"] == ["docs updated"]
    assert req.detail["suggestion"] == "deny" and req.detail["why"].startswith("completion")


# ── policy configs ─────────────────────────────────────────────────────────

def test_policy_configs_accept_triage_and_human_wins():
    cfg = SignoffConfig(preset="custom", complete_final="triage", goal="human")
    policy = signoff_policy_from_config(cfg)
    assert policy["complete_final"] == "triage" and policy["goal"] == "human"
    assert normalize_policy({"add": "triage", "goal": "bogus"})["add"] == "triage"
    assert approval.normalize_tool_policy({"medium": "triage", "x": "maybe"}) == {
        "medium": "triage"}
    both = tool_policy_from_config(ToolApprovalConfig(preset="writes", triage_bands="low_medium",
                                                      tools="run_command",
                                                      triage_tools="run_command"))
    assert both == {"high": "human", "medium": "human", "low": "triage",
                    "run_command": "human"}


def test_the_config_dialog_offers_triage():
    from typing import get_args
    assert get_args(SignoffConfig.model_fields["complete_final"].annotation) == (
        "agent", "triage", "human")
    assert "low_medium" in get_args(ToolApprovalConfig.model_fields["triage_bands"].annotation)


def test_config_dialog_renders_the_third_level(qapp):
    from PySide6.QtWidgets import QComboBox
    from silk.widgets.config_dialog import PydanticConfigDialog
    dlg = PydanticConfigDialog(SignoffConfig, SignoffConfig().model_dump())
    combos = [c for c in dlg.findChildren(QComboBox)
              if [c.itemText(i) for i in range(c.count())] == ["agent", "triage", "human"]]
    assert len(combos) == 5                              # one per change type


# ── binding per run ────────────────────────────────────────────────────────

def test_run_binding_registers_classify_only_while_wired():
    tb = toolbox({"medium": "triage"})
    start = begin_classifier_run(tb, handle(lambda b: {}, mode="shadow"), run_id="r1")
    assert start.error == "" and "classify" in tb.tools and run_classifier(tb) is not None
    assert any("shadow" in n for n in start.notices)
    assert end_classifier_run(tb) == []
    assert "classify" not in tb.tools and run_classifier(tb) is None
    start = begin_classifier_run(tb, None, run_id="r2")
    assert "classify" not in tb.tools
    assert any("no Classifier is connected" in n for n in start.notices)


def test_run_binding_summarises_triage_and_refuses_self_grading():
    tb = toolbox({"medium": "triage"})
    h = handle(lambda b: approval_answers())
    begin_classifier_run(tb, h, run_id="r1")
    call(tb, "write_file", {"path": "a"})
    assert end_classifier_run(tb) == [f"1 call(s) allowed by triage without asking ({JEV})."]
    pool = NS(model_path="/m/qwen.gguf", client=NS())
    h_self = handle(lambda b: {}, judge={"backend": "gguf", "pool": pool})
    start = begin_classifier_run(tb, h_self, run_id="r2",
                                 agent_model={"backend": "gguf", "pool": pool})
    assert start.error.startswith("Classifier: the judge must not be")
    assert run_classifier(tb) is None
    other = NS(model_path="/m/judge.gguf", client=NS())
    ok = begin_classifier_run(tb, handle(lambda b: {}, judge={"backend": "gguf", "pool": other}),
                              run_id="r3", agent_model={"backend": "gguf", "pool": pool})
    assert ok.error == ""


def test_catalog_lists_classify_for_roles_before_it_exists():
    tb = ToolBox()
    assert all(e["name"] != "classify" for e in tool_catalog(tb))   # the toolbox's own tree
    entry = next(e for e in with_port_tools(tool_catalog(tb)) if e["name"] == "classify")
    assert entry["category"] == "classifier"
    begin_classifier_run(tb, handle(lambda b: {}), run_id="r1")
    assert [e["name"] for e in with_port_tools(tool_catalog(tb))].count("classify") == 1
    end_classifier_run(tb)


# ── the orchestrator passes it on ──────────────────────────────────────────

def test_workers_inherit_the_run_classifier_and_get_it_back(monkeypatch):
    parent, child = ToolBox(), ToolBox()
    seen = {}

    def fake_run_subagent(spec, content, **kw):
        seen["rc"] = run_classifier(spec.toolset)
        return NS(ok=True, text="done", tool_calls=[], error=None)
    monkeypatch.setattr(orchestrator, "run_subagent", fake_run_subagent)
    orchestrator.set_orchestrator_workers(parent, [WorkerSpec(
        model_handle={"backend": "gguf", "pool": object()}, toolset=child, name="coder")])
    h = handle(lambda b: {})
    begin_classifier_run(parent, h, run_id="r1")
    res = orchestrator._run_one(parent, "coder", "do it", "", "orchestrator")
    assert res.ok and seen["rc"] is run_classifier(parent)
    assert run_classifier(child) is None                     # restored after the delegation
    assert not hasattr(child, "_run_classifier")


# ── classifier hooks through ToolboxBound ──────────────────────────────────

class _Store:
    def load(self):
        return Plan(plan_id="p", created_at="", updated_at="", revision=1,
                    goal=Goal(text="Ship v2", original_text="Ship v2"),
                    tasks=[Task(id="t1", title="write parser", status="in_progress", parent=None, order=0)])


def _drift_answers(body):
    return {"needed": {"type": "noul", "noul": 0.1}, "duplicate": {"type": "noul", "noul": 0.1}}


def _tb_with_task_add():
    tb = ToolBox()

    class AddArgs(BaseModel):
        title: str
        rationale: str = ""

    @tb.register("task_add", "add", args_model=AddArgs)
    def _a(db_pool, user_session, title, rationale=""):
        return f"added {title}"
    tb._task_store = _Store()
    return tb


def test_role_selected_task_drift_binds_to_the_toolbox():
    tb = _tb_with_task_add()
    role = Role(id="r", hooks=build_hooks(["task_drift"], {}))
    binding = RoleBinding.activate(role, tb)
    try:
        assert call(tb, "task_add", {"title": "Repaint logo"}) == "added Repaint logo"  # unbound
        h = handle(_drift_answers)
        bind_run_classifier(tb, h.for_run("r1"))
        first = json.loads(call(tb, "task_add", {"title": "Repaint logo"}))
        assert first["error"].startswith("Not added")
        assert h.calls[0]["state"] == {"goal": "Ship v2", "existing_tasks": ["write parser"],
                                       "new_task": "Repaint logo", "rationale": ""}
        assert call(tb, "task_add", {"title": "Repaint logo"}) == "added Repaint logo"
    finally:
        binding.deactivate()


def test_toolbox_task_drift_via_the_catalog_attach():
    tb = _tb_with_task_add()
    attach_catalog_hooks(tb, None, names=["task_drift"])
    bind_run_classifier(tb, handle(_drift_answers).for_run("r1"))
    assert "Not added" in call(tb, "task_add", {"title": "x"})


def test_prose_question_and_findings_are_selectable_and_inert_without_sinks():
    tb = _tb_with_task_add()
    attach_catalog_hooks(tb, None, names=["prose_question", "findings_gate"])
    h = handle(lambda b: pytest.fail("asked without a sink"))
    bind_run_classifier(tb, h.for_run("r1"))
    tb.hooks.emit("after_run", final_text="Should I use Postgres?", rounds=1, elapsed_s=0.1)
    assert {"prose_question", "findings_gate", "task_drift"} <= set(HOOK_CATALOG)


def test_a_toolbox_bound_hook_needs_an_owner():
    reg = HookRegistry()
    with pytest.raises(RuntimeError, match="no owner"):
        register_hook_map(reg, {"after_run": [ToolboxBound(lambda tb: (lambda **k: None), "x")]})
    with pytest.raises(RuntimeError, match="before it was bound"):
        ToolboxBound(lambda tb: None, "x")()


# ── the Classifier node and its thresholds table ───────────────────────────

@pytest.fixture
def node(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from silk.functions import presets
    monkeypatch.setattr(presets, "PRESET_DIR", tmp_path / "presets")
    from silk.nodes.classifier import SilkClassifierNode
    n = SilkClassifierNode()
    yield n
    n.cleanup()


def ports(side):
    return {p.name: getattr(p.datatype, "name", p.datatype) for p in side}


def gather(node, wired=None):
    """What the evaluator hands ``compute``: widget values plus wired inputs."""
    core = node._widget_core
    inputs = {name: core.get_port_value(name) for name in core.bindings()}
    inputs["judge_model"] = None
    inputs.update(wired or {})
    return inputs


def test_node_ports_and_safe_defaults(node):
    assert ports(node.inputs) == {"judge_model": "model_handle"}
    assert ports(node.outputs) == {"classifier": "silk_classifier", "stats": "dict"}
    out = node.compute(gather(node))
    h = out["classifier"]
    assert isinstance(h, ClassifierHandle) and h.spec.mode == "shadow"
    assert h.spec.egress_allowed is False and h.spec.bands == {}
    assert out["stats"]["backends"] == ["jev"]           # no judge wired: no local link


def test_node_rebuilds_the_handle_only_when_the_form_changes(node):
    a = node.compute(gather(node))["classifier"]
    b = node.compute(gather(node))["classifier"]
    assert a is b                                         # cache, breaker, meter survive
    node.chk_egress.setChecked(True)
    c = node.compute(gather(node))["classifier"]
    assert c is not a and c.spec.egress_allowed is True


def test_node_with_a_judge_model_adds_the_local_link(node):
    pool = NS(model_path="/models/qwen3-1.7b-judge.gguf",
              client=NS(base_url="http://127.0.0.1:9/v1", model_alias="judge",
                        headers=lambda: {}))
    out = node.compute(gather(node, {"judge_model": {"backend": "gguf", "pool": pool}}))
    assert out["stats"]["backends"] == ["jev", "local"]
    local = out["classifier"].classifier.backends[1]
    assert local.model == "local/qwen3-1.7b-judge"
    assert out["classifier"].judge_model["pool"] is pool


def test_thresholds_table_validates_per_model(node):
    t = node.bands_table
    t.add_model(JEV)
    t.add_model("local/qwen3-1.7b-judge", BandsConfig(reversible=0.95))
    t.add_model("jev/jev-latest")                         # unpinned: refused
    row = t.add_model("local/broken")
    t._table.item(row, 1).setText("1.7")                  # out of range: refused
    assert t.invalid_rows() == [2, 3]
    assert sorted(t.bands()) == [JEV, "local/qwen3-1.7b-judge"]
    spec = node.compute(gather(node))["classifier"].spec
    assert spec.bands_for("local/qwen3-1.7b-judge").reversible == 0.95
    assert spec.bands_for("local/broken") is None         # never loosens anything
    assert "2 calibrated model(s)" in t._hint.text()


def test_presets_roundtrip_the_whole_form(node):
    node.combo_mode.setCurrentIndex(2)
    node.edit_model.setText("jev/jev-1.14.0")
    node.bands_table.add_model("jev/jev-1.14.0", BandsConfig(in_scope=0.9))
    saved = node._collect_preset()
    node.combo_mode.setCurrentIndex(0)
    node.bands_table.set_bands({})
    from silk.functions.presets import ClassifierPreset
    node._apply_preset(ClassifierPreset(name="p", **saved))
    spec = node.compute(gather(node))["classifier"].spec
    assert spec.mode == "live" and spec.backends[0].model == "jev/jev-1.14.0"
    assert spec.bands_for("jev/jev-1.14.0").in_scope == 0.9


def test_port_type_is_registered():
    from silk.nodes.silk_ports import SILK_CLASSIFIER_TYPE
    assert SILK_CLASSIFIER_TYPE.validator(None)
    assert SILK_CLASSIFIER_TYPE.validator(handle(lambda b: {}))
    assert not SILK_CLASSIFIER_TYPE.validator(object())


# ── the Agent node's edits (Weave's node runtime is not available here) ────

def test_agent_binds_after_the_seam_and_unbinds_in_finally():
    src = (SILK / "nodes" / "agent.py").read_text()
    assert 'self.add_input("classifier", datatype="silk_classifier")' in src
    seam_at = src.index("bind_run_seam(toolset, self._seam)")
    begin_at = src.index("begin_classifier_run(")
    finally_at = src.index("finally:", begin_at)
    end_at = src.index("end_classifier_run(toolset)")
    assert seam_at < begin_at < finally_at < end_at
    assert "agent_model=model_handle" in src
