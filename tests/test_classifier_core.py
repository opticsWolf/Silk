"""Silk classifier core (D-S8): questions, backends, breaker, redaction,
spec, triage levels, task drift, findings gate and the classify tool.

Ported from the Macrame 0.20 prototype; the cases that drive the work
ledger / decision flow stay with that prototype until it lands.
"""
import asyncio
import json
import math
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from silk.functions.classifier import (
    BackendConfig, BandsConfig, Breaker, ChainClassifier, ClassifierHandle, ClassifierSpec,
    FakeBackend, JevBackend, LocalJudgeBackend, QuestionError, Redactor, build_classifier,
    validate_questions)
from silk.functions.classifier_hooks import (
    FindingsGateConfig, HookContext, TaskDriftConfig, make_findings_gate, make_task_drift,
    route_finding)
from silk.functions.classifier_binding import bind_run_classifier
from silk.functions.classify_tool import register_classify_tool
from silk.functions.triage import APPROVAL_QUESTIONS, effective_level, stricter
from silk.functions.hooks import HOOK_AFTER_RUN, register_hook_map
from silk.functions.tool_box import ToolBox

JEV = "jev/jev-1.13.0"



def approval_answers(rev=0.97, scope=0.95, cons=0.9, risk=0.3, rconf=0.9, intent="routine",
                     iconf=0.9):
    return {"reversible": {"type": "noul", "noul": rev},
            "in_scope": {"type": "noul", "noul": scope},
            "consistent": {"type": "noul", "noul": cons},
            "risk": {"type": "score", "score": risk, "confidence": rconf,
                     "legend": {}, "probabilities": {}},
            "intent": {"type": "choice", "choice": intent, "confidence": iconf,
                       "probabilities": {}}}


class StubHttp:
    """Plays the /v1/systemone endpoint. ``reply(body) -> answers`` or raises."""

    def __init__(self, reply, model="jev-1.13.0"):
        self.reply, self.model, self.bodies, self.headers = reply, model, [], []

    def __call__(self, url, headers, body, timeout):
        self.bodies.append(body)
        self.headers.append(headers)
        return {"model": self.model, "answers": self.reply(body),
                "usage": {"input_tokens": 1500, "output_tokens": 20}}


def make_run_classifier(reply, *, mode="live", bands=True, local_scorer=None,
                        backends=None, egress=True, clock=None, **spec_kw):
    spec = ClassifierSpec(
        mode=mode, egress_allowed=egress,
        backends=backends or [BackendConfig(kind="jev", model=JEV)],
        bands={JEV: BandsConfig()} if bands else {}, **spec_kw)
    http = StubHttp(reply)
    kw = {"clock": clock} if clock else {}
    chain = build_classifier(spec, secrets=lambda n: "k-test", http=http,
                             local_scorer=local_scorer, **kw)
    rc = ClassifierHandle(spec, chain).for_run("r1")
    rc.http = http
    return rc


@pytest.fixture
def mk():
    return make_run_classifier


NOUL = {"q": {"type": "noul", "instructions": "Is it?"}}


def hook_map(name, ctx, config=None):
    """The hook map a ToolboxBound catalog entry builds, without a toolbox owner."""
    make = {"task_drift": make_task_drift, "findings_gate": make_findings_gate}[name]
    return make(config, ctx)


def noul(p):
    return lambda state, qs: {k: {"type": "noul", "noul": p} for k in qs}


# ── C1: questions and answers ──────────────────────────────────────────────

def test_question_validation():
    validate_questions(APPROVAL_QUESTIONS)
    bad = [
        {},
        {"1bad": {"type": "noul", "instructions": "x"}},
        {"q": {"type": "noul", "instructions": ""}},
        {"q": {"type": "noul", "instructions": "x", "criteria": ["a"]}},
        {"q": {"type": "choice", "instructions": "x", "criteria": {"only": None}}},
        {"q": {"type": "choice", "instructions": "x",
               "criteria": {f"o{i}": None for i in range(256)}}},
        {"q": {"type": "score", "instructions": "x", "criteria": ["one"]}},
        {"q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}},
        {"q": {"type": "rank", "instructions": "x"}},
    ]
    for q in bad:
        with pytest.raises(QuestionError):
            validate_questions(q)


def test_mistyped_or_out_of_range_answers_fall_through():
    c = ChainClassifier([FakeBackend(lambda s, q: {"q": {"type": "noul", "noul": 1.7}}),
                         FakeBackend(lambda s, q: {"q": {"type": "choice", "choice": "x"}},
                                     name="f2"),
                         FakeBackend(noul(0.4), name="f3", model="fake/3")], mode="live")
    v = c.ask("s", NOUL, purpose="gate")
    assert v.ok and v.backend == "f3" and v.noul("q") == 0.4
    assert len(v.tried) == 2 and "out of range" in v.tried[0] and "mistyped" in v.tried[1]


# ── C1: backends ───────────────────────────────────────────────────────────

def test_jev_wire_format_and_usage(mk):
    rc = mk(lambda body: approval_answers())
    v = rc.ask({"goal": "g"}, APPROVAL_QUESTIONS, purpose="gate")
    body, headers = rc.http.bodies[0], rc.http.headers[0]
    assert body == {"model": JEV, "state": {"goal": "g"}, "questions": APPROVAL_QUESTIONS}
    assert headers["Authorization"] == "Bearer k-test"
    assert v.model == JEV and v.input_tokens == 1500
    assert math.isclose(v.cost_usd, 1500 * 0.042 / 1e6)


def test_jev_needs_a_pinned_model_and_a_key():
    with pytest.raises(ValueError):
        JevBackend(key="k", model="jev/jev-latest")
    with pytest.raises(ValidationError):
        BackendConfig(kind="jev", model="jev/jev-latest")
    c = ChainClassifier([JevBackend(key=None)], mode="live", egress_allowed=True)
    v = c.ask("s", NOUL, purpose="gate")
    assert not v.ok and "no Jev key" in v.tried[0]


def test_local_judge_scores_by_log_probability():
    def scorer(prompt, options):
        table = {" yes": -0.1, " no": -2.4, " routine": -0.2, " destructive": -3.0,
                 " exploratory": -2.0, " exfiltration": -6.0, " escalation": -6.0,
                 " none": -3.0, " low": -0.3, " moderate": -1.5, " high": -4.0, " severe": -6.0}
        return [table[o] for o in options]
    judge = LocalJudgeBackend(scorer, model="local/qwen3-1.7b-judge")
    answers, used = judge.answer({"action": "rm build/"}, APPROVAL_QUESTIONS)
    assert used > 0 and judge.egress is False
    assert 0.85 < answers["reversible"]["noul"] < 0.95
    i = answers["intent"]
    assert i["choice"] == "routine" and math.isclose(sum(i["probabilities"].values()), 1.0)
    assert 0 < i["confidence"] < 1
    r = answers["risk"]
    assert 0.5 < r["score"] < 1.5 and r["legend"]["0"] == "none"
    assert "Allowed: yes, no" in judge.prompt({}, "reversible", APPROVAL_QUESTIONS["reversible"])


# ── C3: the fallback chain ─────────────────────────────────────────────────

def test_breaker_rests_a_failing_backend_and_recovers():
    now = [0.0]
    calls = {"n": 0}

    def flaky(s, q):
        calls["n"] += 1
        raise ConnectionError("down")
    c = ChainClassifier([FakeBackend(flaky, name="jev", model="jev/x"),
                         FakeBackend(noul(0.5), name="local", model="local/x")], mode="live",
                        breaker=Breaker(threshold=2, cooldown_s=30, clock=lambda: now[0]))
    for i in range(4):
        c.ask(f"s{i}", NOUL, purpose="gate")
    assert calls["n"] == 2                       # rested after two failures
    now[0] = 31.0
    c.ask("s9", NOUL, purpose="gate")
    assert calls["n"] == 3                       # tried again after the cool-down


def test_everything_down_means_no_answers():
    c = ChainClassifier([FakeBackend(lambda s, q: 1 / 0)], mode="live")
    v = c.ask("s", NOUL, purpose="gate")
    assert not v.ok and v.answers == {} and "ZeroDivisionError" in v.tried[0]


# ── C3: modes and levels ───────────────────────────────────────────────────

def test_levels_order_and_triage_without_classifier_is_human():
    assert stricter("agent", "triage") == "triage" and stricter("triage", "human") == "human"
    assert effective_level("triage", None) == "human"
    assert effective_level("triage", NS(mode="off")) == "human"
    assert effective_level("triage", NS(mode="shadow")) == "triage"
    assert effective_level("agent", None) == "agent"


# ── C5: cache isolation, egress, redaction, metering ───────────────────────

def test_cache_is_per_purpose(mk):
    rc = mk(lambda b: {"q": {"type": "noul", "noul": 0.9}})
    a = rc.ask("same", NOUL, purpose="tool")
    b = rc.ask("same", NOUL, purpose="gate")
    c = rc.ask("same", NOUL, purpose="gate")
    assert not a.cached and not b.cached and c.cached
    assert len(rc.http.bodies) == 2
    assert rc.meter.calls == 3 and rc.meter.by_purpose == {"tool": 1, "gate": 2}
    assert math.isclose(rc.meter.cost_usd, 2 * 1500 * 0.042 / 1e6)


def test_hosted_backends_need_project_egress(mk):
    rc = mk(lambda b: {"q": {"type": "noul", "noul": 0.9}}, egress=False)
    v = rc.ask("s", NOUL, purpose="gate")
    assert not v.ok and "egress not allowed" in v.tried[0] and rc.http.bodies == []


def test_hosted_backends_see_redacted_state_local_ones_do_not():
    seen = {}
    hosted = FakeBackend(lambda s, q: (seen.setdefault("hosted", s), noul(0.5)(s, q))[1],
                         name="jev", model="jev/x", egress=True)
    c = ChainClassifier([hosted], mode="live", egress_allowed=True)
    c.ask({"args": "deploy --token=abc123 key sk-ABCDEFGHIJKLMNOPQRST"}, NOUL, purpose="gate")
    assert seen["hosted"] == {"args": "deploy --[REDACTED] key [REDACTED]"}
    local = FakeBackend(lambda s, q: (seen.setdefault("local", s), noul(0.5)(s, q))[1])
    ChainClassifier([local], mode="live").ask({"args": "--token=abc"}, NOUL, purpose="gate")
    assert seen["local"] == {"args": "--token=abc"}
    assert Redactor()("AKIAABCDEFGHIJKLMNOP") == "[REDACTED]"


# ── C6: the node's spec ────────────────────────────────────────────────────

def test_spec_defaults_are_safe_and_roundtrip():
    spec = ClassifierSpec()
    assert spec.mode == "shadow" and spec.egress_allowed is False
    assert spec.bands_for(JEV) is None                      # uncalibrated: never skips
    again = ClassifierSpec.model_validate_json(spec.model_dump_json())
    assert again == spec
    with pytest.raises(ValidationError):
        ClassifierSpec(mode="aggressive")


def test_build_skips_links_it_cannot_make():
    spec = ClassifierSpec(backends=[BackendConfig(kind="jev", model=JEV),
                                    BackendConfig(kind="local", model="local/j")])
    c = build_classifier(spec)                               # no scorer loaded
    assert [b.name for b in c.backends] == ["jev"]


def test_run_view_snapshots_the_spec():
    spec = ClassifierSpec(mode="live", tool_calls_per_run=3)
    handle = ClassifierHandle(spec, build_classifier(spec))
    run = handle.for_run("r1")
    spec.tool_calls_per_run = 99                             # edited mid-run
    assert run.spec.tool_calls_per_run == 3                  # D38: next run sees it


# ── C4: hooks on Silk's real ToolBox / HookRegistry ────────────────────────

def _toolbox_with_task_add():
    tb = ToolBox()
    from pydantic import BaseModel

    class AddArgs(BaseModel):
        title: str
        rationale: str = ""

    @tb.register("task_add", "add a task", args_model=AddArgs)
    def _add(db_pool, user_session, title, rationale=""):
        return f"added {title}"
    return tb


def _call(tb, name, args, cid="c1"):
    tc = NS(id=cid, function=NS(name=name, arguments=json.dumps(args)))
    return asyncio.run(tb.execute_tool_calls_async([tc]))[0]["content"]


def test_task_drift_bounces_once_then_lets_the_retry_through(mk):
    tb = _toolbox_with_task_add()
    rc = mk(lambda b: {"needed": {"type": "noul", "noul": 0.1},
                       "duplicate": {"type": "noul", "noul": 0.2}})
    ctx = HookContext(classifier=lambda: rc, goal=lambda: "Ship v2",
                      task_titles=lambda: ["write parser"])
    register_hook_map(tb.hooks, hook_map("task_drift", ctx, TaskDriftConfig()))
    first = json.loads(_call(tb, "task_add", {"title": "Repaint the logo", "rationale": "nice"}))
    assert first["error"].startswith("Not added") and first["advisory"]["model"] == JEV
    assert _call(tb, "task_add", {"title": "Repaint the logo",
                                  "rationale": "brand refresh is in the goal"}) == \
        "added Repaint the logo"
    sent = rc.http.bodies[0]["state"]
    assert sent["new_task"] == "Repaint the logo" and sent["existing_tasks"] == ["write parser"]


def test_task_drift_is_inert_without_a_classifier_and_quiet_in_shadow(mk):
    for ctx in (HookContext(), HookContext(classifier=lambda: mk(
            lambda b: {"needed": {"type": "noul", "noul": 0.0},
                       "duplicate": {"type": "noul", "noul": 0.0}}, mode="shadow"))):
        tb = _toolbox_with_task_add()
        register_hook_map(tb.hooks, hook_map("task_drift", ctx))
        assert _call(tb, "task_add", {"title": "x"}) == "added x"
    assert ctx.shadow_log and ctx.shadow_log[0]["would"] == "bounce"


def test_findings_gate_routes_auto_ask_drop(mk):
    cfg = FindingsGateConfig()

    def ans(cat, p, dur, gnd, rel):
        return {"category": {"type": "choice", "choice": cat, "confidence": 0.9,
                             "probabilities": {cat: p}},
                "durable": {"type": "noul", "noul": dur},
                "grounded": {"type": "noul", "noul": gnd},
                "relation": {"type": "choice", "choice": rel, "confidence": 0.9,
                             "probabilities": {rel: 0.9}}}
    assert route_finding(ans("rule", 0.9, 0.8, 0.9, "supports"), cfg) == "auto"
    assert route_finding(ans("rule", 0.9, 0.8, 0.9, "contradicts"), cfg) == "ask"
    assert route_finding(ans("decision", 0.6, 0.6, 0.6, "unrelated"), cfg) == "ask"
    assert route_finding(ans("rule", 0.9, 0.3, 0.9, "unrelated"), cfg) == "drop"
    assert route_finding(ans("none", 0.99, 0.9, 0.9, "unrelated"), cfg) == "drop"

    written, offered = [], []
    rc = mk(lambda b: ans("rule", 0.92, 0.85, 0.9, "supersedes"))
    ctx = HookContext(
        classifier=lambda: rc, run_id=lambda: "r1",
        recent_decisions=lambda run: [
            {"id": "decision:1", "actor": "user", "outcome": "denied",
             "prompt": "Delete ~/notes?", "reason": "never delete outside build/ or dist/"},
            {"id": "decision:2", "actor": "user", "outcome": "approved", "prompt": "x",
             "reason": ""},
            {"id": "decision:3", "actor": "jev", "outcome": "approved", "prompt": "y",
             "reason": "z"}],
        nearest_rule=lambda text: {"id": "rules/delete-outside-build", "text": "ask first"},
        write_finding=written.append, offer_finding=offered.append)
    tb = ToolBox()
    register_hook_map(tb.hooks, hook_map("findings_gate", ctx))
    tb.hooks.emit(HOOK_AFTER_RUN, final_text="", rounds=1, elapsed_s=0.1)
    assert len(rc.http.bodies) == 1 and offered == []
    [note] = written
    assert note["relations"] == {"supersedes": "rules/delete-outside-build"}
    assert note["source_decision"] == "decision:1" and note["category"] == "rule"


# ── C5: the classify tool on Silk's real ToolBox ───────────────────────────

def test_classify_tool_budget_cap_and_advisory_tag(mk):
    tb = ToolBox()
    register_classify_tool(tb)
    assert tb.tools["classify"]["category"] == "classifier"
    qs = {"bug": {"type": "noul", "instructions": "Is this a bug report?"},
          "area": {"type": "choice", "instructions": "Which area?",
                   "criteria": {"ui": None, "api": None}}}
    out = json.loads(_call(tb, "classify", {"items": ["x"], "questions": qs}))
    assert "No classifier" in out["error"]                     # nothing bound this run
    rc = mk(lambda b: {"bug": {"type": "noul", "noul": 0.81},
                       "area": {"type": "choice", "choice": "api", "confidence": 0.7,
                                "probabilities": {"api": 0.8, "ui": 0.2}}},
            tool_calls_per_run=2, tool_items_per_call=3)
    bind_run_classifier(tb, rc)
    out = json.loads(_call(tb, "classify", {"items": ["500 on /v1", "typo", "crash"],
                                            "questions": qs, "context": "issue tracker"}))
    assert out["advisory"] is True and out["model"] == JEV and len(out["results"]) == 3
    assert out["results"][0]["answers"] == {"bug": 0.81, "area": {"choice": "api",
                                                                  "confidence": 0.7}}
    assert rc.http.bodies[0]["state"] == {"context": "issue tracker", "item": "500 on /v1"}
    too_many = json.loads(_call(tb, "classify", {"items": ["a"] * 4, "questions": qs}))
    assert "Too many items" in too_many["error"]
    _call(tb, "classify", {"items": ["a"], "questions": qs})
    spent = json.loads(_call(tb, "classify", {"items": ["b"], "questions": qs}))
    assert "budget" in spent["error"]
    bad = _call(tb, "classify", {"items": ["a"], "questions": {"q": {"type": "rank",
                                                                     "instructions": "x"}}})
    assert "unknown type" in bad


# ── C3: the local judge on the GGUF pool ───────────────────────────────────

def test_pool_scorer_sums_only_the_option_tokens():
    from silk.functions.classifier_local import pool_scorer, refuse_self_grading
    sent = []

    def post(url, headers, body, timeout):
        sent.append((url, body))
        prompt = body["prompt"]
        cut = prompt.rindex(" ")                        # option starts at its leading space
        # prompt tokens, then the option split in two tokens
        return {"choices": [{"logprobs": {
            "text_offset": [0, 5, cut, cut + 2],
            "token_logprobs": [None, -1.0, -0.25 if "yes" in prompt[cut:] else -1.5, -0.25]}}]}

    client = NS(base_url="http://127.0.0.1:8081/v1", model_alias="judge",
                headers=lambda: {"Content-Type": "application/json"})
    score = pool_scorer(client, post=post)
    lp = score("State: x\nAnswer:", [" yes", " no"])
    assert lp == [-0.5, -1.75]
    assert sent[0][0].endswith("/completions") and sent[0][1]["echo"] is True
    judge = LocalJudgeBackend(score, model="local/judge")
    a, _ = judge.answer({"x": 1}, NOUL)
    assert 0.7 < a["q"]["noul"] < 0.8                   # softmax(-0.5, -1.75)
    m = NS(model_path="/m/a.gguf")
    with pytest.raises(ValueError):
        refuse_self_grading(m, NS(model_path="/m/a.gguf"))
    refuse_self_grading(m, NS(model_path="/m/b.gguf"))
