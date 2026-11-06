# -*- coding: utf-8 -*-
"""Jev and local decision models as one interchangeable backend, with the
human as the guaranteed final link.

* ``jev`` and ``systemone`` kinds answer through the same
  :class:`~silk.functions.classifier.SystemOneBackend`: same questions,
  same answers, same usage shape. Only the URL, the credential, the model
  id (the bands key) and the egress flag differ.
* The system always defers to a human (a) when no decision backend can
  answer and (b) when the answer is not clear cut: uncalibrated model,
  failed band checks, non-triage policy level, or anything but live mode.
"""

import math

from silk.functions.classifier import (
    BackendConfig, BandsConfig, ClassifierHandle,
    ClassifierSpec, JevBackend, SystemOneBackend,
    build_classifier,
)
from silk.functions.triage import (
    APPROVAL_QUESTIONS, effective_level, triage_approval,
)


JEV = "jev/jev-1.13.0"
LAYA = "local/laya-en"


class StubHttp:
    """Plays a hosted /v1/systemone endpoint."""

    def __init__(self, reply, model="jev-1.13.0"):
        self.reply, self.model, self.bodies = reply, model, []

    def __call__(self, url, headers, body, timeout):
        self.bodies.append(body)
        return {"model": self.model, "answers": self.reply(body),
                "usage": {"input_tokens": 1500, "output_tokens": 0}}


class FakePool:
    """Plays a Classifier Loader pool (``decide(state, questions)``)."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []
        self.base_url = "http://127.0.0.1:9999/v1"
        self.model_alias = "default"

    def decide(self, state, questions, timeout=60.0):
        self.calls.append((state, questions))
        return self.answers


def gate_answers(conf=0.95, consistent=0.95):
    return {
        "reversible": {"type": "noul", "noul": 0.95},
        "in_scope": {"type": "noul", "noul": 0.95},
        "consistent": {"type": "noul", "noul": consistent},
        "risk": {"type": "score", "score": 0.5, "confidence": conf,
                 "legend": {"0": "none", "1": "low", "2": "moderate",
                            "3": "high", "4": "severe"},
                 "probabilities": {"0": 0.5, "1": 0.5, "2": 0.0,
                                   "3": 0.0, "4": 0.0}},
        "intent": {"type": "choice", "choice": "routine", "confidence": conf,
                   "probabilities": {"routine": 0.9, "exploratory": 0.05,
                                      "destructive": 0.02, "exfiltration": 0.02,
                                      "escalation": 0.01}},
    }


def run_classifier(backends, *, mode="live", bands=True, secrets=None,
                   systemone=None, http=None, egress=True):
    spec = ClassifierSpec(
        mode=mode, egress_allowed=egress, backends=backends,
        bands={backends[0].model: BandsConfig()} if bands else {})
    chain = build_classifier(spec, secrets=secrets or (lambda n: None),
                             systemone=systemone, http=http)
    return ClassifierHandle(spec, chain).for_run("r1")


# ── interchangeable: one wire shape, two transports ───────────────────────

def test_jev_is_a_systemone_backend():
    assert isinstance(JevBackend(key="k"), SystemOneBackend)
    assert JevBackend(key="k").egress is True


def test_same_answers_through_jev_and_pool():
    answers = gate_answers()
    via_jev = run_classifier(
        [BackendConfig(kind="jev", model=JEV)],
        secrets=lambda n: "k-test",
        http=StubHttp(lambda body: answers))
    via_pool = run_classifier(
        [BackendConfig(kind="systemone", model=LAYA)],
        systemone=FakePool(answers))
    state = {"goal": "g"}
    a = via_jev.ask(state, APPROVAL_QUESTIONS, purpose="gate")
    b = via_pool.ask(state, APPROVAL_QUESTIONS, purpose="gate")
    assert a.ok and b.ok and a.answers == b.answers
    assert (a.backend, a.model) == ("jev", JEV)
    assert (b.backend, b.model) == ("systemone", LAYA)
    assert b.cost_usd == 0.0 and b.input_tokens > 0


def test_systemone_sends_the_jev_wire_format():
    http = StubHttp(lambda body: gate_answers())
    # No pool client: hosted fallback uses the shared Bearer + model body.
    rc = run_classifier(
        [BackendConfig(kind="systemone", model="systemone/9",
                       url="https://example.invalid/v1/systemone", egress=True)],
        secrets=lambda n: "k-test", http=http, egress=True)
    v = rc.ask({"goal": "g"}, APPROVAL_QUESTIONS, purpose="gate")
    assert v.ok
    assert http.bodies[0]["model"] == "systemone/9"
    assert set(http.bodies[0]) == {"model", "state", "questions"}
    assert rc.handle.classifier.backends[0].egress is True  # hosted: metered, redacted


def test_local_pool_needs_no_key_or_egress():
    pool = FakePool(gate_answers())
    rc = run_classifier([BackendConfig(kind="systemone", model=LAYA)],
                        systemone=pool)
    v = rc.ask({"goal": "g"}, APPROVAL_QUESTIONS, purpose="gate")
    assert v.ok and pool.calls and v.cost_usd == 0.0


# ── (a) nobody can answer -> the human ────────────────────────────────────

def test_unavailable_backends_defer_to_human():
    rc = run_classifier(
        [BackendConfig(kind="systemone", model=LAYA),
         BackendConfig(kind="jev", model=JEV)])
    v = rc.ask({"goal": "g"}, APPROVAL_QUESTIONS, purpose="gate")
    assert not v.ok and v.tried  # pool missing, key missing: both recorded
    t = triage_approval(rc, {"goal": "g"}, level="triage")
    assert t is not None and t.route == "ask"


def test_mode_off_is_human():
    rc = run_classifier([BackendConfig(kind="jev", model=JEV)], mode="off")
    assert triage_approval(rc, {"goal": "g"}, level="triage") is None
    assert effective_level("triage", rc) == "human"
    assert effective_level("human", rc) == "human"


# ── (b) not clear cut -> the human ────────────────────────────────────────

def test_uncalibrated_model_never_skips():
    rc = run_classifier([BackendConfig(kind="systemone", model=LAYA)],
                        systemone=FakePool(gate_answers()), bands=False)
    t = triage_approval(rc, {"goal": "g"}, level="triage")
    assert t.route == "ask" and "no calibrated bands" in t.why


def test_low_confidence_asks_with_reasons():
    rc = run_classifier([BackendConfig(kind="systemone", model=LAYA)],
                        systemone=FakePool(gate_answers(conf=0.5, consistent=0.5)))
    t = triage_approval(rc, {"goal": "g"}, level="triage")
    assert t.route == "ask" and t.why  # failed band checks are named


def test_clear_cut_allows_only_in_live_triage():
    rc = run_classifier([BackendConfig(kind="systemone", model=LAYA)],
                        systemone=FakePool(gate_answers()))
    assert triage_approval(rc, {"goal": "g"}, level="triage").route == "allow"
    assert triage_approval(rc, {"goal": "g"}, level="human").route == "ask"
    shadow = run_classifier([BackendConfig(kind="systemone", model=LAYA)],
                            systemone=FakePool(gate_answers()), mode="shadow")
    t = triage_approval(shadow, {"goal": "g"}, level="triage")
    assert t.route == "ask" and t.shadow_route == "allow"


def test_hosted_systemone_is_metered_and_redacted():
    http = StubHttp(lambda body: gate_answers())
    denied = run_classifier(
        [BackendConfig(kind="systemone", model="systemone/9",
                       url="https://example.invalid/v1/systemone", egress=True)],
        secrets=lambda n: "k-test", http=http, egress=False)
    v = denied.ask({"goal": "g"}, APPROVAL_QUESTIONS, purpose="gate")
    assert not v.ok and "egress not allowed" in v.tried[0]
    assert http.bodies == []
    rc = run_classifier(
        [BackendConfig(kind="systemone", model="systemone/9",
                       url="https://example.invalid/v1/systemone", egress=True)],
        secrets=lambda n: "k-test", http=http, egress=True)
    v = rc.ask({"args": "deploy --token=abc123"}, APPROVAL_QUESTIONS,
               purpose="gate")
    assert v.ok and math.isclose(v.cost_usd, 1500 * 0.042 / 1e6)
    sent = http.bodies[0]["state"]
    assert sent != {"args": "deploy --token=abc123"}  # redacted for egress
