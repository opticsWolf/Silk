# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Several classifiers as one: the combination math and the policies.

The node surface (ports, presets, bands table) is pinned in
``test_classifier_ensemble_wiring``; this pins the part that decides --
that agreeing members earn a skip together, disagreeing members veto
each other into the human's lap, and an early stop keeps the member's
identity while only a real combination mints the joined id.
"""

import pytest

from silk.functions.classifier import (
    BackendConfig,
    BandsConfig,
    ClassifierError,
    ClassifierSpec,
    EnsembleClassifier,
    EnsembleSpec,
    Verdict,
    build_classifier,
    build_ensemble,
    combine_answers,
)


def ans(choice=None, probs=None, conf=0.95, score=None, noul=None):
    if choice is not None:
        return {"type": "choice", "choice": choice,
                "probabilities": dict(probs or {choice: 1.0}),
                "confidence": conf}
    if score is not None:
        return {"type": "score", "score": score, "confidence": conf}
    return {"type": "noul", "noul": noul, "confidence": conf}


CQs = {"intent": {"type": "choice", "instructions": "i",
                  "criteria": {"read": "r", "write": "w"}}}
SQs = {"risk": {"type": "score", "instructions": "r",
                "criteria": ["none", "low", "moderate", "high", "severe"]}}
NQs = {"reversible": {"type": "noul", "instructions": "r"}}


def chain(model, answers, **kw):
    spec = ClassifierSpec(
        backends=[BackendConfig(kind="fake", model=model)], **kw)
    return build_classifier(spec, secrets=lambda n: None,
                            fake=lambda s, q: dict(answers))


def verdict(model, answers, tried=None):
    """A member verdict, already answered."""
    return Verdict(dict(answers), "fake", model, "gate", False, 10, 0.0,
                   "", list(tried or []))


# ── the combination ──────────────────────────────────────────────────────

def test_agreeing_choices_combine_to_the_winner():
    a = verdict("m/b", {"intent": ans("read", {"read": 0.9, "write": 0.1})})
    b = verdict("m/a", {"intent": ans("read", {"read": 0.7, "write": 0.3})})
    combined, model = combine_answers(CQs, [a, b])
    assert combined["intent"]["choice"] == "read"
    assert combined["intent"]["probabilities"]["read"] == pytest.approx(0.8)
    assert model == "m/a+m/b", "joined and sorted, whatever the answer order"


def test_disagreeing_choices_keep_little_confidence():
    a = verdict("m/a", {"intent": ans("read", {"read": 0.9, "write": 0.1}, conf=0.9)})
    b = verdict("m/b", {"intent": ans("write", {"read": 0.1, "write": 0.9}, conf=0.9)})
    combined, _ = combine_answers(CQs, [a, b])
    assert combined["intent"]["confidence"] == pytest.approx(0.45), \
        "a 50/50 split must not clear any sane min_confidence"


def test_agreeing_scores_keep_confidence_disagreeing_lose_it():
    calm = {"risk": ans(score=0.0, conf=0.9)}
    agree, _ = combine_answers(
        SQs, [verdict("m/a", calm), verdict("m/b", calm)])
    assert agree["risk"]["score"] == pytest.approx(0.0)
    assert agree["risk"]["confidence"] == pytest.approx(0.9)
    split, _ = combine_answers(
        SQs, [verdict("m/a", calm),
              verdict("m/b", {"risk": ans(score=4.0, conf=0.9)})])
    assert split["risk"]["score"] == pytest.approx(2.0)
    assert split["risk"]["confidence"] == pytest.approx(0.0), \
        "maximal spread vetoes into the human's lap"


def test_noul_averages_and_clamps():
    both, _ = combine_answers(
        NQs, [verdict("m/a", {"reversible": ans(noul=0.9)}),
              verdict("m/b", {"reversible": ans(noul=0.7)})])
    assert both["reversible"]["noul"] == pytest.approx(0.8)


def test_a_single_verdict_passes_through_untouched():
    a = verdict("m/a", {"intent": ans("read")})
    combined, model = combine_answers(CQs, [a])
    assert model == "m/a" and combined == a.answers


def test_nothing_to_combine_is_an_error():
    with pytest.raises(ClassifierError, match="no member answered"):
        combine_answers(CQs, [])


# ── the policies ─────────────────────────────────────────────────────────

def member(model, answers, calls=None, **kw):
    def fn(s, q):
        if calls is not None:
            calls.append(model)
        return dict(answers)
    spec = ClassifierSpec(
        backends=[BackendConfig(kind="fake", model=model)], **kw)
    return build_classifier(spec, secrets=lambda n: None, fake=fn)


def test_first_returns_the_first_answer_untouched():
    a = member("m/a", {"intent": ans("read")})
    b = member("m/b", {"intent": ans("write")})
    ens = build_ensemble(EnsembleSpec(policy="first"), [("a", a), ("b", b)])
    v = ens.ask({"g": 1}, CQs, purpose="gate")
    assert v.ok and v.backend == "fake" and v.model == "m/a"


def test_first_skips_a_broken_member_and_records_it():
    def boom(s, q):
        raise RuntimeError("down")
    spec = ClassifierSpec(backends=[BackendConfig(kind="fake", model="m/a")])
    a = build_classifier(spec, secrets=lambda n: None, fake=boom)
    b = member("m/b", {"intent": ans("read")})
    ens = build_ensemble(EnsembleSpec(policy="first"), [("a", a), ("b", b)])
    v = ens.ask({"g": 1}, CQs, purpose="gate")
    assert v.ok and v.model == "m/b"
    assert any("member a" in t for t in v.tried)


def test_first_with_nobody_answering_is_an_error_for_the_human():
    def boom(s, q):
        raise RuntimeError("down")
    spec = ClassifierSpec(backends=[BackendConfig(kind="fake", model="m/a")])
    a = build_classifier(spec, secrets=lambda n: None, fake=boom)
    ens = build_ensemble(EnsembleSpec(policy="first"), [("a", a)])
    v = ens.ask({"g": 1}, CQs, purpose="gate")
    assert not v.ok and v.tried


def test_consensus_combines_and_merges_cost_and_tried():
    a = member("m/a", {"intent": ans("read", {"read": 0.8, "write": 0.2})})
    b = member("m/b", {"intent": ans("read", {"read": 0.6, "write": 0.4})})
    ens = build_ensemble(EnsembleSpec(policy="consensus"), [("a", a), ("b", b)])
    v = ens.ask({"g": 1}, CQs, purpose="gate")
    assert v.ok and v.backend == "ensemble" and v.model == "m/a+m/b"
    assert v.answers["intent"]["choice"] == "read"


def test_cascade_stops_early_on_confidence_and_keeps_the_member():
    calls: list = []
    a = member("m/a", {"intent": ans("read", conf=0.95)}, calls=calls)
    b = member("m/b", {"intent": ans("read", conf=0.95)}, calls=calls)
    ens = build_ensemble(EnsembleSpec(policy="cascade", agreement=0.7),
                         [("a", a), ("b", b)])
    v = ens.ask({"g": 1}, CQs, purpose="gate")
    assert v.ok and v.model == "m/a", "early stop: the member's id, member bands"
    assert calls == ["m/a"], "b was never asked"


def test_cascade_continues_on_doubt_and_combines():
    a = member("m/a", {"intent": ans("read", {"read": 0.9, "write": 0.1}, conf=0.5)})
    b = member("m/b", {"intent": ans("read", {"read": 0.7, "write": 0.3}, conf=0.6)})
    ens = build_ensemble(EnsembleSpec(policy="cascade", agreement=0.7),
                         [("a", a), ("b", b)])
    v = ens.ask({"g": 1}, CQs, purpose="gate")
    assert v.ok and v.model == "m/a+m/b"


def test_off_members_are_skipped_and_off_ensemble_is_silent():
    a = member("m/a", {"intent": ans("read")}, mode="off")
    b = member("m/b", {"intent": ans("write")})
    ens = build_ensemble(EnsembleSpec(policy="consensus"), [("a", a), ("b", b)])
    v = ens.ask({"g": 1}, CQs, purpose="gate")
    assert v.ok and v.model == "m/b", "a single answer passes through"
    ens_off = build_ensemble(EnsembleSpec(mode="off"), [("b", b)])
    assert not ens_off.ask({"g": 1}, CQs, purpose="gate").ok


def test_an_unknown_policy_is_rejected_at_build():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EnsembleSpec(policy="vibes")
    with pytest.raises(ClassifierError, match="unknown ensemble policy"):
        EnsembleClassifier([], policy="vibes")


# ── the joined id earns its skip like any other model ────────────────────

def test_the_ensemble_id_routes_through_bands():
    from silk.functions.triage import route_approval
    full = {
        "reversible": ans(noul=0.95), "in_scope": ans(noul=0.95),
        "consistent": ans(noul=0.95),
        "risk": ans(score=0.0),
        "intent": ans("read", {"read": 0.95, "write": 0.05}, conf=0.95),
    }
    questions = dict(NQs)
    questions["risk"] = SQs["risk"]
    questions["intent"] = CQs["intent"]
    questions["in_scope"] = {"type": "noul", "instructions": "s"}
    questions["consistent"] = {"type": "noul", "instructions": "c"}
    a = member("m/a", full)
    b = member("m/b", full)
    ens = build_ensemble(EnsembleSpec(policy="consensus", mode="live"),
                         [("a", a), ("b", b)])
    v = ens.ask({"g": 1}, questions, purpose="gate")
    assert v.ok
    ask = route_approval(v.answers, level="triage",
                         bands=EnsembleSpec().bands_for(v.model))
    assert ask.route == "ask" and "no calibrated bands" in ask.why
    row = {"m/a+m/b": BandsConfig(allow_intents=["read"], min_confidence=0.5)}
    allow = route_approval(v.answers, level="triage",
                           bands=row["m/a+m/b"])
    assert allow.route == "allow"


def test_decision_member_reads_pool_and_hosted_handles():
    from types import SimpleNamespace as NS
    from silk.functions.classifier import decision_member
    pool = NS(model_path="/models/laya-en.gguf",
              base_url="http://127.0.0.1:8080/v1", model_alias="default")
    model_id, entry, kwargs = decision_member(
        {"backend": "systemone", "role": "decision", "pool": pool})
    assert model_id == "local/laya-en"
    assert entry.model == "local/laya-en" and entry.egress is False
    assert kwargs["systemone"].base_url == "http://127.0.0.1:8080/v1"
    model_id, entry, kwargs = decision_member(
        {"backend": "systemone", "role": "decision", "model": "jev/jev-1.13.0",
         "url": "https://example.invalid/v1/systemone/",
         "credential": "JEV_API_KEY", "egress": True})
    assert model_id == "jev/jev-1.13.0"
    assert entry.url == "https://example.invalid/v1/systemone"
    assert entry.key_name == "JEV_API_KEY" and entry.egress is True
    assert kwargs == {}
    assert decision_member(None) is None
    assert decision_member({}) is None
    assert decision_member({"backend": "openai", "model": object()}) is None
    assert decision_member({"backend": "systemone", "role": "decision"}) is None
    with pytest.raises(ValueError, match="pin a model version"):
        decision_member({"backend": "systemone", "role": "decision",
                         "model": "jev/jev-latest", "url": "https://x/"})


def test_the_fallback_answers_only_when_nobody_did():
    asked: list = []

    def judge_fn(s, q):
        asked.append(True)
        return dict({"intent": ans("read")})
    spec = ClassifierSpec(backends=[BackendConfig(kind="fake", model="local/j")])
    judge = build_classifier(spec, secrets=lambda n: None, fake=judge_fn)
    a = member("m/a", {"intent": ans("read")})
    ens = build_ensemble(EnsembleSpec(policy="consensus"),
                         [("a", a)], fallback=judge)
    v = ens.ask({"g": 1}, CQs, purpose="gate")
    assert v.ok and v.model == "m/a" and not asked, \
        "the fallback stays silent while members answer"
    failing = build_classifier(
        ClassifierSpec(backends=[BackendConfig(kind="fake", model="m/f")]),
        secrets=lambda n: None,
        fake=lambda s, q: (_ for _ in ()).throw(RuntimeError("down")))
    ens = build_ensemble(EnsembleSpec(policy="consensus"),
                         [("f", failing)], fallback=judge)
    v = ens.ask({"g": 1}, CQs, purpose="gate")
    assert v.ok and v.model == "local/j" and len(asked) == 1
    assert v.tried == ["member f: fake: RuntimeError: down"], \
        "the member failure stays on the record; the fallback adds none"


def test_a_failed_fallback_still_defers_to_the_human():
    def down(s, q):
        raise RuntimeError("down")
    spec = ClassifierSpec(backends=[BackendConfig(kind="fake", model="m")])
    bad = build_classifier(spec, secrets=lambda n: None, fake=down)
    ens = build_ensemble(EnsembleSpec(policy="first"),
                         [("a", bad)], fallback=bad)
    v = ens.ask({"g": 1}, CQs, purpose="gate")
    assert not v.ok and any(t.startswith("fallback:") for t in v.tried)


def test_the_ensemble_handle_passes_the_binding_check():
    from silk.functions.classifier import ClassifierHandle, EnsembleHandle
    handle = EnsembleHandle(
        EnsembleSpec(), build_ensemble(EnsembleSpec(), []), members=())
    assert isinstance(handle, ClassifierHandle)
    assert handle.judge_model is None


def test_ensemble_run_mirrors_the_classifier_run():
    handle_like = build_ensemble(EnsembleSpec(), [])
    from silk.functions.classifier import EnsembleHandle
    handle = EnsembleHandle(EnsembleSpec(), handle_like, ("a",))
    run = handle.for_run("r1")
    assert run.mode == "shadow" and run.spec is not handle.spec
    assert run.bands_for("x") is None and run.tool_calls == 0
    cap, budget = run.spec.tool_items_per_call, run.spec.tool_calls_per_run
    assert (cap, budget) == (50, 100), "the classify tool reads both off the spec"
    assert run.meter is handle.classifier.meter
