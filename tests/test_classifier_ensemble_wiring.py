# -*- coding: utf-8 -*-
"""The Ensemble node: ports, wiring, presets.

The combination math and the policies are pinned Qt-free in
``test_classifier_ensemble``; this pins the canvas seam -- that
classifier-model handles become member chains, that the judge is only
a fallback, and that the stats say who answered.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from silk.functions.classifier import BandsConfig


def pool(path, base="http://127.0.0.1:9/v1", alias="default"):
    return NS(model_path=path, base_url=base, model_alias=alias)


def decision_pool(path):
    return {"backend": "systemone", "role": "decision",
            "pool": pool(path)}


def decision_hosted(model="m/a", url="https://example.invalid/v1/systemone",
                    credential=""):
    return {"backend": "systemone", "role": "decision", "model": model,
            "url": url, "credential": credential, "egress": True}


def judge(client=True):
    return {"backend": "gguf",
            "pool": NS(client=NS(base_url="http://127.0.0.1:9/v1")
                       if client else None)}


@pytest.fixture
def node():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from silk.nodes.ensemble import SilkEnsembleNode
    return SilkEnsembleNode()


def gather(node, wired=None):
    core = node._widget_core
    inputs = {name: core.get_port_value(name) for name in core.bindings()}
    inputs.update({"classifier_a": None, "classifier_b": None,
                   "classifier_c": None, "judge_model": None})
    inputs.update(wired or {})
    return inputs


def ports(node_ports):
    return {p.name: p.datatype for p in node_ports}


def test_node_ports_and_safe_defaults(node):
    assert ports(node.inputs) == {"classifier_a": "classifier_model",
                                   "classifier_b": "classifier_model",
                                   "classifier_c": "classifier_model",
                                   "judge_model": "model_handle"}
    assert ports(node.outputs) == {"classifier": "silk_classifier",
                                   "stats": "dict"}
    out = node.compute(gather(node))
    assert out["classifier"].spec.policy == "cascade"
    assert out["classifier"].spec.mode == "shadow"
    assert out["stats"]["members"] == []
    assert "nobody wired" in node._sync_status


def test_two_pool_members_become_member_chains(node):
    out = node.compute(gather(node, {
        "classifier_a": decision_pool("/models/laya-en.gguf"),
        "classifier_b": decision_pool("/models/julia-1.gguf")}))
    h = out["classifier"]
    assert [label for label, _ in h.classifier.members] == ["a", "b"]
    assert out["stats"]["members"] == ["a", "b"]
    assert "a, b" in node._sync_status
    assert h.classifier.fallback is None


def test_a_hosted_member_carries_its_own_endpoint(node):
    out = node.compute(gather(node, {
        "classifier_a": decision_hosted("jev/jev-1.14.0")}))
    link = out["classifier"].classifier.members[0][1].backends[0]
    assert link.model == "jev/jev-1.14.0"
    assert link.url == "https://example.invalid/v1/systemone"
    assert link.egress is True


def test_a_chat_handle_on_a_member_port_is_skipped(node):
    out = node.compute(gather(node, {
        "classifier_a": {"backend": "openai", "model": object()}}))
    assert out["stats"]["members"] == []


def test_a_judge_without_a_client_is_skipped_gracefully(node):
    out = node.compute(gather(node, {
        "classifier_a": decision_pool("/models/laya-en.gguf"),
        "judge_model": judge(client=False)}))
    assert out["stats"]["members"] == ["a"]
    assert out["classifier"].classifier.fallback is None


def test_a_judge_with_a_client_becomes_the_fallback(node):
    out = node.compute(gather(node, {
        "classifier_a": decision_pool("/models/laya-en.gguf"),
        "judge_model": judge(client=True)}))
    h = out["classifier"]
    assert h.classifier.fallback is not None
    assert h.judge_model is not None, "carried for the self-grading refusal"
    assert "+ judge fallback" in node._sync_status


def test_the_table_knows_members_joins_and_judge(node):
    out = node.compute(gather(node, {
        "classifier_a": decision_pool("/models/laya-en.gguf"),
        "classifier_b": decision_hosted("m/b"),
        "judge_model": {"backend": "gguf",
                        "pool": NS(model_path="/models/qwen3-1.7b-judge.gguf",
                                   client=NS(base_url="http://127.0.0.1:9/v1"))}}))
    assert out["stats"]["members"] == ["a", "b"]
    node.on_evaluate_finished()
    known = node.bands_table.known_models()
    assert "local/laya-en" in known and "m/b" in known
    assert "local/laya-en+m/b" in known, "the sorted join is offered upfront"
    assert "local/qwen3-1.7b-judge" in known
    assert "fallback" not in " ".join(known), "tried prefixes are not ids"


def test_policy_comes_from_the_form(node):
    node.combo_policy.setCurrentIndex(
        node.combo_policy.findData("consensus"))
    node.spin_budget.setValue(25)
    out = node.compute(gather(node, {
        "classifier_a": decision_hosted("m/a"),
        "classifier_b": decision_hosted("m/b")}))
    assert out["classifier"].spec.policy == "consensus"
    assert out["classifier"].classifier.policy == "consensus"
    assert out["classifier"].spec.tool_calls_per_run == 25


def test_presets_roundtrip_the_whole_form(node):
    node.combo_policy.setCurrentIndex(
        node.combo_policy.findData("consensus"))
    node.spin_agreement.setValue(0.85)
    node.bands_table.add_model("m/a+m/b", BandsConfig(min_confidence=0.5))
    saved = node._collect_preset()
    node.combo_policy.setCurrentIndex(0)
    node.spin_agreement.setValue(0.7)
    node.bands_table.set_bands({})
    from silk.functions.presets import EnsemblePreset
    node._apply_preset(EnsemblePreset(name="p", **saved))
    spec = node.compute(gather(node))["classifier"].spec
    assert spec.policy == "consensus" and spec.agreement == 0.85
    assert spec.bands_for("m/a+m/b").min_confidence == 0.5
