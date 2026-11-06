# -*- coding: utf-8 -*-
"""The Jev loader: endpoint in, classifier-model handle out.

The wire shape (``hosted_decision_handle``) is pinned Qt-free in
``test_silk_systemone``; this pins the node seam -- that the provider
comes from the Endpoint wire, that nothing is probed, and that the
credential travels by name only.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def node():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from silk.nodes.jev_loader import JevLoaderNode
    return JevLoaderNode()


def gather(node, wired=None):
    core = node._widget_core
    inputs = {name: core.get_port_value(name) for name in core.bindings()}
    inputs.update({"endpoint": None})
    inputs.update(wired or {})
    return inputs


def endpoint(provider="openrouter", url="https://openrouter.ai/api/v1",
             credential="OPENROUTER_API_KEY"):
    return {"provider": provider, "base_url": url, "credential": credential}


def test_an_endpoint_and_a_model_become_a_classifier_model(node):
    out = node.compute(gather(node, {"endpoint": endpoint()}))
    handle = out["model_obj"]
    assert handle == {"backend": "systemone", "role": "decision",
                      "model": "jev/jev-1.13.0",
                      "url": "https://openrouter.ai/api/v1",
                      "credential": "OPENROUTER_API_KEY", "egress": True}
    from weave.node.port_registry import PortRegistry
    import silk.nodes.silk_ports  # noqa: F401 (registers the type)
    assert PortRegistry._by_name["classifier_model"].validator(handle)


def test_without_an_endpoint_there_is_nothing_to_point_at(node):
    out = node.compute(gather(node))
    assert out["model_obj"] is None
    assert "Wire an Endpoint" in node._sync_status


def test_an_unpinned_model_is_refused_not_emitted(node):
    node._model.setText("jev/jev-latest")
    out = node.compute(gather(node, {"endpoint": endpoint()}))
    assert out["model_obj"] is None


def test_the_key_travels_by_name_never_by_value(node, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-a-real-key")
    out = node.compute(gather(node, {"endpoint": endpoint()}))
    handle = out["model_obj"]
    assert handle["credential"] == "OPENROUTER_API_KEY"
    assert "sk-test-not-a-real-key" not in repr(handle)
    assert "key 'OPENROUTER_API_KEY' ✓" in node._sync_status


def test_migration_drops_the_old_provider_widgets(node):
    from silk.nodes.jev_loader import JevLoaderNode
    saved = {"widget_data": {"model": "jev/jev-1.13.0",
                             "url": "https://old.example/v1",
                             "key_name": "OLD_KEY"}}
    out = JevLoaderNode.migrate_state(saved, 1)
    assert out["widget_data"] == {"model": "jev/jev-1.13.0"}
