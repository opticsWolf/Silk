# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The OpenCode nodes: the dropdown, the slider, and what reaches the handle.

The catalogue logic is pinned in ``test_silk_model_catalog``; this pins
the seam a person actually touches -- that the slider's top is the
model's window, that moving it lands in the catalogue file, and that the
handle carries the window the slider shows.
"""

from __future__ import annotations

import json

import pytest

from silk.functions import model_catalog as mc
from silk.functions import model_endpoint as me
from silk.functions.model_catalog import CatalogModel, ModelCatalog, catalog_for


@pytest.fixture
def catalog_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "CATALOG_DIR", tmp_path)
    cat = ModelCatalog(catalog_for("opencode-go"), tmp_path)
    cat.models = {
        "kimi-k2.5": CatalogModel("kimi-k2.5", context_max=262144, tool_call=True),
        "glm-5": CatalogModel("glm-5", context_max=204800),
        "minimax-m2.5": CatalogModel("minimax-m2.5", context_max=204800,
                                     api="messages"),
    }
    cat.fetched_at = 1.0
    cat.save()
    return tmp_path


@pytest.fixture
def node(catalog_dir, monkeypatch):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    # No network: the endpoint "lists" whatever the catalogue holds.
    monkeypatch.setattr(
        me, "list_model_specs",
        lambda base_url, headers=None, timeout=None: {
            "kimi-k2.5": {}, "glm-5": {}, "minimax-m2.5": {}})
    monkeypatch.setenv("OPENCODE_API_KEY", "test-not-a-real-key")
    from silk.nodes.opencode_endpoint import SilkOpenCodeGoNode

    return SilkOpenCodeGoNode()


def test_the_dropdown_comes_from_the_file(node):
    items = [node._model.itemText(i) for i in range(node._model.count())]
    assert items == ["glm-5", "kimi-k2.5", "minimax-m2.5"]


def test_the_slider_tops_out_at_the_model_window(node):
    node._model.setCurrentText("kimi-k2.5")
    assert node._context.maximum() == 262144
    assert node._context.value() == 262144, "starts at the maximum"
    node._model.setCurrentText("glm-5")
    assert node._context.maximum() == 204800


def test_a_lower_context_is_stored_per_model(node, catalog_dir):
    node._model.setCurrentText("kimi-k2.5")
    node._context.setValue(131072)
    node._persist_context()
    data = json.loads((catalog_dir / "opencode-go.json").read_text())
    assert data["context_overrides"] == {"kimi-k2.5": 131072}
    node._model.setCurrentText("glm-5")
    node._model.setCurrentText("kimi-k2.5")
    assert node._context.value() == 131072, "the choice comes back"


def test_the_handle_carries_the_window_and_the_tools_flag(node):
    out = node.compute({"model": "kimi-k2.5", "credential": "OPENCODE_API_KEY",
                        "context_length": 100000, "supports_tools": True})
    handle = out["model_obj"]
    assert handle is not None
    assert handle["context_length"] == 100000
    assert handle["supports_tools"] is True
    assert handle["provider"] == "opencode-go"
    assert handle["base_url"] == "https://opencode.ai/zen/go/v1"
    assert "pricing" not in handle, "Go is a subscription"


# ── the native-tools box is the answer ───────────────────────────────────

def test_picking_a_tool_calling_model_ticks_the_box(node):
    node._model.setCurrentText("kimi-k2.5")
    assert node._native_tools.isChecked(), "the catalogue says it calls tools"


def test_a_model_the_catalogue_says_nothing_about_leaves_the_box(node):
    node._model.setCurrentText("kimi-k2.5")
    node._model.setCurrentText("glm-5")   # tool_call unknown
    assert node._native_tools.isChecked(), "no answer is not a 'no'"


def test_an_untick_means_fences_even_where_the_catalogue_says_tools(node):
    """The box used to be overridden by the catalogue, so it could not
    say "this model calls tools badly, use fences"."""
    out = node.compute({"model": "kimi-k2.5", "credential": "OPENCODE_API_KEY",
                        "supports_tools": False})
    handle = out["model_obj"]
    assert handle is not None
    assert "supports_tools" not in handle
    assert "native tools" not in node._sync_status


@pytest.mark.parametrize("model, saved, expected", [
    ("kimi-k2.5", False, True),   # the old override ran it native
    ("glm-5", False, False),      # no answer then, none now
    ("glm-5", True, True),        # a tick was always a tick
    ("gone-model", False, False),  # nothing left to ask
])
def test_an_old_graph_keeps_the_transport_it_ran_on(catalog_dir, model, saved,
                                                    expected):
    from silk.nodes.opencode_endpoint import SilkOpenCodeGoNode

    state = {"widget_data": {"model": model, "supports_tools": saved}}
    migrated = SilkOpenCodeGoNode.migrate_state(state, 1)
    assert migrated["widget_data"]["supports_tools"] is expected


def test_a_wired_context_cannot_exceed_the_window(node):
    out = node.compute({"model": "glm-5", "credential": "OPENCODE_API_KEY",
                        "context_length": 10_000_000})
    assert out["model_obj"]["context_length"] == 204800


def test_a_non_chat_model_is_flagged_not_refused(node):
    out = node.compute({"model": "minimax-m2.5",
                        "credential": "OPENCODE_API_KEY"})
    assert out["model_obj"] is not None
    assert "/messages" in node._sync_status
