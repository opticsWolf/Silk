# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

OpenCode catalogues: the model list on disk, and the context it implies.

OpenCode's ``/v1/models`` names models and nothing else; models.dev
carries their windows. What is pinned here is how the two are combined,
what happens when one is missing, and that a person's chosen context
length is kept across a refresh and never exceeds the model's window.
"""

from __future__ import annotations

import json

import pytest

from silk.functions import model_catalog as mc
from silk.functions.model_catalog import (
    CatalogModel, ModelCatalog, catalog_for, merge_sources,
    models_from_models_dev,
)
from silk.functions.model_endpoint import provider_for

MODELS_DEV = {
    "opencode-go": {
        "id": "opencode-go",
        "npm": "@ai-sdk/openai-compatible",
        "models": {
            "kimi-k2.5": {"name": "Kimi K2.5", "tool_call": True,
                          "limit": {"context": 262144, "output": 65536}},
            "glm-5": {"name": "GLM-5", "limit": {"context": 204800}},
            "minimax-m2.5": {"name": "MiniMax M2.5",
                             "provider": {"npm": "@ai-sdk/anthropic"},
                             "limit": {"context": 204800}},
        },
    },
    "opencode": {
        "npm": "@ai-sdk/openai-compatible",
        "models": {
            "qwen3-coder": {"limit": {"context": 262144},
                            "cost": {"input": 0.45, "output": 1.8}},
        },
    },
}


@pytest.fixture
def go(tmp_path):
    return ModelCatalog(catalog_for("opencode-go"), tmp_path)


@pytest.fixture
def sources(monkeypatch):
    """Make the endpoint and models.dev answer (or fail) on demand."""
    def _install(endpoint=None, dev=None, endpoint_err=None, dev_err=None):
        def fake_list(base_url, headers=None, timeout=None):
            if endpoint_err:
                raise RuntimeError(endpoint_err)
            return {name: {} for name in (endpoint or [])}

        def fake_dev(timeout=None):
            if dev_err:
                raise RuntimeError(dev_err)
            return dev if dev is not None else MODELS_DEV
        monkeypatch.setattr(mc, "list_model_specs", fake_list)
        monkeypatch.setattr(mc, "fetch_models_dev", fake_dev)
    return _install


# ── reading models.dev ───────────────────────────────────────────────────

def test_models_dev_supplies_the_window_and_the_wire_format():
    models = models_from_models_dev(MODELS_DEV, "opencode-go")
    assert models["kimi-k2.5"].context_max == 262144
    assert models["kimi-k2.5"].tool_call is True
    assert models["glm-5"].speaks_chat
    assert models["minimax-m2.5"].api == "messages", (
        "a model OpenCode serves on /messages must be flagged, not hidden"
    )


def test_an_unknown_provider_is_an_empty_list_not_an_error():
    assert models_from_models_dev(MODELS_DEV, "nobody") == {}
    assert models_from_models_dev({"x": "junk"}, "x") == {}


# ── combining the two sources ────────────────────────────────────────────

def test_the_endpoint_decides_which_models_exist():
    annotated = models_from_models_dev(MODELS_DEV, "opencode-go")
    models, source = merge_sources(["glm-5", "brand-new"], annotated)
    assert sorted(models) == ["brand-new", "glm-5"], (
        "models.dev may lag or list what this key cannot reach"
    )
    assert models["glm-5"].context_max == 204800
    assert models["brand-new"].context_max == 0, "unknown, never guessed"
    assert "models.dev" in source


def test_without_models_dev_the_list_stands_with_windows_unknown():
    models, source = merge_sources(["glm-5"], None)
    assert models["glm-5"].context_max == 0
    assert "unknown" in source


def test_without_the_endpoint_models_dev_still_fills_the_dropdown():
    annotated = models_from_models_dev(MODELS_DEV, "opencode-go")
    models, source = merge_sources(None, annotated)
    assert len(models) == 3 and "models.dev only" in source


# ── the file ─────────────────────────────────────────────────────────────

def test_a_refresh_writes_a_file_the_next_node_reads(go, sources, tmp_path):
    sources(endpoint=["kimi-k2.5", "glm-5"])
    message = go.refresh()
    assert "2 model(s)" in message
    again = ModelCatalog(catalog_for("opencode-go"), tmp_path)
    assert again.model_ids() == ["glm-5", "kimi-k2.5"]
    assert again.context_max("kimi-k2.5") == 262144
    data = json.loads(go.path.read_text())
    assert data["provider"] == "opencode-go" and data["version"] == mc.CACHE_VERSION


def test_a_failed_refresh_leaves_the_file_alone(go, sources):
    sources(endpoint=["glm-5"])
    go.refresh()
    before = go.path.read_text()
    sources(endpoint_err="offline", dev_err="offline")
    with pytest.raises(RuntimeError, match="offline"):
        go.refresh()
    assert go.path.read_text() == before


def test_context_defaults_to_the_model_maximum(go, sources):
    sources(endpoint=["kimi-k2.5"])
    go.refresh()
    assert go.context_for("kimi-k2.5") == 262144


def test_a_chosen_context_is_stored_and_survives_a_refresh(go, sources, tmp_path):
    sources(endpoint=["kimi-k2.5", "glm-5"])
    go.refresh()
    go.set_context("kimi-k2.5", 131072)
    fresh = ModelCatalog(catalog_for("opencode-go"), tmp_path)
    fresh.refresh()
    assert fresh.context_for("kimi-k2.5") == 131072
    assert json.loads(go.path.read_text())["context_overrides"] == {
        "kimi-k2.5": 131072}


def test_choosing_the_maximum_forgets_the_override(go, sources):
    """'Max' stays live: a window that grows is used at its new size."""
    sources(endpoint=["kimi-k2.5"])
    go.refresh()
    go.set_context("kimi-k2.5", 131072)
    go.set_context("kimi-k2.5", 262144)
    assert "kimi-k2.5" not in go.overrides


def test_an_override_never_exceeds_a_shrunk_window(go, sources):
    sources(endpoint=["kimi-k2.5"])
    go.refresh()
    go.set_context("kimi-k2.5", 200000)
    shrunk = {"opencode-go": {"models": {
        "kimi-k2.5": {"limit": {"context": 128000}}}}}
    sources(endpoint=["kimi-k2.5"], dev=shrunk)
    go.refresh()
    assert go.context_for("kimi-k2.5") == 128000


def test_an_old_file_keeps_its_overrides(tmp_path):
    path = tmp_path / "opencode-go.json"
    path.write_text(json.dumps({"version": 0, "models": {"x": {}},
                                "context_overrides": {"x": 8192}}))
    cat = ModelCatalog(catalog_for("opencode-go"), tmp_path)
    assert cat.models == {} and cat.overrides == {"x": 8192}


def test_a_corrupt_file_is_an_empty_catalogue(tmp_path):
    (tmp_path / "opencode.json").write_text("{not json")
    cat = ModelCatalog(catalog_for("opencode"), tmp_path)
    assert cat.models == {}


def test_a_catalogue_row_round_trips():
    row = CatalogModel(id="m", name="M", context_max=1000, output_max=10,
                       api="responses", tool_call=False, cost_input=1.0,
                       cost_output=2.0)
    assert CatalogModel.from_json("m", row.to_json()) == row


# ── the generic node knows them too ──────────────────────────────────────

@pytest.mark.parametrize("key, url", [
    ("opencode", "https://opencode.ai/zen/v1"),
    ("opencode-go", "https://opencode.ai/zen/go/v1"),
])
def test_opencode_is_a_preset_on_the_generic_endpoint(key, url):
    assert provider_for(key).base_url == url
    assert catalog_for(key).base_url == url
