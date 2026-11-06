# -*- coding: utf-8 -*-
"""The Endpoint/Model split and the reworked Model Endpoint.

The settings file is pinned in ``test_silk_endpoint_settings``; this
pins the seam a person actually touches -- that switching providers
fills and remembers URLs, that the dropdown follows the listing, that
the spinner and slider divide the window between them, and that an
Endpoint node's dict drives a Model node.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from silk.functions import endpoint_settings as es


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "ENDPOINT_DIR", tmp_path)
    return tmp_path


LISTING = {
    "anthropic/claude-opus-4-6": {
        "context_length": 200000,
        "supported_parameters": ["tools", "temperature"],
    },
    "x-ai/grok-4": {
        "context_length": 200000,
        "supported_parameters": ["temperature"],
    },
}


@pytest.fixture
def listing(monkeypatch):
    def fake_list(base_url, headers=None, timeout=None):
        return dict(LISTING)
    monkeypatch.setattr(es, "list_model_specs", fake_list)


@pytest.fixture
def sync_threads(monkeypatch):
    """Refresh threads run inline, so the refresh path is deterministic."""
    import silk.nodes.model_picker as picker

    class SyncThread:
        def __init__(self, target, args=(), kwargs=None, daemon=None,
                     name=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(
        picker, "threading", SimpleNamespace(Thread=SyncThread))


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def seed(store_dir, provider, **over):
    payload = {
        "version": 1, "provider": provider, "base_url": "",
        "credential": "", "fetched_at": 0.0, "source": "",
        "models": {}, "max_overrides": {}, "context_overrides": {},
        "selected": "",
    }
    payload.update(over)
    (store_dir / f"{provider}.json").write_text(json.dumps(payload))
    return payload


def make_endpoint_node(store_dir):
    from silk.nodes.model_endpoint import SilkModelEndpointNode
    return SilkModelEndpointNode()


def make_model_node(store_dir):
    from silk.nodes.endpoint import SilkModelNode
    return SilkModelNode()


@pytest.fixture
def endpoint_node(app, store_dir, listing, sync_threads):
    return make_endpoint_node(store_dir)


@pytest.fixture
def model_node(app, store_dir, listing, sync_threads):
    return make_model_node(store_dir)


@pytest.fixture
def endpoint_only_node(app, store_dir, listing, sync_threads):
    from silk.nodes.endpoint import SilkEndpointNode
    return SilkEndpointNode()


# ── the provider rows remember ───────────────────────────────────────────

def test_a_fresh_provider_fills_the_preset(endpoint_node):
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("openrouter"))
    assert endpoint_node._base_url.text() == "https://openrouter.ai/api/v1"
    assert endpoint_node._credential.text() == "OPENROUTER_API_KEY"


def test_a_typed_url_survives_a_round_trip(endpoint_node, store_dir):
    endpoint_node._base_url.setText("http://my-proxy:4000/v1")
    endpoint_node._persist_provider_edits()
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("openrouter"))
    assert endpoint_node._base_url.text() == "https://openrouter.ai/api/v1"
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("custom"))
    assert endpoint_node._base_url.text() == "http://my-proxy:4000/v1"


def test_the_credential_name_is_per_provider(endpoint_node):
    endpoint_node._credential.setText("MY_PROXY_KEY")
    endpoint_node._persist_provider_edits()
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("openrouter"))
    assert endpoint_node._credential.text() == "OPENROUTER_API_KEY"
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("custom"))
    assert endpoint_node._credential.text() == "MY_PROXY_KEY"


# ── the dropdown follows the listing ─────────────────────────────────────

def test_switching_providers_lists_and_restores_the_pick(
        endpoint_node, store_dir):
    seed(store_dir, "openrouter",
         base_url="https://openrouter.ai/api/v1",
         credential="OPENROUTER_API_KEY",
         models={m: {"context_max": 200000} for m in LISTING},
         selected="x-ai/grok-4")
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("openrouter"))
    items = [endpoint_node._model.itemText(i)
             for i in range(endpoint_node._model.count())]
    assert items == ["anthropic/claude-opus-4-6", "x-ai/grok-4"]
    assert endpoint_node._model.currentText() == "x-ai/grok-4"
    endpoint_node._model.setCurrentText("anthropic/claude-opus-4-6")
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("custom"))
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("openrouter"))
    assert endpoint_node._model.currentText() == "anthropic/claude-opus-4-6"


def test_a_saved_model_the_listing_dropped_is_still_usable(
        endpoint_node, store_dir):
    from silk.nodes.model_picker import UNKNOWN_CONTEXT_MAX
    seed(store_dir, "openrouter", selected="retired/model",
         models={"retired/model": {"context_max": 32000}})
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("openrouter"))
    assert endpoint_node._model.currentText() == "retired/model"
    assert endpoint_node._context.maximum() == UNKNOWN_CONTEXT_MAX, \
        "dropped from the listing: the window is unknown, and said so"


# ── the spinner and the slider divide the window ─────────────────────────

def test_the_max_and_slider_follow_the_detected_window(endpoint_node):
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("openrouter"))
    endpoint_node._model.setCurrentText("anthropic/claude-opus-4-6")
    assert endpoint_node._max.value() == 200000
    assert endpoint_node._context.maximum() == 200000
    assert endpoint_node._context.value() == 200000


def test_a_manual_max_is_kept_per_model(endpoint_node, store_dir):
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("openrouter"))
    endpoint_node._model.setCurrentText("x-ai/grok-4")
    endpoint_node._max.setValue(100000)
    endpoint_node._persist_choices()
    data = json.loads((store_dir / "openrouter.json").read_text())
    assert data["max_overrides"] == {"x-ai/grok-4": 100000}
    endpoint_node._model.setCurrentText("anthropic/claude-opus-4-6")
    endpoint_node._model.setCurrentText("x-ai/grok-4")
    assert endpoint_node._max.value() == 100000
    assert endpoint_node._context.maximum() == 100000


def test_a_reduced_slider_is_kept_per_model(endpoint_node, store_dir):
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("openrouter"))
    endpoint_node._model.setCurrentText("anthropic/claude-opus-4-6")
    endpoint_node._context.setValue(50000)
    endpoint_node._persist_choices()
    data = json.loads((store_dir / "openrouter.json").read_text())
    assert data["context_overrides"] == {"anthropic/claude-opus-4-6": 50000}
    endpoint_node._model.setCurrentText("x-ai/grok-4")
    endpoint_node._model.setCurrentText("anthropic/claude-opus-4-6")
    assert endpoint_node._context.value() == 50000


def test_picking_a_tool_calling_model_ticks_the_box(endpoint_node):
    endpoint_node._provider.setCurrentIndex(
        endpoint_node._provider.findData("openrouter"))
    endpoint_node._model.setCurrentText("x-ai/grok-4")
    assert not endpoint_node._native_tools.isChecked()
    endpoint_node._model.setCurrentText("anthropic/claude-opus-4-6")
    assert endpoint_node._native_tools.isChecked()
    endpoint_node._model.setCurrentText("x-ai/grok-4")
    assert not endpoint_node._native_tools.isChecked(), \
        "a known no-tools answer unticks again"


# ── the connect ──────────────────────────────────────────────────────────

@pytest.fixture
def served(monkeypatch):
    import silk.nodes.model_endpoint as node_mod

    def fake_connect(base_url, model="", **kw):
        handle = {"backend": "openai", "model": object(),
                  "model_alias": model or "m", "base_url": base_url,
                  "provider": kw.get("provider", "custom")}
        if kw.get("context_length"):
            handle["context_length"] = kw["context_length"]
        return handle, f"Connected: {model} @ {base_url}", ["m"]

    monkeypatch.setattr(node_mod, "connect", fake_connect)


def test_the_handle_carries_the_effective_window(endpoint_node, served):
    out = endpoint_node.compute({
        "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-opus-4-6", "context_max": 200000,
        "context_length": 50000, "credential": "", "supports_tools": False,
    })
    assert out["model_obj"]["context_length"] == 50000


def test_a_wired_length_above_the_max_is_clamped(endpoint_node, served):
    out = endpoint_node.compute({
        "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
        "model": "m", "context_max": 10000, "context_length": 999999,
        "credential": "", "supports_tools": False,
    })
    assert out["model_obj"]["context_length"] == 10000


def test_an_untick_means_fences_even_where_the_listing_says_tools(
        endpoint_node, monkeypatch):
    import silk.functions.model_endpoint as me_funcs

    def advertised(base_url, headers=None, timeout=None):
        return {"m": {"context_length": 200000,
                       "supported_parameters": ["tools"]}}
    monkeypatch.setattr(me_funcs, "list_model_specs", advertised)
    ticked = endpoint_node.compute({
        "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
        "model": "m", "context_max": 0, "context_length": 0,
        "credential": "", "supports_tools": True,
    })
    assert ticked["model_obj"]["supports_tools"] is True
    unticked = endpoint_node.compute({
        "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
        "model": "m", "context_max": 0, "context_length": 0,
        "credential": "", "supports_tools": False,
    })
    assert "supports_tools" not in unticked["model_obj"], \
        "the person's untick stands, even against the listing"
    assert "native tools" not in endpoint_node._sync_status


def test_a_model_node_untick_means_fences_too(model_node, monkeypatch):
    import silk.functions.model_endpoint as me_funcs

    def advertised(base_url, headers=None, timeout=None):
        return {"m": {"supported_parameters": ["tools"]}}
    monkeypatch.setattr(me_funcs, "list_model_specs", advertised)
    out = model_node.compute({
        "endpoint": {"provider": "openrouter",
                     "base_url": "https://openrouter.ai/api/v1",
                     "credential": ""},
        "model": "m", "context_max": 0, "context_length": 0,
        "supports_tools": False,
    })
    assert "supports_tools" not in out["model_obj"]


def test_migration_keeps_the_transport_the_old_graph_ran_on(
        app, store_dir, listing, sync_threads):
    from silk.nodes.model_endpoint import SilkModelEndpointNode
    seed(store_dir, "openrouter",
         models={"m-tools": {"context_max": 200000, "tool_call": True}})
    ran_native = {"widget_data": {"provider": "openrouter",
                                    "model": "m-tools",
                                    "supports_tools": False}}
    out = SilkModelEndpointNode.migrate_state(ran_native, 1)
    assert out["widget_data"]["supports_tools"] is True
    unknown = {"widget_data": {"provider": "openrouter",
                                "model": "never-listed",
                                "supports_tools": False}}
    out = SilkModelEndpointNode.migrate_state(unknown, 1)
    assert out["widget_data"]["supports_tools"] is False, \
        "no answer left to restore: the saved value stands"
    fences = {"widget_data": {"provider": "openrouter",
                               "model": "m-tools",
                               "supports_tools": True}}
    out = SilkModelEndpointNode.migrate_state(fences, 1)
    assert out["widget_data"]["supports_tools"] is True


# ── the Endpoint node ────────────────────────────────────────────────────

def test_the_endpoint_node_forwards_provider_url_and_key(
        endpoint_only_node):
    endpoint_only_node._provider.setCurrentIndex(
        endpoint_only_node._provider.findData("openrouter"))
    out = endpoint_only_node.compute({
        "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
        "credential": "OPENROUTER_API_KEY",
    })
    assert out["endpoint"] == {
        "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
        "credential": "OPENROUTER_API_KEY",
    }


def test_an_endpoint_without_a_url_is_empty_not_an_error(endpoint_only_node):
    out = endpoint_only_node.compute(
        {"provider": "custom", "base_url": "", "credential": ""})
    assert out["endpoint"] is None


# ── the Model node ───────────────────────────────────────────────────────

def test_a_model_without_an_endpoint_says_so(model_node):
    out = model_node.compute({"endpoint": None, "model": "m"})
    assert out["model_obj"] is None
    assert "Wire an Endpoint" in model_node._sync_status


def test_a_model_connects_through_the_wired_endpoint(model_node, monkeypatch):
    import silk.nodes.endpoint as node_mod

    seen = {}

    def fake_connect(base_url, model="", **kw):
        seen.update(base_url=base_url, credential=kw.get("credential"),
                    provider=kw.get("provider"))
        return ({"backend": "openai", "model": object(),
                 "model_alias": model, "base_url": base_url,
                 "provider": kw.get("provider", "?")},
                "Connected", ["m"])

    monkeypatch.setattr(node_mod, "connect", fake_connect)
    out = model_node.compute({
        "endpoint": {"provider": "openrouter",
                     "base_url": "https://openrouter.ai/api/v1",
                     "credential": "OPENROUTER_API_KEY"},
        "model": "m", "context_max": 0, "context_length": 0,
        "supports_tools": False,
    })
    assert out["model_obj"]["provider"] == "openrouter"
    assert seen == {"base_url": "https://openrouter.ai/api/v1",
                    "credential": "OPENROUTER_API_KEY",
                    "provider": "openrouter"}
    model_node.on_evaluate_finished()
    assert model_node._picker.get("key") == "openrouter"
