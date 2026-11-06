# -*- coding: utf-8 -*-
"""Per-provider endpoint memory: what the node remembers, pinned.

The listing logic keeps no state of its own; this pins the seam the
nodes actually touch -- that a URL edit lands in the provider's file,
that last-picked models and held-back context lengths come back, and
that a refresh never clears a person's choices.
"""

from __future__ import annotations

import pytest

from silk.functions import endpoint_settings as es
from silk.functions.endpoint_settings import EndpointSettings


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "ENDPOINT_DIR", tmp_path)
    return tmp_path


def _settings(store_dir, provider="openrouter"):
    return EndpointSettings(provider, store_dir)


# ── the person's choices survive ─────────────────────────────────────────

def test_a_fresh_provider_has_nothing_stored(store_dir):
    settings = _settings(store_dir)
    assert settings.base_url == "" and settings.credential == ""
    assert settings.model_ids() == [] and settings.selected == ""
    assert settings.describe_age() == "never refreshed"


def test_url_credential_selection_and_context_round_trip(store_dir):
    settings = _settings(store_dir)
    settings.set_base_url("https://openrouter.ai/api/v1")
    settings.set_credential("OPENROUTER_API_KEY")
    settings.set_selected("anthropic/claude-opus-4-6")
    settings.set_context("anthropic/claude-opus-4-6", 100000)

    reread = _settings(store_dir)
    assert reread.base_url == "https://openrouter.ai/api/v1"
    assert reread.credential == "OPENROUTER_API_KEY"
    assert reread.selected == "anthropic/claude-opus-4-6"
    assert reread.context_for("anthropic/claude-opus-4-6") == 100000


def test_settings_are_per_provider(store_dir):
    _settings(store_dir, "openrouter").set_base_url("https://openrouter.ai/api/v1")
    _settings(store_dir, "lmstudio").set_base_url("http://localhost:1234/v1")
    assert _settings(store_dir, "openrouter").base_url == "https://openrouter.ai/api/v1"
    assert _settings(store_dir, "lmstudio").base_url == "http://localhost:1234/v1"


def test_only_names_and_urls_are_stored_never_key_values(store_dir):
    settings = _settings(store_dir)
    settings.set_credential("OPENROUTER_API_KEY")
    raw = (store_dir / "openrouter.json").read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY" in raw
    assert "sk-" not in raw


def test_a_choice_at_or_above_the_max_is_forgotten(store_dir):
    settings = _settings(store_dir)
    settings.models = {"m": es.EndpointModel("m", context_max=200000)}
    settings.set_context("m", 200000)
    settings.set_max("m", 200000)
    assert settings.overrides == {} and settings.max_overrides == {}
    assert settings.max_for("m") == 200000, "the live detected max is the default"
    settings.set_context("m", 50000)
    assert settings.context_for("m") == 50000
    reread = _settings(store_dir)
    reread.models = {"m": es.EndpointModel("m", context_max=200000)}
    assert reread.context_for("m") == 50000


def test_a_manual_max_survives_and_wins_over_the_detected(store_dir):
    settings = _settings(store_dir)
    settings.models = {"m": es.EndpointModel("m", context_max=0)}
    settings.set_max("m", 131072)
    reread = _settings(store_dir)
    assert reread.max_for("m") == 131072
    reread.models = {"m": es.EndpointModel("m", context_max=200000)}
    assert reread.max_for("m") == 131072, "the person's claim wins"
    reread.set_max("m", 200000)
    assert reread.max_overrides == {}, "equal to detected: forgotten"
    assert reread.max_for("m") == 200000


def test_a_choice_above_a_shrunk_max_is_clamped(store_dir):
    settings = _settings(store_dir)
    settings.models = {"m": es.EndpointModel("m", context_max=200000)}
    settings.set_context("m", 150000)
    settings.models = {"m": es.EndpointModel("m", context_max=100000)}
    assert settings.context_for("m") == 100000


def test_a_corrupt_file_reads_as_empty_not_an_error(store_dir):
    (store_dir / "openrouter.json").write_text("{nope", encoding="utf-8")
    assert _settings(store_dir).model_ids() == []


# ── the refresh ──────────────────────────────────────────────────────────

@pytest.fixture
def listing(monkeypatch):
    def fake_list(base_url, headers=None, timeout=None):
        assert base_url == "https://openrouter.ai/api/v1"
        return {
            "anthropic/claude-opus-4-6": {
                "context_length": 200000,
                "supported_parameters": ["tools", "temperature"],
            },
            "x-ai/grok-4": {
                "supported_parameters": ["temperature"],
            },
            "local/thing": {},
        }
    monkeypatch.setattr(es, "list_model_specs", fake_list)


def test_refresh_stores_windows_and_tool_answers(store_dir, listing):
    settings = _settings(store_dir)
    detail = settings.refresh("https://openrouter.ai/api/v1")
    assert "3 model(s)" in detail and "without an advertised context" in detail
    assert settings.model_ids() == [
        "anthropic/claude-opus-4-6", "local/thing", "x-ai/grok-4"]
    assert settings.context_max("anthropic/claude-opus-4-6") == 200000
    assert settings.tool_call("anthropic/claude-opus-4-6") is True
    assert settings.tool_call("x-ai/grok-4") is False, "a list without tools is a no"
    assert settings.tool_call("local/thing") is None, "no answer leaves the box alone"
    assert settings.describe_age() == "refreshed just now"
    reread = _settings(store_dir)
    assert reread.context_max("anthropic/claude-opus-4-6") == 200000


def test_refresh_never_clears_choices(store_dir, listing):
    settings = _settings(store_dir)
    settings.set_selected("x-ai/grok-4")
    settings.set_context("x-ai/grok-4", 50000)
    settings.set_base_url("https://openrouter.ai/api/v1")
    settings.refresh("https://openrouter.ai/api/v1")
    assert settings.selected == "x-ai/grok-4"
    assert settings.context_for("x-ai/grok-4") == 50000
    assert settings.base_url == "https://openrouter.ai/api/v1"


def test_an_empty_listing_is_a_refusal_with_advice(store_dir, monkeypatch):
    monkeypatch.setattr(es, "list_model_specs", lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="listed none"):
        _settings(store_dir).refresh("https://example.invalid/v1")
    assert not (store_dir / "openrouter.json").exists(), "the file is untouched"
