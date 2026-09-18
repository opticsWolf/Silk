# -*- coding: utf-8 -*-
"""A model that runs somewhere else (D45, the remote half).

The loop only ever calls ``create_chat_completion``, so a hosted or
separately-run backend is the same graph with the subprocess removed.
What is pinned here is the part that is not automatic: that the endpoint
is *asked* before a handle is handed out, that the refusals say what to
change, and that a credential stays a name -- the value must not be
reachable from the handle, which is what a saved graph is built from.
"""

from __future__ import annotations

import pytest

from silk.functions import model_endpoint as me
from silk.functions.embeddings import ServerEmbedder, embedder_for
from silk.functions.graph_engine import GraphEngine
from silk.functions.model_endpoint import (
    BACKEND, connect, normalise_base_url, provider_for,
)


# ── the URL people actually paste ────────────────────────────────────────

@pytest.mark.parametrize("given, expected", [
    ("http://localhost:1234", "http://localhost:1234/v1"),
    ("localhost:1234", "http://localhost:1234/v1"),
    ("http://localhost:1234/v1", "http://localhost:1234/v1"),
    ("http://localhost:1234/v1/", "http://localhost:1234/v1"),
    ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1"),
    ("", ""),
])
def test_a_base_url_is_normalised_not_guessed_at(given, expected):
    assert normalise_base_url(given) == expected


def test_a_path_someone_wrote_is_left_alone():
    """A reverse proxy's path is not ours to improve."""
    assert normalise_base_url("https://host/models/gw") == "https://host/models/gw"


def test_an_unknown_preset_is_custom_rather_than_an_error():
    """A graph saved against a later Silk still opens."""
    assert provider_for("some-future-gateway").key == "custom"
    assert provider_for("openrouter").credential == "OPENROUTER_API_KEY"


# ── connecting ───────────────────────────────────────────────────────────

@pytest.fixture
def serves(monkeypatch):
    """Make the endpoint advertise a given model list."""
    def _install(models, err=None):
        def fake(base_url, headers=None, timeout=None):
            if err is not None:
                raise RuntimeError(err)
            return list(models)
        monkeypatch.setattr(me, "list_models", fake)
    return _install


def test_no_url_is_an_instruction_not_a_stack_trace(serves):
    handle, status, _ = connect("")
    assert handle is None and "URL" in status


def test_a_single_model_server_needs_no_typing(serves):
    serves(["qwen3-8b"])
    handle, status, models = connect("http://localhost:1234")
    assert handle is not None
    assert handle["model_alias"] == "qwen3-8b", (
        "one model is not a choice, and making someone retype it is ceremony"
    )
    assert models == ["qwen3-8b"] and "Connected" in status


def test_a_gateway_with_many_models_asks_rather_than_picks(serves):
    serves([f"vendor/model-{n}" for n in range(9)])
    handle, status, models = connect("https://openrouter.ai/api/v1")
    assert handle is None, "picking one of nine would be picking someone's bill"
    assert "Pick a model" in status and "+4 more" in status
    assert len(models) == 9, "and the caller still gets the list to offer"


def test_an_unlisted_model_is_used_anyway(serves):
    """Gateways route aliases they do not list; refusing would be wrong."""
    serves(["a", "b"])
    handle, _, _ = connect("http://host/v1", "my-proxy-alias")
    assert handle is not None and handle["model_alias"] == "my-proxy-alias"


def test_an_endpoint_that_cannot_be_reached_says_so(serves):
    serves([], err="could not reach http://nope/v1/models: refused")
    handle, status, _ = connect("http://nope", "m")
    assert handle is None and "could not reach" in status


def test_nothing_listed_and_nothing_chosen_is_a_dead_end_with_advice(serves):
    serves([])
    handle, status, _ = connect("http://host/v1")
    assert handle is None and "type the model name" in status


# ── the handle ───────────────────────────────────────────────────────────

def test_the_handle_is_the_same_wire_the_loader_emits(serves):
    serves(["m"])
    handle, _, _ = connect("http://host/v1", "m", context_length=32768)
    assert handle["backend"] == BACKEND
    assert hasattr(handle["model"], "create_chat_completion"), (
        "the engine calls this and nothing else; that is the whole contract"
    )
    assert handle["context_length"] == 32768
    assert handle["base_url"] == "http://host/v1"


def test_an_unknown_context_window_is_absent_not_invented(serves):
    serves(["m"])
    handle, status, _ = connect("http://host/v1", "m", context_length=0)
    assert "context_length" not in handle, (
        "compaction needs a real denominator (D25); a guess summarises too "
        "early or overflows, and both look like the agent's fault"
    )
    assert "context unknown" in status


def test_a_credential_value_is_not_reachable_from_the_handle(serves, monkeypatch):
    """D22: the graph carries the name. The value lives in the headers."""
    monkeypatch.setenv("SILK_TEST_KEY", "sk-super-secret")
    serves(["m"])
    handle, status, _ = connect("http://host/v1", "m", credential="SILK_TEST_KEY")

    assert handle is not None
    assert "sk-super-secret" not in repr(handle), "not in the handle"
    assert "sk-super-secret" not in status, "and not in what the node displays"
    assert "SILK_TEST_KEY" in status and "✓" in status, (
        "the name and the fact that it resolved are exactly what helps"
    )
    assert handle["model"].headers()["Authorization"] == "Bearer sk-super-secret"


def test_a_credential_nobody_set_says_where_to_put_it(serves):
    serves(["m"])
    handle, status, _ = connect("http://host/v1", "m", credential="SILK_UNSET_KEY")
    assert handle is None
    assert "SILK_UNSET_KEY" in status and "secrets.json" in status
    assert "never in the graph" in status


# ── downstream, the difference must not exist ────────────────────────────

def test_the_engine_takes_a_remote_handle(serves):
    serves(["m"])
    handle, _, _ = connect("http://host/v1", "m", context_length=8192)
    engine = GraphEngine(handle, system_prompt="hi")
    assert engine.context_length() == 8192
    model, pool = engine._checkout()
    assert pool is None and model is handle["model"], (
        "no pool to check out of: one client, and the engine's pool-less "
        "path was already there"
    )


def test_the_engine_still_refuses_a_handle_with_no_backend():
    with pytest.raises(ValueError):
        GraphEngine({"model": object()})


def test_a_remote_handle_embeds_over_http_not_in_process(serves):
    """The client is an HTTP proxy, not a Llama: it has no create_embedding."""
    serves(["m"])
    handle, _, _ = connect("http://host/v1", "m")
    embedder = embedder_for(handle)
    assert isinstance(embedder, ServerEmbedder)


def test_the_port_accepts_either_backend():
    from weave.node.port_registry import PortRegistry

    import silk.nodes.silk_ports  # noqa: F401  (registers the type)

    port = PortRegistry._by_name["model_handle"]
    assert port.validator({"backend": "gguf", "pool": object()})
    assert port.validator({"backend": "openai", "model": object()})
    assert not port.validator({"backend": "openai"}), "a handle with no client"
    assert not port.validator({"model": object()}), "a client from nowhere"
    assert not port.validator("http://host/v1"), "a URL is not a handle"


def test_the_port_label_names_the_backend_and_the_model():
    from weave.node.port_registry import PortRegistry

    import silk.nodes.silk_ports  # noqa: F401

    port = PortRegistry._by_name["model_handle"]
    label = port.formatter({"backend": "openai", "model_alias": "vendor/m-1"})
    assert "openai" in label and "m-1" in label, (
        "which backend is the thing a reader cannot infer, and for a paid "
        "endpoint it is the thing they most need to see"
    )


# ── native tool calling ──────────────────────────────────────────────────

def test_native_tools_are_off_unless_asked_for(serves):
    """The failure modes are not symmetric, so the default is not neutral.

    A server that does not accept a `tools` field refuses the whole
    request -- the run dies. Fences merely cost some accuracy on a model
    that could have done better. Off is the direction whose worst case is
    "clumsier", not "broken", and it matches the loader's own default.
    """
    serves(["m"])
    handle, status, _ = connect("http://host/v1", "m")
    assert "supports_tools" not in handle
    assert GraphEngine(handle).supports_native_tools() is False
    assert "native tools" not in status


def test_native_tools_reach_the_engine_when_asked_for(serves):
    """The one key `select_transport` consults, from the node to the loop."""
    serves(["m"])
    handle, status, _ = connect("http://host/v1", "m", supports_tools=True)
    assert handle["supports_tools"] is True
    assert GraphEngine(handle).supports_native_tools() is True, (
        "the checkbox exists to flip exactly this gate; if the key does not "
        "arrive under this name the transport silently stays on fences"
    )
    assert "native tools" in status, "and a person can see which protocol"
