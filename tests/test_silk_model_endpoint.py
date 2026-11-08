# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

A model that runs somewhere else (D45, the remote half).

The loop only ever calls ``create_chat_completion``, so a hosted or
separately-run backend is the same graph with the subprocess removed.
What is pinned here is the part that is not automatic: that the endpoint
is *asked* before a handle is handed out, that the refusals say what to
change, and that a credential stays a name -- the value must not be
reachable from the handle, which is what a saved graph is built from.
"""

from __future__ import annotations

import io
import urllib.error

import pytest

from silk.functions import model_endpoint as me
from silk.functions import model_pool as mp
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
    """Make the endpoint advertise a given model list.

    ``specs`` carries the rest of a ``/models`` entry -- pricing, context
    length, supported parameters -- for the tests that care what `connect`
    reads out of it.
    """
    def _install(models, err=None, specs=None):
        def fake(base_url, headers=None, timeout=None):
            if err is not None:
                raise RuntimeError(err)
            return {name: dict((specs or {}).get(name, {})) for name in models}
        monkeypatch.setattr(me, "list_model_specs", fake)
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


# ── the listing request itself ───────────────────────────────────────────

def test_the_listing_names_silk_even_when_the_caller_passes_only_a_key(monkeypatch):
    """The catalogue refresh builds its own headers: just Authorization.

    Without a User-Agent urllib sends `Python-urllib/3.x`, which OpenCode's
    Cloudflare front bans with 403 "error code: 1010" -- with or without a
    valid key.
    """
    sent = {}

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        sent.update({k.lower(): v for k, v in request.header_items()})
        return _Response(b'{"data": [{"id": "m"}]}')

    monkeypatch.setattr(me.urllib.request, "urlopen", fake_urlopen)
    assert me.list_model_specs("https://gw/v1", {"Authorization": "Bearer k"}) == {
        "m": {"id": "m"}}
    assert sent["user-agent"] == me.USER_AGENT
    assert sent["authorization"] == "Bearer k"


def test_a_cloudflare_1010_is_not_blamed_on_the_key(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 403, "Forbidden", {},
            io.BytesIO(b"error code: 1010"))

    monkeypatch.setattr(me.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError) as caught:
        me.list_model_specs("https://gw/v1")
    assert "1010" in str(caught.value)
    assert "refused the credentials" not in str(caught.value), (
        "the key was never looked at; sending someone to rotate it is wrong"
    )


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


# ── a gateway that routes by conversation ────────────────────────────────

def _capture_chat(monkeypatch):
    """Answer one streamed chat request and keep the headers it carried."""
    sent = {}

    def fake_urlopen(request, timeout=None):
        sent.update({k.lower(): v for k, v in request.header_items()})
        return io.BytesIO(b'data: {"choices": [{"delta": {"content": "ok"}, '
                          b'"finish_reason": "stop"}]}\n\ndata: [DONE]\n\n')

    monkeypatch.setattr(mp.urllib.request, "urlopen", fake_urlopen)
    return sent


@pytest.mark.parametrize("url, provider", [
    ("https://opencode.ai/zen/go/v1", "opencode-go"),
    ("https://opencode.ai/zen/v1", "opencode"),
    # Pasted into the generic node: the host is enough.
    ("https://opencode.ai/zen/go/v1", "custom"),
])
def test_opencode_gets_the_engine_session_on_every_request(
        serves, monkeypatch, url, provider):
    """Go refuses chat without it ("MissingSessionID"); it routes and caches
    by conversation, so it is the agent's session key, stable across turns."""
    serves(["m"])
    handle, _, _ = connect(url, "m", provider=provider)
    sent = _capture_chat(monkeypatch)

    engine = GraphEngine(handle, session_id="conv-1")
    engine.append_message("user", "hi")
    assert "".join(engine.stream_response({})) == "ok"

    assert sent["x-opencode-session"] == "conv-1"


def test_other_endpoints_are_not_sent_a_session_header(serves, monkeypatch):
    serves(["m"])
    handle, _, _ = connect("https://openrouter.ai/api/v1", "m",
                           provider="openrouter")
    sent = _capture_chat(monkeypatch)

    engine = GraphEngine(handle, session_id="conv-1")
    engine.append_message("user", "hi")
    list(engine.stream_response({}))

    assert "x-opencode-session" not in sent
    assert "session_id" not in sent, "and nothing leaks into the payload path"


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


def test_the_port_accepts_its_own_default():
    """An unwired handle must not be invalid by its own type.

    ``default()`` returns ``{}`` and an unconnected port delivers
    ``None``; a validator that refused either would call every
    disconnected instance of the type malformed.
    """
    from weave.node.port_registry import PortRegistry

    import silk.nodes.silk_ports  # noqa: F401

    port = PortRegistry._by_name["model_handle"]
    assert port.validator(port.default_factory())
    assert port.validator(None)


def test_the_classifier_model_port_tells_decision_from_chat():
    from weave.node.port_registry import PortRegistry

    import silk.nodes.silk_ports  # noqa: F401

    port = PortRegistry._by_name["classifier_model"]
    assert port.validator({"backend": "systemone", "role": "decision",
                           "pool": object()})
    assert port.validator({"backend": "systemone", "role": "decision",
                           "model": "jev/jev-1.13.0",
                           "url": "https://example.invalid/v1"})
    assert not port.validator({"backend": "systemone",
                               "role": "decision"}), "neither pool nor url"
    assert not port.validator({"backend": "openai", "model": object()}), \
        "a chat handle is not a classifier model"
    assert not port.validator({"backend": "systemone", "pool": object()}), \
        "no role, no deal"
    assert port.validator(port.default_factory())
    assert port.validator(None)


def test_the_endpoint_dict_carries_names_never_values():
    from silk.functions.model_endpoint import endpoint_dict, split_endpoint
    handle = endpoint_dict("OpenRouter", "https://openrouter.ai/api/v1/",
                           "OPENROUTER_API_KEY")
    assert handle == {"provider": "openrouter",
                      "base_url": "https://openrouter.ai/api/v1/",
                      "credential": "OPENROUTER_API_KEY"}
    assert split_endpoint(handle) == handle
    assert split_endpoint({"provider": "x"}) is None, "no URL, no endpoint"
    assert split_endpoint(None) is None
    assert split_endpoint({}) is None


def test_the_endpoint_port_accepts_what_the_endpoint_node_emits():
    from weave.node.port_registry import PortRegistry

    import silk.nodes.silk_ports  # noqa: F401

    port = PortRegistry._by_name["endpoint"]
    assert port.validator({"provider": "openrouter",
                           "base_url": "https://openrouter.ai/api/v1"})
    assert not port.validator({"provider": "openrouter"}), "no URL"
    assert not port.validator({"base_url": "http://x/v1"}), "no provider"
    assert port.validator(port.default_factory())
    assert port.validator(None)
    label = port.formatter({"provider": "openrouter",
                            "base_url": "https://openrouter.ai/api/v1"})
    assert "openrouter" in label and "openrouter.ai" in label


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


@pytest.mark.parametrize("asked, expected", [
    (None, True),    # nobody answered: the gateway's word
    (False, False),  # a person's "no" stands against it
    (True, True),
])
def test_the_gateway_answers_only_when_nobody_else_did(serves, asked, expected):
    serves(["m"], specs={"m": {"supported_parameters": ["tools", "seed"]}})
    handle, _, _ = connect("http://host/v1", "m", supports_tools=asked)
    assert bool(handle.get("supports_tools")) is expected
