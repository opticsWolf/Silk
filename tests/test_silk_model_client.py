# -*- coding: utf-8 -*-
"""The client the pool talks through, and the key it can now carry (D45/D22).

`OpenAIClientMock` was the production HTTP client for as long as it was
called a mock, and it sent exactly one header — `Content-Type`. That was
fine while the only backend was a local server nobody authenticates to,
and it is the single thing that made a remote OpenAI-compatible backend
(litellm, vLLM, a hosted endpoint) unreachable: no way to pass a key.

What is pinned here is the rule around that key rather than the header
plumbing: a credential is a *name*, resolved once when the client
connects, held in memory and never anywhere a saved graph can reach
(D22). A name nobody has set fails loudly at connect time, because the
alternative is a 401 from the backend three layers away from its cause.
"""

from __future__ import annotations

import json

import pytest


from silk.functions import credentials
from silk.functions.embeddings import ServerEmbedder, embedder_for
from silk.functions.model_pool import OpenAICompatClient


def test_no_credential_means_the_headers_are_what_they_always_were():
    client = OpenAICompatClient("http://127.0.0.1:8080/v1")

    assert client.headers() == {"Content-Type": "application/json"}


def test_a_named_credential_is_resolved_from_the_environment(monkeypatch):
    monkeypatch.setenv("LITELLM_KEY", "sk-abc")
    client = OpenAICompatClient("https://api.example/v1", credential="LITELLM_KEY")

    assert client.headers()["Authorization"] == "Bearer sk-abc"
    # The name is what the client keeps; the value is not an attribute.
    assert client.credential == "LITELLM_KEY"
    assert "sk-abc" not in json.dumps(
        {k: v for k, v in vars(client).items() if k != "_headers"}
    )


def test_a_credential_nobody_set_fails_at_connect_time(monkeypatch, tmp_path):
    monkeypatch.delenv("LITELLM_KEY", raising=False)
    monkeypatch.setattr(credentials, "SECRETS_FILE", tmp_path / "nothing.json")

    with pytest.raises(RuntimeError) as caught:
        OpenAICompatClient("https://api.example/v1", credential="LITELLM_KEY")

    # The message says where to put it — the point of D22 is that there is
    # a right place, not that graphs are a wrong one.
    assert "LITELLM_KEY" in str(caught.value)
    assert "never in the graph" in str(caught.value)


def test_the_secrets_file_is_the_other_place_a_key_may_live(monkeypatch, tmp_path):
    monkeypatch.delenv("LITELLM_KEY", raising=False)
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"LITELLM_KEY": "sk-file"}), encoding="utf-8")
    monkeypatch.setattr(credentials, "SECRETS_FILE", secrets)

    client = OpenAICompatClient("https://api.example/v1", credential="LITELLM_KEY")

    assert client.headers()["Authorization"] == "Bearer sk-file"


def test_headers_are_a_copy_so_one_request_cannot_edit_the_next(monkeypatch):
    monkeypatch.setenv("LITELLM_KEY", "sk-abc")
    client = OpenAICompatClient("https://api.example/v1", credential="LITELLM_KEY")

    client.headers().pop("Authorization")

    assert "Authorization" in client.headers()


class _FakePool:
    """Only the surface `embedder_for` is allowed to read (see model_pool)."""

    base_url = "https://api.example/v1"
    model_alias = "alias"
    model_path = "/models/nomic-embed.gguf"

    def __init__(self, client):
        self.client = client


def test_the_embedder_carries_the_same_key_as_the_chat_client(monkeypatch):
    monkeypatch.setenv("LITELLM_KEY", "sk-abc")
    client = OpenAICompatClient("https://api.example/v1", credential="LITELLM_KEY")

    embedder = embedder_for({"backend": "gguf", "pool": _FakePool(client)})

    assert isinstance(embedder, ServerEmbedder)
    # /v1/embeddings on a backend that authenticates is the same door as
    # /v1/chat/completions; taking the headers from the client rather than
    # resolving the name again keeps one connect, one resolution.
    assert embedder._headers["Authorization"] == "Bearer sk-abc"


def test_a_pool_without_a_client_still_yields_an_embedder():
    """The pool is often a stand-in in tests, and memory is optional (§17)."""
    pool = _FakePool(client=None)

    embedder = embedder_for({"backend": "gguf", "pool": pool})

    assert isinstance(embedder, ServerEmbedder)
    assert embedder._headers == {"Content-Type": "application/json"}
