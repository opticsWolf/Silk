# -*- coding: utf-8 -*-
"""Tests for the /v1/systemone client (decision models)."""

import io
import json
import urllib.error

import pytest

from silk.functions.systemone import (
    SystemOneClient,
    hosted_decision_handle,
    validate_questions,
)


def test_hosted_handle_shape_and_pinning():
    handle = hosted_decision_handle(
        "jev/jev-1.13.0", "https://api.typesafe.ai/v1/systemone/", "JEV_API_KEY")
    assert handle == {
        "backend": "systemone", "role": "decision",
        "model": "jev/jev-1.13.0",
        "url": "https://api.typesafe.ai/v1/systemone",
        "credential": "JEV_API_KEY", "egress": True,
    }
    with pytest.raises(ValueError, match="pin a model version"):
        hosted_decision_handle("jev/jev-latest", "https://x/", "k")
    with pytest.raises(ValueError, match="model id"):
        hosted_decision_handle("", "https://x/", "k")


VALID = {
    "route": {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": "payments", "shipping": None}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["can wait", "today"]},
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
}


def test_validate_accepts_all_three_types():
    validate_questions(VALID)


@pytest.mark.parametrize("questions", [
    {},
    {"q": {"type": "unknown", "instructions": "x"}},
    {"q": {"type": "noul"}},
    {"q": {"type": "choice", "instructions": "x"}},
    {"q": {"type": "choice", "instructions": "x", "criteria": {}}},
    {"q": {"type": "score", "instructions": "x",
            "criteria": ["only one"]}},
    {"q": "not-a-dict"},
])
def test_validate_rejects_what_the_server_would_400(questions):
    with pytest.raises(ValueError):
        validate_questions(questions)


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch_urlopen(monkeypatch, payload=None, error=None):
    import silk.functions.systemone as module

    def fake(request, timeout=None):
        if error is not None:
            raise error
        return _FakeResponse(payload)

    monkeypatch.setattr(module.urllib.request, "urlopen", fake)


def test_decide_returns_answers(monkeypatch):
    _patch_urlopen(monkeypatch, {
        "answers": {"angry": {"type": "noul", "noul": 0.9}},
        "usage": {"input_tokens": 12, "output_tokens": 0},
    })
    answers = SystemOneClient("http://127.0.0.1:8080/v1").decide(
        "hi", {"angry": VALID["angry"]})
    assert answers["angry"]["noul"] == 0.9


def test_decide_validates_before_sending(monkeypatch):
    called = []
    import silk.functions.systemone as module

    def fake(request, timeout=None):
        called.append(request)
        return _FakeResponse({"answers": {}})

    monkeypatch.setattr(module.urllib.request, "urlopen", fake)
    with pytest.raises(ValueError):
        SystemOneClient("http://127.0.0.1:8080/v1").decide("hi", {})
    assert called == []


def _http_error(code, payload):
    return urllib.error.HTTPError(
        "http://x/systemone", code, "reason", {},
        io.BytesIO(json.dumps(payload).encode("utf-8")))


def test_decide_400_is_value_error(monkeypatch):
    _patch_urlopen(monkeypatch,
                   error=_http_error(400, {"error": "bad questions"}))
    with pytest.raises(ValueError, match="bad questions"):
        SystemOneClient("http://x/v1").decide(
            "hi", {"angry": VALID["angry"]})


def test_decide_501_names_decision_models(monkeypatch):
    _patch_urlopen(monkeypatch,
                   error=_http_error(501, {"error": "unsupported"}))
    with pytest.raises(RuntimeError, match="decision GGUF"):
        SystemOneClient("http://x/v1").decide(
            "hi", {"angry": VALID["angry"]})
