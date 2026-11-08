# -*- coding: utf-8 -*-
"""
Typed decision models over llama.cpp's ``/v1/systemone`` -- Qt-free.

Decision models (Laya, Julia-1, Kev, lev, OpenJev, Nimble, Clef, ...) do not
generate text, so ``/v1/chat/completions`` is the wrong endpoint for them.
A llama.cpp server of ``b11361`` or newer serves them at ``/v1/systemone``:

    POST {base_url}/systemone
    {"state": "I was charged twice ...",
     "questions": {
       "route":   {"type": "choice", "instructions": "...",
                   "criteria": {"billing": "payments ...", "shipping": None}},
       "urgency": {"type": "score",  "instructions": "...",
                   "criteria": ["can wait", "this week", "today"]},
       "angry":   {"type": "noul",   "instructions": "..."}}}

    -> {"answers": {
          "route":   {"type": "choice", "choice": "billing",
                      "probabilities": {...}, "confidence": 0.0-1.0},
          "urgency": {"type": "score", "score": 1.2,
                      "legend": {"0": "can wait", ...},
                      "probabilities": {...}, "confidence": 0.0-1.0},
          "angry":   {"type": "noul", "noul": 0.0-1.0}},
        "usage": {"input_tokens": N, "output_tokens": 0}}

``state`` may also be an object; the server reads it as JSON text. The
server answers ``400`` + ``{"error": ...}`` for malformed requests and
``501`` when the loaded model (or an old server) cannot serve decisions.
This module mirrors the request validation client-side so a bad question
set fails on the canvas with a plain message instead of a round trip.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

#: How long one decision request may take. A decision is a single forward
#: pass, so this is generous; a hung server must still not hold a run open.
TIMEOUT_S = 60.0

#: Question types the endpoint accepts (mirrors the server's 400 rules).
VALID_TYPES = ("choice", "score", "noul")


def hosted_decision_handle(model: str, url: str, credential: str = "") -> Dict[str, Any]:
    """A ``model_handle`` for a hosted ``/v1/systemone`` endpoint (Jev, …).

    Qt-free so nodes and tests share it. The credential travels by *name*
    (D22) -- the value is resolved at answer time and never enters the graph.
    The model must be pinned: bands are calibrated per model id.
    """
    name = str(model or "").strip()
    if not name:
        raise ValueError("a decision model id is required (e.g. jev/jev-1.13.0).")
    if name.endswith("latest"):
        raise ValueError("pin a model version: bands are calibrated per model.")
    return {
        "backend": "systemone",
        "role": "decision",
        "model": name,
        "url": str(url or "").strip().rstrip("/"),
        "credential": str(credential or "").strip(),
        "egress": True,
    }


def validate_questions(questions: Any) -> None:
    """Raise ``ValueError`` for anything the endpoint would answer 400 to.

    Mirrors ``test_systemone_invalid_request`` in llama.cpp: a non-empty
    dict, every question typed, instructed, and -- for choice/score --
    given a usable criteria set.
    """
    if not isinstance(questions, dict) or not questions:
        raise ValueError(
            "questions must be a non-empty dict of question id -> spec.")
    for qid, spec in questions.items():
        if not isinstance(spec, dict):
            raise ValueError(f"question {qid!r} must be a dict.")
        qtype = spec.get("type")
        if qtype not in VALID_TYPES:
            raise ValueError(
                f"question {qid!r} has unknown type {qtype!r} "
                f"(expected one of {', '.join(VALID_TYPES)}).")
        if not str(spec.get("instructions") or "").strip():
            raise ValueError(f"question {qid!r} needs instructions.")
        criteria = spec.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError(
                    f"choice question {qid!r} needs a non-empty "
                    f"criteria dict.")
        elif qtype == "score":
            if not isinstance(criteria, (list, tuple)) or len(criteria) < 2:
                raise ValueError(
                    f"score question {qid!r} needs at least two levels "
                    f"in criteria.")


class SystemOneClient:
    """One decision-model server behind ``llama_cpp.server``.

    Built from the pool's ``base_url`` exactly like
    :class:`functions.model_pool.OpenAICompatClient` -- no session, no
    subprocess, so a separately-run ``llama-server`` is the same object
    with a different URL.
    """

    def __init__(
        self,
        base_url: str,
        model_alias: str = "default",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.model_alias = model_alias or "default"
        self._headers = dict(headers or {})
        self._headers.setdefault("Content-Type", "application/json")

    def decide(
        self,
        state: Any,
        questions: Dict[str, Dict[str, Any]],
        timeout: float = TIMEOUT_S,
    ) -> Dict[str, Any]:
        """Ask the typed questions about *state*; returns the ``answers``.

        Raises ``ValueError`` for a question set the server would refuse,
        ``RuntimeError`` for transport failures, ``501``-style capability
        mismatches (old server, chat model loaded), and anything else the
        server reports. Never returns a partial answer.
        """
        if state is None or (isinstance(state, str) and not state.strip()):
            raise ValueError("state must be a non-empty string or object.")
        validate_questions(questions)
        payload = json.dumps(
            {"state": state, "questions": questions}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/systemone", data=payload,
            headers=dict(self._headers), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise _classify_http(exc) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"could not reach {self.base_url}/systemone: {exc.reason}. "
                f"Is the decision server running (llama.cpp b11361+)?"
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"decision request to {self.base_url} failed: {exc}") from exc
        if not isinstance(body, dict) or "answers" not in body:
            raise RuntimeError(
                f"{self.base_url}/systemone answered without 'answers'.")
        answers = body["answers"]
        if not isinstance(answers, dict):
            raise RuntimeError(
                f"{self.base_url}/systemone answered with a malformed "
                f"'answers' section.")
        return answers


class PoolSystemOneClient(SystemOneClient):
    """A :class:`SystemOneClient` on a Silk pool, which it asks for the
    address at every question rather than once.

    The Classifier Loader's pool starts its server on first use, so
    building a classifier must not touch it; and a server stopped and
    started again (Unload, then the next question) is on a new port. A
    server that will not start fails the question, like an unreachable
    one, with the reason it gave.
    """

    def __init__(self, pool: Any, headers: Optional[Dict[str, str]] = None) -> None:
        self._pool = pool
        super().__init__("", headers=headers)

    @property  # type: ignore[override]
    def base_url(self) -> str:
        try:
            return str(self._pool.base_url or "").rstrip("/")
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - one failure type for the chain
            raise RuntimeError(f"the decision server did not start: {exc}") from exc

    @base_url.setter
    def base_url(self, value: str) -> None:
        pass   # the pool's, never set

    @property  # type: ignore[override]
    def model_alias(self) -> str:
        return str(getattr(self._pool, "model_alias", "") or "default")

    @model_alias.setter
    def model_alias(self, value: str) -> None:
        pass


def _classify_http(exc: urllib.error.HTTPError) -> Exception:
    """The server's own verdict, translated to the caller's terms."""
    detail = ""
    try:
        raw = exc.read().decode("utf-8", errors="replace").strip()
        parsed = json.loads(raw) if raw else {}
        detail = parsed.get("error", raw) if isinstance(parsed, dict) else raw
        if isinstance(detail, dict):
            detail = json.dumps(detail)
        detail = str(detail).strip().replace("\n", " ")[:300]
    except Exception:  # noqa: BLE001 - a failed read must not mask the status
        pass
    suffix = f": {detail}" if detail else ""
    if exc.code == 400:
        return ValueError(f"the decision server refused the request{suffix}")
    if exc.code == 501:
        return RuntimeError(
            "this model cannot serve decisions -- load a decision GGUF "
            "(Laya, Julia-1, Kev, ...) on llama.cpp b11361 or newer"
            f"{suffix}")
    return RuntimeError(
        f"the decision server answered {exc.code}{suffix}")
