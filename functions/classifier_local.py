# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

A scorer for :class:`LocalJudgeBackend` on Silk's GGUF pool (C3, local link).

Silk's pool runs ``llama_cpp.server`` behind an OpenAI-compatible client
(``model_pool.OpenAICompatClient``). Its ``/completions`` endpoint can echo the
prompt with per-token log-probabilities, which is all a judge needs: for each
allowed answer, send ``prompt + answer``, and sum the log-probabilities of the
tokens that fall after the prompt. No sampling, nothing generated.

The judge must be a model the Agent is *not* running on: wire a second GGUF
Loader into the Classifier node's ``judge_model`` port. Self-grading by the
acting model is refused at the node (C3).
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable, Optional, Sequence


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers,
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310
        return json.loads(r.read())


def option_logprob(resp: dict, prompt_chars: int) -> float:
    """Sum of the echoed token log-probs at or after ``prompt_chars``.

    ``text_offset`` is in characters of the echoed text. The first token of a
    completion has no log-prob (``None``); it is always part of the prompt.
    """
    lp = resp["choices"][0]["logprobs"]
    total, seen = 0.0, 0
    for off, val in zip(lp["text_offset"], lp["token_logprobs"]):
        if off >= prompt_chars and val is not None:
            total += float(val)
            seen += 1
    if not seen:
        raise ValueError("the option produced no scored tokens")
    return total


def pool_scorer(client: Any, *, timeout: float = 20.0,
                post: Optional[Callable[[str, dict, dict, float], dict]] = None
                ) -> Callable[[str, Sequence[str]], list[float]]:
    """``scorer(prompt, options) -> [logprob per option]`` over a pool client."""
    send = post or _post
    url = f"{client.base_url}/completions"

    def score(prompt: str, options: Sequence[str]) -> list[float]:
        out = []
        for opt in options:
            resp = send(url, client.headers(), {
                "model": client.model_alias, "prompt": prompt + opt, "max_tokens": 1,
                "temperature": 0.0, "echo": True, "logprobs": 1}, timeout)
            out.append(option_logprob(resp, len(prompt)))
        return out

    return score


def _pool_of(handle: Any) -> Any:
    """The pool behind a ``model_handle`` port value (``{"backend": "gguf",
    "pool": GGUFModelPool}``), or the object itself."""
    if isinstance(handle, dict):
        return handle.get("pool") or handle.get("model")
    return handle


def judge_client(judge_model: Any) -> Any:
    """The OpenAI-compatible client of a ``model_handle`` port value."""
    pool = _pool_of(judge_model)
    client = getattr(pool, "client", None)
    if client is None:
        raise ValueError("the judge model has no server client (load it in a GGUF Loader)")
    return client


def refuse_self_grading(judge_model: Any, agent_model: Any) -> None:
    """Raise when the judge is the model the agent runs on (C3).

    Accepts ``model_handle`` port values (dicts) or pools. The same pool, or a
    pool serving the same model file, is the same judge.
    """
    j, a = _pool_of(judge_model), _pool_of(agent_model)
    if j is None or a is None:
        return
    jp, ap = getattr(j, "model_path", None), getattr(a, "model_path", None)
    if j is a or (jp and jp == ap):
        raise ValueError("the judge must not be the model the agent runs on")
