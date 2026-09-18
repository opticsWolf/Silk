# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Remote model endpoints -- the other half of D45.

The GGUF Loader spawns a `llama_cpp.server` and hands out clients to it.
Everything downstream of that -- the engine, the loop, the agent -- only
ever calls ``create_chat_completion``. So a hosted or separately-run
backend is the same graph with the subprocess removed, which is what D45
said and what `OpenAICompatClient` was already built for: it is the whole
client surface, constructed from a bare ``base_url``.

What was missing was a way to *say* one on the canvas. That is this
module (the decisions) plus `nodes/model_endpoint.py` (the surface).

**Why no litellm dependency.** OpenRouter, LM Studio, a llama.cpp or vLLM
server, Ollama and litellm's own proxy all speak the OpenAI chat API, so
one HTTP client reaches all of them and a Python SDK would only wrap what
`OpenAICompatClient` already does. litellm is still the answer for
providers that *don't* speak it (Anthropic, Gemini, Bedrock): run
``litellm --config ...`` and point this node at the proxy. That keeps
provider-specific translation in a process that specialises in it, and
keeps Silk's model layer one wire format wide.

**Credentials are names (D22).** A provider key is looked up at connect
time, from the environment or `~/.weave/silk/secrets.json`, and lives only
in the client's header dict. A saved graph carries the *name*, so it stays
shareable; nothing here ever returns a value, and the status text says
whether a key was found, never what it was.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from weave.logger import get_logger

from .credentials import missing_credential
from .model_pool import OpenAICompatClient

__all__ = [
    "BACKEND",
    "PROVIDERS",
    "Provider",
    "connect",
    "list_models",
    "normalise_base_url",
    "provider_for",
    "provider_keys",
]

log = get_logger("SilkModelEndpoint")

#: The handle's backend name. Not a vendor: it names the *wire format*,
#: which is the only thing the engine cares about. Everything reachable
#: here speaks it, and the provider is recorded separately for the reader.
BACKEND = "openai"

#: How long to wait on a model listing before calling the endpoint down.
#: Generous for a cold local server, short enough that a wrong URL is a
#: mistake you find out about rather than a node that sits there.
PROBE_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class Provider:
    """One preset: where it listens, and what it calls its key.

    A preset is a *default*, never a constraint -- every field stays
    editable, because the useful endpoint is often someone's own proxy on
    a port nobody can guess.
    """

    key: str
    label: str
    base_url: str
    credential: str = ""
    note: str = ""


#: The presets, local first: a local server is the case with no bill and
#: no key, so it should be the one reachable without reading anything.
PROVIDERS: Tuple[Provider, ...] = (
    Provider("custom", "Custom (any OpenAI-compatible URL)", "", "",
             "Anything that serves /v1/chat/completions -- a hosted "
             "gateway, a colleague's box, your own proxy."),
    Provider("lmstudio", "LM Studio", "http://localhost:1234/v1", "",
             "LM Studio's local server, default port 1234."),
    Provider("llamacpp", "llama.cpp server", "http://localhost:8080/v1", "",
             "A llama-server you started yourself, rather than the one "
             "the GGUF Loader spawns."),
    Provider("ollama", "Ollama", "http://localhost:11434/v1", "",
             "Ollama's OpenAI-compatible endpoint."),
    Provider("vllm", "vLLM", "http://localhost:8000/v1", "",
             "A vLLM server -- including one serving an Unsloth "
             "fine-tune, which is an ordinary model once it is served."),
    Provider("litellm", "LiteLLM proxy", "http://localhost:4000/v1",
             "LITELLM_API_KEY",
             "The proxy, not the library: it fronts providers that do "
             "not speak the OpenAI API, and speaks it to us."),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1",
             "OPENROUTER_API_KEY",
             "Hosted, many models, one key. Requests cost money, so the "
             "model name is worth reading twice."),
)

_BY_KEY: Dict[str, Provider] = {p.key: p for p in PROVIDERS}


def provider_keys() -> List[str]:
    """Preset keys in display order."""
    return [p.key for p in PROVIDERS]


def provider_for(key: str) -> Provider:
    """The preset by key; ``custom`` for anything unknown.

    Unknown is not an error: a graph saved against a later Silk may name
    a preset this one has never heard of, and the URL in the graph is
    what actually matters.
    """
    return _BY_KEY.get(str(key or "").strip().lower(), _BY_KEY["custom"])


def normalise_base_url(url: str) -> str:
    """The base URL as the client wants it: no trailing slash.

    A bare host gains ``/v1``, because every server here mounts the
    OpenAI API there and "http://localhost:1234" is what a person copies
    out of the UI. A URL that already has a path is left exactly alone --
    guessing at someone's reverse proxy is how you get a 404 that reads
    like an outage.
    """
    text = str(url or "").strip().rstrip("/")
    if not text:
        return ""
    if "://" not in text:
        text = f"http://{text}"
    tail = text.split("://", 1)[1]
    if "/" not in tail:
        text = f"{text}/v1"
    return text


def list_models(base_url: str, headers: Optional[Dict[str, str]] = None,
                timeout: float = PROBE_TIMEOUT_S) -> List[str]:
    """Model ids the endpoint advertises on ``/models``.

    Raises ``RuntimeError`` with something a person can act on. An empty
    list is a real answer -- some proxies serve chat without listing --
    and is not an error here; the caller decides what to do with it.
    """
    url = f"{normalise_base_url(base_url)}/models"
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _http_detail(exc)
        if exc.code in (401, 403):
            raise RuntimeError(
                f"{url} refused the credentials ({exc.code}). The key is "
                f"resolved from the credential *name* in this node{detail}."
            ) from exc
        raise RuntimeError(f"{url} answered {exc.code}{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not reach {url}: {exc.reason}. Is the server running, "
            f"and is the URL its OpenAI base (usually ending in /v1)?"
        ) from exc
    except (TimeoutError, OSError) as exc:
        raise RuntimeError(f"could not reach {url}: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(
            f"{url} did not answer with JSON -- this is usually a URL that "
            f"points at a web page rather than at an API ({exc})."
        ) from exc

    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    found: List[str] = []
    for row in rows:
        name = row.get("id") if isinstance(row, dict) else row
        if isinstance(name, str) and name:
            found.append(name)
    return sorted(found)


def connect(
    base_url: str,
    model: str = "",
    *,
    credential: str = "",
    context_length: int = 0,
    provider: str = "custom",
    supports_tools: bool = False,
    timeout: float = PROBE_TIMEOUT_S,
) -> Tuple[Optional[Dict[str, Any]], str, List[str]]:
    """Build a model handle for a remote endpoint.

    Returns ``(handle, status, models)``. ``handle`` is ``None`` when the
    endpoint could not be used, and ``status`` then says why in terms of
    the thing to change. Nothing raises: an unreachable endpoint is an
    ordinary state of the world, and a node that throws on it takes the
    graph evaluation with it.

    The endpoint is *asked* before the handle is handed out. Discovering
    a bad URL at connect time costs one request; discovering it inside an
    agent run costs the run, and reads like the agent failed.
    """
    url = normalise_base_url(base_url)
    if not url:
        return None, "No URL. Pick a preset, or type the endpoint's base URL.", []

    name = str(credential or "").strip()
    try:
        client = OpenAICompatClient(
            url, model_alias=str(model or "").strip() or "default",
            credential=name,
        )
    except RuntimeError:
        # The one failure that is about the user's machine, not the
        # network: a name nobody has set. Its message already says where
        # to put the value.
        return None, missing_credential(name), []

    try:
        models = list_models(url, headers=client.headers(), timeout=timeout)
    except RuntimeError as exc:
        return None, str(exc), []

    chosen = str(model or "").strip()
    if not chosen:
        if len(models) == 1:
            # A single-model server (LM Studio, a llama-server) has only
            # one answer, and making someone retype it is ceremony.
            chosen = models[0]
        elif models:
            preview = ", ".join(models[:5])
            more = f" (+{len(models) - 5} more)" if len(models) > 5 else ""
            return None, f"Pick a model. This endpoint serves: {preview}{more}", models
        else:
            return None, (
                "No model set, and this endpoint lists none -- type the "
                "model name the provider documents."
            ), []
    elif models and chosen not in models:
        # Reported, not refused: gateways route names they do not list,
        # and a proxy's alias is a legitimate thing to type.
        log.warning(
            f"Model '{chosen}' is not in {url}'s list of {len(models)}; "
            f"using it anyway -- gateways often route unlisted aliases."
        )

    client.model_alias = chosen
    handle: Dict[str, Any] = {
        "backend": BACKEND,
        "model": client,
        "model_alias": chosen,
        "base_url": url,
        "provider": provider_for(provider).key,
    }
    # Only when it is known. `GraphEngine.context_length` prefers an
    # explicit value and returns None otherwise, and None is honest:
    # compaction (D25) needs a real denominator, and a guessed one would
    # summarise too early or overflow the window.
    if int(context_length or 0) > 0:
        handle["context_length"] = int(context_length)
    # Opt-in, like the loader's: a server that does not accept a `tools`
    # field refuses the whole request, so the fence protocol -- which
    # works everywhere -- stays the default. `GraphEngine` reads this key.
    if supports_tools:
        handle["supports_tools"] = True

    return (handle,
            _status(url, chosen, name, models, context_length, supports_tools),
            models)


def _status(url: str, model: str, credential: str, models: List[str],
            context_length: int, native_tools: bool = False) -> str:
    """One line a person can check the important facts against."""
    parts = [f"Connected: {model} @ {url}"]
    if credential:
        # That it resolved is provable -- the request went through with
        # it. The value is not printed, here or anywhere.
        parts.append(f"key '{credential}' ✓")
    if models:
        parts.append(f"{len(models)} model(s) advertised")
    parts.append(
        f"context {int(context_length)}" if int(context_length or 0) > 0
        else "context unknown (compaction has no denominator)"
    )
    if native_tools:
        parts.append("native tools")
    return "  ·  ".join(parts)


def _http_detail(exc: urllib.error.HTTPError) -> str:
    """The server's own words, when it bothered to say any."""
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001 - a failed read must not mask the HTTP error
        return ""
    if not body:
        return ""
    try:
        parsed = json.loads(body)
        message = parsed.get("error", parsed)
        if isinstance(message, dict):
            message = message.get("message") or json.dumps(message)
    except ValueError:
        message = body
    text = str(message).strip().replace("\n", " ")
    return f": {text[:200]}" if text else ""
