# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Embeddings for memory -- the vector half of §17.

``recall`` shipped as FTS5 keyword search because FTS5 needs no model and
works on day one. The half it was waiting for is here: something that
turns a turn into a vector, so the ledger's ``hybrid_search`` (FTS5 and
DiskANN fused by RRF) can answer *what did we conclude about the lexer*
when the words used then are not the words used now.

**Nothing in Silk produces embeddings by itself.** llama.cpp does, and a
GGUF handle is already a first-class thing in the graph, so an embedder is
built from one -- an *embedding* model wired to the ToolSet, not the chat
model the agent is talking to. Two reasons it is a separate wire: an
embedding model is a different (much smaller) file, and embedding through
the agent's own model would push its KV cache out from under it for no
reason.

**Failure is silence, not an error.** An embedder that cannot embed --
missing model, a server started without embedding support, a request that
fails -- disables itself after saying so once, and memory carries on as
keyword search. The alternative is a run that fails because the *optional*
half of a search index was unavailable, which nobody would want.

**The name matters.** Vectors from two models are not comparable, and
Macrame keeps one table per model name at one dimension, so an embedder is
identified by the model it came from. Change the model and the vectors go
to a new table; the old ones stay addressable and are simply not searched.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from weave.logger import get_logger

log = get_logger("SilkEmbeddings")

#: How long a single embedding request may take. Embedding is on the write
#: path of a turn, so a hung request must not hold a run open; the turn is
#: written either way and only its vector is lost.
TIMEOUT_S = 30.0


class Embedder:
    """One embedding model, addressed by name.

    Subclasses implement :meth:`_vector`. Everything else -- the dimension
    (learned from the first vector, because that is the only place the
    truth is), disabling on failure, and the empty-text case -- is shared.
    """

    def __init__(self, name: str) -> None:
        self.name = model_name(name)
        self._dim: Optional[int] = None
        self._enabled = True
        self._lock = threading.Lock()

    @property
    def dim(self) -> Optional[int]:
        """The vector width, once one vector has been produced."""
        return self._dim

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disable(self, reason: str) -> None:
        """Stop trying, once, out loud."""
        if self._enabled:
            self._enabled = False
            log.warning(
                f"Embeddings disabled for '{self.name}': {reason}. Memory "
                f"stays keyword search."
            )

    def embed(self, text: str) -> Optional[list[float]]:
        """A vector for *text*, or ``None`` -- never an exception.

        ``None`` means "no vector for this one", which every caller must
        already handle: it is what an embedder that was never wired
        returns for everything.
        """
        body = str(text or "").strip()
        if not body or not self._enabled:
            return None
        try:
            vector = self._vector(body)
        except Exception as exc:  # noqa: BLE001 - optional half, never fatal
            self.disable(f"{type(exc).__name__}: {exc}")
            return None
        if not vector:
            self.disable("the model returned an empty vector")
            return None
        with self._lock:
            if self._dim is None:
                self._dim = len(vector)
                log.info(f"Embeddings ready: '{self.name}', dim {self._dim}.")
            elif len(vector) != self._dim:
                # Two widths from one name would be a corrupt index, and
                # Macrame refuses the second one anyway (DimMismatchError).
                self.disable(f"width changed from {self._dim} to {len(vector)}")
                return None
        return vector

    def _vector(self, text: str) -> list[float]:
        raise NotImplementedError


class LlamaEmbedder(Embedder):
    """An in-process ``llama_cpp.Llama`` (or anything with its shape)."""

    def __init__(self, model: Any, name: str = "") -> None:
        super().__init__(name or _model_name(model))
        self._model = model

    def _vector(self, text: str) -> list[float]:
        create = getattr(self._model, "create_embedding", None)
        if create is not None:
            return _from_response(create(text))
        embed = getattr(self._model, "embed", None)
        if embed is None:
            raise RuntimeError("model exposes neither create_embedding nor embed")
        return _floats(embed(text))


class ServerEmbedder(Embedder):
    """A llama.cpp server behind the pool, over ``/v1/embeddings``.

    A server started without embedding support answers with an error, and
    that is the ordinary case for a chat model: the first request disables
    this embedder and memory carries on as keyword search.
    """

    def __init__(self, base_url: str, name: str = "",
                 model_alias: str = "default",
                 headers: Optional[dict] = None) -> None:
        super().__init__(name or "server")
        self.base_url = str(base_url).rstrip("/")
        self.model_alias = model_alias
        # The same headers the pool's chat client uses, so a backend that
        # needs a key (D45) is reachable for embeddings too -- taken from
        # the client rather than resolved again here, because the value
        # belongs to whoever connected (D22).
        self._headers = dict(headers or {})
        self._headers.setdefault("Content-Type", "application/json")

    def _vector(self, text: str) -> list[float]:
        payload = json.dumps(
            {"model": self.model_alias, "input": text}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings", data=payload,
            headers=dict(self._headers),
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                f"server refused embeddings: {detail or exc}"
            ) from exc
        return _from_response(body)


def embedder_for(handle: Any, name: str = "") -> Optional[Embedder]:
    """The embedder a wired model handle can serve, or ``None``.

    ``None`` is the normal answer -- nothing is wired, so ``recall`` is
    keyword search, exactly as it was. A handle is only ever *offered*
    here; whether the model behind it can actually embed is answered by
    the first request, not guessed from the file name.
    """
    if not isinstance(handle, dict) or handle.get("backend") != "gguf":
        return None
    model = handle.get("model")
    if model is not None:
        return LlamaEmbedder(model, name=name)
    pool = handle.get("pool")
    base_url = getattr(pool, "base_url", "") if pool is not None else ""
    if base_url:
        client = getattr(pool, "client", None)
        headers = client.headers() if hasattr(client, "headers") else None
        return ServerEmbedder(
            base_url, name=name or _pool_name(pool),
            model_alias=getattr(pool, "model_alias", "default") or "default",
            headers=headers,
        )
    return None


# ── helpers ─────────────────────────────────────────────────────────────

#: Macrame names an embedding table after the model and accepts
#: ``[a-z][a-z0-9_]*`` up to 48 characters, so a model file called
#: ``Qwen3-Embedding-0.6B-Q8_0.gguf`` cannot be used as-is. Normalising here
#: rather than at the call site keeps one answer to "which table do this
#: model's vectors live in" -- two spellings of one model would silently be
#: two indexes, and a search would see half its own memory.
_NAME_MAX = 48


def model_name(raw: str) -> str:
    """A Macrame-legal embedding-model name for *raw*."""
    body = "".join(
        char if (char.isascii() and (char.isalnum() or char == "_")) else "_"
        for char in str(raw or "").lower()
    ).strip("_")
    while "__" in body:
        body = body.replace("__", "_")
    if not body or not body[0].isalpha():
        body = f"e_{body}" if body else "embedding"
    return body[:_NAME_MAX].rstrip("_") or "embedding"


def _from_response(body: Any) -> list[float]:
    """The vector inside an OpenAI-shaped embedding response."""
    if isinstance(body, dict):
        data = body.get("data") or []
        if data and isinstance(data[0], dict):
            return _floats(data[0].get("embedding"))
        return _floats(body.get("embedding"))
    return _floats(body)


def _floats(value: Any) -> list[float]:
    """Flatten one vector out of whatever shape the caller produced.

    ``llama_cpp`` returns a list of per-token vectors for some pooling
    settings and a single vector for others; a list of lists is meant as
    one embedding either way, so the first row is the answer.
    """
    if value is None:
        return []
    rows = list(value)
    if rows and isinstance(rows[0], (list, tuple)):
        rows = list(rows[0])
    return [float(x) for x in rows]


def _model_name(model: Any) -> str:
    for attr in ("model_path", "name", "model"):
        value = getattr(model, attr, "")
        if isinstance(value, str) and value:
            return Path(value).stem
    return "embedding"


def _pool_name(pool: Any) -> str:
    path = getattr(pool, "model_path", "")
    return Path(path).stem if path else "embedding"
