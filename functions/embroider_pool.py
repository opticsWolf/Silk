# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

An ONNX model (embroider) served by a child process.

The ONNX twin of the GGUF pools: ``python -m embroider.serve`` run on a
free loopback port, waited on until it answers, released through Weave's
shutdown registry (:class:`~functions.rerank_pool.ChildServerPool` does
all of that). Wrapped in a :class:`~functions.model_pool.LazyGGUFPool`
it starts on first use, which is how the ONNX loaders hand it out.

embroider owns the host: its routes, its model opening, its
``describe()``. Silk only chooses the process and the port. Nothing here
imports embroider either: ONNX Runtime must never load in Weave's
process, so whether embroider is installed is asked of the import
system, and the model's identity is asked of the running server.

``base_url`` is what each model's consumers expect of its GGUF twin: an
embedder's and a vision model's is the OpenAI ``.../v1`` base (``okf``
appends ``/embeddings`` and ``/images/embeddings``), a reranker's and a
decision model's the host root (okf appends ``/v1/rerank``, and the
decision client ``/systemone``).
"""
from __future__ import annotations

import importlib.util
import json
import urllib.request
from typing import Any, Dict

from .onnx_roles import serve_args
from .rerank_pool import ChildServerPool

#: The module the child runs: embroider's own host, not a Silk copy.
SERVER_MODULE = "embroider.serve"

#: The kinds embroider serves.
KINDS = ("embed", "vision", "rerank", "decide", "decide-laya", "decide-2b")

#: Kinds whose API mounts under ``/v1`` (the OpenAI shape).
_V1_KINDS = ("embed", "vision")

#: Kinds that answer ``POST /tokenize`` (embroider's TEXT_KINDS).
TEXT_KINDS = ("embed", "rerank", "decide", "decide-laya", "decide-2b")

#: What each kind calls itself in the log and the shutdown registry.
_LABELS = {
    "embed": "ONNX embedder server",
    "vision": "ONNX vision server",
    "rerank": "ONNX reranker server",
    "decide": "ONNX classifier server",
    "decide-laya": "ONNX Laya server",
    "decide-2b": "ONNX 2B decider server",
}


def base_url_for(kind: str, url: str) -> str:
    """The base a *kind*'s consumers expect, from the host root."""
    return f"{url}/v1" if kind in _V1_KINDS else url


def embroider_available() -> bool:
    """Whether embroider is installed, without importing it here."""
    return importlib.util.find_spec("embroider") is not None


class EmbroiderPool(ChildServerPool):
    """One running embroider model server."""

    #: Kinds that count tokens exactly (a vision model serves no text).
    serves_tokenize = True
    serves_tokenize_kinds = frozenset(TEXT_KINDS)

    def __init__(self, model_id: str, kind: str = "embed", device: str = "auto",
                 precision: str = "auto", cache_dir: str = "",
                 dim: int = 0) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
        if not embroider_available():
            raise RuntimeError(
                "embroider is not installed: install Silk's knowledge extra "
                "(okfgraph[cpu] or okfgraph[gpu]).")
        self.kind = kind
        self.serves_tokenize = kind in TEXT_KINDS
        super().__init__(
            model_id, "", serve_args(kind, model_id, device=device,
                                     precision=precision, cache_dir=cache_dir,
                                     dim=dim),
            _LABELS[kind], log_prefix="silk-onnx-", module=SERVER_MODULE)
        self._info = self._read_info()

    @property
    def base_url(self) -> str:
        return base_url_for(self.kind, self._url)

    @property
    def model_alias(self) -> str:
        return self._model_path

    @property
    def info(self) -> Dict[str, Any]:
        """What the host says of its model (embroider ``describe()``):
        ``device``, ``precision``, ``dim``, ``space``, ``max_length`` and
        the rest of the identity."""
        return dict(self._info)

    def snapshot(self) -> dict:
        out = super().snapshot()
        out["model_path"] = self._model_path
        out["kind"] = self.kind
        out.update({k: v for k, v in self._info.items()
                    if k in ("dim", "native_dim", "max_length", "precision",
                             "device", "space", "family", "text_partner")})
        out["cuda"] = str(self._info.get("device") or "") == "cuda"
        return out

    def _read_info(self) -> Dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self._url}/v1/models", timeout=5.0) as r:
                rows = json.loads(r.read().decode("utf-8")).get("data") or []
            return dict(rows[0]) if rows else {}
        except Exception:  # noqa: BLE001 - a running model answers anyway
            return {}
