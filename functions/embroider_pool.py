# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

An ONNX model (embroider) served by a child process.

The ONNX twin of the GGUF pools: ``functions/embroider_server.py`` run as
a script on a free loopback port, waited on until it answers, released
through Weave's shutdown registry (:class:`~functions.rerank_pool.ChildServerPool`
does all of that). Wrapped in a :class:`~functions.model_pool.LazyGGUFPool`
it starts on first use, which is how the ONNX loaders hand it out.

Nothing here imports embroider: ONNX Runtime must never load in Weave's
process, so whether embroider is installed is asked of the import
system, and the model's width is asked of the running server.

``base_url`` is what each model's consumers expect of its GGUF twin: an
embedder's is the OpenAI ``.../v1`` base (``functions.embeddings``
appends ``/embeddings``), a reranker's the host root (okf's external
rerank lane appends ``/v1/rerank`` and ``/tokenize``).
"""
from __future__ import annotations

import importlib.util
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict

from .rerank_pool import ChildServerPool

#: The script the child runs, as a file: it needs embroider, not Silk.
SERVER_SCRIPT = str(Path(__file__).with_name("embroider_server.py"))

#: The kinds the server hosts.
KINDS = ("embed", "rerank")


def embroider_available() -> bool:
    """Whether embroider is installed, without importing it here."""
    return importlib.util.find_spec("embroider") is not None


class EmbroiderPool(ChildServerPool):
    """One running embroider model server."""

    #: Its /tokenize counts exactly (see embroider_server).
    serves_tokenize = True

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
        args = ["--kind", kind, "--model", model_id, "--device", device,
                "--precision", precision or "auto"]
        if cache_dir:
            args += ["--cache_dir", cache_dir]
        if dim:
            args += ["--dim", str(int(dim))]
        label = "ONNX embedder server" if kind == "embed" else "ONNX reranker server"
        super().__init__(model_id, SERVER_SCRIPT, args, label,
                         log_prefix="silk-onnx-")
        self._info = self._read_info()

    @property
    def base_url(self) -> str:
        return f"{self._url}/v1" if self.kind == "embed" else self._url

    @property
    def model_alias(self) -> str:
        return self._model_path

    @property
    def info(self) -> Dict[str, Any]:
        """What the server says of its model: ``dim`` (an embedder's
        width), ``max_length``, ``precision``, ``cuda``."""
        return dict(self._info)

    def snapshot(self) -> dict:
        out = super().snapshot()
        out["model_path"] = self._model_path
        out.update({k: v for k, v in self._info.items()
                    if k in ("dim", "max_length", "precision", "cuda")})
        return out

    def _read_info(self) -> Dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self._url}/v1/models", timeout=5.0) as r:
                rows = json.loads(r.read().decode("utf-8")).get("data") or []
            return dict(rows[0]) if rows else {}
        except Exception:  # noqa: BLE001 - a running model answers anyway
            return {}
