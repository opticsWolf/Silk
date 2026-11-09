# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

An ONNX model (embroider) behind HTTP, as its own process.

ONNX never runs in Weave's process: a second ONNX Runtime beside one
already loaded takes the interpreter down. So an ONNX model a node hands
out lives here, in a child the pool can stop
(:class:`functions.embroider_pool.EmbroiderPool`), and its consumers
speak HTTP to it -- the same shapes the GGUF servers answer, so a
consumer cannot tell the runtimes apart:

``--kind embed``
    ``POST /v1/embeddings`` ``{input, task?}`` -> the OpenAI shape.
    *input* is a string or a list of them; *task* is embroider's
    (``Document``, the default, or ``Query``) -- the retrieval models
    embed a query and a passage differently. ``input_type``
    (``query`` / ``document``) is read as *task* too.
``--kind rerank``
    ``POST /v1/rerank`` ``{query, documents, top_n?}`` ->
    ``{model, results: [{index, relevance_score}], usage}`` best first,
    the shape of llama.cpp's and Jina's rerank, which okf's external
    rerank lane speaks.

Both: ``GET /health`` and ``GET /v1/models`` (readiness, and the model's
width), and ``POST /tokenize`` ``{content}`` -> ``{tokens}`` for exact
counts. embroider counts tokens but does not hand out their ids, so
``tokens`` holds one placeholder per token: its length is the count,
which is all a counter reads.

It imports nothing from Silk and its own folder leaves ``sys.path``
first, so it runs as a plain script. The ONNX Runtime library is found
the way okf finds it, when okfgraph is installed.
"""
from __future__ import annotations

import os
import sys

if sys.path and os.path.abspath(sys.path[0] or ".") == os.path.dirname(
        os.path.abspath(__file__)):
    sys.path.pop(0)

import argparse  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

#: Largest request body accepted, in bytes.
MAX_BODY = 16 * 1024 * 1024

#: ``input_type`` spellings read as embroider's tasks.
_TASKS = {"query": "Query", "search_query": "Query",
          "document": "Document", "passage": "Document",
          "search_document": "Document"}


def _locate_runtime() -> None:
    """Point ``ORT_DYLIB_PATH`` at the installed ONNX Runtime, as okf does."""
    try:
        from okfgraph.components.embedding import resolve_ort_dylib
    except Exception:  # noqa: BLE001 - embroider may find it itself
        return
    try:
        resolve_ort_dylib()
    except Exception:  # noqa: BLE001
        pass


def registry_row(model_id: str) -> Dict[str, str]:
    """embroider's registry entry for *model_id*, or ``{}``."""
    import embroider
    for row in embroider.available_models():
        if row.get("id") == model_id:
            return dict(row)
    return {}


class Host:
    """One embroider model; calls serialise (one ONNX session)."""

    def __init__(self, kind: str, model_id: str, *, device: str = "auto",
                 precision: str = "auto", cache_dir: str = "",
                 dim: int = 0) -> None:
        _locate_runtime()
        import embroider

        self.kind = kind
        self.model_id = model_id
        precision_arg = None if precision in ("", "auto") else precision
        cache = cache_dir or None
        if kind == "embed":
            native = int(registry_row(model_id).get("native_dim") or 0)
            self.model = embroider.JinaV5.open(
                model_id, cache_dir=cache, truncate_dim=dim or native or 512,
                device=device, precision=precision_arg)
        elif kind == "rerank":
            self.model = embroider.Reranker.open(
                model_id, cache_dir=cache, device=device, precision=precision_arg)
        else:
            raise ValueError(f"unknown kind {kind!r}")
        self._lock = threading.Lock()

    def describe(self) -> Dict[str, Any]:
        model = self.model
        row: Dict[str, Any] = {
            "id": self.model_id, "object": "model", "kind": self.kind,
            "max_length": int(getattr(model, "max_length", 0) or 0),
            "precision": str(getattr(model, "precision", "") or ""),
            "cuda": bool(getattr(model, "used_cuda", False)),
        }
        if self.kind == "embed":
            row["dim"] = int(getattr(model, "dim", 0) or 0)
        return row

    def count(self, text: str) -> int:
        with self._lock:
            return int(self.model.count_tokens(text))

    def embed(self, texts: List[str], task: str) -> Dict[str, Any]:
        with self._lock:
            vectors = self.model.encode_batch(texts, task=task)
            used = sum(int(self.model.count_tokens(t)) for t in texts)
        return {"object": "list", "model": self.model_id,
                "data": [{"object": "embedding", "index": i,
                          "embedding": [float(x) for x in vector]}
                         for i, vector in enumerate(vectors)],
                "usage": {"prompt_tokens": used, "total_tokens": used}}

    def rerank(self, query: str, documents: List[str],
               top_n: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            scores = self.model.score_batch([(query, d) for d in documents])
            used = sum(int(self.model.count_tokens(d)) for d in documents)
        results = [{"index": i, "relevance_score": float(s)}
                   for i, s in enumerate(scores)]
        results.sort(key=lambda row: row["relevance_score"], reverse=True)
        if top_n is not None and top_n > 0:
            results = results[:top_n]
        return {"model": self.model_id, "object": "list", "results": results,
                "usage": {"prompt_tokens": used, "total_tokens": used}}


def _texts(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("input must be a string or a list of strings")
    return list(value)


def _documents(value: Any) -> List[str]:
    """Plain strings, or ``{text}`` rows as Cohere-style clients send."""
    if not isinstance(value, list):
        raise ValueError("documents must be a list")
    return [str(item.get("text", "") if isinstance(item, dict) else item)
            for item in value]


def _task(data: Dict[str, Any]) -> str:
    task = str(data.get("task") or "")
    if task:
        return task
    return _TASKS.get(str(data.get("input_type") or "").lower(), "Document")


def make_handler(host: Host) -> type:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:   # quiet: the pool logs
            pass

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("request body too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            path = self.path.split("?", 1)[0].rstrip("/")
            if path in ("/health", ""):
                self._send(200, {"status": "ok"})
            elif path in ("/v1/models", "/models"):
                self._send(200, {"object": "list", "data": [host.describe()]})
            else:
                self._send(404, {"error": f"no route {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            try:
                data = self._body()
                if path in ("/v1/embeddings", "/embeddings") and host.kind == "embed":
                    self._send(200, host.embed(_texts(data.get("input")), _task(data)))
                elif path in ("/v1/rerank", "/rerank") and host.kind == "rerank":
                    docs = _documents(data.get("documents", data.get("texts")))
                    top_n = data.get("top_n")
                    self._send(200, host.rerank(
                        str(data.get("query") or ""), docs,
                        int(top_n) if top_n else None))
                elif path == "/tokenize":
                    count = host.count(str(data.get("content") or ""))
                    self._send(200, {"tokens": [0] * count})
                else:
                    self._send(404, {"error": f"no route {self.path} "
                                              f"on a {host.kind} model"})
            except (ValueError, TypeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - reported to the client
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", choices=("embed", "rerank"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--dim", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    host = Host(args.kind, args.model, device=args.device,
                precision=args.precision, cache_dir=args.cache_dir, dim=args.dim)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(host))
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
