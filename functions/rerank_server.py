# -*- coding: utf-8 -*-
"""A GGUF reranker behind ``POST /v1/rerank``, as its own process.

``llama_cpp.server`` serves chat and embeddings but not rerank, and
llama.cpp's own ``llama-server --rerank`` is a native binary Silk does not
ship. llama.cpp itself can score a pair: a reranker GGUF loaded with
rank pooling (``LLAMA_POOLING_TYPE_RANK``) puts its classification head on
the graph, and the pooled output of a sequence is its score. This script
is that, plus the HTTP shape OKFgraph's external rerank lane speaks
(llama.cpp's and Jina's ``/v1/rerank``).

Run by :class:`functions.rerank_pool.RerankPool`, never imported into
Weave's process: like the chat server, the weights live in a child the
pool can stop. It imports nothing from Silk, so it runs as a plain script.

The pair is built the way ``llama-server`` builds it (``format_rerank``):
the GGUF's ``tokenizer.chat_template.rerank`` with ``{query}`` and
``{document}`` filled in when it has one (Qwen3-Reranker), else
``[BOS] query [EOS] [SEP] document [EOS]`` with each special token where
the vocabulary asks for it (bge / XLM-R rerankers). A pair longer than the
context keeps the whole query and as much of the document as fits.

A Qwen3 reranker GGUF converted as a plain language model (no rerank
template, hence no head) is scored as Qwen3 itself scores: the same
prompt, then P("yes") against "no" at the last position.

Endpoints: ``GET /health`` and ``GET /v1/models`` (readiness),
``POST /v1/rerank`` ``{query, documents, top_n?}`` ->
``{model, results: [{index, relevance_score}], usage}`` best first, and
``POST /tokenize`` ``{content}`` -> ``{tokens}`` for okf's exact counter.
Scores are the head's raw output, as llama-server returns them.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

#: The metadata key llama.cpp reads a rerank prompt template from.
RERANK_TEMPLATE_KEY = "tokenizer.chat_template.rerank"

#: Qwen3-Reranker's own prompt (its model card; llama.cpp's converter
#: writes the same into the GGUF). For a Qwen3 reranker file converted
#: without it, which otherwise scores bare text and lands near zero.
QWEN3_RERANK_TEMPLATE = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: "
    "Given a web search query, retrieve relevant passages that answer the "
    "query\n<Query>: {query}\n<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

#: Largest request body accepted, in bytes.
MAX_BODY = 8 * 1024 * 1024


class Reranker:
    """One reranker GGUF, scoring one pair at a time (calls serialise)."""

    def __init__(self, model_path: str, *, n_ctx: int = 2048,
                 n_gpu_layers: int = 0, n_threads: Optional[int] = None,
                 use_mmap: bool = True) -> None:
        import llama_cpp
        from llama_cpp import Llama

        self._llama_cpp = llama_cpp
        self.model_path = model_path
        self.model_id = Path(model_path).stem
        meta = Llama(model_path=model_path, vocab_only=True, verbose=False).metadata
        self.template = str(meta.get(RERANK_TEMPLATE_KEY) or "")
        #: A Qwen3 reranker converted as a plain language model (no rerank
        #: template, so no head either: llama.cpp's converter writes the
        #: two together) is scored the way Qwen3 scores: P("yes").
        self.causal = not self.template and bool(_fallback_template(meta))
        if self.causal:
            self.template = _fallback_template(meta)
        self.llm = Llama(
            model_path=model_path, embedding=not self.causal,
            pooling_type=(llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED if self.causal
                          else llama_cpp.LLAMA_POOLING_TYPE_RANK),
            n_ctx=n_ctx, n_batch=n_ctx, n_ubatch=n_ctx,
            n_gpu_layers=n_gpu_layers, n_threads=n_threads,
            use_mmap=use_mmap, verbose=False,
        )
        self.n_ctx = self.llm.n_ctx()
        if self.causal:
            self._yes = self.tokenize("yes")[0]
            self._no = self.tokenize("no")[0]
        vocab = llama_cpp.llama_model_get_vocab(self.llm._model.model)
        eos = llama_cpp.llama_vocab_eos(vocab)
        if eos < 0:
            eos = llama_cpp.llama_vocab_sep(vocab)
        self._bos = (llama_cpp.llama_vocab_bos(vocab)
                     if llama_cpp.llama_vocab_get_add_bos(vocab) else None)
        self._eos = eos if llama_cpp.llama_vocab_get_add_eos(vocab) else None
        self._sep = (llama_cpp.llama_vocab_sep(vocab)
                     if llama_cpp.llama_vocab_get_add_sep(vocab) else None)
        self._lock = threading.Lock()

    def tokenize(self, text: str, special: bool = False) -> List[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=False,
                                 special=special)

    def pair_tokens(self, query: str, document: str) -> List[int]:
        """The scored sequence for one pair, cut to the context."""
        if self.template:
            head, _, tail = self.template.partition("{document}")
            head = head.replace("{query}", query)
            tail = tail.replace("{query}", query)
            pre = self.tokenize(head, special=True)
            post = self.tokenize(tail, special=True)
            doc = self.tokenize(document)
            room = max(0, self.n_ctx - len(pre) - len(post))
            return pre + doc[:room] + post
        pre: List[int] = [] if self._bos is None else [self._bos]
        pre += self.tokenize(query)
        if self._eos is not None:
            pre.append(self._eos)
        if self._sep is not None:
            pre.append(self._sep)
        post = [] if self._eos is None else [self._eos]
        doc = self.tokenize(document)
        room = max(0, self.n_ctx - len(pre) - len(post))
        return pre + doc[:room] + post

    def score(self, tokens: List[int]) -> float:
        llama_cpp = self._llama_cpp
        with self._lock:
            if self.causal:
                self.llm.reset()
                self.llm.eval(tokens)
                # eval keeps no logits unless logits_all; the context's
                # last row is the one the answer is read from.
                logits = llama_cpp.llama_get_logits_ith(self.llm._ctx.ctx, -1)
                margin = float(logits[self._yes]) - float(logits[self._no])
                return 1.0 / (1.0 + math.exp(-margin))
            batch = self.llm._batch
            batch.reset()
            batch.add_sequence(tokens, 0, True)
            self.llm._ctx.kv_cache_clear()
            self.llm._ctx.decode(batch)
            batch.reset()
            return float(llama_cpp.llama_get_embeddings_seq(self.llm._ctx.ctx, 0)[0])

    def rerank(self, query: str, documents: List[str],
               top_n: Optional[int] = None) -> Dict[str, Any]:
        results, used = [], 0
        for index, document in enumerate(documents):
            tokens = self.pair_tokens(query, document)
            used += len(tokens)
            results.append({"index": index, "relevance_score": self.score(tokens)})
        results.sort(key=lambda row: row["relevance_score"], reverse=True)
        if top_n is not None and top_n > 0:
            results = results[:top_n]
        return {"model": self.model_id, "object": "list", "results": results,
                "usage": {"prompt_tokens": used, "total_tokens": used}}


def _fallback_template(metadata: Dict[str, Any]) -> str:
    """The prompt a reranker GGUF should have carried, when it is known."""
    arch = str(metadata.get("general.architecture") or "")
    name = " ".join(str(metadata.get(k) or "") for k in
                    ("general.name", "general.basename")).lower()
    if arch == "qwen3" and "rerank" in name:
        return QWEN3_RERANK_TEMPLATE
    return ""


def _documents(value: Any) -> List[str]:
    """Plain strings, or ``{text}`` rows as Cohere-style clients send."""
    if not isinstance(value, list):
        raise ValueError("documents must be a list")
    out = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("text", "")
        out.append(str(item))
    return out


def make_handler(reranker: Reranker) -> type:
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
                self._send(200, {"object": "list", "data": [
                    {"id": reranker.model_id, "object": "model"}]})
            else:
                self._send(404, {"error": f"no route {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            try:
                data = self._body()
                if path in ("/v1/rerank", "/rerank"):
                    query = str(data.get("query") or "")
                    docs = _documents(data.get("documents", data.get("texts")))
                    top_n = data.get("top_n")
                    self._send(200, reranker.rerank(
                        query, docs, int(top_n) if top_n else None))
                elif path == "/tokenize":
                    text = str(data.get("content") or "")
                    self._send(200, {"tokens": reranker.tokenize(text)})
                else:
                    self._send(404, {"error": f"no route {self.path}"})
            except (ValueError, TypeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - reported to the client
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--n_gpu_layers", type=int, default=0)
    parser.add_argument("--n_threads", type=int, default=0)
    parser.add_argument("--no_mmap", action="store_true")
    args = parser.parse_args(argv)
    reranker = Reranker(args.model, n_ctx=args.n_ctx,
                        n_gpu_layers=args.n_gpu_layers,
                        n_threads=args.n_threads or None,
                        use_mmap=not args.no_mmap)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(reranker))
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
