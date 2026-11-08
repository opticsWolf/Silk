# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The Reranker Loader: a GGUF reranker on Silk's own /v1/rerank server.

The server (``functions/rerank_server.py``) is exercised over HTTP with a
stand-in scorer, so no model is needed; the node and the Knowledge Graph
are exercised with a fake pool, as the embedder tests do.
"""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from silk.functions import rerank_server
from silk.functions.okf_graph import OkfSpec
from silk.nodes.knowledge_graph import SilkKnowledgeGraphNode


class FakeReranker:
    """Scores a pair by how many query words the document repeats."""

    model_id = "fake-reranker"

    def tokenize(self, text, special=False):
        return [len(word) for word in text.split()]

    def rerank(self, query, documents, top_n=None):
        words = set(query.lower().split())
        results = [{"index": i, "relevance_score": float(
            len(words & set(d.lower().split())))} for i, d in enumerate(documents)]
        results.sort(key=lambda r: r["relevance_score"], reverse=True)
        return {"model": self.model_id, "object": "list",
                "results": results[:top_n] if top_n else results, "usage": {}}


@pytest.fixture()
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                rerank_server.make_handler(FakeReranker()))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def call(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_the_server_speaks_okfs_rerank_shape(server):
    assert call(f"{server}/health") == (200, {"status": "ok"})
    status, body = call(f"{server}/v1/rerank", {
        "query": "capital of France",
        "documents": ["bananas", "Paris is the capital of France", "France"]})
    assert status == 200
    assert [r["index"] for r in body["results"]] == [1, 2, 0], "best first"
    status, body = call(f"{server}/rerank", {
        "query": "France", "documents": [{"text": "France"}, {"text": "x"}],
        "top_n": 1})
    assert [r["index"] for r in body["results"]] == [0], "{text} rows and top_n"
    assert call(f"{server}/tokenize", {"content": "a bb ccc"}) == (
        200, {"tokens": [1, 2, 3]})


def test_the_server_refuses_bad_requests(server):
    status, body = call(f"{server}/v1/rerank", {"query": "q", "documents": "x"})
    assert status == 400 and "list" in body["error"]
    assert call(f"{server}/v1/nothing", {})[0] == 404


def test_a_qwen3_reranker_without_its_template_gets_qwens_own():
    meta = {"general.architecture": "qwen3", "general.name": "Qwen3 Reranker 0.6B"}
    assert rerank_server._fallback_template(meta) == rerank_server.QWEN3_RERANK_TEMPLATE
    assert rerank_server._fallback_template(
        {"general.architecture": "qwen3", "general.name": "Qwen3 0.6B"}) == ""
    assert rerank_server._fallback_template(
        {"general.architecture": "bert", "general.name": "bge reranker"}) == ""


class FakeRerankPool:
    def __init__(self, path="C:/models/bge-reranker-v2-m3-Q8_0.gguf"):
        self.model_path = path

    @property
    def base_url(self):
        return "http://127.0.0.1:8944"


def gguf_reranker(pool=None):
    return {"role": "rerank", "runtime": "gguf", "pool": pool or FakeRerankPool(),
            "model_id": "gguf/bge-reranker-v2-m3-Q8_0", "family": "bge-reranker-v2-m3",
            "max_tokens": 2048}


def test_a_gguf_reranker_reaches_okf_on_the_external_lane():
    spec = SilkKnowledgeGraphNode.spec_from(
        {"db_path": "kb.db", "reranker": gguf_reranker()})
    assert (spec.rerank_backend, spec.rerank_endpoint_url) == (
        "external", "http://127.0.0.1:8944")
    assert spec.rerank_model_id == "gguf/bge-reranker-v2-m3-Q8_0"
    env = spec.launch_env()
    assert env["OKFGRAPH_RERANK_FAMILY"] == "bge-reranker-v2-m3"
    assert env["OKFGRAPH_RERANK_MAX_TOKENS"] == "2048"
    assert env["OKFGRAPH_RERANK_COUNT"] == "server-tokenize"


def test_the_rerank_rows_are_spelled_only_when_legal():
    env = OkfSpec(db_path="kb.db", rerank_backend="external", rerank_endpoint_url="u",
                  rerank_max_tokens=10, rerank_count="guess").launch_env()
    assert "OKFGRAPH_RERANK_MAX_TOKENS" not in env
    assert "OKFGRAPH_RERANK_COUNT" not in env
    onnx = OkfSpec(db_path="kb.db", rerank_model_id="r/m",
                   rerank_max_tokens=2048).launch_env()
    assert "OKFGRAPH_RERANK_MAX_TOKENS" not in onnx, "the external lane's row"


def test_a_reranker_server_that_will_not_start_is_reported():
    class DeadPool(FakeRerankPool):
        @property
        def base_url(self):
            raise RuntimeError("no head")

    inputs = {"db_path": "kb.db", "reranker": gguf_reranker(DeadPool())}
    assert SilkKnowledgeGraphNode.wiring_problem(inputs) == (
        "The reranker's server did not start: no head")
    assert SilkKnowledgeGraphNode.spec_from(inputs).rerank_model_id == ""


def test_the_port_takes_a_gguf_reranker():
    from silk.nodes.silk_ports import RERANKER_MODEL_TYPE
    assert RERANKER_MODEL_TYPE.validator(gguf_reranker())
    assert not RERANKER_MODEL_TYPE.validator(
        {"role": "rerank", "runtime": "gguf"}), "a gguf reranker needs its pool"


def test_the_loader_hands_out_a_lazy_server(qt_app, tmp_path, monkeypatch):
    from silk.functions import model_pool
    from silk.nodes import reranker_loader

    started = []

    class Pool(FakeRerankPool):
        def __init__(self, path, **kwargs):
            super().__init__(path)
            started.append(kwargs)

        def cleanup(self, timeout=None):
            started.append("stopped")

    monkeypatch.setattr(reranker_loader, "RerankPool", Pool)
    monkeypatch.setattr(reranker_loader, "LLAMA_CPP_AVAILABLE", True)
    model = tmp_path / "bge-reranker-v2-m3-Q8_0.gguf"
    model.write_bytes(b"not a gguf")
    node = reranker_loader.RerankerLoaderNode()
    try:
        handle = node.compute({"model_path": str(model), "device": "gpu",
                               "n_ctx": 4096, "n_threads": -1, "use_mmap": True})["reranker"]
        assert isinstance(handle["pool"], model_pool.LazyGGUFPool)
        assert started == [], "nothing runs until the Knowledge Graph asks"
        assert handle["family"] == "bge-reranker-v2-m3-q8-0", "the file name will do"
        assert handle["max_tokens"] == 4096
        spec = SilkKnowledgeGraphNode.spec_from({"db_path": "kb.db", "reranker": handle})
        assert spec.rerank_endpoint_url == "http://127.0.0.1:8944"
        assert started == [{"n_ctx": 4096, "n_gpu_layers": -1, "n_threads": None,
                            "use_mmap": True}]
        same = node.compute({"model_path": str(model), "device": "gpu",
                             "n_ctx": 4096, "n_threads": -1, "use_mmap": True})
        assert same["reranker"]["pool"] is handle["pool"], "unchanged settings keep it"
    finally:
        node.cleanup()
    assert started[-1] == "stopped"
