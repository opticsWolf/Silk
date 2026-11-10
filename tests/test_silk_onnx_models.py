# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

ONNX models two ways: a specification okf loads, or a model Silk hosts.

**A specification says what it is.** Silk KG Embedder (ONNX) and Silk KG
Reranker (ONNX) sit with the Knowledge Graph and put ``embedder_spec`` /
``reranker_spec`` on the wire; scenes saved under the old port names
keep their wires.

**A loader hands out a real model, outside Weave's process.** The ONNX
Embedder / Reranker Loader's handle is a lazy pool over an embroider
host process: ``embedder_for`` embeds through it, the Knowledge Graph
reranks through it on okf's external lane, and nothing starts until it
is used.

**The host speaks the GGUF servers' shapes.** ``/v1/embeddings``,
``/v1/rerank``, ``/tokenize``, ``/health``, ``/v1/models``.

Offline except the last test, which runs the real host when the small
English cross-encoder is in the Hugging Face cache.
"""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from silk.functions import embroider_server
from silk.functions.embeddings import ServerEmbedder, embedder_for
from silk.functions.embroider_pool import EmbroiderPool
from silk.functions.model_pool import LazyGGUFPool
from silk.nodes.knowledge_graph import SilkKnowledgeGraphNode
from silk.nodes.silk_ports import RERANKER_MODEL_TYPE


@pytest.fixture()
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class FakePool:
    """What consumers read off an EmbroiderPool, nothing started."""

    def __init__(self, model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 base_url="http://127.0.0.1:8955"):
        self.model_path, self.base_url = model_id, base_url


# -- A2: the specification nodes ----------------------------------------

def test_the_spec_nodes_say_what_they_are(qt_app):
    from silk.nodes.knowledge_graph import SilkKnowledgeGraphNode as KG
    from silk.nodes.onnx_loader import SilkOnnxEmbedderNode, SilkOnnxRerankerNode
    assert SilkOnnxEmbedderNode.node_name == "Silk KG Embedder (ONNX)"
    assert SilkOnnxRerankerNode.node_name == "Silk KG Reranker (ONNX)"
    for cls in (SilkOnnxEmbedderNode, SilkOnnxRerankerNode, KG):
        assert cls.node_subclass == "Knowledge"
    for cls in (SilkOnnxEmbedderNode, SilkOnnxRerankerNode):
        assert "loader" not in cls.node_tags
    embedder, reranker = SilkOnnxEmbedderNode(), SilkOnnxRerankerNode()
    assert [p.name for p in embedder.outputs] == ["embedder_spec"]
    assert [p.name for p in reranker.outputs] == ["reranker_spec"]
    assert embedder.outputs[0].datatype == "onnx_embedder", "the type is unchanged"


def test_old_scenes_keep_their_spec_wires(qt_app):
    from weave.canvas.serializer import _named_port
    from silk.nodes.onnx_loader import SilkOnnxEmbedderNode, SilkOnnxRerankerNode
    old = {"outputs": [{"name": "onnx_embedder", "datatype": "onnx_embedder"}],
           "widget_data": {}}
    state = SilkOnnxEmbedderNode.migrate_state(old, 2)
    assert [p["name"] for p in state["outputs"]] == ["embedder_spec"]
    first = SilkOnnxEmbedderNode.migrate_state(
        {"outputs": [{"name": "onnx_embedder"}], "widget_data": {}}, 1)
    assert first["outputs"][0]["name"] == "embedder_spec"
    assert first["widget_data"]["image_model_id"] == "auto", "api 1's step still runs"
    rerank = SilkOnnxRerankerNode.migrate_state(
        {"outputs": [{"name": "reranker", "datatype": "reranker_model"}]}, 1)
    assert rerank["outputs"][0]["name"] == "reranker_spec"

    assert _named_port(SilkOnnxEmbedderNode(), "outputs", "onnx_embedder").name == "embedder_spec"
    assert _named_port(SilkOnnxRerankerNode(), "outputs", "reranker").name == "reranker_spec"


# -- L1: what the Knowledge Graph and embedder_for make of a loader -----

def onnx_reranker(pool=None):
    return {"role": "rerank", "runtime": "onnx", "pool": pool or FakePool(),
            "model_id": "onnx/cross-encoder/ms-marco-MiniLM-L-6-v2",
            "family": "ms-marco-minilm-l-6-v2", "max_tokens": 512}


def test_an_onnx_reranker_loader_reranks_on_okfs_external_lane():
    assert RERANKER_MODEL_TYPE.validator(onnx_reranker())
    assert not RERANKER_MODEL_TYPE.validator(dict(onnx_reranker(), pool=None))
    spec = SilkKnowledgeGraphNode.spec_from({"db_path": "kb.db", "reranker": onnx_reranker()})
    assert (spec.rerank_backend, spec.rerank_endpoint_url) == (
        "external", "http://127.0.0.1:8955")
    env = spec.launch_env()
    assert env["OKFGRAPH_RERANK_FAMILY"] == "ms-marco-minilm-l-6-v2"
    assert env["OKFGRAPH_RERANK_MAX_TOKENS"] == "512"
    assert env["OKFGRAPH_RERANK_COUNT"] == "server-tokenize"


def test_the_graph_refuses_an_onnx_loader_as_its_embedder():
    handle = {"backend": "onnx", "role": "embedding", "pool": FakePool()}
    problem = SilkKnowledgeGraphNode.wiring_problem(
        {"db_path": "kb.db", "embed_source": "gguf", "embedder": handle})
    assert "Silk KG Embedder (ONNX)" in problem


def test_embedder_for_embeds_through_the_onnx_host():
    pool = FakePool("google/embeddinggemma-2", "http://127.0.0.1:8956/v1")
    embedder = embedder_for({"backend": "onnx", "role": "embedding", "pool": pool})
    assert isinstance(embedder, ServerEmbedder)
    assert embedder.base_url == "http://127.0.0.1:8956/v1"


def test_the_pool_gives_each_consumer_the_base_it_expects():
    def made(kind):
        pool = object.__new__(EmbroiderPool)
        pool.kind, pool._url, pool._model_path = kind, "http://127.0.0.1:9", "m/x"
        return pool
    assert made("embed").base_url == "http://127.0.0.1:9/v1", "embedder_for adds /embeddings"
    assert made("rerank").base_url == "http://127.0.0.1:9", "okf adds /v1/rerank"
    assert EmbroiderPool.serves_tokenize
    with pytest.raises(ValueError):
        EmbroiderPool("m/x", kind="decide")


# -- L1: the loader node --------------------------------------------------

@pytest.fixture()
def fake_host(monkeypatch):
    from silk.nodes import onnx_model_loader as loader
    started = []

    class Pool:
        serves_tokenize = True

        def __init__(self, model_id, **settings):
            started.append((model_id, settings))
            self.model_path, self.base_url = model_id, "http://127.0.0.1:9/v1"

        def snapshot(self):
            return {"model_path": self.model_path, "url": "http://127.0.0.1:9",
                    "dim": 256, "precision": "fp32", "cuda": False}

        def cleanup(self, timeout=None):
            started.append("stopped")

    monkeypatch.setattr(loader, "EmbroiderPool", Pool)
    monkeypatch.setattr(loader, "embroider_available", lambda: True)
    return started


def test_the_loader_as_embedder_hands_out_a_model_that_starts_on_use(qt_app, fake_host):
    from silk.nodes.onnx_model_loader import OnnxLoaderNode
    node = OnnxLoaderNode()
    out = node.compute({"model_id": "google/embeddinggemma-2", "dim": "256"})
    handle = out["model_obj"]
    assert out["reranker"] is None, "the role fills one output"
    assert (handle["backend"], handle["role"]) == ("onnx", "embedding")
    assert fake_host == [], "nothing runs until it is used"
    embedder = embedder_for(handle)
    assert fake_host[0] == ("google/embeddinggemma-2", {
        "kind": "embed", "device": "cpu", "precision": "auto", "cache_dir": "", "dim": 256})
    assert embedder.base_url == "http://127.0.0.1:9/v1"

    same = node.compute({"model_id": "google/embeddinggemma-2", "dim": "256"})
    assert same["model_obj"]["pool"] is handle["pool"], "unchanged settings keep it"
    node.compute({"model_id": "google/embeddinggemma-2", "dim": "native"})
    assert fake_host[-1] == "stopped", "new settings: the old process stops"
    with pytest.raises(ValueError, match="ships no int8"):
        node.compute({"model_id": "google/embeddinggemma-2", "precision": "int8"})


def test_the_loader_offers_the_models_widths(qt_app, fake_host):
    from silk.nodes.onnx_model_loader import NATIVE, OnnxLoaderNode
    node = OnnxLoaderNode()
    node._combo_model.setCurrentText("jinaai/jina-embeddings-v5-text-small-retrieval")
    widths = [node._combo_dim.itemText(i) for i in range(node._combo_dim.count())]
    assert widths[0] == NATIVE and widths[-1] == "1024"
    node._combo_dim.setCurrentText("1024")
    node._combo_model.setCurrentText("google/embeddinggemma-2")
    assert node._combo_dim.currentText() == NATIVE, "gemma stops at 768"


def test_the_role_row_swaps_the_models_and_keeps_a_typed_one(qt_app, fake_host):
    from silk.functions.okf_graph import KNOWN_MODELS, KNOWN_RERANK_MODELS
    from silk.nodes.onnx_model_loader import OnnxLoaderNode
    node = OnnxLoaderNode()
    node._combo_role.setCurrentIndex(1)
    models = [node._combo_model.itemText(i) for i in range(node._combo_model.count())]
    assert models == list(KNOWN_RERANK_MODELS)
    assert node._combo_model.currentText() == KNOWN_RERANK_MODELS[0]
    assert not node._combo_dim.isEnabled(), "a reranker has no width"
    node._combo_model.setCurrentText("my-org/my-cross-encoder")
    node._combo_role.setCurrentIndex(0)
    assert node._combo_model.currentText() == "my-org/my-cross-encoder"
    node._combo_role.setCurrentIndex(1)
    node._combo_role.setCurrentIndex(0)
    node._combo_model.setCurrentText(KNOWN_RERANK_MODELS[0])
    node._combo_role.setCurrentIndex(1)
    node._combo_role.setCurrentIndex(0)
    assert node._combo_model.currentText() == KNOWN_MODELS[0], \
        "a model of the other role is swapped for this role's first"


def test_the_loader_as_reranker_feeds_the_graph_and_unloads(qt_app, fake_host):
    from silk.nodes.onnx_model_loader import OnnxLoaderNode
    node = OnnxLoaderNode()
    out = node.compute({"role": "reranker", "precision": "q8",
                        "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2"})
    reranker = out["reranker"]
    assert out["model_obj"] is None, "the role fills one output"
    assert RERANKER_MODEL_TYPE.validator(reranker)
    assert (reranker["runtime"], reranker["max_tokens"]) == ("onnx", 512)
    spec = SilkKnowledgeGraphNode.spec_from({"db_path": "kb.db", "reranker": reranker})
    assert spec.rerank_backend == "external" and fake_host[0][1]["kind"] == "rerank"
    assert fake_host[0][1]["precision"] == "q8" and "dim" not in fake_host[0][1]
    assert node._is_loaded()
    node._on_unload_clicked()
    assert not node._is_loaded() and fake_host[-1] == "stopped"


def test_the_loader_survives_the_canvas(qt_app, fake_host):
    """A widget named ``_width`` once shadowed the node's own width and
    took the app down on the first resize."""
    from silk.nodes.onnx_model_loader import OnnxLoaderNode
    node = OnnxLoaderNode()
    assert isinstance(node._width, (int, float))
    node.auto_resize()
    node.cleanup()


def test_the_loader_without_embroider_says_so(qt_app, monkeypatch):
    from silk.nodes import onnx_model_loader as loader
    monkeypatch.setattr(loader, "embroider_available", lambda: False)
    node = loader.OnnxLoaderNode()
    errors = []
    node.compute_error.connect(errors.append)
    assert node.compute({}) == {"model_obj": None, "reranker": None}
    assert errors and "embroider is not installed" in errors[0]


# -- L1: the host's HTTP --------------------------------------------------

class FakeModel:
    """Two words, two tokens; vectors say which task made them."""

    max_length, precision, used_cuda, dim = 512, "fp32", False, 3

    def count_tokens(self, text):
        return len(text.split())

    def encode_batch(self, texts, task="Document"):
        return [[float(len(t)), 1.0 if task == "Query" else 0.0, 0.0] for t in texts]

    def score_batch(self, pairs):
        return [float(len(set(q.split()) & set(d.split()))) for q, d in pairs]


def serve(kind):
    host = object.__new__(embroider_server.Host)
    host.kind, host.model_id, host.model = kind, "m/x", FakeModel()
    host._lock = threading.Lock()
    server = ThreadingHTTPServer(("127.0.0.1", 0), embroider_server.make_handler(host))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def call(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=5) as r:
        return json.loads(r.read())


def test_the_host_embeds_in_the_openai_shape_with_tasks():
    server, root = serve("embed")
    try:
        assert call(root + "/health") == {"status": "ok"}
        assert call(root + "/v1/models")["data"][0]["dim"] == 3
        out = call(root + "/v1/embeddings", {"input": ["a b", "c"]})
        assert [row["index"] for row in out["data"]] == [0, 1]
        assert out["data"][0]["embedding"][1] == 0.0, "Document by default"
        assert out["usage"]["prompt_tokens"] == 3
        query = call(root + "/v1/embeddings", {"input": "a", "input_type": "query"})
        assert query["data"][0]["embedding"][1] == 1.0
        assert len(call(root + "/tokenize", {"content": "one two three"})["tokens"]) == 3
        with pytest.raises(urllib.error.HTTPError) as no_route:
            call(root + "/v1/rerank", {"query": "q", "documents": []})
        assert no_route.value.code == 404, "an embedder does not rerank"
        with pytest.raises(urllib.error.HTTPError) as bad:
            call(root + "/v1/embeddings", {"input": 5})
        assert bad.value.code == 400
    finally:
        server.shutdown()


def test_the_host_reranks_best_first():
    server, root = serve("rerank")
    try:
        out = call(root + "/v1/rerank", {"query": "a cat", "top_n": 1,
                                         "documents": ["a dog", {"text": "a cat sat"}]})
        assert out["results"] == [{"index": 1, "relevance_score": 2.0}]
    finally:
        server.shutdown()


_SMALL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.mark.skipif(
    not (Path.home() / ".cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2").is_dir(),
    reason="the small cross-encoder is not cached")
def test_the_real_host_reranks_in_its_own_process():
    pool = LazyGGUFPool(_SMALL, pool_class=EmbroiderPool, kind="rerank", device="cpu")
    try:
        out = call(pool.base_url + "/v1/rerank", {
            "query": "what is a cat",
            "documents": ["Paris is in France", "A cat is a small feline"]})
        assert out["results"][0]["index"] == 1
        assert pool.snapshot()["max_length"] == 512
    finally:
        pool.cleanup()
    assert not pool.started
