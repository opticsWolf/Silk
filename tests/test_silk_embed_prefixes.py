# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

A GGUF embedder on okf's external lane, told what okf cannot guess.

Three claims, offline (no ``okf serve``, no model loaded anywhere).

**A known family embeds with its prefixes.** The table comes from the
model's name, the Embedder Loader can override it, and the graph's pin
names it: vectors made with one table are never searched with another.

**A graph embedded before that keeps working.** It is served without
prefixes, as it was made, until Re-embed is pressed.

**Tokens are counted, not estimated.** Silk's llama.cpp server answers
llama-server's ``POST /tokenize``, and the Knowledge Graph says so.
"""
import pytest

from silk.functions import embed_prefixes as ep
from silk.functions import model_pool
from silk.functions.gguf_meta import pack_header, pack_kv_int, pack_kv_str
from silk.functions.okf_graph import OkfSpec
from silk.nodes.knowledge_graph import SilkKnowledgeGraphNode


class FakePool:
    def __init__(self, path, serves_tokenize=False):
        self.model_path = str(path)
        self.serves_tokenize = serves_tokenize

    @property
    def base_url(self):
        return "http://127.0.0.1:8931/v1"


def write_gguf(path, basename="Qwen3-Embedding-0.6B", width=1024, pooling_type=3):
    kvs = [pack_kv_str("general.architecture", "qwen3"),
           pack_kv_str("general.basename", basename),
           pack_kv_int("qwen3.embedding_length", width),
           pack_kv_int("qwen3.pooling_type", pooling_type)]
    path.write_bytes(pack_header(0, len(kvs)) + b"".join(kvs))
    return path


def gguf_inputs(path, prefixes=None, serves_tokenize=False, **more):
    handle = {"backend": "gguf", "role": "embedding",
              "pool": FakePool(path, serves_tokenize)}
    if prefixes is not None:
        handle["prefixes"] = prefixes
    return {"db_path": "kb.db", "embed_source": "gguf", "embedder": handle, **more}


@pytest.fixture()
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("name, query, document", [
    ("nomic-embed-text-v1.5.Q8_0", "search_query: ", "search_document: "),
    ("multilingual-e5-large-instruct", "query: ", "passage: "),
    ("e5_base_v2", "query: ", "passage: "),
    ("bge-small-en-v1.5", ep._REPRESENT, None),
    ("mxbai-embed-large-v1", ep._REPRESENT, None),
    ("snowflake-arctic-embed-m-v2.0", "query: ", None),
    ("snowflake-arctic-embed-l", ep._REPRESENT, None),
    ("Qwen3-Embedding-0.6B", ep._QWEN, None),
    ("e5-mistral-7b-instruct", ep._QWEN + " ", None),
])
def test_known_families_get_their_tables(name, query, document):
    table = ep.prefixes_for(name)
    assert table.get("Query") == query
    assert table.get("Document") == document


def test_unknown_models_and_lookalikes_get_none():
    assert ep.prefixes_for("all-MiniLM-L6-v2") == {}
    assert ep.prefixes_for("bge-m3") == {}, "bge-m3 takes no instruction"
    assert ep.prefixes_for("embeddinggemma-300m") == {}, "its prefixes hold '|'"
    assert ep.prefixes_for("") == {}


def test_an_override_is_auto_none_or_a_typed_table():
    assert ep.resolve("auto", "bge-base-en") == ep.prefixes_for("bge-base-en")
    assert ep.resolve("", "bge-base-en") == ep.prefixes_for("bge-base-en")
    assert ep.resolve("none", "bge-base-en") == {}
    assert ep.resolve("Query=q: |Document=d: ", "bge-base-en") == {
        "Query": "q: ", "Document": "d: "}, "the prefix keeps its space"
    with pytest.raises(ValueError):
        ep.resolve("Passage=p: ", "x")
    with pytest.raises(ValueError):
        ep.resolve("query: ", "x")


def test_a_table_declares_and_tags_stably():
    table = {"Document": "d: ", "Query": "q: "}
    assert ep.declaration(table) == "Query=q: |Document=d: ", "okf's row, Query first"
    assert ep.declaration({}) == "" and ep.pin_tag({}) == ""
    assert ep.pin_tag(table) == ep.pin_tag(dict(table)) != ep.pin_tag({"Query": "q: "})
    assert len(ep.pin_tag(table)) == 6


def test_launch_env_declares_a_table_over_the_family():
    env = OkfSpec(db_path="kb.db", backend="external", endpoint_url="u",
                  embed_family="nomic-v1.5",
                  embed_prefixes="Query=q: |Document=d: ").launch_env()
    assert env["OKFGRAPH_EMBED_PREFIXES"] == "Query=q: |Document=d: "


def test_a_known_gguf_declares_its_prefixes_and_tags_the_pin(tmp_path):
    path = write_gguf(tmp_path / "q.gguf")
    spec = SilkKnowledgeGraphNode.spec_from(gguf_inputs(path))
    tag = ep.pin_tag(ep.prefixes_for("Qwen3-Embedding-0.6B"))
    assert spec.model_id == f"gguf/q+p{tag}"
    assert spec.embed_prefixes == ep.declaration(ep.prefixes_for("qwen3-embedding"))

    none = SilkKnowledgeGraphNode.spec_from(gguf_inputs(path, prefixes="none"))
    assert (none.model_id, none.embed_prefixes) == ("gguf/q", "")

    typed = SilkKnowledgeGraphNode.spec_from(gguf_inputs(path, prefixes="Query=x: "))
    assert typed.embed_prefixes == "Query=x: "
    assert typed.model_id == f"gguf/q+p{ep.pin_tag({'Query': 'x: '})}"


def test_nomic_keeps_okfs_own_table_and_its_old_pin(tmp_path):
    path = tmp_path / "nomic-embed-text-v1.5.Q8_0.gguf"
    path.write_bytes(b"")
    spec = SilkKnowledgeGraphNode.spec_from(gguf_inputs(path))
    assert spec.model_id == "gguf/nomic-embed-text-v1.5.Q8_0", "graphs made before still match"
    assert spec.embed_prefixes == "" and spec.embed_family == "nomic-v1.5"
    assert spec.launch_env()["OKFGRAPH_EMBED_PREFIXES"] == "nomic-v1.5"


def test_a_bad_override_is_refused_by_name(tmp_path):
    inputs = gguf_inputs(write_gguf(tmp_path / "q.gguf"), prefixes="Passage=p")
    problem = SilkKnowledgeGraphNode.wiring_problem(inputs)
    assert problem.startswith("The Embedder Loader's prefixes")


def test_silks_server_counts_tokens_exactly(tmp_path):
    path = write_gguf(tmp_path / "q.gguf")
    exact = SilkKnowledgeGraphNode.spec_from(gguf_inputs(path, serves_tokenize=True))
    assert exact.embed_counter == "server-tokenize"
    other = SilkKnowledgeGraphNode.spec_from(gguf_inputs(path))
    assert other.embed_counter == "gguf-meta", "another server: the file's estimate"


def test_asking_whether_it_tokenizes_starts_no_server(monkeypatch):
    started = []

    class Pool:
        serves_tokenize = True

        def __init__(self, path, **kwargs):
            started.append(path)

    lazy = model_pool.LazyGGUFPool("e.gguf", pool_class=Pool)
    assert lazy.serves_tokenize is True and started == []
    assert model_pool.LazyGGUFPool("e.gguf", pool_class=object).serves_tokenize is False
    assert model_pool.GGUFModelPool.serves_tokenize is True
    assert model_pool.SERVER_SCRIPT.endswith("llama_server.py")


def test_the_server_script_answers_llama_servers_tokenize():
    pytest.importorskip("llama_cpp.server.app")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from llama_cpp.server import app as server_app
    import silk.functions.llama_server  # noqa: F401  (adds the route)

    class Llama:
        def tokenize(self, text, add_bos=True, special=False):
            assert isinstance(text, bytes) and special
            return ([1] if add_bos else []) + [7] * len(text.split())

    app = FastAPI()
    app.include_router(server_app.router)
    app.dependency_overrides[server_app.get_llama_proxy] = lambda: (lambda model=None: Llama())
    app.dependency_overrides[server_app.authenticate] = lambda: None
    client = TestClient(app)
    assert client.post("/tokenize", json={"content": "a b c"}).json() == {"tokens": [7, 7, 7]}
    assert client.post("/tokenize", json={"content": "a", "add_special": True}
                       ).json() == {"tokens": [1, 7]}


def test_the_gguf_loaders_embedding_model_feeds_the_graph(qt_app, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from silk.nodes import gguf_loader

    class Pool:
        serves_tokenize = True

        def __init__(self, model_path, **kwargs):
            self.model_path = model_path

        base_url = "http://127.0.0.1:8931/v1"

        def snapshot(self):
            return {}

    monkeypatch.setattr(gguf_loader, "GGUFModelPool", Pool)
    monkeypatch.setattr(gguf_loader, "LLAMA_CPP_AVAILABLE", True)
    monkeypatch.setattr(gguf_loader, "LLAMA_SERVER_AVAILABLE", True)
    node = gguf_loader.GGUFLNode()
    node._model_store = SimpleNamespace(remember=lambda *a, **k: None)
    path = str(write_gguf(tmp_path / "q.gguf"))

    chat = node.compute({"model_path": path})["model_obj"]
    assert "role" not in chat, "a chat model stays off the embedder port"
    assert SilkKnowledgeGraphNode.wiring_problem(
        {"db_path": "kb.db", "embed_source": "gguf", "embedder": chat})

    handle = node.compute({"model_path": path, "embedding": True})["model_obj"]
    assert handle["role"] == "embedding"
    inputs = {"db_path": "kb.db", "embed_source": "gguf", "embedder": handle}
    assert not SilkKnowledgeGraphNode.wiring_problem(inputs)
    spec = SilkKnowledgeGraphNode.spec_from(inputs)
    assert spec.backend == "external" and spec.model_id.startswith("gguf/q+p")
    assert spec.embed_counter == "server-tokenize"


class FakeGraph:
    """An okf serve whose graph was embedded with *pinned*."""

    def __init__(self, spec, pinned):
        self.spec, self.running, self.owned = spec, True, True
        self._pinned = pinned

    def describe(self):
        return "kb.db (fake)"

    def pins(self):
        same = self.spec.model_id == self._pinned
        return {"pinned": {"model_id": self._pinned, "embedding_dim": 1024},
                "configured": {"model_id": self.spec.model_id},
                "mismatch": [] if same else ["model_id"], "concepts": 3}

    def close(self):
        pass


def test_a_graph_embedded_without_prefixes_is_served_that_way(
        qt_app, tmp_path, monkeypatch):
    from silk.nodes import knowledge_graph as kg
    opened = []

    def fake_open(spec, should_stop=None):
        opened.append(spec.model_id)
        return FakeGraph(spec, pinned="gguf/q")

    monkeypatch.setattr(kg.OkfGraph, "open", staticmethod(fake_open))
    monkeypatch.setattr(kg, "model_cache", lambda spec: {})
    node = SilkKnowledgeGraphNode()
    reembedded = []
    monkeypatch.setattr(node, "_reembed",
                        lambda graph, spec: reembedded.append(spec.model_id) or "re-embedded")
    inputs = gguf_inputs(write_gguf(tmp_path / "q.gguf"), db_path=str(tmp_path / "kb.db"))

    out = node.compute(inputs)
    assert opened[-1] == "gguf/q" and out["knowledge"].spec.embed_prefixes == ""
    assert "Embedded without the query/document prefixes" in node._sync_status
    assert not node._failed and not reembedded, "nothing re-embedded unasked"
    assert node._applied["model_id"] == "gguf/q"

    count = len(opened)
    node.compute(inputs)
    assert len(opened) == count, "held as it was: no second start"
    assert "Embedded without" in node._sync_status and not reembedded

    node._reembed_requested = True
    node.compute(inputs)
    assert reembedded and reembedded[0].startswith("gguf/q+p"), "Re-embed adds them"
