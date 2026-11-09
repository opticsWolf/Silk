# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Models on wires into the Knowledge Graph node.

Four claims, offline (no ``okf serve``, no model loaded anywhere).

**The wires are the only model settings.** The Embedder row (onnx or
gguf) picks the wire. The Embedder Loader's GGUF handle puts okf on its
external lane against Silk's llama.cpp server, which starts when the
graph asks for it; an ONNX Embedder description keeps okf on embroider
with that model; a wired reranker reranks every search, and none wired
reranks none.

**What okf must be told about a GGUF comes from the file.** Family,
pooling and width are read from its header, never typed.

**Wiring that cannot work is refused, not guessed at.** A chat model's
handle, a model wider than okf stores, a server that will not start.

**An unplugged loader does not re-embed.** Unwired, the ONNX lane keeps
the model the graph was embedded with, and a GGUF graph waits for its
model.

The Qt cases need the Weave env, like the other node suites.
"""
import pytest

from silk.functions.gguf_meta import (
    pack_header, pack_kv_int, pack_kv_str, read_embedding_meta,
)
from silk.functions.okf_graph import DEFAULT_EMBED_MODEL, KNOWN_MODELS, OkfSpec
from silk.nodes.knowledge_graph import SilkKnowledgeGraphNode


class FakePool:
    """The two things the node reads off a GGUFModelPool (both properties,
    as on the real one)."""

    def __init__(self, path="C:/models/nomic-embed-text-v1.5.Q8_0.gguf"):
        self.model_path = str(path)

    @property
    def base_url(self):
        return "http://127.0.0.1:8931/v1"


def gguf_inputs(handle=None, **more):
    return {"db_path": "kb.db", "embed_source": "gguf",
            "embedder": handle or gguf_handle(), **more}


def gguf_handle(path=None, role="embedding"):
    pool = FakePool(path) if path else FakePool()
    return {"backend": "gguf", "pool": pool, "role": role}


def write_gguf(path, basename, width, pooling_type):
    kvs = [pack_kv_str("general.architecture", "qwen3"),
           pack_kv_str("general.basename", basename),
           pack_kv_int("qwen3.embedding_length", width),
           pack_kv_int("qwen3.pooling_type", pooling_type)]
    path.write_bytes(pack_header(0, len(kvs)) + b"".join(kvs))
    return path


ONNX = {"runtime": "embroider", "role": "embedding",
        "model_id": "google/embeddinggemma-2", "device": "cuda",
        "precision": "fp16", "cache_dir": "D:/hf",
        "image_model_id": "", "image_precision": "auto"}
RERANK = {"runtime": "embroider", "role": "rerank",
          "model_id": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
          "device": "cpu", "precision": "q8", "cache_dir": "D:/rerank"}


@pytest.fixture()
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_launch_env_spells_the_new_rows():
    assert OkfSpec(db_path="kb.db").launch_env() == {}, "defaults add nothing"
    env = OkfSpec(db_path="kb.db", backend="external",
                  endpoint_url="http://127.0.0.1:8931",
                  embed_family="nomic-v1.5", embed_pooling="cls",
                  embed_counter="gguf-meta", embed_counter_gguf="m.gguf",
                  precision="int8", cache_dir="D:/hf",
                  rerank_model_id="r/m", rerank_device="cpu",
                  rerank_precision="q8").launch_env()
    assert env["OKFGRAPH_EMBED_PREFIXES"] == "nomic-v1.5", "a built-in table"
    assert env["OKFGRAPH_EMBED_POOLING"] == "cls"
    assert env["OKFGRAPH_EMBED_COUNTER"] == "gguf-meta"
    assert env["OKFGRAPH_EMBED_COUNTER_GGUF"] == "m.gguf"
    assert env["OKFGRAPH_PRECISION"] == "int8"
    assert env["OKFGRAPH_CACHE_DIR"] == "D:/hf"
    assert env["OKFGRAPH_RERANK_MODEL_ID"] == "r/m"
    assert env["OKFGRAPH_RERANK_DEVICE"] == "cpu"
    assert env["OKFGRAPH_RERANK_PRECISION"] == "q8"

    other = OkfSpec(db_path="kb.db", backend="external", endpoint_url="u",
                    embed_family="my-model").launch_env()
    assert "OKFGRAPH_EMBED_PREFIXES" not in other, "no table: no prefixes"

    ext = OkfSpec(db_path="kb.db", rerank_backend="external",
                  rerank_model_id="bge/reranker",
                  rerank_endpoint_url="http://h:8080", rerank_family="bge",
                  rerank_device="cpu").launch_env()
    assert ext["OKFGRAPH_RERANK_BACKEND"] == "external"
    assert ext["OKFGRAPH_RERANK_ENDPOINT_URL"] == "http://h:8080"
    assert ext["OKFGRAPH_RERANK_FAMILY"] == "bge"
    assert "OKFGRAPH_RERANK_DEVICE" not in ext, "a host's device is its own"


def test_an_off_ladder_model_width_reaches_okf():
    args = OkfSpec(db_path="kb.db", embedding_dim=384).settings_args()
    assert args[-2:] == ["--embedding-dim", "384"]


def test_the_gguf_header_says_family_pooling_and_width(tmp_path):
    path = write_gguf(tmp_path / "q.gguf", "Qwen3-Embedding-0.6B", 1024, 3)
    meta = read_embedding_meta(str(path))
    assert (meta.name, meta.width, meta.pooling) == ("Qwen3-Embedding-0.6B", 1024, "last")

    spec = SilkKnowledgeGraphNode.spec_from(gguf_inputs(gguf_handle(path)))
    assert spec.backend == "external"
    assert spec.embed_family == "qwen3-embedding-0.6b"
    assert (spec.embed_pooling, spec.embedding_dim) == ("last", 1024), \
        "no ladder: the model's own width"


def test_a_nomic_gguf_gets_okf_prefixes_and_the_rows_width():
    spec = SilkKnowledgeGraphNode.spec_from(gguf_inputs(embedding_dim="256"))
    assert spec.backend == "external"
    assert spec.endpoint_url == "http://127.0.0.1:8931", "okf adds /v1 itself"
    assert spec.model_id == "gguf/nomic-embed-text-v1.5.Q8_0"
    assert spec.embed_family == "nomic-v1.5", "known from the file name"
    assert spec.embedding_dim == 256, "on nomic's ladder"
    assert spec.embed_counter == "gguf-meta"
    assert spec.embed_counter_gguf.endswith(".gguf")
    wide = SilkKnowledgeGraphNode.spec_from(gguf_inputs(embedding_dim="1024"))
    assert wide.embedding_dim == 768, "nomic stops at 768"


def test_an_onnx_embedder_keeps_okf_on_embroider_with_its_model():
    spec = SilkKnowledgeGraphNode.spec_from({"db_path": "kb.db", "onnx_embedder": ONNX})
    assert spec.backend == "embroider"
    assert (spec.model_id, spec.device, spec.precision, spec.cache_dir) == (
        "google/embeddinggemma-2", "cuda", "fp16", "D:/hf")
    assert spec.endpoint_url == "" and spec.embed_family == ""


def test_the_embedder_row_picks_between_two_wires():
    both = {"db_path": "kb.db", "embedder": gguf_handle(), "onnx_embedder": ONNX}
    onnx = SilkKnowledgeGraphNode.spec_from(dict(both, embed_source="onnx"))
    assert (onnx.backend, onnx.model_id) == ("embroider", "google/embeddinggemma-2")
    gguf = SilkKnowledgeGraphNode.spec_from(dict(both, embed_source="gguf"))
    assert gguf.backend == "external"


def test_unwired_onnx_keeps_the_model_the_graph_has():
    applied = {"backend": "embroider", "model_id": "google/embeddinggemma-2",
               "device": "cuda", "precision": "fp16", "cache_dir": "D:/hf"}
    spec = SilkKnowledgeGraphNode.spec_from({"db_path": "kb.db"}, applied)
    assert (spec.model_id, spec.device, spec.cache_dir) == (
        "google/embeddinggemma-2", "cuda", "D:/hf")
    assert SilkKnowledgeGraphNode.spec_from({"db_path": "kb.db"}) \
        == OkfSpec(db_path="kb.db", model_id=DEFAULT_EMBED_MODEL), \
        "a new node: Silk's default model"
    legacy = SilkKnowledgeGraphNode.spec_from(
        {"db_path": "kb.db"}, {"backend": "embroider", "model_id": ""})
    assert legacy.model_id == "", "a graph made on okf's default keeps it"
    source = SilkKnowledgeGraphNode.embed_source
    assert source({}) == "onnx" and source({"embed_source": "auto"}) == "onnx"
    assert source({"embedder": gguf_handle()}) == "onnx", "the row picks, not the wire"


def test_a_reranker_adds_rows_and_shares_one_cache():
    spec = SilkKnowledgeGraphNode.spec_from({
        "db_path": "kb.db", "onnx_embedder": ONNX, "reranker": RERANK})
    assert spec.rerank_model_id == RERANK["model_id"]
    assert (spec.rerank_device, spec.rerank_precision) == ("cpu", "q8")
    assert spec.cache_dir == "D:/hf", "the embedder's folder wins"

    alone = SilkKnowledgeGraphNode.spec_from({"db_path": "kb.db", "reranker": RERANK})
    assert alone.cache_dir == "D:/rerank"
    assert alone.backend == "embroider" and alone.model_id == DEFAULT_EMBED_MODEL

    ext = SilkKnowledgeGraphNode.spec_from({"db_path": "kb.db", "reranker": {
        "runtime": "external", "role": "rerank", "model_id": "bge/r",
        "endpoint_url": "http://h:8080", "family": "bge"}})
    assert (ext.rerank_backend, ext.rerank_endpoint_url, ext.rerank_family) == (
        "external", "http://h:8080", "bge")


def test_wiring_that_cannot_work_is_refused(tmp_path):
    problem = SilkKnowledgeGraphNode.wiring_problem
    assert problem({}) == ""
    assert problem({"onnx_embedder": ONNX}) == ""
    assert problem(gguf_inputs()) == "", "no family to declare"
    both = dict(gguf_inputs(), onnx_embedder=ONNX)
    assert problem(both) == "" and problem(dict(both, embed_source="onnx")) == ""
    chat = gguf_handle(role="chat")
    assert "not one" in problem(gguf_inputs(chat))
    assert problem({"embedder": chat}) == "", "ignored on the onnx lane"
    wide = write_gguf(tmp_path / "w.gguf", "Qwen3-Embedding-4B", 2560, 3)
    assert "2560-dim" in problem(gguf_inputs(gguf_handle(wide)))

    class DeadPool(FakePool):
        @property
        def base_url(self):
            raise RuntimeError("port in use")

    dead = {"backend": "gguf", "pool": DeadPool(), "role": "embedding"}
    assert problem(gguf_inputs(dead)) == "The embedder's server did not start: port in use"


def test_ports_take_descriptions_not_live_models():
    from silk.nodes.silk_ports import ONNX_EMBEDDER_TYPE, RERANKER_MODEL_TYPE
    assert ONNX_EMBEDDER_TYPE.validator(ONNX)
    assert ONNX_EMBEDDER_TYPE.validator(None)
    assert not ONNX_EMBEDDER_TYPE.validator(RERANK)
    assert not ONNX_EMBEDDER_TYPE.validator(gguf_handle())
    assert RERANKER_MODEL_TYPE.validator(RERANK)
    assert RERANKER_MODEL_TYPE.validator(
        {"runtime": "external", "role": "rerank", "endpoint_url": "http://h"})
    assert not RERANKER_MODEL_TYPE.validator(
        {"runtime": "external", "role": "rerank"}), "a host needs its URL"
    assert not RERANKER_MODEL_TYPE.validator(ONNX)


def test_loader_nodes_describe_and_never_load(qt_app):
    from silk.nodes.onnx_loader import SilkOnnxEmbedderNode, SilkOnnxRerankerNode
    out = SilkOnnxEmbedderNode().compute(
        {"model_id": "google/embeddinggemma-2", "device": "cuda",
         "precision": "fp16", "cache_dir": "D:/hf"})
    assert out == {"embedder_spec": ONNX}
    rerank = SilkOnnxRerankerNode().compute({"precision": "fp16"})["reranker_spec"]
    assert rerank["role"] == "rerank" and rerank["runtime"] == "embroider"
    assert rerank["model_id"] == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert rerank["precision"] == "auto", "fp16 is not a reranker precision"


def test_the_embedder_names_its_image_model_and_refuses_a_mismatch(qt_app):
    from silk.nodes.onnx_loader import SilkOnnxEmbedderNode
    node = SilkOnnxEmbedderNode()
    gemma = {"model_id": "google/embeddinggemma-2", "device": "cuda",
             "image_model_id": "google/embeddinggemma-2-vision",
             "image_precision": "fp16"}
    out = node.compute(gemma)["embedder_spec"]
    assert (out["image_model_id"], out["image_precision"]) == (
        "google/embeddinggemma-2-vision", "fp16")
    assert "Images: google/embeddinggemma-2-vision (named" in node._sync_status
    auto = node.compute(dict(gemma, image_model_id="auto", image_precision="int8"))
    assert auto["embedder_spec"]["image_model_id"] == "", "auto is okf's pairing"
    assert auto["embedder_spec"]["image_precision"] == "auto", "no int8 image weights"
    with pytest.raises(ValueError, match="embeds images for google/embeddinggemma-2"):
        node.compute(dict(gemma, model_id="jinaai/jina-embeddings-v5-text-nano-retrieval"))
    small = node.compute({"model_id": "jinaai/jina-embeddings-v5-text-small-retrieval"})
    assert small["embedder_spec"]["image_model_id"] == ""
    assert "searched by their captions" in node._sync_status


def test_the_embedder_offers_only_the_precisions_a_model_ships(qt_app):
    from silk.nodes.onnx_loader import SilkOnnxEmbedderNode
    node = SilkOnnxEmbedderNode()
    with pytest.raises(ValueError, match="ships no int8"):
        node.compute({"model_id": "google/embeddinggemma-2", "precision": "int8"})
    nano = "jinaai/jina-embeddings-v5-text-nano-retrieval"
    assert node.compute({"model_id": nano, "precision": "int8"})[
        "embedder_spec"]["precision"] == "int8"

    def enabled(combo):
        return [combo.itemText(i) for i in range(combo.count())
                if combo.model().item(i).isEnabled()]
    node._model.setCurrentText(nano)
    node._precision.setCurrentText("int8")
    node._image.setCurrentText("jina-v5-omni-nano-retrieval-vision")
    node._model.setCurrentText("google/embeddinggemma-2")
    assert enabled(node._precision) == ["auto", "fp32", "fp16"]
    assert enabled(node._image) == ["auto", "google/embeddinggemma-2-vision"]
    assert node._precision.currentText() == "auto", "int8 went grey"
    assert node._image.currentText() == "auto", "the jina image model went grey"


def test_image_rows_reach_okf_and_the_info():
    from silk.nodes.knowledge_graph import _image_line
    onnx = dict(ONNX, image_model_id="google/embeddinggemma-2-vision",
                image_precision="fp32")
    spec = SilkKnowledgeGraphNode.spec_from({"db_path": "kb.db", "onnx_embedder": onnx})
    assert (spec.image_model_id, spec.image_precision) == (
        "google/embeddinggemma-2-vision", "fp32")
    env = spec.launch_env()
    assert env["OKFGRAPH_IMAGE_MODEL_ID"] == "google/embeddinggemma-2-vision"
    assert env["OKFGRAPH_IMAGE_PRECISION"] == "fp32"
    plain = SilkKnowledgeGraphNode.spec_from({"db_path": "kb.db", "onnx_embedder": ONNX})
    assert not {"OKFGRAPH_IMAGE_MODEL_ID", "OKFGRAPH_IMAGE_PRECISION"} & set(
        plain.launch_env()), "okf's pairing: nothing to say"
    assert "google/embeddinggemma-2-vision" in _image_line(plain)
    assert "captions" in _image_line(OkfSpec(
        db_path="", model_id="jinaai/jina-embeddings-v5-text-small-retrieval"))
    assert "at most 768" in _image_line(OkfSpec(
        db_path="", model_id="google/embeddinggemma-2", embedding_dim=1024))


def test_the_node_has_no_model_fields(qt_app):
    node = SilkKnowledgeGraphNode()
    names = {p.name for p in node.inputs}
    assert {"embedder", "onnx_embedder", "reranker", "embed_source"} <= names
    assert not names & {"device", "model_id", "backend", "endpoint_url",
                        "embed_family", "api_key_env", "embed_pooling"}


def test_old_scenes_keep_their_lane():
    migrate = SilkKnowledgeGraphNode.migrate_state
    ext = migrate({"widget_data": {"backend": "external"}}, 4)
    assert ext["widget_data"]["embed_source"] == "gguf"
    crate = migrate({"widget_data": {"backend": "embroider"}}, 3)
    assert crate["widget_data"]["embed_source"] == "onnx"
    auto = migrate({"widget_data": {"embed_source": "auto"},
                    "applied": {"backend": "external"}}, 5)
    assert auto["widget_data"]["embed_source"] == "gguf", "the lane it served"

    from silk.nodes.onnx_loader import SilkOnnxEmbedderNode
    onnx = SilkOnnxEmbedderNode.migrate_state(
        {"widget_data": {"model_id": "google/embeddinggemma-2"}}, 1)
    assert onnx["widget_data"]["image_model_id"] == "auto", "okf's pairing, as before"
    assert onnx["widget_data"]["image_precision"] == "auto"

    from silk.nodes.embedder_loader import EmbedderLoaderNode
    loader = EmbedderLoaderNode.migrate_state({"dataflow": {"manual_mode": True}}, 1)
    assert loader["dataflow"]["manual_mode"] is False, "no Load button to wait on"


def test_the_embedder_server_starts_on_first_use(monkeypatch):
    from silk.functions import model_pool

    started = []

    class Pool:
        def __init__(self, path, **kwargs):
            started.append(path)
            self.model_path, self.base_url = path, "http://127.0.0.1:9/v1"

        def cleanup(self, timeout=None):
            started.append("stopped")

    monkeypatch.setattr(model_pool, "GGUFModelPool", Pool)
    seen = []
    lazy = model_pool.LazyGGUFPool("e.gguf", on_start=seen.append, n_ctx=2048)
    assert lazy.model_path == "e.gguf" and lazy.snapshot()["started"] is False
    assert not lazy.started and started == [], "nothing runs until it is used"
    assert lazy.base_url == "http://127.0.0.1:9/v1"
    assert lazy.started and started == ["e.gguf"] and len(seen) == 1
    assert lazy.base_url
    assert started == ["e.gguf"], "one server, however often it is asked"
    lazy.cleanup()
    assert not lazy.started and started[-1] == "stopped"


def test_a_reranker_reranks_every_search_and_none_reranks_none():
    from types import SimpleNamespace
    from silk.functions.tools.okf_tools import rerank_args
    plain = SimpleNamespace(spec=OkfSpec(db_path="kb.db"))
    wired = SimpleNamespace(spec=OkfSpec(db_path="kb.db", rerank_model_id="r/m"))
    args = {"query": "q", "rank": "none"}
    assert rerank_args(plain, args) == args
    assert rerank_args(wired, args)["rank"] == "cross"
    assert rerank_args(wired, dict(args, target="chunks")) == dict(
        args, target="chunks", rerank=True)
    assert rerank_args(wired, dict(args, rank="ppr"))["rank"] == "ppr", \
        "a graph walk keeps its order"


def test_an_unplugged_gguf_waits_instead_of_reembedding(qt_app, tmp_path):
    node = SilkKnowledgeGraphNode()
    db = str(tmp_path / "kb.db")
    node._applied = {"db_path": db, "source": "wire", "backend": "external",
                     "model_id": "gguf/nomic-embed-text-v1.5.Q8_0"}
    assert node.compute({"db_path": db, "embed_source": "gguf"}) == {"knowledge": None}
    assert node._sync_status.startswith("Waiting for the embedder")
    assert "gguf/nomic-embed-text-v1.5.Q8_0" in node._sync_status

    assert node._failed, "red: there is no graph to search"

    node._applied = {}
    node.compute({"db_path": db, "embed_source": "gguf"})
    assert node._sync_status.startswith("The Embedder row is gguf")
    assert node._failed

    node.compute({})
    assert not node._failed, "no database chosen yet is a new node, not a broken one"


class FakeGraph:
    """An okf serve that reports the pins of a graph embedded with *pinned*."""

    def __init__(self, spec, pinned="jinaai/jina-embeddings-v5-text-small-retrieval"):
        self.spec, self.running, self.owned = spec, True, True
        self._pinned, self.closed = pinned, False

    def describe(self):
        return "kb.db (fake)"

    def pins(self):
        same = self.spec.model_id == self._pinned
        return {"pinned": {"model_id": self._pinned, "embedding_dim": 512},
                "configured": {"model_id": self.spec.model_id},
                "mismatch": [] if same else ["model_id"], "concepts": 3}

    def close(self):
        self.closed = True


def test_silks_default_is_embeddinggemma_2_and_old_graphs_keep_theirs(
        qt_app, tmp_path, monkeypatch):
    from silk.nodes import knowledge_graph as kg
    assert DEFAULT_EMBED_MODEL == "google/embeddinggemma-2" == KNOWN_MODELS[0]
    opened = []

    def fake_open(spec, should_stop=None):
        opened.append(spec.model_id)
        return FakeGraph(spec)

    monkeypatch.setattr(kg.OkfGraph, "open", staticmethod(fake_open))
    monkeypatch.setattr(kg, "model_cache", lambda spec: {})
    node = SilkKnowledgeGraphNode()
    db = str(tmp_path / "kb.db")
    out = node.compute({"db_path": db})
    assert opened == [DEFAULT_EMBED_MODEL, "jinaai/jina-embeddings-v5-text-small-retrieval"]
    assert out["knowledge"].spec.model_id.startswith("jinaai/"), "kept, not re-embedded"
    assert "Kept jinaai/" in node._sync_status and not node._failed
    assert node._applied["model_id"].startswith("jinaai/")

    node.compute({"db_path": db})
    assert len(opened) == 2, "the kept model is the node's now: no second start"


def test_a_graph_that_cannot_search_turns_the_node_red(qt_app, tmp_path, monkeypatch):
    from silk.nodes import knowledge_graph as kg
    monkeypatch.setattr(kg.OkfGraph, "open", staticmethod(lambda spec, should_stop=None:
                                                          FakeGraph(spec, pinned="gguf/x")))
    monkeypatch.setattr(kg, "model_cache", lambda spec: {})
    node = SilkKnowledgeGraphNode()
    node.compute({"db_path": str(tmp_path / "kb.db")})
    assert "! Embedded with other settings" in node._sync_status and node._failed
    entered = []
    monkeypatch.setattr(node, "_enter_error_state", lambda: entered.append(True))
    node.on_evaluate_finished()
    assert entered == [True]


def test_the_info_says_whether_the_model_is_downloaded():
    from silk.nodes.knowledge_graph import _cache_text
    assert _cache_text({}) == [], "okf could not be asked: say nothing"
    here = _cache_text({"model_id": DEFAULT_EMBED_MODEL, "cached": True,
                        "repo": "onnx-community/embeddinggemma-2-ONNX",
                        "precision": "fp32", "cache_dir": "C:/hf",
                        "disk_usage_bytes": 1116762171})
    assert here == ["Model files: onnx-community/embeddinggemma-2-ONNX "
                    "(fp32, 1.04 GB) in C:/hf."]
    missing = _cache_text({"model_id": DEFAULT_EMBED_MODEL, "cached": False,
                           "repo": "r/m", "precision": "fp16", "cache_dir": "C:/hf"})
    assert missing[0].startswith("Not downloaded yet: r/m (fp16)")
    assert "offline" in missing[0]
