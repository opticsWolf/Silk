# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

S2, Qt-free: the Knowledge Graph's wires as okf sees them (plan section
3.3). The node only builds widgets; ``functions/okf_wiring.py`` is what
it tells ``okf serve``, and every row of the table is checked here.

- ``embedder`` <- ONNX Loader -> ``backend=external``, identity ``auto``:
  the host names its family, width, precision and space, so Silk declares
  no family, no prefixes and no counter.
- ``embedder`` <- Embedder Loader (GGUF) -> ``backend=external`` with the
  declared family, pooling, prefixes and ``/tokenize`` counts.
- ``embedder`` unwired -> okf's in-process embroider on the graph's pin.
- ``image_model`` <- ONNX Loader (vision) -> ``image_backend=external``,
  and the host's text partner must be the graph's text model.
"""
from __future__ import annotations

from silk.functions.okf_wiring import (
    EMBED_KEYS, _image_line, _model_lines, embed_source, spec_from,
    wiring_problem,
)
from silk.functions.onnx_roles import handle_for

TEXT = "google/embeddinggemma-2"
VISION = "google/embeddinggemma-2-vision"


class Pool:
    """What the KG reads off a loader's lazy pool (nothing starts)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8931/v1",
                 model_path: str = "D:/models/nomic-embed-text-v1.5.Q8_0.gguf",
                 serves_tokenize: bool = False) -> None:
        self.base_url = base_url
        self.model_path = model_path
        self.serves_tokenize = serves_tokenize


def onnx_embedder(**settings) -> dict:
    pool = Pool()
    opts = {"device": "cpu", "precision": "fp32", "dim": 512, **settings}
    return handle_for("embedding", TEXT, pool, opts)


def vision_handle(**settings) -> dict:
    pool = Pool()
    opts = {"precision": "auto", "dim": 512, **settings}
    return handle_for("vision", VISION, pool, opts)


RRF = {"db_path": "kb.db", "embed_source": "gguf"}


def test_the_embedder_row_still_picks_the_onx_lane():
    assert embed_source({}) == "onnx"
    assert embed_source({"embed_source": "auto"}) == "onnx"
    assert embed_source({"embedder": onnx_embedder()}) == "onnx", \
        "the row picks, not the wire"
    assert embed_source({"embed_source": "gguf"}) == "gguf"


def test_an_onnx_host_is_okfs_external_lane_with_identity_auto():
    spec = spec_from({**RRF, "embedder": onnx_embedder()})
    assert (spec.backend, spec.endpoint_url) == (
        "external", "http://127.0.0.1:8931/v1")
    assert spec.model_id == TEXT
    assert spec.embed_family == "", "the host names its own family"
    assert spec.embed_prefixes == "", "the host applies its own prompts"
    assert spec.embed_counter == "", "the host counts"
    assert spec.precision == "auto" and spec.cache_dir == ""
    env = spec.launch_env()
    assert env["OKFGRAPH_BACKEND"] == "external"
    assert env["OKFGRAPH_ENDPOINT_URL"] == "http://127.0.0.1:8931/v1"
    assert "OKFGRAPH_EMBED_FAMILY" not in env
    assert "OKFGRAPH_EMBED_PREFIXES" not in env
    assert "OKFGRAPH_EMBED_COUNTER" not in env


def test_a_gguf_embedder_keeps_its_declared_contract():
    handle = {"backend": "gguf", "role": "embedding",
              "pool": Pool("http://127.0.0.1:9000/v1", serves_tokenize=True)}
    spec = spec_from({**RRF, "embedder": handle})
    assert spec.backend == "external"
    assert spec.endpoint_url == "http://127.0.0.1:9000"
    assert spec.model_id.startswith("gguf/nomic-embed-text-v1.5")
    assert spec.embed_family == "nomic-v1.5"
    assert spec.embed_counter == "server-tokenize"
    env = spec.launch_env()
    assert env["OKFGRAPH_EMBED_FAMILY"] == "nomic-v1.5"


def test_no_embedder_wired_keeps_okf_in_process():
    spec = spec_from({"db_path": "kb.db"})
    assert (spec.backend, spec.endpoint_url) == ("embroider", "")
    assert spec.model_id == "google/embeddinggemma-2"
    assert spec.launch_env() == {}


def test_a_gguf_handle_on_the_onx_lane_is_ignored():
    handle = {"backend": "gguf", "role": "embedding", "pool": Pool()}
    spec = spec_from({"db_path": "kb.db", "embedder": handle})
    assert spec.backend == "embroider", "the row decides, as before"


def test_a_wired_image_model_maps_to_the_external_image_lane():
    spec = spec_from({**RRF, "embedder": onnx_embedder(),
                      "image_model": vision_handle()})
    assert spec.image_backend == "external"
    assert spec.image_endpoint_url == "http://127.0.0.1:8931/v1"
    assert spec.image_model_id == VISION
    env = spec.launch_env()
    assert env["OKFGRAPH_IMAGE_BACKEND"] == "external"
    assert env["OKFGRAPH_IMAGE_ENDPOINT_URL"] == "http://127.0.0.1:8931/v1"
    assert env["OKFGRAPH_IMAGE_MODEL_ID"] == VISION
    assert "OKFGRAPH_IMAGE_PRECISION" not in env, "auto: the host's landed one"
    assert "Embedding images with" in _image_line(spec)
    assert "http://127.0.0.1:8931/v1" in _image_line(spec)


def test_the_image_model_row_changes_restart_the_server():
    plain = spec_from({**RRF, "embedder": onnx_embedder()})
    imaged = spec_from({**RRF, "embedder": onnx_embedder(),
                        "image_model": vision_handle()})
    assert plain != imaged
    assert "image_backend" in EMBED_KEYS and "image_endpoint_url" in EMBED_KEYS


def test_the_vision_partner_must_be_the_graphs_text_model():
    handle = vision_handle()
    fine = wiring_problem({**RRF, "embedder": onnx_embedder(),
                           "image_model": handle})
    assert fine == ""
    jina = onnx_embedder()  # same graph model id in this fixture
    wrong = dict(handle, text_partner="jinaai/jina-embeddings-v5-text-nano-retrieval")
    problem = wiring_problem({**RRF, "embedder": jina, "image_model": wrong})
    assert TEXT in problem and "jinaai/jina-embeddings" in problem
    assert "shares vectors with" in problem
    no_partner = dict(handle, text_partner="")
    assert "names no text partner" in wiring_problem(
        {**RRF, "embedder": jina, "image_model": no_partner})


def test_the_host_width_must_be_the_graphs_width():
    problem = wiring_problem({**RRF, "embedder": onnx_embedder(dim=768),
                              "embedding_dim": "512"})
    assert "open at 768 dims" in problem and "512" in problem
    assert wiring_problem({**RRF, "embedder": onnx_embedder(dim=512),
                           "embedding_dim": "512"}) == ""


def test_the_info_line_names_the_host_lane():
    spec = spec_from({**RRF, "embedder": onnx_embedder()})
    line = "\n".join(_model_lines(spec, "onnx-host"))
    assert "ONNX host" in line and TEXT in line
    assert "http://127.0.0.1:8931/v1" in line
    assert "512 dims" in line


def test_a_dead_loader_is_reported_not_raised():
    class DeadPool:
        model_path = "D:/models/x.gguf"
        serves_tokenize = False

        @property
        def base_url(self):
            raise RuntimeError("port in use")

    handle = handle_for("embedding", TEXT, DeadPool(),
                        {"device": "cpu", "precision": "fp32", "dim": 512})
    assert "port in use" in wiring_problem({**RRF, "embedder": handle})
    assert spec_from({**RRF, "embedder": handle}).backend == "embroider"


def test_a_chat_handle_is_not_an_embedder():
    chat = {"backend": "gguf", "role": "chat", "pool": Pool()}
    problem = wiring_problem({**RRF, "embedder": chat})
    assert "embedding model" in problem
    assert wiring_problem({"db_path": "kb.db", "embedder": chat}) == "", \
        "ignored on the onnx lane, as before"
