# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The ONNX host boundary, Qt-free: the command Silk starts, the bases each
consumer appends to, the four roles' handles, and -- when embroider and
a model are cached -- the real ``python -m embroider.serve`` child.

S1 moved the host from Silk's own ``functions/embroider_server.py`` to
embroider's ``embroider.serve``. These tests are what kept that swap
honest: the script file is gone, the command is ``-m embroider.serve``,
and every wire handle a role hands out still validates on its port type.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from silk.functions.embroider_pool import (
    KINDS, SERVER_MODULE, TEXT_KINDS, base_url_for, embroider_available,
)
from silk.functions.model_pool import LazyGGUFPool
from silk.functions.onnx_roles import (
    DECIDE_MODELS, DECISION, ROLE_KINDS, ROLE_MODELS,
    ROLE_PRECISIONS, handle_for, kind_for, onnx_family, serve_args, shipped,
)
from silk.functions.rerank_pool import child_command

_SMALL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_the_child_runs_embroiders_own_host():
    assert SERVER_MODULE == "embroider.serve"
    args = serve_args("embed", "google/embeddinggemma-2", device="cpu",
                      precision="fp16", cache_dir="C:/cache", dim=256)
    cmd = child_command(args, "127.0.0.1", 8000, module=SERVER_MODULE)
    assert cmd[:3] == [sys.executable, "-m", "embroider.serve"]
    assert "--kind" in cmd and cmd[cmd.index("--kind") + 1] == "embed"
    assert cmd[cmd.index("--model") + 1] == "google/embeddinggemma-2"
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--precision") + 1] == "fp16"
    assert cmd[cmd.index("--cache-dir") + 1] == "C:/cache"
    assert cmd[cmd.index("--dim") + 1] == "256"
    assert cmd[-4:] == ["--host", "127.0.0.1", "--port", "8000"]
    assert not any(part.endswith("embroider_server.py") for part in cmd)


def test_a_file_script_still_runs_as_before():
    cmd = child_command(["--model", "m.gguf"], "127.0.0.1", 9,
                        script="C:/Silk/functions/rerank_server.py")
    assert cmd == [sys.executable, "C:/Silk/functions/rerank_server.py",
                   "--model", "m.gguf", "--host", "127.0.0.1", "--port", "9"]


def test_each_kind_answers_the_base_its_consumers_append():
    assert base_url_for("embed", "http://h:9") == "http://h:9/v1"
    assert base_url_for("vision", "http://h:9") == "http://h:9/v1"
    assert base_url_for("rerank", "http://h:9") == "http://h:9"
    assert base_url_for("decide", "http://h:9") == "http://h:9"
    assert base_url_for("decide-laya", "http://h:9") == "http://h:9"
    assert base_url_for("decide-2b", "http://h:9") == "http://h:9"


def test_tokenize_is_claimed_where_it_exists_and_not_for_a_vision_model():
    assert set(TEXT_KINDS) < set(KINDS)
    for kind in ("embed", "rerank", "decide", "decide-laya", "decide-2b"):
        pool = LazyGGUFPool("m/x", pool_class=_FakePool, kind=kind)
        assert pool.serves_tokenize, f"{kind} serves /tokenize"
    vision = LazyGGUFPool("m/x", pool_class=_FakePool, kind="vision")
    assert not vision.serves_tokenize, "a vision host serves no text"
    assert not vision.started, "and asking starts nothing"


class _FakePool:
    serves_tokenize = True
    serves_tokenize_kinds = frozenset(TEXT_KINDS)


def test_the_roles_map_to_embroiders_kinds():
    assert ROLE_KINDS == {"embedding": "embed", "reranker": "rerank",
                          "vision": "vision"}
    assert kind_for("embedding", "google/embeddinggemma-2") == "embed"
    assert kind_for("vision", "google/embeddinggemma-2-vision") == "vision"
    assert kind_for(DECISION, "convaiinnovations/laya") == "decide-laya"
    assert kind_for(DECISION, "knowledgator/gliclass-v4-edge") == "decide"
    assert kind_for(DECISION, "Mapika/decider-2b") == "decide-2b"
    assert kind_for(DECISION, "someone/else") == "decide", "unknown falls to GLiClass"
    for role, models in ROLE_MODELS.items():
        assert models, f"{role} offers a model"
        for model in models:
            assert kind_for(role, model) in KINDS


def test_the_precision_rows_match_what_each_kind_ships():
    assert shipped("embedding", "google/embeddinggemma-2") == (
        "auto", "fp32", "fp16"), "only what the registry ships"
    assert shipped("vision", "google/embeddinggemma-2-vision")[0] == "auto"
    assert shipped(DECISION, "convaiinnovations/laya") == (
        "auto", "fp32", "q8")
    for role, precisions in ROLE_PRECISIONS.items():
        assert precisions[0] == "auto", f"{role} defaults to auto"


def test_each_role_hands_out_the_handle_its_consumer_reads():
    pool = object()

    embedding = handle_for("embedding", "google/embeddinggemma-2", pool)
    assert (embedding["backend"], embedding["role"]) == ("onnx", "embedding")
    assert embedding["pool"] is pool

    reranker = handle_for("reranker", _SMALL, pool)
    assert (reranker["role"], reranker["runtime"]) == ("rerank", "onnx")
    assert reranker["max_tokens"] == 512
    assert reranker["family"] == "ms-marco-minilm-l-6-v2"

    vision = handle_for("vision", "google/embeddinggemma-2-vision", pool)
    assert (vision["role"], vision["runtime"]) == ("vision", "onnx")
    assert vision["text_partner"] == "google/embeddinggemma-2", \
        "an image model names the text model its vectors compare against"

    decision = handle_for(DECISION, "convaiinnovations/laya", pool)
    assert (decision["backend"], decision["role"]) == ("systemone", "decision")
    assert decision["pool"] is pool
    assert onnx_family("convaiinnovations/laya") == "laya"


def test_the_handles_pass_their_port_validators():
    from silk.nodes.silk_ports import (
        CLASSIFIER_MODEL_TYPE, IMAGE_MODEL_TYPE, RERANKER_MODEL_TYPE,
    )
    pool = object()
    assert IMAGE_MODEL_TYPE.validator(
        handle_for("vision", "google/embeddinggemma-2-vision", pool))
    assert not IMAGE_MODEL_TYPE.validator({"role": "rerank", "pool": pool})
    assert not IMAGE_MODEL_TYPE.validator({"role": "vision"}), "no pool, no host"
    assert RERANKER_MODEL_TYPE.validator(handle_for("reranker", _SMALL, pool))
    assert CLASSIFIER_MODEL_TYPE.validator(
        handle_for(DECISION, "convaiinnovations/laya", pool))


def test_the_pinned_decision_and_vision_models_match_embroiders_registry():
    """The Role rows are a pinned copy; this refuses a stale one.

    Asked in a child, so importing embroider (and ONNX Runtime) can never
    touch the test process's runtime, and skipped where it is absent.
    """
    code = (
        "import embroider, json\n"
        "print(json.dumps(embroider.available_models()))\n")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=120)
    except Exception as exc:  # noqa: BLE001 - no child, no drift check
        pytest.skip(f"embroider could not be asked: {exc}")
    if out.returncode != 0:
        pytest.skip(f"embroider could not be asked: {out.stderr[-200:]}")
    rows = {row["id"]: row["kind"] for row in json.loads(out.stdout)}
    for model, kind in DECIDE_MODELS.items():
        assert rows.get(model) == kind, f"{model} is {rows.get(model)}, not {kind}"
    for model in ROLE_MODELS["vision"]:
        assert rows.get(model) == "vision"
    registry_decide = {m for m, k in rows.items() if k.startswith("decide")}
    assert registry_decide == set(DECIDE_MODELS), \
        "a decision model was added or removed: update DECIDE_MODELS"
    registry_vision = {m for m, k in rows.items() if k == "vision"}
    assert registry_vision == set(ROLE_MODELS["vision"])


def call(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=60) as r:
        return json.loads(r.read())


@pytest.mark.skipif(
    not (Path.home() / ".cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2").is_dir(),
    reason="the small cross-encoder is not cached")
def test_the_real_host_serves_identity_counts_and_reranks():
    """``python -m embroider.serve``, as the pool starts it."""
    if not embroider_available():
        pytest.skip("embroider is not installed")
    from silk.functions.embroider_pool import EmbroiderPool
    pool = LazyGGUFPool(_SMALL, pool_class=EmbroiderPool, kind="rerank",
                        device="cpu")
    try:
        root = pool.base_url
        assert root.startswith("http://127.0.0.1:")
        identity = call(root + "/v1/models")["data"][0]
        assert identity["engine"] == "embroider"
        assert identity["kind"] == "rerank"
        assert identity["contract"], "the host publishes its wire contract"
        assert identity["family"] == "ms-marco-MiniLM-L-6-v2"
        assert identity["device"] == "cpu"
        out = call(root + "/v1/rerank", {
            "query": "what is a cat",
            "documents": ["Paris is in France", "A cat is a small feline"]})
        assert out["results"][0]["index"] == 1
        counted = call(root + "/tokenize", {"content": "one two three"})
        assert counted["count"] == 3
        info = pool.snapshot()
        assert info["precision"] == identity["precision"]
        assert info["max_length"] == 512
    finally:
        pool.cleanup()
    assert not pool.started
