# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

What an embroider model is for, as plain data.

The ONNX Loader's Role row picks one of four things: an embedder, a
reranker, a vision (image) model, or a typed decision model. Each role
maps to embroider's host kind, its models, the precisions they ship and
the handle the node puts on the wire -- and every one of those is a
table here, Qt-free, so the node, the tests and the Knowledge Graph read
the same definitions.

The decision models' kinds are a pinned copy of embroider's registry
(``embroider.available_models()``); the drift test in
``tests/test_silk_onnx_host.py`` refuses a copy that no longer matches.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .okf_graph import (
    DEVICES, EMBED_PRECISIONS, IMAGE_PRECISIONS, KNOWN_MODELS,
    KNOWN_RERANK_MODELS, MODEL_LADDERS, RERANK_MODEL_MAX_TOKENS,
    RERANK_PRECISIONS, VISION_MODELS, precisions_for,
)

__all__ = [
    "DEVICES", "NATIVE", "ROLE_KINDS", "ROLE_MODELS", "ROLE_PRECISIONS",
    "ROLE_NOUNS", "DECIDE_MODELS", "DECIDE_PRECISIONS", "kind_for",
    "shipped", "handle_for", "serve_args", "onnx_family", "width_ladder",
    "opened_width", "landing_precision",
]

#: The Width row's first entry: the model's own width.
NATIVE = "native"

#: Role row value -> embroider's host kind, for the roles whose kind is
#: the role itself. The Decision role's kind is the model's (see
#: :data:`DECIDE_MODELS`).
ROLE_KINDS = {"embedding": "embed", "reranker": "rerank", "vision": "vision"}

#: Role row value of the typed decision models.
DECISION = "decider"

#: Each decision model and embroider's kind for it (registry ``kind``):
#: ``decide`` is GLiClass (zero-shot), ``decide-laya`` the Laya family,
#: ``decide-2b`` Mapika's 2B decider. Default first.
DECIDE_MODELS: Dict[str, str] = {
    "convaiinnovations/laya": "decide-laya",
    "convaiinnovations/laya-multilingual": "decide-laya",
    "convaiinnovations/laya-typed-decisions": "decide-laya",
    "knowledgator/gliclass-v4-edge": "decide",
    "knowledgator/gliclass-small-v1.0": "decide",
    "Mapika/decider-2b": "decide-2b",
}

#: Precisions a decision model may be asked for (registry ``precisions``).
DECIDE_PRECISIONS = ("auto", "fp32", "q8")

#: Role -> the models the Model row offers.
ROLE_MODELS: Dict[str, Sequence[str]] = {
    "embedding": KNOWN_MODELS,
    "reranker": KNOWN_RERANK_MODELS,
    "vision": tuple(VISION_MODELS),
    DECISION: tuple(DECIDE_MODELS),
}

#: Role -> the precisions the Precision row offers.
ROLE_PRECISIONS: Dict[str, Sequence[str]] = {
    "embedding": EMBED_PRECISIONS,
    "reranker": RERANK_PRECISIONS,
    "vision": IMAGE_PRECISIONS,
    DECISION: DECIDE_PRECISIONS,
}

#: Role -> what the status line calls the model.
ROLE_NOUNS = {"embedding": "embedder", "reranker": "reranker",
              "vision": "image model", DECISION: "decision model"}


def onnx_family(model_id: str) -> str:
    """okf's family label for an embroider model: its name, made legal."""
    text = model_id.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9.-]+", "-", text).strip("-.")[:64] or "onnx-reranker"


def kind_for(role: str, model_id: str) -> str:
    """embroider's host kind for *model_id* in *role*."""
    if role == DECISION:
        return DECIDE_MODELS.get(model_id.strip(), "decide")
    return ROLE_KINDS.get(role, "embed")


def shipped(role: str, model_id: str) -> Sequence[str]:
    """The precisions *model_id* can be asked for in *role*."""
    if role == "embedding":
        return precisions_for(model_id)
    return ROLE_PRECISIONS.get(role, ("auto",))


def width_ladder(role: str, model_id: str) -> Sequence[int]:
    """The widths *model_id* can be opened at in *role*, native last.

    A vision model shares its text partner's space, and with it the
    partner's Matryoshka ladder (embroider's registry lists the same one),
    so a vision host can be opened at the graph's width. Empty for a
    model Silk has no table for, and for the roles without a width.
    """
    if role == "embedding":
        return MODEL_LADDERS.get(model_id.strip(), ())
    if role == "vision":
        return MODEL_LADDERS.get(VISION_MODELS.get(model_id.strip(), ""), ())
    return ()


def opened_width(role: str, model_id: str, dim: int) -> int:
    """The width the host will serve: *dim*, or the model's native width
    when *dim* is 0 (``native``). 0 only for a model Silk has no table for."""
    if dim:
        return int(dim)
    ladder = width_ladder(role, model_id)
    return int(ladder[-1]) if ladder else 0


def landing_precision(model_id: str, device: str, precision: str) -> str:
    """The precision an embedding host will land on, or ``""`` when that
    cannot be told before it opens.

    An explicit precision is what it says. ``auto`` is FP32 on the CPU and
    FP16 on CUDA where the model ships FP16 weights. ``auto`` on the
    ``auto`` device depends on whether a GPU answers, so it is unknown.
    """
    precision = (precision or "auto").strip().lower()
    if precision != "auto":
        return precision
    device = (device or "cpu").strip().lower()
    if device == "cpu":
        return "fp32"
    if device == "cuda":
        return "fp16" if "fp16" in precisions_for(model_id) else "fp32"
    return ""


def handle_for(role: str, model_id: str, pool: Any,
               settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The wire handle for a running (or lazy) embroider pool.

    Embedding and reranker keep their old shapes: the embedder's handle is
    what ``functions.embeddings.embedder_for`` reads, the reranker's what
    the Knowledge Graph hands okf's external rerank lane. Embedding and
    vision also carry the **opened** device, precision and width, so the
    Knowledge Graph can predict a mismatch before the host starts (okf
    still asks the host itself at boot). Vision names its text partner
    (the graph's own text model must be that partner for image vectors to
    compare). Decision models use the ``classifier_model`` shape the
    Classifier and Ensemble nodes read.
    """
    settings = settings or {}
    if role == "embedding":
        return {"backend": "onnx", "role": "embedding", "pool": pool,
                "model_id": model_id,
                "device": str(settings.get("device") or "cpu"),
                "precision": str(settings.get("precision") or "auto"),
                "dim": opened_width(role, model_id,
                                    int(settings.get("dim") or 0))}
    if role == "reranker":
        return {"role": "rerank", "runtime": "onnx", "pool": pool,
                "model_id": f"onnx/{model_id}", "family": onnx_family(model_id),
                "max_tokens": RERANK_MODEL_MAX_TOKENS.get(model_id, 512)}
    if role == "vision":
        # VISION_MODELS maps an image model to the text model whose vectors
        # it shares (embroider's registry ``text_partner``).
        return {"role": "vision", "runtime": "onnx", "pool": pool,
                "model_id": model_id,
                "text_partner": VISION_MODELS.get(model_id, ""),
                "precision": str(settings.get("precision") or "auto"),
                "dim": opened_width(role, model_id,
                                    int(settings.get("dim") or 0))}
    return {"backend": "systemone", "role": "decision", "pool": pool,
            "model": model_id}


def serve_args(kind: str, model_id: str, *, device: str = "cpu",
               precision: str = "auto", cache_dir: str = "",
               dim: int = 0) -> List[str]:
    """The arguments ``python -m embroider.serve`` takes for one model.

    One builder for the pool and the tests, so a renamed flag fails in a
    test rather than in a child process.
    """
    args = ["--kind", kind, "--model", model_id,
            "--device", device, "--precision", precision or "auto"]
    if cache_dir:
        args += ["--cache-dir", cache_dir]
    if dim:
        args += ["--dim", str(int(dim))]
    return args
