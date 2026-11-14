# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

What the Knowledge Graph tells ``okf serve``: the spec, and which lane
each wired model travels on.

Qt-free on purpose: the node builds its widgets, but what okf is told is
plain data that the tests read directly (the S7 seam). The rule is
section 3.3 of ``docs/ONNX_UNIFICATION_PLAN.md``:

==========================  ==============================================
wire                        okf runs
==========================  ==============================================
``embedder`` <- ONNX Loader ``backend=external``, identity ``auto`` (the
                            host names its family, width, precision and
                            space; no prefixes), counted by the host. On
                            either Embedder row: the wire is the choice
``embedder`` <- Embedder    ``backend=external``, declared family, pooling,
Loader (GGUF)               prefixes and ``/tokenize`` counts (gguf row;
                            the onnx row refuses it with the fix)
``embedder`` unwired        okf's own in-process embroider, on the graph's
                            pin (or okf's default for a new graph)
``reranker`` <- either      ``rerank_backend=external``
``image_model`` <- ONNX     ``image_backend=external`` at the host, which
Loader (vision)             okf checks against the graph's text model
``image_model`` unwired     okf's in-process vision slot, or captions
==========================  ==============================================

With no embedder wired the graph keeps the model it was last embedded
with -- unplugging a loader is not a request to re-embed.

Every refusal that can be read off a handle (width, text partner, the
precision the host would land on against the graph's pin) is checked
before a lazy pool is asked for its address, because asking starts it:
a wiring that will be refused never launches a model server.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import replace

from . import embed_prefixes
from .gguf_meta import EmbeddingMeta, read_embedding_meta
from .okf_graph import (
    DEFAULT_EMBED_MODEL, DEVICES, DIM_RANGE, EMBED_PRECISIONS, FAMILY_LADDERS,
    FAMILY_PREFIXES, IMAGE_PRECISIONS, LADDER_DIMS, RERANK_PRECISIONS,
    VISION_MAX_DIM, OkfSpec, vision_partner,
)
from .onnx_roles import landing_precision
from .port_text import text_input

#: The Embedder row: which wire supplies the embedding model.
EMBED_SOURCES = ("onnx", "gguf")

#: The settings that decide what the vectors are made with, and which
#: restart the server when they change (image rows included: the image
#: lane is a boot setting too).
EMBED_KEYS = ("device", "model_id", "embedding_dim", "precision", "backend",
              "embed_family", "endpoint_url", "image_backend",
              "image_endpoint_url", "image_model_id", "image_precision")


def embed_source(inputs: Dict[str, Any]) -> str:
    """``gguf`` or ``onnx``: the lane the Embedder row picks."""
    chosen = text_input(inputs, "embed_source", "onnx")
    return chosen if chosen in EMBED_SOURCES else "onnx"


def _graph_dim(inputs: Dict[str, Any]) -> int:
    """The graph's vector width, as the node's row spells it."""
    raw = text_input(inputs, "embedding_dim", "512").strip()
    try:
        dim = int(raw)
    except ValueError:
        dim = 512
    return dim if dim in LADDER_DIMS else 512


def _is_onnx_embedder(handle: Any) -> bool:
    """Whether *handle* is the ONNX Loader's embedding handle (read off the
    dict alone: nothing starts)."""
    return (isinstance(handle, dict) and handle.get("backend") == "onnx"
            and handle.get("role") == "embedding"
            and handle.get("pool") is not None)


def _is_gguf_embedder(handle: Any) -> bool:
    """Whether *handle* is the Embedder Loader's (nothing starts)."""
    return (isinstance(handle, dict) and handle.get("backend") == "gguf"
            and handle.get("role") == "embedding"
            and handle.get("pool") is not None)


def _onnx_host_embedder(handle: Any) -> Dict[str, Any]:
    """``{endpoint_url, model_id, dim, precision, device}`` for the ONNX
    Loader's embedding handle, ``{error}`` when its server would not
    start, empty for anything else.

    okf's external lane with ``embed_identity=auto`` asks the host who it
    is and pins the space it names, so Silk only passes the address, the
    model id (the host must serve exactly it) and the width the host was
    opened at. Asking for the address starts a lazy pool's server: this
    is the demand the loader waits for.
    """
    if not (isinstance(handle, dict) and handle.get("backend") == "onnx"
            and handle.get("role") == "embedding"):
        return {}
    pool = handle.get("pool")
    if pool is None:
        return {}
    try:
        base = pool.base_url
        base = str(base() if callable(base) else base).rstrip("/")
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return {"error": str(exc) or type(exc).__name__}
    model_id = str(handle.get("model_id") or "").strip()
    if not base or "/" not in model_id:
        return {}
    return {"endpoint_url": base, "model_id": model_id,
            "dim": int(handle.get("dim") or 0),
            "precision": str(handle.get("precision") or "auto"),
            "device": str(handle.get("device") or "cpu")}


def _onnx_vision(handle: Any) -> Dict[str, Any]:
    """``{endpoint_url, model_id, dim, precision, text_partner}`` for the
    ONNX Loader's Vision handle, ``{error}`` when its server would not
    start, empty for anything else."""
    if not (isinstance(handle, dict) and handle.get("role") == "vision"):
        return {}
    pool = handle.get("pool")
    if pool is None:
        return {}
    try:
        base = pool.base_url
        base = str(base() if callable(base) else base).rstrip("/")
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return {"error": str(exc) or type(exc).__name__}
    if not base:
        return {}
    return {"endpoint_url": base,
            "model_id": str(handle.get("model_id") or ""),
            "dim": int(handle.get("dim") or 0),
            "precision": str(handle.get("precision") or "auto"),
            "text_partner": str(handle.get("text_partner") or "")}


def _with_image(spec: OkfSpec, image: Any) -> OkfSpec:
    """*spec* with the wired image model's rows; unchanged when none is."""
    host = _onnx_vision(image)
    if not host.get("endpoint_url"):
        return spec
    return replace(
        spec, image_backend="external",
        image_endpoint_url=host["endpoint_url"],
        image_model_id=host["model_id"],
        image_precision=_choice(host["precision"], IMAGE_PRECISIONS),
    )


def _image_problem(image: Any, text: str, graph_dim: int) -> str:
    """Why the wired image model cannot serve a graph whose text model is
    *text* at *graph_dim*, or ``""``. Read off the handle: nothing starts.

    okf checks the same rules at boot on the host's own ``text_partner``;
    saying it here turns the node red instead of the server.
    """
    if not (isinstance(image, dict) and image.get("role") == "vision"
            and image.get("pool") is not None):
        return ""
    model = str(image.get("model_id") or "")
    partner = str(image.get("text_partner") or "")
    dim = int(image.get("dim") or 0)
    if not partner:
        return (f"{model} names no text partner: image vectors would "
                "compare against nothing.")
    if partner != text:
        return (f"{model} shares vectors with {partner}, not {text}: wire "
                f"{partner}'s embedder, or pick the vision model paired "
                f"with {text}.")
    if graph_dim > VISION_MAX_DIM:
        return (f"Image search needs a vector width of at most "
                f"{VISION_MAX_DIM} ({graph_dim} here): images are searched "
                "by their captions. Unwire image_model, or rebuild the graph "
                "narrower.")
    if dim and dim != graph_dim:
        return (f"The vision host would open at {dim} dims but the graph is "
                f"{graph_dim}: set the loader's Width to {graph_dim}.")
    return ""


def _precision_problem(handle: Dict[str, Any], db_path: str,
                       applied: Dict[str, str]) -> str:
    """Why the ONNX host would land on another precision than the graph's
    pin, or ``""`` (plan section 6, Q5).

    The pin is the one this node last served the graph with (``applied``):
    a graph it has not served yet is okf's to check at boot. The loader's
    Device is the user's call, so the fix names both ways out.
    """
    pinned = str(applied.get("pinned_precision") or "").strip().lower()
    model = str(handle.get("model_id") or "")
    if (not pinned or not db_path or applied.get("db_path") != db_path
            or applied.get("model_id") != model):
        return ""
    device = str(handle.get("device") or "cpu")
    lands = landing_precision(model, device, str(handle.get("precision") or "auto"))
    if not lands or lands == pinned:
        return ""
    if pinned == "fp16" and device == "cpu":
        fix = "set the loader's Device to cuda"
    else:
        fix = f"set the loader's Precision to {pinned}"
    return (f"The graph is pinned to {model} at {pinned}; this loader would "
            f"land {lands} on {device}: {fix}, or press Re-embed to rebuild "
            f"the graph at {lands}.")


def _text_model(inputs: Dict[str, Any], applied: Dict[str, str]) -> str:
    """The text model the graph's vectors come from, asked of no server."""
    handle = inputs.get("embedder")
    if _is_onnx_embedder(handle):
        return str(handle.get("model_id") or "")
    if embed_source(inputs) == "gguf" and _is_gguf_embedder(handle):
        try:
            return f"gguf/{Path(str(handle['pool'].model_path)).stem}"
        except Exception:  # noqa: BLE001 - only a label
            return "a GGUF embedder"
    plain = {k: v for k, v in inputs.items()
             if k not in ("embedder", "image_model", "reranker")}
    return spec_from(plain, applied).model_id or DEFAULT_EMBED_MODEL


def spec_from(inputs: Dict[str, Any],
              applied: Optional[Dict[str, str]] = None) -> OkfSpec:
    """The server spec: paths from *inputs*, the model from the wires.

    The ONNX Loader on ``embedder`` is the host lane on either Embedder
    row. With no embedder wired on the ONNX lane, the model the graph was
    last embedded with (*applied*), else okf's default.
    """
    dim = _graph_dim(inputs)
    applied = applied or {}
    spec = OkfSpec(
        db_path=text_input(inputs, "db_path"),
        executable=text_input(inputs, "executable"),
        bundle_root=text_input(inputs, "bundle_root"),
        embedding_dim=dim,
    )
    handle = inputs.get("embedder")
    host = _onnx_host_embedder(handle) if _is_onnx_embedder(handle) else {}
    if host.get("endpoint_url"):
        # embroider's own host on okf's external lane with identity
        # 'auto': the host names its family, width, precision and space,
        # and applies its own prompts, so silence is right.
        spec = replace(
            spec, backend="external", endpoint_url=host["endpoint_url"],
            model_id=host["model_id"],
        )
    elif embed_source(inputs) == "gguf":
        gguf = _gguf_embedder(handle)
        if gguf.get("endpoint_url"):
            # Silk's llama.cpp server, on okf's external lane. The server
            # answers any model name with the one it has loaded, so the
            # wire's name can be the pin: stable across ports and restarts.
            spec = replace(
                spec, backend="external", endpoint_url=gguf["endpoint_url"],
                model_id=gguf["model_id"], embed_family=gguf["family"],
                embed_pooling=gguf["pooling"],
                embedding_dim=_gguf_dim(gguf, dim),
                embed_counter=gguf["counter"], embed_counter_gguf=gguf["path"],
                embed_prefixes=gguf["prefixes"],
            )
    else:
        onnx = _onnx_description(inputs.get("onnx_embedder"))
        if onnx is None and applied.get("backend", "embroider") == "embroider":
            onnx = applied
        if onnx:
            # An empty model id in a saved record is okf's own default:
            # what graphs made before Silk had one were embedded with.
            spec = replace(
                spec, model_id=str(onnx.get("model_id") or ""),
                device=_choice(onnx.get("device"), DEVICES),
                precision=_choice(onnx.get("precision"), EMBED_PRECISIONS),
                cache_dir=str(onnx.get("cache_dir") or ""),
                image_model_id=str(onnx.get("image_model_id") or ""),
                image_precision=_choice(onnx.get("image_precision"),
                                        IMAGE_PRECISIONS),
            )
        else:
            spec = replace(spec, model_id=DEFAULT_EMBED_MODEL)
    return _with_image(_with_reranker(spec, inputs.get("reranker")),
                       inputs.get("image_model"))


def wiring_problem(inputs: Dict[str, Any],
                   applied: Optional[Dict[str, str]] = None, *,
                   reembedding: bool = False) -> str:
    """Why the wired models cannot be used as they are, or ``""``.

    The checks a handle can answer run first, so a refused wiring starts
    no server. *reembedding* (the Re-embed button) lifts the precision
    check: rebuilding at the new precision is the other way out.
    """
    applied = applied or {}
    handle = inputs.get("embedder")
    source = embed_source(inputs)
    dim = _graph_dim(inputs)
    image = inputs.get("image_model")

    # -- read off the handles: nothing starts ---------------------------
    if _is_onnx_embedder(handle):
        width = int(handle.get("dim") or 0)
        if width and width != dim:
            return (f"The ONNX host would open at {width} dims "
                    f"({handle.get('model_id')}) but the graph is {dim}: set "
                    f"the loader's Width to {dim}, or the graph's Vector dim "
                    f"to {width}.")
        if not reembedding:
            pinned = _precision_problem(
                handle, text_input(inputs, "db_path"), applied)
            if pinned:
                return pinned
    elif (source == "onnx" and _is_gguf_embedder(handle)
          and _onnx_description(inputs.get("onnx_embedder")) is None):
        # With a spec node on onnx_embedder too, the row is choosing
        # between two wires; alone, the GGUF wire would be ignored silently.
        return ("A GGUF embedder is wired on embedder, but the Embedder row "
                "is onnx: set the row to gguf to embed with it, or unwire it.")
    problem = _image_problem(image, _text_model(inputs, applied), dim)
    if problem:
        return problem

    # -- ask the pools: these start their servers -------------------------
    failed = _gguf_reranker(inputs.get("reranker")).get("error")
    if failed:
        return f"The reranker's server did not start: {failed}"
    if _is_onnx_embedder(handle):
        host = _onnx_host_embedder(handle)
        if host.get("error"):
            return f"The embedder's server did not start: {host['error']}"
    elif source == "gguf" and handle:
        gguf = _gguf_embedder(handle)
        if gguf.get("prefix_error"):
            return f"The Embedder Loader's prefixes: {gguf['prefix_error']}"
        if gguf.get("error"):
            return f"The embedder's server did not start: {gguf['error']}"
        if not gguf:
            return ("The embedder input takes an embedding model: the "
                    "Embedder Loader's model_obj (a GGUF model on Silk's "
                    "llama.cpp server) or the ONNX Loader's (embroider's own "
                    "host); this model handle is not one.")
        width = gguf["width"]
        if (gguf["family"] not in FAMILY_LADDERS and width
                and not DIM_RANGE[0] <= width <= DIM_RANGE[1]):
            return (f"{gguf['model_id']} makes {width}-dim vectors; okf "
                    f"stores {DIM_RANGE[0]} to {DIM_RANGE[1]}. Load a "
                    "smaller embedding model.")
    if isinstance(image, dict) and image.get("role") == "vision":
        started = _onnx_vision(image)
        if started.get("error"):
            return f"The image model's server did not start: {started['error']}"
    return ""


def _choice(value: Any, choices: Tuple[str, ...]) -> str:
    text = str(value or "").strip().lower()
    return text if text in choices else "auto"


def _onnx_description(value: Any) -> Optional[Dict[str, Any]]:
    """A Silk KG Embedder (ONNX) node's wire value, or ``None``."""
    if isinstance(value, dict) and value.get("runtime") == "embroider":
        return value
    return None


@lru_cache(maxsize=32)
def _embedding_meta_cached(path: str, _mtime: float) -> EmbeddingMeta:
    try:
        return read_embedding_meta(path)
    except (OSError, ValueError):
        return EmbeddingMeta()


def _embedding_meta(path: str) -> EmbeddingMeta:
    """The GGUF's embedding identity, read once per file version."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return EmbeddingMeta()
    return _embedding_meta_cached(path, mtime)


def _family_for(name: Optional[str], path: str) -> str:
    """okf's family label for a GGUF: ``nomic-v1.5`` when it is one (okf
    has its prefixes and ladder), else the model's own name, made a legal
    label (no ``/``)."""
    text = (name or Path(path).stem).lower().replace("_", "-")
    if "nomic-embed-text-v1.5" in text or "nomic-embed-text-v1-5" in text:
        return "nomic-v1.5"
    return re.sub(r"[^a-z0-9.-]+", "-", text).strip("-.")[:64] or "gguf"


def _gguf_dim(gguf: Dict[str, Any], dim: int) -> int:
    """The width to store: the row's step where the family has a ladder,
    else the model's own width (okf cannot shorten it)."""
    ladder = FAMILY_LADDERS.get(gguf["family"])
    if ladder:
        return dim if dim in ladder else ladder[-1]
    return int(gguf["width"] or dim)


def _gguf_embedder(handle: Any) -> Dict[str, Any]:
    """``{endpoint_url, model_id, path, family, pooling, width}`` for an
    Embedder Loader handle.

    Empty for anything else: a chat model's pool, a remote endpoint, or
    nothing wired. okf appends ``/v1/embeddings`` to the endpoint itself,
    so the pool's ``.../v1`` base loses its suffix here. The family,
    pooling and width come from the GGUF file (``width`` 0 when it does
    not say); llama.cpp pools by the file's own key, so that is what the
    vectors are.
    """
    if not (isinstance(handle, dict) and handle.get("backend") == "gguf"
            and handle.get("role") == "embedding"):
        return {}
    pool = handle.get("pool")
    if pool is None:
        return {}
    try:
        # Asking for the address starts a lazy pool's server: this is the
        # demand the Embedder Loader waits for.
        base = pool.base_url
        base = str(base() if callable(base) else base).rstrip("/")
        path = str(pool.model_path)
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return {"error": str(exc) or type(exc).__name__}
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    if not base or not path:
        return {}
    meta = _embedding_meta(path)
    family = _family_for(meta.name, path)
    name = f"{meta.name or ''} {Path(path).stem}"
    try:
        table = embed_prefixes.resolve(str(handle.get("prefixes") or ""), name)
    except ValueError as exc:
        return {"prefix_error": str(exc)}
    # okf brings its own table for the families it knows: declared as the
    # family, under the pin graphs made before Silk had tables carry.
    builtin = (family in FAMILY_PREFIXES
               and table == embed_prefixes.prefixes_for(name))
    tag = "" if builtin else embed_prefixes.pin_tag(table)
    return {"endpoint_url": base,
            "model_id": f"gguf/{Path(path).stem}" + (f"+p{tag}" if tag else ""),
            "plain_model_id": f"gguf/{Path(path).stem}",
            "prefixes": "" if builtin else embed_prefixes.declaration(table),
            "counter": ("server-tokenize" if getattr(pool, "serves_tokenize", False)
                        else "gguf-meta"),
            "path": path, "family": family,
            "pooling": meta.pooling or "mean", "width": meta.width or 0}


def _wired_embedder(inputs: Dict[str, Any], source: str) -> str:
    """The wire that supplies the model on *source*'s lane, or ``""``.

    The ONNX Loader's embroider host (identity ``auto``) on ``embedder``
    is the wire on either row; a GGUF embedder (declared family) only on
    the ``gguf`` row.
    """
    handle = inputs.get("embedder")
    if _is_onnx_embedder(handle):
        return "onnx-host" if _onnx_host_embedder(handle).get("endpoint_url") else ""
    if source == "gguf":
        return "gguf" if _gguf_embedder(handle).get("endpoint_url") else ""
    return "onnx" if _onnx_description(inputs.get("onnx_embedder")) else ""


def _model_lines(spec: OkfSpec, wired: str) -> List[str]:
    """What embeds and what reranks."""
    lines: List[str] = []
    if wired == "gguf":
        lines.append(
            f"Embedding through the wired GGUF embedder {spec.model_id} "
            f"(from its file: family {spec.embed_family}, "
            f"{spec.embed_pooling} pooling, {spec.embedding_dim} dims); it "
            "must stay loaded for search and ingest to work.")
    elif wired == "onnx-host":
        lines.append(
            f"Embedding through the wired ONNX host {spec.model_id} at "
            f"{spec.endpoint_url} ({spec.embedding_dim} dims): it names its "
            "own precision and vector space, and must stay loaded for "
            "search and ingest to work.")
    elif wired == "onnx":
        lines.append(
            f"Embedding with the wired ONNX model {spec.model_id or 'default'} "
            f"on {spec.device} ({spec.precision}).")
    else:
        lines.append(
            f"No embedder wired: okf loads "
            f"{spec.model_id or 'its default model (jina v5 small)'} itself, on "
            f"{spec.device}. Wire Silk KG Embedder (ONNX) or an Embedder Loader to "
            "change it.")
    if spec.image_backend == "external":
        lines.append(
            f"Embedding images with {spec.image_model_id} at "
            f"{spec.image_endpoint_url}.")
    elif wired != "gguf":
        lines.append(_image_line(spec))
    if spec.rerank_model_id:
        where = (f"at {spec.rerank_endpoint_url}" if spec.rerank_backend == "external"
                 else f"on {spec.rerank_device} ({spec.rerank_precision})")
        lines.append(f"Reranking every search with {spec.rerank_model_id} {where}.")
    else:
        lines.append("No reranker wired: searches are not reranked.")
    return lines


def _image_line(spec: OkfSpec) -> str:
    """How images are embedded for image-content search."""
    if spec.image_backend == "external":
        return (f"Embedding images with {spec.image_model_id} at "
                f"{spec.image_endpoint_url}.")
    image = spec.image_model_id or vision_partner(spec.model_id)
    if not image:
        return (f"No image model pairs with {spec.model_id or 'the default model'}: "
                "images are searched by their captions.")
    if spec.embedding_dim > VISION_MAX_DIM:
        return (f"Image search needs a vector width of at most {VISION_MAX_DIM} "
                f"({spec.embedding_dim} here): images are searched by their captions.")
    return f"Embedding images with {image} ({spec.image_precision})."


def _gguf_reranker(handle: Any) -> Dict[str, Any]:
    """``{endpoint_url}`` for a Reranker Loader (or ONNX Loader as a reranker)
    handle, ``{error}`` when its server would not start, empty for
    anything else.

    Asking the lazy pool for its address starts the server: this is the
    demand the loader waits for.
    """
    if not (isinstance(handle, dict) and handle.get("runtime") in _POOL_RUNTIMES
            and handle.get("role") == "rerank"):
        return {}
    pool = handle.get("pool")
    if pool is None:
        return {}
    try:
        base = pool.base_url
        base = str(base() if callable(base) else base).rstrip("/")
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return {"error": str(exc) or type(exc).__name__}
    return {"endpoint_url": base} if base else {}


#: Reranker runtimes that are a Silk server behind a lazy pool: the
#: Reranker Loader's GGUF one and the ONNX Loader's embroider one.
_POOL_RUNTIMES = ("gguf", "onnx")


def _with_reranker(spec: OkfSpec, reranker: Any) -> OkfSpec:
    """*spec* with the wired reranker's rows; unchanged when none is."""
    if not (isinstance(reranker, dict) and reranker.get("role") == "rerank"):
        return spec
    model_id = str(reranker.get("model_id") or "")
    if reranker.get("runtime") in _POOL_RUNTIMES:
        gguf = _gguf_reranker(reranker)
        if not gguf.get("endpoint_url"):
            return spec
        # A Silk rerank server on okf's external lane: okf cuts pairs to
        # the server's context and counts them with its /tokenize.
        return replace(
            spec, rerank_backend="external", rerank_model_id=model_id,
            rerank_endpoint_url=gguf["endpoint_url"],
            rerank_family=str(reranker.get("family") or "gguf-reranker"),
            rerank_max_tokens=int(reranker.get("max_tokens") or 0),
            rerank_count="server-tokenize",
        )
    if reranker.get("runtime") == "external" and reranker.get("endpoint_url"):
        return replace(
            spec, rerank_backend="external", rerank_model_id=model_id,
            rerank_endpoint_url=str(reranker.get("endpoint_url") or ""),
            rerank_family=str(reranker.get("family") or ""),
            rerank_api_key_env=str(reranker.get("api_key_env") or ""),
        )
    if reranker.get("runtime") != "embroider":
        return spec
    return replace(
        spec, rerank_backend="embroider", rerank_model_id=model_id,
        rerank_device=_choice(reranker.get("device"), DEVICES),
        rerank_precision=_choice(reranker.get("precision"), RERANK_PRECISIONS),
        # One cache per server: the embedder's folder, if it named one.
        cache_dir=spec.cache_dir or str(reranker.get("cache_dir") or ""),
    )
