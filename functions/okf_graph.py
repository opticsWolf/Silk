# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

One OKFgraph knowledge graph, served by ``okf serve`` and owned by a node.

OKFgraph runs **out of process**, always. Importing it here would put its
ONNX Runtime and Ladybug's native libraries into Weave's process, where
OKFgraph's own crash log records a second ONNX Runtime aborting the whole
interpreter; and Ladybug lets one process open a database file, so Weave
holding it would lock the ``okf`` CLI and every MCP client out of the same
graph. ``okf serve`` is OKFgraph's answer to the second problem and makes
the first one moot: one process owns the file and keeps the embedding
model loaded, and everyone else -- Silk included -- calls the router's ops
as JSON over HTTP (``POST /ops/<op>``). Nothing here imports ``okfgraph``;
the client is the standard library.

The shape is the MCP node's (D19): the node owns the server the way it
would own a session, the handle on the wire is a connection that already
exists, and derived ToolSets attach to that same connection by closing
over it. A server already running for the database (the CLI or another
harness started it) is attached to, never replaced, and never stopped.

**Pins.** A graph's vectors are pinned to one model and one precision
(CPU means FP32, GPU means FP16), and OKFgraph refuses to mix them.
:meth:`OkfGraph.pins` reads the pin without loading a model, so the node
can warn on every evaluation; :meth:`OkfGraph.reembed` rebuilds every
vector under the configured model and re-pins, keeping the database file
as a backup until it succeeds.
"""
from __future__ import annotations

import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from weave.logger import get_logger

# The ledger's inverse lane rides the graph's pinned encoder (0.12); the
# Embedder base is the same class the GGUF lanes use, so the memory stack
# cannot tell the difference -- one threadable duck, one failure rule.
from .embeddings import Embedder

#: Embedding rides the /ops/embed wire (+ a server-side batch); a hung
#: server must not hold an agent turn open much beyond that.
EMBED_OP_TIMEOUT = 60.0

log = get_logger("SilkOkfGraph")

#: Inference devices ``okf --device`` accepts. ``auto`` is CUDA when the
#: GPU runtime is installed and a GPU answers, else the CPU.
DEVICES = ("auto", "cpu", "cuda")

#: The model a new graph is embedded with when nothing is wired: Silk's
#: default, not okf's (okf's is the jina small model). embroider >= 0.13
#: (mean pooling + Gemma prompts, native 768).
DEFAULT_EMBED_MODEL = "google/embeddinggemma-2"

#: The text models OKFgraph's registry knows, Silk's default first. An
#: empty model id still means "OKFgraph's default" (the jina small one),
#: which is what graphs built before Silk had its own default use.
KNOWN_MODELS = (
    DEFAULT_EMBED_MODEL,
    "jinaai/jina-embeddings-v5-text-small-retrieval",
    "jinaai/jina-embeddings-v5-text-nano-retrieval",
)

#: The cross-encoders embroider ships, default first (okfgraph's
#: ``DEFAULT_RERANK_MODEL_ID``); embroider 0.14 adds the English MiniLM.
KNOWN_RERANK_MODELS = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

#: The external rerank lane's pair ceiling okf accepts, in tokens.
RERANK_MAX_TOKENS = (64, 1048576)
#: How okf counts tokens for the external rerank lane: by estimate, or by
#: asking the host's root ``/tokenize`` (Silk's rerank server has one).
RERANK_COUNTS = ("heuristic", "server-tokenize")

#: Precisions ``okf`` accepts per stage. ``auto`` follows the device
#: (FP32 on CPU, FP16 on a GPU); an explicit one is part of the pin.
EMBED_PRECISIONS = ("auto", "fp32", "fp16", "int8")
RERANK_PRECISIONS = ("auto", "fp32", "q8")
#: Image weights: no int8 artifact ships, and explicit fp16 on the CPU
#: fails fast in okf.
IMAGE_PRECISIONS = ("auto", "fp32", "fp16")

#: The explicit precisions each text model ships (embroider's registry);
#: a model not listed here may be asked for any of EMBED_PRECISIONS.
MODEL_PRECISIONS = {
    "google/embeddinggemma-2": ("fp32", "fp16"),
    "jinaai/jina-embeddings-v5-text-small-retrieval": ("fp32", "fp16"),
    "jinaai/jina-embeddings-v5-text-nano-retrieval": ("fp32", "fp16", "int8"),
}

#: Each image (vision) model and the text model whose vector space it
#: shares -- embroider's registry ``text_partner``. Image vectors only
#: compare against their partner, so an image model belongs to one text
#: model; okf picks the pair itself when none is named. The jina small
#: text model has no partner: its graphs search images by caption.
VISION_MODELS = {
    "google/embeddinggemma-2-vision": "google/embeddinggemma-2",
    "jina-v5-omni-nano-retrieval-vision":
        "jinaai/jina-embeddings-v5-text-nano-retrieval",
}
#: The widest vector an image model makes; wider graphs get no image search.
VISION_MAX_DIM = 768


def precisions_for(model_id: str) -> Tuple[str, ...]:
    """The precisions an embedding model can be asked for, ``auto`` first."""
    return ("auto",) + MODEL_PRECISIONS.get(model_id, EMBED_PRECISIONS[1:])


def vision_partner(model_id: str) -> str:
    """The image model sharing *model_id*'s vectors, or ``""`` for none."""
    for image_model, text_model in VISION_MODELS.items():
        if text_model == model_id:
            return image_model
    return ""

#: How the external lane pools token vectors; declared, never probed.
POOLINGS = ("mean", "last", "cls")

#: The families okfgraph has a built-in prefix table for. Any other
#: family name is legal and embeds without prefixes.
FAMILY_PREFIXES = ("nomic-v1.5",)

#: The Matryoshka steps okfgraph knows for a family on the external lane;
#: any other family is served at its model's own width.
FAMILY_LADDERS = {"nomic-v1.5": (64, 128, 256, 512, 768)}

#: The widths okf accepts at all (a ladder step is only recommended).
DIM_RANGE = (32, 1024)

#: How long an ordinary op may take. Ingest embeds, so it is not instant,
#: but a hung server must not hold an agent's tool call forever.
DEFAULT_CALL_TIMEOUT = 300.0

#: Re-embedding a large graph is minutes, not seconds.
REEMBED_TIMEOUT = 6 * 3600.0

#: How long ``okf serve`` may take to open the database and listen.
START_TIMEOUT = 120.0

#: How long a stopped server gets to close the database cleanly.
STOP_TIMEOUT = 15.0


class OkfError(RuntimeError):
    """An op failed: OKFgraph's error envelope, or no server to ask."""

    def __init__(self, code: str, message: str, remedy: str = "",
                 fields: Optional[dict] = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.remedy = remedy
        self.fields = dict(fields or {})


def default_executable() -> str:
    """The ``okf`` that came with this environment, else the one on PATH.

    Silk's ``knowledge`` extra installs okfgraph next to Weave, so the
    entry point sits beside the interpreter. Looking there first means
    the graph is served by the same release the extra pinned, not by
    whichever ``okf`` happens to be first on PATH.
    """
    name = "okf.exe" if os.name == "nt" else "okf"
    beside = Path(sys.executable).parent / name
    if beside.is_file():
        return str(beside)
    return shutil.which("okf") or ""


LADDER_DIMS = (32, 64, 128, 256, 512, 768, 1024)


@dataclass(frozen=True)
class OkfSpec:
    """Which graph, and how to embed into it. Plain data, saved with the node.

    ``backend="external"`` (0.12) trades OKFgraph's own ONNX crate for any
    OpenAI-shaped ``POST /v1/embeddings`` host the user points at
    (llama.cpp ``--embeddings``, LM Studio, TEI) -- the same *declared*
    family contract okfgraph boots with: ``embed_family`` is required then,
    and ``endpoint_url`` must be http(s). ``embedding_dim`` is the graph's
    vector width (a Matryoshka step); it applies when the graph is created
    and is refused as a re-embed once it has vectors (okfgraph reports the
    pinned width; a change needs a fresh database by design).

    ``embed_pooling`` / ``embed_counter`` / ``embed_counter_gguf`` are the
    rest of that declaration: a GGUF embedder from Silk's loader counts
    tokens from its own file (``gguf-meta``), because the Python
    ``llama_cpp.server`` serves no root ``/tokenize``.

    The ``rerank_*`` rows choose the second retrieval stage. ``embroider``
    (the default) is a cross-encoder in okf's own process; ``external`` is
    a ``POST /v1/rerank`` host, told its pair ceiling (``rerank_max_tokens``)
    and how to count toward it (``rerank_count``). Rerank writes no
    vectors, so none of it is pinned: changing it restarts the server and
    nothing else.

    ``image_model_id`` / ``image_precision`` choose how images are embedded
    for image-content search. Empty is okf's own pairing (the vision model
    sharing the text model's vectors); okf pins the image model on the
    first image it embeds.
    """

    db_path: str
    executable: str = ""
    bundle_root: str = ""
    device: str = "auto"
    model_id: str = ""
    embedding_dim: int = 512
    precision: str = "auto"
    cache_dir: str = ""
    backend: str = "embroider"
    embed_family: str = ""
    endpoint_url: str = ""
    api_key_env: str = ""
    embed_pooling: str = "mean"
    embed_counter: str = ""
    embed_counter_gguf: str = ""
    image_model_id: str = ""
    image_precision: str = "auto"
    rerank_backend: str = "embroider"
    rerank_model_id: str = ""
    rerank_device: str = "auto"
    rerank_precision: str = "auto"
    rerank_endpoint_url: str = ""
    rerank_family: str = ""
    rerank_api_key_env: str = ""
    rerank_max_tokens: int = 0
    rerank_count: str = ""
    allow_repin: bool = False

    def resolved_executable(self) -> str:
        return self.executable.strip() or default_executable()

    def settings_args(self) -> list[str]:
        """The global flags that choose the graph and its embedding."""
        args = ["--db-path", self.db_path]
        if self.bundle_root.strip():
            args += ["--bundle-root", self.bundle_root.strip()]
        if self.device in DEVICES:
            args += ["--device", self.device]
        if self.model_id.strip():
            args += ["--model-id", self.model_id.strip()]
        # A GGUF embedder's own width (384, say) is legal, if off-ladder.
        if (DIM_RANGE[0] <= self.embedding_dim <= DIM_RANGE[1]
                and self.embedding_dim != 512):
            args += ["--embedding-dim", str(self.embedding_dim)]
        return args

    def launch_env(self) -> dict[str, str]:
        """OKFGRAPH_* env vars for the rows okf serve reads at boot that
        have no CLI spelling worth maintaining: the external-embedder lane
        and the re-embed escape. Empty dict: launch inherits nothing
        beyond the process environment (the default crate lane)."""
        env: dict[str, str] = {}
        if self.precision in EMBED_PRECISIONS and self.precision != "auto":
            env["OKFGRAPH_PRECISION"] = self.precision
        if self.cache_dir.strip():
            env["OKFGRAPH_CACHE_DIR"] = self.cache_dir.strip()
        if self.image_model_id.strip():
            env["OKFGRAPH_IMAGE_MODEL_ID"] = self.image_model_id.strip()
        if self.image_precision in IMAGE_PRECISIONS and self.image_precision != "auto":
            env["OKFGRAPH_IMAGE_PRECISION"] = self.image_precision
        if self.backend == "external":
            env["OKFGRAPH_BACKEND"] = "external"
            if self.endpoint_url.strip():
                env["OKFGRAPH_ENDPOINT_URL"] = self.endpoint_url.strip()
            family = self.embed_family.strip()
            if family:
                env["OKFGRAPH_EMBED_FAMILY"] = family
                if family in FAMILY_PREFIXES:
                    env["OKFGRAPH_EMBED_PREFIXES"] = family
            if self.api_key_env.strip():
                env["OKFGRAPH_API_KEY_ENV"] = self.api_key_env.strip()
            if self.embed_pooling in POOLINGS:
                env["OKFGRAPH_EMBED_POOLING"] = self.embed_pooling
            if self.embed_counter.strip():
                env["OKFGRAPH_EMBED_COUNTER"] = self.embed_counter.strip()
            if self.embed_counter_gguf.strip():
                env["OKFGRAPH_EMBED_COUNTER_GGUF"] = self.embed_counter_gguf.strip()
        if self.rerank_model_id.strip():
            env["OKFGRAPH_RERANK_MODEL_ID"] = self.rerank_model_id.strip()
        if self.rerank_backend == "external":
            env["OKFGRAPH_RERANK_BACKEND"] = "external"
            for key, value in (("OKFGRAPH_RERANK_ENDPOINT_URL", self.rerank_endpoint_url),
                               ("OKFGRAPH_RERANK_FAMILY", self.rerank_family),
                               ("OKFGRAPH_RERANK_API_KEY_ENV", self.rerank_api_key_env)):
                if value.strip():
                    env[key] = value.strip()
            if RERANK_MAX_TOKENS[0] <= self.rerank_max_tokens <= RERANK_MAX_TOKENS[1]:
                env["OKFGRAPH_RERANK_MAX_TOKENS"] = str(self.rerank_max_tokens)
            if self.rerank_count in RERANK_COUNTS:
                env["OKFGRAPH_RERANK_COUNT"] = self.rerank_count
        else:
            if self.rerank_device in DEVICES and self.rerank_device != "auto":
                env["OKFGRAPH_RERANK_DEVICE"] = self.rerank_device
            if (self.rerank_precision in RERANK_PRECISIONS
                    and self.rerank_precision != "auto"):
                env["OKFGRAPH_RERANK_PRECISION"] = self.rerank_precision
        if self.allow_repin:
            env["OKFGRAPH_ALLOW_REPIN"] = "1"
        return env

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


# ── the wire ───────────────────────────────────────────────────────────────


class OkfClient:
    """``POST /ops/<op>`` against one ``okf serve``, standard library only."""

    def __init__(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "http" or not parsed.hostname:
            # okf serve falls back to TCP loopback on Windows; a unix://
            # owner is reachable from POSIX only, and then not by this.
            raise OkfError("BAD_VALUE", f"not an http okf serve URL: {url!r}")
        self.url = url
        self._host = parsed.hostname
        self._port = parsed.port or 80

    def _request(self, method: str, path: str, body: Optional[bytes],
                 timeout: float) -> dict:
        conn = http.client.HTTPConnection(self._host, self._port, timeout=timeout)
        try:
            headers = {"Content-Type": "application/json"} if body is not None else {}
            conn.request(method, path, body=body, headers=headers)
            raw = conn.getresponse().read()
        except (OSError, http.client.HTTPException) as exc:
            raise OkfError("UNREACHABLE", f"okf serve at {self.url}: {exc}") from exc
        finally:
            conn.close()
        try:
            envelope = json.loads(raw or b"{}")
        except ValueError as exc:
            raise OkfError("BAD_RESPONSE", f"okf serve sent no JSON: {exc}") from exc
        return envelope if isinstance(envelope, dict) else {}

    def call(self, op: str, timeout: float = DEFAULT_CALL_TIMEOUT, **kwargs: Any) -> Any:
        """Run *op* with keyword arguments; its ``data``, or :class:`OkfError`."""
        body = json.dumps({k: v for k, v in kwargs.items() if v is not None})
        envelope = self._request("POST", f"/ops/{op}", body.encode("utf-8"), timeout)
        if envelope.get("ok"):
            return envelope.get("data")
        error = envelope.get("error") or {}
        raise OkfError(
            str(error.get("code") or "FAILED"),
            str(error.get("message") or f"{op} failed"),
            remedy=str(error.get("remedy") or ""),
            fields=error.get("fields") if isinstance(error.get("fields"), dict) else None,
        )

    def health(self, timeout: float = 5.0) -> dict:
        envelope = self._request("GET", "/health", None, timeout)
        if not envelope.get("ok"):
            raise OkfError("UNHEALTHY", f"okf serve at {self.url} did not answer /health")
        return dict(envelope.get("data") or {})

    # -- the job lane (okfgraph 0.12.0) -------------------------------------

    def job(self, job_id: str, timeout: float = DEFAULT_CALL_TIMEOUT) -> dict:
        """One job record (``GET /jobs/<id>``): status, phase progress and,
        when terminal, the op's full result."""
        envelope = self._request("GET", f"/jobs/{job_id}", None, timeout)
        if envelope.get("ok"):
            return dict(envelope.get("data") or {})
        error = envelope.get("error") or {}
        raise OkfError(
            str(error.get("code") or "BAD_VALUE"),
            str(error.get("message") or f"no job record for {job_id!r}"),
            remedy=str(error.get("remedy") or ""))

    def cancel_job(self, job_id: str,
                   timeout: float = DEFAULT_CALL_TIMEOUT) -> dict:
        """Ask the op to stop at its next batch boundary.

        This never queues behind the op lock: the flag is checked by the
        op thread itself, so cancelling a 6 h re-embed answers now.
        """
        envelope = self._request("POST", f"/jobs/{job_id}/cancel", b"{}", timeout)
        if envelope.get("ok"):
            return dict(envelope.get("data") or {})
        error = envelope.get("error") or {}
        raise OkfError(
            str(error.get("code") or "BAD_VALUE"),
            str(error.get("message") or f"cancel request for {job_id!r} failed"),
            remedy=str(error.get("remedy") or ""))

    def shutdown(self, token: str, timeout: float = 5.0) -> dict:
        """Ask the server to close the database and exit (``POST /shutdown``).

        *token* is the ``shutdown_token`` from its discovery file: only
        whoever can read that file next to the database may stop it.
        """
        body = json.dumps({"token": token}).encode("utf-8")
        envelope = self._request("POST", "/shutdown", body, timeout)
        if envelope.get("ok"):
            return dict(envelope.get("data") or {})
        error = envelope.get("error") or {}
        raise OkfError(str(error.get("code") or "FAILED"),
                       str(error.get("message") or "shutdown refused"),
                       remedy=str(error.get("remedy") or ""))


def serve_file_for(db_path: str) -> Path:
    """The discovery file ``okf serve`` writes next to the database."""
    path = Path(db_path).expanduser().resolve()
    return path.with_name(path.name + ".serve.json")


def _same_file(a: str, b: str) -> bool:
    try:
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except OSError:
        return False


def discover(db_path: str) -> Optional[str]:
    """URL of a live server owning *db_path*, or ``None``.

    Live means it answers ``/health`` for this same file. The discovery
    file alone is not trusted: a server that crashed leaves it behind.
    """
    url = str(_serve_info(db_path).get("url") or "")
    if not url.startswith("http://"):
        return None
    try:
        health = OkfClient(url).health(timeout=2.0)
    except OkfError:
        return None
    return url if _same_file(str(health.get("db_path", "")), db_path) else None


# ── where a project keeps its graph ────────────────────────────────────────

#: The folder, directly under a root, that holds a project's graph -- the
#: database and its bundles -- out of the way of the project's own files.
GRAPH_DIR = ".okfgraph"

#: OKFgraph's own default database name.
DEFAULT_DB_NAME = "okfgraph.db"


def default_db_path(root: str) -> str:
    """``<root>/.okfgraph/okfgraph.db``: where a new graph goes."""
    return str(Path(root).expanduser() / GRAPH_DIR / DEFAULT_DB_NAME)


def default_bundle_root(db_path: str) -> str:
    """``<project>/.okfgraph/bundle`` for the graph at *db_path*.

    The project is the database's folder, or the one above it when the
    database already lives in ``.okfgraph`` -- either way the bundles land
    beside the database, inside ``.okfgraph``.
    """
    parent = Path(db_path).expanduser().parent
    graph_dir = parent if parent.name == GRAPH_DIR else parent / GRAPH_DIR
    return str(graph_dir / "bundle")


def find_databases(roots: Iterable[str]) -> List[str]:
    """Every graph database under *roots*, first root first.

    Looked for in two places per root and nowhere deeper: OKFgraph's
    default ``<root>/okfgraph.db``, and any ``*.db`` in
    ``<root>/.okfgraph/`` -- that folder is the graph's own, so whatever
    database it holds is one. Other ``*.db`` files at the root are the
    project's (SQLite and the like) and are left alone.
    """
    found: List[str] = []
    for root in roots:
        base = Path(str(root)).expanduser()
        candidates = [base / DEFAULT_DB_NAME]
        try:
            candidates += sorted((base / GRAPH_DIR).glob("*.db"))
        except OSError:
            pass
        for path in candidates:
            if path.exists() and str(path) not in found:
                found.append(str(path))
    return found


# ── the server and the handle ──────────────────────────────────────────────


class OkfGraph:
    """One served graph: the handle on the ``okf_graph`` wire.

    Owned (this process started ``okf serve``) or attached (someone else
    did). Only an owned graph is ever stopped by :meth:`close`.
    """

    def __init__(self, spec: OkfSpec, url: str, *,
                 process: Optional[subprocess.Popen] = None,
                 log_path: str = "") -> None:
        self.spec = spec
        self.client = OkfClient(url)
        self.owned = process is not None
        self._process = process
        self._log_path = log_path
        self._lock = threading.Lock()
        # The process that actually holds the database. Behind an entry-
        # point launcher it is the launcher's child, so stopping has to
        # make sure of it separately.
        self._server_pid = 0
        # What lets close() ask the server to stop rather than kill it.
        # No signal reaches a server started without a console on Windows.
        self._shutdown_token = ""
        if process is not None:
            try:
                self._server_pid = int(self.client.health().get("pid") or 0)
            except (OkfError, TypeError, ValueError):
                self._server_pid = 0
            self._shutdown_token = str(_serve_info(spec.db_path).get("shutdown_token") or "")

    # -- lifecycle --------------------------------------------------------

    @classmethod
    def open(cls, spec: OkfSpec, timeout: float = START_TIMEOUT, *,
             should_stop: Optional[Callable[[], bool]] = None) -> "OkfGraph":
        """Attach to the server owning ``spec.db_path``, or start one.

        *should_stop* is polled while a new server comes up: truthy stops
        the half-started process and raises ``OkfError("CANCELLED")``.
        """
        if not spec.db_path.strip():
            raise OkfError("MISSING_PARAM", "choose a database file for the graph")
        url = discover(spec.db_path)
        if url:
            log.info(f"attaching to the okf serve that owns {spec.db_path}: {url}")
            return cls(spec, url)
        return cls._start(spec, timeout, should_stop=should_stop)

    @classmethod
    def repinned(cls, spec: OkfSpec, timeout: float = START_TIMEOUT, *,
                 should_stop: Optional[Callable[[], bool]] = None,
                 on_progress: Optional[Callable[[dict], None]] = None,
                 ) -> tuple["OkfGraph", dict]:
        """Open *spec* under the re-embed escape, re-embed, then restart
        the server cleanly (no escape) and return ``(fresh_graph, report)``.

        The escape is a declared, logged exception meant to live for
        exactly one re-embed, so the server the node leaves behind never
        runs pin-exempt. The caller closes any existing server first --
        the node's compute does (one server owns the file).

        The re-embed rides the 0.12.0 job lane: *on_progress* sees each
        job record, and *should_stop* cancels the job at its next batch
        boundary. A cancelled run still reopens the graph cleanly -- the
        report then says ``cancelled: true``, the old pin stands, and a
        re-run continues where this one stopped.
        """
        temp_spec = replace(spec, allow_repin=True)
        temp = cls._start(temp_spec, timeout, should_stop=should_stop)
        try:
            report = temp.reembed(on_progress=on_progress, should_stop=should_stop)
        except OkfError:
            temp.close()
            raise
        temp.close()
        # Never *should_stop* here: the graph must be left served, and a
        # cancel that stopped the re-embed would stop this boot as well.
        return cls.open(spec), report

    @classmethod
    def _start(cls, spec: OkfSpec, timeout: float, *,
               should_stop: Optional[Callable[[], bool]] = None) -> "OkfGraph":
        exe = spec.resolved_executable()
        if not exe:
            raise OkfError(
                "NOT_INSTALLED",
                "okf was not found: install Silk's knowledge-cpu or "
                "knowledge-gpu extra, or choose the okf executable",
            )
        Path(spec.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        # Port 0: the server binds a free port and publishes the real URL
        # in its discovery file, so two graphs never fight over 8765.
        argv = [exe, "serve", "--listen", "127.0.0.1:0", *spec.settings_args()]
        log_fd, log_path = tempfile.mkstemp(prefix="silk-okf-serve-", suffix=".log")
        flags = 0
        if os.name == "nt":
            # Its own process group, so a Ctrl+Break reaches it alone, and
            # no console window flashing up behind the canvas.
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        launched = int(time.time()) - 1
        # Rows with no CLI spelling (the external-embedder lane, the
        # re-embed escape) ride OKFGRAPH_* env; everything inherits the
        # process environment so CUDA_VISIBLE_DEVICES etc. still work.
        env = {**os.environ, **spec.launch_env()}
        with os.fdopen(log_fd, "w", encoding="utf-8") as sink:
            try:
                process = subprocess.Popen(
                    argv, stdout=sink, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, creationflags=flags, env=env,
                )
            except OSError as exc:
                # A chosen path that is gone, or not executable: the same
                # report as no okf at all, never a crash in the node.
                raise OkfError(
                    "NOT_INSTALLED", f"could not run okf at {exe}: {exc.strerror or exc}",
                    remedy="choose the okf executable, or install Silk's "
                           "knowledge-cpu or knowledge-gpu extra",
                ) from exc
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                _stop(process)
                raise OkfError("CANCELLED", "okf serve start was cancelled")
            code = process.poll()
            if code is not None:
                raise OkfError(
                    "START_FAILED",
                    f"okf serve exited with code {code}: {_tail(log_path)}",
                )
            url = _published_url(spec.db_path, launched)
            if url:
                try:
                    OkfClient(url).health(timeout=2.0)
                except OkfError:
                    pass
                else:
                    log.info(f"okf serve (pid {process.pid}) owns {spec.db_path} at {url}")
                    return cls(spec, url, process=process, log_path=log_path)
            time.sleep(0.25)
        _stop(process)
        raise OkfError("START_TIMEOUT",
                       f"okf serve did not come up in {timeout:.0f}s: {_tail(log_path)}")

    @property
    def running(self) -> bool:
        if self._process is not None:
            return self._process.poll() is None
        try:
            self.client.health(timeout=2.0)
        except OkfError:
            return False
        return True

    def close(self) -> None:
        """Stop the server if this handle started it; detach otherwise."""
        with self._lock:
            process, self._process = self._process, None
        if process is None:
            return
        if self._shutdown_token:
            try:
                self.client.shutdown(self._shutdown_token)
                process.wait(timeout=STOP_TIMEOUT)
            except OkfError as exc:
                log.warning(f"okf serve refused to shut down ({exc}); stopping it")
            except subprocess.TimeoutExpired:
                log.warning(f"okf serve (pid {process.pid}) is slow to exit; stopping it")
        _stop(process)
        server = self._server_pid
        if server and server != process.pid:
            # The launcher passes the stop on to its child; give the child
            # the same grace to close the database before insisting.
            deadline = time.monotonic() + STOP_TIMEOUT
            while _pid_alive(server) and time.monotonic() < deadline:
                time.sleep(0.1)
            if _pid_alive(server):
                try:
                    os.kill(server, signal.SIGTERM)
                except OSError as exc:
                    log.warning(f"could not stop okf serve (pid {server}): {exc}")
        if _serve_info(self.spec.db_path).get("pid") in (process.pid, server):
            # A hard stop skips the server's own cleanup; a stale file is
            # ignored by discover(), but there is no reason to leave it.
            try:
                serve_file_for(self.spec.db_path).unlink()
            except OSError:
                pass

    # -- ops --------------------------------------------------------------

    def call(self, op: str, timeout: float = DEFAULT_CALL_TIMEOUT, **kwargs: Any) -> Any:
        return self.client.call(op, timeout=timeout, **kwargs)

    def pins(self) -> dict:
        """What the vectors are pinned to, against what the server runs with."""
        return dict(self.call("pin_info", timeout=30.0) or {})

    #: Seconds between polls of ``GET /jobs/<id>`` (0.12.0 job lane).
    REEMBED_POLL = 2.0

    def reembed(self, *, dry_run: bool = False,
                on_progress=None, should_stop=None,
                poll: Optional[float] = None) -> dict:
        """Rebuild every vector under the configured model, then re-pin.

        The 0.12 report carries the per-table counts under ``reembedded``
        (``concepts`` / ``chunks``), ``seconds``, and ``pins_after`` (the
        pin view straight after the re-pin, which the node displays). A
        graph pinned to another model boots only with
        ``OKFGRAPH_ALLOW_REPIN=1`` -- :meth:`open`/`repinned` handles
        that cycle.

        0.12.0 job lane: submitted as a job (``mode: "job"``) instead of
        one blocking request — a blocking call owns the graph's op lock
        for hours and nothing could stop it, so every other call (search,
        ingest, pin_info) queued behind it. The job id answers inline;
        this client polls ``GET /jobs/<id>`` every ``poll`` seconds,
        hooks ``on_progress(record)`` per poll, and when ``should_stop()``
        answers truthy between polls cancels via
        ``POST /jobs/<id>/cancel`` — the server-side op then stops at its
        next batch boundary and the report comes back ``cancelled: true``
        with partial counts (a re-run converges the graph; per-step
        commits are the graph's own recovery contract).
        """
        client = self.client
        sub = self.call("reembed", mode="job", dry_run=dry_run) or {}
        job_id = str(sub.get("job") or "")
        if not job_id:
            # An owner older than 0.12.0 falls through to the blocking call;
            # its report shape is the same, nothing else changes.
            return dict(self.call("reembed", timeout=REEMBED_TIMEOUT,
                                  dry_run=dry_run) or {})
        wait = float(poll or self.REEMBED_POLL)
        deadline = time.time() + REEMBED_TIMEOUT
        record: Dict[str, Any] = {}
        while time.time() < deadline:
            time.sleep(wait)
            try:
                record = client.job(job_id)
            except OkfError as exc:
                # one dropped poll must never kill the loop: the job itself
                # runs on undisturbed server-side
                log.warning("reembed: poll of job %s failed (%s)", job_id, exc)
                continue
            if on_progress is not None:
                try:
                    on_progress(record)
                except Exception:            # a reporting hook never kills it
                    pass
            if record.get("status") in ("done", "cancelled", "failed"):
                break
            if should_stop is not None:
                try:
                    stop = bool(should_stop())
                except Exception:
                    stop = False
                if stop:
                    try:
                        client.cancel_job(job_id)
                    except OkfError:
                        pass
        else:
            # the poll window closed and the job never finished:
            # cancel it out of the server so it cannot queue the graph lock
            try:
                client.cancel_job(job_id)
            except OkfError:
                pass
            raise OkfError(
                "TIMEOUT",
                f"the re-embed did not finish within its "
                f"{int(REEMBED_TIMEOUT // 3600)} h window; job {job_id} "
                "was cancelled",
                remedy="re-run okf reembed later; the graph recovers")
        if record.get("status") == "failed":
            err = record.get("error") or {}
            raise OkfError(
                str(err.get("code") or "INTERNAL"),
                str(err.get("message") or "the re-embed job failed"),
                remedy=str(err.get("remedy") or ""))
        report = dict(record.get("result") or {})
        report.setdefault("job", job_id)
        return report

    def describe(self) -> str:
        who = f"pid {self._process.pid}" if self._process is not None else "attached"
        return f"{Path(self.spec.db_path).name} at {self.client.url} ({who})"


def mismatch_text(pins: dict) -> str:
    """One line naming each pin the configured settings would break, or ''."""
    fields = [str(f) for f in (pins.get("mismatch") or ())]
    if not fields or pins.get("empty"):
        return ""
    pinned = pins.get("pinned") or {}
    configured = pins.get("configured") or {}
    parts = []
    for name in fields:
        want = configured.get(f"expected_{name}", configured.get(name))
        parts.append(f"{name}: graph has {pinned.get(name)!s}, settings give {want!s}")
    return "; ".join(parts)


def model_cache(spec: OkfSpec, timeout: float = 60.0) -> dict:
    """``okf model-info`` for *spec*'s embedding model, or ``{}``.

    Offline by construction (embroider's cache walker): whether the files
    ``okf serve`` would open are in the hub cache (``cached``), which repo
    and precision that resolves to, ``cache_dir`` and ``disk_usage_bytes``.
    Only the embroider lane has a cache to ask about; ``{}`` too when okf
    cannot be run or answers with an error.
    """
    exe = spec.resolved_executable()
    if not exe or spec.backend != "embroider":
        return {}
    argv = [exe, "model-info", *spec.settings_args(), "--json"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, stdin=subprocess.DEVNULL,
            env={**os.environ, **spec.launch_env()}, creationflags=flags)
        envelope = json.loads(done.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    data = envelope.get("data") if isinstance(envelope, dict) and envelope.get("ok") else None
    return dict(data) if isinstance(data, dict) else {}


def _published_url(db_path: str, since: int) -> str:
    """The URL a server started at or after *since* published for *db_path*.

    Matched by time, not by pid: ``okf.exe`` is an entry-point launcher, and
    the pid in the file is the Python process it started, not the launcher
    ``Popen`` knows about.
    """
    info = _serve_info(db_path)
    if int(info.get("started_at") or 0) < since:
        return ""
    return str(info.get("url") or "")


def _serve_info(db_path: str) -> dict:
    """The discovery file for *db_path*, or ``{}`` if there is none to read."""
    try:
        info = json.loads(serve_file_for(db_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def _stop(process: subprocess.Popen) -> None:
    """Ask the server to shut down, and insist only if it will not.

    A graceful stop lets it close the database and remove its discovery
    file. On Windows the request is Ctrl+Break to its process group.
    """
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGTERM)
        process.wait(timeout=STOP_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        process.kill()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            log.warning(f"okf serve (pid {process.pid}) did not exit")


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        # os.kill(pid, 0) is not a probe on Windows: ask the kernel.
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _tail(path: str, limit: int = 600) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "(no output)"
    return text[-limit:] or "(no output)"


# ── the ledger riding the graph's pinned encoder (0.12) ───────────────────


class OkfEmbedder(Embedder):
    """The inverse lane: the ledger's vectors come from the graph's OWN
    pinned embed engine via ``POST /ops/embed``.

    One model host instead of two: the vectors this produces are encoded
    by the very engine okfgraph has already pinned the graph to, so they
    land in the same space its search index lives in. Provenance is the
    server's answer, not a guess: ``dim`` rides the first vector and
    ``name`` is the ``model_id`` okfgraph reports (Macrame-slugified by
    the base class). The optional-half rule is inherited from
    :class:`Embedder` -- the first failed call disables this embedder,
    once, out loud, and memory stays keyword search.
    """

    # A hung server must not hold an agent turn open much beyond the
    # embedder's own budget; the call also has to survive a server batch.
    EMBED_OP_TIMEOUT = 60.0

    def __init__(self, graph: OkfGraph, name: str = "") -> None:
        super().__init__(name or graph.spec.model_id or "okf")
        self._graph = graph

    def _vector(self, text: str) -> list[float]:
        data = self._graph.client.call(
            "embed", timeout=OkfEmbedder.EMBED_OP_TIMEOUT,
            texts=[text], task="Document",
        )
        vectors = (data or {}).get("vectors")
        if not vectors:
            raise RuntimeError("the /ops/embed answer carried no vectors")
        return [float(x) for x in vectors[0]]

    @property
    def model_id(self) -> str:
        """What okfgraph says it pinned, straight from the source."""
        try:
            pins = self._graph.client.call("pin_info", timeout=10.0) or {}
        except OkfError:
            return self.spec_model()
        pinned = pins.get("pinned") or {}
        return str(pinned.get("model_id") or self.spec_model())

    def spec_model(self) -> str:
        return self._graph.spec.model_id or "okf"


EMBED_OP_TIMEOUT = 60.0
