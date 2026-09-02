# -*- coding: utf-8 -*-
"""Server-based GGUF model pool.

Replaces the in-memory multiple-``Llama`` pool with a single background
``llama_cpp.server`` process (LM Studio-like): the model weights load exactly
once, and every agent talks to it over the local OpenAI-compatible HTTP API.

The server is configured through a generated JSON **config file** (not per-CLI
flags), so the full set of model settings — GPU layers, context, threads, seed,
flash-attention, mmap, KV-cache quant — is forwarded robustly across
``llama-cpp-python`` versions without guessing flag names or bool spellings.

``checkout()`` hands agents an :class:`OpenAIClientMock` that mimics the
``llama_cpp.Llama`` API (``create_chat_completion``) the GraphEngine/AgentLoop
already speak, translating calls into HTTP + SSE.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import socket
import sys
import subprocess
import tempfile
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

from weave.logger import get_logger

from .prefix_stats import LogDrain, PrefixMeter

log = get_logger("SilkModelPool")

# ── Dependency guards ────────────────────────────────────────────────────────
# The loader/tests import these to degrade gracefully. LLAMA_CPP_AVAILABLE is the
# base library; LLAMA_SERVER_AVAILABLE is the ``[server]`` extra
# (fastapi/uvicorn/sse-starlette) that the background process needs.
try:
    from llama_cpp import Llama  # noqa: F401  (re-exported for back-compat)
    LLAMA_CPP_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means "not usable"
    Llama = None  # type: ignore[assignment,misc]
    LLAMA_CPP_AVAILABLE = False

# The ``[server]`` extra ships the module files but NOT its third-party deps, so
# checking find_spec("llama_cpp.server") alone is not enough (the subprocess would
# still crash importing fastapi). Verify the deps the server imports at startup.
_SERVER_DEPS = (
    "fastapi", "uvicorn", "sse_starlette", "starlette_context", "pydantic_settings",
)


def _missing_server_deps() -> list[str]:
    if not LLAMA_CPP_AVAILABLE:
        return ["llama-cpp-python"]
    try:
        if importlib.util.find_spec("llama_cpp.server") is None:
            return ["llama_cpp.server"]
    except Exception:  # noqa: BLE001
        return ["llama_cpp.server"]
    missing: list[str] = []
    for dep in _SERVER_DEPS:
        try:
            if importlib.util.find_spec(dep) is None:
                missing.append(dep)
        except Exception:  # noqa: BLE001
            missing.append(dep)
    return missing


#: Third-party modules the server needs but which are absent (empty = all present).
LLAMA_SERVER_MISSING = _missing_server_deps()
LLAMA_SERVER_AVAILABLE = not LLAMA_SERVER_MISSING


def server_missing_deps_message() -> str:
    """Actionable message naming what to install for the server to run."""
    return (
        "llama_cpp.server cannot run — missing: "
        f"{', '.join(LLAMA_SERVER_MISSING)}. Install the server extra: "
        "pip install 'llama-cpp-python[server]'"
    )

#: ModelSettings fields forwarded from the loader's llama_kwargs to the server
#: config (verified against llama_cpp.server.settings.ModelSettings). Keys absent
#: or None in llama_kwargs are simply omitted, so server defaults apply.
_SERVER_MODEL_KEYS = (
    "n_ctx", "n_gpu_layers", "n_threads", "n_threads_batch", "n_batch",
    "seed", "use_mmap", "use_mlock", "flash_attn", "type_k", "type_v", "verbose",
)

#: How long to wait for the server to answer ``/v1/models`` before giving up.
_READY_TIMEOUT_S = 120.0
#: How long the server may take to terminate before it is killed.
_SHUTDOWN_TIMEOUT_S = 5.0


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OpenAIClientMock:
    """Proxy that mimics the ``llama_cpp.Llama`` API used by GraphEngine, routing
    ``create_chat_completion`` to the local server over HTTP (+ SSE for streams).
    """

    def __init__(self, base_url: str, model_alias: str = "default") -> None:
        self.base_url = base_url
        self.model_alias = model_alias

    def create_chat_completion(
        self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs: Any,
    ) -> Any:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model_alias, "messages": messages, "stream": stream,
        }
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        # NOTE: do NOT wrap the streaming response in `with` — returning the
        # generator would exit the context and close the response *before* the
        # caller iterates it (yielding zero tokens). The generator owns the
        # response lifecycle and closes it in its finally.
        try:
            response = urllib.request.urlopen(req)
        except urllib.error.URLError as exc:
            detail = ""
            if hasattr(exc, "read"):
                try:
                    detail = exc.read().decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    pass
            log.error(f"Llama server request failed: {detail or exc}")
            raise RuntimeError(f"Failed to reach local Llama server: {exc}") from exc

        if not stream:
            try:
                return json.loads(response.read().decode("utf-8"))
            finally:
                response.close()

        def generator() -> Any:
            try:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        pass
            finally:
                response.close()

        return generator()

    def tokenize(self, text: bytes) -> list[int]:
        # Approximation only — GraphEngine estimates prompt tokens itself and
        # never checks out a client just to count, so an exact server round-trip
        # here would be wasted. ~4 chars/token is close enough for budgeting.
        body = text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text)
        return [0] * max(1, len(body) // 4)

    def reset(self) -> None:
        pass


class GGUFModelPool:
    """Single background ``llama_cpp.server`` process, shared by all agents."""

    def __init__(
        self,
        model_path: str,
        n_instances: int = 4,
        clear_on_return: bool = True,
        **llama_kwargs: Any,
    ) -> None:
        if not LLAMA_SERVER_AVAILABLE:
            raise RuntimeError(server_missing_deps_message())

        self._lock = threading.RLock()
        self._max_instances = n_instances  # display only (single shared server)
        self._clear_on_return = clear_on_return
        self._model_path = model_path
        self._model_alias = "default"
        self._host = "127.0.0.1"
        self._port = find_free_port()
        self._server_url = f"http://{self._host}:{self._port}/v1"
        self._process: Optional[subprocess.Popen] = None
        self._config_path: Optional[str] = None
        self._log_path: Optional[str] = None
        self._log_handle = None
        self._participant = None
        # The context window the server was actually started with. The one
        # number a context budget is a fraction of; kept here because this
        # is where it stops being a request and becomes a fact (G14c).
        self.context_length: Optional[int] = (
            int(llama_kwargs["n_ctx"]) if llama_kwargs.get("n_ctx") else None
        )

        # Build the server config from the requested settings (robust: JSON, so
        # bools/ints serialize correctly and no CLI flag names are guessed).
        model_settings: Dict[str, Any] = {
            "model": model_path, "model_alias": self._model_alias,
        }
        for key in _SERVER_MODEL_KEYS:
            value = llama_kwargs.get(key)
            if value is not None:
                model_settings[key] = value
        # interrupt_requests defaults to True in llama_cpp.server, and it
        # does not mean what it sounds like: while one request streams, the
        # arrival of a second one *truncates the first* with a well-formed
        # [DONE]. The reader cannot tell that from a natural stop, so an
        # agent reasons over half an answer whenever two agents share the
        # server — which is the normal case here (spec D43). Turned off; the
        # missing-finish_reason check in GraphEngine stays anyway, because a
        # remote backend can truncate for its own reasons.
        config = {
            "host": self._host,
            "port": self._port,
            "interrupt_requests": False,
            "models": [model_settings],
        }

        fd, self._config_path = tempfile.mkstemp(prefix="silk-llama-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle)
        # Capture server stderr to a file so a failed start reports *why*.
        self._log_path = self._config_path + ".log"
        self._log_handle = open(self._log_path, "w", encoding="utf-8")
        # That same file is where the backend already reports, per request,
        # how much of the prompt it had cached. Reading it is the whole of
        # the D41 measurement -- nothing is added to the model path.
        self._prefix_meter = PrefixMeter()
        self._prefix_drain = LogDrain(self._log_path)
        self._prefix_lock = threading.Lock()

        cmd = [sys.executable, "-m", "llama_cpp.server",
               "--config_file", self._config_path]
        log.info(
            f"GGUF server starting on {self._server_url} "
            f"(model {Path(model_path).name})"
        )
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=self._log_handle,
        )

        self._wait_until_ready()  # raises (and cleans up) on failure
        self._client = OpenAIClientMock(self._server_url, self._model_alias)
        # Which conversations are bound to this server, by session id.
        # A *set*, not a counter: `checkout` runs once per request, so a
        # counter measured requests-ever and called them sessions -- it
        # only ever grew, and the Pool Monitor showed that growth as bound
        # conversations (spec D47).
        self._bound_sessions: set[str] = set()
        self._register_for_shutdown()

    # -- shutdown registry ------------------------------------------------

    def _register_for_shutdown(self) -> None:
        """Let the process release this server even if no node does.

        ``cleanup`` used to be reachable only through the loader node —
        so a quit that never deleted the node, or a relaunch that spawns
        a replacement, left a ``llama_cpp.server`` holding VRAM and a
        port with nothing left to shut it down.  Registering here fixes
        that at the only place that knows the server is actually up, and
        the registry's reverse order releases it before the things it
        was built underneath.
        """
        try:
            from weave.engine.shutdown import (
                get_shutdown_registry, install_shutdown_handlers,
            )
        except Exception:  # noqa: BLE001 - a plugin must not need the host
            self._participant = None
            return
        install_shutdown_handlers()
        self._participant = get_shutdown_registry().register(
            f"GGUF server :{self._port}", self._release, force=self._force,
            timeout_s=_SHUTDOWN_TIMEOUT_S,
        )

    def _release(self, timeout: float = _SHUTDOWN_TIMEOUT_S) -> bool:
        """The registry's graceful tier: terminate and wait."""
        self.cleanup(timeout=timeout)
        process = self._process
        return process is None or process.poll() is not None

    def _force(self) -> None:
        """The registry's forceful tier: the server is holding VRAM."""
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except Exception:  # noqa: BLE001
                log.exception("Could not kill the GGUF server")

    # -- lifecycle --------------------------------------------------------

    def _read_log_tail(self, limit: int = 2000) -> str:
        try:
            if self._log_handle is not None:
                self._log_handle.flush()
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()[-limit:].strip()
        except Exception:  # noqa: BLE001
            return ""

    def _wait_until_ready(self, timeout: float = _READY_TIMEOUT_S) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                tail = self._read_log_tail()
                self.cleanup()
                raise RuntimeError(
                    f"llama_cpp.server exited during startup "
                    f"(code {self._process.returncode}).\n{tail}"
                )
            try:
                with urllib.request.urlopen(
                    f"{self._server_url}/models", timeout=2.0
                ) as response:
                    if response.status == 200:
                        log.info("GGUF server ready.")
                        return
            except Exception:  # noqa: BLE001 - not up yet; keep polling
                time.sleep(0.5)
        tail = self._read_log_tail()
        self.cleanup()
        raise RuntimeError(
            f"llama_cpp.server did not become ready within {timeout:.0f}s.\n{tail}"
        )

    def cleanup(self, timeout: float = _SHUTDOWN_TIMEOUT_S) -> None:
        """Stop the server and free its VRAM.  Safe to call twice."""
        log.info("GGUF server: shutting down process and freeing VRAM.")
        self._unregister()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            if self._log_handle is not None:
                try:
                    self._log_handle.close()
                except Exception:  # noqa: BLE001
                    pass
                self._log_handle = None
            for path in (self._config_path, self._log_path):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            self._config_path = self._log_path = None
        gc.collect()

    def _unregister(self) -> None:
        """Drop the registry handle: a node that ejected the model has
        already released this server, and a participant whose resource is
        gone would report a failure at quit."""
        participant = getattr(self, "_participant", None)
        if participant is None:
            return
        self._participant = None
        try:
            from weave.engine.shutdown import get_shutdown_registry
            get_shutdown_registry().unregister(participant)
        except Exception:  # noqa: BLE001
            pass

    # -- checkout API (kept for GraphEngine; a shared client, not a slot) --

    def register_instance(self, instance: Any) -> None:
        pass

    def add_idle(self, instance: Any) -> None:
        pass

    def checkout(self, session_id: str = "default") -> Optional[Any]:
        with self._lock:
            self._bound_sessions.add(str(session_id))
            log.debug(f"Checkout: session {session_id[:8]}… → server client")
            return self._client

    def checkin(
        self, instance: Any, session_id: str = "default",
        release_session: bool = False,
    ) -> None:
        """Return the client. Idempotent, because a set has no double-free.

        A checkin only *unbinds* when asked to: a run ends every round with
        one, and the conversation it belongs to is still live.
        """
        if release_session:
            self.release_session(session_id)

    def release_session(self, session_id: str = "default") -> bool:
        """Forget a conversation; returns whether it was bound.

        The public way to do what `Clear Context` wants. It used to reach
        into a `_session_instances` dict that this pool does not have --
        an AttributeError swallowed by a broad except, so the release never
        happened and the count it was meant to correct kept climbing.
        """
        with self._lock:
            known = str(session_id) in self._bound_sessions
            self._bound_sessions.discard(str(session_id))
        if known:
            log.debug(f"Released session {str(session_id)[:8]}…")
        return known

    @property
    def bound_sessions(self) -> int:
        """How many distinct conversations are bound to this server."""
        with self._lock:
            return len(self._bound_sessions)

    # -- prefix-reuse measurement (spec D41/D47) --------------------------

    def begin_request(self, session_id: str = "default") -> None:
        """Drop log lines written before this request starts.

        Attribution is sequential because the server is: every request goes
        through ``llama_outer_lock``, so what appears between begin and end
        belongs to the request in between (spec D43/D53).
        """
        with self._prefix_lock:
            self._prefix_drain.drain()

    def end_request(
        self, session_id: str = "default", wall_s: Optional[float] = None
    ) -> None:
        """Fold this request's log lines into the meter."""
        with self._prefix_lock:
            lines = self._prefix_drain.drain()
            self._prefix_meter.record_lines(
                lines, session=session_id, wall_s=wall_s
            )

    def prefix_report(self) -> dict:
        """D47's three numbers so far; ``None`` where there is no data."""
        with self._prefix_lock:
            return self._prefix_meter.report().as_dict()

    def reset_prefix_stats(self) -> None:
        with self._prefix_lock:
            self._prefix_meter.reset()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "model_path": str(Path(self._model_path).name),
                "full_path": self._model_path,
                "capacity": self._max_instances,
                "total_instances": 1,
                "bound_sessions": len(self._bound_sessions),
                "idle": 0,
                # KV usage isn't exposed by the server; reported as 0 (the loader
                # shows the bar as informational only).
                "kv_used_tokens": 0,
                "kv_total_tokens": 0,
                "kv_fill_pct": 0.0,
                "clear_on_return": self._clear_on_return,
                # The number the context design hangs on, surfaced where the
                # pool is already being watched (D41; G15).
                "prefix_reuse": self.prefix_report(),
            }
