# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

A GGUF reranker served on ``/v1/rerank`` by a child process.

The rerank twin of :class:`functions.model_pool.GGUFModelPool`: one
background process per model (``functions/rerank_server.py``, run as a
script), started on a free loopback port, waited on until it answers,
released through Weave's shutdown registry so a quit never leaves it
holding RAM or VRAM. Wrapped in a
:class:`~functions.model_pool.LazyGGUFPool` it starts on first use: the
Reranker Loader hands it out at once and the Knowledge Graph starts it
when it serves a graph.

Its consumer is OKFgraph's external rerank lane, which posts to
``<base_url>/v1/rerank`` and, for exact pair counts, ``<base_url>/tokenize``
-- so ``base_url`` here is the host root, not an OpenAI ``.../v1`` base.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from weave.logger import get_logger

from .model_pool import LLAMA_CPP_AVAILABLE, find_free_port

log = get_logger("SilkRerankPool")

#: The script the child runs. A file path, not ``-m``: the child must not
#: need Silk's package identity, only llama-cpp-python.
SERVER_SCRIPT = str(Path(__file__).with_name("rerank_server.py"))

_READY_TIMEOUT_S = 120.0
_SHUTDOWN_TIMEOUT_S = 10.0


class RerankPool:
    """One running reranker server."""

    def __init__(self, model_path: str, n_ctx: int = 2048,
                 n_gpu_layers: int = 0, n_threads: Optional[int] = None,
                 use_mmap: bool = True) -> None:
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python is not installed: pip install llama-cpp-python")
        self._model_path = model_path
        self._lock = threading.Lock()
        self._host = "127.0.0.1"
        self._port = find_free_port()
        self._url = f"http://{self._host}:{self._port}"
        self.context_length = int(n_ctx)
        self._participant: Any = None
        fd, self._log_path = tempfile.mkstemp(prefix="silk-rerank-", suffix=".log")
        os.close(fd)
        self._log_handle: Any = open(self._log_path, "w", encoding="utf-8")
        cmd = [sys.executable, SERVER_SCRIPT, "--model", model_path,
               "--host", self._host, "--port", str(self._port),
               "--n_ctx", str(int(n_ctx)), "--n_gpu_layers", str(int(n_gpu_layers))]
        if n_threads:
            cmd += ["--n_threads", str(int(n_threads))]
        if not use_mmap:
            cmd.append("--no_mmap")
        log.info(f"Reranker server starting on {self._url} "
                 f"(model {Path(model_path).name})")
        self._process: Optional[subprocess.Popen] = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=self._log_handle)
        self._wait_until_ready()
        self._register_for_shutdown()

    # -- what the Knowledge Graph reads -----------------------------------

    @property
    def base_url(self) -> str:
        """The host root; okf appends ``/v1/rerank`` and ``/tokenize``."""
        return self._url

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def model_alias(self) -> str:
        return Path(self._model_path).stem

    def snapshot(self) -> dict:
        process = self._process
        return {"model_path": Path(self._model_path).name,
                "full_path": self._model_path, "url": self._url,
                "started": process is not None and process.poll() is None}

    # -- lifecycle ----------------------------------------------------------

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
            process = self._process
            if process is not None and process.poll() is not None:
                tail = self._read_log_tail()
                self.cleanup()
                raise RuntimeError(
                    f"the reranker server exited during startup "
                    f"(code {process.returncode}).\n{tail}")
            try:
                with urllib.request.urlopen(f"{self._url}/health", timeout=2.0) as r:
                    if r.status == 200:
                        log.info("Reranker server ready.")
                        return
            except Exception:  # noqa: BLE001 - not up yet
                time.sleep(0.5)
        tail = self._read_log_tail()
        self.cleanup()
        raise RuntimeError(
            f"the reranker server did not answer within {timeout:.0f}s.\n{tail}")

    def _register_for_shutdown(self) -> None:
        try:
            from weave.engine.shutdown import (
                get_shutdown_registry, install_shutdown_handlers,
            )
        except Exception:  # noqa: BLE001 - a plugin must not need the host
            return
        install_shutdown_handlers()
        self._participant = get_shutdown_registry().register(
            f"Reranker server :{self._port}", self._release, force=self._force,
            timeout_s=_SHUTDOWN_TIMEOUT_S)

    def _release(self, timeout: float = _SHUTDOWN_TIMEOUT_S) -> bool:
        self.cleanup(timeout=timeout)
        process = self._process
        return process is None or process.poll() is not None

    def _force(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()

    def cleanup(self, timeout: float = _SHUTDOWN_TIMEOUT_S) -> None:
        """Stop the server.  Safe to call twice."""
        participant, self._participant = self._participant, None
        if participant is not None:
            try:
                from weave.engine.shutdown import get_shutdown_registry
                get_shutdown_registry().unregister(participant)
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
            if self._log_handle is not None:
                try:
                    self._log_handle.close()
                except Exception:  # noqa: BLE001
                    pass
                self._log_handle = None
                try:
                    os.unlink(self._log_path)
                except OSError:
                    pass
