# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

A warm bobine converter on a loopback port (``functions/bobine_server.py``).

The document workers spawn one child per conversion, which is exact for
cancel and cheap for a file or two. A folder of PDFs pays bobine's model
loads over and over; this pool pays them once. It is bobine's own
``python -m bobine.serve`` shape -- one converter per settings key, kept
warm, released when idle, requests for one key serialised -- run through
Silk's small launcher so the settings dict and the stack vocabulary
arrive intact (see ``bobine_server.py`` for why ``bobine.serve`` itself
cannot take them).

:class:`~functions.rerank_pool.ChildServerPool` is the process half, the
same one the reranker and the ONNX host use: a free port, a health
check, the shutdown registry. ``base_url`` is the host root; the routes
are bobine's ``POST /v1/convert`` (NDJSON: a ``{"page", "total"}`` line
per page, then the result), ``GET /v1/models`` and ``GET /health``.

**Cancel stops the host.** A shared child cannot abort one conversion
(bobine's pool serialises per key and the native converter is not
interruptible), so stopping mid-document kills the process, exactly as
the per-call worker does. The next conversion starts a fresh host; the
warmth is lost, the cancel is honest.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .bobine_convert import ConversionCancelled, clean_settings
from .rerank_pool import ChildServerPool, child_command

#: The launcher script, run by this checkout's interpreter.
SERVER_SCRIPT = str(Path(__file__).with_name("bobine_server.py"))

#: How long a socket read waits for the next line (a page or the result).
READ_TIMEOUT_S = 900.0


def server_args(cache_dir: str = "", idle_seconds: float = 300.0) -> list:
    """The launcher's argv, minus ``--host``/``--port`` (the pool adds them)."""
    args = ["--idle-seconds", str(float(idle_seconds))]
    if cache_dir:
        args += ["--cache-dir", str(cache_dir)]
    return args


def pool_command(cache_dir: str = "", idle_seconds: float = 300.0) -> list:
    """The whole argv a host starts with, for the tests to pin."""
    return child_command(server_args(cache_dir, idle_seconds),
                         "127.0.0.1", 0, script=SERVER_SCRIPT)


class BobinePool(ChildServerPool):
    """One warm bobine host, started on first use and stopped on cleanup."""

    def __init__(self, cache_dir: str = "", idle_seconds: float = 300.0) -> None:
        self.cache_dir = str(cache_dir or "")
        self.idle_seconds = float(idle_seconds)
        super().__init__("bobine", SERVER_SCRIPT,
                         server_args(self.cache_dir, self.idle_seconds),
                         "bobine host", log_prefix="silk-bobine-")

    # -- requests ----------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]):
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method="POST",
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=READ_TIMEOUT_S)

    def convert(self, path: str, output_dir: str, settings: Any = None, *,
                on_page: Optional[Callable[[int, int], None]] = None,
                should_continue: Optional[Callable[[], bool]] = None
                ) -> Dict[str, Any]:
        """Convert one file; the shape :func:`bobine_convert.convert_file`
        returns."""
        cleaned = clean_settings(settings)
        try:
            response = self._post("/v1/convert", {
                "path": str(path), "output_dir": str(output_dir),
                "settings": cleaned})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[-600:]
            raise RuntimeError(f"bobine host refused the request: {detail}") from exc
        except OSError as exc:
            raise RuntimeError(f"the bobine host is not answering: {exc}") from exc

        final: Optional[Dict[str, Any]] = None
        with response:
            for raw in response:
                if should_continue is not None and not should_continue():
                    # A shared child cannot abort one conversion: stop it.
                    self.cleanup()
                    raise ConversionCancelled("conversion stopped")
                try:
                    message = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                if "page" in message:
                    if on_page is not None:
                        on_page(int(message["page"]),
                                int(message.get("total") or 0))
                else:
                    final = message
        if final is None:
            raise RuntimeError("the bobine host ended without a result")
        if final.get("error"):
            raise RuntimeError(str(final["error"]))
        md_path = str(final.get("md_path") or "")
        markdown = ""
        if md_path:
            try:
                markdown = Path(md_path).read_text(encoding="utf-8")
            except OSError:
                pass
        return {"source": str(path), "md_path": md_path,
                "markdown": markdown,
                "pages": int(final.get("page_count") or 0),
                "images": int(final.get("image_count") or 0),
                "image_dir": str(final.get("image_dir") or ""),
                "data_files": [], "describe": str(final.get("describe") or "")}

    def models(self) -> Dict[str, Any]:
        """bobine's model cache and the warm converters (``GET /v1/models``)."""
        with urllib.request.urlopen(f"{self.base_url}/v1/models",
                                    timeout=30.0) as response:
            return json.loads(response.read().decode("utf-8"))
