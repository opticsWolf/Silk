# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

bobine's warm converter behind HTTP, speaking Silk's settings dict.

Run by :mod:`functions.bobine_pool` as a plain script in a child process
(``import bobine`` loads ONNX Runtime, which must not happen in Weave's
process). It reuses ``bobine.serve``'s handler and pool -- the routes,
the NDJSON page lines, the idle sweep -- and supplies the one thing that
host cannot do for Silk: translate a settings *dict* into a
``ConverterConfig``.

``python -m bobine.serve`` builds ``ConverterConfig(**settings)`` itself,
which is exactly right for a caller that sends enum objects and useless
over JSON: ``routing_mode``, ``table_backend`` and ``image_mode`` must
be enum instances, and Silk's wire carries words (``surgical``,
``potatr``, ``link``) plus its own key names. So this script keeps the
same request/response shapes and turns the dict into the kwargs via
:func:`bobine_worker.config_kwargs`, the function the per-call worker
uses too.

Settings are the stack vocabulary (bobine 0.9): ``device``
``auto|cpu|cuda``, ``precision`` ``auto|fp32|fp16|int8``. The warm pool
keeps one converter per distinct settings key, releases it after
``--idle-seconds`` without use, and serialises requests that share a key
-- bobine's own behaviour, unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from typing import Any, Dict, Optional

import bobine
import bobine.serve as host

from bobine_convert import clean_settings
from bobine_worker import config_kwargs


class SilkPool(host.Pool):
    """bobine's pool, keyed by Silk's cleaned settings and built from them."""

    def acquire(self, settings: Optional[Dict[str, Any]] = None) -> Any:
        cleaned = clean_settings(settings)
        key = json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self._closed:
                raise RuntimeError("pool is shut down")
            warm = self._warm.get(key)
            if warm is None:
                config = bobine.ConverterConfig(**config_kwargs(bobine, cleaned))
                warm = host.Warm(
                    dict(cleaned),
                    self._factory(config, cache_dir=self.cache_dir),
                )
                self._warm[key] = warm
            warm.last_used = self._clock()
            return warm


def serve(host_ip: str = "127.0.0.1", port: int = 8757,
          cache_dir: Optional[str] = None,
          idle_seconds: float = 300.0) -> None:
    """bobine.serve's loop, with :class:`SilkPool` in place of its pool."""
    pool = SilkPool(cache_dir=cache_dir, idle_seconds=idle_seconds)
    handler = type("BoundHandler", (host.Handler,), {"pool": pool})
    try:
        httpd = host.Server((host_ip, port), handler)
    except OSError as exc:
        sys.stderr.write(f"silk bobine host: cannot bind {host_ip}:{port}: {exc}\n")
        raise SystemExit(3) from exc

    stop = threading.Event()

    def sweep() -> None:
        interval = max(0.05, min(5.0, idle_seconds / 4 or 0.05))
        while not stop.wait(interval):
            pool.release_idle()

    threading.Thread(target=sweep, name="bobine-idle", daemon=True).start()
    sys.stderr.write(f"listening http://{host_ip}:{port}\n")
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()
        pool.close()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="silk-bobine-host",
        description="One warm bobine converter behind HTTP, for Silk.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8757)
    parser.add_argument("--cache-dir", default=None,
                        help="converter-model hub cache (default: bobine's)")
    parser.add_argument("--idle-seconds", type=float, default=300.0,
                        help="release a warm converter after this long unused")
    args = parser.parse_args(argv)
    if args.port < 0 or args.port > 65535:
        parser.error("--port must be between 0 and 65535")
    serve(args.host, args.port, args.cache_dir, max(0.0, args.idle_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
