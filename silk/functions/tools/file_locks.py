"""
Process-wide per-path file locks.

The file tools run synchronously inside ``asyncio.to_thread``, so when
``ToolBox.execute_tool_calls_async`` fires a batch in parallel, two writes to the
same file execute in different worker *threads*. Atomic writes stop the file from
being corrupted, but a read-modify-write pair (edit_file, insert_text) could still
lose an update â€” both threads read the original, and the second os.replace clobbers
the first.

This module serialises those operations with a registry of ``threading.Lock``s
keyed by resolved path (``threading.Lock`` â€” not ``asyncio.Lock`` â€” because the
contention is between threads). The registry is module-global so the guarantee
holds even across multiple FileToolSandbox instances pointing at the same files.

Locks are acquired in canonical (sorted) order so the two-path operations
(copy_file, move_file) can't deadlock against each other.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_registry: dict[str, threading.Lock] = {}
_registry_guard = threading.Lock()


def _get_lock(key: str) -> threading.Lock:
    with _registry_guard:
        lock = _registry.get(key)
        if lock is None:
            lock = threading.Lock()
            _registry[key] = lock
        return lock


@contextmanager
def lock_paths(*paths: "str | Path | None") -> Iterator[None]:
    """Acquire write locks for *paths* (deduped, canonical order), release on exit."""
    keys = sorted({str(Path(p).resolve()) for p in paths if p is not None})
    locks = [_get_lock(k) for k in keys]
    acquired: list[threading.Lock] = []
    try:
        for lk in locks:
            lk.acquire()
            acquired.append(lk)
        yield
    finally:
        for lk in reversed(acquired):
            lk.release()
