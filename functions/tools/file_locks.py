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

Tier 2 -- the per-root gate (spec D67, closes G19)
--------------------------------------------------
Per-path locks cover the file tools and nothing else. A toolchain
subprocess that rewrites files -- ``ruff format``, ``cargo fmt``,
``run_python``, which can write anything -- never sees ``lock_paths``, and
its ``sequential=True`` flag orders execution only *within one agent's
batch*. Across agents nothing serialised them at all: agent A's formatter
could interleave with agent B's ``edit_file`` on the same tree, and neither
was told.

So there is a second tier, the same registry pattern one level up: a
readers-writer gate per **registered sandbox root**.

* File tools take the gate **shared** for every registered root containing
  the target, then the per-path locks. Among themselves their behaviour is
  unchanged -- shared never excludes shared.
* A subprocess that may write takes the gate **exclusive** for its root, for
  the subprocess's whole duration. Coarse on purpose: nothing can know which
  files a subprocess will touch, and a formatter run is short.
* Read-only subprocesses (``--no-fix`` lints, mypy, radon) take nothing.

Ordering rule, extending tier 1's: **root gates before path locks, both in
sorted canonical order.** Where registered roots nest, a writer takes the
gate of every registered root in an ancestor/descendant relation with its
own, so a write under ``/project`` and a write under ``/project/sub``
cannot proceed at once.

Scope, stated once (spec D68): the gate is **advisory and per-process**. It
protects cooperating tools. It cannot bind an external editor, a second
Weave process, or an MCP server with its own file access, and it is held
per *operation*, never per turn -- "this file is mine for the task" is
ownership, which is a claim in the task ledger, not a lock.
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
    """Acquire write locks for *paths* (deduped, canonical order), release on exit.

    Takes the containing roots' gates **shared** first, so a file write
    cannot land in the middle of a subprocess that holds a root exclusively
    (spec D67). Shared does not exclude shared, so file tools among
    themselves behave exactly as before.
    """
    keys = sorted({str(Path(p).resolve()) for p in paths if p is not None})
    with _shared_roots(keys):
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


# -- tier 2: the per-root readers-writer gate ------------------------------


class _RootGate:
    """A readers-writer gate over one sandbox root.

    Writer-preferring: a waiting writer blocks new readers, so a stream of
    file-tool calls cannot starve a formatter indefinitely. Not reentrant --
    nothing here nests, and a gate that quietly allowed it would hide the
    one case that must not happen (a writer waiting on itself).
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    def acquire_shared(self) -> None:
        with self._cond:
            while self._writer or self._waiting_writers:
                self._cond.wait()
            self._readers += 1

    def release_shared(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_exclusive(self) -> None:
        with self._cond:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._cond.wait()
            finally:
                self._waiting_writers -= 1
            self._writer = True

    def release_exclusive(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()


_roots: set[str] = set()
_root_gates: dict[str, _RootGate] = {}


def register_root(root: "str | Path") -> str:
    """Register a sandbox root so writes under it can be gated.

    Idempotent, and called from ``FileToolSandbox.__init__`` — the gate has
    to know about a root before anything under it is written, and that is
    the moment the root comes into existence.
    """
    key = str(Path(root).resolve())
    with _registry_guard:
        _roots.add(key)
        _root_gates.setdefault(key, _RootGate())
    return key


def registered_roots() -> frozenset[str]:
    """The roots currently registered (a snapshot; the set is live)."""
    with _registry_guard:
        return frozenset(_roots)


def _gate(key: str) -> _RootGate:
    with _registry_guard:
        gate = _root_gates.get(key)
        if gate is None:
            gate = _RootGate()
            _root_gates[key] = gate
        return gate


def _containing_roots(paths: "list[str]") -> list[str]:
    """Registered roots that contain any of *paths*, in canonical order."""
    out = set()
    for root in registered_roots():
        for path in paths:
            if path == root or path.startswith(root + "\\") or path.startswith(root + "/"):
                out.add(root)
                break
    return sorted(out)


def _related_roots(root: str) -> list[str]:
    """*root* plus every registered root nested in it or containing it.

    The registry is small, so walking it is cheaper than maintaining a tree,
    and a writer that misses a nested root is exactly the hole tier 2 exists
    to close.
    """
    out = {root}
    for other in registered_roots():
        if other.startswith(root) or root.startswith(other):
            out.add(other)
    return sorted(out)


@contextmanager
def _shared_roots(paths: "list[str]") -> Iterator[None]:
    gates = [_gate(k) for k in _containing_roots(paths)]
    held: list[_RootGate] = []
    try:
        for gate in gates:
            gate.acquire_shared()
            held.append(gate)
        yield
    finally:
        for gate in reversed(held):
            gate.release_shared()


@contextmanager
def write_gate(root: "str | Path | None") -> Iterator[None]:
    """Hold *root* (and every root nested with it) exclusively.

    For a subprocess that may write: nothing can know which files it will
    touch, so the whole tree is taken for its duration. The wait is visible
    rather than silent — the blocked call has already emitted its
    ``before_tool_execute``, so a queued tool reads as queued, not as hung.

    ``None`` is a no-op, so a caller with no sandbox needs no branch.
    """
    if root is None:
        yield
        return
    key = str(Path(root).resolve())
    gates = [_gate(k) for k in _related_roots(key)]
    held: list[_RootGate] = []
    try:
        for gate in gates:
            gate.acquire_exclusive()
            held.append(gate)
        yield
    finally:
        for gate in reversed(held):
            gate.release_exclusive()
