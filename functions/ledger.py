# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The Macrame ledger seam: one handle per file, process-wide (spec D62).

Macrame is a bitemporal graph ledger with **one Write Actor per open
handle**. Two handles on one file is outside its contract, and the
library does not stop you -- opening the same path twice succeeds and
gives you two writers racing on one file. So the rule has to live
somewhere, and the only place it can be complete is the single place that
opens: this registry. Nodes, tools, the hub and the compactor never call
``Database.open`` themselves.

What the registry is:

- a process singleton mapping resolved path -> open handle, refcounted;
- the owner of ``close()``, which Macrame says has exactly one owner --
  plugin unload or graph close closes, never a node and never a run;
- the answer to "is this file already open here", which is what turns
  the Task Hub's file discovery (D58, D60(2)) into *discovery plus
  lookup* rather than a second open of a live ledger.

What it is not: a backend. ``macrame-db`` is an optional extra (D66).
Absent, :func:`available` is False and the caller falls back to
``SqliteTaskStore`` -- loudly, with one log line, never silently.

Edge-type note: Macrame validates edge types against ``[A-Z0-9]+``, so
the vocabulary here is ``CLAIMEDBY`` / ``SUBTASKOF`` / ``INRUN`` rather
than the underscored names §17 writes prose in.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from weave.logger import get_logger

log = get_logger("SilkLedger")

#: The optional extra that provides the backend (D66, G5).
DISTRIBUTION = "macrame-db"

#: Snapshot cadence for a Silk ledger. Silk's writes are small and its
#: processes are long-lived, so the library default is right; the field
#: exists so a test can ask for none without reaching into the registry.
DEFAULT_SNAPSHOT_ENTRIES: Optional[int] = None

_IMPORT_ERROR: Optional[BaseException] = None

try:                                    # pragma: no cover - import shape
    import macrame as _macrame
except BaseException as exc:            # noqa: BLE001 - a binary wheel can
    _macrame = None                     #   fail to load, not just be absent
    _IMPORT_ERROR = exc


def available() -> bool:
    """Whether the ledger backend can be used at all."""
    return _macrame is not None


def unavailable_reason() -> str:
    """Why it cannot, in a sentence a log line can carry."""
    if _macrame is not None:
        return ""
    detail = f" ({_IMPORT_ERROR})" if _IMPORT_ERROR else ""
    return (
        f"the {DISTRIBUTION} extra is not installed{detail}; Silk is using "
        f"the SQLite task store and keeping history in the node"
    )


class LedgerUnavailable(RuntimeError):
    """Raised when a ledger is asked for and the backend is not there."""


@dataclass
class _Entry:
    handle: Any
    refs: int
    path: str


class LedgerRegistry:
    """One open handle per ledger file, for the life of the process.

    Refcounted rather than opened-and-closed per user: several agents in
    one sandbox root share one task ledger, and closing it under the
    others because one run ended would take the write actor with it.

    Closing is deliberately *not* automatic when the count reaches zero.
    A graph that finishes a run and starts another would otherwise pay a
    full open per run, and Macrame's close writes a final snapshot --
    cheap once, wasteful in a loop. :meth:`close_all` at plugin unload or
    graph close is the owner, as D62 requires.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._open: dict[str, _Entry] = {}

    # ── opening ───────────────────────────────────────────────────────

    def acquire(self, path: str | os.PathLike, *,
                snapshot_every_entries: Optional[int] = DEFAULT_SNAPSHOT_ENTRIES,
                ) -> Any:
        """The shared handle for *path*, opening it once if needed.

        Raises :class:`LedgerUnavailable` when the extra is missing --
        the caller's cue to fall back (D66), not a crash to swallow.
        """
        if _macrame is None:
            raise LedgerUnavailable(unavailable_reason())

        key = self.key(path)
        with self._lock:
            entry = self._open.get(key)
            if entry is not None and not _is_closed(entry.handle):
                entry.refs += 1
                return entry.handle
            if entry is not None:
                # A handle that died under us (Macrame closes hard on some
                # errors). Reopening is right; pretending it is live is not.
                log.warning(f"Ledger handle for {key} was closed; reopening")
                self._open.pop(key, None)

            Path(key).parent.mkdir(parents=True, exist_ok=True)
            handle = _macrame.Database.open(
                key, snapshot_every_entries=snapshot_every_entries)
            self._open[key] = _Entry(handle=handle, refs=1, path=key)
            log.debug(f"Opened ledger {key}")
            return handle

    def release(self, path: str | os.PathLike) -> int:
        """Give up one reference. The handle stays open (see the class doc)."""
        key = self.key(path)
        with self._lock:
            entry = self._open.get(key)
            if entry is None:
                return 0
            entry.refs = max(0, entry.refs - 1)
            return entry.refs

    # ── looking ───────────────────────────────────────────────────────

    def get(self, path: str | os.PathLike) -> Optional[Any]:
        """The handle for *path* if this process already has it open.

        This is the half D58's scan needs: files are found on disk, and
        a file that is already open is read through the handle that owns
        it rather than opened a second time.
        """
        with self._lock:
            entry = self._open.get(self.key(path))
        if entry is None or _is_closed(entry.handle):
            return None
        return entry.handle

    def is_open(self, path: str | os.PathLike) -> bool:
        return self.get(path) is not None

    def paths(self) -> list[str]:
        with self._lock:
            return sorted(self._open)

    def refs(self, path: str | os.PathLike) -> int:
        with self._lock:
            entry = self._open.get(self.key(path))
            return entry.refs if entry is not None else 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._open)

    # ── closing (the registry's job, nobody else's) ───────────────────

    def close(self, path: str | os.PathLike) -> bool:
        """Close one ledger regardless of refcount. Owner-only.

        Callers that merely finished with a ledger want :meth:`release`.
        This is for the owner -- graph close, plugin unload -- and it is
        blunt on purpose: a handle nobody closes loses its final
        snapshot.
        """
        key = self.key(path)
        with self._lock:
            entry = self._open.pop(key, None)
        if entry is None:
            return False
        _close(entry.handle, key)
        return True

    def close_all(self) -> int:
        """Close every ledger this process holds. Plugin unload calls this."""
        with self._lock:
            entries = list(self._open.values())
            self._open.clear()
        for entry in entries:
            _close(entry.handle, entry.path)
        return len(entries)

    # ── identity ──────────────────────────────────────────────────────

    @staticmethod
    def key(path: str | os.PathLike) -> str:
        """One canonical string per file.

        Two names for one file must be one entry, or the sole-writer rule
        is enforced against spellings rather than files -- which is no
        rule at all on a case-insensitive filesystem.
        """
        resolved = Path(path).expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:                 # pragma: no cover - unreachable path
            resolved = resolved.absolute()
        text = str(resolved)
        return os.path.normcase(text) if os.name == "nt" else text


def _is_closed(handle: Any) -> bool:
    """Whether a handle has already shut down. Defensive: the flag moved
    from method to property once, and a wrong answer here reopens a live
    ledger."""
    flag = getattr(handle, "is_closed", False)
    if callable(flag):
        try:
            return bool(flag())
        except Exception:               # noqa: BLE001
            return False
    return bool(flag)


def _close(handle: Any, key: str) -> None:
    try:
        handle.close()
        log.debug(f"Closed ledger {key}")
    except Exception as exc:            # noqa: BLE001 - closing twice, or a
        log.debug(f"Ledger {key} closed with: {exc}")   # dead actor


#: The process's registry -- the only thing that opens a ledger (D62).
REGISTRY = LedgerRegistry()


def ledger_path(root: str | os.PathLike, stem: str = "ledger") -> Path:
    """Where a sandbox root's task ledger lives.

    One ledger per sandbox root is the default placement (§17), which
    keeps T4/D58 file discovery working unchanged: a ledger is a file
    under a root like a plan is.
    """
    return Path(root).expanduser() / f"{stem}.macrame"
