# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The durable event sink: one JSONL file per run (T7, D85).

Compaction (§12) is a lossy projection -- it drops turns to keep the
window under a cap, and once dropped they are gone. That is what turned
T7 from "nice for debugging" into a precondition: the events are the only
remaining account of a run whose middle was thrown away.

What this writes is the same wire vocabulary the ``events`` port carries
(D2), one JSON object per line, with one extra rule on top:

**Metadata only, always.** The wire form already drops the big content
fields, but not all of them -- a delta carries its text, a tool call its
arguments, a decision its rendered prompt. A file on disk outlives the
run and the window it was shown in, so the sink replaces every free-text
field with its *length* (``delta`` -> ``delta_chars``) and a tool call's
arguments with their key names and total size. What is left answers
"what happened, in what order, how big" and never "what did it say".

Three operational rules, because a log that damages the run it records is
worse than no log:

- **A broken sink is a silent sink.** Any OSError disables it for the
  rest of the run and is logged once. Nothing propagates into the run.
- **Bounded.** A run stops writing at ``max_lines`` and says so in a
  final line, so one runaway loop cannot fill a disk.
- **Pruned.** Only the newest ``keep_runs`` files survive an open, so the
  directory does not grow without end.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from weave.logger import get_logger

__all__ = [
    "DEFAULT_DIR",
    "DEFAULT_KEEP_RUNS",
    "DEFAULT_MAX_LINES",
    "REDACTED",
    "RunSink",
    "redact",
]

log = get_logger("SilkEventSink")

#: Sibling of the grant store and the secrets file -- outside the graph,
#: because a saved graph must stay shareable (D22).
DEFAULT_DIR = Path.home() / ".weave" / "silk" / "runs"

#: How many run files survive. Debugging looks at the last few runs; the
#: hundredth is landfill.
DEFAULT_KEEP_RUNS = 50

#: A cap, not a target. A model looping on one tool can emit events far
#: faster than anyone will read them.
DEFAULT_MAX_LINES = 20_000

#: Free-text fields, replaced by ``<name>_chars``. Everything not named
#: here travels as-is, so adding an event type cannot silently start
#: logging content: a new *content* field has to be added here to be
#: redacted, and the review that adds it is the point.
REDACTED = (
    "delta",
    "cumulative_text",
    "text",
    "content",
    "result",
    "message",
    "answer",
    "response",
    "prompt",
    "system_prompt",
    "instructions",
    "summary",
    "note",
    "goal",
)


def _sizeof(value: Any) -> int:
    """How much text this was, without keeping any of it."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(value))


def redact(wire: dict[str, Any]) -> dict[str, Any]:
    """One wire event as the sink may keep it: shape and sizes, no text.

    Never mutates the caller's dict -- the same event is on its way to a
    widget that still wants the whole thing.
    """
    out: dict[str, Any] = {}
    for key, value in (wire or {}).items():
        if key in REDACTED:
            out[f"{key}_chars"] = _sizeof(value)
        elif key == "tool_args":
            args = value if isinstance(value, dict) else {}
            out["tool_args_keys"] = sorted(str(k) for k in args)
            out["tool_args_chars"] = _sizeof(value)
        else:
            out[key] = value
    return out


class RunSink:
    """The JSONL file for one run. Opened on the first event, not before.

    An agent that never runs must not leave a file behind, and a run_id
    is only known once there is an event carrying it.
    """

    def __init__(
        self,
        directory: Optional[Path] = None,
        *,
        keep_runs: int = DEFAULT_KEEP_RUNS,
        max_lines: int = DEFAULT_MAX_LINES,
    ) -> None:
        self.directory = Path(directory) if directory else DEFAULT_DIR
        self.keep_runs = max(1, int(keep_runs))
        self.max_lines = max(1, int(max_lines))
        self.path: Optional[Path] = None
        self.lines = 0
        self._handle: Any = None
        self._disabled = False
        self._truncated = False

    # ── lifecycle ────────────────────────────────────────────────────

    def _open(self, run_id: str) -> bool:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        # The run id is in the name so two runs a second apart cannot
        # collide, and a stamp is in front so the newest sort last.
        name = f"{stamp}-{(run_id or 'run')[:8]}.jsonl"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._prune()
            self._handle = open(self.directory / name, "a", encoding="utf-8")
        except OSError as exc:
            self._fail(f"could not open a run log in {self.directory}: {exc}")
            return False
        self.path = self.directory / name
        return True

    def _prune(self) -> None:
        """Keep the newest ``keep_runs`` files. Best effort, never fatal."""
        try:
            files = sorted(self.directory.glob("*.jsonl"))
        except OSError:
            return
        for stale in files[: max(0, len(files) - self.keep_runs + 1)]:
            try:
                stale.unlink()
            except OSError:
                pass    # someone else's file, or a file in use; leave it

    def _fail(self, reason: str) -> None:
        """Disable the sink once, loudly enough to find, quietly enough
        that the run does not care."""
        if not self._disabled:
            log.warning(f"Run event log disabled: {reason}")
        self._disabled = True
        self.close()

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            handle.close()
        except OSError:
            pass

    def __enter__(self) -> "RunSink":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ── writing ──────────────────────────────────────────────────────

    def write(self, wire: dict[str, Any]) -> bool:
        """Append one event. Returns whether it was written.

        `False` is an ordinary answer -- the sink is disabled, full, or
        was handed something that is not an event -- and never an error
        the caller has to handle.
        """
        if self._disabled or not isinstance(wire, dict) or not wire:
            return False
        if self._handle is None and not self._open(str(wire.get("run_id", ""))):
            return False
        if self.lines >= self.max_lines:
            if not self._truncated:
                self._truncated = True
                self._raw({"type": "sink_truncated", "ts": time.time(),
                           "run_id": wire.get("run_id", ""),
                           "lines": self.lines,
                           "note": "cap reached; later events are not recorded"})
            return False
        if self._raw(redact(wire)):
            self.lines += 1
            return True
        return False

    def _raw(self, record: dict[str, Any]) -> bool:
        handle = self._handle
        if handle is None:
            return False
        try:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            # Flushed per line on purpose: the run this records is exactly
            # the kind of thing that ends by crashing, and a buffered tail
            # is the part worth having.
            handle.flush()
            return True
        except (OSError, ValueError) as exc:
            self._fail(f"write failed: {exc}")
            return False
