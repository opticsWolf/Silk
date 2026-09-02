# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Approve once, and re-ask when the bytes change (spec §22 q10, D77, D81).

A human approving a load (`functions/load_floor.py`) approves *this code*,
and D77's floor makes them do it every time an agent asks. Question 10 was
the other side of that: after the approval, must the next start ask again?

- **Every session** is safe and unusable: the loop the agent's plugin
  authoring exists for is "write a node, load it, use it tomorrow", and a
  dialog on every start makes tomorrow not come.
- **Once, then always** is usable and wrong: the pin would name a
  *directory the agent can write to*, so "approved" would drift into
  "approved whatever is in there now", and an agent could edit its way
  past a human's approval between two sessions.
- **Once, pinned to the bytes** is the answer. The approval records a
  SHA-256 per file; a start loads a pinned suite only when every digest
  still matches. One edited character and the pin no longer applies, so
  the next load goes back through the floor and a human sees the diff.

Where the pins live follows D35 exactly, and for the same reason grants
do not live under the sandbox root: this is **authority, not
configuration**. `~/.weave/silk/suite_pins.json`, beside `grants.json`,
outside every sandbox root and outside the plugin root -- a pin file the
agent could write would be an agent that approves its own code. It is
allow-only in the same sense: a missing, corrupt or unreadable file means
*nothing is pinned*, so every failure path leads back to asking a human.

**Quarantine breaks the pin** (§22 q11). If a suite took the process down,
the fact that a human once approved those bytes is no longer a reason to
import them unattended: `record_quarantine` unpins, so the fix has to go
back through the floor with a person looking at the diff. An agent may
*read* the traceback and try again -- that is the feedback loop D81 exists
for -- it simply cannot do the last step alone. Auto-load plus auto-retry
is precisely the configuration in which a self-improving loop runs
unattended, which is exactly when it should not.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError

from weave.logger import get_logger

from .grants import GRANT_DIR

log = get_logger("SilkSuitePins")

PIN_FILE = "suite_pins.json"

FORMAT_VERSION = 1

#: Files whose contents the pin covers. Everything Python can import or
#: read at import time; a stray .txt beside the module cannot change what
#: ``import`` does, but a .pth or a .pyd can.
PINNED_SUFFIXES = (".py", ".pyi", ".pth", ".pyd", ".so", ".dll", ".json")

#: A suite bigger than this is not pinned at all -- it is asked about every
#: time. Digesting a tree without limit turns a start into a disk scan, and
#: an agent-authored node suite that needs a thousand files is not the case
#: this feature was built for.
MAX_PINNED_FILES = 400

# ── statuses a start reports ─────────────────────────────────────────────

STATUS_OK = "ok"               # pinned, unchanged: load it
STATUS_CHANGED = "changed"     # pinned, edited since: ask again
STATUS_MISSING = "missing"     # pinned, but the files are gone
STATUS_QUARANTINED = "quarantined"   # it crashed a start; the pin is void
STATUS_UNPINNED = "unpinned"   # never approved, or the approval was withdrawn


def digest_file(path: str | Path) -> str:
    """SHA-256 of one file, or ``""`` if it cannot be read."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
    except OSError as exc:
        log.warning(f"Cannot digest '{path}' ({exc}); treating it as unpinned")
        return ""
    return digest.hexdigest()


def digest_suite(path: str | Path) -> dict[str, str]:
    """Every importable file in a suite, relative path -> SHA-256.

    Returns ``{}`` for anything that is not a readable directory or that is
    larger than :data:`MAX_PINNED_FILES`, and an empty digest map never
    matches -- so a suite this cannot describe is a suite that gets asked
    about.
    """
    root = Path(path)
    if not root.is_dir():
        return {}
    digests: dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        if item.is_dir() or "__pycache__" in item.parts:
            continue
        if item.suffix.lower() not in PINNED_SUFFIXES:
            continue
        if len(digests) >= MAX_PINNED_FILES:
            log.warning(
                f"Suite at '{root}' has more than {MAX_PINNED_FILES} files; "
                "it will be asked about on every start rather than pinned."
            )
            return {}
        stamp = digest_file(item)
        if not stamp:
            return {}
        digests[str(item.relative_to(root)).replace("\\", "/")] = stamp
    return digests


class SuitePin(BaseModel):
    """One human approval, frozen to the bytes it was given."""

    name: str
    #: Where it was when it was approved. Informational: a suite that moved
    #: is matched by name and re-digested where it is now.
    path: str = ""
    #: Relative path -> SHA-256 at the moment of approval.
    digests: dict[str, str] = Field(default_factory=dict)
    pinned_at: float = Field(default_factory=time.time)
    pinned_by: str = ""
    note: str = ""


class PinStore:
    """Which suites a human approved, and for exactly which bytes.

    Reads are total and fail closed: every way the file can be unusable
    answers *not pinned*, which means asking a person.
    """

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.path = (directory or GRANT_DIR) / PIN_FILE
        self._pins: dict[str, SuitePin] = {}
        self.reload()

    # -- persistence ------------------------------------------------------

    def reload(self) -> None:
        """Re-read from disk. Every failure means *nothing is pinned*."""
        self._pins.clear()
        if not self.path.is_file():
            return
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                f"Suite pin file '{self.path}' unreadable ({exc}); treating "
                "it as empty -- every load will ask."
            )
            return
        if not isinstance(document, dict):
            return
        for row in document.get("pins", []):
            try:
                pin = SuitePin(**row)
            except (TypeError, ValidationError) as exc:
                log.warning(f"Skipping an unreadable suite pin ({exc})")
                continue
            self._pins[pin.name] = pin

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "version": FORMAT_VERSION,
                "pins": [pin.model_dump() for pin in self._pins.values()],
            }, indent=2), encoding="utf-8")
        except OSError as exc:
            log.error(
                f"Cannot write the suite pin file '{self.path}' ({exc}); "
                "this approval will not survive the session."
            )

    # -- the two verbs ----------------------------------------------------

    def pin(self, name: str, path: str | Path, *, pinned_by: str = "",
            note: str = "") -> Optional[SuitePin]:
        """Record an approval of the bytes currently at *path*.

        Returns ``None`` when there is nothing to pin -- an unreadable or
        oversized tree -- because a pin that matches nothing would either
        never fire or, worse, fire on emptiness.
        """
        digests = digest_suite(path)
        if not digests:
            log.warning(
                f"Not pinning '{name}': nothing importable could be "
                f"digested at '{path}'. It will be asked about again."
            )
            return None
        self.reload()
        pin = SuitePin(name=str(name), path=str(path), digests=digests,
                       pinned_by=pinned_by, note=note)
        self._pins[pin.name] = pin
        self._flush()
        log.info(f"Pinned suite '{name}' ({len(digests)} file(s)) for "
                 "loading at the next start")
        return pin

    def unpin(self, name: str) -> bool:
        """Withdraw an approval. Returns whether there was one."""
        self.reload()
        if self._pins.pop(str(name), None) is None:
            return False
        self._flush()
        log.info(f"Unpinned suite '{name}'; the next load will ask")
        return True

    # -- reads ------------------------------------------------------------

    def get(self, name: str) -> Optional[SuitePin]:
        return self._pins.get(str(name))

    def names(self) -> list[str]:
        return sorted(self._pins)

    def all(self) -> list[SuitePin]:
        """Every pin, newest first -- the shape a revocation surface wants."""
        return sorted(self._pins.values(),
                      key=lambda p: (-p.pinned_at, p.name))

    def status(self, name: str, path: Optional[str | Path] = None,
               *, quarantine: Sequence[str] = ()) -> str:
        """What a start should do about *name*, and why."""
        if str(name) in set(quarantine or ()):
            return STATUS_QUARANTINED
        pin = self.get(name)
        if pin is None:
            return STATUS_UNPINNED
        where = Path(path) if path is not None else Path(pin.path)
        if not where.is_dir():
            return STATUS_MISSING
        current = digest_suite(where)
        if not current:
            return STATUS_MISSING
        return STATUS_OK if current == pin.digests else STATUS_CHANGED


# ── what a start would do ────────────────────────────────────────────────

def autoload_plan(rows: Optional[Iterable[dict]] = None, *,
                  store: Optional[PinStore] = None,
                  quarantine: Sequence[str] = ()) -> list[dict]:
    """One row per pinned suite: whether this start may load it, and why.

    Nothing here loads anything. The plan is separable from the act on
    purpose -- it is what `list_suites` shows an agent, what a test can
    assert against, and what a start logs when it declines.
    """
    from .self_modify import ORIGIN_USER, suites as discover

    store = store if store is not None else PinStore()
    known = {row.get("name"): row for row in
             (rows if rows is not None else discover())}
    plan: list[dict] = []
    for pin in store.all():
        row = known.get(pin.name) or {}
        status = store.status(pin.name, row.get("path") or pin.path,
                              quarantine=quarantine)
        if status == STATUS_OK and row and row.get("origin") != ORIGIN_USER:
            # It was approved as agent-authored code and is now something
            # else -- a shipped suite of the same name, say. The pin does
            # not carry over (D76).
            status = STATUS_CHANGED
        plan.append({
            "name": pin.name,
            "status": status,
            "path": row.get("path") or pin.path,
            "loaded": bool(row.get("loaded")),
            "pinned_at": pin.pinned_at,
            "files": len(pin.digests),
            "reason": _REASONS[status],
        })
    return plan


_REASONS = {
    STATUS_OK: "approved, and unchanged since",
    STATUS_CHANGED: "edited since it was approved -- a human must see the "
                    "diff again",
    STATUS_MISSING: "approved, but its files are not there any more",
    STATUS_QUARANTINED: "it crashed a previous start; the approval is void "
                        "until a human loads it by hand",
    STATUS_UNPINNED: "never approved",
}


def autoload_names(*, store: Optional[PinStore] = None,
                   quarantine: Sequence[str] = ()) -> list[str]:
    """Just the suites this start may load without asking anyone."""
    return [row["name"] for row in
            autoload_plan(store=store, quarantine=quarantine)
            if row["status"] == STATUS_OK and not row["loaded"]]


def annotate_pins(rows: Iterable[dict], *,
                  store: Optional[PinStore] = None) -> list[dict]:
    """Add ``pinned`` to suite rows, for `list_suites` (D75)."""
    store = store if store is not None else PinStore()
    out = []
    for row in rows:
        row = dict(row)
        pin = store.get(row.get("name", ""))
        row["pinned"] = pin is not None
        out.append(row)
    return out
