# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The decision log (S2): what the user answered, and why.

Every question the approval gate puts to a human ends here: the tool, the
outcome, who answered and the reason, when one was given. Two readers:

* the gate's triage reads the decisions on the same tool in the same
  project (``similar``), so "consistent with what the user decided before"
  is a question about real answers rather than an empty list;
* the ``findings_gate`` hook reads this run's decisions (``for_run``) and
  files the reasoned ones into the knowledge graph.

Kept in memory for the run and appended to ``~/.weave/silk/decisions.jsonl``
beside the grants, one JSON line per decision, so the next run's triage
still knows. A log that cannot be written stays in memory; it never fails
the decision it records.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

from weave.logger import get_logger

from .grants import GRANT_DIR

log = get_logger("SilkDecisionLog")

DECISION_FILE = "decisions.jsonl"
#: Where the gate keeps the log on its toolbox.
DECISION_LOG_ATTR = "_decision_log"
#: How many decisions are read back from the file, newest kept.
MAX_LOADED = 500


class DecisionLog:
    """Decisions, newest last; optionally appended to a JSONL file."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self._entries: deque[dict[str, Any]] = deque(maxlen=MAX_LOADED)
        if path is not None:
            self._load()

    def _load(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()  # type: ignore[union-attr]
        except OSError:
            return
        for line in lines[-MAX_LOADED:]:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                self._entries.append(entry)

    def record(self, *, decision_id: str, tool: str, prompt: str, outcome: str,
               actor: str = "user", reason: str = "", run_id: str = "",
               project: str = "") -> dict[str, Any]:
        entry = {"id": decision_id, "tool": tool, "prompt": prompt, "outcome": outcome,
                 "actor": actor, "reason": reason, "run_id": run_id, "project": project,
                 "at": time.time()}
        self._entries.append(entry)
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError as exc:
                log.warning(f"Decision not written to {self.path}: {exc}")
        return entry

    def for_run(self, run_id: str) -> list[dict[str, Any]]:
        return [e for e in self._entries if run_id and e.get("run_id") == run_id]

    def similar(self, tool: str, project: str = "", limit: int = 12) -> list[dict[str, Any]]:
        """Decisions on *tool* (in *project*, when given), newest first."""
        out = [e for e in reversed(self._entries) if e.get("tool") == tool
               and (not project or e.get("project") == project)]
        return out[:limit]


def default_decision_path() -> Path:
    """``~/.weave/silk/decisions.jsonl``, or ``$SILK_DECISION_LOG`` (the
    test suites point it at a temporary file)."""
    override = os.environ.get("SILK_DECISION_LOG", "").strip()
    return Path(override) if override else GRANT_DIR / DECISION_FILE


def decision_log_of(toolbox: Any, path: Optional[Path] = None) -> DecisionLog:
    """The log on *toolbox*, made on first use.

    In memory unless *path* is given: the Agent node passes
    :func:`default_decision_path` when it binds a run, so a test or a
    headless embedder never writes to the user's home by accident. A log
    made in memory first is given the file when one arrives later.
    """
    found = getattr(toolbox, DECISION_LOG_ATTR, None)
    if isinstance(found, DecisionLog):
        if path is not None and found.path is None:
            found.path = path
        return found
    made = DecisionLog(path)
    try:
        setattr(toolbox, DECISION_LOG_ATTR, made)
    except AttributeError:
        pass
    return made
