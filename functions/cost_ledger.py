# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

What the runs cost, kept across them (D90).

D88 made money a ceiling: `UsageLimits` refuses to overspend a cap, and
`snapshot()` says what this run has used. That answers *am I about to go
over*. It does not answer the question people actually ask, which is
**what did this week cost me** -- because every counter dies with the run
that owned it.

This is the other half, and it is deliberately the smaller one. One
append-only JSONL file beside the grant store and the event sink, one
line per run, no content of any kind: when, how long, which models, how
many tokens each way, and how much. A line is small enough that the
interesting file is years of them.

**It writes only when money was spent.** A run on a local model produces
no line at all, so somebody who never touches a paid endpoint never
acquires a file -- which is the objection D85 answered with a checkbox,
answered here by the data instead. When there *is* a bill, recording it
is not a feature someone should have had to enable beforehand: you find
out you wanted the ledger at the moment you get the invoice, which is
exactly too late to switch it on.

**A broken ledger is a silent ledger.** Any `OSError` is logged once and
swallowed. Bookkeeping that can fail the run it is keeping books on is
worse than no bookkeeping -- the same rule the event sink follows.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from weave.logger import get_logger

__all__ = [
    "DEFAULT_MAX_LINES",
    "DEFAULT_PATH",
    "describe_summary",
    "read_lines",
    "record_run",
    "summarize",
]

log = get_logger("SilkCostLedger")

#: Sibling of the grant store, the secrets file and the run sink --
#: outside the graph, because a saved graph must stay shareable (D22).
DEFAULT_PATH = Path.home() / ".weave" / "silk" / "spend.jsonl"

#: Lines kept. A run is one line of a few hundred bytes, so this is
#: years of ordinary use; the cap exists so an automated loop cannot
#: grow the file without bound, not to expire anything a person wanted.
DEFAULT_MAX_LINES = 50_000

#: Written on every line so a reader never has to guess which shape it
#: is looking at. Bumped only when an existing field changes meaning.
SCHEMA = 1


def _total(entries: Iterable[Dict[str, Any]]) -> Optional[float]:
    """The priced part of *entries*, or ``None`` when nothing was priced.

    A chain that fell back from a metered gateway to a local model has a
    real, partial bill; the total is what the priced members spent, and
    the unpriced ones are visible in the per-model breakdown rather than
    silently folded in as zero.
    """
    costs = [e["cost"] for e in entries if e.get("cost") is not None]
    return round(sum(costs), 8) if costs else None


def record_run(
    spend: List[Dict[str, Any]],
    *,
    run_id: str = "",
    session_id: str = "",
    agent: str = "",
    elapsed_s: float = 0.0,
    outcome: str = "",
    path: Optional[Path] = None,
) -> bool:
    """Append one line for a finished run. Returns whether it wrote.

    Args:
        spend: :meth:`GraphEngine.spend_report` -- one entry per model
            the run actually reached.
        run_id: the run's own id, so a line can be matched against the
            event sink's file for the same run.
        session_id: which conversation it belonged to.
        agent: the node's title, which is what a person recognises in a
            report far better than a session id.
        elapsed_s: wall time, because "expensive" and "slow" are
            different complaints and the ledger should not conflate them.
        outcome: how the run ended (``EventRunResult.outcome``). A run
            stopped by its budget still spent what it spent, and a report
            that cannot tell those apart invites the wrong conclusion.
        path: override, for tests.

    Returns:
        False when there was nothing to record, or when writing failed.
        Never raises -- see the module docstring.
    """
    entries = [e for e in (spend or []) if e.get("requests")]
    total = _total(entries)
    if total is None:
        # Unpriced run: a local model, or a gateway that quotes nothing.
        # No bill, no line. This is the whole of "off by default".
        return False

    line = {
        "schema": SCHEMA,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "session_id": session_id,
        "agent": agent,
        "elapsed_s": round(float(elapsed_s or 0.0), 3),
        "outcome": outcome,
        "cost": total,
        "currency": "USD",
        "models": [
            {
                "model": e.get("model", "?"),
                "backend": e.get("backend", "?"),
                "requests": int(e.get("requests", 0)),
                "input_tokens": int(e.get("input_tokens", 0)),
                "output_tokens": int(e.get("output_tokens", 0)),
                "cost": (round(e["cost"], 8)
                         if e.get("cost") is not None else None),
            }
            for e in entries
        ],
    }

    target = Path(path) if path is not None else DEFAULT_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning(f"Cost ledger not written ({target}): {exc}")
        return False
    _prune(target)
    return True


def _prune(target: Path, max_lines: int = DEFAULT_MAX_LINES) -> None:
    """Keep the newest *max_lines* lines, cheaply and only when needed.

    The file is read only when it has grown past the cap, and the rewrite
    goes through a temporary file so an interrupted prune cannot leave a
    truncated ledger where a complete one used to be.
    """
    try:
        if target.stat().st_size < max_lines * 64:
            return      # cannot possibly be over the cap; do not read it
        lines = target.read_text(encoding="utf-8").splitlines()
        if len(lines) <= max_lines:
            return
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text("\n".join(lines[-max_lines:]) + "\n", encoding="utf-8")
        os.replace(tmp, target)
        log.info(f"Cost ledger pruned to the newest {max_lines} runs.")
    except OSError as exc:
        log.warning(f"Cost ledger not pruned ({target}): {exc}")


def read_lines(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every readable line of the ledger, oldest first.

    A malformed line is skipped rather than fatal: the file is appended
    to by a process that can be killed mid-write, and one torn line must
    not cost you the other nine hundred.
    """
    target = Path(path) if path is not None else DEFAULT_PATH
    out: List[Dict[str, Any]] = []
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return out
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def summarize(
    days: Optional[float] = 7.0,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """What the last *days* cost, in total and per model.

    ``days=None`` reads the whole ledger. The per-model breakdown is what
    makes the number actionable: a total says you spent eleven dollars,
    the breakdown says nine of them went to one model you could have put
    a cheaper one behind (D89).
    """
    cutoff = None
    if days is not None:
        cutoff = time.time() - days * 86_400

    runs = 0
    total = 0.0
    tokens_in = 0
    tokens_out = 0
    per_model: Dict[str, Dict[str, Any]] = {}
    first: Optional[str] = None
    last: Optional[str] = None

    for record in read_lines(path):
        stamp = _epoch(record.get("ts"))
        if cutoff is not None and stamp is not None and stamp < cutoff:
            continue
        runs += 1
        total += float(record.get("cost") or 0.0)
        ts = record.get("ts")
        if isinstance(ts, str):
            first = ts if first is None else min(first, ts)
            last = ts if last is None else max(last, ts)
        for entry in record.get("models") or []:
            name = str(entry.get("model", "?"))
            acc = per_model.setdefault(
                name, {"model": name, "runs": 0, "requests": 0,
                       "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
                       "unpriced": False},
            )
            acc["runs"] += 1
            acc["requests"] += int(entry.get("requests", 0))
            acc["input_tokens"] += int(entry.get("input_tokens", 0))
            acc["output_tokens"] += int(entry.get("output_tokens", 0))
            tokens_in += int(entry.get("input_tokens", 0))
            tokens_out += int(entry.get("output_tokens", 0))
            if entry.get("cost") is None:
                # It ran, and nobody said what it charged. Kept as a flag
                # rather than folded in as zero: the total below is what
                # is *known* to have been spent, never a claim about what
                # was not quoted.
                acc["unpriced"] = True
            else:
                acc["cost"] += float(entry["cost"])

    return {
        "days": days,
        "runs": runs,
        "cost": round(total, 8),
        "currency": "USD",
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "first": first,
        "last": last,
        "models": sorted(per_model.values(),
                         key=lambda m: (-m["cost"], m["model"])),
    }


def _epoch(stamp: Any) -> Optional[float]:
    """An ISO timestamp as seconds, or ``None`` if it will not read."""
    if not isinstance(stamp, str):
        return None
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except ValueError:
        return None


def describe_summary(summary: Dict[str, Any]) -> str:
    """The summary as a person reads it, most expensive model first."""
    from .usage_limits import format_cost

    runs = summary.get("runs", 0)
    if not runs:
        window = ("ever" if summary.get("days") is None
                  else f"in the last {summary['days']:g} day(s)")
        return f"No priced runs {window}."
    window = ("all time" if summary.get("days") is None
              else f"the last {summary['days']:g} day(s)")
    head = (f"{format_cost(summary.get('cost', 0.0))} over {runs} run(s), "
            f"{window}")
    lines = [head]
    for model in summary.get("models", [])[:8]:
        note = "  (some runs unpriced)" if model.get("unpriced") else ""
        lines.append(
            f"  {model['model']}: {format_cost(model['cost'])} "
            f"· {model['requests']} request(s) "
            f"· {model['input_tokens']:,} in / "
            f"{model['output_tokens']:,} out{note}"
        )
    return "\n".join(lines)
