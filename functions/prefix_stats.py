# -*- coding: utf-8 -*-
"""Prompt-prefix reuse measurement (spec D41/D47, Phase 1 item 6; G15).

Silk's whole context design hangs on one number nobody has ever looked at:
how much of a request's prompt the backend already had in its KV cache. If
reuse is high, every mechanism in D47 is solving a problem that does not
exist; if it is near zero, `delegate_parallel` is far more expensive than it
looks, and compaction's double-prefill cost is a footnote next to it.

The measurement is nearly free, which is why it comes before any of the
tuning. `verbose` is already forwarded to the spawned `llama_cpp.server`
(``_SERVER_MODEL_KEYS``) and its stderr is already captured to a file so a
failed start can report why. That file therefore already contains, per
request::

    Llama.generate: 512 prefix-match hit, remaining 8 prompt tokens to eval
    llama_perf_context_print: prompt eval time = 210.11 ms / 8 tokens
    llama_perf_context_print:       total time = 1980.44 ms / ...

Nothing needs to be added to the model path. This module reads what is
already being written and turns it into D47's three numbers:

===================  ==========================================  ====================================
Metric               Definition                                  What it decides
===================  ==========================================  ====================================
**Reuse rate**       matched / (matched + evaluated), summed     whether reuse is being lost at all
**Contention rate**  requests whose predecessor was another      whether the loss is interleaving
                     session, over requests with a predecessor   or prefix instability
**Prefill share**    prompt-eval time / total time, summed       whether any of this is worth building
===================  ==========================================  ====================================

Attribution is sequential on purpose. `llama_cpp.server` serialises every
request through ``llama_outer_lock`` (spec D43/D53), so at most one request
is in the model at a time and the lines a drain finds belong to the request
that just finished. That is also precisely why *contention* is measurable at
all: the predecessor of a request is a well-defined thing.

Everything here reports ``None`` rather than a guess when it has no data. A
reuse rate of 0.0 and an unknown reuse rate lead to opposite decisions, and
the point of the exercise is to tell them apart.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# ── the lines we read ─────────────────────────────────────────────────────

#: ``Llama.generate: <n> prefix-match hit, remaining <m> prompt tokens to eval``
_PREFIX_NUMERIC = re.compile(
    r"(\d+)\s+prefix-match hit,\s*remaining\s+(\d+)\s+prompt tokens"
)
#: Older/newer builds print the bare form with no numbers. It says a hit
#: happened and nothing about its size, so it is counted separately rather
#: than folded in as a zero (which would drag the rate down) or as a full
#: match (which would invent one).
_PREFIX_BARE = re.compile(r"prefix-match hit")
#: ``... prompt eval time =  210.11 ms /   8 tokens`` — both the current
#: ``llama_perf_context_print`` prefix and the older ``llama_print_timings``.
_PROMPT_EVAL = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
)
#: ``...       total time =  1980.44 ms / ...``
_TOTAL_TIME = re.compile(r"total time\s*=\s*([\d.]+)\s*ms")


@dataclass(frozen=True)
class PrefixLine:
    """One fact read off a server log line."""

    kind: str          # "prefix" | "prompt_eval" | "total"
    matched: Optional[int] = None
    evaluated: Optional[int] = None
    ms: Optional[float] = None


def parse_line(line: str) -> Optional[PrefixLine]:
    """Read one log line, or ``None`` if it says nothing we measure."""
    match = _PREFIX_NUMERIC.search(line)
    if match:
        return PrefixLine("prefix", matched=int(match.group(1)),
                          evaluated=int(match.group(2)))
    match = _PROMPT_EVAL.search(line)
    if match:
        return PrefixLine("prompt_eval", ms=float(match.group(1)),
                          evaluated=int(match.group(2)))
    match = _TOTAL_TIME.search(line)
    if match:
        return PrefixLine("total", ms=float(match.group(1)))
    if _PREFIX_BARE.search(line):
        return PrefixLine("prefix")
    return None


# ── one request's worth of numbers ────────────────────────────────────────


@dataclass
class PrefixSample:
    """What one model request cost, and whose it was."""

    session: str = ""
    matched: Optional[int] = None
    evaluated: Optional[int] = None
    prompt_eval_ms: Optional[float] = None
    total_ms: Optional[float] = None
    wall_s: Optional[float] = None
    #: True when the immediately preceding request came from another
    #: session — the definition of contention in D47's table.
    contended: Optional[bool] = None

    @property
    def prompt_tokens(self) -> Optional[int]:
        if self.matched is None or self.evaluated is None:
            return None
        return self.matched + self.evaluated

    @property
    def reuse(self) -> Optional[float]:
        total = self.prompt_tokens
        if not total:
            return None
        return (self.matched or 0) / total


@dataclass
class PrefixReport:
    """D47's three numbers, plus enough context to trust them."""

    requests: int = 0
    measured_requests: int = 0
    matched_tokens: int = 0
    evaluated_tokens: int = 0
    reuse_rate: Optional[float] = None
    contention_rate: Optional[float] = None
    prefill_share: Optional[float] = None
    bare_hits: int = 0

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "measured_requests": self.measured_requests,
            "matched_tokens": self.matched_tokens,
            "evaluated_tokens": self.evaluated_tokens,
            "reuse_rate": self.reuse_rate,
            "contention_rate": self.contention_rate,
            "prefill_share": self.prefill_share,
            "bare_hits": self.bare_hits,
        }

    def describe(self) -> str:
        """A human-readable line per metric, saying 'unknown' when it is."""
        def pct(value: Optional[float]) -> str:
            return "unknown" if value is None else f"{value * 100:.1f}%"

        return (
            f"requests={self.requests} (measured {self.measured_requests})\n"
            f"reuse rate      : {pct(self.reuse_rate)}"
            f"  [{self.matched_tokens} matched / "
            f"{self.matched_tokens + self.evaluated_tokens} prompt tokens]\n"
            f"contention rate : {pct(self.contention_rate)}\n"
            f"prefill share   : {pct(self.prefill_share)}"
        )


class PrefixMeter:
    """Collects per-request samples and reduces them to a report.

    Thread-safe, because the pool it hangs off is shared by every agent in
    the graph (spec D52.4 made the same point about usage limits).
    """

    def __init__(self, max_samples: int = 500) -> None:
        self._lock = threading.RLock()
        self._samples: list[PrefixSample] = []
        self._max_samples = max_samples
        self._last_session: Optional[str] = None

    # -- recording --------------------------------------------------------

    def record(self, sample: PrefixSample) -> PrefixSample:
        """Record one request, filling in contention from its predecessor."""
        with self._lock:
            if self._last_session is not None:
                sample.contended = sample.session != self._last_session
            self._last_session = sample.session
            self._samples.append(sample)
            # A bounded window: this runs for the life of a graph, and the
            # rates are what matter, not the tail of individual requests.
            if len(self._samples) > self._max_samples:
                del self._samples[: len(self._samples) - self._max_samples]
            return sample

    def record_lines(
        self, lines: Iterable[str], *, session: str = "", wall_s: Optional[float] = None
    ) -> Optional[PrefixSample]:
        """Fold the log lines of one finished request into a sample.

        Returns ``None`` when the lines carried nothing measurable — a
        request the server logged nothing about is not a request with zero
        reuse, and must not be counted as one.
        """
        sample = PrefixSample(session=session, wall_s=wall_s)
        seen = False
        for line in lines:
            fact = parse_line(line)
            if fact is None:
                continue
            if fact.kind == "prefix":
                seen = True
                if fact.matched is not None:
                    sample.matched = fact.matched
                    sample.evaluated = fact.evaluated
            elif fact.kind == "prompt_eval":
                seen = True
                sample.prompt_eval_ms = fact.ms
                if sample.evaluated is None:
                    sample.evaluated = fact.evaluated
                if sample.matched is None:
                    # No prefix line at all means no hit: the server
                    # evaluated the whole prompt.
                    sample.matched = 0
            elif fact.kind == "total":
                seen = True
                sample.total_ms = fact.ms
        if not seen:
            return None
        return self.record(sample)

    # -- reporting --------------------------------------------------------

    def samples(self) -> list[PrefixSample]:
        with self._lock:
            return list(self._samples)

    def report(self) -> PrefixReport:
        with self._lock:
            samples = list(self._samples)
        return summarize(samples)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._last_session = None


def summarize(samples: list[PrefixSample]) -> PrefixReport:
    """Reduce samples to D47's three numbers."""
    report = PrefixReport(requests=len(samples))
    matched = evaluated = 0
    measured = 0
    prompt_ms = total_ms = 0.0
    contended = comparable = 0
    for sample in samples:
        if sample.matched is None and sample.evaluated is None:
            report.bare_hits += 1
        if sample.prompt_tokens:
            measured += 1
            matched += sample.matched or 0
            evaluated += sample.evaluated or 0
        if sample.prompt_eval_ms is not None and sample.total_ms:
            prompt_ms += sample.prompt_eval_ms
            total_ms += sample.total_ms
        if sample.contended is not None:
            comparable += 1
            contended += 1 if sample.contended else 0

    report.measured_requests = measured
    report.matched_tokens = matched
    report.evaluated_tokens = evaluated
    if matched + evaluated:
        report.reuse_rate = matched / (matched + evaluated)
    if comparable:
        report.contention_rate = contended / comparable
    if total_ms:
        report.prefill_share = prompt_ms / total_ms
    return report


# ── reading a live server log ─────────────────────────────────────────────


@dataclass
class LogDrain:
    """Incremental reader over the server's stderr log.

    The pool already writes that file; this only ever reads forward from
    where it last stopped, so it costs one seek and a read per request and
    can never disturb the server.
    """

    path: str
    _offset: int = field(default=0, repr=False)

    def drain(self) -> list[str]:
        """Return whatever has been appended since the last drain."""
        try:
            size = Path(self.path).stat().st_size
        except OSError:
            return []
        if size < self._offset:
            # The file was replaced (a relaunched server); start over rather
            # than reading a stale tail as if it were new.
            self._offset = 0
        if size == self._offset:
            return []
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._offset)
                text = handle.read()
                self._offset = handle.tell()
        except OSError:
            return []
        return text.splitlines()


def summarize_log(path: "str | Path") -> PrefixReport:
    """Summarize a whole server log offline, with no session attribution.

    For reading a log after the fact: reuse rate and prefill share are
    recoverable, contention is not (nothing in the file says whose request
    a line belonged to), so it is reported as unknown rather than zero.
    """
    meter = PrefixMeter(max_samples=1_000_000)
    current: list[str] = []
    try:
        with open(str(path), "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fact = parse_line(line)
                if fact is None:
                    continue
                current.append(line)
                if fact.kind == "total":
                    # The timing block ends a request; flush it.
                    meter.record_lines(current)
                    current = []
    except OSError:
        return PrefixReport()
    if current:
        meter.record_lines(current)
    report = meter.report()
    # Every sample carried the same empty session, so the meter's contention
    # is a meaningless 0.0. Say unknown instead.
    report.contention_rate = None
    return report


if __name__ == "__main__":       # pragma: no cover - operator entry point
    # Offline read of a captured server log:
    #   python -m weave.plugins.silk.functions.prefix_stats <server.log>
    import sys as _sys

    if len(_sys.argv) != 2:
        print(__doc__.strip().splitlines()[0])
        print("usage: python -m weave.plugins.silk.functions.prefix_stats <log>")
        raise SystemExit(2)
    print(summarize_log(_sys.argv[1]).describe())
