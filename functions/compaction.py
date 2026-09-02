# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Compaction: shrink the model-visible history instead of failing the run
(spec §12, D24/D25/D40/D41; closes G14).

A long run walks into the context window and, today, stops there: the
pre-request seam checks the estimated input against the budget and raises.
The budget is a fact about the model, not about the conversation, so the
run dying at it is a missing feature rather than a limit.

**What triggers it (D24).** Two things, and only two. Pressure at the
pre-request seam, when the estimated input crosses ``context - reserve``;
and a *classified* overflow from a model request (D40) -- never a generic
stream error, because a dead ``llama_cpp.server`` produces one of those
too, and answering that by spending a summarization request against the
same dead server is the failure mode the classifier exists to prevent.

**What it costs (D41).** Two full prefills, not one. The summarization is
itself a model request with a different prompt, so it evicts the backend's
resident prefix; then the rebuilt context -- which is, by construction,
near the ceiling -- is prefilled from scratch because its head just
changed. On a local GPU that is dead wall-clock in the middle of a run,
while the user waits. Three consequences are wired in here:

* compaction is **rare and decisive** -- hysteresis on the trigger and a
  generous keep-recent, so a run compacts once rather than every round;
* it is **last**, not first -- the spill hook (option A, ``spill.py``)
  rewrites results before they are appended and is prefix-preserving, so
  it should carry as much of the load as it can before this runs at all;
* :class:`EventCompaction` reports the **prefill cost**, not only the
  turns dropped, or the expensive half stays invisible.

**Where it cuts (I9).** On whole-round boundaries. An assistant turn and
the tool results answering it move together or not at all -- a surviving
``tool`` message whose call was summarized away corrupts the *next*
request, not the current one, which is the kind of bug that surfaces three
rounds later as the model talking about a tool it never called.

**How it fails.** To nothing. Every failure path -- no context window, no
summarizable prefix, a summarizer that errors, refuses, or returns empty,
an engine with no rewrite operation -- returns ``None`` and leaves the
history exactly as it was. The pre-existing ``EventUsageLimit`` /
``EventError`` path then protects the run as it did before. Compaction
never kills a run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from weave.logger import get_logger

from .stream_events import EventCompaction

log = get_logger("SilkCompaction")

#: Why a compaction ran. Both reach :meth:`Compactor.maybe_compact`; only
#: the first consults the pressure threshold.
REASON_PRESSURE = "pressure"
REASON_OVERFLOW = "overflow"

#: Headroom kept below the context window, as a fraction of it and as an
#: absolute floor -- whichever is larger. The generation still has to fit,
#: and so does the round of tool results that follows it.
DEFAULT_RESERVE_RATIO = 0.25
DEFAULT_RESERVE = 1024

#: Turns kept verbatim at the tail. Generous on purpose (D41): the recent
#: turns are what the model is actually working from, and a stingy tail
#: buys a second compaction two rounds later.
DEFAULT_KEEP_RECENT = 8

#: Below this many dropped turns, compaction is not worth two prefills.
DEFAULT_MIN_DROPPED = 4

#: How much the context must grow past the last compaction before another
#: one is allowed, as a fraction of the window. This is the hysteresis: it
#: stops a run that lands just under the threshold from compacting again
#: on the very next round, dropping one turn for two full prefills.
DEFAULT_HYSTERESIS = 0.10

#: Per-turn cap when rendering the dropped prefix for the summarizer. The
#: prefix can be nearly a whole context window, and a summarization request
#: that itself overflows is a wasted prefill on top of a failed compaction.
DEFAULT_MAX_TURN_CHARS = 2000

SUMMARY_SYSTEM = (
    "You compress conversation transcripts for an autonomous agent that "
    "will keep working from your summary alone. You never invent facts and "
    "you never address the user."
)

SUMMARY_PROMPT = """\
Below is the earlier part of an agent's working transcript. It is about to \
be removed from the agent's context to make room, and your summary is all \
that will remain of it.

Write a compact summary that preserves, in this order:
1. the task as stated, and any constraints or preferences given;
2. decisions made and their reasons;
3. facts discovered -- file paths, names, values, results of tool calls;
4. what has been done already, and what is still outstanding.

Omit pleasantries, restatements and anything the agent can rediscover \
trivially. Write it as notes to itself, not as a report to a reader.

--- transcript begins ---
{transcript}
--- transcript ends ---
"""

#: How the summary re-enters the history. Labelled, because an unlabelled
#: summary reads to the model as something it said or was told verbatim.
SUMMARY_MARKER = "[Earlier conversation, summarized to free context]\n\n"


class CompactionUnavailable(RuntimeError):
    """The engine cannot summarize, or cannot rewrite its history."""


# -- where a cut may land (I9) ----------------------------------------------


def round_boundaries(history: Sequence[dict[str, Any]]) -> list[int]:
    """Indices at which the history may be cut without orphaning a result.

    A ``tool`` turn belongs to the assistant turn above it. Cutting at its
    index would keep the answer and drop the question, so every index whose
    turn is a tool result is excluded -- and that single rule is the whole
    of I9, because any other index begins a fresh user or assistant turn
    with its own results still behind it.
    """
    return [i for i, entry in enumerate(history)
            if (entry or {}).get("role") != "tool"]


def plan_cut(
    history: Sequence[dict[str, Any]],
    *,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    min_dropped: int = DEFAULT_MIN_DROPPED,
) -> Optional[int]:
    """How many leading turns to drop, or ``None`` if compaction is moot.

    The target is "everything before the last *keep_recent* turns", snapped
    **down** to a round boundary -- down rather than up, so the snap always
    errs toward keeping more recent context than asked for, never less.
    """
    target = len(history) - max(0, keep_recent)
    if target < min_dropped:
        return None
    candidates = [i for i in round_boundaries(history) if 0 < i <= target]
    if not candidates:
        return None
    cut = max(candidates)
    return cut if cut >= min_dropped else None


def render_transcript(
    entries: Sequence[dict[str, Any]],
    *,
    max_turn_chars: int = DEFAULT_MAX_TURN_CHARS,
) -> str:
    """The dropped turns as plain text for the summarizer."""
    lines: list[str] = []
    for raw in entries:
        entry = raw or {}
        role = str(entry.get("role", "user"))
        content = str(entry.get("content", ""))
        if len(content) > max_turn_chars:
            half = max_turn_chars // 2
            omitted = len(content) - 2 * half
            content = (f"{content[:half].rstrip()}\n[... {omitted} characters "
                       f"omitted ...]\n{content[-half:].lstrip()}")
        if role == "tool":
            role = f"tool result ({entry.get('name', 'unknown')})"
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


# -- the compactor ----------------------------------------------------------


@dataclass
class Compactor:
    """Decides when to compact, and does it atomically when it does.

    Held by the :class:`~.agent_loop.AgentLoop` as an optional collaborator,
    exactly like ``output_validator``: the loop owns the turn, the engine
    owns one request, and the compactor owns the summarization request and
    the swap. A loop without one behaves precisely as it did before.
    """

    keep_recent: int = DEFAULT_KEEP_RECENT
    min_dropped: int = DEFAULT_MIN_DROPPED
    reserve: int = DEFAULT_RESERVE
    reserve_ratio: float = DEFAULT_RESERVE_RATIO
    hysteresis: float = DEFAULT_HYSTERESIS
    max_turn_chars: int = DEFAULT_MAX_TURN_CHARS
    #: ``(transcript) -> str``. Defaults to the agent's own model (D25).
    summarizer: Optional[Callable[[str], str]] = None
    #: Generation parameters for the summarization request.
    gen_params: dict[str, Any] = field(default_factory=dict)
    #: Optional :class:`~.spill.SpillWriter` for the dropped transcript, so
    #: ``EventCompaction`` can carry a reference rather than the content.
    writer: Any = None

    #: How many compactions this run has performed.
    compactions: int = 0
    #: Prompt size straight after the last compaction, for the hysteresis.
    _floor: Optional[int] = None

    # -- the trigger --------------------------------------------------------

    def threshold(self, budget: Optional[int]) -> Optional[int]:
        """The prompt size above which pressure compaction is allowed."""
        if not budget or budget <= 0:
            return None
        reserve = max(self.reserve, int(budget * self.reserve_ratio))
        limit = budget - reserve
        return limit if limit > 0 else None

    def should_compact(
        self,
        engine: Any,
        *,
        context_length: Optional[int] = None,
        estimated: Optional[int] = None,
    ) -> bool:
        """Whether pressure alone justifies a compaction right now.

        An unknown context window is not a reason to guess: with no
        denominator there is no pressure to measure, and compacting on a
        hunch would spend two prefills to answer a number nobody has.
        """
        budget = context_length if context_length is not None else _budget(engine)
        limit = self.threshold(budget)
        if limit is None:
            return False
        estimate = estimated if estimated is not None else _tokens(engine)
        if estimate is None or estimate <= limit:
            return False
        if self._floor is not None and budget:
            # Hysteresis (D41): the context has to have actually grown since
            # the last compaction. Otherwise a run that lands just under the
            # threshold compacts again next round, dropping a turn or two
            # for two more full prefills.
            if estimate - self._floor < self.hysteresis * budget:
                log.debug(
                    f"compaction suppressed by hysteresis: {estimate} tokens, "
                    f"{self._floor} after the last one"
                )
                return False
        return True

    # -- the act ------------------------------------------------------------

    def maybe_compact(
        self,
        engine: Any,
        *,
        reason: str = REASON_PRESSURE,
        force: bool = False,
        context_length: Optional[int] = None,
    ) -> Optional[EventCompaction]:
        """Compact if warranted; return the event, or ``None`` if not.

        ``force`` is the overflow retry (D24's second trigger): the backend
        has already said the prompt does not fit, so its verdict outranks
        our estimate and the threshold is not consulted. Everything else --
        the boundary rule, the summary, the atomic swap -- is identical,
        because an overflow is not a licence to cut carelessly.
        """
        if not force and not self.should_compact(
            engine, context_length=context_length
        ):
            return None
        try:
            return self._compact(engine, reason=reason)
        except Exception as exc:  # noqa: BLE001 - compaction never kills a run
            log.warning(f"Compaction ({reason}) failed and was skipped: {exc}")
            return None

    def _compact(self, engine: Any, *, reason: str) -> Optional[EventCompaction]:
        history = getattr(engine, "history", None)
        replace = getattr(engine, "replace_history_prefix", None)
        if not history or replace is None:
            raise CompactionUnavailable(
                "this engine has no rewritable history (spec D25 shape)"
            )
        cut = plan_cut(history, keep_recent=self.keep_recent,
                       min_dropped=self.min_dropped)
        if cut is None:
            log.debug("nothing to compact: no round boundary worth cutting at")
            return None

        transcript = render_transcript(history[:cut],
                                       max_turn_chars=self.max_turn_chars)
        tokens_before = _tokens(engine)

        # The summary is produced *before* anything is dropped. If it fails,
        # the history is still whole and the run continues without a
        # compaction -- which is the pre-D24 behaviour, not a new failure.
        summary, summary_tokens = self._summarize(engine, transcript)
        summary = (summary or "").strip()
        if not summary:
            raise CompactionUnavailable("the summarizer returned nothing")

        ref = self._archive(transcript)
        dropped = replace(cut, SUMMARY_MARKER + summary)
        tokens_after = _tokens(engine)
        self.compactions += 1
        self._floor = tokens_after

        # Both prefills, because both are real and the second one is the
        # one nobody expects (D41): the summarization request evicts the
        # resident prefix, and the rebuilt context is then evaluated whole.
        prefill: Optional[int] = None
        if summary_tokens is not None or tokens_after is not None:
            prefill = (summary_tokens or 0) + (tokens_after or 0)
        log.info(
            f"Compacted {dropped} turns ({reason}): {tokens_before} -> "
            f"{tokens_after} tokens, about {prefill} tokens to re-prefill"
        )
        return EventCompaction(
            turns_dropped=dropped,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            prefill_tokens=prefill,
            summary_ref=ref,
        )

    # -- the summarization request (D25) ------------------------------------

    def _summarize(self, engine: Any, transcript: str) -> tuple[str, Optional[int]]:
        """Return the summary and what producing it cost to prefill."""
        if self.summarizer is not None:
            return self.summarizer(transcript), _estimate_chars(transcript)

        factory = getattr(engine, "sibling", None)
        if factory is None:
            raise CompactionUnavailable(
                "this engine cannot make a sibling for the summarization "
                "request; pass a summarizer instead"
            )
        sibling = factory(system_prompt=SUMMARY_SYSTEM)
        sibling.append_message("user", SUMMARY_PROMPT.format(transcript=transcript))
        cost = _tokens(sibling)
        text = "".join(sibling.stream_response(dict(self.gen_params)))
        stats = getattr(sibling, "last_stats", None) or {}
        if stats.get("error"):
            raise CompactionUnavailable(
                f"the summarization request failed: {stats['error']}"
            )
        return text, cost

    def _archive(self, transcript: str) -> str:
        """Where the dropped turns can still be read, if anywhere.

        ``EventCompaction`` is content-free by design (§5), so the event
        carries a path and not a transcript. Best-effort: no writer, or a
        write that fails, costs a reference and nothing else.
        """
        if self.writer is None:
            return ""
        try:
            path = self.writer.write(transcript, tool_name="compaction")
            return self.writer.relative(path) if path is not None else ""
        except Exception as exc:  # noqa: BLE001
            log.debug(f"Could not archive the compacted transcript: {exc}")
            return ""


# -- small helpers ----------------------------------------------------------


def _budget(engine: Any) -> Optional[int]:
    getter = getattr(engine, "context_length", None)
    if getter is None:
        return None
    try:
        value = getter()
    except Exception:  # noqa: BLE001 - a backend probe must not fail a run
        return None
    return int(value) if value else None


def _tokens(engine: Any) -> Optional[int]:
    counter = getattr(engine, "count_prompt_tokens", None)
    if counter is None:
        return None
    try:
        return int(counter())
    except Exception:  # noqa: BLE001
        return None


def _estimate_chars(text: str) -> int:
    """The usual four-characters-a-token approximation, for a custom summarizer."""
    return max(1, len(text) // 4)
