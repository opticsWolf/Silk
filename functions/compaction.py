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

**What triggers it (D24).** Pressure at the pre-request seam, when the
estimated input crosses ``context - reserve``; and an overflow -- either a
*classified* overflow error from a model request (D40), or an answer that
finished with ``length`` short of its token cap, which is the same overflow
arriving as a finish instead of a refusal. Never a generic stream error,
because a dead ``llama_cpp.server`` produces one of those too, and
answering that by spending a summarization request against the same dead
server is the failure mode the classifier exists to prevent. These are the
*automatic* triggers, and the ToolBox's ``auto_compact`` hook switches them
(:func:`compactor_for`). A caller may also ask -- :meth:`Compactor.compact_now`,
or the model through the ``compact_context`` tool -- which skips only the
threshold and works whether or not the hook is ticked.

**What it keeps.** A tail sized in tokens (a share of the window), snapped
to a round; a summary in fixed sections, so the next compaction can update
it rather than re-tell it -- an earlier summary is handed to the update
prompt whole, never re-summarized as one more clipped turn; and the files
the dropped turns read and modified, recorded by code from the tool calls
and appended outside the model's text. The summarizer sees each call's name
and arguments, not only its result.

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
summarizable prefix, a summarizer that errors, refuses, returns empty or
is cut off,
an engine with no rewrite operation -- returns ``None`` and leaves the
history exactly as it was. The pre-existing ``EventUsageLimit`` /
``EventError`` path then protects the run as it did before. Compaction
never kills a run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from weave.logger import get_logger

from .stream_events import EventCompaction

log = get_logger("SilkCompaction")

#: Why a compaction ran. Pressure consults the threshold; overflow and
#: manual compactions do not (the backend, or the user, already decided).
REASON_PRESSURE = "pressure"
REASON_OVERFLOW = "overflow"
REASON_MANUAL = "manual"

#: Headroom kept below the context window, as a fraction of it and as an
#: absolute floor -- whichever is larger. The generation still has to fit,
#: and so does the round of tool results that follows it.
DEFAULT_RESERVE_RATIO = 0.25
DEFAULT_RESERVE = 1024

#: How much of the window the verbatim tail may hold, when the window is
#: known. A token budget rather than a turn count: eight turns can be eight
#: one-line answers or eight 50k-token file reads, and a tail that already
#: fills the window frees nothing and buys a second compaction two rounds
#: later. Generous on purpose (D41): the recent turns are what the model is
#: actually working from.
DEFAULT_KEEP_RECENT_RATIO = 0.30

#: Turns kept verbatim at the tail when the window is unknown (a forced
#: overflow compaction on a backend that reports no context length), or
#: when a caller pins a turn count explicitly.
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

#: Cap on one tool call's rendered arguments. A write_file call carries the
#: whole file; the summarizer needs to know which file, not its bytes.
DEFAULT_MAX_CALL_ARG_CHARS = 400

SUMMARY_SYSTEM = (
    "You compress conversation transcripts for an autonomous agent that "
    "will keep working from your summary alone. You never invent facts and "
    "you never address the user. Do not continue the conversation and do "
    "not answer questions in it: output only the summary."
)

#: The fixed sections. A stable shape is what lets the *next* compaction
#: merge into this one instead of re-telling it, and what makes a missing
#: section visible rather than silently absent.
SUMMARY_TEMPLATE = """\
## Task
[What the agent was asked to do. Several items if the work changed.]

## Constraints & preferences
- [Requirements, limits and preferences that were stated, or "(none)"]

## Progress
### Done
- [x] [Completed steps and changes]

### In progress
- [ ] [The step that was under way]

### Blocked
- [What is stuck and why, or "(none)"]

## Key decisions
- **[Decision]**: [the reason]

## Facts discovered
- [File paths, names, values, tool results and errors the work depends on]

## Next steps
1. [What to do next, in order]"""

SUMMARY_RULES = (
    "Keep every section short. Preserve exact file paths, function and node "
    "names, values and error messages. Omit pleasantries, restatements and "
    "anything the agent can rediscover trivially. Write it as notes to "
    "itself, not as a report to a reader."
)

SUMMARY_PROMPT = """\
Below is the earlier part of an agent's working transcript. It is about to \
be removed from the agent's context to make room, and your summary is all \
that will remain of it.

--- transcript begins ---
{transcript}
--- transcript ends ---

Write the summary in exactly this format:

{template}

{rules}{focus}
"""

SUMMARY_UPDATE_PROMPT = """\
Below are the agent's existing summary of its earlier work, and the part of \
its transcript that came after it. Both are about to be removed from the \
agent's context, and your updated summary is all that will remain of them.

--- previous summary begins ---
{previous}
--- previous summary ends ---

--- transcript begins ---
{transcript}
--- transcript ends ---

Update the summary. Rules:
- PRESERVE everything in the previous summary that still holds
- ADD the progress, decisions and facts from the new transcript
- MOVE items from "In progress" to "Done" when they were completed
- REWRITE "Next steps" for where the work now stands
- Remove an item only when the transcript shows it no longer matters

Write the summary in exactly this format:

{template}

{rules}{focus}
"""

#: How the summary re-enters the history. Labelled, because an unlabelled
#: summary reads to the model as something it said or was told verbatim --
#: and because the next compaction finds the previous summary by it.
SUMMARY_MARKER = "[Earlier conversation, summarized to free context]\n\n"

#: Files a dropped turn touched, recorded by code rather than by the model,
#: so they cannot be summarized away. Tool name -> (argument, how) pairs.
FILE_TOOL_ARGS: dict[str, tuple[tuple[str, str], ...]] = {
    "read_file": (("path", "read"),),
    "view_file": (("path", "read"),),
    "file_info": (("path", "read"),),
    "diff_files": (("file_a", "read"), ("file_b", "read")),
    "write_file": (("path", "modified"),),
    "append_file": (("path", "modified"),),
    "edit_file": (("path", "modified"),),
    "insert_text": (("path", "modified"),),
    "delete_file": (("path", "modified"),),
    "copy_file": (("source", "read"), ("destination", "modified")),
    "move_file": (("source", "modified"), ("destination", "modified")),
}

_FILE_BLOCK = re.compile(
    r"\n*<(read-files|modified-files)>\n(.*?)\n?</\1>", re.DOTALL)
_TOOL_FENCE = re.compile(r"```tool_call\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


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
    keep_tokens: Optional[int] = None,
) -> Optional[int]:
    """How many leading turns to drop, or ``None`` if compaction is moot.

    With *keep_tokens*, the tail is the newest turns whose estimated size
    fits in that many tokens (always at least the newest turn); without it,
    the last *keep_recent* turns. Either target is then snapped **down** to
    a round boundary -- down rather than up, so the snap always errs toward
    keeping more recent context than asked for, never less.
    """
    if keep_tokens is not None:
        target = len(history)
        used = 0
        for i in range(len(history) - 1, -1, -1):
            used += estimate_entry_tokens(history[i])
            if used > keep_tokens:
                break
            target = i
        target = min(target, len(history) - 1)
    else:
        target = len(history) - max(0, keep_recent)
    if target < min_dropped:
        return None
    candidates = [i for i in round_boundaries(history) if 0 < i <= target]
    if not candidates:
        return None
    cut = max(candidates)
    return cut if cut >= min_dropped else None


def estimate_entry_tokens(entry: dict[str, Any]) -> int:
    """One history turn's size, at the usual four characters a token."""
    entry = entry or {}
    size = len(str(entry.get("content", "")))
    calls = entry.get("tool_calls")
    if calls:
        try:
            size += len(json.dumps(calls, default=str))
        except (TypeError, ValueError):
            size += len(str(calls))
    return max(1, size // 4)


def is_summary(entry: dict[str, Any]) -> bool:
    """Whether *entry* is a summary an earlier compaction put there."""
    return str((entry or {}).get("content", "")).startswith(SUMMARY_MARKER)


def tool_calls_of(entry: dict[str, Any]) -> list[tuple[str, str]]:
    """``(name, arguments-json)`` for every call an assistant turn made.

    Both transports: the native one keeps them in ``tool_calls``, beside
    the content; the fence one keeps them inside the content.
    """
    entry = entry or {}
    out: list[tuple[str, str]] = []
    for call in entry.get("tool_calls") or ():
        fn = (call or {}).get("function") or {}
        name = fn.get("name") or (call or {}).get("name")
        if not name:
            continue
        args = fn.get("arguments", (call or {}).get("arguments", "{}"))
        if not isinstance(args, str):
            try:
                args = json.dumps(args, default=str)
            except (TypeError, ValueError):
                args = str(args)
        out.append((str(name), args))
    if entry.get("role") == "assistant":
        for block in _TOOL_FENCE.findall(str(entry.get("content", ""))):
            try:
                data = json.loads(block.strip())
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            name = data.get("name") or data.get("tool")
            if not name or not isinstance(name, str):
                continue
            args = data.get("arguments", data.get("args", {}))
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, default=str)
                except (TypeError, ValueError):
                    args = "{}"
            out.append((name, args))
    return out


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return (f"{text[:half].rstrip()}\n[... {omitted} characters "
            f"omitted ...]\n{text[-half:].lstrip()}")


def render_transcript(
    entries: Sequence[dict[str, Any]],
    *,
    max_turn_chars: int = DEFAULT_MAX_TURN_CHARS,
    max_arg_chars: int = DEFAULT_MAX_CALL_ARG_CHARS,
) -> str:
    """The dropped turns as plain text for the summarizer.

    An assistant turn's tool calls are rendered as their own line, name and
    arguments, so the summarizer knows *which* file was read and not only
    that something came back. Fenced calls are lifted out of the prose
    first, so clipping a long turn cannot cut a call in half.
    """
    lines: list[str] = []
    for raw in entries:
        entry = raw or {}
        role = str(entry.get("role", "user"))
        content = str(entry.get("content", ""))
        calls = tool_calls_of(entry)
        if role == "assistant":
            content = _TOOL_FENCE.sub("", content).strip()
        content = _clip(content, max_turn_chars)
        if role == "tool":
            role = f"tool result ({entry.get('name', 'unknown')})"
        if content or not calls:
            lines.append(f"{role}: {content}")
        if calls:
            rendered = "; ".join(
                f"{name}({_clip(args, max_arg_chars)})" for name, args in calls
            )
            lines.append(f"{role} tool calls: {rendered}")
    return "\n\n".join(lines)


def file_operations(
    entries: Iterable[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """``(read, modified)`` paths the given turns' tool calls touched."""
    read: set[str] = set()
    modified: set[str] = set()
    for entry in entries:
        for name, args in tool_calls_of(entry):
            spec = FILE_TOOL_ARGS.get(name)
            if not spec:
                continue
            try:
                data = json.loads(args) if args else {}
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            for key, how in spec:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    (modified if how == "modified" else read).add(value.strip())
    return read, modified


def split_file_lists(summary: str) -> tuple[str, set[str], set[str]]:
    """A stored summary's text, and the file lists code appended to it."""
    read: set[str] = set()
    modified: set[str] = set()

    def _take(match: "re.Match[str]") -> str:
        bucket = read if match.group(1) == "read-files" else modified
        bucket.update(p.strip() for p in match.group(2).splitlines() if p.strip())
        return ""

    return _FILE_BLOCK.sub(_take, summary).strip(), read, modified


def format_file_lists(read: set[str], modified: set[str]) -> str:
    """The file lists as they are appended to a summary (empty if none)."""
    read = read - modified
    parts: list[str] = []
    if read:
        parts.append("<read-files>\n" + "\n".join(sorted(read)) + "\n</read-files>")
    if modified:
        parts.append("<modified-files>\n" + "\n".join(sorted(modified))
                     + "\n</modified-files>")
    return "\n\n".join(parts)


def build_summary_prompt(
    transcript: str, previous: str = "", focus: str = "",
) -> str:
    """The summarization request's user message."""
    focus_line = f"\n\nAdditional focus: {focus.strip()}" if focus.strip() else ""
    if previous.strip():
        return SUMMARY_UPDATE_PROMPT.format(
            previous=previous.strip(), transcript=transcript,
            template=SUMMARY_TEMPLATE, rules=SUMMARY_RULES, focus=focus_line,
        )
    return SUMMARY_PROMPT.format(
        transcript=transcript, template=SUMMARY_TEMPLATE,
        rules=SUMMARY_RULES, focus=focus_line,
    )


# -- the compactor ----------------------------------------------------------


@dataclass
class Compactor:
    """Decides when to compact, and does it atomically when it does.

    Held by the :class:`~.agent_loop.AgentLoop` as an optional collaborator,
    exactly like ``output_validator``: the loop owns the turn, the engine
    owns one request, and the compactor owns the summarization request and
    the swap. A loop without one behaves precisely as it did before.
    """

    #: Turns kept verbatim. ``None`` (the default) sizes the tail by tokens
    #: instead -- *keep_recent_ratio* of the window -- and falls back to
    #: :data:`DEFAULT_KEEP_RECENT` turns only when the window is unknown.
    #: An explicit count pins the tail to that many turns.
    keep_recent: Optional[int] = None
    keep_recent_ratio: float = DEFAULT_KEEP_RECENT_RATIO
    min_dropped: int = DEFAULT_MIN_DROPPED
    reserve: int = DEFAULT_RESERVE
    reserve_ratio: float = DEFAULT_RESERVE_RATIO
    hysteresis: float = DEFAULT_HYSTERESIS
    max_turn_chars: int = DEFAULT_MAX_TURN_CHARS
    #: Whether the context itself may trigger a compaction: pressure at the
    #: pre-request seam, an overflow error, an early ``length`` finish. Off,
    #: only an explicit request (:meth:`compact_now`, the ``compact_context``
    #: tool) compacts -- that is the ToolBox's ``auto_compact`` hook unticked.
    auto: bool = True
    #: Steers what the summary keeps ("Additional focus: ..."), for every
    #: compaction of this run. :meth:`compact_now` can pass a one-off focus.
    focus: str = ""
    #: ``(prompt) -> str``: replaces the model request (a cheaper model, a
    #: test double). It receives the whole summarization prompt -- the
    #: transcript, the previous summary when there is one, and the format.
    #: Defaults to the agent's own model (D25).
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
        focus: Optional[str] = None,
    ) -> Optional[EventCompaction]:
        """Compact if warranted; return the event, or ``None`` if not.

        ``force`` is the overflow retry (D24's second trigger): the backend
        has already said the prompt does not fit, so its verdict outranks
        our estimate and the threshold is not consulted. Everything else --
        the boundary rule, the summary, the atomic swap -- is identical,
        because an overflow is not a licence to cut carelessly.
        """
        if not self.auto and reason != REASON_MANUAL:
            return None
        if not force and not self.should_compact(
            engine, context_length=context_length
        ):
            return None
        budget = context_length if context_length is not None else _budget(engine)
        try:
            return self._compact(engine, reason=reason, budget=budget,
                                 focus=self.focus if focus is None else focus)
        except Exception as exc:  # noqa: BLE001 - compaction never kills a run
            log.warning(f"Compaction ({reason}) failed and was skipped: {exc}")
            return None

    def compact_now(
        self,
        engine: Any,
        *,
        focus: Optional[str] = None,
        context_length: Optional[int] = None,
    ) -> Optional[EventCompaction]:
        """Compact because someone asked, not because the context is full.

        The threshold and the hysteresis are skipped -- the asker has
        decided the cost is worth it -- but the boundary rule, the minimum
        worth dropping and the atomic swap are not. *focus* steers this one
        summary, in place of :attr:`focus`.
        """
        return self.maybe_compact(
            engine, reason=REASON_MANUAL, force=True,
            context_length=context_length, focus=focus,
        )

    def keep_tokens(self, budget: Optional[int]) -> Optional[int]:
        """The verbatim tail's token budget, or ``None`` to count turns."""
        if self.keep_recent is not None or not budget or budget <= 0:
            return None
        return max(1, int(budget * self.keep_recent_ratio))

    def _compact(
        self, engine: Any, *, reason: str, budget: Optional[int] = None,
        focus: str = "",
    ) -> Optional[EventCompaction]:
        history = getattr(engine, "history", None)
        replace = getattr(engine, "replace_history_prefix", None)
        if not history or replace is None:
            raise CompactionUnavailable(
                "this engine has no rewritable history (spec D25 shape)"
            )
        keep_recent = (self.keep_recent if self.keep_recent is not None
                       else DEFAULT_KEEP_RECENT)
        cut = plan_cut(history, keep_recent=keep_recent,
                       min_dropped=self.min_dropped,
                       keep_tokens=self.keep_tokens(budget))
        if cut is None:
            log.debug("nothing to compact: no round boundary worth cutting at")
            return None

        # An earlier compaction's summary is carried forward *as a summary*:
        # handed to the update prompt whole, never re-summarized as one more
        # turn (which clipped it to max_turn_chars and lost its middle).
        dropping = list(history[:cut])
        previous_parts: list[str] = []
        read, modified = file_operations(dropping)
        for entry in dropping:
            if is_summary(entry):
                text, prev_read, prev_modified = split_file_lists(
                    str(entry.get("content", ""))[len(SUMMARY_MARKER):])
                read |= prev_read
                modified |= prev_modified
                if text:
                    previous_parts.append(text)
        previous = "\n\n".join(previous_parts)
        transcript = render_transcript(
            [e for e in dropping if not is_summary(e)],
            max_turn_chars=self.max_turn_chars,
        )
        prompt = build_summary_prompt(transcript, previous, focus)
        tokens_before = _tokens(engine)

        # The summary is produced *before* anything is dropped. If it fails,
        # the history is still whole and the run continues without a
        # compaction -- which is the pre-D24 behaviour, not a new failure.
        summary, summary_tokens = self._summarize(engine, prompt)
        summary = (summary or "").strip()
        if not summary:
            raise CompactionUnavailable("the summarizer returned nothing")
        # The file lists are the code's record, not the model's: a model
        # that echoes or invents them does not get to change them.
        summary, _, _ = split_file_lists(summary)
        files = format_file_lists(read, modified)
        if files:
            summary = f"{summary}\n\n{files}"

        archived = transcript if not previous else (
            f"--- previous summary ---\n{previous}\n\n--- transcript ---\n"
            f"{transcript}")
        ref = self._archive(archived)
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

    def _summarize(self, engine: Any, prompt: str) -> tuple[str, Optional[int]]:
        """Return the summary and what producing it cost to prefill."""
        if self.summarizer is not None:
            return self.summarizer(prompt), _estimate_chars(prompt)

        factory = getattr(engine, "sibling", None)
        if factory is None:
            raise CompactionUnavailable(
                "this engine cannot make a sibling for the summarization "
                "request; pass a summarizer instead"
            )
        sibling = factory(system_prompt=SUMMARY_SYSTEM)
        sibling.append_message("user", prompt)
        cost = _tokens(sibling)
        text = "".join(sibling.stream_response(dict(self.gen_params)))
        stats = getattr(sibling, "last_stats", None) or {}
        if stats.get("error"):
            raise CompactionUnavailable(
                f"the summarization request failed: {stats['error']}"
            )
        # A summary cut off by the token cap, or by a stream that ended
        # without a finish reason, is half a checkpoint. Swapping it in would
        # replace the history for good with something that stops mid-thought,
        # so it counts as no summary at all.
        if stats.get("truncated") or stats.get("finish_reason") == "length":
            raise CompactionUnavailable(
                "the summary was cut off before it finished"
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


# -- where the settings come from -------------------------------------------

#: Where the ToolBox's ``auto_compact`` hook leaves its settings, and where
#: the ``compact_context`` tool leaves its requests. Both on the toolbox,
#: because that is what a derived ToolSet rebuilds from its recipe and
#: what the Agent node is handed.
COMPACTION_CONFIG_ATTR = "compaction_config"
COMPACTION_REQUESTS_ATTR = "compaction_requests"

#: Where a tool pack names the tool the agent should use before a
#: compaction, and what to tell it: ``(tool_name, prompt)``. The knowledge
#: graph pack sets it to ``okf_ingest``, so the agent writes its findings
#: into the graph before the turns that hold them are summarized away.
PRE_COMPACTION_ATTR = "pre_compaction"


def compactor_for(toolbox: Any, role: Any = None) -> Compactor:
    """The run's compactor, configured from its toolbox and its role.

    No toolbox at all: the defaults, compacting automatically -- an agent
    without tools still runs into its context window. A toolbox without
    the ``auto_compact`` hook: the defaults with ``auto`` off, so only an
    explicit request (the ``compact_context`` tool) compacts. With it: the
    hook's settings. A Role that ticks ``auto_compact`` adjusts that for
    its own runs: its settings replace the box's, and it switches automatic
    compaction on even where the box left it off. A Role that does not
    tick it has no say, so it cannot switch the box's off.
    """
    cfg: Any = None
    if toolbox is not None:
        cfg = getattr(toolbox, COMPACTION_CONFIG_ATTR, None)
    names = role.hook_names() if hasattr(role, "hook_names") else frozenset()
    if "auto_compact" in names:
        from .hook_catalog import HOOK_CATALOG, resolve_config
        cfg = resolve_config(HOOK_CATALOG["auto_compact"],
                             role.hook_config("auto_compact")).model_dump()
    elif toolbox is None:
        return Compactor()
    if not isinstance(cfg, dict):
        return Compactor(auto=False)
    out = Compactor(auto=True)
    try:
        out.keep_recent_ratio = min(max(float(cfg.get("keep_recent_pct", 30)), 5), 60) / 100
        out.reserve_ratio = min(max(float(cfg.get("reserve_pct", 25)), 5), 50) / 100
        out.hysteresis = min(max(float(cfg.get("hysteresis_pct", 10)), 0), 50) / 100
        out.min_dropped = max(int(cfg.get("min_dropped", DEFAULT_MIN_DROPPED)), 1)
    except (TypeError, ValueError) as exc:
        log.warning(f"Compaction settings partly ignored ({exc}).")
    out.focus = str(cfg.get("focus") or "")
    return out


def take_compaction_request(toolbox: Any) -> Optional[str]:
    """Pop what ``compact_context`` asked for: its focus, or ``None``.

    Several calls in one round are one compaction; the last non-empty
    focus wins, because it is the model's latest word on what matters.
    """
    pending = getattr(toolbox, COMPACTION_REQUESTS_ATTR, None)
    if not pending:
        return None
    focus = next((f for f in reversed(pending) if f), "")
    pending.clear()
    return focus


def set_pre_compaction(toolbox: Any, tool: str, prompt: str) -> None:
    """Ask for one round with *tool* before each compaction of a run.

    Set by the pack that registers *tool*, on the toolbox, so a ToolSet
    replaying the recipe carries it along. Whether the round happens is
    :func:`pre_compaction_prompt`'s call, made when it would happen.
    """
    setattr(toolbox, PRE_COMPACTION_ATTR, (str(tool), str(prompt)))


def pre_compaction_prompt(toolbox: Any) -> Optional[str]:
    """What to tell the agent before compacting, or ``None`` to just compact.

    ``None`` unless a pack asked for the round *and* its tool would run if
    called: it is in this box (a ToolSet that left it out leaves the
    marker behind with the recipe) and the active role permits it. Asked
    at the moment of compaction rather than at build time, because the
    role is the run's, not the box's.
    """
    spec = getattr(toolbox, PRE_COMPACTION_ATTR, None) if toolbox is not None else None
    if not spec:
        return None
    tool, prompt = spec
    if tool not in (getattr(toolbox, "tools", None) or {}):
        return None
    permits = getattr(toolbox, "role_permits", None)
    if callable(permits) and not permits(tool):
        return None
    return prompt
