# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Spill: keep a huge tool result out of the context (spec D41 option A, D57).

A tool result the model does not need in full still costs the whole run:
it is appended to history and re-sent, in full, on every subsequent
request. The spill hook rewrites the result **before it is appended** --
the full text goes to a file inside the sandbox, and what the model sees
is a head, a tail, and the path.

**Why this is the mechanism that lands first.** Compaction (D24/D25)
rewrites the *head* of the context, which collapses the longest common
prefix with the previous request to roughly the system prompt and forces a
full re-prefill of a context that is, by construction, near the ceiling --
twice, since the summary is itself a model request with a different prompt
(D41). Spill touches only the newest message, so history stays append-only
and invariant I11 holds: it defers compaction rather than causing it.

**Fan-out first (D57).** ``delegate_parallel`` returns every worker's full
answer inline, in one round: eight workers times a long answer is the
single largest result Silk can produce, and an orchestrator that compacts
pays D41's double prefill at the worst possible moment. So the hook
understands the delegation result shapes structurally and spills each
worker's answer separately, rather than truncating one opaque blob -- the
model keeps the per-worker framing (who answered, whether it worked) and
loses only the bulk.

**What the model is left with must be actionable.** A truncated result
that does not say where the rest went is worse than no truncation: the
model cannot tell a short answer from an amputated one. Every preview
names its file, and the file is inside the sandbox, so reading it back is
an ordinary tool call the agent already has.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

from weave.logger import get_logger

from .hooks import HOOK_WRAP_TOOL_EXECUTE

log = get_logger("SilkSpill")

#: Where spilled results go, relative to the sandbox root. Inside the tree
#: the agent can read, because a path it cannot open is not a reference.
SPILL_DIR = ".silk/spill"

#: Results shorter than this are left alone. Below a few kilobytes the
#: bookkeeping (a file, a path, a sentence explaining it) costs more
#: context than the text it replaces.
DEFAULT_THRESHOLD = 4000

#: How much of the result survives inline. The head carries the shape of
#: the answer, the tail carries its conclusion -- which is the half a
#: summary usually loses.
DEFAULT_HEAD = 800
DEFAULT_TAIL = 400

_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(text: str, limit: int = 40) -> str:
    return (_SLUG.sub("-", str(text)).strip("-") or "result")[:limit]


class SpillWriter:
    """Writes spilled text under one sandbox root, numbering as it goes.

    One per attached hook, not one per call, so the numbering is stable
    within a run and two spills in one batch cannot collide on a name.
    """

    def __init__(self, root: Optional[str | Path], *, subdir: str = SPILL_DIR):
        self.root = Path(root).resolve() if root else None
        self.subdir = subdir
        self._n = 0
        #: Every path written, for tests and for a UI that wants to list them.
        self.written: list[Path] = []

    @property
    def available(self) -> bool:
        return self.root is not None

    def write(self, text: str, *, tool_name: str, label: str = "") -> Optional[Path]:
        """Write *text*; return its path, or ``None`` if it could not be.

        A failure here must not fail the tool call: the result is still
        correct, merely large. The caller falls back to leaving it inline,
        which is the behaviour of not having the hook at all.
        """
        if self.root is None:
            return None
        self._n += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        name = f"{stamp}-{_slug(tool_name)}-{self._n:02d}"
        if label:
            name = f"{name}-{_slug(label)}"
        path = self.root / self.subdir / f"{name}.md"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            log.warning(f"Could not spill a {tool_name} result to '{path}': {exc}")
            return None
        self.written.append(path)
        return path

    def relative(self, path: Path) -> str:
        """The path as the model should see it: relative to the sandbox root."""
        if self.root is None:
            return str(path)
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)


def preview(text: str, path_text: str, *, head: int, tail: int) -> str:
    """Head, tail, and where the rest is.

    The marker is deliberately verbose about *why* there is a gap: a model
    that reads "[... truncated ...]" has no way to know whether the missing
    part mattered, and no way to go and find out.
    """
    kept_head = text[:head].rstrip()
    kept_tail = text[len(text) - tail:].lstrip() if tail else ""
    dropped = len(text) - len(kept_head) - len(kept_tail)
    marker = (
        f"\n\n[... {dropped} characters omitted. The complete result was "
        f"written to `{path_text}` -- read that file if you need the part "
        f"that is not shown here ...]\n\n"
    )
    return f"{kept_head}{marker}{kept_tail}".strip()


def _spill_text(
    text: str, writer: SpillWriter, *, tool_name: str, label: str,
    threshold: int, head: int, tail: int,
) -> str:
    """Replace *text* with a preview, or return it unchanged."""
    if len(text) <= threshold:
        return text
    path = writer.write(text, tool_name=tool_name, label=label)
    if path is None:
        return text          # no sandbox, or the write failed: leave it whole
    return preview(text, writer.relative(path), head=head, tail=tail)


def attach_spill_hook(
    toolbox: Any,
    sandbox: Any = None,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    head: int = DEFAULT_HEAD,
    tail: int = DEFAULT_TAIL,
    tools: tuple[str, ...] = ("delegate", "delegate_parallel"),
    writer: Optional[SpillWriter] = None,
) -> Optional[Any]:
    """Register the spill middleware; returns its :class:`HookEntry`.

    Bound to *tools* (D13) -- by default the two delegation tools, which
    D57 identifies as the dominant context-growth term. Pass ``tools=()``
    to spill every tool's result over the threshold.

    Returns ``None`` when there is no sandbox root to write into: spilling
    into a directory the agent cannot read back would replace a large
    result with a dangling reference, which is strictly worse than the
    large result.
    """
    spill = writer or SpillWriter(getattr(sandbox, "root_dir", None))
    if not spill.available:
        log.debug("spill hook not attached: no sandbox root to write into")
        return None

    def _rewrite(value: str, label: str, tool_name: str) -> str:
        return _spill_text(value, spill, tool_name=tool_name, label=label,
                           threshold=threshold, head=head, tail=tail)

    def _spill_delegation(output: Any, tool_name: str) -> bool:
        """Rewrite the answer fields of a delegation result in place.

        Returns whether *output* was one -- structural, not stringly: the
        per-worker framing (who ran, whether it worked, which tools it
        used) is small and worth keeping, and only ``answer`` is ever the
        large field.
        """
        results = getattr(output, "results", None)
        if isinstance(results, list) and results:
            for item in results:
                answer = getattr(item, "answer", None)
                if isinstance(answer, str):
                    item.answer = _rewrite(
                        answer, str(getattr(item, "worker", "")), tool_name)
            return True
        answer = getattr(output, "answer", None)
        if isinstance(answer, str):
            output.answer = _rewrite(
                answer, str(getattr(output, "worker", "")), tool_name)
            return True
        return False

    async def spill_hook(
        handler: Callable = None, tool_name: str = "",
        tool_args: Optional[dict] = None, **_kw: Any,
    ) -> Any:
        output = await handler()
        try:
            if isinstance(output, BaseModel):
                if _spill_delegation(output, tool_name):
                    return output
                return _rewrite(output.model_dump_json(), "", tool_name)
            if isinstance(output, str):
                return _rewrite(output, "", tool_name)
            return _rewrite(json.dumps(output, default=str), "", tool_name)
        except Exception as exc:  # noqa: BLE001 - never fail a call over size
            log.warning(f"spill hook left the {tool_name} result inline: {exc}")
            return output

    return toolbox.hooks.register_middleware(
        HOOK_WRAP_TOOL_EXECUTE, spill_hook, tools=tools,
    )
