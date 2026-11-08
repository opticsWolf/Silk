# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

``compact_context``: the model asks for its own compaction (spec §12, T8
option B).

Automatic compaction answers the context window; this answers the model's
judgement. A run that has finished one phase of its work and is about to
start another knows better than any threshold that the old phase's raw
transcript is now dead weight -- and it can say what the summary must keep.

The tool does not compact. A tool runs in the middle of a round, between
an assistant turn and the results that answer it, and cutting the history
there would orphan its own result (I9). It records the request on the
toolbox; the loop takes it before the next model request and compacts
then, through the same :class:`~..compaction.Compactor` -- same boundary
rule, same atomic swap, same failure to nothing. It compacts whether or
not the ``auto_compact`` hook is ticked: that hook decides whether the
*context* may trigger compaction, and this is the model asking.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..compaction import (
    COMPACTION_REQUESTS_ATTR,
    REQUEST_INGEST_ATTR,
    coerce_ingest_mode,
)

if TYPE_CHECKING:
    from ..tool_box import ToolBox
    from .file_sandbox import FileToolSandbox


class CompactContextArgs(BaseModel):
    focus: str = Field(
        "",
        max_length=500,
        description=(
            "What the summary must keep, e.g. 'the three failing test names "
            "and their errors'. Empty for a general summary."
        ),
    )


class CompactContextResponse(BaseModel):
    ok: bool = Field(..., description="Whether the request was recorded.")
    message: str = Field("", description="What happens next.")


def attach_context_tools(toolbox: "ToolBox", sandbox: "FileToolSandbox" = None,
                         ingest_before: Any = None, **_kw: Any) -> None:
    """Mount ``compact_context`` on *toolbox*. *sandbox* is unused.

    *ingest_before* (``off`` / ``ask`` / ``require``) is what happens before
    a compaction this tool asks for, when the box has a Knowledge Graph the
    agent may write to: no round, a round asking it to save its findings,
    or a round that must call ``okf_ingest``. Set on the ToolBox's gear.
    """
    if not isinstance(getattr(toolbox, COMPACTION_REQUESTS_ATTR, None), list):
        setattr(toolbox, COMPACTION_REQUESTS_ATTR, [])
    setattr(toolbox, REQUEST_INGEST_ATTR, coerce_ingest_mode(ingest_before))

    @toolbox.register(
        name="compact_context",
        tags=("context", "memory"), category="context", risk="low",
        description=(
            "Summarize your older turns to free context, keeping the recent "
            "ones word for word. Use it between phases of a long task, when "
            "the earlier transcript is no longer needed verbatim. It costs a "
            "summarization request, so do not call it routinely."
        ),
        args_model=CompactContextArgs,
        procedure=(
            "Request a compaction before your next step.\n"
            "- focus: optional, what the summary must keep.\n"
            "- The summary replaces older turns permanently; anything you "
            "will need exactly (paths, values, errors) belongs in focus.\n"
            "- Dropped turns stay searchable with `recall` when memory is on."
        ),
    )
    def _compact_context(
        db_pool: Any, user_session: dict, focus: str = "",
    ) -> CompactContextResponse:
        pending = getattr(toolbox, COMPACTION_REQUESTS_ATTR, None)
        if not isinstance(pending, list):
            pending = []
            setattr(toolbox, COMPACTION_REQUESTS_ATTR, pending)
        pending.append(focus.strip())
        return CompactContextResponse(
            ok=True,
            message=(
                "Compaction requested. Older turns will be summarized before "
                "your next step, if there are enough of them to be worth it."
            ),
        )
