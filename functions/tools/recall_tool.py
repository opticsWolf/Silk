# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Long-term memory for the silk agent: search over remembered turns (§17, D66).

The agent's own scrollback is bounded by the context window and dies with
the run. The history ledger is neither: a turn is a concept in an
append-only store under the sandbox root, so `recall` reaches across runs
and across sessions -- *"what did we conclude about the lexer last week"*
rather than *"what is still in the prompt"*.

Ranking is FTS5 keyword search today. §17's plan is hybrid (FTS5 plus
vectors fused by RRF); FTS5 needs no embedding model, so it ships first
and the result shape does not change when the vector half arrives.

Nothing here imports `macrame`. Tools talk to `functions/ledger.py` and
nothing else (D66), which is also what makes the missing-extra case a
plain refusal with a reason rather than an ImportError at load time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

# Absolute import: this module is exec'd by the ToolLoader as
# ``dynamic_tools.recall_tool`` (no parent package), so a ``..`` relative
# import would be "beyond top-level".
from weave.plugins.silk.functions.ledger import (
    DISTRIBUTION, HistoryLedger, KIND_RUN, KIND_TURN, LedgerUnavailable,
    available, unavailable_reason,
)

if TYPE_CHECKING:
    from ..tool_box import ToolBox
    from .file_sandbox import FileToolSandbox


# ── schemas ──────────────────────────────────────────────────────────────

class RecallArgs(BaseModel):
    query: str = Field(
        ..., min_length=1,
        description="What to look for, in plain words (keyword search).",
    )
    top_k: int = Field(
        5, gt=0, le=50,
        description="How many hits to return (most relevant first).",
    )
    scope: str = Field(
        "turns",
        description=(
            "'turns' searches remembered conversation turns; 'runs' searches "
            "run goals and summaries; 'all' searches both."
        ),
    )
    run_id: Optional[str] = Field(
        None,
        description="Optional: restrict the search to one run's history.",
    )


class RecallHit(BaseModel):
    id: str = Field(..., description="Ledger id of the remembered item.")
    kind: str = Field(..., description="'turn' or 'run'.")
    run_id: str = Field("", description="The run this came from.")
    role: str = Field("", description="Who spoke, for a turn.")
    at: str = Field("", description="When it was recorded (UTC ISO-8601).")
    text: str = Field("", description="The remembered text.")


class RecallResponse(BaseModel):
    ok: bool = Field(..., description="False when memory is unavailable.")
    query: str = Field(..., description="The query that was searched.")
    hits: list[RecallHit] = Field(
        default_factory=list, description="Matches, most relevant first.")
    total: int = Field(0, description="How many hits were returned.")
    message: str = Field("", description="Human-readable status or reason.")


_SCOPES = {
    "turns": (KIND_TURN,),
    "runs": (KIND_RUN,),
    "all": (KIND_TURN, KIND_RUN),
}


# ── attach ───────────────────────────────────────────────────────────────

def attach_recall_tool(toolbox: "ToolBox", sandbox: "FileToolSandbox",
                       history: Any = None) -> None:
    """Mount ``recall`` against this sandbox root's history ledger.

    *history* lets a node pass a ledger it already holds (one Write Actor
    per file is the whole concurrency model, D62); without one the tool
    opens the ledger under the sandbox's working root, which is where
    the run that wrote the turns put it.
    """
    ledger = history
    if ledger is None and available():
        ledger = HistoryLedger(getattr(sandbox, "root_dir", "."))
    toolbox._history_ledger = ledger  # type: ignore[attr-defined]

    note = "" if available() else (
        f"\n- NOTE: memory is unavailable ({DISTRIBUTION} is not installed); "
        "this tool will say so rather than return stale or empty results."
    )

    @toolbox.register(
        name="recall",
        tags=("memory", "search"), category="search", risk="low",
        description=(
            "Search your own memory of earlier turns and runs -- including "
            "ones from previous sessions and ones dropped from the current "
            "context by compaction. Use it before asking the user to repeat "
            "something, or to check what was already tried."
        ),
        args_model=RecallArgs,
        procedure=(
            "Search remembered history (keyword ranked).\n"
            "- query: plain words; scope: 'turns' (default), 'runs', 'all'.\n"
            "- run_id: optional, restricts to one run.\n"
            "- Hits carry {id, kind, run_id, role, at, text}: quote them "
            "rather than paraphrasing from memory.\n"
            "- An empty result means nothing was remembered, not that it "
            "never happened -- memory starts when the ledger does." + note
        ),
    )
    def _recall(
        db_pool: Any, user_session: dict,
        query: str, top_k: int = 5, scope: str = "turns",
        run_id: Optional[str] = None,
    ) -> RecallResponse:
        if ledger is None:
            return RecallResponse(
                ok=False, query=query, message=unavailable_reason(),
            )
        kinds = _SCOPES.get(str(scope).lower(), (KIND_TURN,))
        try:
            hits = ledger.recall(query, top_k=int(top_k), kinds=kinds)
        except LedgerUnavailable as exc:
            return RecallResponse(ok=False, query=query, message=str(exc))
        except Exception as exc:  # noqa: BLE001 - a tool reports, never raises
            return RecallResponse(
                ok=False, query=query,
                message=f"recall failed: {type(exc).__name__}: {exc}",
            )

        if run_id:
            hits = [h for h in hits if h.get("run_id") == run_id]
        rows = [
            RecallHit(
                id=h.get("id", ""), kind=h.get("kind", ""),
                run_id=h.get("run_id", ""), role=h.get("role", ""),
                at=h.get("at", ""), text=str(h.get("text", ""))[:4000],
            )
            for h in hits
        ]
        return RecallResponse(
            ok=True, query=query, hits=rows, total=len(rows),
            message=("" if rows else
                     "nothing remembered matches that; try fewer words"),
        )
