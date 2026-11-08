# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The knowledge sink (S6) over a served OKFgraph.

What the approval triage and the ``findings_gate`` hook ask of "the
knowledge", answered by the graph the Knowledge Graph node serves:

``rules_for(tool)``     the titles of the rules and dogmas nearest a tool, for
                        the triage state's ``rules``;
``nearest_rule(text)``  the one rule or dogma nearest a decision, which the
                        classifier compares the decision against;
``write_finding(note)`` a finding the gate trusts: ingested as its kind, and
                        linked to the rule it supports or supersedes;
``offer_finding(note)`` a borderline one: ingested too, but tagged
                        ``proposed`` so the user can keep or delete it --
                        nothing waits on an answer.

A graph the box may only read still answers the first two; the writes are
skipped and logged. Every call is best-effort: an OKF error costs the
answer, never the run.
"""
from __future__ import annotations

from typing import Any, Optional

from weave.logger import get_logger

from .knowledge_kinds import KNOWLEDGE_KINDS, RULE_KINDS
from .okf_graph import OkfError

log = get_logger("SilkOkfKnowledge")

#: Where the knowledge sink sits on a toolbox (read by the gate and hooks).
KNOWLEDGE_ATTR = "_knowledge"
#: The topic findings are filed under.
FINDINGS_TOPIC = "user-decisions"
FINDINGS_TAG = "findings_gate"
PROPOSED_TAG = "proposed"


class OkfKnowledge:
    """``rules_for`` / ``nearest_rule`` / ``write_finding`` / ``offer_finding``."""

    def __init__(self, graph: Any, *, can_write: bool = True) -> None:
        self.graph = graph
        self.can_write = can_write

    def _search(self, query: str, kind: str, limit: int) -> list[dict]:
        try:
            rows = self.graph.call("search", query=query, target="concepts",
                                   concept_type=kind, limit=limit)
        except (OkfError, OSError) as exc:
            log.debug(f"Knowledge search failed: {exc}")
            return []
        if isinstance(rows, dict):
            rows = rows.get("results") or []
        return [r for r in rows if isinstance(r, dict) and r.get("id")]

    def _rules(self, query: str, limit: int) -> list[dict]:
        rows = [r for kind in RULE_KINDS for r in self._search(query, kind, limit)]
        rows.sort(key=lambda r: -float(r.get("score") or 0.0))
        return [r for r in rows
                if "superseded" not in (r.get("tags") or ())][:limit]

    def rules_for(self, tool: str, limit: int = 6) -> list[str]:
        return [str(r.get("title") or r["id"]) for r in self._rules(tool, limit) if tool]

    def nearest_rule(self, text: str) -> Optional[dict]:
        if not str(text or "").strip():
            return None
        found = self._rules(text, 1)
        if not found:
            return None
        r = found[0]
        return {"id": r["id"], "title": r.get("title") or "",
                "text": str(r.get("body") or r.get("description") or "")[:600]}

    def write_finding(self, note: dict) -> Optional[str]:
        return self._ingest(note, proposed=False)

    def offer_finding(self, note: dict) -> Optional[str]:
        return self._ingest(note, proposed=True)

    def _ingest(self, note: dict, *, proposed: bool) -> Optional[str]:
        if not self.can_write:
            log.info("Finding not filed: this box may only read the knowledge graph.")
            return None
        kind = str(note.get("category") or "")
        body = str(note.get("body") or "").strip()
        if kind not in KNOWLEDGE_KINDS or not body:
            return None
        relations = dict(note.get("relations") or {})
        related = [cid for rel, cid in relations.items() if rel in ("supports", "contradicts")]
        tags = [FINDINGS_TAG] + ([PROPOSED_TAG] if proposed else [])
        try:
            made = self.graph.call(
                "ingest", kind="thoughts", thoughts=body, topic=FINDINGS_TOPIC,
                title=str(note.get("title") or "")[:80] or None, type=kind, tags=tags,
                related=related or None)
            new_id = str((made or {}).get("concept_id") or (made or {}).get("id") or "")
            old = relations.get("supersedes")
            if new_id and old and not proposed:
                self.graph.call("supersede", old_id=old, new_id=new_id)
            return new_id or None
        except (OkfError, OSError) as exc:
            log.warning(f"Finding not filed in the knowledge graph: {exc}")
            return None
