# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The kinds of knowledge Silk keeps, in one place.

The same four words are the ``type`` an agent gives ``okf_ingest``, the
categories the ``findings_gate`` hook sorts a user's decisions into, and
the choice the ``classify_ingest`` hook makes for an untyped thought -- so
a concept the classifier filed and one the agent filed can be searched
for with the same ``concept_type``.

A thought nobody typed keeps OKFgraph's own default, ``thought``: an
honest "not classified", rather than a guess dressed as one of the four.
"""
from __future__ import annotations

#: kind -> what it means, as the classifier is told it.
KNOWLEDGE_KINDS: dict[str, str] = {
    "dogma": "a value or principle the user holds",
    "rule": "a standing rule for how work should be done",
    "evidence": "an observed fact about this project",
    "decision": "a choice made for a specific situation that will recur",
}

#: What an untyped thought is stored as (OKFgraph's default).
UNCLASSIFIED = "thought"

#: The kinds a triage reads as rules: what the user holds and what they set.
RULE_KINDS: tuple[str, ...] = ("rule", "dogma")


def kind_criteria(*, with_none: bool = True) -> dict[str, str | None]:
    """The classifier's ``choice`` criteria, optionally with a ``none``."""
    out: dict[str, str | None] = dict(KNOWLEDGE_KINDS)
    if with_none:
        out["none"] = "nothing worth keeping"
    return out
