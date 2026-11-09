# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Query and document prefixes for embedding models on okf's external lane.

Most retrieval embedders are trained to read a query and a passage
differently, and they are told which one they read by a prefix on the
text: ``query: `` / ``passage: `` for e5, an instruction before the query
for bge and Qwen3-Embedding. okf's crate lane does this itself; on the
external lane it adds only the prefixes it is *declared*, and it knows a
table for nomic-v1.5 alone. Without one, queries and documents embed
alike and search quietly gets worse.

:func:`prefixes_for` is the table for the families Silk knows, matched on
the model's name; :func:`resolve` applies an Embedder Loader's override
(``auto``, ``none``, or ``Query=...|Document=...``). okf pins the model
id and not the prefixes, so :func:`pin_tag` names a table for the pin:
a graph is never searched with prefixes its vectors were made without.

embeddinggemma is not here: its prefixes hold ``|``, which okf's
``Task=prefix|Task=prefix`` declaration cannot carry.
"""

import hashlib
import re
from typing import Dict, Sequence, Tuple

#: Override values the Embedder Loader offers besides a typed table.
AUTO = "auto"
NONE = "none"

#: The sides okf prefixes.
SIDES = ("Query", "Document")

_REPRESENT = "Represent this sentence for searching relevant passages: "
_QWEN = ("Instruct: Given a web search query, retrieve relevant passages "
         "that answer the query\nQuery:")

#: ``(name pattern, table)``, first match wins. Patterns run on the
#: lower-cased model name with ``_`` read as ``-``.
KNOWN: Sequence[Tuple[str, Dict[str, str]]] = (
    (r"nomic-embed-text", {"Query": "search_query: ",
                           "Document": "search_document: "}),
    (r"e5-mistral", {"Query": _QWEN + " "}),
    (r"(^|[^a-z0-9])e5-", {"Query": "query: ", "Document": "passage: "}),
    (r"arctic-embed-[ml]-v2", {"Query": "query: "}),
    (r"arctic-embed", {"Query": _REPRESENT}),
    (r"mxbai-embed-large", {"Query": _REPRESENT}),
    (r"bge-(small|base|large)-en", {"Query": _REPRESENT}),
    (r"qwen3-embedding|gte-qwen2", {"Query": _QWEN}),
)


def prefixes_for(name: str) -> Dict[str, str]:
    """The table for the model called *name*, or ``{}`` (no prefixes)."""
    text = (name or "").lower().replace("_", "-")
    for pattern, table in KNOWN:
        if re.search(pattern, text):
            return dict(table)
    return {}


def parse(text: str) -> Dict[str, str]:
    """``Query=...|Document=...`` as a table; raises ValueError."""
    table: Dict[str, str] = {}
    for pair in text.split("|"):
        if not pair.strip():
            continue
        side, sep, prefix = pair.partition("=")
        side = side.strip()
        if not sep or side not in SIDES:
            raise ValueError(
                f"Prefixes are Query=...|Document=...; not {pair!r}.")
        table[side] = prefix
    return table


def resolve(override: str, name: str) -> Dict[str, str]:
    """The table an override asks for: *name*'s own on ``auto`` (or
    nothing typed), none on ``none``, else the typed one."""
    text = (override or "").strip()
    if not text or text.lower() == AUTO:
        return prefixes_for(name)
    if text.lower() == NONE:
        return {}
    return parse(override)


def declaration(table: Dict[str, str]) -> str:
    """The table as okf's ``embed_prefixes`` row, ``""`` for none."""
    return "|".join(f"{side}={table[side]}" for side in SIDES if table.get(side))


def pin_tag(table: Dict[str, str]) -> str:
    """A short name for a table, for the graph's model pin."""
    text = declaration(table)
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
