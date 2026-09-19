# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

A model handle that names its successor (D89).

D87 taught the loop to ask the *same* backend again, because a 429 passes.
This is the other half: what to do when it does not. A retry budget runs
out, or the failure was terminal from the first word -- the server is
gone, the key was revoked, the model was retired -- and the run ends
holding nothing, on a graph where a perfectly good local model was
sitting one node away.

The chain rides on the handle, under ``fallbacks``. That keeps every
consumer that only wants a model working exactly as before: a chained
handle *is* the primary handle, with one extra key that a plain
GraphEngine would ignore. Only the engine reads it, and only when the
current model has failed for good.

**A chain is as capable as its weakest link**, and that is the load-bearing
rule here. The transport is chosen once, before the first request, and
the system prompt is written to match it (see ``compose_system_prompt``);
the context window is the denominator compaction plans against. Neither
can be renegotiated mid-run -- the conversation has already been written
in one of them. So the chain answers those questions for *all* its
members at once:

* ``supports_tools`` only if every member does; one fence-only fallback
  puts the whole chain on fences.
* ``context_length`` is the smallest known one, because a switch must not
  hand a 32k conversation to an 8k model.
* a price only if every member quotes one -- a cost cap that stops
  binding halfway through a run is not a cap (D88).

Each of those costs something on the primary. That is the trade the node
makes explicit, and why fallback is a wire you draw rather than a default.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .pricing import price_from_handle

__all__ = [
    "FALLBACK_KEY",
    "build_chain",
    "chain_can_price",
    "chain_context_length",
    "chain_of",
    "chain_supports_tools",
    "describe_chain",
    "model_label",
]

#: Where the rest of the chain lives on a handle. A list of handles, each
#: already flattened -- nesting is resolved at build time so nothing
#: downstream has to recurse.
FALLBACK_KEY = "fallbacks"


def _is_handle(value: Any) -> bool:
    """Whether *value* is something a GraphEngine could run against.

    The same test the ``model_handle`` port applies, repeated here because
    this module is Qt-free and must not import the port registry.
    """
    return (isinstance(value, dict) and bool(value.get("backend"))
            and ("model" in value or "pool" in value))


def _identity(handle: Dict[str, Any]) -> Any:
    """What makes two handles the same *model*, for deduplication.

    The client object, when there is one: two nodes pointing at the same
    endpoint produce two dicts around one client, and falling back from a
    model to itself is the one arrangement that is certainly useless --
    the failure that exhausted the retries would simply happen again.
    """
    client = handle.get("model", handle.get("pool"))
    return id(client) if client is not None else id(handle)


def model_label(handle: Any) -> str:
    """How one model in a chain is named in a status line or a log.

    Backend first: what a reader needs from a fallback chain is *which
    kind of thing backs which*, and "gguf behind openai" says more about
    the run's cost and failure modes than two model names would.
    """
    if not isinstance(handle, dict):
        return "<no model>"
    backend = str(handle.get("backend") or "?")
    alias = (handle.get("model_alias") or handle.get("model_name")
             or handle.get("model_path") or "")
    name = str(alias).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return f"{backend}:{name}" if name else backend


def chain_of(handle: Any) -> List[Dict[str, Any]]:
    """Flatten *handle* into the list of models to try, in order.

    The head is the handle itself with ``fallbacks`` stripped, so the
    engine never re-reads a chain out of a member it is already walking.
    An ordinary handle flattens to a one-element list, which is what lets
    every caller treat "has a fallback" as the general case.
    """
    if not _is_handle(handle):
        return []
    head = {k: v for k, v in handle.items() if k != FALLBACK_KEY}
    chain: List[Dict[str, Any]] = [head]
    seen = {_identity(head)}
    for nxt in (handle.get(FALLBACK_KEY) or []):
        for member in chain_of(nxt):
            key = _identity(member)
            if key not in seen:
                seen.add(key)
                chain.append(member)
    return chain


def build_chain(*handles: Any) -> Optional[Dict[str, Any]]:
    """One handle standing for all of *handles*, tried in the order given.

    Returns the first usable handle, carrying the rest under
    ``fallbacks``. ``None`` when nothing usable was passed -- an
    unconnected endpoint upstream must not become an empty chain that
    fails at the agent instead of here.

    Nested chains are flattened and duplicates dropped, so wiring two
    fallback nodes in series composes the way the canvas reads it.
    """
    chain: List[Dict[str, Any]] = []
    seen: set = set()
    for handle in handles:
        for member in chain_of(handle):
            key = _identity(member)
            if key not in seen:
                seen.add(key)
                chain.append(member)
    if not chain:
        return None
    head = dict(chain[0])
    rest = chain[1:]
    if rest:
        head[FALLBACK_KEY] = rest
    return head


def chain_supports_tools(chain: List[Dict[str, Any]]) -> bool:
    """Whether *every* member advertises native tool calling.

    All, not any: the transport is picked once and the system prompt is
    written to match it. A chain that went native on the primary and fell
    back to a model that cannot read a ``tools`` field would have taught
    the conversation one protocol and then asked it to speak another.
    """
    return bool(chain) and all(
        bool(h.get("supports_tools", False)) for h in chain
    )


def chain_context_length(chain: List[Dict[str, Any]]) -> Optional[int]:
    """The smallest context window in the chain, or ``None`` if unknown.

    The denominator has to hold across a switch: compaction plans cuts
    against it, and a conversation grown to fit the primary must still
    fit whatever catches it. Members that do not know their own window
    are skipped rather than treated as zero -- an unknown is not a
    smaller one (D25).
    """
    known = []
    for handle in chain:
        explicit = handle.get("context_length")
        if explicit:
            known.append(int(explicit))
            continue
        pool = handle.get("pool")
        value = getattr(pool, "context_length", None) if pool is not None else None
        if value:
            known.append(int(value))
    return min(known) if known else None


def chain_can_price(chain: List[Dict[str, Any]]) -> bool:
    """Whether spend stays measurable however far the chain is walked.

    One unpriced member is enough to answer no. A cost cap that binds
    until the fallback takes over is worse than none: it reads as a
    ceiling right up to the point where it stops being one (D88).
    """
    return bool(chain) and all(
        price_from_handle(h) is not None for h in chain
    )


def describe_chain(chain: List[Dict[str, Any]]) -> str:
    """``openai:gpt-x → gguf:qwen`` -- the chain as the canvas reads it."""
    if not chain:
        return "<no model>"
    return " → ".join(model_label(h) for h in chain)
