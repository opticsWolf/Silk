# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The knowledge graph as agent tools: OKFgraph's search, read, traverse and
ingest over a served graph (``functions/okf_graph.py``).

``recall`` is the agent's memory of what was *said*; this is what was
*learned* -- a curated graph of documents, findings and decisions that the
user also reads, edits and imports into with the ``okf`` CLI. The tools
close over the :class:`~..okf_graph.OkfGraph` handle the Knowledge Graph
node owns, so a ToolSet replaying the recipe attaches to the same served
graph rather than starting a second owner for the file.

**Before compaction, the agent writes down what it learned.** A compaction
summary is a lossy squeeze written for the next round, not knowledge, so it
is not what goes into the graph. Instead this pack marks ``okf_ingest`` as
the box's pre-compaction tool: when the context is about to be compacted,
the loop gives the agent one round to ingest its findings and reasoning
first (``compaction.pre_compaction_prompt``). Whether that round happens is
decided when it would happen, by the same two checks a call would pass --
the tool is in this run's box (a ToolSet can leave it out) and the active
role permits it -- so a worker that may not write to the graph is never
asked to.

**Access levels.** The graph is shared with the user and every other run,
so what an agent may do to it is a grant, set where the sandbox ceiling is
set: ``read`` (search, read, traverse), ``write`` (also ingest -- adding,
never changing), ``curate`` (also update, supersede and delete; deletes
are soft and recoverable for a day). The ToolBox's level is the ceiling
and decides which tools are attached at all; a ToolSet may lower it,
never raise it (:func:`narrow_okf_level`); ticks and roles narrow further.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..compaction import set_pre_compaction
from ..okf_graph import OkfError

if TYPE_CHECKING:
    from ..okf_graph import OkfGraph
    from ..tool_box import ToolBox
    from .file_sandbox import FileToolSandbox

#: The tool the pre-compaction round asks the agent to use.
OKF_INGEST_TOOL = "okf_ingest"

#: Access levels, lowest first.
OKF_LEVELS = ("read", "write", "curate")

#: What a fresh ToolBox grants: what the pack always offered.
DEFAULT_OKF_LEVEL = "write"

#: Each tool, in the order an agent reads them, and the level it needs.
OKF_TOOL_LEVELS: Dict[str, str] = {
    "okf_search": "read",
    "okf_read": "read",
    "okf_traverse": "read",
    OKF_INGEST_TOOL: "write",
    "okf_update": "curate",
    "okf_supersede": "curate",
    "okf_delete": "curate",
}

#: The tools this pack registers, in the order an agent reads them.
OKF_TOOLS = tuple(OKF_TOOL_LEVELS)

#: Attribute on a ToolBox naming the level its okf tools were granted.
OKF_LEVEL_ATTR = "okf_level"


def coerce_level(value: Any, default: str = DEFAULT_OKF_LEVEL) -> str:
    text = str(value or "").strip().lower()
    return text if text in OKF_LEVELS else default


def lower_level(a: str, b: str) -> str:
    """The lower of two levels."""
    return min(coerce_level(a), coerce_level(b), key=OKF_LEVELS.index)


def tools_up_to(level: str) -> tuple[str, ...]:
    """The okf tools *level* grants."""
    rank = OKF_LEVELS.index(coerce_level(level))
    return tuple(n for n, need in OKF_TOOL_LEVELS.items()
                 if OKF_LEVELS.index(need) <= rank)


def narrow_okf_level(toolbox: Any, level: str) -> str:
    """Lower *toolbox*'s okf grant to *level* (never raise it); the level
    it ends up with. Tools above it are unregistered, so neither the model
    nor a later ToolSet sees them."""
    current = getattr(toolbox, OKF_LEVEL_ATTR, None)
    if current is None:
        return ""    # no graph attached: nothing to narrow
    effective = lower_level(current, level)
    keep = set(tools_up_to(effective))
    for name in OKF_TOOLS:
        if name not in keep and name in toolbox.tools:
            toolbox.unregister(name)
    setattr(toolbox, OKF_LEVEL_ATTR, effective)
    return effective

#: Characters of a result handed back to the model.
OUTPUT_LIMIT = 20_000

#: What the agent is told when its context is about to be compacted.
PRE_COMPACTION_PROMPT = (
    "[Silk] Your context is nearly full, and older turns are about to be "
    "summarized and dropped. Before that happens, save what you have "
    "learned so far into the knowledge graph with okf_ingest: one call per "
    "finding, decision or line of reasoning worth keeping, with a "
    "consistent topic for this project. Use type 'finding' for facts you "
    "established and 'thought' for reasoning; list the concepts it is about "
    "in 'related'. Skip anything already ingested or not worth finding "
    "again. Then continue with the task."
)


class OkfSearchArgs(BaseModel):
    query: str = Field(..., min_length=1, description="What to look for, in plain words.")
    limit: int = Field(8, ge=1, le=50, description="How many hits to return.")
    target: Literal["concepts", "chunks"] = Field(
        "concepts", description="'concepts' ranks whole documents; 'chunks' ranks passages.",
    )
    rank: Literal["none", "hub", "ppr"] = Field(
        "none",
        description=(
            "'none': semantic + keyword fusion. 'hub': favour well-linked "
            "concepts. 'ppr': graph walk from keyword seeds (no model needed)."
        ),
    )
    tags: Optional[List[str]] = Field(None, description="Only concepts carrying all these tags.")
    concept_type: Optional[str] = Field(
        None, description="Only this concept type, e.g. 'finding' or 'thought'.",
    )


class OkfReadArgs(BaseModel):
    concept_id: str = Field(..., min_length=1, description="The concept to read (an id from search).")
    include: Literal["body", "chunks", "document", "context"] = Field(
        "body",
        description=(
            "'body': the concept with its text. 'context': its links, parents "
            "and siblings. 'chunks' / 'document': the stored passages."
        ),
    )
    max_tokens: Optional[int] = Field(
        None, ge=50, le=20_000,
        description="Token budget: the concept first, then its best neighbours.",
    )


class OkfTraverseArgs(BaseModel):
    start_id: str = Field("", description="Concept or directory id; empty lists the root.")
    relationship: Literal["CONTAINS", "LINKS_TO", "PART_OF"] = Field(
        "LINKS_TO", description="Which edges to follow.",
    )
    direction: Literal["OUTGOING", "INCOMING", "BOTH"] = Field(
        "BOTH", description="Which way along them.",
    )
    depth: int = Field(1, ge=1, le=5, description="How many steps.")
    target: Optional[str] = Field(
        None, description="Instead of walking: the shortest path from start_id to this id.",
    )


class OkfIngestArgs(BaseModel):
    thoughts: str = Field(
        ..., min_length=1,
        description="The finding or reasoning, in full sentences that stand on their own.",
    )
    topic: str = Field(
        ..., min_length=1,
        description="A stable topic for this line of work, e.g. 'lexer-rewrite'.",
    )
    title: Optional[str] = Field(
        None, description="Short title, unique within the topic. Default: the first sentence.",
    )
    type: Optional[str] = Field(
        None, description="'finding', 'decision' or 'thought' (the default).",
    )
    tags: Optional[List[str]] = Field(None, description="Extra tags.")
    related: Optional[List[str]] = Field(
        None, description="Ids or titles of the concepts this is about; linked in the graph.",
    )


class OkfUpdateArgs(BaseModel):
    concept_id: str = Field(..., min_length=1, description="The concept to edit.")
    title: Optional[str] = Field(None, description="New title.")
    description: Optional[str] = Field(None, description="New one-line description.")
    type: Optional[str] = Field(None, description="New concept type.")
    add_tags: Optional[List[str]] = Field(None, description="Tags to add.")
    remove_tags: Optional[List[str]] = Field(None, description="Tags to remove.")


class OkfSupersedeArgs(BaseModel):
    old_id: str = Field(..., min_length=1, description="The concept that is out of date.")
    new_id: str = Field(..., min_length=1, description="The concept that replaces it.")


class OkfDeleteArgs(BaseModel):
    concept_id: str = Field(..., min_length=1, description="The concept to delete.")
    remove_file: bool = Field(
        False,
        description=(
            "Also move its bundle file to the bundle's trash, so a later "
            "import does not bring it back. Without it, a concept backed by "
            "a file is refused."
        ),
    )


def _format(data: Any) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str, indent=1)
    if len(text) > OUTPUT_LIMIT:
        text = text[:OUTPUT_LIMIT] + f"\n… (truncated at {OUTPUT_LIMIT} chars)"
    return text


def _failure(exc: OkfError) -> str:
    """An op error as the model should read it: what, and what to do."""
    line = f"error: {exc.code}: {exc.message}"
    return f"{line}\nremedy: {exc.remedy}" if exc.remedy else line


def attach_okf_tools(toolbox: "ToolBox", sandbox: "FileToolSandbox",
                     graph: Optional["OkfGraph"] = None, *,
                     level: str = DEFAULT_OKF_LEVEL,
                     catalog_only: bool = False) -> None:
    """Register the knowledge-graph tools *level* grants, against *graph*.

    Recipe-shaped (``attacher(toolbox, sandbox)`` once bound); *sandbox* is
    unused -- the graph is the node's, not a path in the root. With no
    graph wired nothing is registered: a tool that can only answer "no
    graph" is noise in the model's list. *catalog_only* registers every
    tool regardless, for the ToolBox tree's probe, which never calls them.
    """
    del sandbox
    if graph is None and not catalog_only:
        return
    granted = set(OKF_TOOLS if catalog_only else tools_up_to(level))

    def run(op: str, **kwargs: Any) -> str:
        if graph is None:
            return "error: NO_GRAPH: no knowledge graph is wired"
        try:
            return _format(graph.call(op, **kwargs))
        except OkfError as exc:
            return _failure(exc)

    def register(name: str, **meta: Any):
        """``toolbox.register`` for a granted tool; a no-op otherwise."""
        if name not in granted:
            return lambda fn: fn
        return toolbox.register(name=name, **meta)

    @register(
        "okf_search",
        description=(
            "Search the knowledge graph: documents, findings and decisions "
            "from earlier work, yours and the user's. Search here before "
            "re-deriving something or asking the user."
        ),
        args_model=OkfSearchArgs,
        tags=("knowledge", "search", "okfgraph"), category="knowledge", risk="low",
    )
    def okf_search(_db, _session, **args: Any) -> str:
        return run("search", **args)

    @register(
        "okf_read",
        description="Read one knowledge-graph concept by id: its text, or its links.",
        args_model=OkfReadArgs,
        tags=("knowledge", "read", "okfgraph"), category="knowledge", risk="low",
    )
    def okf_read(_db, _session, **args: Any) -> str:
        return run("read", **args)

    @register(
        "okf_traverse",
        description=(
            "Walk the knowledge graph from a concept: what it links to, what "
            "links to it, or the path between two concepts."
        ),
        args_model=OkfTraverseArgs,
        tags=("knowledge", "graph", "okfgraph"), category="knowledge", risk="low",
    )
    def okf_traverse(_db, _session, **args: Any) -> str:
        return run("traverse", **args)

    @register(
        OKF_INGEST_TOOL,
        description=(
            "Save a finding, decision or piece of reasoning into the "
            "knowledge graph, where later runs and the user can find it."
        ),
        args_model=OkfIngestArgs,
        tags=("knowledge", "write", "okfgraph"), category="knowledge", risk="medium",
    )
    def okf_ingest(_db, _session, **args: Any) -> str:
        if graph is None:
            return run("ingest")
        try:
            made = graph.call("ingest", kind="thoughts", **args)
        except OkfError as exc:
            return _failure(exc)
        # Not the rendered Markdown: the agent just wrote the text, and
        # echoing it back costs the context this round is trying to save.
        if isinstance(made, dict):
            made = {k: v for k, v in made.items() if k != "markdown"}
        return _format(made)

    @register(
        "okf_update",
        description=(
            "Edit a knowledge-graph concept's title, description, type or "
            "tags (never its body). For a changed belief, ingest the new "
            "one and okf_supersede the old instead."
        ),
        args_model=OkfUpdateArgs,
        tags=("knowledge", "write", "okfgraph"), category="knowledge", risk="medium",
    )
    def okf_update(_db, _session, **args: Any) -> str:
        return run("update", **args)

    @register(
        "okf_supersede",
        description=(
            "Mark a concept as replaced by a newer one: the old stays "
            "readable, tagged superseded, and links to its replacement."
        ),
        args_model=OkfSupersedeArgs,
        tags=("knowledge", "write", "okfgraph"), category="knowledge", risk="medium",
    )
    def okf_supersede(_db, _session, **args: Any) -> str:
        return run("supersede", **args)

    @register(
        "okf_delete",
        description=(
            "Delete a knowledge-graph concept. Soft: the user can recover "
            "it for 24 hours. Prefer okf_supersede for something that was "
            "merely wrong or out of date."
        ),
        args_model=OkfDeleteArgs,
        tags=("knowledge", "delete", "okfgraph"), category="knowledge", risk="high",
    )
    def okf_delete(_db, _session, **args: Any) -> str:
        return run("delete", **args)

    if graph is None:
        return    # the catalog probe: no grant to record, no round to ask
    setattr(toolbox, OKF_LEVEL_ATTR, coerce_level(level))
    if OKF_INGEST_TOOL in granted:
        set_pre_compaction(toolbox, OKF_INGEST_TOOL, PRE_COMPACTION_PROMPT)
