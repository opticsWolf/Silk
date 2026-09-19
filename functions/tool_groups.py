# -*- coding: utf-8 -*-
"""What Silk *can* attach, known before anything is attached.

The ToolBox node's tree is the only tool selector, so it has to be
populated before there is a ToolBox to read it off — a tree that fills in
only once a sandbox root is wired is a tree with nothing to tick, and the
root is the thing the user is usually wiring *because* they want tools.

The catalog is therefore built once, from a throwaway sandbox rooted at a
temporary directory, by running every group's attacher against a scratch
:class:`ToolBox` and recording what each one registered. Nothing is
executed: attaching a tool registers a name, a description and a
parameter schema, and that is all this reads. The scratch box is
discarded.

Two things are deliberately *not* here. Toolchain packs depend on which
toolchains are wired, and MCP tools on which servers are connected — both
arrive at build time and the node adopts them then. This is the fixed
half: the groups that are always available and differ only in whether the
user wants them.
"""
from __future__ import annotations

import tempfile
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional

from weave.logger import get_logger
from weave.node.port_registry import coerce_path_list

from .tool_box import ToolBox
from .tools.file_sandbox import FileToolSandbox
from .tools.file_manipulate import attach_file_manipulate_tools
from .tools.file_read import attach_file_read_tools
from .tools.file_write import attach_file_write_tools
from .tools.graph_authoring import attach_graph_tools
from .tools.recall_tool import attach_recall_tool
from .tools.ripgrep_tool import attach_ripgrep_tools
from .tools.suite_tools import attach_suite_tools
from .tools.task_tracker import attach_task_tools

log = get_logger("SilkToolGroups")

#: Group id → the attacher that mounts it. Order is the order tools are
#: registered in, which is the order they reach the model's tool list:
#: read before write, plan before memory.
GROUP_ATTACHERS: Dict[str, Callable[..., None]] = {
    "file_read": attach_file_read_tools,
    "file_write": attach_file_write_tools,
    "file_manipulate": attach_file_manipulate_tools,
    "ripgrep": attach_ripgrep_tools,
    "task_tracker": attach_task_tools,
    "graph_authoring": attach_graph_tools,
    "suite_tools": attach_suite_tools,
    "recall": attach_recall_tool,
}

#: Groups whose tools modify the filesystem. The sandbox's write flag is
#: derived from the *selection*, not from a separate switch: ticking
#: ``write_file`` is the request for write access, and nothing else is.
WRITING_GROUPS: frozenset[str] = frozenset({"file_write", "file_manipulate"})

#: What a fresh node starts with ticked. Reading and searching a folder
#: you just wired is the request; writing to it is a different one, and
#: graph authoring and plugin authoring are different ones again.
DEFAULT_GROUPS: frozenset[str] = frozenset({"file_read", "ripgrep"})

#: Groups that carry configuration of their own, and the label the tree
#: shows on the row that opens it. Graph authoring's whitelist is not a
#: preference — it *is* the grant (D71), so it belongs beside the tools
#: it governs rather than in a field elsewhere on the node.
GROUP_CONFIG_LABELS: Dict[str, str] = {
    "graph_authoring": "Allowed node classes…",
}


def _probe_kwargs(group: str) -> Dict[str, Any]:
    """Arguments that let an attacher run with nothing wired.

    Each is the emptiest legal value, chosen so the probe registers the
    same tool *names* the real build would: an empty whitelist still
    mounts all eight graph tools (the grant is enforced per call, not by
    leaving tools out), and recall without an embedder is the keyword
    half it has always been.
    """
    if group == "graph_authoring":
        return {"whitelist": ()}
    return {}


@lru_cache(maxsize=1)
def _probe() -> Dict[str, List[Dict[str, Any]]]:
    """``{group: [catalog entry, …]}`` for every static group."""
    from .toolset_build import tool_catalog

    catalog: Dict[str, List[Dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="silk-probe-") as scratch:
        sandbox = FileToolSandbox(root_dir=scratch, write_enabled=True)
        for group, attacher in GROUP_ATTACHERS.items():
            box = ToolBox()
            try:
                attacher(box, sandbox, **_probe_kwargs(group))
            except Exception as exc:  # noqa: BLE001 — a group is optional
                # A group that cannot even be described is a group the
                # user cannot be offered. Logged, not raised: one missing
                # optional dependency must not cost the whole tree.
                log.warning(f"tool group {group!r} could not be probed: {exc}")
                catalog[group] = []
                continue
            catalog[group] = tool_catalog(box)
    return catalog


def group_catalog() -> Dict[str, List[Dict[str, Any]]]:
    """``{group: [catalog entry, …]}`` — every tool Silk can attach."""
    return {group: list(entries) for group, entries in _probe().items()}


def static_catalog() -> List[Dict[str, Any]]:
    """Every static tool as one flat catalog, in group order."""
    return [entry for entries in _probe().values() for entry in entries]


def tools_of(group: str) -> frozenset[str]:
    """The tool names *group* registers."""
    return frozenset(e["name"] for e in _probe().get(group, ()))


def group_of(tool: str) -> Optional[str]:
    """Which group registers *tool*, or None for a dynamic one."""
    for group, entries in _probe().items():
        if any(e["name"] == tool for e in entries):
            return group
    return None


def groups_for(selection: Iterable[str]) -> List[str]:
    """The groups that must be attached to satisfy *selection*.

    A group earns its place by having at least one tool ticked. That is
    the whole rule, and it is why the group checkboxes are gone: a group
    switch and a tool tick answered the same question twice, and could
    disagree.
    """
    wanted = set(selection)
    return [g for g in GROUP_ATTACHERS if wanted & tools_of(g)]


def default_selection() -> List[str]:
    """The tool names a fresh ToolBox node starts with ticked."""
    names: set[str] = set()
    for group in DEFAULT_GROUPS:
        names |= tools_of(group)
    return sorted(names)


def categories_of(group: str) -> frozenset[str]:
    """The catalog categories *group*'s tools fall under.

    The tree groups by category, not by group, so this is how a
    category row finds the configuration that belongs to it.
    """
    return frozenset(e.get("category", "uncategorized")
                     for e in _probe().get(group, ()))


def configurable_categories() -> Dict[str, str]:
    """``{category: label}`` for the rows that open a config dialog."""
    out: Dict[str, str] = {}
    for group, label in GROUP_CONFIG_LABELS.items():
        for category in categories_of(group):
            out[category] = label
    return out


def coerce_paths(value: Any) -> List[str]:
    """Whatever arrived on the ``sandbox_roots`` port, as a list of strings.

    Thin alias for Weave's :func:`coerce_path_list`: a Folder node may
    connect to this list-typed port thanks to the ``dirpath`` ->
    ``dirpath_list`` cast, but the cast is checked when the wire is drawn
    and never applied to the value, so a bare ``Path`` is what actually
    turns up. Named locally because "sandbox roots, whatever shape they
    came in" is the thing this module is about.
    """
    return coerce_path_list(value)
