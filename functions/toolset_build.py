# -*- coding: utf-8 -*-
"""Graph-level ToolSet construction — Qt-free.

A **ToolSet** in the silk node graph is a real :class:`ToolBox` rebuilt
from the source toolbox's *recipe* (the attach functions it was built
with), restricted to an explicit selection of tool names and optionally
re-rooted onto its own :class:`FileToolSandbox`.

Rebuilding (instead of view-wrapping) keeps every guarantee of the
ToolBox intact for downstream consumers: hard role enforcement at
dispatch, hooks, schema generation — and each agent gets an independent
instance, so concurrent agents never fight over one RoleBinding.

The recipe protocol: whoever builds the source ToolBox stamps it with
``build_recipe`` (tuple of ``(source_name, attacher)`` pairs, each
attacher callable as ``attacher(toolbox, sandbox)``) and
``base_sandbox`` (the sandbox those attachers were bound to).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from .file_grants import MODE_ORDER, FileGrants
from .hooks import is_middleware_event
from .tool_box import ToolBox
from .tools.file_sandbox import FileToolSandbox

#: Tools that are ToolBox infrastructure, never part of a user selection.
INFRASTRUCTURE_TOOLS = frozenset({"load_capability", "search_tools"})

#: Permission modes in ascending order of access. Defined once, in
#: ``file_grants``; re-exported here because this is where callers of the
#: sandbox builder already look.
PERMISSION_ORDER: dict[str, int] = MODE_ORDER


def tool_catalog(toolbox: Any) -> list[dict[str, Any]]:
    """Flatten a ToolBox's registry into plain-data catalog entries.

    Each entry: ``{name, description, parameters, category, tags, risk}``.
    Infrastructure tools (``load_capability``) are excluded. The result is
    plain dicts/lists only, safe to hand across threads to UI code.
    """
    catalog: list[dict[str, Any]] = []
    for name, meta in sorted(toolbox.tools.items()):
        if name in INFRASTRUCTURE_TOOLS:
            continue
        definition = meta.get("definition") or meta
        function = definition.get("function", {})
        catalog.append({
            "name": name,
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
            "category": meta.get("category") or "uncategorized",
            "tags": sorted(meta.get("tags") or ()),
            "risk": meta.get("risk", "low"),
        })
    return catalog


def catalog_categories(catalog: Iterable[dict[str, Any]]) -> list[str]:
    """Sorted unique category names of a catalog."""
    return sorted({entry["category"] for entry in catalog})


def split_by_ceiling(
    permissions: Any,
    base: Optional[FileToolSandbox],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition permission entries into (inside, outside) the ceiling.

    The *base* sandbox's allowed paths (the ToolBox's sandbox roots) are
    the **hard ceiling**: entries outside every allowed root are never
    granted, no matter what an upstream permission structure claims.
    A missing or disabled base sandbox imposes no ceiling.

    This is the one narrowing step that is *not* grant-against-grant: it is
    checked against a live sandbox, because only the sandbox knows how a
    path resolves on this machine (spec D16).

    Accepts a :class:`FileGrants` or the equivalent dict; returns plain
    dicts, which is what the callers and the status readouts already use.
    """
    grants = FileGrants.coerce(permissions)
    entries = [] if grants is None else [
        {"path": e.path, "mode": e.mode} for e in grants.entries
    ]
    if base is None or not base.enabled:
        return entries, []
    inside = [e for e in entries if base.is_allowed(Path(e["path"]))]
    outside = [e for e in entries if not base.is_allowed(Path(e["path"]))]
    return inside, outside


def sandbox_from_permissions(
    permissions: Any,
    base: Optional[FileToolSandbox] = None,
) -> FileToolSandbox:
    """Build a sandbox from a :class:`FileGrants` (or the equivalent dict).

    The *base* sandbox (from the ToolBox) is the hard ceiling: entries
    outside its allowed roots are dropped here — a toolset can only
    narrow, never escape. Size limits are inherited from *base*.
    """
    grants = FileGrants.coerce(permissions) or FileGrants()
    permissions = grants.to_dict()
    entries, _outside = split_by_ceiling(grants, base)
    # Hierarchical mode map: a directory rule covers its subtree; a per-path
    # entry (including an explicit "blocked" override) beats an ancestor. Files
    # created after selection inherit the nearest granted directory, so the
    # agent's own writes and files a user drops in are covered automatically.
    path_modes = {
        e["path"]: e["mode"]
        for e in entries
        if e.get("mode") in ("read", "read_write", "blocked")
    }
    grants = any(m in ("read", "read_write") for m in path_modes.values())

    # Root: prefer a declared root that survives the ceiling, else the
    # base root — never a root outside the ceiling.
    roots = [str(r) for r in (permissions.get("roots") or ()) if r]
    if not roots and permissions.get("root"):
        roots = [str(permissions["root"])]
    root = next(
        (r for r in roots
         if base is None or not base.enabled or base.is_allowed(Path(r))),
        str(base.root_dir) if base is not None else ".",
    )

    kwargs: dict[str, Any] = {}
    if base is not None:
        kwargs["max_read_bytes"] = base.max_read_bytes
        kwargs["max_write_bytes"] = base.max_write_bytes

    if not grants:
        # No readable grant = nothing visible.
        return FileToolSandbox(
            root_dir=root, read_enabled=False, write_enabled=False, **kwargs
        )

    return FileToolSandbox(
        root_dir=root,
        path_modes=path_modes,
        write_enabled=any(m == "read_write" for m in path_modes.values()),
        **kwargs,
    )


def carry_essential_hooks(source: Any, derived: Any) -> int:
    """Copy *source*'s essential hooks onto *derived*; returns how many (D14).

    Replaying the recipe already reinstalls anything the recipe attached,
    which is most of the infrastructure tier -- so the usual answer is
    zero, and that is fine. What this catches is the hook registered
    *outside* the recipe: the approval gate the Agent node installs on a
    live toolbox, say. Invariant I7 says such a hook survives derivation,
    and without this it would not, because nothing in the recipe knows it
    exists.

    Skips anything the replay already produced -- same event, same callable
    -- so a recipe hook is not installed twice.
    """
    registry = getattr(source, "hooks", None)
    target = getattr(derived, "hooks", None)
    if registry is None or target is None:
        return 0

    carried = 0
    for event, entry in registry.essential_entries():
        existing = (
            target.middleware_entries(event) if is_middleware_event(event)
            else target.entries(event)
        )
        if any(e.callback is entry.callback for e in existing):
            continue
        if is_middleware_event(event):
            target.register_middleware(event, entry)
        else:
            target.register(event, entry)
        carried += 1
    return carried


def build_toolset(
    source: Any,
    selected_names: Iterable[str],
    permissions: Any = None,
) -> ToolBox:
    """Rebuild *source* as an independent ToolBox restricted to a selection.

    Re-runs the source's recorded ``build_recipe`` against either the
    source's own sandbox or one derived from *permissions*, then removes
    every tool not in *selected_names* (infrastructure tools stay).

    Raises ``ValueError`` if the source carries no recipe — a toolset can
    only be derived from a toolbox that recorded how it was built.
    """
    recipe = getattr(source, "build_recipe", None)
    if not recipe:
        raise ValueError(
            "Source toolbox has no build recipe; a ToolSet can only be "
            "derived from a ToolBox node output."
        )
    base_sandbox = getattr(source, "base_sandbox", None)

    sandbox = (
        sandbox_from_permissions(permissions, base_sandbox)
        if permissions else base_sandbox
    )

    toolset = ToolBox(db_pool=source.db_pool, user_session=source.user_session)
    for source_name, attacher in recipe:
        with toolset._attributing_to(source_name):
            attacher(toolset, sandbox)

    keep = set(selected_names) | INFRASTRUCTURE_TOOLS
    for name in list(toolset.tools):
        if name not in keep:
            toolset.unregister(name)

    carry_essential_hooks(source, toolset)

    # Propagate the recipe so a toolset remains introspectable downstream.
    toolset.build_recipe = recipe  # type: ignore[attr-defined]
    toolset.base_sandbox = sandbox  # type: ignore[attr-defined]
    return toolset
