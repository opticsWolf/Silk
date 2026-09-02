# -*- coding: utf-8 -*-
"""Model-facing tool discovery (spec D4-D6).

The index in `tool_search.py` has always existed and has never been
reachable by a model. The only agent-visible discovery tool was
`load_capability`, which takes an id and no query -- you can only load
what you already know the name of. This module is the missing half: one
core tool, `search_tools(query, category=None, capability=None)`, that
returns *individual* tools by what they do.

Three decisions shape what is here.

**One tool, not two (D5).** There is no companion `load_tool`. A search
result is complete -- it carries the parameter schema -- so the model can
call what it just found, and the dispatcher deals with the loading. The
always-present surface is one tool wide.

**Auto-load at dispatch (D6).** A call to a tool that was discovered but
never loaded is not an error; the dispatcher loads the tool and runs the
call. The role gate runs unchanged before and after, so auto-loading can
never widen what an agent may do -- it only saves a round trip. A load
that fails comes back as a structured error naming the fix.

**Loading does not re-advertise (D6/I11).** A loaded tool becomes
dispatchable, not advertised: it stays out of `get_tool_definitions`, so
the schema block at the head of the prompt does not change mid-run and
the KV prefix survives (D41). The model does not need the advertisement
-- it has the schema from the search result. The spec allows either this
or a deliberate, counted invalidation, and forbids only the third thing:
silently recomposing the prompt. `load_capability` remains the explicit,
model-initiated path that *does* re-advertise.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from weave.logger import get_logger


log = get_logger("SilkDiscovery")

#: The one always-present discovery tool (D5).
SEARCH_TOOL_NAME = "search_tools"

#: How many hits one search returns by default. Small on purpose: the
#: result carries full parameter schemas, and a search that returns twenty
#: of those has spent more context than advertising the tools would have.
DEFAULT_RESULTS = 6
MAX_RESULTS = 20


def _function(tool_def: dict) -> dict:
    """The OpenAI-shaped ``function`` block, however the def is nested."""
    if not isinstance(tool_def, dict):
        return {}
    return tool_def.get("function", tool_def) or {}


def catalog_entry(toolbox: Any, name: str, tool_def: dict) -> dict:
    """One search hit as plain data, schema included.

    The schema is the point: it is what lets D5 hold, because the model
    can call this tool from the search result alone.
    """
    meta = (getattr(toolbox, "tools", None) or {}).get(name) or {}
    function = _function(meta.get("definition", meta) if meta else tool_def)
    if not function.get("name"):
        function = _function(tool_def)
    entry = {
        "name": name,
        "description": function.get("description", ""),
        "parameters": function.get("parameters")
                      or {"type": "object", "properties": {}},
    }
    category = meta.get("category")
    if category:
        entry["category"] = str(category)
    tags = meta.get("tags")
    if tags:
        entry["tags"] = sorted(str(tag) for tag in tags)
    risk = meta.get("risk")
    if risk:
        entry["risk"] = str(risk)
    capability = capability_of(toolbox, name)
    if capability:
        entry["capability"] = capability
    if not meta:
        # Known to the index but not registered: it lives in a capability
        # that has not been loaded. Saying so is honest, and the model
        # needs no action from it -- dispatch will load it (D6).
        entry["loaded"] = False
    return entry


def capability_of(toolbox: Any, name: str) -> Optional[str]:
    """Which capability provides *name*, if one does."""
    sources = getattr(toolbox, "_tool_capabilities", None) or {}
    return sources.get(name)


def discover(
    toolbox: Any,
    query: str,
    *,
    category: Optional[str] = None,
    capability: Optional[str] = None,
    limit: int = DEFAULT_RESULTS,
) -> list[dict]:
    """Ranked, role-filtered hits for *query* (D4).

    The role gate is applied inside :class:`ToolSearch`, not here, so that
    every path into the index obeys it (I8) -- including the ones this
    module does not own.
    """
    search = getattr(toolbox, "tool_search", None)
    if search is None:
        return []

    limit = max(1, min(int(limit or DEFAULT_RESULTS), MAX_RESULTS))
    previous = search.max_results
    # Filters cut into the ranked list, so ask for enough to still fill a
    # page after they have run.
    search.max_results = limit if not (category or capability) else MAX_RESULTS
    try:
        hits = search.search(query or "")
    finally:
        search.max_results = previous

    entries: list[dict] = []
    seen: set[str] = set()
    for tool_def in hits:
        name = _function(tool_def).get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        entry = catalog_entry(toolbox, name, tool_def)
        if category and entry.get("category") != category:
            continue
        if capability and entry.get("capability") != capability:
            continue
        entries.append(entry)
        if len(entries) >= limit:
            break
    return entries


def attach_search_tool(toolbox: Any) -> None:
    """Register ``search_tools`` on *toolbox* (D4, D5).

    Written straight into the flat tool dict rather than through
    ``register``, for the same reason ``load_capability`` is: this is
    infrastructure, not a plugin tool, and it must survive every derived
    ToolSet (see ``INFRASTRUCTURE_TOOLS``).
    """
    definition = {
        "type": "function",
        "function": {
            "name": SEARCH_TOOL_NAME,
            "description": (
                "Find tools by what they do. Returns matching tools with "
                "their full parameter schemas, ranked by relevance -- call "
                "any of them directly from the result; nothing else has to "
                "be loaded first. Use this before concluding that a "
                "capability is missing.\n"
                "Optional filters: 'category' narrows to one family of "
                "tools, 'capability' to one capability's tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you need to do, in words.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter.",
                    },
                    "capability": {
                        "type": "string",
                        "description": "Optional capability-id filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"Max hits (default {DEFAULT_RESULTS}).",
                    },
                },
                "required": ["query"],
            },
        },
    }

    def _search(**kwargs: Any) -> str:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return json.dumps({
                "error": "search_tools needs a query.",
                "suggestion": "Describe the operation you want to perform, "
                              "for example 'read a file' or 'run the tests'.",
            }, ensure_ascii=False)
        entries = discover(
            toolbox, query,
            category=kwargs.get("category") or None,
            capability=kwargs.get("capability") or None,
            limit=int(kwargs.get("limit") or DEFAULT_RESULTS),
        )
        if not entries:
            return json.dumps({
                "query": query,
                "results": [],
                "message": "No tool matches that. Either the capability does "
                           "not exist here, or the active role forbids it -- "
                           "try other words before assuming it is missing.",
            }, ensure_ascii=False)
        return json.dumps({"query": query, "results": entries},
                          ensure_ascii=False)

    toolbox.tools[SEARCH_TOOL_NAME] = {
        "definition": definition,
        "args_model": None,
        "executable": _search,
        "is_async": False,
        "procedure": None,
        "source": None,
        "timeout": None,
        "requires_approval": False,
        "sequential": False,
        "tags": frozenset({"discovery"}),
        "category": "infrastructure",
        "risk": "low",
    }
    toolbox.tool_search.register_tool(SEARCH_TOOL_NAME, definition)


def autoload(toolbox: Any, name: str) -> Optional[dict]:
    """Make *name* dispatchable if discovery has promised it (D6).

    Returns ``None`` when the tool is ready to run, or an error payload in
    the "errors carry the fix" style when it cannot be. Nothing here
    consults the role gate: dispatch has already run it and will run it
    again on the loaded tool, so this can only ever affect *availability*,
    never permission.
    """
    if name in (getattr(toolbox, "tools", None) or {}):
        return None

    capability_id = capability_of(toolbox, name)
    if capability_id is None:
        return None       # Not ours; the caller reports the unknown tool.

    result = toolbox.load_capability(capability_id)
    if not result.get("success") and name not in toolbox.tools:
        return {
            "error": f"Tool '{name}' comes from capability "
                     f"'{capability_id}', which failed to load: "
                     f"{result.get('error', 'unknown error')}",
            "error_type": "autoload_failed",
            "suggestion": "Use search_tools to find another tool for this, "
                          "or proceed without it.",
        }

    if name not in toolbox.tools:
        return {
            "error": f"Capability '{capability_id}' loaded but does not "
                     f"provide '{name}'.",
            "error_type": "autoload_failed",
            "suggestion": "Call search_tools again -- the catalog has "
                          "changed since that name was found.",
        }

    # Loaded, and deliberately left unadvertised: the model already has the
    # schema from its search, and re-advertising would rewrite the head of
    # the prompt mid-run and cost a full prefill (D41, I11).
    toolbox.defer_tools(
        tool.get("function", tool).get("name")
        for tool in (toolbox.capability(capability_id).get_tools() or [])
        if (tool.get("function", tool) or {}).get("name")
    )
    log.info(f"Auto-loaded '{name}' from capability '{capability_id}' "
             f"(not re-advertised; prompt prefix preserved)")
    return None
