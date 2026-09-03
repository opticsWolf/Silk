from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Optional, Type

from pydantic import BaseModel, ValidationError

# Command-advice engine. BashHint/REDIRECT are re-exported so tools can write
# `from .tool_box import BashHint` and the advice surface lives on the ToolBox.
from .tools.command_advice import (
    BashHint,
    BashHintIndex,
    REDIRECT,
    hint_for_command,
)

# Phase 2: Lifecycle hooks
from .hooks import (
    HookRegistry,
    HOOK_BEFORE_TOOL_EXECUTE,
    HOOK_AFTER_TOOL_EXECUTE,
    HOOK_ON_TOOL_EXECUTE_ERROR,
    HOOK_ON_TOOL_VALIDATE_ERROR,
    HOOK_WRAP_TOOL_VALIDATE,
    HOOK_TOOL_DENIED,
    HOOK_WRAP_TOOL_EXECUTE,
    register_hook_map,
)
from .capabilities import BaseCapability, DeferredCapability
from .tool_discovery import (
    SEARCH_TOOL_NAME, attach_search_tool, autoload,
)
from .tool_search import ToolSearch
from .toolset import (
    CombinedToolSet,
    FilteredToolSet,
    PrefixedToolSet,
    StaticToolSet,
    ToolSet,
)

try:  # MCP support is optional — the rest of the toolbox works without it.
    from .mcp_toolset import MCPToolset
except ImportError:  # pragma: no cover - optional dependency
    MCPToolset = None  # type: ignore[assignment,misc]

__all__ = ["ToolBox", "BashHint", "REDIRECT", "ToolSet", "CombinedToolSet", "FilteredToolSet", "PrefixedToolSet", "StaticToolSet", "MCPToolset"]

# Signature of the role predicate installed via ToolBox.set_role_filter():
# (tool_name, tool_meta_dict_or_None) -> bool. Returning False hard-denies
# the call at dispatch time and hides the tool from get_tool_schemas().
RoleFilter = Callable[[str, Optional[dict]], bool]


class ToolBox:
    """
    Tool registry with per-session dependency injection, Pydantic-based
    schema generation + validation, and parallel async execution.

    A fresh instance is created per web-request / user-session; `db_pool`
    and `user_session` are injected into every tool call.

    Tools also contribute *bash-equivalent* advice as part of the registration
    protocol (``register(..., replaces=[BashHint(...)])``). The ToolBox folds
    those into an index and answers :meth:`bash_equivalent_hint`, so a shell
    tool can nudge the model toward the structured tool without importing any
    static command table.

    ToolSets are used internally as composable building blocks. Each registered
    tool (via ``register()``) is stored in a ``StaticToolSet`` which is then
    combined into a ``CombinedToolSet``. MCP servers, filtered sets, and
    prefixed sets can also be added directly via ``add_toolset()``.
    """

    def __init__(self, db_pool: Any = None, user_session: Optional[dict] = None):
        self.db_pool = db_pool
        self.user_session = user_session

        # Active role enforcement predicate (None = allow everything).
        # Installed/removed by RoleBinding.activate()/deactivate(); shared by
        # get_tool_schemas() (prompt side) and execute_tool_calls_async()
        # (dispatch side) so advertisement and enforcement cannot drift.
        self._role_filter: Optional[RoleFilter] = None
        # Flat tools dict (legacy compatibility, kept for direct access)
        self.tools: dict[str, dict[str, Any]] = {}

        # Bash â†’ native-tool advice, populated from each tool's `replaces`.
        self._bash_index = BashHintIndex()

        # Phase 2: Lifecycle hooks
        self.hooks = HookRegistry()
        # A category-bound hook (D13) needs to know a tool's category, and
        # the registry has no tool index of its own. This is that index.
        self.hooks.bind_categories(self.tool_category)

        # Phase 4: Capabilities
        self._capabilities: dict[str, BaseCapability] = {}
        self._loaded_capability_ids: set[str] = set()
        # Hooks each capability registered, keyed by capability id, so
        # deactivation (RoleBinding) can remove them precisely.
        self._capability_hooks: dict[str, list[tuple[str, Callable]]] = {}
        # Discovery reads the same role gate dispatch enforces, so what
        # search offers and what dispatch accepts cannot drift (I8).
        self.tool_search = ToolSearch(permits=self.role_permits)
        # Tools that are registered and dispatchable but deliberately not
        # advertised (spec D6): per-tool deferral. Auto-loading one mid-run
        # must not rewrite the schema block at the head of the prompt, so a
        # loaded tool joins this set rather than the advertisement.
        self.deferred_tools: set[str] = set()
        # Which capability provides which tool name -- populated for
        # *deferred* capabilities too, so a tool can be found by search
        # before anything has loaded it, and loaded by the dispatcher when
        # it is called (D4, D6).
        self._tool_capabilities: dict[str, str] = {}

        # ToolSet architecture
        # Each statically registered tool lives in its own StaticToolSet,
        # and all are combined into one CombinedToolSet.
        self._static_toolsets: list[StaticToolSet] = []
        self._external_toolsets: list[ToolSet] = []  # MCP, filtered, prefixed, etc.
        self._combined: CombinedToolSet | None = None
        self._combined_context: Any = None

        # Provenance: which plugin source registered which tool names. Lets a
        # reload prune a module's tools (and its hints) before re-attaching.
        self._sources: dict[str, set[str]] = {}
        self._current_source: Optional[str] = None

        # Phase 6: Register load_capability tool for deferred tool loading
        self.register_load_capability_tool()
        # Spec D4/D5: the one always-present discovery tool.
        attach_search_tool(self)

    # -- ToolSet management -----------------------------------------------

    async def _rebuild_combined(self) -> CombinedToolSet:
        """Rebuild the CombinedToolSet from all static and external toolsets."""
        all_toolsets: list[ToolSet] = list(self._static_toolsets) + list(self._external_toolsets)
        if not all_toolsets:
            return CombinedToolSet([])
        if len(all_toolsets) == 1:
            return CombinedToolSet(all_toolsets)
        return CombinedToolSet(all_toolsets)

    async def _enter_combined(self) -> None:
        """Enter the CombinedToolSet context (connects all external toolsets)."""
        if self._combined is None:
            self._combined = await self._rebuild_combined()
        if self._combined_context is None:
            self._combined_context = await self._combined.__aenter__()

    async def _exit_combined(self) -> None:
        """Exit the CombinedToolSet context (disconnects all external toolsets)."""
        if self._combined_context is not None:
            await self._combined_context.__aexit__(None, None, None)
            self._combined_context = None
        self._combined = None

    async def add_toolset(self, toolset: ToolSet) -> None:
        """Add an external toolset (MCP, filtered, prefixed, etc.).

        Args:
            toolset: The toolset to add.
        """
        self._external_toolsets.append(toolset)
        self._combined = None  # Invalidate cache
        self._combined_context = None  # Will be re-entered on next use

    def remove_toolset(self, toolset: ToolSet) -> None:
        """Remove an external toolset."""
        if toolset in self._external_toolsets:
            self._external_toolsets.remove(toolset)
        self._combined = None
        self._combined_context = None

    # -- capability management (Phase 4) --------------------------------

    def register_capability(self, capability: BaseCapability) -> None:
        """Register a capability with the toolbox.

        Args:
            capability: The capability to register.
        """
        self._capabilities[capability.id] = capability
        self.tool_search.register_capability(capability)

        # Index the capability's tools by name even when it is deferred.
        # Without this, a deferred capability is discoverable only by id --
        # which means only by an agent that already knows the id, which is
        # the gap D4 exists to close.
        for tool_def in (capability.get_tools() or []):
            tool_name = (tool_def.get("function", tool_def) or {}).get("name")
            if not tool_name:
                continue
            self._tool_capabilities[tool_name] = capability.id
            if capability.defer_loading and tool_name not in self.tools:
                self.tool_search.register_tool(tool_name, tool_def)

        # If the capability is not deferred, load it immediately
        if not capability.defer_loading:
            self._load_capability(capability)

        # Refresh the load_capability tool description to include new deferred caps
        self.register_load_capability_tool()

    def _load_capability(self, capability: BaseCapability) -> str:
        """Load a capability's tools and hooks.

        Args:
            capability: The capability to load.

        Returns:
            The capability's instructions, or empty string if none.
        """
        if capability.id in self._loaded_capability_ids:
            return ""  # Already loaded

        # Mark DeferredCapability as loaded FIRST so get_tools() works
        if isinstance(capability, DeferredCapability) and not capability.is_loaded():
            capability.load()

        # Now load tools from the capability
        for tool_def in capability.get_tools():
            tool_name = tool_def.get("function", {}).get("name")
            if tool_name and tool_name not in self.tools:
                self.tools[tool_name] = tool_def
                self.tool_search.register_tool(tool_name, tool_def)

        # Register the capability's hooks (wrap_* events route to the
        # middleware layer), tracked per capability id so a RoleBinding
        # deactivation can remove exactly these again.
        registered = register_hook_map(self.hooks, capability.get_hooks())
        if registered:
            self._capability_hooks[capability.id] = registered

        self._loaded_capability_ids.add(capability.id)

        # Return instructions for the loaded capability
        return capability.get_instructions()

    def load_capability(self, capability_id: str) -> dict:
        """Load a capability by ID.

        Args:
            capability_id: The ID of the capability to load.

        Returns:
            Dict with 'success', 'capability_id', 'loaded_tools', 'instructions',
            and optionally 'error' if not found or already loaded.
        """
        capability = self._capabilities.get(capability_id)
        if capability is None:
            return {
                "success": False,
                "capability_id": capability_id,
                "error": f"Capability '{capability_id}' not found.",
            }

        # Check if already loaded
        if capability_id in self._loaded_capability_ids:
            return {
                "success": False,
                "capability_id": capability_id,
                "error": f"Capability '{capability_id}' is already loaded. Use its existing tools and instructions.",
            }

        instructions = self._load_capability(capability)
        loaded_tools = capability.get_tools()
        tool_names = [t.get("function", {}).get("name", "") for t in loaded_tools]

        return {
            "success": True,
            "capability_id": capability_id,
            "loaded_tools": tool_names,
            "instructions": instructions,
            "message": f"Capability '{capability_id}' loaded. Available tools: {', '.join(tool_names)}",
        }

    # -- deferred tool loading (Phase 6) ----------------------------------

    def register_load_capability_tool(self) -> None:
        """Register or refresh the load_capability tool so the model can call it.

        This enables deferred tool loading: the model can discover and load
        capabilities on-demand via a tool call, reducing prompt size by only
        sending tools that are actually needed.

        Called automatically in __init__ and after each capability registration.
        """
        # Build the list of available deferred capabilities for the tool description
        deferred_caps = [
            f"{cap.id}: {cap.description}"
            for cap in self._capabilities.values()
            if getattr(cap, "defer_loading", False)
        ]
        available = "\n".join(deferred_caps) if deferred_caps else "None"

        definition = {
            "type": "function",
            "function": {
                "name": "load_capability",
                "description": (
                    "Load a deferred capability to unlock access to its tools. "
                    "Use this when the user's request requires tools from a capability "
                    "that hasn't been loaded yet.\n\n"
                    "Available deferred capabilities:\n" + available
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "capability_id": {
                            "type": "string",
                            "description": "The ID of the capability to load.",
                        },
                    },
                    "required": ["capability_id"],
                },
            },
        }

        async def _load_capability_exec(**kwargs):
            capability_id = kwargs.get("capability_id", "")
            result = self.load_capability(capability_id)

            if result["success"]:
                return json.dumps({
                    "success": True,
                    "capability_id": result["capability_id"],
                    "loaded_tools": result["loaded_tools"],
                    "instructions": result["instructions"],
                    "message": result["message"],
                })
            else:
                return json.dumps({
                    "success": False,
                    "capability_id": result["capability_id"],
                    "error": result["error"],
                })

        self.tools["load_capability"] = {
            "definition": definition,
            "args_model": None,
            "executable": _load_capability_exec,
            "is_async": True,
            "procedure": None,
            "source": "core",
        }
        self.tool_search.register_tool("load_capability", definition)

    def capability(self, capability_id: str) -> Optional[BaseCapability]:
        """The registered capability with this id, if any."""
        return self._capabilities.get(capability_id)

    # -- per-tool deferral (spec D6) --------------------------------------

    def defer_tools(self, names: Iterable[str]) -> None:
        """Keep these tools dispatchable but out of the advertisement.

        Deferral is about the *prompt*, not about permission: a deferred
        tool is refused or allowed by exactly the same role gate as any
        other. What it does not do is occupy schema space at the head of
        every request -- the model finds it with ``search_tools`` and calls
        it from the schema in that result (D4-D6).
        """
        self.deferred_tools.update(str(name) for name in names if name)

    def undefer_tools(self, names: Iterable[str]) -> None:
        """Advertise these tools again.

        Deliberate, and never done mid-run by the dispatcher: adding a
        schema to the head of the prompt invalidates the KV prefix for
        every remaining round (D41, I11).
        """
        for name in names:
            self.deferred_tools.discard(str(name))

    def is_deferred(self, name: str) -> bool:
        """Whether *name* is registered but not advertised."""
        return name in self.deferred_tools

    def get_deferred_capability_names(self) -> list[str]:
        """Get names of deferred (not yet loaded) capabilities.

        Returns:
            List of capability IDs that are deferred.
        """
        return [
            cap.id
            for cap in self._capabilities.values()
            if getattr(cap, "defer_loading", False) and cap.id not in self._loaded_capability_ids
        ]

    def get_all_capability_info(self) -> list[dict]:
        """Get info about all registered capabilities (loaded and deferred).

        Returns:
            List of dicts with capability info.
        """
        return [
            {
                "id": cap.id,
                "description": cap.description,
                "defer_loading": getattr(cap, "defer_loading", False),
                "loaded": cap.id in self._loaded_capability_ids,
            }
            for cap in self._capabilities.values()
        ]

    def get_loaded_capability_ids(self) -> set[str]:
        """Get the IDs of loaded capabilities.

        Returns:
            A set of capability IDs.
        """
        return self._loaded_capability_ids.copy()

    def get_available_tool_names(self) -> set[str]:
        """Get the names of available tools.

        Returns:
            A set of tool names.
        """
        return set(self.tools.keys())

    # -- registration ---------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        args_model: Optional[Type[BaseModel]] = None,
        procedure: Optional[str] = None,
        replaces: Optional[Iterable[BashHint]] = None,
        timeout: Optional[float] = None,
        requires_approval: bool = False,
        sequential: bool = False,
        tags: Optional[Iterable[str]] = None,
        category: Optional[str] = None,
        risk: str = "low",
    ) -> Callable:
        """
        Decorator to register a tool.

        `args_model` drives both the JSON schema handed to the LLM and the
        runtime validation of the model's arguments. `procedure` is optional
        prose appended to the system prompt for this specific tool. `replaces`
        is an optional list of :class:`BashHint`\\ s declaring the shell
        commands this tool stands in for â€” the registration-time half of the
        command-advice protocol.

        The wrapped function is always called as:
            func(db_pool, user_session, **validated_args)
        """
        def decorator(func: Callable) -> Callable:
            parameters = (
                args_model.model_json_schema()
                if args_model
                else {"type": "object", "properties": {}}
            )
            definition = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }

            # Detect async/sync once, at registration, instead of re-sniffing
            # the wrapper later.
            is_async = asyncio.iscoroutinefunction(func)
            executable: Callable[..., Any]
            if is_async:
                async def _async_executable(**kwargs):
                    return await func(self.db_pool, self.user_session, **kwargs)
                executable = _async_executable
            else:
                def _sync_executable(**kwargs):
                    return func(self.db_pool, self.user_session, **kwargs)
                executable = _sync_executable

            # Create a StaticToolSet for this tool
            toolset = StaticToolSet(
                _id=f"tool:{name}",
                _tools={name: definition},
                _instructions=None,
            )
            self._static_toolsets.append(toolset)
            self._combined = None  # Invalidate cache

            # Also add to flat dict for legacy compatibility
            self.tools[name] = {
                "definition": definition,
                "args_model": args_model,
                "executable": executable,
                "is_async": is_async,
                "procedure": procedure,
                "source": self._current_source,
                "timeout": timeout,
                "requires_approval": requires_approval,
                "sequential": sequential,
                # Role-selector metadata (see role.ToolSelector.permits).
                "tags": frozenset(tags or ()),
                "category": category,
                "risk": risk,
            }

            if requires_approval:
                # The flag arrives with its gate (D81). Declaring it is
                # the whole of what a tool author has to do; a host that
                # wants grants or a fixed seam attaches the floor itself
                # first, and this then finds it already there.
                from .approval_floor import ensure_approval_floor
                ensure_approval_floor(self)

            # Index it for discovery. Without this a registered tool is
            # findable only if it is advertised, which is exactly backwards:
            # search exists to reach the tools the prompt does not carry
            # (D4).
            self.tool_search.register_tool(name, definition)

            # Record provenance so a plugin reload can prune precisely.
            if self._current_source is not None:
                self._sources.setdefault(self._current_source, set()).add(name)

            # Fold this tool's shell-command advice into the index, stamping each
            # hint with the owning tool name.
            for hint in (replaces or ()):
                self._bash_index.add(hint.with_tool(name))

            return func

        return decorator

    def unregister(self, name: str) -> None:
        """Remove a single tool and any bash hints it contributed."""
        self.tools.pop(name, None)
        self.tool_search.tools.pop(name, None)
        self.deferred_tools.discard(name)
        self._bash_index.remove_tool(name)
        for names in self._sources.values():
            names.discard(name)

    # -- plugin provenance / discovery ----------------------------------

    @contextmanager
    def _attributing_to(self, source: Optional[str]):
        """Tag every tool registered inside this block with *source*."""
        previous = self._current_source
        self._current_source = source
        try:
            yield
        finally:
            self._current_source = previous

    def prune_source(self, source: str) -> None:
        """Drop every tool (and its hints) registered by *source*."""
        for name in self._sources.pop(source, set()):
            self.tools.pop(name, None)
            self._bash_index.remove_tool(name)

    def attach_plugin(self, source: str, attacher: Callable, /, **context: Any) -> None:
        """
        Run a single ``attach_*`` callable, tagging whatever it registers with
        *source*. Extra dependencies are injected by parameter name from
        *context* (e.g. ``sandbox=...``) â€” no positional-count guessing.
        """
        from .tools.tool_loader import inject  # local import avoids a cycle
        with self._attributing_to(source):
            inject(attacher, self, context)

    def load_directory(self, directory, /, **context: Any) -> "LoadReport":
        """
        Discover and attach every tool plugin in *directory*, remembering the
        directory + context so :meth:`refresh` can re-scan later. Safe to call
        repeatedly; an already-loaded module is re-attached only if its file
        changed. Returns a :class:`LoadReport`.
        """
        from .tools.tool_loader import ToolLoader
        loader = getattr(self, "_loader", None)
        if loader is None or loader.directory != _as_path(directory):
            loader = ToolLoader(directory)
            self._loader = loader
        if context:
            self._loader_context = dict(context)
        return loader.sync(self, getattr(self, "_loader_context", {}))

    def refresh(self) -> "LoadReport":
        """Re-scan the directory passed to :meth:`load_directory`."""
        loader = getattr(self, "_loader", None)
        if loader is None:
            from .tools.tool_loader import LoadReport
            return LoadReport()
        return loader.sync(self, getattr(self, "_loader_context", {}))

    # -- command advice -------------------------------------------------

    def bash_equivalent_hint(self, command: str) -> Optional[str]:
        """
        If *command* looks like a shell invocation with a native-tool
        equivalent among the currently registered tools, return a one-line hint;
        otherwise ``None``. This is the advice surface a shell tool calls (via
        closure capture of the toolbox) before/after running a raw command.
        """
        return hint_for_command(command, self._bash_index)

    # -- role enforcement -------------------------------------------------

    def set_role_filter(self, predicate: Optional[RoleFilter]) -> None:
        """Install (or clear, with ``None``) the active role's tool predicate.

        The predicate is consulted live at every schema build and every
        dispatch, so tools mounted after activation are covered too.
        """
        self._role_filter = predicate

    def role_permits(self, name: str) -> bool:
        """Whether the active role (if any) permits tool *name*."""
        if self._role_filter is None:
            return True
        return self._role_filter(name, self.tools.get(name))

    def tool_category(self, name: str) -> str | None:
        """The registered category of tool *name*, if it has one.

        Lent to the hook registry so a hook can bind to a category rather
        than to a list of tool names (D13) -- the binding a capability
        wants, since a capability adds tools over time and a name list goes
        stale the moment it does.
        """
        meta = self.tools.get(name) or {}
        category = meta.get("category")
        return str(category) if category else None

    # -- prompt / schema ------------------------------------------------

    def build_system_prompt(self, base_prompt: str) -> str:
        """Append each registered tool's `procedure` block and capability
        instructions to the base prompt.

        Also includes information about available deferred capabilities so the
        model knows what it can load on-demand (Phase 6).

        Args:
            base_prompt: The base system prompt.

        Returns:
            The complete system prompt.
        """
        sections = [base_prompt]

        # Add tool procedures
        for name, meta in self.tools.items():
            procedure = meta.get("procedure")
            if procedure:
                sections.append(f"[PROCEDURE: {name}]\n{procedure.strip()}")

        # Add capability instructions for loaded capabilities
        for capability in self._capabilities.values():
            if capability.id in self._loaded_capability_ids:
                instructions = capability.get_instructions()
                if instructions:
                    sections.append(f"[CAPABILITY: {capability.id}]\n{instructions}")

        # Say that discovery exists (D4/D5). One line, unconditional: the
        # tools an agent was not given are exactly the ones it cannot see,
        # so nothing else in the prompt can imply that they might be there.
        if SEARCH_TOOL_NAME in self.tools:
            sections.append(
                "[TOOL DISCOVERY]\n"
                "Not every available tool is listed above. Call "
                "'search_tools' with a description of what you need before "
                "concluding that something cannot be done here; the result "
                "carries each tool's full schema, and you can call what you "
                "find straight away."
            )

        # Add deferred capability info (Phase 6)
        deferred = self.get_deferred_capability_names()
        if deferred:
            sections.append(
                f"[DEFERRED CAPABILITIES]\n"
                f"The following capabilities are available but not yet loaded. "
                f"Use the 'load_capability' tool to load them when needed:\n"
                f"{', '.join(deferred)}"
            )

        return "\n\n".join(sections)

    def get_tool_schemas(self) -> list[dict]:
        """Tool schemas for the LLM API call.

        Returns:
            A list of tool schemas from all loaded tools.
        """
        schemas = []
        for name, meta in self.tools.items():
            if not self.role_permits(name):
                continue
            # Deferred: registered, dispatchable, deliberately unadvertised
            # (D6). `search_tools` is how the model reaches these.
            if name in self.deferred_tools:
                continue
            # Tools registered via @register have a "definition" key
            # Tools loaded from Capability are stored directly
            if "definition" in meta:
                schemas.append(meta["definition"])
            else:
                schemas.append(meta)
        return schemas

    # -- execution ------------------------------------------------------

    async def execute_tool_calls_async(self, tool_calls: list[Any]) -> list[dict]:
        """
        Validate and run a batch of tool calls in parallel. Failures are
        returned as tool results (never raised) so the LLM can self-correct.

        Sequential tools are executed one at a time to avoid conflicts.
        """
        # Ensure external toolsets (MCP, etc.) are connected
        await self._enter_combined()

        results: list[dict] = []
        sequential_tasks: list[tuple[str, str, dict, dict]] = []
        parallel_tasks: list[tuple[str, str, dict, dict]] = []

        for tc in tool_calls:
            name = tc.function.name
            meta = self.tools.get(name)

            if meta is None:
                # Discovered but never loaded: load it and run the call
                # rather than spending a round trip telling the model to
                # load it itself (D6). The role gate below is unchanged, so
                # this can only ever save time, never widen permission.
                failure = autoload(self, name)
                if failure is not None:
                    results.append(self._error(
                        tc.id, name, failure["error"],
                        error_type=failure.get("error_type"),
                        suggestion=failure.get("suggestion"),
                    ))
                    continue
                meta = self.tools.get(name)

            if meta is None:
                results.append(
                    self._error(
                        tc.id, name, f"Tool '{name}' is not registered.",
                        suggestion="Call search_tools to find what is "
                                   "available for this.",
                    )
                )
                continue

            # Hard role boundary: a denied call never reaches the executable,
            # even if the model hallucinated a fence for an unadvertised tool.
            if not self.role_permits(name):
                self.hooks.emit(HOOK_TOOL_DENIED, tool_name=name)
                results.append(
                    self._error(
                        tc.id,
                        name,
                        f"Tool '{name}' is not available to the active role.",
                        error_type="role_denied",
                        suggestion="Use one of the tools listed in your instructions.",
                    )
                )
                continue

            try:
                args = await self._validate_args(name, meta, tc.function.arguments)
            except ValidationError as e:
                details = [
                    f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                    for err in e.errors()
                ]
                self.hooks.emit(
                    HOOK_ON_TOOL_VALIDATE_ERROR,
                    tool_name=name, error=str(e), details=details,
                )
                results.append(
                    self._error(
                        tc.id,
                        name,
                        "Validation error: arguments do not match the required schema.",
                        details=details,
                        args_model=meta["args_model"],  # include correct schema for self-correction
                    )
                )
                continue
            except json.JSONDecodeError as e:
                self.hooks.emit(
                    HOOK_ON_TOOL_VALIDATE_ERROR,
                    tool_name=name, error=str(e), details=[],
                )
                results.append(
                    self._error(
                        tc.id,
                        name,
                        "Failed to parse tool arguments. Ensure they are valid JSON.",
                    )
                )
                continue
            except Exception as e:      # noqa: BLE001
                # A wrap_tool_validate middleware refused (or broke). It is
                # a refusal of one call, not a failure of the run: the
                # model gets a result it can read, the same as every other
                # validation outcome.
                self.hooks.emit(
                    HOOK_ON_TOOL_VALIDATE_ERROR,
                    tool_name=name, error=str(e), details=[],
                )
                results.append(
                    self._error(
                        tc.id,
                        name,
                        f"Arguments for '{name}' were refused before the "
                        f"tool ran: {e}",
                    )
                )
                continue

            # Separate sequential and parallel tasks
            if meta.get("sequential"):
                sequential_tasks.append((tc.id, name, meta, args))
            else:
                parallel_tasks.append((tc.id, name, meta, args))

        # Execute parallel tasks concurrently
        if parallel_tasks:
            parallel_results = await asyncio.gather(*[
                self._safe_execute(id, name, meta, args)
                for id, name, meta, args in parallel_tasks
            ])
            results.extend(parallel_results)

        # Execute sequential tasks one at a time
        if sequential_tasks:
            for id, name, meta, args in sequential_tasks:
                result = await self._safe_execute(id, name, meta, args)
                results.append(result)

        return results

    async def _validate_args(self, name: str, meta: dict,
                             raw: Optional[str]) -> dict:
        """Parse and validate one call's arguments, wrappable (§22 q2).

        `wrap_tool_validate` is the one of the five open ``WRAP_*`` events
        with a shape middleware can actually express: it is async, it is
        per-call, it knows the tool name (so `tools=` / `categories=`
        filtering works), and the thing it wraps returns a value. A
        middleware here can repair arguments a model got slightly wrong,
        or refuse a call before the executable is reached -- and raising
        `ValidationError` lands in the same self-correction path the
        model already knows how to read.
        """
        def _parse(**kw: Any) -> dict:
            return self._parse_args(meta["args_model"],
                                    kw.get("raw_args", raw))

        if not self.hooks.has_middleware(HOOK_WRAP_TOOL_VALIDATE, name):
            return _parse()

        async def _innermost(**kw: Any) -> dict:
            return _parse(**kw)

        return await self.hooks.emit_middleware(
            HOOK_WRAP_TOOL_VALIDATE,
            innermost=_innermost,
            tool_name=name,
            raw_args=raw,
        )

    @staticmethod
    def _parse_args(args_model: Optional[Type[BaseModel]], raw: Optional[str]) -> dict:
        if args_model is not None:
            # Pydantic handles JSON parsing + strict validation in one step.
            return args_model.model_validate_json(raw or "{}").model_dump()
        return json.loads(raw) if raw else {}

    async def _safe_execute(
        self, call_id: str, name: str, meta: dict, args: dict
    ) -> dict:
        # The flag is enforced by the floor in `approval_floor.py` (D81),
        # which is middleware and has already run by the time execution
        # reaches here. What is left at this site is the fail-closed half:
        # a tool that declares it needs a human, in a toolbox where
        # nothing can ask one, does not run. This is where G1's TODO sat.
        if meta.get("requires_approval"):
            from .approval_floor import APPROVAL_FLOOR_ATTR, NO_FLOOR_TEXT
            if getattr(self, APPROVAL_FLOOR_ATTR, None) is None:
                return self._error(
                    call_id, name, NO_FLOOR_TEXT,
                    error_type="approval_required",
                    suggestion="Ask the user to run this step themselves.",
                )

        # Emit before_tool_execute hook (Phase 2)
        self.hooks.emit(
            HOOK_BEFORE_TOOL_EXECUTE,
            tool_name=name,
            tool_args=args,
        )

        try:
            # The actual execution, as the innermost layer of the
            # wrap_tool_execute middleware chain: middleware can
            # short-circuit it (deny) or post-process its result.
            async def _run_tool(**_kw: Any) -> Any:
                executable = meta["executable"]
                if meta.get("timeout") and meta["is_async"]:
                    return await asyncio.wait_for(
                        executable(**args), timeout=meta["timeout"]
                    )
                if meta.get("timeout") and not meta["is_async"]:
                    # For sync tools, use asyncio.to_thread with timeout
                    return await asyncio.wait_for(
                        asyncio.to_thread(executable, **args),
                        timeout=meta["timeout"],
                    )
                if meta["is_async"]:
                    return await executable(**args)
                # Keep sync tools off the event loop.
                return await asyncio.to_thread(executable, **args)

            if self.hooks.has_middleware(HOOK_WRAP_TOOL_EXECUTE, name):
                output = await self.hooks.emit_middleware(
                    HOOK_WRAP_TOOL_EXECUTE,
                    innermost=_run_tool,
                    tool_name=name,
                    tool_args=args,
                )
            else:
                output = await _run_tool()

            # Emit after_tool_execute hook (Phase 2)
            self.hooks.emit(
                HOOK_AFTER_TOOL_EXECUTE,
                tool_name=name,
                tool_result=_to_content(output),
            )

            return {
                "tool_call_id": call_id,
                "name": name,
                "content": _to_content(output),
            }
        except asyncio.TimeoutError as e:
            self.hooks.emit(
                HOOK_ON_TOOL_EXECUTE_ERROR,
                tool_name=name, tool_args=args, error=f"timeout: {e}",
            )
            return self._error(
                call_id,
                name,
                f"Tool '{name}' timed out after {meta.get('timeout', 'unknown')}s",
                suggestion="Try again with simpler arguments or fewer operations.",
            )
        except Exception as e:
            self.hooks.emit(
                HOOK_ON_TOOL_EXECUTE_ERROR,
                tool_name=name, tool_args=args, error=str(e),
            )
            # Emit after_tool_execute hook with error (Phase 2)
            self.hooks.emit(
                HOOK_AFTER_TOOL_EXECUTE,
                tool_name=name,
                tool_result=f"Error: {e}",
            )
            return self._error(
                call_id,
                name,
                f"Execution failed: {e}",
                suggestion="Check your arguments and try again.",
            )

    # â”€â”€ Structured Error Response (Phase 1: Reflection) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _error(
        call_id: str,
        name: str,
        error: str,
        *,
        details: Optional[list] = None,
        suggestion: Optional[str] = None,
        error_type: Optional[str] = None,
        args_model: Optional[Type[BaseModel]] = None,
    ) -> dict:
        """Build a structured error response for the model.

        The error is returned as JSON so the model can parse it and retry.
        ``error_type`` tags machine-readable categories (e.g. ``role_denied``)
        that reflection treats as non-retryable.

        When ``args_model`` is provided (validation errors), the correct
        JSON schema is included in the error payload so the model can
        self-correct on retry — it sees the exact field names and types
        it got wrong.
        """
        payload: dict[str, Any] = {"error": error}
        if error_type is not None:
            payload["error_type"] = error_type
        if details is not None:
            payload["details"] = details
        if suggestion is not None:
            payload["suggestion"] = suggestion
        if args_model is not None:
            # Resurface the correct call pattern as JSON schema
            payload["correct_schema"] = args_model.model_json_schema()
        return {
            "tool_call_id": call_id,
            "name": name,
            "content": json.dumps(payload, ensure_ascii=False),
        }


def _as_path(directory):
    from pathlib import Path
    return Path(directory).resolve()


def _to_content(output: Any) -> str:
    """
    Normalise a tool's return value into the string `content` field.

    Plain strings pass through unchanged (so human-readable tool output isn't
    double-encoded into an escaped JSON string), Pydantic models are dumped,
    and everything else is JSON-serialised with a str fallback so a stray
    non-serialisable object can't crash the loop.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    return json.dumps(output, default=str)


# Re-exported for type hints on load_directory/refresh without importing at top
# (avoids a circular import with tool_loader, which imports ToolBox for typing).
from .tools.tool_loader import LoadReport  # noqa: E402
