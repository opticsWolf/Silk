"""ToolSet abstraction for composable, lifecycle-managed tool collections.

Inspired by Pydantic AI's toolset architecture, this module provides:
- Base ToolSet class with lifecycle management (__aenter__/__aexit__)
- Composable variants: CombinedToolSet, FilteredToolSet, PrefixedToolSet
- Each toolset can discover, manage, and describe its own tools
- Tool provenance tracking (which toolset provided each tool)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .run_context import RunContext


if TYPE_CHECKING:
    from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Tool definition metadata
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ToolMeta:
    """Metadata for a single tool, tracking its source toolset."""

    toolset: ToolSet
    """The toolset that provided this tool."""
    definition: dict
    """The tool definition dict (for the LLM API)."""
    args_model: Optional[type["BaseModel"]]
    """Pydantic model for argument validation."""
    executable: Callable[..., Any]
    """The callable that executes the tool."""
    is_async: bool
    """Whether the executable is a coroutine function."""
    procedure: Optional[str]
    """Optional prose appended to the system prompt."""
    source: Optional[str]
    """Provenance tag (e.g. 'mcp', 'core', 'plugin')."""
    timeout: Optional[float]
    """Max execution time in seconds."""
    requires_approval: bool
    """Whether this tool requires human approval before execution."""
    sequential: bool
    """Whether this tool must be executed sequentially with other sequential tools."""


# ---------------------------------------------------------------------------
# Base ToolSet
# ---------------------------------------------------------------------------


class ToolSet(ABC):
    """A composable collection of tools with lifecycle management.

    Each ToolSet is responsible for:
    - Discovering/listing its tools
    - Validating tool arguments (via args_model in ToolMeta)
    - Executing tools (via executable in ToolMeta)
    - Providing instructions for the model
    - Managing its own lifecycle (connections, etc.)

    ToolSets can be combined, filtered, prefixed, and wrapped to compose
    complex tool registries from simple building blocks.
    """

    @property
    @abstractmethod
    def id(self) -> str | None:
        """Unique ID for this toolset (used for error messages and provenance)."""
        ...

    @property
    def label(self) -> str:
        """Human-readable label for this toolset (for error messages)."""
        name = self.__class__.__name__
        if self.id:
            name += f"({self.id!r})"
        return name

    @abstractmethod
    async def get_tools(self) -> dict[str, dict]:
        """Return the tools in this toolset.

        Returns:
            Dict mapping tool names to tool definition dicts.
        """
        ...

    async def get_instructions(self) -> str | None:
        """Return instructions for how to use this toolset's tools.

        Returns:
            Instruction string, or None if no instructions.
        """
        return None

    async def __aenter__(self) -> "ToolSet":
        """Enter the toolset context (e.g., establish connections)."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the toolset context (e.g., close connections)."""
        pass

    def apply(self, visitor: Callable[[ToolSet], None]) -> None:
        """Run a visitor function on this leaf toolset."""
        visitor(self)

    def visit_and_replace(
        self, visitor: Callable[[ToolSet], ToolSet]
    ) -> ToolSet:
        """Run a visitor on this leaf toolset and return the result."""
        return visitor(self)

    # -- Composable combinators (convenience methods) ----------------

    def combined(self, other: ToolSet) -> CombinedToolSet:
        """Combine this toolset with another."""
        return CombinedToolSet([self, other])

    def filtered(
        self,
        filter_fn: Callable[[dict], bool],
    ) -> FilteredToolSet:
        """Filter tools from this toolset."""
        return FilteredToolSet(self, filter_fn)

    def prefixed(self, prefix: str) -> PrefixedToolSet:
        """Prefix all tool names in this toolset."""
        return PrefixedToolSet(self, prefix)

    def with_metadata(self, **metadata: Any) -> "MetadataToolset":
        """Merge custom metadata onto all tools from this toolset."""
        return MetadataToolset(self, metadata)

    def prepared(
        self,
        prepare_fn: Callable[[RunContext, list[dict]], list[dict]],
    ) -> "PreparedToolset":
        """Returns a new toolset that prepares this toolset's tools using a prepare function.

        The prepare function receives the RunContext and the original tool definitions,
        and returns a modified list.

        Args:
            prepare_fn: A callable that receives (ctx, tool_defs) and returns modified tool_defs.

        Returns:
            A PreparedToolset wrapping this toolset.
        """
        return PreparedToolset(self, prepare_fn)

    def renamed(self, name_map: dict[str, str]) -> "RenamedToolset":
        """Returns a new toolset that renames this toolset's tools.

        Args:
            name_map: Dict mapping original names to new names.

        Returns:
            A RenamedToolset wrapping this toolset.
        """
        return RenamedToolset(self, name_map)

    def approval_required(
        self,
        approval_fn: Callable[[RunContext, dict, dict], bool] = lambda ctx, tool_def, tool_args: True,
    ) -> "ApprovalRequiredToolset":
        """Returns a new toolset that requires (some) calls to tools to be approved.

        Args:
            approval_fn: A callable that receives (ctx, tool_def, tool_args) and returns
                True if the tool call requires approval.

        Returns:
            An ApprovalRequiredToolset wrapping this toolset.
        """
        return ApprovalRequiredToolset(self, approval_fn)

    def defer_loading(self, tool_names: Sequence[str] | None = None) -> "DeferredLoadingToolset":
        """Returns a new toolset that marks tools for deferred loading.

        Args:
            tool_names: Optional sequence of tool names to mark for deferred loading.
                If None, all tools are marked for deferred loading.

        Returns:
            A DeferredLoadingToolset wrapping this toolset.
        """
        return DeferredLoadingToolset(self, tool_names)

    def include_return_schemas(self) -> "IncludeReturnSchemasToolset":
        """Returns a new toolset that sets include_return_schema=True on all tools.

        This causes the model to receive return type information for the tools
        in this toolset.

        Returns:
            An IncludeReturnSchemasToolset wrapping this toolset.
        """
        return IncludeReturnSchemasToolset(self)


# ---------------------------------------------------------------------------
# CombinedToolSet
# ---------------------------------------------------------------------------


class CombinedToolSet(ToolSet):
    """Combines multiple toolsets into one.

    Manages the lifecycle of all child toolsets (enters/exits them all).
    Detects name conflicts between child toolsets.
    """

    def __init__(
        self,
        toolsets: Sequence[ToolSet],
    ) -> None:
        self._toolsets = list(toolsets)
        self._exit_stack: AsyncExitStack | None = None

    @property
    def id(self) -> str | None:
        return None

    @property
    def label(self) -> str:
        names = ", ".join(ts.label for ts in self._toolsets)
        return f"CombinedToolSet({names})"

    async def __aenter__(self) -> "CombinedToolSet":
        self._exit_stack = AsyncExitStack()
        for ts in self._toolsets:
            await self._exit_stack.enter_async_context(ts)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None

    async def get_tools(self) -> dict[str, dict]:
        """Merge tools from all child toolsets.

        Raises:
            ValueError: If two child toolsets define the same tool name.
        """
        all_tools: dict[str, dict] = {}

        for child in self._toolsets:
            child_tools = await child.get_tools()
            for name, tool_def in child_tools.items():
                if name in all_tools:
                    existing = all_tools[name].get("_source_toolset_label", "unknown")
                    raise ValueError(
                        f"Tool name conflict: {name!r} is defined by both "
                        f"{existing} and {child.label}. "
                        "Rename one of the tools or wrap in a PrefixedToolset."
                    )
                # Track provenance
                tool_def_copy = dict(tool_def)
                tool_def_copy["_source_toolset_label"] = child.label
                all_tools[name] = tool_def_copy

        return all_tools

    async def get_instructions(self) -> str | None:
        """Concatenate instructions from all child toolsets."""
        instructions = []
        for ts in self._toolsets:
            inst = await ts.get_instructions()
            if inst:
                instructions.append(inst)
        return "\n\n".join(instructions) if instructions else None

    def apply(self, visitor: Callable[[ToolSet], None]) -> None:
        for ts in self._toolsets:
            ts.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[ToolSet], ToolSet]
    ) -> ToolSet:
        new_toolsets = [
            ts.visit_and_replace(visitor) for ts in self._toolsets
        ]
        return CombinedToolSet(new_toolsets)

    def combined(self, other: ToolSet) -> CombinedToolSet:
        """Flatten: combine this combined set with another."""
        return CombinedToolSet(self._toolsets + [other])


# ---------------------------------------------------------------------------
# FilteredToolSet
# ---------------------------------------------------------------------------


class FilteredToolSet(ToolSet):
    """Filters tools from a wrapped toolset using a predicate."""

    def __init__(
        self,
        toolset: ToolSet,
        filter_fn: Callable[[dict], bool],
    ) -> None:
        self._toolset = toolset
        self._filter_fn = filter_fn

    @property
    def id(self) -> str | None:
        return self._toolset.id

    @property
    def label(self) -> str:
        return f"FilteredToolSet({self._toolset.label})"

    async def __aenter__(self) -> "FilteredToolSet":
        await self._toolset.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._toolset.__aexit__(*args)

    async def get_tools(self) -> dict[str, dict]:
        tools = await self._toolset.get_tools()
        return {
            name: tool
            for name, tool in tools.items()
            if self._filter_fn(tool)
        }

    async def get_instructions(self) -> str | None:
        return await self._toolset.get_instructions()

    def apply(self, visitor: Callable[[ToolSet], None]) -> None:
        self._toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[ToolSet], ToolSet]
    ) -> ToolSet:
        return FilteredToolSet(
            self._toolset.visit_and_replace(visitor),
            self._filter_fn,
        )


# ---------------------------------------------------------------------------
# PrefixedToolSet
# ---------------------------------------------------------------------------


class PrefixedToolSet(ToolSet):
    """Prefixes all tool names in a wrapped toolset."""

    def __init__(
        self,
        toolset: ToolSet,
        prefix: str,
    ) -> None:
        self._toolset = toolset
        self._prefix = prefix

    @property
    def id(self) -> str | None:
        return self._toolset.id

    @property
    def label(self) -> str:
        return f"PrefixedToolSet({self._toolset.label}, prefix={self._prefix!r})"

    async def __aenter__(self) -> "PrefixedToolSet":
        await self._toolset.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._toolset.__aexit__(*args)

    async def get_tools(self) -> dict[str, dict]:
        tools = await self._toolset.get_tools()
        return {
            f"{self._prefix}{name}": tool
            for name, tool in tools.items()
        }

    async def get_instructions(self) -> str | None:
        return await self._toolset.get_instructions()

    def apply(self, visitor: Callable[[ToolSet], None]) -> None:
        self._toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[ToolSet], ToolSet]
    ) -> ToolSet:
        return PrefixedToolSet(
            self._toolset.visit_and_replace(visitor),
            self._prefix,
        )


# ---------------------------------------------------------------------------
# MetadataToolset
# ---------------------------------------------------------------------------


class MetadataToolset(ToolSet):
    """Merges custom metadata onto all tools from a wrapped toolset."""

    def __init__(
        self,
        toolset: ToolSet,
        metadata: dict[str, Any],
    ) -> None:
        self._toolset = toolset
        self._metadata = metadata

    @property
    def id(self) -> str | None:
        return self._toolset.id

    @property
    def label(self) -> str:
        return f"MetadataToolset({self._toolset.label})"

    async def __aenter__(self) -> "MetadataToolset":
        await self._toolset.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._toolset.__aexit__(*args)

    async def get_tools(self) -> dict[str, dict]:
        tools = await self._toolset.get_tools()
        result = {}
        for name, tool in tools.items():
            tool_copy = dict(tool)
            tool_copy["_metadata"] = {**self._metadata, **tool_copy.get("_metadata", {})}
            result[name] = tool_copy
        return result

    async def get_instructions(self) -> str | None:
        return await self._toolset.get_instructions()

    def apply(self, visitor: Callable[[ToolSet], None]) -> None:
        self._toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[ToolSet], ToolSet]
    ) -> ToolSet:
        return MetadataToolset(
            self._toolset.visit_and_replace(visitor),
            self._metadata,
        )


# ---------------------------------------------------------------------------
# StaticToolSet â€” for toolsets that don't need connection lifecycle
# ---------------------------------------------------------------------------


@dataclass
class StaticToolSet(ToolSet):
    """A simple toolset backed by a static dict of tool definitions.

    Useful for toolsets that don't need connection lifecycle management
    (e.g., statically registered tools).
    """

    _tools: dict[str, dict] = field(default_factory=dict)
    _instructions: str | None = None
    _id: str | None = None

    @property
    def id(self) -> str | None:
        return self._id

    async def get_tools(self) -> dict[str, dict]:
        return self._tools

    async def get_instructions(self) -> str | None:
        return self._instructions

    async def __aenter__(self) -> "StaticToolSet":
        """StaticToolSet is a no-op context manager."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """StaticToolSet is a no-op context manager."""
        pass

    def add_tool(self, name: str, definition: dict) -> None:
        """Add a tool to this static toolset."""
        self._tools[name] = definition

    def apply(self, visitor: Callable[[ToolSet], None]) -> None:
        visitor(self)

    def visit_and_replace(
        self, visitor: Callable[[ToolSet], ToolSet]
    ) -> ToolSet:
        return visitor(self)


# ---------------------------------------------------------------------------
# PreparedToolset
# ---------------------------------------------------------------------------


class PreparedToolset(ToolSet):
    """Prepares this toolset's tools using a prepare function."""

    def __init__(
        self,
        toolset: ToolSet,
        prepare_fn: Callable[[RunContext, list[dict]], list[dict]],
    ) -> None:
        self._toolset = toolset
        self._prepare_fn = prepare_fn

    @property
    def id(self) -> str | None:
        return self._toolset.id

    @property
    def label(self) -> str:
        return f"PreparedToolset({self._toolset.label})"

    async def __aenter__(self) -> "PreparedToolset":
        await self._toolset.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._toolset.__aexit__(*args)

    async def get_tools(self) -> dict[str, dict]:
        tools = await self._toolset.get_tools()
        # prepare_fn expects a list of tool defs, returns modified list
        tool_list = list(tools.values())
        modified = self._prepare_fn(None, tool_list)  # type: ignore
        return {tool["name"]: tool for tool in modified if "name" in tool}

    async def get_instructions(self) -> str | None:
        return await self._toolset.get_instructions()

    def apply(self, visitor: Callable[[ToolSet], None]) -> None:
        self._toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[ToolSet], ToolSet]
    ) -> ToolSet:
        return PreparedToolset(
            self._toolset.visit_and_replace(visitor),
            self._prepare_fn,
        )


# ---------------------------------------------------------------------------
# RenamedToolset
# ---------------------------------------------------------------------------


class RenamedToolset(ToolSet):
    """Renames tools in a wrapped toolset using a name map."""

    def __init__(
        self,
        toolset: ToolSet,
        name_map: dict[str, str],
    ) -> None:
        self._toolset = toolset
        self._name_map = name_map

    @property
    def id(self) -> str | None:
        return self._toolset.id

    @property
    def label(self) -> str:
        return f"RenamedToolset({self._toolset.label})"

    async def __aenter__(self) -> "RenamedToolset":
        await self._toolset.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._toolset.__aexit__(*args)

    async def get_tools(self) -> dict[str, dict]:
        tools = await self._toolset.get_tools()
        result = {}
        for name, tool in tools.items():
            new_name = self._name_map.get(name, name)
            tool_copy = dict(tool)
            tool_copy["name"] = new_name
            result[new_name] = tool_copy
        return result

    async def get_instructions(self) -> str | None:
        return await self._toolset.get_instructions()

    def apply(self, visitor: Callable[[ToolSet], None]) -> None:
        self._toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[ToolSet], ToolSet]
    ) -> ToolSet:
        return RenamedToolset(
            self._toolset.visit_and_replace(visitor),
            self._name_map,
        )


# ---------------------------------------------------------------------------
# ApprovalRequiredToolset
# ---------------------------------------------------------------------------


class ApprovalRequiredToolset(ToolSet):
    """Requires (some) tool calls to be approved before execution."""

    def __init__(
        self,
        toolset: ToolSet,
        approval_fn: Callable[[RunContext, dict, dict], bool] = lambda ctx, tool_def, tool_args: True,
    ) -> None:
        self._toolset = toolset
        self._approval_fn = approval_fn

    @property
    def id(self) -> str | None:
        return self._toolset.id

    @property
    def label(self) -> str:
        return f"ApprovalRequiredToolset({self._toolset.label})"

    async def __aenter__(self) -> "ApprovalRequiredToolset":
        await self._toolset.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._toolset.__aexit__(*args)

    async def get_tools(self) -> dict[str, dict]:
        tools = await self._toolset.get_tools()
        result = {}
        for name, tool in tools.items():
            tool_copy = dict(tool)
            tool_copy["_requires_approval"] = True
            tool_copy["_approval_fn"] = self._approval_fn
            result[name] = tool_copy
        return result

    async def get_instructions(self) -> str | None:
        return await self._toolset.get_instructions()

    def apply(self, visitor: Callable[[ToolSet], None]) -> None:
        self._toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[ToolSet], ToolSet]
    ) -> ToolSet:
        return ApprovalRequiredToolset(
            self._toolset.visit_and_replace(visitor),
            self._approval_fn,
        )


# ---------------------------------------------------------------------------
# DeferredLoadingToolset
# ---------------------------------------------------------------------------


class DeferredLoadingToolset(ToolSet):
    """Marks tools for deferred loading, hiding them until discovered via tool search."""

    def __init__(
        self,
        toolset: ToolSet,
        tool_names: Sequence[str] | None = None,
    ) -> None:
        self._toolset = toolset
        self._tool_names = frozenset(tool_names) if tool_names is not None else None

    @property
    def id(self) -> str | None:
        return self._toolset.id

    @property
    def label(self) -> str:
        return f"DeferredLoadingToolset({self._toolset.label})"

    async def __aenter__(self) -> "DeferredLoadingToolset":
        await self._toolset.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._toolset.__aexit__(*args)

    async def get_tools(self) -> dict[str, dict]:
        tools = await self._toolset.get_tools()
        result = {}
        for name, tool in tools.items():
            tool_copy = dict(tool)
            if self._tool_names is None or name in self._tool_names:
                tool_copy["_defer_loading"] = True
            result[name] = tool_copy
        return result

    async def get_instructions(self) -> str | None:
        return await self._toolset.get_instructions()

    def apply(self, visitor: Callable[[ToolSet], None]) -> None:
        self._toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[ToolSet], ToolSet]
    ) -> ToolSet:
        return DeferredLoadingToolset(
            self._toolset.visit_and_replace(visitor),
            self._tool_names,
        )


# ---------------------------------------------------------------------------
# IncludeReturnSchemasToolset
# ---------------------------------------------------------------------------


class IncludeReturnSchemasToolset(ToolSet):
    """Sets include_return_schema=True on all tools in a wrapped toolset."""

    def __init__(self, toolset: ToolSet) -> None:
        self._toolset = toolset

    @property
    def id(self) -> str | None:
        return self._toolset.id

    @property
    def label(self) -> str:
        return f"IncludeReturnSchemasToolset({self._toolset.label})"

    async def __aenter__(self) -> "IncludeReturnSchemasToolset":
        await self._toolset.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._toolset.__aexit__(*args)

    async def get_tools(self) -> dict[str, dict]:
        tools = await self._toolset.get_tools()
        result = {}
        for name, tool in tools.items():
            tool_copy = dict(tool)
            tool_copy["include_return_schema"] = True
            result[name] = tool_copy
        return result

    async def get_instructions(self) -> str | None:
        return await self._toolset.get_instructions()

    def apply(self, visitor: Callable[[ToolSet], None]) -> None:
        self._toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[ToolSet], ToolSet]
    ) -> ToolSet:
        return IncludeReturnSchemasToolset(
            self._toolset.visit_and_replace(visitor),
        )
