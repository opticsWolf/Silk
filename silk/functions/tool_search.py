"""Tool search for deferred tool discovery.

Enables searching for tools by name or description, useful for
on-demand capability loading where only relevant tools are sent to
the model.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .capabilities import BaseCapability


# Type alias for custom search functions.
# Takes a list of search queries and a list of tool definitions,
# returns a list of matching tool names ordered by relevance.
# Both sync and async implementations are supported.
ToolSearchFunc = Callable[
    [list[str], list[dict]],
    list[str] | Awaitable[list[str]],
]


@dataclass
class ToolSearch:
    """Tool search for deferred tool discovery.

    Supports multiple search strategies:
    - 'keywords': Local keyword-overlap algorithm (default)
    - 'bm25': Placeholder for BM25-based search
    - 'regex': Regex-based search
    - Callable: Custom search function `(queries, tools) -> names`

    Attributes:
        strategy: The search strategy to use.
        max_results: Maximum number of results to return.
        tools: Dict mapping tool names to tool definitions.
        capabilities: Dict mapping capability IDs to capabilities.
        search_fn: Optional custom search function.
    """

    strategy: str = "keywords"
    """The search strategy to use.

    Can be a string ('keywords', 'bm25', 'regex') or 'custom' when
    a custom search function is provided.
    """

    max_results: int = 10
    """Maximum number of results to return."""

    tools: dict[str, dict] = None  # type: ignore[assignment]
    """Dict mapping tool names to tool definitions."""

    capabilities: dict[str, BaseCapability] = None  # type: ignore[assignment]
    """Dict mapping capability IDs to capabilities."""

    search_fn: ToolSearchFunc | None = None
    """Optional custom search function."""

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = {}
        if self.capabilities is None:
            self.capabilities = {}

    @classmethod
    def create(
        cls,
        strategy: str | ToolSearchFunc = "keywords",
        max_results: int = 10,
        tools: dict[str, dict] | None = None,
        capabilities: dict[str, BaseCapability] | None = None,
    ) -> "ToolSearch":
        """Create a ToolSearch with a custom search function.

        Args:
            strategy: The search strategy. Can be a string or a callable.
            max_results: Maximum number of results to return.
            tools: Dict mapping tool names to tool definitions.
            capabilities: Dict mapping capability IDs to capabilities.

        Returns:
            A ToolSearch instance with the specified strategy.
        """
        # If strategy is a callable, store it as search_fn
        if callable(strategy):
            search_fn = strategy
            strategy = "custom"
        else:
            search_fn = None

        return cls(
            strategy=strategy,
            max_results=max_results,
            tools=tools or {},
            capabilities=capabilities or {},
            search_fn=search_fn,
        )

    def register_tool(self, tool_name: str, tool_def: dict) -> None:
        """Register a tool for search.

        Args:
            tool_name: The name of the tool.
            tool_def: The tool definition (as a dict).
        """
        self.tools[tool_name] = tool_def

    def register_capability(self, capability: BaseCapability) -> None:
        """Register a capability for search.

        Args:
            capability: The capability to register.
        """
        self.capabilities[capability.id] = capability

    def search(self, query: str) -> list[dict]:
        """Search for tools matching the query.

        Args:
            query: The search query (single string, will be split into words).

        Returns:
            A list of matching tool definitions, sorted by relevance.
        """
        queries = [query]
        tool_defs = list(self.tools.values())

        # Use custom search function if provided
        if self.search_fn is not None:
            return self._run_custom_search(queries, tool_defs)

        if self.strategy == "keywords":
            return self._keyword_search(query)
        elif self.strategy == "bm25":
            return self._bm25_search(query)
        elif self.strategy == "regex":
            return self._regex_search(query)
        else:
            return self._keyword_search(query)

    def _keyword_search(self, query: str) -> list[dict]:
        """Keyword-based search.

        Args:
            query: The search query.

        Returns:
            A list of matching tool definitions, sorted by relevance.
        """
        query_words = set(query.lower().split())
        scores: dict[str, int] = {}

        for tool_name, tool_def in self.tools.items():
            score = 0
            tool_desc = tool_def.get("function", {}).get("description", "").lower()
            tool_name_lower = tool_name.lower()

            for word in query_words:
                if word in tool_desc:
                    score += 2
                if word in tool_name_lower:
                    score += 3

            if score > 0:
                scores[tool_name] = score

        results = [self.tools[name] for name in sorted(scores.keys(), key=lambda n: scores[n], reverse=True)]
        return results[:self.max_results]

    def _bm25_search(self, query: str) -> list[dict]:
        """BM25-based search (placeholder for now).

        Args:
            query: The search query.

        Returns:
            A list of matching tool definitions, sorted by relevance.
        """
        # TODO: Implement BM25 search
        return self._keyword_search(query)

    def _regex_search(self, query: str) -> list[dict]:
        """Regex-based search.

        Args:
            query: The search query (regex pattern).

        Returns:
            A list of matching tool definitions.
        """
        try:
            pattern = re.compile(query, re.IGNORECASE)
            results = [
                tool_def
                for tool_name, tool_def in self.tools.items()
                if pattern.search(tool_def.get("function", {}).get("description", ""))
            ]
            return results[:self.max_results]
        except re.error:
            return []

    def _run_custom_search(
        self,
        queries: list[str],
        tool_defs: list[dict],
    ) -> list[dict]:
        """Run a custom search function and return matching tools.

        Args:
            queries: List of search queries.
            tool_defs: List of tool definitions to search.

        Returns:
            A list of matching tool definitions.
        """
        # Check if the result is a coroutine (async search function)
        result = self.search_fn(queries, tool_defs)
        if asyncio.iscoroutine(result):
            # For sync usage, run the coroutine
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, create a new one
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(result)
                loop.close()
            else:
                result = loop.run_until_complete(result)
        else:
            # Sync result
            pass

        # Convert tool names to tool definitions
        matched_names = set(result)
        matched_tools = [
            tool_def
            for tool_def in tool_defs
            if tool_def.get("function", {}).get("name") in matched_names
        ]
        return matched_tools[:self.max_results]

    def search_capabilities(self, query: str) -> list[BaseCapability]:
        """Search for capabilities matching the query.

        Args:
            query: The search query.

        Returns:
            A list of matching capabilities.
        """
        query_lower = query.lower()
        return [
            cap
            for cap in self.capabilities.values()
            if query_lower in cap.id.lower() or query_lower in cap.description.lower()
        ]
