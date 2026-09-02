"""Tool search for deferred tool discovery.

Enables searching for tools by name or description, useful for
on-demand capability loading where only relevant tools are sent to
the model.

Two things make this model-facing rather than internal (spec D4, G2):

*The role gate applies here too* (I8). The index is populated at attach
time and knows nothing about which role is active, so without a filter
discovery would advertise exactly what dispatch is going to refuse --
half of I4, from the wrong side. ``permits`` is that filter; the ToolBox
installs its own ``role_permits`` when it builds the index.

*Ranking is load-bearing* now that a search result is how a tool reaches
the model at all, so ``bm25`` is a real ranking function rather than an
alias for ``keywords`` (closes G2).
"""
from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING


#: BM25 term-frequency saturation and length normalisation. The defaults
#: from the literature; tool descriptions are short and uniform enough that
#: tuning them here would be superstition.
BM25_K1 = 1.5
BM25_B = 0.75

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Words of *text*, lowercased, with ``snake_case`` split apart.

    Tool names carry most of the signal and are written as identifiers, so
    ``read_file`` has to match a query saying "read a file".
    """
    return _WORD.findall((text or "").lower())


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

    permits: Callable[[str], bool] | None = field(default=None)
    """Role gate, or ``None`` to allow everything.

    Called with a tool name; a false answer removes the tool from every
    result. The ToolBox installs :meth:`ToolBox.role_permits` here so that
    what discovery offers and what dispatch accepts cannot drift (I8).
    """

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

    # -- the visible corpus ------------------------------------------

    def visible(self) -> dict[str, dict]:
        """The tools this search may return, after the role gate."""
        if self.permits is None:
            return dict(self.tools)
        return {name: tool_def for name, tool_def in self.tools.items()
                if self.permits(name)}

    @staticmethod
    def _text(tool_name: str, tool_def: dict) -> str:
        function = tool_def.get("function", tool_def) or {}
        return f"{tool_name} {function.get('description', '')}"

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
        tool_defs = list(self.visible().values())

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

        for tool_name, tool_def in self.visible().items():
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
        """BM25 ranking over tool names and descriptions.

        Okapi BM25 with the usual constants: a term saturates rather than
        scoring linearly (``k1``), a long description is not rewarded for
        its length (``b``), and a term appearing in every tool carries
        almost no information (the idf factor). That last property is what
        makes this worth having over keyword overlap in a tool corpus,
        where words like "file" or "list" are in half the descriptions.

        Args:
            query: The search query.

        Returns:
            A list of matching tool definitions, sorted by relevance.
        """
        corpus = self.visible()
        if not corpus:
            return []

        documents = {name: tokenize(self._text(name, tool_def))
                     for name, tool_def in corpus.items()}
        lengths = {name: len(tokens) for name, tokens in documents.items()}
        total = sum(lengths.values())
        if not total:
            return []
        average = total / len(documents)
        frequencies = {name: Counter(tokens) for name, tokens in documents.items()}

        containing: Counter = Counter()
        for counts in frequencies.values():
            containing.update(counts.keys())

        count = len(documents)
        scores: dict[str, float] = {}
        for term in tokenize(query):
            appearances = containing.get(term, 0)
            if not appearances:
                continue
            idf = math.log(1 + (count - appearances + 0.5) / (appearances + 0.5))
            for name, counts in frequencies.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                norm = 1 - BM25_B + BM25_B * (lengths[name] / average)
                scores[name] = scores.get(name, 0.0) + idf * (
                    frequency * (BM25_K1 + 1) / (frequency + BM25_K1 * norm)
                )

        ranked = sorted(scores, key=lambda n: (-scores[n], n))
        return [corpus[name] for name in ranked[:self.max_results]]

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
                for tool_name, tool_def in self.visible().items()
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

        # Convert tool names to tool definitions. A custom strategy is
        # handed the visible corpus, but it may return anything it likes,
        # so the gate is applied to its answer as well.
        matched_names = {
            name for name in set(result)
            if self.permits is None or self.permits(name)
        }
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
        matches = [
            cap
            for cap in self.capabilities.values()
            if query_lower in cap.id.lower() or query_lower in cap.description.lower()
        ]
        if self.permits is None:
            return matches
        # A capability every one of whose tools the role forbids is not a
        # candidate to load: offering it would be the same drift between
        # discovery and dispatch that `permits` exists to prevent (I8). One
        # declaring no tools at all is instructions or hooks, and stays.
        allowed = []
        for cap in matches:
            names = [
                (tool.get("function", tool) or {}).get("name")
                for tool in (cap.get_tools() or [])
            ]
            names = [name for name in names if name]
            if not names or any(self.permits(name) for name in names):
                allowed.append(cap)
        return allowed
