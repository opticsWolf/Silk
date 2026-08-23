# -*- coding: utf-8 -*-
"""Interface contracts between the Silk agent loop, engine, and tool registry.

These Protocols pin the previously duck-typed wiring: the AgentLoop binds to
``AgentEngine`` + ``ToolRegistry`` instead of reaching into concrete classes,
so tests can substitute fakes and alternative engines (remote APIs, mock
models) can drop in without touching the loop.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentEngine(Protocol):
    """What the AgentLoop needs from an engine.

    An engine owns the conversation history and performs exactly one model
    request per :meth:`stream_response` call. It never executes tools and
    never loops — multi-turn behaviour belongs to the AgentLoop.
    """

    usage_limits: Any            # UsageLimits
    reflection_config: Any       # ReflectionConfig
    history: list[dict[str, Any]]
    last_stats: dict[str, Any]

    def stream_response(self, gen_params: dict[str, Any]) -> Iterator[str]:
        """Yield incremental text deltas for one model request."""
        ...

    def append_message(self, role: str, content: str, **stats: Any) -> None:
        """Append a turn (user / assistant / tool / system) to the history."""
        ...

    def count_prompt_tokens(self) -> int:
        """Best-effort input-token count for the current prompt state."""
        ...

    def request_stop(self) -> None:
        """Ask the in-flight generation to stop at the next token."""
        ...

    def stop_requested(self) -> bool:
        """Whether a stop has been requested for the current run."""
        ...


@runtime_checkable
class ToolRegistry(Protocol):
    """What the AgentLoop needs from a tool registry (ToolBox satisfies it)."""

    tools: dict[str, dict[str, Any]]

    async def execute_tool_calls_async(self, tool_calls: list[Any]) -> list[dict]:
        """Validate and run a batch of tool calls; errors return as results."""
        ...
