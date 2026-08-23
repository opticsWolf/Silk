"""Usage limits for runs.

Prevents runaway costs by limiting output tokens, requests, and tool calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class UsageLimits:
    """Limits for a single run.

    All limits are optional; ``None`` means no limit.
    """

    output_tokens_limit: int | None = None
    """Maximum output tokens for the entire run."""

    input_tokens_limit: int | None = None
    """Maximum input tokens for the entire run."""

    request_limit: int | None = None
    """Maximum number of model requests (turns) for the entire run."""

    tool_calls_limit: int | None = None
    """Maximum number of successful tool executions for the entire run."""

    # Internal counters (not serialized)
    _output_tokens_used: int = field(default=0, repr=False)
    _input_tokens_used: int = field(default=0, repr=False)
    _request_count: int = field(default=0, repr=False)
    _tool_call_count: int = field(default=0, repr=False)

    # â”€â”€ Checkers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_output_tokens(self, tokens: int) -> None:
        """Raise ``UsageLimitExceeded`` if *tokens* would exceed the limit."""
        if self.output_tokens_limit is not None:
            if self._output_tokens_used + tokens > self.output_tokens_limit:
                raise UsageLimitExceeded(
                    f"output_tokens_limit of {self.output_tokens_limit} "
                    f"(would use {_format_tokens(self._output_tokens_used + tokens)} "
                    f"but only {self.output_tokens_limit - self._output_tokens_used} remaining)"
                )

    def check_input_tokens(self, tokens: int) -> None:
        """Raise ``UsageLimitExceeded`` if *tokens* would exceed the limit."""
        if self.input_tokens_limit is not None:
            if self._input_tokens_used + tokens > self.input_tokens_limit:
                raise UsageLimitExceeded(
                    f"input_tokens_limit of {self.input_tokens_limit} "
                    f"(would use {_format_tokens(self._input_tokens_used + tokens)} "
                    f"but only {self.input_tokens_limit - self._input_tokens_used} remaining)"
                )

    def check_request(self) -> None:
        """Raise ``UsageLimitExceeded`` if this would be the Nth request."""
        if self.request_limit is not None:
            if self._request_count >= self.request_limit:
                raise UsageLimitExceeded(
                    f"request_limit of {self.request_limit} "
                    f"(already made {self._request_count} requests)"
                )

    def check_tool_calls(self, count: int = 1) -> None:
        """Raise ``UsageLimitExceeded`` if *count* tool calls would exceed the limit."""
        if self.tool_calls_limit is not None:
            if self._tool_call_count + count > self.tool_calls_limit:
                raise UsageLimitExceeded(
                    f"tool_calls_limit of {self.tool_calls_limit} "
                    f"(would use {_format_tool_calls(self._tool_call_count + count)} "
                    f"but only {self.tool_calls_limit - self._tool_call_count} remaining)"
                )

    # â”€â”€ Recorders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def record_output_tokens(self, tokens: int) -> None:
        """Record *tokens* of output."""
        self._output_tokens_used += tokens

    def record_input_tokens(self, tokens: int) -> None:
        """Record *tokens* of input."""
        self._input_tokens_used += tokens

    def record_request(self) -> None:
        """Record a model request."""
        self._request_count += 1

    def record_tool_calls(self, count: int = 1) -> None:
        """Record *count* successful tool executions."""
        self._tool_call_count += count

    # â”€â”€ Snapshot / restore â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of current counters."""
        return {
            "_output_tokens_used": self._output_tokens_used,
            "_input_tokens_used": self._input_tokens_used,
            "_request_count": self._request_count,
            "_tool_call_count": self._tool_call_count,
        }

    def restore(self, snapshot: dict) -> None:
        """Restore counters from a snapshot."""
        self._output_tokens_used = snapshot.get("_output_tokens_used", 0)
        self._input_tokens_used = snapshot.get("_input_tokens_used", 0)
        self._request_count = snapshot.get("_request_count", 0)
        self._tool_call_count = snapshot.get("_tool_call_count", 0)


class UsageLimitExceeded(Exception):
    """Raised when a usage limit is exceeded."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _format_tokens(tokens: int) -> str:
    """Format token count for error messages."""
    return f"{tokens:,}"


def _format_tool_calls(count: int) -> str:
    """Format tool call count for error messages."""
    return f"{count:,}"
