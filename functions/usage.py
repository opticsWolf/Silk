"""Usage statistics for runs.

Tracks token usage and request counts.
"""
from __future__ import annotations


class UsageStats:
    """Usage statistics for a run.

    Attributes:
        input_tokens: Total input tokens used.
        output_tokens: Total output tokens used.
        requests: Total number of requests made.
        tool_calls: Total number of tool calls made.
    """

    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        requests: int = 0,
        tool_calls: int = 0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.requests = requests
        self.tool_calls = tool_calls

    def record_request(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record a request with token counts."""
        self.requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def record_tool_call(self) -> None:
        """Record a tool call."""
        self.tool_calls += 1

    def snapshot(self) -> dict:
        """Return a serialisable snapshot."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "requests": self.requests,
            "tool_calls": self.tool_calls,
        }

    def restore(self, snapshot: dict) -> None:
        """Restore from a snapshot."""
        self.input_tokens = snapshot.get("input_tokens", 0)
        self.output_tokens = snapshot.get("output_tokens", 0)
        self.requests = snapshot.get("requests", 0)
        self.tool_calls = snapshot.get("tool_calls", 0)

    def __repr__(self) -> str:
        return (
            f"UsageStats(input_tokens={self.input_tokens}, "
            f"output_tokens={self.output_tokens}, "
            f"requests={self.requests}, "
            f"tool_calls={self.tool_calls})"
        )
