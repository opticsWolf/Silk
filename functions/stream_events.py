"""Typed streaming events for structured LLM output.

Provides a typed event system that replaces ad-hoc token deltas with
structured events. This enables better observability, tool-call tracking,
and UI rendering without changing the underlying streaming mechanism.

Event types
-----------
* ``EventStart``       â€“ generation started (with model settings)
* ``EventDelta``       â€“ incremental text delta (like current token signal)
* ``EventToolCall``    â€“ a tool call was detected in the output
* ``EventToolResult``  â€“ tool result received
* ``EventFinalResult`` â€“ generation finished (text + stats)
* ``EventRunResult``   â€“ complete run result (text, tokens, stats, metadata)
* ``EventError``       â€“ an error occurred during generation
* ``EventUsageLimit``  â€“ a usage limit was exceeded
* ``EventReflection``  â€“ a reflection/retry was triggered
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# â”€â”€ Event Types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class EventType(Enum):
    """Enum of all streaming event types."""
    START = "start"
    DELTA = "delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_RESULT = "final_result"
    RUN_RESULT = "run_result"
    ERROR = "error"
    USAGE_LIMIT = "usage_limit"
    REFLECTION = "reflection"


#: How a run ended, on ``EventRunResult.outcome``. A consumer must key off
#: this and never off "is there a final text": a run that hit ``max_rounds``
#: produces text and is not a success, which is exactly how that abort used
#: to be reported as a clean finish (G13).
OUTCOME_COMPLETED = "completed"
OUTCOME_STOPPED = "stopped"
OUTCOME_USAGE_LIMITED = "usage_limited"
OUTCOME_ERROR = "error"


# â”€â”€ Event Classes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class EventStart:
    """Emitted when generation starts.

    Attributes:
        timestamp: When the event was created.
        model: Model identifier (if available).
        settings: Generation settings (temperature, top_p, etc.).
        input_tokens: Estimated input token count.
        system_prompt: The system prompt being used.
        context_length: The backend's context window, when it is knowable —
            the denominator every context-pressure decision needs, and which
            never used to reach the loop at all (G14c).
    """
    timestamp: datetime = field(default_factory=datetime.now)
    model: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    input_tokens: int | None = None
    system_prompt: str | None = None
    context_length: int | None = None


@dataclass
class EventDelta:
    """Emitted for each incremental text delta during streaming.

    Attributes:
        timestamp: When the event was created.
        delta: The text delta (substring to append).
        total_tokens: Cumulative token count so far.
        cumulative_text: Full text accumulated so far.
        tps: Current tokens-per-second rate.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    delta: str = ""
    total_tokens: int = 0
    cumulative_text: str = ""
    tps: float = 0.0


@dataclass
class EventToolCall:
    """Emitted when a tool call is detected in the model output.

    Attributes:
        timestamp: When the event was created.
        tool_name: Name of the tool being called.
        tool_args: Arguments passed to the tool.
        call_id: Unique identifier for this tool call.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass
class EventToolResult:
    """Emitted when a tool result is received.

    Attributes:
        timestamp: When the event was created.
        tool_name: Name of the tool that produced the result.
        result: The tool result text.
        call_id: The call ID this result corresponds to.
        error: Whether this result indicates an error.
        error_message: Error message if error=True.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    tool_name: str = ""
    result: str = ""
    call_id: str = ""
    error: bool = False
    error_message: str | None = None


@dataclass
class EventFinalResult:
    """Emitted when generation finishes (normal or errored).

    Attributes:
        timestamp: When the event was created.
        text: The full generated text.
        tokens: Output token count.
        input_tokens: Input token count.
        tps: Tokens per second.
        finish_reason: Reason the generation finished.
        error: Error message if an error occurred.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    text: str = ""
    tokens: int = 0
    input_tokens: int = 0
    tps: float = 0.0
    finish_reason: str = "stop"
    error: str | None = None


@dataclass
class EventRunResult:
    """Emitted when a complete run finishes (after all tool calls).

    This is the final event in a run sequence and contains all
    accumulated results including tool call history.

    Attributes:
        timestamp: When the event was created.
        text: The final generated text.
        tokens: Output token count.
        input_tokens: Input token count.
        tps: Tokens per second.
        finish_reason: Reason the generation finished.
        tool_calls: List of tool calls made during the run.
        tool_results: List of tool results received during the run.
        error: Error message if an error occurred.
        usage_stats: Usage statistics for the run.
        outcome: How the run ended — one of ``completed``, ``stopped``,
            ``usage_limited``, ``error``. See the OUTCOME_* constants.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    text: str = ""
    tokens: int = 0
    input_tokens: int = 0
    tps: float = 0.0
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    usage_stats: dict[str, Any] = field(default_factory=dict)
    outcome: str = OUTCOME_COMPLETED


@dataclass
class EventError:
    """Emitted when an error occurs during generation.

    Attributes:
        timestamp: When the event was created.
        error: The exception or error message.
        context: Additional context about where the error occurred.
        recoverable: Whether the error can be recovered from.
        kind: For a model-request failure, the classification from
            ``model_errors`` (``overflow`` / ``retryable`` / ``terminal`` /
            ``truncated``). Empty for errors raised elsewhere. Consumers key
            compaction off ``kind == "overflow"``, never off ``context``,
            which covers every stream failure alike (spec D40).
    """
    timestamp: datetime = field(default_factory=datetime.now)
    error: str = ""
    context: str = ""
    recoverable: bool = True
    kind: str = ""


@dataclass
class EventUsageLimit:
    """Emitted when a usage limit is exceeded.

    Attributes:
        timestamp: When the event was created.
        limit_type: The type of limit that was exceeded.
        limit_value: The limit value.
        current_value: The current usage value.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    limit_type: str = ""
    limit_value: int = 0
    current_value: int = 0


@dataclass
class EventReflection:
    """Emitted when a reflection/retry is triggered.

    Attributes:
        timestamp: When the event was created.
        retry_count: Current retry count.
        max_retries: Maximum number of retries allowed.
        error_type: Type of error that triggered the retry.
        error_message: The error message.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    error_type: str = ""
    error_message: str = ""


# â”€â”€ Event Builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class EventBuilder:
    """Builder for creating typed events from raw data.

    Provides static factory methods for creating events from
    the raw data structures used by the streaming system.

    Example:
        >>> start = EventBuilder.build_start(
        ...     model="llama-3.1",
        ...     settings={"temperature": 0.7}
        ... )
        >>> delta = EventBuilder.build_delta(
        ...     text="Hello",
        ...     total_tokens=10,
        ...     tps=25.5
        ... )
    """

    @staticmethod
    def build_start(
        model: str | None = None,
        settings: dict[str, Any] | None = None,
        input_tokens: int | None = None,
        system_prompt: str | None = None,
        context_length: int | None = None,
    ) -> EventStart:
        """Build an EventStart from raw data."""
        return EventStart(
            timestamp=datetime.now(),
            model=model,
            settings=settings or {},
            input_tokens=input_tokens,
            system_prompt=system_prompt,
            context_length=context_length,
        )

    @staticmethod
    def build_delta(
        delta: str,
        total_tokens: int = 0,
        cumulative_text: str = "",
        tps: float = 0.0,
    ) -> EventDelta:
        """Build an EventDelta from raw data."""
        return EventDelta(
            timestamp=datetime.now(),
            delta=delta,
            total_tokens=total_tokens,
            cumulative_text=cumulative_text,
            tps=tps,
        )

    @staticmethod
    def build_tool_call(
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        call_id: str = "",
    ) -> EventToolCall:
        """Build an EventToolCall from raw data."""
        return EventToolCall(
            timestamp=datetime.now(),
            tool_name=tool_name,
            tool_args=tool_args or {},
            call_id=call_id,
        )

    @staticmethod
    def build_tool_result(
        tool_name: str,
        result: str,
        call_id: str = "",
        error: bool = False,
        error_message: str | None = None,
    ) -> EventToolResult:
        """Build an EventToolResult from raw data."""
        return EventToolResult(
            timestamp=datetime.now(),
            tool_name=tool_name,
            result=result,
            call_id=call_id,
            error=error,
            error_message=error_message,
        )

    @staticmethod
    def build_final_result(
        text: str = "",
        tokens: int = 0,
        input_tokens: int = 0,
        tps: float = 0.0,
        finish_reason: str = "stop",
        error: str | None = None,
    ) -> EventFinalResult:
        """Build an EventFinalResult from raw data."""
        return EventFinalResult(
            timestamp=datetime.now(),
            text=text,
            tokens=tokens,
            input_tokens=input_tokens,
            tps=tps,
            finish_reason=finish_reason,
            error=error,
        )

    @staticmethod
    def build_run_result(
        text: str = "",
        tokens: int = 0,
        input_tokens: int = 0,
        tps: float = 0.0,
        finish_reason: str = "stop",
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        error: str | None = None,
        usage_stats: dict[str, Any] | None = None,
        outcome: str = OUTCOME_COMPLETED,
    ) -> EventRunResult:
        """Build an EventRunResult from raw data."""
        return EventRunResult(
            timestamp=datetime.now(),
            text=text,
            tokens=tokens,
            input_tokens=input_tokens,
            tps=tps,
            finish_reason=finish_reason,
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
            error=error,
            usage_stats=usage_stats or {},
            outcome=outcome,
        )

    @staticmethod
    def build_error(
        error: str,
        context: str = "",
        recoverable: bool = True,
        kind: str = "",
    ) -> EventError:
        """Build an EventError from raw data."""
        return EventError(
            timestamp=datetime.now(),
            error=error,
            context=context,
            recoverable=recoverable,
            kind=kind,
        )

    @staticmethod
    def build_usage_limit(
        limit_type: str,
        limit_value: int,
        current_value: int,
    ) -> EventUsageLimit:
        """Build an EventUsageLimit from raw data."""
        return EventUsageLimit(
            timestamp=datetime.now(),
            limit_type=limit_type,
            limit_value=limit_value,
            current_value=current_value,
        )

    @staticmethod
    def build_reflection(
        retry_count: int = 0,
        max_retries: int = 3,
        error_type: str = "",
        error_message: str = "",
    ) -> EventReflection:
        """Build an EventReflection from raw data."""
        return EventReflection(
            timestamp=datetime.now(),
            retry_count=retry_count,
            max_retries=max_retries,
            error_type=error_type,
            error_message=error_message,
        )


# â”€â”€ Event Stream â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class EventStream:
    """Async iterator over streaming events.

    Wraps an async generator that yields typed events.
    Provides a clean interface for consuming the event stream.

    Example:
        >>> async for event in EventStream(async_gen()):
        ...     if isinstance(event, EventDelta):
        ...         print(event.delta, end="")
        ...     elif isinstance(event, EventFinalResult):
        ...         print(f"\\nDone: {event.tokens} tokens")
    """

    def __init__(self, async_generator):
        self._async_gen = async_generator

    def __aiter__(self):
        return self._async_gen

    async def collect(self) -> list:
        """Collect all events from the stream into a list."""
        events = []
        async for event in self._async_gen:
            events.append(event)
        return events

    async def get_final_result(self) -> EventFinalResult | EventRunResult | None:
        """Get the final result event from the stream."""
        async for event in self._async_gen:
            if isinstance(event, (EventFinalResult, EventRunResult)):
                return event
        return None

    async def get_text(self) -> str:
        """Get the full text from the stream by collecting all deltas."""
        text = ""
        async for event in self._async_gen:
            if isinstance(event, EventDelta):
                text += event.delta
        return text
