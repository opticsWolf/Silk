"""Usage limits for runs.

Prevents runaway costs by limiting output tokens, requests, and tool calls.

**Thread safety.** One ``UsageLimits`` is threaded into every worker of an
orchestrator fan-out so the whole fan-out respects a single cap. That makes
the counters shared mutable state, and the check/record pair a
check-then-act race: several workers can pass the same check and then all
record, collectively overrunning the limit the object exists to enforce
(spec D52.4). Every counter access therefore happens under one lock, and
the ``reserve_*`` methods do the check **and** the record inside it -- those
are what callers on the hot path use. ``check_*`` / ``record_*`` remain for
callers that genuinely need the two halves apart.

**Nesting (spec D26, closes T3).** One shared cap answers "the fan-out may
not cost more than this" and nothing else: a greedy worker can still spend
the whole allowance and leave the rest of the fan-out with nothing but
``USAGE_LIMIT`` events. :class:`SubBudget` is the second half -- a worker's
own caps *inside* the shared one. It claims from itself first, then from
its parent, and refunds itself if the parent refuses, so a worker can
never be charged for a request it was not allowed to make. Both caps bind:
whichever is exhausted first is the one the worker hears about, and no
sub-budget can raise the global ceiling.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

    #: Guards every counter read and write. Reentrant so a ``reserve_*``
    #: can be written in terms of the ``check_*`` it already owns.
    _lock: Any = field(
        default_factory=threading.RLock, repr=False, compare=False,
    )

    # â”€â”€ Checkers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_output_tokens(self, tokens: int) -> None:
        """Raise ``UsageLimitExceeded`` if *tokens* would exceed the limit."""
        with self._lock:
            if self.output_tokens_limit is None:
                return
            if self._output_tokens_used + tokens > self.output_tokens_limit:
                raise UsageLimitExceeded(
                    f"output_tokens_limit of {self.output_tokens_limit} "
                    f"(would use {_format_tokens(self._output_tokens_used + tokens)} "
                    f"but only {self.output_tokens_limit - self._output_tokens_used} remaining)"
                )

    def check_input_tokens(self, tokens: int) -> None:
        """Raise ``UsageLimitExceeded`` if *tokens* would exceed the limit."""
        with self._lock:
            if self.input_tokens_limit is None:
                return
            if self._input_tokens_used + tokens > self.input_tokens_limit:
                raise UsageLimitExceeded(
                    f"input_tokens_limit of {self.input_tokens_limit} "
                    f"(would use {_format_tokens(self._input_tokens_used + tokens)} "
                    f"but only {self.input_tokens_limit - self._input_tokens_used} remaining)"
                )

    def check_request(self) -> None:
        """Raise ``UsageLimitExceeded`` if this would be the Nth request."""
        with self._lock:
            if self.request_limit is None:
                return
            if self._request_count >= self.request_limit:
                raise UsageLimitExceeded(
                    f"request_limit of {self.request_limit} "
                    f"(already made {self._request_count} requests)"
                )

    def check_tool_calls(self, count: int = 1) -> None:
        """Raise ``UsageLimitExceeded`` if *count* tool calls would exceed the limit."""
        with self._lock:
            if self.tool_calls_limit is None:
                return
            if self._tool_call_count + count > self.tool_calls_limit:
                raise UsageLimitExceeded(
                    f"tool_calls_limit of {self.tool_calls_limit} "
                    f"(would use {_format_tool_calls(self._tool_call_count + count)} "
                    f"but only {self.tool_calls_limit - self._tool_call_count} remaining)"
                )

    # â”€â”€ Recorders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def record_output_tokens(self, tokens: int) -> None:
        """Record *tokens* of output."""
        with self._lock:
            self._output_tokens_used += tokens

    def record_input_tokens(self, tokens: int) -> None:
        """Record *tokens* of input."""
        with self._lock:
            self._input_tokens_used += tokens

    def record_request(self) -> None:
        """Record a model request."""
        with self._lock:
            self._request_count += 1

    def record_tool_calls(self, count: int = 1) -> None:
        """Record *count* successful tool executions."""
        with self._lock:
            self._tool_call_count += count

    # -- Reservations (check + record, atomically) ---------------------------

    def reserve_request(self) -> None:
        """Claim one model request, or raise without claiming it.

        The atomic form of ``check_request`` + ``record_request``. A shared
        budget must be claimed in one step, or two workers both pass the
        check before either records (spec D52.4).
        """
        with self._lock:
            self.check_request()
            self.record_request()

    def reserve_tool_calls(self, count: int = 1) -> None:
        """Claim *count* tool executions, or raise without claiming them."""
        with self._lock:
            self.check_tool_calls(count)
            self.record_tool_calls(count)

    def reserve_output_tokens(self, tokens: int) -> None:
        """Claim *tokens* of output, or raise without claiming them."""
        with self._lock:
            self.check_output_tokens(tokens)
            self.record_output_tokens(tokens)

    def reserve_input_tokens(self, tokens: int) -> None:
        """Claim *tokens* of input, or raise without claiming them."""
        with self._lock:
            self.check_input_tokens(tokens)
            self.record_input_tokens(tokens)


    # â”€â”€ Snapshot / restore â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of current counters."""
        with self._lock:
            return {
            "_output_tokens_used": self._output_tokens_used,
            "_input_tokens_used": self._input_tokens_used,
            "_request_count": self._request_count,
            "_tool_call_count": self._tool_call_count,
        }

    def restore(self, snapshot: dict) -> None:
        """Restore counters from a snapshot."""
        with self._lock:
            self._output_tokens_used = snapshot.get("_output_tokens_used", 0)
            self._input_tokens_used = snapshot.get("_input_tokens_used", 0)
            self._request_count = snapshot.get("_request_count", 0)
            self._tool_call_count = snapshot.get("_tool_call_count", 0)


    # -- Nesting (spec D26) -----------------------------------------------

    def _refund(self, *, output_tokens: int = 0, input_tokens: int = 0,
                requests: int = 0, tool_calls: int = 0) -> None:
        """Give back what was claimed a moment ago and cannot be spent.

        Only :class:`SubBudget` uses this, and only to undo its *own*
        claim when its parent refuses. Counters never go below zero: a
        refund of something that was never claimed is a bug, and reading
        it as a negative allowance would hide it.
        """
        with self._lock:
            self._output_tokens_used = max(
                0, self._output_tokens_used - output_tokens)
            self._input_tokens_used = max(
                0, self._input_tokens_used - input_tokens)
            self._request_count = max(0, self._request_count - requests)
            self._tool_call_count = max(0, self._tool_call_count - tool_calls)


@dataclass
class SubBudget(UsageLimits):
    """One worker's own caps, inside a shared one (spec D26, T3).

    Every check consults both, and every reservation claims from both --
    this budget first, because it is uncontended, then the parent. If the
    parent refuses, this one refunds itself before the exception leaves,
    so a worker is never charged for what it did not get.

    Ordering is deliberate: claiming the parent first and this one second
    would leave the *shared* counter transiently over-charged, which is
    the counter other threads read.
    """

    parent: Any = None

    # â”€â”€ checks: both ceilings bind â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check_output_tokens(self, tokens: int) -> None:
        super().check_output_tokens(tokens)
        if self.parent is not None:
            self.parent.check_output_tokens(tokens)

    def check_input_tokens(self, tokens: int) -> None:
        super().check_input_tokens(tokens)
        if self.parent is not None:
            self.parent.check_input_tokens(tokens)

    def check_request(self) -> None:
        super().check_request()
        if self.parent is not None:
            self.parent.check_request()

    def check_tool_calls(self, count: int = 1) -> None:
        super().check_tool_calls(count)
        if self.parent is not None:
            self.parent.check_tool_calls(count)

    # â”€â”€ reservations: mine, then the shared one, or neither â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def reserve_request(self) -> None:
        with self._lock:
            UsageLimits.check_request(self)
            self.record_request()
        if self.parent is None:
            return
        try:
            self.parent.reserve_request()
        except UsageLimitExceeded:
            self._refund(requests=1)
            raise

    def reserve_tool_calls(self, count: int = 1) -> None:
        with self._lock:
            UsageLimits.check_tool_calls(self, count)
            self.record_tool_calls(count)
        if self.parent is None:
            return
        try:
            self.parent.reserve_tool_calls(count)
        except UsageLimitExceeded:
            self._refund(tool_calls=count)
            raise

    def reserve_output_tokens(self, tokens: int) -> None:
        with self._lock:
            UsageLimits.check_output_tokens(self, tokens)
            self.record_output_tokens(tokens)
        if self.parent is None:
            return
        try:
            self.parent.reserve_output_tokens(tokens)
        except UsageLimitExceeded:
            self._refund(output_tokens=tokens)
            raise

    def reserve_input_tokens(self, tokens: int) -> None:
        with self._lock:
            UsageLimits.check_input_tokens(self, tokens)
            self.record_input_tokens(tokens)
        if self.parent is None:
            return
        try:
            self.parent.reserve_input_tokens(tokens)
        except UsageLimitExceeded:
            self._refund(input_tokens=tokens)
            raise

    def snapshot(self) -> dict:
        """This worker's counters, plus the shared ones it is spending."""
        body = super().snapshot()
        if self.parent is not None:
            body["shared"] = self.parent.snapshot()
        return body


def nest(shared: Any, own: Any) -> Any:
    """The budget a worker actually runs under (spec D26).

    Either half may be missing: a fan-out with only a global cap behaves
    exactly as it did before sub-budgets existed, and a worker with only
    its own caps and no orchestrator keeps them. With both, the worker's
    caps become a :class:`SubBudget` of the shared one -- *inside* it,
    never beside it, because a per-worker allowance that could exceed the
    global cap would not be a sub-budget at all.

    An ``own`` that is already a :class:`SubBudget` of this parent is
    returned unchanged, so re-entering a run does not nest twice.
    """
    if own is None:
        return shared
    if shared is None:
        return own
    if isinstance(own, SubBudget) and own.parent is shared:
        return own
    return SubBudget(
        output_tokens_limit=own.output_tokens_limit,
        input_tokens_limit=own.input_tokens_limit,
        request_limit=own.request_limit,
        tool_calls_limit=own.tool_calls_limit,
        parent=shared,
    )


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
