# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The seam a tool uses to touch the canvas (spec D70).

A tool runs inside ``ToolBox.execute_tool_calls_async`` on the agent's
``ThreadedNode`` worker. Qt says the scene, the nodes and the undo stack
belong to the main thread. The existing worker→main channels (``pulse``,
``emit_stream``) are one-way and return nothing, which is precisely the
gap D49 filled for human decisions -- so this is D49's machinery with a
different resolver: not a person, the event loop.

The waiting, the correlation, the ordering rule and the four failure
causes are :class:`~.blocking_seam.BlockingSeam`'s. What this module adds
is the vocabulary of a *call*: an operation name with arguments, and a
result that is either a value or a reason there is none.

**Fail-closed, same as the human seam** (D36). No resolver (a headless
evaluation, a graph with no canvas, a subagent), a canvas destroyed
mid-request, a handler that raises, a timeout: each wakes the worker with
a named cause and no mutation. A graph-authoring tool that cannot reach
the main thread refuses; it never hangs, and it never half-builds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from weave.logger import get_logger

from .blocking_seam import (
    CAUSE_ANSWERED, CAUSE_CANCELLED, CAUSE_NO_ANSWERER, CAUSE_TIMEOUT,
    CAUSE_TRANSPORT, BlockingSeam, DriveGate, new_request_id,
)

log = get_logger("SilkMainCall")

#: How long a worker waits for the main thread. Far shorter than the
#: human seam's five minutes: nobody is thinking about this one. If the
#: event loop has not run a queued slot in ten seconds it is wedged, busy
#: in another node's compute, or gone -- and the honest answer to the
#: model is a refusal it can act on, not a worker parked behind a
#: RoleBinding for minutes.
DEFAULT_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class CallRequest:
    """One thing to do on the main thread."""

    call_id: str
    op: str
    args: dict = field(default_factory=dict)
    #: Free-form context for a UI that wants to show what is happening.
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CallResult:
    """What happened. ``ok`` is False for every cause but a real answer.

    There is no path through this object that reports a mutation nobody
    performed: the only ``ok=True`` results are the ones a resolver
    committed after its handler returned.
    """

    call_id: str
    op: str = ""
    ok: bool = False
    value: Any = None
    error: str = ""
    cause: str = CAUSE_ANSWERED

    @property
    def performed(self) -> bool:
        """Whether the main thread actually ran the handler."""
        return self.cause == CAUSE_ANSWERED

    def failure_text(self) -> str:
        """A sentence for the model, saying which of the five happened."""
        if self.ok:
            return "Done."
        if self.cause == CAUSE_ANSWERED:
            return self.error or "The operation was refused."
        return {
            CAUSE_CANCELLED: "Refused: the run was stopped before the graph "
                             "change could be made.",
            CAUSE_TIMEOUT: "Refused: the canvas did not respond in time; no "
                           "change was made.",
            CAUSE_NO_ANSWERER: "Refused: this run has no canvas to edit "
                               "(headless evaluation or a closed graph).",
            CAUSE_TRANSPORT: "Refused: the request could not reach the "
                             "canvas.",
        }.get(self.cause, "Refused.")


class MainThreadCall(BlockingSeam):
    """Run something on the main thread and block the worker for its result.

    *deliver* hands a :class:`CallRequest` to the main thread -- in Weave
    that is a queued Qt signal emitted from the node. ``None`` means there
    is no canvas, and every call then refuses immediately rather than
    spending the timeout to reach the same answer.

    The main thread answers with :meth:`serve`, which runs the handler and
    commits the outcome **under the lock before waking the worker**. That
    ordering is the seam's reason to exist; see
    :mod:`~.blocking_seam`.
    """

    def __init__(
        self,
        deliver: Optional[Callable[[CallRequest], Any]] = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        drive: Optional[DriveGate] = None,
    ) -> None:
        super().__init__(deliver, timeout_s=timeout_s, drive=drive)

    # -- the worker side --------------------------------------------------

    def call(self, op: str, *, timeout: Optional[float] = None,
             detail: Optional[dict] = None, **args: Any) -> CallResult:
        """Ask the main thread to do *op*, and wait. **Worker thread.**

        Returns rather than raises on every path, because the caller is
        producing a tool result and an exception there would be a crash
        where a refusal was meant.
        """
        request = CallRequest(call_id=new_request_id(), op=op, args=dict(args),
                              detail=dict(detail or {}))
        return self.submit(
            request.call_id, request, timeout=timeout,
            failed=lambda cause, reason: CallResult(
                call_id=request.call_id, op=op, ok=False, cause=cause,
                error=reason,
            ),
        )

    # -- the resolver side ------------------------------------------------

    def serve(self, request: CallRequest,
              handler: Callable[[CallRequest], Any]) -> bool:
        """Run *handler* and commit its outcome. **Main thread.**

        A handler that raises becomes a refusal carrying the exception's
        text, never a propagating exception: this runs inside a Qt slot,
        where an escaping exception would take out the event loop and
        leave the worker to time out for no reason.
        """
        try:
            value = handler(request)
        except Exception as exc:  # noqa: BLE001 -- the handler is arbitrary
            log.warning(f"main-thread call '{request.op}' failed: {exc}")
            return self.complete(request.call_id, ok=False, op=request.op,
                                 error=f"{type(exc).__name__}: {exc}")
        if isinstance(value, CallResult):
            return self.commit(request.call_id, value)
        if isinstance(value, dict) and "ok" in value:
            return self.complete(
                request.call_id, ok=bool(value.get("ok")), op=request.op,
                value=value.get("value"), error=str(value.get("error", "")),
            )
        return self.complete(request.call_id, ok=True, op=request.op,
                             value=value)

    def complete(self, call_id: str, *, ok: bool, op: str = "",
                 value: Any = None, error: str = "") -> bool:
        """Commit a result by hand, for a resolver that is not a handler."""
        return self.commit(call_id, CallResult(
            call_id=call_id, op=op, ok=ok, value=value, error=error,
        ))
