# -*- coding: utf-8 -*-
"""AgentLoop — the single, Qt-free autonomous agent run loop.

This is the only place multi-turn agent behaviour lives: stream one model
response, parse ``tool_call`` fences, dispatch them through the ToolBox
(where the active role hard-enforces its toolset), feed results back, apply
reflection retries and usage limits, and finish with an optional
output-schema validation pass.

The loop is a plain generator yielding typed events from ``stream_events``
— the single canonical event shape. Consumers decide what to do with them:
the Weave Agent node forwards them to Qt signals / ``emit_stream``; tests
collect them into lists; a CLI could print them. Nothing here imports
PySide6.

Chaining agent networks: the loop itself knows nothing about the graph.
The Agent node maps ``EventRunResult`` to its ``done`` Exec pulse, which is
how downstream agent nodes are edge-triggered.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from typing import Any, Optional

from . import tool_calling
from .hooks import (
    HOOK_AFTER_MODEL_REQUEST,
    HOOK_AFTER_MODEL_RESPONSE,
    HOOK_AFTER_RUN,
    HOOK_BEFORE_MODEL_REQUEST,
    HOOK_BEFORE_RUN,
    HOOK_ON_MODEL_REQUEST_ERROR,
    HOOK_ON_OUTPUT_PROCESS_ERROR,
    HOOK_ON_OUTPUT_VALIDATE_ERROR,
)
from .model_errors import classify_model_error
from .protocols import AgentEngine, ToolRegistry
from .reflection import is_retryable_tool_error, parse_tool_error
from .stream_events import (
    OUTCOME_COMPLETED,
    OUTCOME_ERROR,
    OUTCOME_STOPPED,
    EventDelta,
    EventError,
    EventFinalResult,
    EventReflection,
    EventRunResult,
    EventStart,
    EventToolCall,
    EventToolResult,
    EventUsageLimit,
)
from .tool_transport import FenceTransport, ToolTransport, select_transport
from .usage_limits import UsageLimitExceeded

#: Hard ceiling on model requests per run — replaces an unbounded loop so a
#: model that keeps emitting tool calls cannot spin forever.
DEFAULT_MAX_ROUNDS = 16

AgentEvent = Any  # union of the Event* dataclasses in stream_events


class AgentLoop:
    """The autonomous run loop over an engine + tool registry.

    Args:
        engine: Anything satisfying the :class:`AgentEngine` protocol.
        toolbox: Tool registry (or ``None`` for pure chat — tool fences are
            then treated as final output).
        output_validator: Optional object with
            ``validate_with_reflection(text, max_retries=...)`` returning
            ``(is_valid, result, retries)`` (see ``output_schema.py``).
        max_rounds: Model requests allowed per run.
    """

    def __init__(
        self,
        engine: AgentEngine,
        toolbox: Optional[ToolRegistry] = None,
        output_validator: Any = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> None:
        self.engine = engine
        self.toolbox = toolbox
        self.output_validator = output_validator
        self.max_rounds = max_rounds
        # Run bookkeeping surfaced to the after_run hook (set defensively
        # here: the finally in run() may fire before the first round).
        self._final_text = ""
        self._rounds_used = 0
        # Chosen per run() by select_transport; fence is the safe default so
        # a direct _run_rounds call (tests) never hits an unset transport.
        self._transport: ToolTransport = FenceTransport()

    def stop(self) -> None:
        """Request a graceful stop; the engine breaks at the next token."""
        self.engine.request_stop()

    def context_length(self) -> Optional[int]:
        """The engine's context window, or ``None`` if it does not say.

        Optional on the AgentEngine protocol, because an engine over a
        backend that never reports one is still a usable engine — the
        budget is then simply unknown, and callers must treat it as such
        rather than substituting a guess.
        """
        getter = getattr(self.engine, "context_length", None)
        if getter is None:
            return None
        try:
            value = getter()
        except Exception:      # a backend probe must not fail a run
            return None
        return int(value) if value else None

    def _emit(self, event: str, **kwargs: Any) -> None:
        """Emit a lifecycle hook via the toolbox's registry, if present.

        Run/model-level events surface here (the ToolBox itself emits the
        tool-level trio at dispatch). No toolbox → pure chat, no hooks.
        """
        hooks = getattr(self.toolbox, "hooks", None)
        if hooks is not None:
            hooks.emit(event, **kwargs)

    # -- the loop ----------------------------------------------------------

    def run(
        self,
        user_input: Optional[str],
        gen_params: Optional[dict[str, Any]] = None,
    ) -> Iterator[AgentEvent]:
        """Run one agent turn; yields typed events, ending in EventRunResult."""
        gen_params = gen_params or {}
        engine = self.engine

        # Native structured tool calls when the model supports them, else the
        # universal ```tool_call fence protocol. Selected before the first
        # request so native schemas are advertised to the model up front.
        self._transport = select_transport(engine, self.toolbox)

        if user_input:
            engine.append_message("user", user_input)

        tool_calls_made: list[dict[str, Any]] = []
        tool_results_made: list[dict[str, Any]] = []
        start_time = time.time()

        # after_run is emitted via finally so every exit path — normal
        # completion, usage limit, stream error, early generator close —
        # reports run end exactly once (invariant I2).
        #
        # EventStart and before_run are *inside* the try for that last case:
        # a consumer that takes the first event and walks away closes the
        # generator at that first yield, and with the try opening any later
        # the close would land outside it — before_run fired, after_run
        # never did. Found by the I2 fixture, which is what those fixtures
        # are for.
        try:
            yield EventStart(
                settings=dict(gen_params),
                input_tokens=engine.count_prompt_tokens(),
                context_length=self.context_length(),
            )
            self._emit(
                HOOK_BEFORE_RUN, user_input=user_input, settings=dict(gen_params),
            )
            yield from self._run_rounds(
                gen_params, tool_calls_made, tool_results_made, start_time,
            )
        finally:
            self._emit(
                HOOK_AFTER_RUN,
                final_text=self._final_text,
                rounds=self._rounds_used,
                elapsed_s=time.time() - start_time,
            )

    def _model_failed(
        self, error: Any, round_index: int, *, truncated: bool = False,
    ) -> EventError:
        """One place where a model-request failure becomes an event.

        The classification rides along as ``kind`` so a consumer can tell an
        overflow (which compaction may answer) from a dead server (which it
        must not) without re-parsing the message (spec D40), and the same
        verdict goes to the hook so a logger sees what the loop saw.
        """
        verdict = classify_model_error(error, truncated=truncated)
        self._emit(
            HOOK_ON_MODEL_REQUEST_ERROR,
            error=verdict.message,
            kind=verdict.kind,
            round_index=round_index,
        )
        return EventError(
            error=verdict.message,
            context="stream_response",
            recoverable=verdict.is_retryable,
            kind=verdict.kind,
        )

    def _run_rounds(
        self,
        gen_params: dict[str, Any],
        tool_calls_made: list[dict[str, Any]],
        tool_results_made: list[dict[str, Any]],
        start_time: float,
    ) -> Iterator[AgentEvent]:
        engine = self.engine
        full_text = ""
        total_tokens = 0
        retry_count = 0
        self._final_text = ""
        self._rounds_used = 0

        outcome = OUTCOME_COMPLETED

        for _round in range(self.max_rounds):
            # 1. Usage gates before each request.
            try:
                engine.usage_limits.check_request()
                engine.usage_limits.check_input_tokens(engine.count_prompt_tokens())
            except UsageLimitExceeded as exc:
                yield EventUsageLimit(limit_type="request")
                yield EventError(error=str(exc), context="usage_limits", recoverable=False)
                return

            # 2. Stream exactly one model response.
            self._rounds_used = _round + 1
            self._emit(HOOK_BEFORE_MODEL_REQUEST, round_index=_round)
            full_text = ""
            try:
                for delta in engine.stream_response(gen_params):
                    full_text += delta
                    total_tokens += 1
                    elapsed = time.time() - start_time
                    yield EventDelta(
                        delta=delta,
                        total_tokens=total_tokens,
                        cumulative_text=full_text,
                        tps=(total_tokens / elapsed) if elapsed > 0 else 0.0,
                    )
            except UsageLimitExceeded as exc:
                yield EventUsageLimit(limit_type="output_tokens")
                yield EventError(error=str(exc), context="usage_limits", recoverable=False)
                return
            except Exception as exc:
                yield self._model_failed(exc, _round)
                return

            stats = engine.last_stats or {}
            # The request is over either way: after_model_request reports
            # that a request happened, after_model_response that a usable
            # answer came back. Distinct events because a failed request is
            # still a request -- it cost time, and on a metered backend,
            # money (D15).
            self._emit(
                HOOK_AFTER_MODEL_REQUEST,
                round_index=_round,
                ok=not stats.get("error") and not stats.get("truncated"),
                finish_reason=stats.get("finish_reason"),
            )
            if stats.get("error"):
                yield self._model_failed(stats["error"], _round)
                return
            if stats.get("truncated"):
                # The request succeeded and the answer was cut off anyway:
                # a shared llama_cpp.server interrupts an in-flight stream
                # when a second agent asks (spec D43), and the [DONE] it
                # sends is indistinguishable from a clean finish except by
                # the missing finish_reason. Reasoning over half an
                # assistant turn is worse than failing the round.
                yield self._model_failed(
                    "The response stream ended without a finish reason — the "
                    "answer was truncated. If two agents share one llama.cpp "
                    "server, start it with interrupt_requests disabled.",
                    _round,
                    truncated=True,
                )
                return

            self._final_text = full_text
            self._emit(
                HOOK_AFTER_MODEL_RESPONSE,
                text=full_text,
                round_index=_round,
                finish_reason=stats.get("finish_reason"),
            )

            # 3. Persist the assistant turn (tool fences stay inside it).
            engine.append_message(
                "assistant",
                full_text,
                input_tokens=stats.get("input_tokens"),
                output_tokens=stats.get("tokens"),
                tps=stats.get("tps"),
                finish_reason=stats.get("finish_reason"),
            )

            # 4. Tool round? Calls come from the active transport — parsed
            #    from ```tool_call fences, or the engine's structured
            #    tool_calls when the model supports native tool calling.
            calls = self._transport.extract_calls(engine, full_text)
            if engine.stop_requested():
                outcome = OUTCOME_STOPPED
                break
            if not calls or self.toolbox is None:
                break

            try:
                # Claim the whole batch before dispatching any of it: the
                # budget may be shared with sibling workers (spec D52.4).
                engine.usage_limits.reserve_tool_calls(len(calls))
            except UsageLimitExceeded as exc:
                yield EventUsageLimit(limit_type="tool_calls")
                yield EventError(error=str(exc), context="usage_limits", recoverable=False)
                return

            for call in calls:
                yield EventToolCall(
                    tool_name=call.function.name,
                    tool_args=_args_as_dict(call.function.arguments),
                    call_id=call.id,
                )

            try:
                results = asyncio.run(self.toolbox.execute_tool_calls_async(calls))
            except Exception as exc:
                results = [{
                    "tool_call_id": c.id,
                    "name": c.function.name,
                    "content": tool_calling.tool_result_content(
                        c.function.name, f"Error: {exc}"),
                } for c in calls]

            # Feed EVERY result back in the normal slot — the model sees the
            # actual tool output (or error), the UI gets an EventToolResult,
            # and the run records the call, even when it errored and we also
            # nudge a retry below. Retryable errors accumulate a note; we emit
            # at most ONE reflection nudge per round (not per call) so a
            # fan-out round can neither spam the transcript nor burn the whole
            # retry budget at once.
            retry_notes: list[str] = []

            for res in results:
                name = res.get("name", "tool")
                body = res.get("content", "")
                call_id = res.get("tool_call_id", "")
                is_error, error_msg = parse_tool_error(body)

                yield EventToolResult(
                    tool_name=name,
                    result=body,
                    call_id=call_id,
                    error=is_error,
                    error_message=error_msg if is_error else None,
                )
                self._transport.append_tool_result(engine, name, call_id, body)
                tool_calls_made.append({"name": name})
                tool_results_made.append({"name": name, "result": body})

                if is_error and is_retryable_tool_error(body):
                    note = error_msg or "tool error"
                    schema = _correct_schema(body)
                    if schema:
                        note += f"\nCorrect schema:\n{schema}"
                    retry_notes.append(f"{name}: {note}")

            # One consolidated reflection nudge per erroring round, gated by
            # the run-wide retry budget so we still give up eventually. The
            # transport delivers it in a template-safe role.
            if retry_notes and retry_count < engine.reflection_config.max_retries:
                yield EventReflection(
                    retry_count=retry_count,
                    max_retries=engine.reflection_config.max_retries,
                    error_type="tool_validation",
                    error_message="; ".join(retry_notes),
                )
                self._transport.append_retry_nudge(
                    engine,
                    f"{engine.reflection_config.tool_error_prompt}\n\n"
                    + "\n".join(retry_notes),
                )
                retry_count += 1

            # Loop back so the model can read the tool output(s) / retry.
            continue
        else:
            # The loop ran out of rounds with the model still calling tools.
            # There *is* a final text -- the last assistant turn -- which is
            # why this used to read as a clean finish downstream (G13). The
            # outcome is what says otherwise.
            outcome = OUTCOME_ERROR
            yield EventError(
                error=f"max_rounds ({self.max_rounds}) reached without a final answer.",
                context="agent_loop",
                recoverable=True,
            )

        # 5. Optional structured-output validation.
        if self.output_validator is not None:
            is_valid, result, _retries = self.output_validator.validate_with_reflection(
                full_text,
                max_retries=engine.reflection_config.max_output_retries,
            )
            if not is_valid:
                self._emit(HOOK_ON_OUTPUT_VALIDATE_ERROR, error=str(result))
                yield EventError(
                    error=f"Output validation failed: {result}",
                    context="output_validation",
                    recoverable=False,
                )
                return
            if hasattr(result, "model_dump_json"):
                try:
                    full_text = result.model_dump_json()
                except Exception as exc:
                    # Serialising a validated object is the "output process"
                    # step; it can still fail (a field that does not
                    # round-trip), and the run keeps the validated text
                    # rather than dying over the rendering of it.
                    self._emit(HOOK_ON_OUTPUT_PROCESS_ERROR, error=str(exc))
                    yield EventError(
                        error=f"Output processing failed: {exc}",
                        context="output_processing",
                        recoverable=True,
                    )

        stats = engine.last_stats or {}
        elapsed = time.time() - start_time
        self._final_text = full_text  # includes output-validation rewrites
        yield EventFinalResult(
            text=full_text,
            tokens=int(stats.get("tokens", total_tokens) or 0),
            input_tokens=int(stats.get("input_tokens", 0) or 0),
            tps=float(stats.get("tps", 0.0) or 0.0),
            finish_reason=str(stats.get("finish_reason", "stop")),
        )
        yield EventRunResult(
            text=full_text,
            tokens=int(stats.get("tokens", total_tokens) or 0),
            input_tokens=int(stats.get("input_tokens", 0) or 0),
            tps=float(stats.get("tps", 0.0) or 0.0),
            finish_reason=str(stats.get("finish_reason", "stop")),
            tool_calls=tool_calls_made,
            tool_results=tool_results_made,
            usage_stats={
                "total_tokens": total_tokens,
                "elapsed_s": elapsed,
                **engine.usage_limits.snapshot(),
            },
            outcome=outcome,
        )


def _args_as_dict(arguments: Any) -> dict[str, Any]:
    """Tool-call arguments arrive as a JSON string; events carry dicts."""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (json.JSONDecodeError, TypeError):
        return {"raw": str(arguments)}


def _correct_schema(body: str) -> Optional[str]:
    """Pull a ``correct_schema`` out of a structured tool-error payload.

    Returns the schema as indented JSON so the reflection nudge can show the
    model exactly which fields/types it got wrong, or ``None`` when the body
    is not a structured error (e.g. a plain-text ``"Error: …"`` result).
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict) and "correct_schema" in data:
        return json.dumps(data["correct_schema"], indent=2)
    return None
