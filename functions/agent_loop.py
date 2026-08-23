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
    HOOK_AFTER_MODEL_RESPONSE,
    HOOK_AFTER_RUN,
    HOOK_BEFORE_MODEL_REQUEST,
    HOOK_BEFORE_RUN,
)
from .protocols import AgentEngine, ToolRegistry
from .reflection import is_retryable_tool_error, parse_tool_error
from .stream_events import (
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

        yield EventStart(
            settings=dict(gen_params),
            input_tokens=engine.count_prompt_tokens(),
        )
        self._emit(HOOK_BEFORE_RUN, user_input=user_input, settings=dict(gen_params))

        tool_calls_made: list[dict[str, Any]] = []
        tool_results_made: list[dict[str, Any]] = []
        start_time = time.time()

        # after_run is emitted via finally so every exit path — normal
        # completion, usage limit, stream error, early generator close —
        # reports run end exactly once.
        try:
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
                yield EventError(error=str(exc), context="stream_response", recoverable=False)
                return

            stats = engine.last_stats or {}
            if stats.get("error"):
                yield EventError(error=str(stats["error"]), context="stream_response",
                                 recoverable=False)
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
            if not calls or engine.stop_requested() or self.toolbox is None:
                break

            try:
                engine.usage_limits.check_tool_calls(len(calls))
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
            engine.usage_limits.record_tool_calls(len(results))

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
                yield EventError(
                    error=f"Output validation failed: {result}",
                    context="output_validation",
                    recoverable=False,
                )
                return
            if hasattr(result, "model_dump_json"):
                full_text = result.model_dump_json()

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
