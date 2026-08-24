## The agent loop

`functions/agent_loop.py` — `AgentLoop` is the single autonomous
multi-turn runtime. It is a generator, not a coroutine: `run(...)` yields a
stream of typed events (see [Event streams](15-event-streams.md#event-streams)) and you consume
it at your own pace.

```python
loop = AgentLoop(
    engine,                 # an AgentEngine (e.g. GraphEngine)
    toolbox=None,           # a ToolRegistry (a ToolBox); None → pure chat
    output_validator=None,  # object with validate_with_reflection(text, max_retries=...)
    max_rounds=16,          # DEFAULT_MAX_ROUNDS hard ceiling
)
for event in loop.run(user_input, gen_params=None):
    ...
```

Note what is *not* a constructor argument: the system prompt, usage limits,
and reflection config all live on the **engine** (the `AgentEngine` protocol
exposes `usage_limits`, `reflection_config`, and `history`), and hooks are
read from the **toolbox** (`toolbox.hooks`). The loop itself carries only
the round ceiling and an optional output validator.

One **round** is: one model request → extract any tool calls → dispatch them
→ feed the results back. The verified flow of `run(...)`:

1. `select_transport(engine, toolbox)` — chosen **before the first request**
   so native tool schemas are advertised up front (see
   [Tool transport](07-tool-transport.md#tool-transport)).
2. Append the user message (if any), yield `EventStart(settings,
   input_tokens)`, emit `HOOK_BEFORE_RUN`.
3. `_run_rounds(...)`, wrapped so `HOOK_AFTER_RUN` fires **exactly once on
   every exit path** (normal completion, usage limit, stream error, early
   generator close) via `finally`, carrying `final_text`, `rounds`, and
   `elapsed_s`.

   Each round (up to `max_rounds`):

   1. **Usage gates before the request**: `usage_limits.check_request()` and
      `check_input_tokens(...)`; a breach yields `EventUsageLimit` +
      `EventError(recoverable=False)` and ends the run.
   2. Emit `HOOK_BEFORE_MODEL_REQUEST`; stream **exactly one** response from
      `engine.stream_response(gen_params)`, yielding an `EventDelta`
      (`delta`, `total_tokens`, `cumulative_text`, `tps`) per chunk. A
      mid-stream `UsageLimitExceeded` or any exception ends the run with an
      `EventError`.
   3. Emit `HOOK_AFTER_MODEL_RESPONSE` (with `finish_reason`); persist the
      assistant turn via `engine.append_message("assistant", ...)` with the
      request's token stats — tool fences stay inside that message.
   4. `calls = transport.extract_calls(engine, full_text)` — from fenced
      `tool_call` blocks, or the engine's structured `tool_calls` when the
      model supports native tool calling.
   5. **No calls** (or a stop was requested, or `toolbox is None` — pure
      chat) → the model produced a final answer; break out of the rounds.
   6. **Calls present**: `usage_limits.check_tool_calls(len(calls))`, yield
      one `EventToolCall` per call, then
      `results = asyncio.run(toolbox.execute_tool_calls_async(calls))` —
      the loop is a *synchronous* generator, so each tool batch runs its own
      asyncio event loop. A batch-level exception becomes per-call error
      results, never a raised error.
   7. Per result: yield `EventToolResult` (including structured error
      payloads — the model sees its own failures), and
      `transport.append_tool_result(engine, ...)` writes the result into the
      engine history in the transport's own format.
   8. **Reflection**: if any result is a *retryable* tool error, the loop
      collects notes (pulling the error's `correct_schema` out of the
      structured payload, if present) and — within
      `reflection_config.max_retries` — emits **one consolidated**
      `EventReflection` per round (not one per failing call) and appends the
      nudge via `transport.append_retry_nudge(...)`. The loop then continues
      so the model can read the outputs / retry.

4. If the rounds are exhausted without a final answer →
   `EventError(context="agent_loop", recoverable=True)`.
5. **Output validation**: if `output_validator` is set,
   `validate_with_reflection(full_text, max_retries=reflection_config.max_output_retries)`
   runs; a failure ends the run with an `EventError`, a pass may rewrite
   `full_text` (e.g. the validated model dumped back to JSON).
6. Yield `EventFinalResult` then `EventRunResult` (text, token counts,
   `tps`, `finish_reason`, the `tool_calls`/`tool_results` run trace, and a
   `usage_stats` snapshot).

**The loop never executes tools itself** — it only dispatches batches to the
`ToolRegistry` and interprets the results. That is the whole division of
labour: the loop owns *turns*, the engine owns *one request*, the ToolBox
owns *one tool batch*.

`DEFAULT_MAX_ROUNDS = 16` is a hard ceiling on autonomy; `usage_limits`
(request / input-token / output-token / tool-call caps) are the second,
independent brake. `loop.stop()` requests a graceful stop, honoured at the
next token boundary via the engine.

### Engine-side config

Because the loop consults the engine's `usage_limits` and
`reflection_config`, those knobs are set where the engine is built (the
`Silk Agent` node / `GraphEngine` construction), not on the loop. The
`ReflectionConfig` fields the loop uses: `max_retries` (tool-error nudges),
`max_output_retries` (output validation), and `tool_error_prompt` (the nudge
text).

