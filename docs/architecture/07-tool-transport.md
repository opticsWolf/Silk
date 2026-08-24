## Tool transport

`functions/tool_transport.py` — models disagree on how to express a tool
call. Silk supports both and picks per run:

- **`FenceTransport`** — the model emits a fenced JSON block
  (```tool_call ... ```), parsed from the streamed text by
  `functions/tool_calling.py` (`parse_tool_calls` / `has_tool_call`;
  `tool_result_content` defines the canonical result shape and
  `tool_call_instructions` builds the tool-protocol section of the system
  prompt). Works with any instruct model, any backend.
- **`NativeTransport`** — the backend returns structured tool calls
  (e.g. llama.cpp's native tool calling). No text parsing; exact types.

`select_transport(engine, toolbox)` chooses based on the handle/backend
capabilities and **degrades safely** — if native isn't available it falls
back to fenced text rather than failing. The loop calls it once per run,
before the first request, so native schemas are advertised to the model up
front.

The transport also owns the *format* of the round-trip, which is what keeps
the loop backend-agnostic:

- `extract_calls(engine, full_text)` → the `tool_calls` list (parsed from
  fences, or the engine's structured calls);
- `append_tool_result(engine, name, call_id, body)` → writes each result
  back into the engine history in the transport's native format;
- `append_retry_nudge(engine, text)` → delivers the reflection nudge in a
  template-safe role.

The `AgentLoop` only consumes `tool_calls` and results either way.

