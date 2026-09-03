## Event streams

`functions/stream_events.py` defines the canonical event types the
`AgentLoop` yields — an `EventType` enum (`start`, `delta`, `tool_call`,
`tool_result`, `final_result`, `run_result`, `error`, `usage_limit`,
`reflection`) plus one dataclass per type, and `EventBuilder` / `EventStream`
helpers. This is the one shape every layer speaks, and the one the Qt layer
renders:

| Event | Carries |
|---|---|
| `EventStart` | `settings` (gen params), `input_tokens`, `context_length`, `system_prompt` — the last one is read off the engine and stays on the typed event: `to_wire` drops it as content and sends `system_prompt_chars` (§22 q3) |
| `EventDelta` | `delta`, `total_tokens`, `cumulative_text`, `tps` |
| `EventToolCall` | `tool_name`, `tool_args` (parsed dict), `call_id` |
| `EventToolResult` | `tool_name`, `result`, `call_id`, `error`, `error_message` |
| `EventFinalResult` | `text`, `tokens`, `input_tokens`, `tps`, `finish_reason` |
| `EventRunResult` | the final-result fields **plus** `tool_calls`, `tool_results` (the run trace) and `usage_stats` (`total_tokens`, `elapsed_s`, and the `UsageLimits` snapshot) |
| `EventError` | `error`, `context` (`stream_response` / `usage_limits` / `agent_loop` / `output_validation`), `recoverable` (set by the loop; no consumer reads it yet — see OPEN_TOPICS G13) |
| `EventUsageLimit` | `limit_type` (`request` — request-count or input-token cap — / `output_tokens` / `tool_calls`) |
| `EventReflection` | `retry_count`, `max_retries`, `error_type`, `error_message` |

`functions/event_format.py` is the human-facing side: `format_event(event)`
turns a tool-events dict into one log line (e.g. `▶ run started`,
`■ run finished — 4 round(s), 12.3s`, `· model round 2…`, `· response round
2 (1830 chars)`, tool call/result previews). Event dicts carry `event`
(kind), `ts`, `run_id`, and `seq` — a monotonic-per-run sequence that is the
dedup key for re-evaluations. `functions/plan_render.py` renders plan
markdown to HTML for the Plan Viewer via the *optional* `mordant` parser
(`highlighting_mode="Attribute"`, so Qt's limited HTML subset resolves the
styles); when `mordant` is absent it returns `None` and the caller falls
back to plain text — a missing optional dependency never breaks the graph.

### The durable sink (T7, D85)

`functions/event_sink.py` is the second consumer of that one stream. When
the Agent node's **Record run log** box is ticked, every wire event is
also appended to `~/.weave/silk/runs/<stamp>-<run>.jsonl`, one JSON
object per line. It exists because compaction (§12) drops turns
permanently: after a long run, this file is the only account of its
middle.

It redacts *more* than `to_wire` does, and deliberately: the wire form
keeps a delta's text and a tool call's arguments, which is right for a
widget showing the run it belongs to and wrong for a file that outlives
it. Every free-text field becomes a length (`delta` → `delta_chars`),
arguments become `tool_args_keys` + `tool_args_chars`. Off by default,
capped per run (a final `sink_truncated` line says so), pruned to the
newest 50 files, and disabled after one logged `OSError` — a log that
damages the run it records is worse than no log.
