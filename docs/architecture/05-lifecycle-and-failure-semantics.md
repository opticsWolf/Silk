## Lifecycle and failure semantics

A **run** is the unit of execution: one `AgentLoop.run(...)` consumed from
`EventStart` to its end. **Rounds** are a run's internal iterations (one
model request → tool calls → results, above).

**A run is atomic.** New user input is admitted only *between* runs: the
`Silk Agent` node consumes its `inbox` when the run starts and emits
`response` / `outbox` when it ends (and a `chat.turn` event on the
`events` stream). There is no mid-run
steering or injection — the only in-run influences are `loop.stop()` (below)
and the sign-off park, where the agent *ends its own run* so control returns
to the user ([Task system and sign-off](11-task-system-signoff.md#task-system-and-sign-off)). This is
deliberate: interactivity at run boundaries matches the sign-off pause and
keeps the loop free of an inbox mechanism.

**Stop semantics.** `loop.stop()` is cooperative, honoured at the next token
boundary or between rounds. A tool batch already in flight runs to
completion before the stop is observed — there is no per-call cancellation
(OPEN_TOPICS G8).

**How a run ends.** Today the stream carries no single "outcome" field; the
terminal event(s) tell you how a run ended:

| Exit | Terminal event(s) |
|---|---|
| Normal — final answer, output validation passes (or none set) | `EventFinalResult`, `EventRunResult` |
| Stop requested | after a round's model response the round loop breaks and the run ends *as if it had produced a final answer*: `EventFinalResult` + `EventRunResult` carry the text accumulated so far — currently indistinguishable from a normal finish |
| Output validation fails (after `max_output_retries`) | `EventError(context="output_validation", recoverable=False)` — no `EventRunResult` |
| Usage limit breach (request / input-token / output-token / tool-call) | `EventUsageLimit(limit_type=…)`, `EventError(context="usage_limits", recoverable=False)` — a controlled stop, not a fault |
| Model stream error (exception or `stats["error"]`) | `EventError(context="stream_response", recoverable=False)` |
| `max_rounds` exhausted (every round produced tool calls) | `EventError(context="agent_loop", recoverable=True)` *mid-stream* — the run then **still** ends with `EventFinalResult` + `EventRunResult` carrying the last round's text, so on the wire it looks like a normal finish |

**How consumers read that.** Both consumers of the loop record the last
`EventError` and surface it only when no `EventRunResult` arrived
(`run_error and not final_text` — `nodes/agent.py`, `functions/subagent.py`).
That makes the `max_rounds` row the one `EventError` that is *not* surfaced:
it arrives together with a final result, so both consumers report a normal
finish (OPEN_TOPICS G13).

**Failure semantics.** Every failure below is model- or stream-visible; none
crosses the loop boundary as an exception:

| Failure | What the model / stream sees | Retryable? | Who recovers |
|---|---|---|---|
| Unknown tool name | `EventToolResult` with error content (`"Tool 'x' is not registered."`) | — | the model: it sees the error and can retry with a registered tool |
| Role denies a tool | `EventToolResult` with `error_type="role_denied"` + suggestion; `HOOK_TOOL_DENIED` fires | **no** — reflection is told the denial is final | the model: it picks a permitted tool |
| Argument validation error | `EventToolResult` carrying `correct_schema` | yes | reflection nudge (≤ `max_retries`, at most one per round) + model self-correction |
| Tool exception / timeout | `EventToolResult` with structured error | yes | reflection nudge; the model retries or works around |
| Batch-level failure (the executor itself raised) | per-call error results for the whole batch | yes | same as above |
| Usage limit | `EventUsageLimit` + `EventError` | no | the user: lower a cap, or re-run |
| Model stream error | `EventError(context="stream_response")` | no | the user: re-run |
| `max_rounds` exhausted | `EventError(context="agent_loop", recoverable=True)` mid-stream, then a normal-looking `EventRunResult` with the last round's text | flagged recoverable | nobody today — both consumers drop it (OPEN_TOPICS G13); more autonomy is a config change |
| Output validation failure | `EventError(context="output_validation")` | no | the user: re-run |

**Deliberately out of scope** (relative to larger harness designs): no
durable session log / resume / forking — the graph is the persistence
boundary and a run's trace is its `EventRunResult`; no compaction **yet** —
it is a required mechanism (OPEN_TOPICS G14, options and sequencing in
T8); until it lands, autonomy is bounded by `max_rounds` and `UsageLimits`;
one-shot subagents only — no durable or continuable children (there is no
durable session to resume them into). A fuller statement of that line —
durable runtime, steering queues, lanes, each with the reason it was
declined — sits in [OPEN_TOPICS, "Deliberately not
planned"](../OPEN_TOPICS.md#deliberately-not-planned) (from the
pi-harness review).

