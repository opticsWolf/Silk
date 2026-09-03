## Model layer

### `functions/graph_engine.py` — `GraphEngine`

The production `AgentEngine`. It adapts a Weave `gguf_model` handle to the
protocol:

- Owns the conversation history (a list of role/content dicts) and appends
  a turn per `append_message(...)`.
- `stream_response(gen_params)` performs **exactly one** model request,
  yielding incremental text deltas. It checks out a model from the pool (if
  the handle is pool-backed) and checks it back in on completion.
- Tracks `last_stats` (token counts, request metadata) and
  `count_prompt_tokens()` (best-effort input-token count) — these feed
  `UsageLimits`.
- Captures native structured tool calls if the backend exposes them (see
  transport), so the loop doesn't have to regex the text.
- `request_stop()` / `stop_requested()` cooperate with an in-flight
  generation: the stop is honoured at the next token boundary.

### `functions/model_pool.py`

A server-based model pool so several agents can share loaded GGUF models:

- Spawns **one** background `llama_cpp.server` process and talks to it over
  its OpenAI-compatible HTTP/SSE API — no in-process `Llama` object per
  agent.
- Generates a JSON server config on the fly (paths, context, etc.).
- `OpenAIClientMock` mimics the `llama_cpp.Llama` API on top of the HTTP
  client, so code written against the in-process API works unchanged against
  a pooled model.
- A pool-backed `gguf_model` handle carries `"pool": pool` instead of
  `"model": Llama`; the engine checks a model out/in around a request.

### `functions/gguf_meta.py`

A small binary GGUF probe. `GGUFMeta(context_length, block_count)` reads just
the two integer values the loader UI needs (to clamp its spinboxes). It
implements the GGUF v1 vs v2+ length-encoding difference and *skips*
non-integer KV values by seeking — the probe only cares about integers but
must advance the stream past everything else.

### `functions/prefix_guard.py` — invariant I11

Between two requests in one run, the earlier request's message sequence
must be a **prefix** of the later one. `GraphEngine` carries a
`PrefixGuard` and observes every request through it.

The rule is invisible at the call site: nothing in `build_messages` breaks
if a section starts rendering the time of day, if a tool schema is added
mid-run, or if a hook edits an already-sent message in place. What breaks
is the backend's KV cache — llama.cpp keeps the longest common prefix
between the new prompt and the last one it evaluated and re-evaluates only
the suffix, so a change at position *k* costs a re-prefill of everything
after *k*. The failure is silent, it costs in proportion to how far back
it happened, and it looks exactly like the model being slow (D41).

The guard therefore **reports and never repairs**: a break is a bug in
whatever built the messages, and an automatic fix would leave the cost in
place and remove the signal. It names three kinds, because they have three
causes — `system` (volatile prompt content; the most expensive, since it
invalidates everything), `tools` (the advertised schemas changed, which
deferred capability loading does by design), and `history` (an
already-sent message rewritten or dropped, the one kind that is never
intentional). Compaction declares itself with `note_compaction()` and is
forgiven exactly once: forever would make the guard go quiet for the rest
of any run that compacts.

### Session accounting on the pool

`GGUFModelPool` is a shared client, not a slot allocator: one
`llama_cpp.server` serves every agent. What it still has to track is which
*conversations* are bound to it, because the Pool Monitor reports the
number and `Clear Context` needs to drop one.

That is a **set of session ids**, not a counter. `checkout` runs once per
request, so counting checkouts measured requests-ever and only ever grew
— the monitor showed that growth as bound conversations. Both operations
are idempotent: a conversation is bound or it is not, so neither a second
request nor a second `Clear Context` can move the number anywhere it
should not go. `release_session(session_id)` is the public way to unbind,
and it reports whether it knew the session.

**The queue is counted, not hidden (§22 q1c).** One server serves one
request at a time (D43), so an orchestrator fan-out of eight workers is
correct *and* sequential — which looks exactly like a hang. Scheduling
stays the pool's business; legibility does not. `begin_request` /
`end_request` now bracket a flight count, `snapshot()` carries
`serialization` (`in_flight`, `peak_in_flight`, `queued_requests`,
`serialising`), and the first request that has to wait logs one line
saying so is correct. The Pool Monitor appends it to the flags line only
after something has actually queued — a line that is always there is one
nobody reads. This is the same answer D53 and §22 q1d gave: say it once,
then count.
