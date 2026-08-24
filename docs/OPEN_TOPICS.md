# Open Topics & Identified Gaps

A living inventory of known gaps and undecided design questions in Silk.
Every entry cites the code that establishes it, so each item can be
verified against the source in seconds. When an item is resolved, delete
it — the commit history is the archive.

**Last audited:** 2026-08-24.

Legend:

- **GAP** — the machinery exists (or is declared) but the implementation
  is missing or partial.
- **TOPIC** — works as-is, but a design decision is pending.

## Identified gaps

### G1. The `requires_approval` gate is a no-op

`ToolBox._safe_execute` contains the entire gate — it is a `pass`:

```python
# Check requires_approval
if meta.get("requires_approval"):
    # TODO: Check approval status (stored in session or context)
    # For now, skip approval check if not implemented
    pass
```

(`functions/tool_box.py`.) Everything *around* the gate is built — the
`ToolMeta.requires_approval` field, the `approval_required(...)` ToolSet
operation, the registration `meta` key — but a tool registered with
`requires_approval=True` executes unchecked. There is no approval-state
store, no grant surface, and no structured denial result. See **T1** for
the open design question.

### G2. The BM25 tool-search strategy is a keyword alias

`ToolSearch._bm25_search` is documented as "a placeholder for now" and
delegates to `_keyword_search` (`functions/tool_search.py`,
`# TODO: Implement BM25 search`). Selecting the `bm25` strategy today
returns exactly the same results as `keywords`.

### G3. 11 of the 19 hook events are defined but never emitted

`functions/hooks.py` declares 19 event constants; only 8 are actually
fired (see the wiring table in [the tool system section](architecture/08-tool-system.md#hooks-and-middleware)).
Never emitted: `HOOK_AFTER_MODEL_REQUEST`, `HOOK_WRAP_MODEL_REQUEST`,
`HOOK_ON_MODEL_REQUEST_ERROR`, `HOOK_WRAP_TOOL_VALIDATE`,
`HOOK_ON_TOOL_VALIDATE_ERROR`, `HOOK_ON_TOOL_EXECUTE_ERROR`,
`HOOK_WRAP_OUTPUT_VALIDATE`, `HOOK_ON_OUTPUT_VALIDATE_ERROR`,
`HOOK_WRAP_OUTPUT_PROCESS`, `HOOK_ON_OUTPUT_PROCESS_ERROR`,
`HOOK_WRAP_RUN_EVENT_STREAM`.

Two consequences:

- The `*_ERROR` family means hooks cannot react to model-request failures
  or to validation/execution failures — those only appear as stream
  events (`EventError`).
- Role/capability hook maps accept any event name, so a hook registered on
  a reserved event registers cleanly and then *silently never fires*.

Related: the loop calls `output_validator.validate_with_reflection(...)`
directly (no `HOOK_WRAP_OUTPUT_VALIDATE` in between) and `_parse_args`
calls `model_validate_json` directly (no `HOOK_WRAP_TOOL_VALIDATE`), so
middleware currently cannot observe or intervene in either phase.

### G4. No test suite

There are no tests in the repository (no `tests/`, no `test_*.py` files).
The runtime is explicitly headless-testable (design rule: no Qt in
`functions/`), so nothing blocks adding them. Highest-value first targets:
the `AgentLoop` generator contract (rounds, reflection, usage limits,
`HOOK_AFTER_RUN` exactly-once), the ToolBox execution path (role gate,
structured errors, timeouts, sequential-vs-parallel), `SqliteTaskStore`
concurrency (revision conflicts → `Conflict`), and the orchestrator
guards (depth / cycle / unknown worker). When it lands, encode the five
invariants as executable fixture data (one record per invariant and
violation class) rather than free-standing test functions, so the doc and
the suite cannot drift apart (pi review D.5).

### G5. Runtime dependencies are declared nowhere

`pyproject.toml` has no `[project]` table: the repo is designed to run in
place as a Weave submodule, so runtime dependencies (PySide6,
`llama-cpp-python[server]`, optional `mordant`) are neither declared nor
version-pinned anywhere. A standalone clone cannot `pip install` Silk, and
there is no declared floor for the `llama_cpp` server API the pool's HTTP
client depends on. (There is a runtime probe:
`server_missing_deps_message()` tells you what to install if the server
extra is missing.)

### G6. The model pool has no recovery when the server dies

`GGUFModelPool` spawns one `llama_cpp.server` subprocess and waits for it
to become ready at startup (raising with a log tail on failure). After
that there is no liveness check: if the server process dies mid-run, the
in-flight request fails (the loop turns it into an `EventError`) and
nothing restarts the pool. `cleanup()` exists; there is no supervisor.

### G7. `EventUsageLimit` cannot tell which cap fired

The request-count and input-token gates share one `try`, and a breach of
either yields `EventUsageLimit(limit_type="request")` followed by an
`EventError` (`functions/agent_loop.py`). The only distinguishing signal is
the error message text. (Related: `count_prompt_tokens()` is a
best-effort estimate, so the input-token gate itself is approximate.)

### G8. Stops are not honoured mid-tool-batch

`stop_requested()` is checked between rounds and at token boundaries
inside the engine. A tool batch already in flight runs to completion;
there is no per-call cancellation (the `asyncio.wait_for` wrappers are
timeout-only). For long-running tools, the registration `timeout` is the
only bound.

### G9. Type coverage is scoped to `functions/`

mypy is configured with `files = ["functions"]` — deliberate staged
adoption per the `pyproject.toml` comment ("widen `files` as more modules
gain types"). `nodes/` and `widgets/` are untyped. Tracked here so the
intentional gap doesn't become an accidental one.

### G10. `EventStart.system_prompt` is never set

The field exists on `EventStart`, but the loop constructs the event with
only `settings` and `input_tokens` — it is always `None`. Either populate
it (useful for a viewer that shows the model its instructions) or drop
the field.

### G11. `OpenAIClientMock` is the production client

The name says "Mock", but `GGUFModelPool` instantiates it as its live
HTTP client (`functions/model_pool.py`). A rename (e.g.
`OpenAICompatClient`) would stop it being confused with a test double.

### G12. No version metadata

The package has no `__version__` (or equivalent), so a running graph
cannot report which Silk commit it is running — only the submodule pin in
the Weave checkout can. Trivial to add; useful for logs and bug reports.

### G13. The `max_rounds` error is silently dropped by every consumer

When the round budget is exhausted, the loop yields
`EventError(context="agent_loop", recoverable=True)`
(`functions/agent_loop.py`, the `for/else` on the rounds loop) **and then
still yields `EventFinalResult` + `EventRunResult`** with the last round's
text. Both consumers guard with `if run_error and not final_text:` before
surfacing an error (`nodes/agent.py`, `functions/subagent.py`), so such a
run reports a normal finish (status "Done." / `SubagentResult(ok=True)`)
and the error event vanishes. Related: `recoverable` is declared and set,
but no consumer reads it — today it is a dead field, and `agent_loop` is
the only context whose error arrives together with a final result.
Candidate fixes: on the consumer side, key the guard off the error context
/ `recoverable` instead of the presence of a final result; on the loop
side, make `max_rounds` a true terminal exit with no `EventRunResult`; or
the pi-harness review's option (D.1), which fixes all three conflated
exits at once — add `outcome: completed | stopped | usage_limited | error`
to `EventRunResult`, set at the loop's exit classes (one field, four
assignments; it also makes a user-stopped run distinguishable from a
finished one).

## Open topics

### T1. Design of the approval gate (closes G1)

Where should approval state live — per-run `RunContext` (simple, but
approvals don't survive a re-run) or a session store (durable, but needs a
home and a lifecycle)? And what is the grant surface: a
`HOOK_WRAP_TOOL_EXECUTE` middleware that parks the call until a UI
responds, a pre-execution check, or an extension of sign-off? Note that
**sign-off is already a working approval mechanism for task changes**
(park → human approves → held-and-applied action). The question is whether
generic tool approval should reuse that plumbing instead of a parallel
one. The in-code TODO suggests "session or context", which is the
unresolved part.

### T2. Hook vocabulary: wire it up or prune it (closes G3)

Either implement emission for the 11 reserved events — starting with the
`*_ERROR` family and `HOOK_AFTER_MODEL_REQUEST`, which are cheap and
useful for logging/metrics — or delete the unused constants so the
vocabulary matches reality. Pruning is safe as long as no hook map
references a constant (the bundled catalog hooks only use wired events).

### T3. Multi-agent budgeting

A fan-out can share one `UsageLimits`, but there is no per-worker
sub-budget: one greedy worker can exhaust the shared budget and every
other worker in the fan-out starts getting `USAGE_LIMIT` events. Decide
whether global-only is the intended semantics, or add nested budgets.

### T4. Plan discovery policy (task store)

The store picks the *newest* `plan-*.db` by mtime across `root` and
`root/.silk/plan`, and both plan nodes (`Plan Viewer`, `Sign-Off`) take a
plain `root` path — there is no plan-id input. One plan per root works
fine; multiple concurrent plans in one root can be cross-discovered.
Options: a plan-id node input, or a directory-per-plan convention. (The
schema already keys rows on `(plan_id, …)`, so the ambiguity is in
discovery, not storage.)

### T5. Default delegation depth

The orchestrator runtime treats `max_depth=None` as `1`, while the
`Silk Orchestrator` node ships `DELEGATION_MAX_DEPTH = 2`. Two defaults
for the same concept — pick one, or make the node's value an editable
port.

### T6. HTML rendering floor

`plan_render` degrades to `None` (→ plain text in the Plan Viewer) when
`mordant` is missing. Decide the minimum rendering guarantee: plain text
always, or `mordant` as a soft requirement with a visible notice when the
styled path is unavailable.

### T7. Durable event sink (JSONL per run)

The event dicts already carry `event` / `ts` / `run_id` / `seq`
([Event streams](architecture/15-event-streams.md#event-streams)).
Writing them as JSONL per run gives debug replay at a small fraction of a
session-substrate cost; the dsh and pi reviews each recommend it
independently. Decision on record: build only when a real debugging need
appears, not speculatively — retrofitting persistence *shapes* into a
running system is the expensive direction. If it lands, it must honour the
content-free observability rule
([18 — Design rules](architecture/18-design-rules.md#design-rules)):
metadata only, never prompts / completions / tool payloads.

### T8. Context budget under raised autonomy (spill first, compaction conditionally)

`DEFAULT_MAX_ROUNDS = 16` is only the constructor default
(`functions/agent_loop.py:76`); `Role.max_rounds` overrides it per agent
(`nodes/agent.py`: `role.max_rounds or DEFAULT_MAX_ROUNDS`), so
`max_rounds=100` is legal today. `GraphEngine.history` grows monotonically
(append-only, never pruned — `functions/graph_engine.py`), and growth is
dominated by verbatim tool results. Failure ladder for long runs: quality
decay (invisible — no event fires) → the `UsageLimits.input_tokens`
controlled stop, checked pre-request every round (`agent_loop.py:166` —
the graceful brake) → the backend `n_ctx` wall: a hard request error
(`EventError(context="stream_response")`) or silent middle-truncation
(the ugly brake). Mitigation ladder, cheapest first:

1. **P2 — rises to P1 once any role or preset raises `max_rounds`:** a
   `spill_large_results(max_chars, spill_dir)` entry in `hook_catalog`,
   beside `redact_secrets`: above threshold, write the full tool result to
   `<spill_dir>/<call_id>.txt` and replace the model-visible content with
   a head/tail preview + the file path (the model has file tools).
   Deterministic, zero extra model calls, ~60–100 lines on existing
   middleware.
2. **Consider, niche:** an agent-invoked compaction capability
   (e.g. `summarize_and_drop_history(keep_recent_n)`) — only if models are
   observed drowning in-context within the round budget before hitting
   caps; any rewrite must preserve the active transport's round-trip
   format.
3. **Not rational today:** auto-compaction at the `check_input_tokens`
   pressure seam — rational only with a long-run product target, and only
   after the outcome field (G13) and the event sink (T7) land.

Interim invariant: whenever `max_rounds` is raised, set
`UsageLimits.input_tokens` too — for long autonomy the token cap, not the
round cap, is the safety bound.

## Deliberately not planned

Machinery a much larger harness (pi — ~149k lines of TypeScript) needs but
Silk (~11k lines of Python, atomic runs over a graph) declines, with the
reason on record (pi-harness review, D.6). Revisit only if the stated
trigger changes; the list exists so the question isn't re-derived from
scratch.

| Machinery | Why not |
|---|---|
| Durable session runtime (write-once entry tree, mutable registers, usage ledger, crash-position recovery) | Silk runs are atomic and graph-pulsed; a dead run is re-pulsed. The product shape excludes the problem. |
| Mid-run steering / follow-up queues | Atomic runs + the sign-off park express the same interactivity at run boundaries; no inbox mechanism needed. |
| Multiple interception generations (callbacks → events → durable hooks) | One audience (graph authors), one surface. Revisit only if third-party Python extension packs become a real demand. |
| Lanes / continuable subagents | Need a session substrate; one-shot delegation with depth/cycle guards and a shared budget covers the current fan-out (T3 aside). |
| Compaction + token metering + cache management | Unnecessary at stock bounds; the spill hook (T8) covers the concrete risk. |
| Multi-package workspace machinery (sub-path exports, lockstep versions) | Organizational overhead for a monorepo Silk is not; the two-layer import rule is the same invariant at the right scale. |
