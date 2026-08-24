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

### G14. Compaction is not implemented (required mechanism)

The declaration here is the requirement, not the code — unlike G1–G13
(machinery declared, implementation missing), compaction is not declared
anywhere in the codebase; it is entirely absent. The requirement is on
record: decision 2026-07-25 — long-running runs are a product goal, so
compaction **will be needed as a mechanism**. Both architecture reviews
converge on this gap independently (dsh review C.10: "the one operational
gap worth closing"; pi review D.4: failure ladder "quality decay → token
brake → backend `n_ctx` wall").

**Verified facts.** `GraphEngine.history` is a plain append-only list
(`graph_engine.py:52/93/134`); the `AgentEngine` protocol exposes
`append_message`, `count_prompt_tokens`, `stream_response`
(`protocols.py:19-56`) but **no** rewrite/drop/summarize operation; tool
results are fed back verbatim through the transport; the only
prompt-growth guard is the pre-request gate at `agent_loop.py:166`, which
*fails* the run rather than shrinking it — and per G13 even that failure
is swallowed by consumers. Nothing summarizes, prunes, or spills.

**What is missing (implementation checklist):**

| # | Piece | Note |
|---|---|---|
| (a) | Engine operation to replace the model-visible history prefix | The protocol is append-only today; the engine stays the owner of its own history (design rule: the engine is a single request) |
| (b) | Compactor seam in the loop | The pressure point at `agent_loop.py:166` today only aborts; it must be able to *shrink and retry* |
| (c) | Context-budget number at loop level | `GGUFMeta.context_length` exists at model load (`gguf_meta.py:42`) and `n_ctx` is passed to the server (`model_pool.py:92`), but nothing plumbs it to the engine or the loop — a pressure threshold needs a denominator |
| (d) | Event type for compaction | "Everything observable is an event"; content-free per the observability rule (metadata + summary reference, not prompt text) |
| (e) | Observability preconditions | G13 (outcome) so consumers can tell a compacted run from a clean one; T7 (sink) so the dropped range is debuggable — compaction is a lossy projection of the run |

**Reference designs (from the reviews).** dsh §11.2 — compaction triggers
on *pressure* (`agent/pre-step`, before request derivation) and on
*canonical overflow* (`agent/request-error`, after a failed model
request); a model-free `toolResultPruner` rewrites oversized tool results
before summary selection; the replacement **shadows the original nodes in
derived history** (the append-only log is never rewritten). pi §6.3 —
auto-trigger at `contextTokens > contextWindow − reserveTokens`;
append-only: find the cut point walking back past `keepRecentTokens`,
summarize the older range (passing previous summaries forward), append a
`CompactionEntry` whose inline `retainedTail` makes each compaction a
self-contained checkpoint; "compaction changes provider context, **not
storage**".

**Implementation options (A: spill hook / B: agent-invoked tool / C: loop
policy at the pressure seam) with trade-offs, Silk implementation shapes,
and the recommended sequencing: [T8](#t8-context-budget-under-raised-autonomy-compaction-is-a-required-mechanism-g14).**

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

### T8. Context budget under raised autonomy (compaction is a required mechanism — G14)

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
(the ugly brake).

**Decision (2026-07-25):** compaction is not just a mitigation — it is a
**required mechanism** for long-running runs (the absence is tracked as
[G14](#g14-compaction-is-not-implemented-required-mechanism)). Three
options, cheapest first, drawn from both architecture reviews (dsh §11.2
+ review C.10/D.10; pi §6.3 + review D.4):

**Option A — Spill hook (deterministic, model-free, tool results only).**
A `spill_large_results(max_chars, spill_dir)` entry in `hook_catalog`,
beside `redact_secrets` (pydantic `SpillConfig`, like the existing
`RedactSecretsConfig`): above threshold, write the full tool result to
`<spill_dir>/<call_id>.txt` and replace the model-visible content with a
head/tail preview + the file path (the model has file tools and can
re-read). Runs on the wired `HOOK_WRAP_TOOL_EXECUTE` middleware — no new
subsystem, zero extra model calls, ~60–100 lines, no new failure mode.
Provenance: dsh Layer 3 "spill-policy" / `toolResultPruner` (model-free,
replayable replacements); dsh review D.10 calls it "directly portable".
Limit: covers only the dominant growth term (verbatim tool results) —
model text and long dialogue still grow; spill files need a cleanup policy
(tie the directory to the run/plan root).

**Option B — Compaction as an agent-invoked tool (escape hatch).**
A tool (e.g. `compact_context(instruction)`) the model calls when it wants
a reset; on invocation a summarization request runs and the history window
is replaced with summary + kept tail. Shape in Silk: the tool needs engine
access (history is the engine's) — the plumbing is a sub-decision
(`RunContext` action vs. dedicated engine operation). Safety: a failed
summary request leaves history untouched and the tool returns an error;
the swap is atomic. Budget: the summarization call consumes tokens —
metered against `UsageLimits` or explicitly excluded (decision). Pi
review's position: demoted — "auto-compaction is loop policy (trigger on
pressure, not model whim); make the tool an escape hatch, not the
default." Niche; after C.

**Option C — Auto-compaction as loop policy at the pressure seam (the primary mechanism).**
At the existing `check_input_tokens` seam (`agent_loop.py:166`): when
estimated input tokens exceed a threshold, **compact and re-check** before
failing the run — summarize older turns, keep the recent K verbatim,
continue. This is pi's shape ("compaction is a loop policy"; auto-trigger
at `contextTokens > contextWindow − reserveTokens`, 16k reserve / 20k
keep-recent defaults, settings-configurable) and dsh's pressure trigger
(`agent/pre-step`, before request derivation). Implementation shape in
Silk:

- New optional `compactor` on `AgentLoop` (constructor argument, like the
  existing optional `output_validator`) — the loop keeps owning the turn,
  the engine keeps owning one request, the compactor owns the
  summarization request (one nested provider call; pi uses one or two).
- New `AgentEngine` operation to replace the history prefix (G14(a)) —
  built and swapped in **after** the summary succeeds (atomic).
- New event (e.g. `EventCompaction`: turns dropped, tokens before/after,
  summary reference) — content-free per the observability rule (G14(d)).
- A second trigger maps onto Silk's stream-error path: a backend `n_ctx`
  overflow arrives today as `EventError(context="stream_response")` (the
  G13 family) → compact once and retry (dsh's `agent/request-error`
  trigger).
- **Transport safety invariant:** the cut point must land on whole-round
  boundaries — an assistant turn and all its tool results move together;
  the native transport pairs `tool_calls` with `tool`-role results, and
  dropping one side of a pair corrupts the next request.
- A failed compaction degrades to no compaction (the existing
  `EventUsageLimit`/`EventError` path still protects); it never kills the
  run — "failures don't cross the loop boundary."
- **What compaction does not do:** rewrite the run's record. The
  `EventRunResult` trace + (once T7 lands) the JSONL sink keep the full
  run; compaction rewrites only the model-visible history — dsh's "the
  replacement shadows the original nodes in derived history," with the
  append-only log intact.
- Cache note (non-issue, recorded so it isn't re-derived): pi's
  tail-growth invariant makes compaction the "single deliberate cache
  invalidation"; Silk's pool does not depend on cross-request prompt
  caching, so no equivalent protection is needed.

**Sequencing (recommended):**

1. **A** — first, whenever any role or preset raises `max_rounds`
   (the P1 trigger already on record). No preconditions.
2. **C** — the required mechanism. Precondition chain: (a) context-budget
   plumbing (G14(c) — `GGUFMeta.context_length` exists at model load but
   never reaches the loop), (b) the G13 outcome field, (c) the T7 sink
   (recommended, for debuggability of the dropped range).
3. **B** — only after C, if the model itself needs to ask for a reset.

**Interim invariant until C lands** (A covers tool results only): whenever
`max_rounds` is raised for a role, also set `UsageLimits.input_tokens` —
for long autonomy the token cap, not the round cap, is the safety bound.

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
| Token metering + cache management | Unnecessary at stock bounds. (Compaction was on this list until 2026-07-25 — it is now a required mechanism, tracked as [G14](#g14-compaction-is-not-implemented-required-mechanism) / T8.) |
| Multi-package workspace machinery (sub-path exports, lockstep versions) | Organizational overhead for a monorepo Silk is not; the two-layer import rule is the same invariant at the right scale. |
