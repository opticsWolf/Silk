# Open Topics & Identified Gaps

A living inventory of known gaps and undecided design questions in Silk.
Every entry cites the code that establishes it, so each item can be
verified against the source in seconds.

**Relationship to the design spec.** Design *decisions* now live in
[DESIGN_SPEC_DRAFT.md](DESIGN_SPEC_DRAFT.md). This file tracks what is not
yet **built** and what is still **undecided**. An entry whose design question
the spec has answered carries a **Decided** line citing its D-number and
stays only until the code lands.

**Last audited:** 2026-08-30 — reworked against the design spec.

Legend:

- **GAP** — the machinery exists (or is declared) but the implementation
  is missing or partial.
- **TOPIC** — works as-is, but a design decision is pending.
- **Decided** — the design question is settled in the spec; only the
  implementation is outstanding.

The standing rule is *delete an item when it is resolved; the commit history
is the archive*. Three resolved topics (T1, T4, T8) are kept as one-line
stubs instead, because the spec and other entries link to their anchors.

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

(`functions/tool_box.py:697`.) Everything *around* the gate is built — the
`ToolMeta.requires_approval` field, the `approval_required(...)` ToolSet
operation, the registration `meta` key — but a tool registered with
`requires_approval=True` executes unchecked.

**Decided:** spec §7 (D30–D35), closing **T1**. The in-code TODO's premise
("stored in session or context") is rejected: there is **no** approval-state
store. The gate emits a request on the run's stream, blocks the call, and
resolves it inside the same live run (D30). Sign-off and tool approval become
one hook (D31); the parked-state machinery is deleted, not migrated (D31–D33).
The only persistence is durable per-tool grants at `~/.weave/silk/grants.json`,
keyed by resolved project root, allow-only (D34, D35).

**Remaining (implementation, spec Phase 2):** the decision transport
(request id + thread-safe wait resolved from the UI thread); cancellation
reaching a blocked waiter (see **G8**); timeout and default-deny, since a
blocked gate holds both the worker thread and the exclusive `RoleBinding`;
the no-answerer case (nothing wired to the events port, or a subagent) —
still open. Also still open: the grant record schema and the revocation
surface.

### G2. The BM25 tool-search strategy is a keyword alias

`ToolSearch._bm25_search` is documented as "a placeholder for now" and
delegates to `_keyword_search` (`functions/tool_search.py`,
`# TODO: Implement BM25 search`). Selecting the `bm25` strategy today
returns exactly the same results as `keywords`.

**Re-scoped by spec §6.** This was cosmetic while nothing model-facing
called the search index. Under D4 (`search_tools`) discovery becomes the
primary path into context, so ranking quality is load-bearing. The decision
is binary and still open: implement BM25, or drop the strategy from the
public surface. The spec parks it under "Later", which is only tenable if
discovery ships with `keywords` and the `bm25` option is hidden until real.

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

**Decided (partly):** spec §8 (D15) — wire the five `*_ERROR` events plus
`HOOK_AFTER_MODEL_REQUEST`, and make registration on a still-unwired event
fail loudly instead of registering silently. The five `WRAP_*` events are
explicitly **not** pruned pending review; their disposition is **T2**.

### G4. No test suite

There are no tests in the repository (no `tests/`, no `test_*.py` files).
The runtime is explicitly headless-testable (design rule: no Qt in
`functions/`), so nothing blocks adding them. Highest-value first targets:
the `AgentLoop` generator contract (rounds, reflection, usage limits,
`HOOK_AFTER_RUN` exactly-once), the ToolBox execution path (role gate,
structured errors, timeouts, sequential-vs-parallel), `SqliteTaskStore`
concurrency (revision conflicts → `Conflict`), and the orchestrator
guards (depth / cycle / unknown worker).

**Decided:** spec §14 (D27) and Phase 1 item 1 — encode the five existing
invariants plus the spec's I6–I9 as executable fixture data, one record per
invariant and violation class, **before** the implementation lands. This is
the main risk control for a change of the spec's size. Note the spec does
*not* ask for characterization fixtures on the sign-off park path: that path
is deleted (D31), not refactored, so pinning it would be wasted work.

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

**Sharpened by D30.** The inline approval gate introduces a *deliberate*
block inside a tool batch, of unbounded duration, waiting on a human. The
consumer loop that polls `is_compute_cancelled()` (`nodes/agent.py:477`) is
inside a single `next()` call while that happens and cannot run — so Stop
must reach the blocked waiter directly, or Stop will not stop a run sitting
on an approval prompt. What was a latency annoyance becomes a correctness
requirement of the gate.

### G9. Type coverage is scoped to `functions/`

mypy is configured with `files = ["functions"]` — deliberate staged
adoption per the `pyproject.toml` comment ("widen `files` as more modules
gain types"). `nodes/` and `widgets/` are untyped. Tracked here so the
intentional gap doesn't become an accidental one.

### G10. `EventStart.system_prompt` is never set

The field exists on `EventStart`, but the loop constructs the event with
only `settings` and `input_tokens` — it is always `None`. Either populate
it (useful for a viewer that shows the model its instructions) or drop
the field. Still open; the spec carries it as an implementation-time call
(§5, open question 3), since it is a one-line change either way.

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
but no consumer reads it — today it is a dead field.

**Decided:** spec §5 (D2) and Phase 1 item 3 — add
`outcome: completed | stopped | usage_limited | error` to `EventRunResult`,
set at the loop's exit classes, and have consumers key off `outcome` rather
than off the presence of a final text. One field, four assignments; it also
makes a user-stopped run distinguishable from a finished one, and it is a
precondition for telling a compacted run from a clean one (G14(e)).

### G14. Compaction is not implemented (required mechanism)

The declaration here is the requirement, not the code — unlike G1–G13
(machinery declared, implementation missing), compaction is not declared
anywhere in the codebase; it is entirely absent. The requirement is on
record: decision 2026-07-25 — long-running runs are a product goal, so
compaction **will be needed as a mechanism**.

**Verified facts.** `GraphEngine.history` is a plain append-only list
(`graph_engine.py:52/93/134`); the `AgentEngine` protocol exposes
`append_message`, `count_prompt_tokens`, `stream_response`
(`protocols.py:19-56`) but **no** rewrite/drop/summarize operation; tool
results are fed back verbatim through the transport; the only
prompt-growth guard is the pre-request gate at `agent_loop.py:166`, which
*fails* the run rather than shrinking it — and per G13 even that failure
is swallowed by consumers. Nothing summarizes, prunes, or spills.

**Decided:** spec §12 (D24, D25) — **T8 option C in full, with option A
alongside**. Pressure trigger at the `agent_loop.py:166` seam against a
`GGUFMeta.context_length` denominator, plus a compact-once-and-retry on a
backend `n_ctx` overflow arriving as `EventError(context="stream_response")`;
the agent's own model and pool session does the summarizing (no second
resident model). Cuts land on whole-round boundaries (spec I9). A failed
compaction degrades to no compaction and never kills a run.

**What is missing (implementation checklist):**

| # | Piece | Status |
|---|---|---|
| (a) | Engine operation to replace the model-visible history prefix | spec §12; built and swapped in only after the summary succeeds (atomic) |
| (b) | Compactor seam in the loop | optional `compactor` on `AgentLoop`, like the existing optional `output_validator` |
| (c) | Context-budget number at loop level | spec Phase 1 item 4 — plumb `GGUFMeta.context_length` (`gguf_meta.py:42`) to the loop |
| (d) | Event type for compaction | `EventCompaction`, content-free per the observability rule (spec §5) |
| (e) | Observability preconditions | G13 lands in Phase 1. **T7 (durable sink) is still undecided** — the spec recommends it for debugging the dropped range but does not schedule it |

**Interim invariant until compaction lands.** Whenever `max_rounds` is
raised for a role, also set `UsageLimits.input_tokens`: for long autonomy the
token cap, not the round cap, is the safety bound. The spill hook (option A)
covers only the dominant growth term — verbatim tool results — and needs a
cleanup policy tied to the run/plan root, which is also still open.

## Open topics

### T1. Design of the approval gate (closes G1)

**Resolved** by spec §7, D30–D35: one inline blocking gate hook, no
approval-state store, durable grants only. Stub kept for inbound links; the
implementation gap is **G1**.

### T2. Hook vocabulary: wire it up or prune it (closes G3)

Narrowed by D15. The six event-family members are decided (**wire**). What
remains open is the disposition of the five middleware events —
`HOOK_WRAP_MODEL_REQUEST`, `HOOK_WRAP_TOOL_VALIDATE`,
`HOOK_WRAP_OUTPUT_VALIDATE`, `HOOK_WRAP_OUTPUT_PROCESS`,
`HOOK_WRAP_RUN_EVENT_STREAM` — one decision per event, against the review
table in spec §8. Pruning is safe as long as no hook map references the
constant (the bundled catalog hooks only use wired events). Note that
`HOOK_WRAP_TOOL_EXECUTE`, the one middleware event that *is* wired, is what
both the sign-off gate and the future approval gate hang off — so the class
is proven useful, and "prune the rest" is not the obvious default.

### T3. Multi-agent budgeting

A fan-out can share one `UsageLimits`, but there is no per-worker
sub-budget: one greedy worker can exhaust the shared budget and every
other worker in the fan-out starts getting `USAGE_LIMIT` events.

**Decided (semantics only):** spec §13 (D26) — a global cap plus optional
per-worker sub-budgets, stated now so the compaction and approval work do
not have to guess, implemented after the core surface. The gap is that
nothing is built; the design question is closed.

### T4. Plan discovery policy (task store)

**Resolved** by spec §11, D23: a `Task Node` carries explicit plan identity
(plan id + store location) and feeds the ToolBox, so the Plan Viewer takes
that identity instead of guessing. Stub kept for inbound links. Until it
ships, the store still picks the *newest* `plan-*.db` by mtime across `root`
and `root/.silk/plan`, so concurrent plans in one root can cross-discover.
(The `Sign-Off` node named in the original entry is deleted by D32; the Plan
Viewer is the only remaining plan consumer.)

### T5. Default delegation depth

The orchestrator runtime treats `max_depth=None` as `1`, while the
`Silk Orchestrator` node ships `DELEGATION_MAX_DEPTH = 2`. Two defaults
for the same concept — pick one, or make the node's value an editable
port. Untouched by the spec.

### T6. HTML rendering floor

`plan_render` degrades to `None` (→ plain text in the Plan Viewer) when
`mordant` is missing. Decide the minimum rendering guarantee: plain text
always, or `mordant` as a soft requirement with a visible notice when the
styled path is unavailable. Untouched by the spec.

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

**Still open, and now load-bearing.** The spec schedules compaction in full
(§12) and lists T7 under G14(e) as a recommended precondition, because
compaction is a lossy projection of the run and the dropped range is
otherwise unrecoverable. The call — in or out of the spec's Phase 2 — has
not been made. Two arguments that did not exist when this entry was written:
the unified event vocabulary (D2) means a sink writes one typed stream
rather than three ad-hoc ones, which is the cheap moment to add it; and D30
puts a human decision inside the run, which is exactly the kind of thing an
audit trail should retain.

### T8. Context budget under raised autonomy (compaction — G14)

**Resolved** by spec §12: option C (loop-policy auto-compaction) in full,
with option A (the spill hook) alongside; option B (an agent-invoked
`compact_context` tool) stays deferred as an escape hatch, only if the model
itself needs to ask for a reset. Stub kept for inbound links; the
implementation checklist and the interim `UsageLimits.input_tokens`
invariant live in **G14**.

## Deliberately not planned

Machinery a much larger harness (pi — ~149k lines of TypeScript) needs but
Silk (~11k lines of Python, atomic runs over a graph) declines, with the
reason on record (pi-harness review, D.6). Revisit only if the stated
trigger changes; the list exists so the question isn't re-derived from
scratch.

| Machinery | Why not |
|---|---|
| Durable session runtime (write-once entry tree, mutable registers, usage ledger, crash-position recovery) | Silk runs are atomic and graph-pulsed; a dead run is re-pulsed. The product shape excludes the problem. D30 makes a run *block* on a human without making it resumable: a run that dies while waiting loses the prompt and is re-pulsed like any other. |
| Mid-run steering / follow-up queues | Runs stay atomic. D30 does put a human decision inside a run, but an approval gate is not a steering channel: it answers one yes/no about one specific call and accepts no new instructions. Revisit only if users need to redirect a run in flight. |
| Multiple interception generations (callbacks → events → durable hooks) | One audience (graph authors), one surface. Revisit only if third-party Python extension packs become a real demand. |
| Lanes / continuable subagents | Need a session substrate; one-shot delegation with depth/cycle guards and a shared budget covers the current fan-out (T3 aside). Note the spec leaves one subagent question open: who answers an approval prompt raised inside a subagent (spec §7). |
| Token metering + cache management | Unnecessary at stock bounds. (Compaction was on this list until 2026-07-25; it is now a required mechanism, specified in spec §12 and tracked as [G14](#g14-compaction-is-not-implemented-required-mechanism).) |
| Multi-package workspace machinery (sub-path exports, lockstep versions) | Organizational overhead for a monorepo Silk is not; the two-layer import rule is the same invariant at the right scale. |
