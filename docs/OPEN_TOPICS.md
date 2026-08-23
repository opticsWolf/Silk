# Open Topics & Identified Gaps

A living inventory of known gaps and undecided design questions in Silk.
Every entry cites the code that establishes it, so each item can be
verified against the source in seconds. When an item is resolved, delete
it — the commit history is the archive.

**Last audited:** 2026-08-23.

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
fired (see the wiring table in [ARCHITECTURE.md](ARCHITECTURE.md#hooks-and-middleware)).
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
guards (depth / cycle / unknown worker).

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
