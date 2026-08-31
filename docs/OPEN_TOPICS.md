# Open Topics & Identified Gaps

A living inventory of known gaps and undecided design questions in Silk.
Every entry cites the code that establishes it, so each item can be
verified against the source in seconds.

**Relationship to the design spec.** Design *decisions* now live in
[DESIGN_SPEC_DRAFT.md](DESIGN_SPEC_DRAFT.md). This file tracks what is not
yet **built** and what is still **undecided**. An entry whose design question
the spec has answered carries a **Decided** line citing its D-number and
stays only until the code lands.

**Last audited:** 2026-08-30 — reworked against the design spec, then
re-audited against the dsh and pi harness reviews in `docs/references/`.

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

**Remaining (implementation, spec Phase 2):** one decision seam carrying
correlation, cancel-before-wake ordering, timeout, and a policy snapshot
(D38); the gate forced outermost as a monotonic guard, since
`attach_catalog_hooks` currently registers catalog middleware *before*
`attach_signoff_gate` and `emit_middleware` runs first-registered-outermost
(D37, spec I10); `PRAGMA user_version` on the plan store (D39); and the
race catalog that makes any of it testable (D42). The no-answerer question is
**closed** — D36 makes every missing-answer path (no consumer wired,
transport raises, timeout, gated subagent) deny. Still open: the grant record
schema and the revocation surface.

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

**New option (2026-08-31):** spec §17 brings `macrame-db` in reach, whose
`keyword_search` (FTS5) and `hybrid_search` (FTS5 + DiskANN, RRF) are real
ranked search. The binary choice gains a third arm: delegate tool search to
the ledger where present, keeping `keywords` as the no-ledger fallback.

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

**Now forced (2026-08-31):** spec §17 adopts `macrame-db` as an optional
extra — Silk's first declared *binary* dependency (abi3 wheels on PyPI).
D66 makes declaring dependencies a precondition of the ledger work, so G5
stops being deferrable the moment that lands.

### G6. The model pool has no recovery when the server dies

`GGUFModelPool` spawns one `llama_cpp.server` subprocess and waits for it
to become ready at startup (raising with a log tail on failure). After
that there is no liveness check: if the server process dies mid-run, the
in-flight request fails (the loop turns it into an `EventError`) and
nothing restarts the pool. `cleanup()` exists; there is no supervisor.

**Escalated by spec D24.** Compaction's second trigger reacts to
`EventError(context="stream_response")`, which is also what a dead server
produces — so without classification, a crash would be answered by spending a
summarization request against the corpse and retrying. Spec D40 makes a
model-request error classifier (overflow vs retryable vs terminal, the shape
pi uses) a **precondition of D24**. Silk has the tool-side half already
(`is_retryable_tool_error` in `functions/reflection.py`) and nothing for model
requests. The classifier is also what a future supervisor would key off; the
restart itself stays out of scope.

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
requirement of the gate. D38 adds the ordering rule that makes it decidable:
the cancellation reason is recorded *before* the waiter is woken, so Stop,
the timeout and a real approval never arrive as one indistinguishable
wakeup. D49 names the object that owns this (`DecisionSeam`, run-scoped) and
generalises the ordering to all four wake causes: write the outcome under the
lock, then set the event. Stop calls `cancel()` on it **directly** — routing
Stop through the consumer loop cannot work, because that loop is not running
while the gate blocks.

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

It is also the whole client surface — `create_chat_completion`, `tokenize`,
`reset` — built from a bare `base_url`, which makes it the natural adapter
for a remote OpenAI-compatible backend (litellm, vLLM, hosted). Two things
block that today: it sends only `Content-Type` (`model_pool.py:128`), so
**no API key can be passed**, and `tokenize` is a `len // 4` approximation
(`:171`) that a remote backend has no way to improve. Decided: spec **D45**;
the key follows D22 (a credential *name*, resolved at connect time).

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
| (f) | Prefill accounting | `EventCompaction` reports the KV-prefill cost, not only tokens dropped — otherwise the most expensive thing compaction does is invisible (**G15**, spec D41) |
| (g) | Error classification | compact only on classified overflow, never on a generic stream error (spec D40, **G6**) |

**Option A is not merely the cheap first step — it is the cache-safe one.**
Spill rewrites a tool result *before* it is appended, so history stays
append-only and the KV prefix survives. Summarization rewrites the head and
destroys it (**G15**). That reorders the two: spill should carry as much load
as it can before compaction is triggered at all.

**Interim invariant until compaction lands.** Whenever `max_rounds` is
raised for a role, also set `UsageLimits.input_tokens`: for long autonomy the
token cap, not the round cap, is the safety bound. The spill hook covers only
the dominant growth term — verbatim tool results — and needs a cleanup policy
tied to the run/plan root, which is also still open.

### G15. Prompt-prefix reuse is unconfigured and unmeasured

The machinery exists in the runtime and Silk neither configures nor observes
it. Verified against the installed llama-cpp-python 0.3.34:

- `Llama.generate` (`llama_cpp/llama.py`) scans for the longest common prefix
  between the incoming tokens and `self._input_ids`, calls
  `kv_cache_seq_rm(-1, reuse_prefix, -1)`, and evaluates **only the suffix**.
  So prefix reuse is real and automatic.
- But there is **one resident context per instance**, not one per
  conversation: `_input_ids` holds whichever prompt was evaluated last. Reuse
  therefore survives only *consecutive* requests from the same conversation.
- `LlamaCache` — the prefix-keyed multi-state store that would survive
  interleaving — is opt-in via the server's `cache` / `cache_type` /
  `cache_size` settings (`llama_cpp/server/settings.py:143`), and
  `_SERVER_MODEL_KEYS` (`functions/model_pool.py:92`) does not forward them.
  Server defaults apply, so it is off.

**Two consequences, one live today and one incoming.**

*Live:* Silk shares one server across every agent by design
(`functions/model_pool.py`: "a shared client, not a slot"). Two Agent nodes
alternating — or an orchestrator fan-out via `delegate_parallel` — clobber
each other's resident context, so each request re-prefills everything past
the shared system prompt. A, B, A, B reuses almost nothing; A, A, B, B reuses
almost everything. **Parallel delegation is considerably more expensive than
it looks**, and nothing reports this.

*Incoming:* compaction (G14) rewrites the head of the context, collapsing the
common prefix to roughly the system prompt — so the whole surviving context
is re-prefilled, and by construction that context is near the ceiling. Since
spec D25 has the agent's own model produce the summary, the nested
summarization request clobbers `_input_ids` on its way through: **a
compaction costs two full prefills, not one**. On a hosted API that is a
billing line; on a local GPU it is dead wall-clock mid-run.

**Decided:** spec §12 (D41) and invariant I11 — the model-visible prefix
grows only at the tail except at a deliberate compaction. Measurement is
**Phase 1** and nearly free: `verbose` is already forwarded in
`_SERVER_MODEL_KEYS`, and `generate()` prints
`"<n> prefix-match hit, remaining <m> prompt tokens to eval"`. Capture it
across a multi-round run and a fan-out before tuning anything.

**The obvious fix is not free.** Forwarding `cache` / `cache_type` /
`cache_size` is a three-name addition to `_SERVER_MODEL_KEYS`, and
`llama_cpp/server/model.py:334` does the rest. But every completion then ends
with `self.cache[prompt + completion] = self.save_state()`
(`llama.py:1700`), and `save_state` (`:2199`) memcpies the full serialized
context blob plus a copy of the scores array. Against a default `cache_size`
of `2 << 30` (2 GiB) and a large `n_ctx`, the LRU can evict on nearly every
insert — paying the copy and keeping nothing. Enabling it is a measured
trade, not a correction. Decided: spec **D44**.

**The identity needed to fix this already exists.** Each Agent node carries
a persistent `session_id` (`nodes/agent.py:105`), sub-agents get fresh ones
(`subagent.py:177`), and GraphEngine passes it on every request
(`graph_engine.py:260`). `GGUFModelPool.checkout()` increments a counter and
throws it away (`model_pool.py:308`). Three mechanisms hang off that one
seam — affinity (group the queue by session), `LlamaCache` (keep more than
one resident state), multiple backends (one resident context each) — and
they address only the *interleaving* half of the problem; the *rewriting*
half is I11 and the spill hook.

**Decided:** spec **D46** (honour the session id) and **D47**, which sets out
how the three compose and gives a measurement-driven rule for choosing
between them. **"Do nothing" is an explicitly reachable outcome** of that
rule — if prefill turns out to be a small share of request time, none of the
three pays for itself, and the correct response is to stop.

**Still open:** the sub-questions the rule does not settle — `LlamaCache`
size and backing if B is selected; who chooses the backend, and the
down-backend path, if C is.

### G21. Write authority over an importable directory is process authority

Every file tool is sandboxed; `import` is not. Module-level code in a file the
agent wrote runs with the **full authority of the Weave process** -- network,
whole filesystem, the user's keys -- however narrow the sandbox was when the
file was written. So a sandbox root that is on the import path is a
*deferred* grant of process authority, redeemable with one `load_suite` call.

Spec **D77** is the mitigation, not a fix: always-approve with a floor no
preset can lower, the diff shown at approval time, and validation moved into
a subprocess *after* approval. Two residues stay open:

- A user who registers a sandbox root that happens to be importable (inside
  the venv, inside `weave/`, anywhere on `sys.path`) has granted more than the
  file-permissions UI suggests. Cheap check, worth doing: **warn at ToolBox
  evaluation when a writable root is importable** -- the same place a
  whitelisted-but-unregistered node class is reported (D71).
- MCP servers with their own file access (§10) sit outside D77's gate
  entirely: they can write wherever their process can, and Silk neither
  sandboxes nor sees it. Already true today; §19 makes it consequential,
  because now something in the process is willing to import what appears on
  disk.

### G19. Toolchain subprocess writes bypass the file locks across agents

The per-path write locks (`functions/tools/file_locks.py`) are process-wide
and deliberately module-global, so every mutating *file tool* is already
serialised across agents -- that part is solved and now spec-level (D67
tier 1). But subprocess tools that rewrite files (`ruff format`,
`cargo fmt`, every `run_*`) never acquire `lock_paths`, and their
`sequential=True` flag only orders execution within one agent's batch
(`functions/tool_box.py:669`). Two agents sharing a tree can interleave a
formatter with an `edit_file` and neither is told. Silent, and concurrent
tool execution across agents is real today (tool batches run on independent
`ThreadedNode` workers even while model calls serialise).

Fix specified at spec **D67 tier 2**: a per-root readers-writer gate, a
`writes_files` flag on `CommandSpec`, root-gates-before-path-locks ordering.
Scope ruling at **D68** (advisory, per-process; long-term ownership via
ledger claims, not locks). Phase 1 item 12.

### G18. `delegate_parallel` is unsound with more than one assignment

The orchestrator's shape is right — delegation is a normal tool, so it
inherits hooks, role enforcement and `tool_events` for free — but the
fan-out path has four independent defects, and every one of them is silent.
Full treatment in spec §15 (**D52**); in brief:

1. **Same-worker fan-out fails.** `_run_one` writes depth and chain onto
   `spec.toolset` (`functions/orchestrator.py:231-234`), a shared live graph
   object; two assignments naming one worker race, then `RoleBinding.activate`
   refuses the second (`subagent.py:165`). *Run `researcher` on A and on B* is
   a reasonable request that returns `ok=False`.
2. **Depth and chain leak** — set on the child toolset, never cleared, no
   `finally`. Run-scoped state on a graph-scoped object.
3. **The assignment list is truncated silently.**
   `items = [...][:_MAX_PARALLEL]` with `_MAX_PARALLEL = 8`
   (`orchestrator.py:78`, `:356`): twelve assignments become eight and the
   reply says `ok=True`, "8/8 delegations succeeded".
4. **The shared budget races** — see the T3 correction.

And it is the worst case for **G16**: N workers interleaving against one
server that truncates in-flight streams, each with a fresh `session_id`
(`subagent.py:177`) so prefix reuse is zero (**G15**). Spec **D53** rules
that the fan-out runs sequentially until G16 is fixed — which costs nothing,
because `llama_outer_lock` was serializing the requests anyway.

Separately, `_run_one` discards the `on_event` and `should_stop` parameters
`run_subagent` already accepts (`subagent.py:125-126`), so a fan-out is both
invisible while it runs and **uninterruptible** — G8's most severe instance.
Spec **D54**.

### G17. `Clear Context` silently fails to release the pool session

`nodes/agent.py:258` reads `pool._session_instances`, an attribute of the
**old multi-`Llama` pool**; `GGUFModelPool` has no such attribute. The access
is wrapped in `except Exception: log.debug(...)`, so the failure is invisible
and the session is never checked in — `_active_sessions` only ever grows, and
`snapshot()["bound_sessions"]` (the loader's display) drifts upward for the
life of the process.

Low severity on its own — nothing downstream gates on the count today — but
it becomes load-bearing the moment `checkout()` starts using the session id
for affinity or backend routing (**G15**, spec D46/D47), because then a stale
session is a stale route. Fix it with that work.

### G16. The shared server truncates in-flight streams

Not a performance gap: a correctness one, and it is live.

`llama_cpp/server/app.py` serializes every request through
`llama_outer_lock` / `llama_inner_lock`, so concurrency never reaches the
model — `delegate_parallel` queues. On top of that, the streaming publisher
checks per chunk:

```python
if interrupt_requests and llama_outer_lock.locked():
    await inner_send_chan.send(dict(data="[DONE]"))
    raise anyio.get_cancelled_exc_class()()
```

`ServerSettings.interrupt_requests` **defaults to `True`**
(`llama_cpp/server/settings.py:223`), and Silk's generated config sets only
`host`, `port` and `models` (`model_pool.py:217`) — so the default applies.
**Agent B starting a request truncates agent A's response mid-stream.** A
receives a well-formed `[DONE]`; `OpenAIClientMock.generator()` (`:153`)
breaks on it exactly as it would on a natural stop, so the loop cannot tell
a cut-off turn from a finished one and reasons on against the fragment.

**Decided:** spec **D43** — forward `interrupt_requests: false`, and treat a
stream ending without a terminal `finish_reason` as
`EventError(context="stream_response")`, which is the classification spec D40
already requires. **Phase 1**, ahead of the Phase 2 safety work: until it
lands, no concurrent multi-agent graph is sound.

Note this corrects a prior "non-issue" note carried in T8, which held that
Silk's pool does not depend on cross-request prompt caching. That was
inherited from a hosted-API framing where a cache miss costs money. Locally
it costs latency, and the code shows the dependency is real.

### G20. Silk depends on Weave behaviour that has no version or contract

Silk reaches into Weave internals that are stable by convention rather than
by declaration: `PortRegistry._by_name` / `_cast_registry` for port
registration (ARCHITECTURE_REVIEW R9), `NODE_REGISTRY` metadata and the undo
command tuple shapes for graph authoring (spec §18), the NodePanel mirror
system for the Decision Inbox (D59), and `emit_stream` / `pulse` semantics
throughout. None of this is versioned, and Weave's own graph files carry a
*format* version that the deserialiser explicitly does not gate on
(`canvas/serializer.py`).

Two consequences. A Weave refactor can break Silk silently at import time,
and there is no declared floor to test against — the same shape as G5 for
Python dependencies, one layer up. And Silk's own nodes carry no version, so
a Silk graph saved today has no defined behaviour when a node's ports change
(a real prospect: D55 makes delegation depth a port, D16 makes file access a
port, §18 adds a whitelist to the ToolBox node).

That plan has since grown the two pieces §19 depends on: a **programmatic**
load API returning a report rather than writing a log line (§3.10), and
**relaunch** -- Weave restarting itself with the session preserved (§3.11),
which is the only correct answer to a core or port-type change and therefore
the escape hatch that makes reload's permanent limits acceptable.

`docs/HOT_RELOAD_PLAN.md` (Weave) specifies the mechanism that closes the
second half — `node_version` / `node_state_api` / `migrate_state`, and a
`GhostNode` so an unknown class never silently drops the node *and its
connections*. Silk should adopt the metadata as soon as it lands, starting
with the nodes whose ports the spec is already changing.

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

**Correction (2026-08-30):** this entry assumed the *shared* half already
worked. It does not. `UsageLimits` (`functions/usage_limits.py`) is a plain
dataclass with `+=` counters and separate `check_*` / `record_*` calls, and
imports no lock; `delegate_parallel` threads one instance into N concurrent
workers. Check-then-record is a TOCTOU race, so several workers pass the same
check and collectively overrun the cap — the one global cap fails in exactly
the case it exists for. A correctness bug, not an ergonomics gap. See
**G18** and spec **D52(4)**.

### T4. Plan discovery policy (task store)

**Note (2026-08-31):** spec §16 (D58/D60) now depends on this store: the
Task Hub scans **all** `plan-*.db` under the graph's sandbox roots
(`scan_all`, additive) while agents keep the newest-only discovery. Whatever
policy T4 settles on must keep both readers coherent.

**Second note (2026-08-31):** spec §17 puts a `MacrameTaskStore`
(`TaskLedger`) behind the same protocol. Under that backend, discovery
*finds files* but reading goes through the process-local `LedgerRegistry`
(D62) — never a second open of a live ledger (sole Write Actor per
process). `scan_all` stays as specified for the SQLite fallback.


**Resolved** by spec §11, D23: a `Task Node` carries explicit plan identity
(plan id + store location) and feeds the ToolBox, so the Plan Viewer takes
that identity instead of guessing. Stub kept for inbound links. Until it
ships, the store still picks the *newest* `plan-*.db` by mtime across `root`
and `root/.silk/plan`, so concurrent plans in one root can cross-discover.
(The `Sign-Off` node named in the original entry is deleted by D32; the Plan
Viewer is the only remaining plan consumer.)

### T5. Default delegation depth

**Resolved** by spec §15, D55: the node's value wins, becomes an editable
port so the graph shows it, and the runtime default follows it rather than
diverging. Recorded chiefly so the divergence stops being re-discovered.
Stub kept for inbound links.

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

**Boundary set by spec §17 (2026-08-31):** the Macrame ledger takes the
*distilled* layer (turns, runs, task transitions, sign-offs, compaction
events); the raw `tool_events` firehose stays JSONL — events are a log,
not belief. T7's open call (build it, and when) is unchanged, but its role
is now defined rather than speculative, and D65 keeps the firehose out of
the ledger by construction.

### T9. Graph authoring: how far does an agent's build authority go?

Spec §18 (D69–D74) settles v1: six tools, a default-deny whitelist on the
ToolBox node, every mutation an undoable command, destructive calls scoped to
what the run created, and a refusal to touch the agent's own execution path.
What is deliberately unsettled:

- **Widget configuration** (§22 q9) — without it the agent builds skeletons.
- **Whether an agent may edit the user's pre-existing graph** rather than only
  its own additions. v1 says no. The approval story for "yes" is much
  larger than one gate: the human would need to see a *diff* of the proposed
  change, which is a different UI from the per-call approval of D48.
- **Whether a built graph should be reviewable before it is applied** — an
  agent proposing a subgraph as a *plan* (a value on a port, or a ledger
  assertion) that a human applies with one gesture, rather than mutating the
  canvas call by call. This is the strictly safer design and it inverts D70:
  no main-thread seam is needed at all, because nothing is applied from a
  worker thread. It is not chosen for v1 because incremental placement is
  what makes the tool usable interactively — but if D73's guard rails start
  accumulating exceptions, that is the signal to switch.

### T10. Self-modification: how far does the agent's build authority go?

Distinct from **T9**, which is about the *graph*; this is about *code*. Spec
§19 settles v1: the agent writes plugins into its own root (D76), loading is
always human-approved with the diff in view (D77), the linter gates version
discipline (D78), and a restart is a queued request at a turn boundary (D79).
What is deliberately unsettled:

- **Editing Silk itself**, or Weave core. Currently refused by D76 for the
  same reason D73 refuses graph edits to the agent's own execution path -- and
  with an extra problem the graph case does not have: the code that would
  review the change is the code being changed. If it is ever allowed, the
  shape is almost certainly *propose a patch, relaunch to apply*, never
  in-session reload.
- **Auto-load and auto-retry** (§22 q10, q11). These two together are what
  turn a supervised loop into an unattended one; they should be decided
  together, not separately, and probably not affirmatively.
- **Whether an agent may approve another agent's load.** No, under D77 --
  approval is a human act. Worth recording because an orchestrator with a
  reviewer sub-agent makes the question look reasonable, and it is not: the
  reviewer is the same class of thing as the author, running with the same
  authority the load is about to grant.

### T8. Context budget under raised autonomy (compaction — G14)

**Resolved** by spec §12: option C (loop-policy auto-compaction) in full,
with option A (the spill hook) alongside — and per **G15** / spec D41, A
carries the load first because it is prefix-preserving. Option B (an agent-invoked
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
| Lanes / continuable subagents | Need a session substrate; one-shot delegation with depth/cycle guards and a shared budget covers the current fan-out (T3 aside). The subagent question this row used to leave open — who answers an approval prompt raised inside a subagent — is now closed: a subagent has no node UI of its own (spec D48), so under D36 it denies, and gated tools reach it only through a durable grant. |
| Token metering (per-session replay folds, revisioned measurements) | Unnecessary at stock bounds. (Compaction was on this list until 2026-07-25; it is now a required mechanism, specified in spec §12 and tracked as [G14](#g14-compaction-is-not-implemented-required-mechanism).) |
| ~~KV-cache management~~ | **Removed 2026-08-30.** Held to be unnecessary because Silk was thought not to depend on cross-request prompt caching. It does — locally the dependency is paid in latency rather than money. Now [G15](#g15-prompt-prefix-reuse-is-unconfigured-and-unmeasured) / spec D41, D44, I11. |
| ~~Single model backend~~ | **Removed 2026-08-30.** `GGUFModelPool` assumes one spawned local server; the pool is to hold N named backends, local or remote (litellm and similar). Now [G11](#g11-openaiclientmock-is-the-production-client) / spec D45. |
| Approval / acknowledgement node | The answerer is the Agent node's own stream output UI (spec **D48**). A node form is impossible without an inbound mid-compute channel Weave does not have — inputs are gathered once, before `compute()`, and the Agent blocks *inside* it. Worked through and rejected in full at spec **D51**, including what the node form would have bought. Re-derived three times now (D12, D32, D51); the idea may return as a *configuration* surface — an Approval **Policy** node feeding the run-start policy snapshot (D38) — never as a runtime backchannel. *Centralizing* the N-agent case is solved without a node: a Decision Inbox **dock** built on Weave's NodePanel mirror system, spec **D59** — and the rule that makes all of this stop recurring is now invariant **I12** (node iff turn boundary). |
| Multi-package workspace machinery (sub-path exports, lockstep versions) | Organizational overhead for a monorepo Silk is not; the two-layer import rule is the same invariant at the right scale. |
