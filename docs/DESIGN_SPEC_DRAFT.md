# Silk — Design Spec (draft)

**Status:** draft, 2026-08-30. Decisions recorded here are settled unless
marked *open*; the rationale is kept so a later refactor doesn't silently
undo one. Companion to [ARCHITECTURE](architecture/README.md) (what exists
today), [OPEN_TOPICS](OPEN_TOPICS.md) (gaps and undecided questions), and
[NODES](NODES.md) (the current node surface).

This document describes a **target state**, not the current one. Where the
target differs from the code, the difference is stated with a citation so
the delta is checkable in seconds.

---

## 1. What this spec changes

Five things, in dependency order:

1. **One typed event vocabulary on one port** — replacing three parallel
   channels.
2. **A real approval gate** — closing the `requires_approval` no-op.
3. **Compaction as loop policy** — the mechanism on record as required.
4. **Tool-granular discovery** — exposing the search index the model
   cannot currently reach.
5. **New graph surface** — MCP and Task nodes; per-tool hook binding;
   file access as an explicit, narrowing-only port.

Everything else in the source draft either already exists (§2) or was
rejected against a load-bearing rule (§3).

---

## 2. Already implemented — cite, don't re-specify

The following were in the input draft and need no new work. They are listed
so the spec does not re-open them, and so the implementation plan does not
budget for them.

| Item | Where it lives |
|---|---|
| Native tool auto-discovery, categories, tags, risk | `ToolBox.register()` meta; `tool_catalog()` |
| Toolchains (python, pip, maturin, …), chainable, version-probed | `nodes/toolchain.py` |
| Sandbox integral to the ToolBox; roots as hard ceiling | `sandbox_roots` (`dirpath_list`) → `FileToolSandbox` |
| Read / Read+Write / Blocked per path | `file_permissions` = `{"root", "roots", "entries":[{"path","mode"}]}`; blocked = absent |
| **New files inherit the parent directory's rules** | `FileToolSandbox.resolve_mode()` (`functions/tools/file_sandbox.py:104`) — longest match wins, per-file override beats the parent rule, a new file inherits the nearest granted directory. Promoted to an invariant in §4 |
| Toolset = tool subselection + re-rooted file access | `nodes/toolset.py`, `build_toolset(source, selected_names, permissions)` |
| Role = instructions + hard-enforced tool gate | `nodes/role.py`, `ToolSelector` / `RoleBinding` |
| Agent control in/out | `run`/`done` exec pulses, `inbox`/`outbox` (`agent_message`) |
| Checkbox tree UI for tool selection | `widgets/tool_tree.py` |
| MCP runtime (SSE / Streamable-HTTP / stdio, auth, resources, prompts, sampling) | `functions/mcp_toolset.py` — runtime only; there is no MCP *node* |
| Conversation history across runs | `nodes/agent.py` owns `self._history` (persisted via `get_state`/`restore_state`) and feeds it to each `GraphEngine` — a parked run resumes with full context |

---

## 3. Rejected from the input draft

**D1. The Agent node does not accept a ToolBox.** It keeps `toolset` +
`role` only. Two agents sharing one live ToolBox contend for a single
`RoleBinding` (which refuses to activate when one is already active), which
is why ToolSets are derived by recipe rebuild rather than view-wrap. The
convenience of wiring a box straight to an agent is not worth hiding that
rule inside a node; users insert a `Silk ToolSet` node, and the constraint
stays visible in the graph.

---

## 4. Invariants

The five existing invariants ([17-invariants](architecture/17-invariants.md))
stand unchanged. This spec adds four, all **enforced, not conventional**:

**I6. File access narrows monotonically.** The ToolBox's `sandbox_roots` is
the hard ceiling. Every downstream layer (ToolSet → Role → Agent) may only
narrow the grant it received; no layer can widen, and no layer can re-enable
the sandbox escape hatch. Violations are a build-time error, not a silent
widening.

**I7. Essential hooks survive derivation.** A hook declared *essential*
rides the `build_recipe` into every derived ToolBox and cannot be dropped by
a ToolSet or Role. The safety classes — approval gate, secret redaction,
sandbox enforcement — are essential by definition.

**I8. Discovery obeys the role gate.** Discovery results are filtered by the
same `role_permits` predicate that governs advertisement and dispatch. The
model must never discover a tool it cannot call — that would half-break
invariant 4, which depends on advertisement and dispatch agreeing.

**I9. Compaction cuts on whole-round boundaries.** An assistant turn and all
its tool results move together. The native transport pairs `tool_calls` with
`tool`-role results; dropping one side of a pair corrupts the next request.

---

## 5. Event model

**D2. One typed vocabulary, one port.** Every event becomes a typed
`EventType` member — including hook, tool, task and compaction events — and
flows out a single `events` port.

**D3. Hard break.** The `tool_events`, `plan_events` and `chat_turn` ports
are removed and the consuming nodes (Hook Monitor, Plan Viewer, Chat Log
Display) are migrated -- the Sign-Off node is deleted outright (D32), not
migrated in the same change. No deprecated views, no
loader migration: saved graphs wired to the old ports must be re-wired.

Rationale: three vocabularies exist today — typed dataclasses from the loop
(`functions/stream_events.py`), untyped hook/tool dicts on `tool_events`, and
plan snapshots on `plan_events` built by `plan_changed_event`. G13 (an
`outcome` field) and G14(d) (a compaction event) both force changes here
anyway; doing it once is cheaper than three times.

**Additions to the vocabulary:**

- `outcome: completed | stopped | usage_limited | error` on `EventRunResult`
  — closes G13. Consumers key off `outcome`, not off "is there a final text",
  so a `max_rounds` abort stops reporting as a clean finish.
- `EventCompaction` — turns dropped, tokens before/after, summary reference.
  Content-free, per the observability rule.
- `EventApprovalRequest` / `EventApprovalDecision` — see §7. Both are
  emitted from hook callbacks rather than yielded by the loop generator,
  so the vocabulary must span both emission paths (D30).
- Hook and task events as typed members rather than dicts.

**Open:** whether `EventStart.system_prompt` (G10, always `None`) is
populated or dropped. Decide during implementation; it is a one-line change
either way.

---

## 6. Discovery

**D4. Tool-granular search, by category and capability.** A single core tool
`search_tools(query, category=None, capability=None)` returns matching
individual tools. Backed by the existing `ToolSearch` index
(`functions/tool_search.py`), which is already built and populated but which
**nothing model-facing currently calls** — the only agent-visible discovery
tool today is `load_capability`, which takes an id and no query
(`functions/tool_box.py:248`).

**D5. One core tool, not two.** Loading is implicit: there is no separate
load tool. This is the smallest possible always-present surface.

**D6. Auto-load at dispatch.** When the model calls a tool that was
discovered but never loaded, the dispatcher loads it and executes the call.
The role gate still runs, so auto-load cannot widen permissions — it only
saves a round-trip. A failed load returns a structured error in the
"errors carry the fix" style.

**Consequences:**

- Per-tool deferral is needed. The machinery partly exists — the ToolSet
  layer already has `defer_loading(tool_names)` → `DeferredLoadingToolset` —
  but the model-facing loader is capability-granular.
- G2 stops being cosmetic. `ToolSearch._bm25_search` currently delegates to
  `_keyword_search` with a TODO; once discovery is the primary path into
  context, ranking quality is load-bearing. Either implement BM25 or drop
  the strategy from the public surface.
- `load_capability` stays for capability-granular loading; it is no longer
  the only discovery path.

---

## 7. Approval gate

Closes **G1** (the gate is a `pass` with a TODO in `_safe_execute`) and
answers **T1** (where approval state lives).

> **Superseded 2026-08-30.** D7 (park and end the run), D8 (resume follows
> the exec chain), D9 (execute the held call on the resuming run), D28 (a
> pending-action subsystem) and D29 (a typed *pause* event) all assumed
> approval is an out-of-band interaction mediated by a node. It is not.
> D30-D34 replace them. D11 (essential hook) stands unchanged; D10 (grant
> scopes) stands but is amended on placement by D34.

**D30. Approval is inline, blocking, and lives in the stream.** The gate does
not park and does not end the run. It emits an approval **request** on the
run's stream, blocks the tool call, and waits for a decision delivered back
on the same channel. Approve -> the held call executes and the loop continues
*in the same run*. Deny -> a refusal is returned as that tool's result and the
model adapts. No pending-action store, no resume run, no exec-chain handoff.

This is cheaper than what it replaces, and it dissolves the problem D7 was
working around: a run is never interrupted, so *runs are atomic* needs no
special case.

*Why the threading permits it.* `AgentLoop.run` is a sync generator on the
`ThreadedNode` worker; each tool batch executes inside `asyncio.run(...)`
(`functions/agent_loop.py:241`). Hook callbacks fire inside that call and
**already reach the UI** -- `_stream_event` (`nodes/agent.py:356`) emits a Qt
queued signal from the worker thread while the generator sits mid-`next()`.
So a prompt can leave while the call is blocked, and the decision returns to
the worker through a thread-safe primitive.

*The constraint this creates.* While the gate blocks, the consumer loop
`for event in loop.run(...)` (`nodes/agent.py:477`) is inside a single
`next()` call and cannot pump. **The approval request therefore cannot be a
yielded event.** It rides the hook-emitted stream path, not the generator's
event path. The unified vocabulary in Section 5 must cover *both* emission
paths, or approval events cannot be typed like the rest.

**D31. Sign-off and tool approval are one hook, not one subsystem.** The
unification T1 asked for happens in the *gate*, not in a store: one
`wrap_tool_execute` middleware with two policy domains -- task changes
(`{change_type: level}`) and tool calls (`{tool_or_risk: level}`, using the
`risk` metadata that already exists). The `signoff` catalog entry generalises
to cover both; it is already the right shape, wired by `attach_catalog_hooks`
(`functions/hook_catalog.py:494`) because it needs the toolbox in scope.

Because nothing is parked any more, the parked-state machinery is **deleted,
not migrated**: `STATUS_AWAITING` / `awaiting_signoff`, `Plan.pending_goal`
and `pending_goal_summary`, the four `signoff_*` task columns,
`request_signoff` / `request_goal_signoff` / `sign_off`, and the
held-and-applied `signoff_action`. Audit survives in the existing `revision`
and `deviation` tables, which already record who did what, when and why.

**D32. No approval node.** `nodes/signoff_node.py` is deleted, along with the
pause inference it fed: the `signoff_hold` latch (`nodes/agent.py:340`), the
plan-shape check (`nodes/agent.py:387`) and its early-stop branch
(`nodes/agent.py:477`). The Agent node stops inspecting plan contents
entirely.

**D33. Forward-only; no migration.** Early development -- `plan-*.db` files
are recreated, not upgraded. The store has no `user_version` and no `ALTER`
path today (`functions/task_store.py:416` is `CREATE TABLE IF NOT EXISTS`
only), and none is added. Dropping the columns outright is correct here; the
price is that an in-flight plan does not survive the change, which is
acceptable now and would not be later.

**D11 (unchanged). The approval gate is an essential hook** (I7) -- a Role
cannot drop it.

**D10 (amended). Two grant scopes.**

- *Run-scoped*: "approve this tool for the rest of the run" lives in the
  gate's closure and dies with the run. No persistence at all.
- *Durable*: per-tool grants persist, scoped to the project. **Amended:** the
  original text placed them in `<root>/.silk/`, beside the plan store. That is
  wrong -- see D34 for why and D35 for where. Scoped *to* the project root,
  stored *outside* it.

**D34. The grant store is allow-only, keyed by project, and lives outside the
project.** Three constraints, each from a distinct failure:

- **Not the plan DB.** It exists only when the task tools are mounted
  (`functions/tools/task_tracker.py:199`), and both current readers fall
  through when it is absent (`functions/signoff.py:158`, `nodes/agent.py:370`).
  An agent with file tools and no planning tools is exactly the configuration
  that most needs a gate; a gate that isn't there when planning is off is not
  a gate. Under D30 the gate no longer *needs* a store to function, so this
  reduces to a code requirement: the tool-call domain must never consult the
  task store.
- **Not under the sandbox root.** The plan store is rooted at
  `sandbox.root_dir` -- inside the tree the agent can write to. A durable
  "always allow `write_file`" record is precisely the thing an agent should
  not be able to author. Grants live user-scoped, with the project root as
  the lookup key -- location settled by D35.
- **Allow-only.** A grant is a record that exists; revoking one deletes it.
  No deny records. That makes a missing, corrupt or unreadable store degrade
  to *nothing is granted* -- ask every time -- which is the safe direction. A
  store that held denials would resurrect revoked permissions on data loss.

Lifetime falls out of the same key: `_locate_db()` resolves to the *newest*
`plan-*.db` and a new plan mints a new file, so anything keyed to a plan
silently vanishes when a plan starts. Grants are keyed to the project and
outlive every plan in it.

**D35. Grants live at `~/.weave/silk/grants.json`, following the preset
precedent.** Silk already owns a user-scoped directory --
`~/.weave/presets/<kind>.json` (`functions/presets.py:21`) -- so grants need no
new convention, only a sibling. The mechanics copy `PresetStore`: a
`version` field (`FORMAT_VERSION`), a Pydantic model per record, `mkdir
(parents=True, exist_ok=True)` then `write_text` on flush, and a reload that
tolerates a missing or unparseable file by treating it as empty.

- **Keyed by resolved project root.** `Path(root).resolve()`, matching
  `SqliteTaskStore.__init__`. A grant made in a scratch project never applies
  to a sensitive one.
- **Allow-only** (D34). Revocation deletes the entry. There are no deny
  records, so a lost, empty or unparseable file means *nothing is granted* --
  the gate asks again. Every failure path leads to more prompting, never less.
- **A grant is not a preset.** It must not live in `PRESET_DIR`, must not
  appear in any preset model, and must never travel in a preset export.
  Presets are made to be shared and copied between projects; a grant that
  travels is consent nobody gave -- the same reasoning as D22's rule against
  persisting MCP credentials.
- **Concurrency: read-modify-write, last writer wins.** Two Agent nodes
  granting at once can clobber one another, and that is acceptable precisely
  because the records are allow-only: the lost grant costs one extra prompt.
  This is the one place where Silk does *not* follow `SqliteTaskStore`'s
  optimistic-concurrency habit, and the reason is that the failure is benign
  in the safe direction.

**What is left to build.** The deletions above are the easy half.

1. **Decision transport.** A request id correlating prompt to answer, and a
   thread-safe wait on the worker side resolved from the UI thread.
2. **Cancellation.** `is_compute_cancelled()` is polled by the *consumer*
   (`nodes/agent.py:477`), which is not running while the gate blocks. Stop
   must reach the waiter directly, or Stop will not stop a run that is
   sitting on a prompt.
3. **Timeout and default-deny.** A blocked gate holds the worker thread *and*
   the exclusive `RoleBinding` on the toolset, so no other Agent node can use
   that toolset meanwhile. "No answer" must not mean "forever".
4. **No-answerer detection.** Nothing wired to the `events` port, or a
   subagent -- the orchestrator runs each one under its own `asyncio.run` on
   its own thread (`functions/orchestrator.py:360`) and its stream is not
   necessarily surfaced anywhere. Default-deny, inherit the parent's grants,
   or refuse to build a gated subagent toolset: **open**.

---

## 8. Hooks

**D12. No Hooks node — for now.** The current two-seam split stays: ToolBox
hooks ride the recipe into every derived toolset; Role hooks are installed
by `RoleBinding.activate` and reversed by `deactivate()`. That distinction is
load-bearing and a single node would hide it. Revisit if a concrete scenario
needs hook composition the two selectors cannot express.

**D13. Per-tool binding as a first-class field.** A hook entry declares which
tools or categories it applies to; the registry does the filtering. Today
hooks fire for every tool and a "tool-specific" hook filters by name inside
its own body, so which hooks touch which tools is invisible in config.

**D14. An essential tier.** Hooks declare `essential: bool`. Essential hooks
ride the recipe and cannot be dropped downstream (I7). This formalises the
informal "infrastructure hooks: part of the recipe" comment already in
`nodes/toolbox.py`.

**D15. Wire the error family; review the rest before pruning.** Emit
`HOOK_ON_MODEL_REQUEST_ERROR`, `HOOK_ON_TOOL_VALIDATE_ERROR`,
`HOOK_ON_TOOL_EXECUTE_ERROR`, `HOOK_ON_OUTPUT_VALIDATE_ERROR`,
`HOOK_ON_OUTPUT_PROCESS_ERROR` and `HOOK_AFTER_MODEL_REQUEST` — cheap, and
useful for logging and metrics. The remaining unwired events are **not
pruned**; they get a review table (purpose, cost to wire, disposition) and a
decision per event.

Partly closes **G3** / **T2**. Until every event either fires or is removed,
registration on an unwired event should fail loudly rather than register
cleanly and never fire.

### Unwired event review (to be completed)

| Event | Kind | Purpose | Disposition |
|---|---|---|---|
| `HOOK_AFTER_MODEL_REQUEST` | event | after a request, distinct from the response | **wire** |
| `HOOK_ON_MODEL_REQUEST_ERROR` | event | model-request failure | **wire** |
| `HOOK_ON_TOOL_VALIDATE_ERROR` | event | argument validation failure | **wire** |
| `HOOK_ON_TOOL_EXECUTE_ERROR` | event | tool execution failure | **wire** |
| `HOOK_ON_OUTPUT_VALIDATE_ERROR` | event | final-output validation failure | **wire** |
| `HOOK_ON_OUTPUT_PROCESS_ERROR` | event | output post-processing failure | **wire** |
| `HOOK_WRAP_MODEL_REQUEST` | middleware | wrap a model request | review |
| `HOOK_WRAP_TOOL_VALIDATE` | middleware | wrap argument validation | review |
| `HOOK_WRAP_OUTPUT_VALIDATE` | middleware | wrap output validation | review |
| `HOOK_WRAP_OUTPUT_PROCESS` | middleware | wrap output post-processing | review |
| `HOOK_WRAP_RUN_EVENT_STREAM` | middleware | wrap the run's event stream | review |

---

## 9. File access

**D16. Explicit port, narrowing-only.** File access travels as a visible
port through ToolSet → Role → Agent, and each layer may only narrow what it
received (I6). The derived ToolBox's sandbox is rebuilt from the incoming
grant, so there is still exactly one source of truth — the port makes the
effective permission set visible in the graph rather than buried inside a
handle.

**D17. Pydantic for the grant structure.** `file_permissions` is currently a
dict described only in a docstring (`nodes/silk_ports.py`). It becomes a
Pydantic model, validated at the port boundary.

**D18. The sandbox escape hatch survives, but is never inheritable.** The
"Enable sandbox" toggle stays as a deliberate, clearly-labelled ToolBox-level
choice; a derived ToolSet or Role can never turn it back on. Consistent with
I6.

---

## 10. MCP nodes

New graph surface over the existing runtime (`functions/mcp_toolset.py`).

**D19. `MCP Node` owns one shared session per server.** Like the GGUF pool:
the node holds the connection; derived toolboxes attach to the same live
session by handle. One connection per server regardless of how many agents
derive from the box. Node cleanup closes it.

This is the answer to the sharpest problem in the input draft: ToolSets are
derived by *replaying* `build_recipe` attachers per agent, so a naive
recipe-level MCP attach would re-handshake per agent per evaluation.

**D20. `MCP Aggregator Node`** connects multiple MCP nodes to the ToolBox's
single `mcp` input, with a checkbox tree to enable/disable servers or
individual tools. Reuse `widgets/tool_tree.py`.

**D21. Always namespace by server.** Every MCP tool is prefixed with its
server id via the existing `prefixed()` ToolSet operation, so names are
collision-free by construction and the model can see a tool's origin.

**D22. Credentials are never persisted.** The node stores a credential
*name*; the value resolves at connect time from the environment or a secrets
file outside the graph. Presets (`~/.weave/presets/`) and saved graph files
stay shareable by construction.

MCP tools participate in discovery (§6) like any other tool.

---

## 11. Task node

**D23. A `Task Node` carries explicit plan identity** — the plan id and store
location — and feeds the ToolBox. The Plan Viewer takes that identity instead
of guessing. (With the Sign-Off node deleted (D32), the Plan Viewer is the
only remaining plan consumer.)

Promotes tasks from a ToolBox checkbox to a visible graph element (as the
input draft wants) and closes **T4** in the same move: today the store picks
the *newest* `plan-*.db` by mtime across `root` and `root/.silk/plan`, and
the plan nodes take a bare `root`, so concurrent plans in one root
cross-discover.

---

## 12. Compaction

Closes **G14**; implements **T8 option C** in full, with option A alongside.

**D24. Pressure trigger plus overflow retry, against a model-derived
budget.**

- Plumb `GGUFMeta.context_length` (available at model load, currently never
  reaching the loop — G14(c)) to the loop as the denominator.
- Auto-trigger at the existing pre-request seam (`agent_loop.py:166`) when
  estimated input exceeds `context − reserve`. Today that seam *fails* the
  run; it must be able to shrink and retry.
- Second trigger: a backend `n_ctx` overflow arrives as
  `EventError(context="stream_response")` → compact once and retry.

**D25. The agent's own model and pool session does the summarizing.** No
second model, no new port. A dedicated summarizer would need a second model
resident, which the single-server pool does not support and typical VRAM
budgets do not allow.

**Shape:**

- An optional `compactor` on `AgentLoop` (constructor argument, like the
  existing optional `output_validator`). The loop keeps owning the turn, the
  engine keeps owning one request, the compactor owns the summarization
  request.
- A new `AgentEngine` operation to replace the model-visible history prefix
  (G14(a)). `GraphEngine.history` is append-only today and the protocol has
  no rewrite operation. The replacement is built and swapped in **after** the
  summary succeeds — atomic.
- Cuts land on whole-round boundaries (I9).
- A failed compaction degrades to no compaction; the existing
  `EventUsageLimit` / `EventError` path still protects. Compaction never
  kills a run.
- Compaction rewrites only the model-visible history, never the run record.
- `EventCompaction` is emitted, content-free (§5).

**Option A (spill hook) ships alongside:** `spill_large_results(max_chars,
spill_dir)` in `hook_catalog`, beside `redact_secrets` — above threshold,
write the full tool result to a file and replace the model-visible content
with a head/tail preview plus the path. Deterministic, model-free, no extra
model calls, covers the dominant growth term. Needs a cleanup policy tied to
the run/plan root.

---

## 13. Budgets

**D26. Specify nested budgets; build later.** The intended semantics: a
global cap plus optional per-worker sub-budgets. Stated now so the compaction
and approval work do not have to guess, implemented after the core surface.
Today a fan-out shares one `UsageLimits` and one greedy worker can starve the
rest (**T3**).

---

## 14. Testing

**D27. Invariants first, as fixture data.** Encode the five existing
invariants plus I6–I9 as executable fixtures — one record per invariant and
per violation class — *before* the implementation lands, so the document and
the suite cannot drift apart. This is what **G4** already proposes, and it is
the main risk control for a change of this size: there are currently no tests
at all.

Highest-value targets beyond the invariants: the `AgentLoop` generator
contract (rounds, reflection, usage limits, `HOOK_AFTER_RUN` exactly-once),
the ToolBox execution path (role gate, structured errors, timeouts,
sequential vs parallel), `SqliteTaskStore` concurrency, and the orchestrator
guards.

---

## 15. Phasing

**Foundations first, then surface.** Each phase leaves the tree working.

**Phase 1 — foundations**
1. Invariant fixtures (D27) — the harness, plus the five existing invariants,
   plus fixtures for the parts of the sign-off gate that **survive** into D31:
   preset → policy resolution and the `complete` → `complete_final`
   resolution. The park/hold/apply path is deleted (D31), so it is not worth
   pinning.
2. Unified event vocabulary + single `events` port, including the approval
   request / decision events and the dual emission path (D2, D3, D30).
3. `outcome` on `EventRunResult` (G13).
4. Context-budget plumbing: `context_length` → loop (G14(c)).
5. Hook error-family emission + registration validation (D15).

**Phase 2 — safety and context**
6. Per-tool hook binding and the essential tier (D13, D14).
7. Delete the parked-state machinery (D31–D33): `awaiting_signoff`,
   `pending_goal`, the `signoff_*` columns, the park/apply store methods,
   `nodes/signoff_node.py`, and the Agent node's plan-shape pause inference.
   Forward-only — no migration.
8. Inline approval gate (D30, D11): decision transport, cancellation reaching
   the blocked waiter, timeout + default-deny, no-answerer policy;
   run-scoped grants in the gate closure, durable per-tool grants in
   `~/.weave/silk/grants.json` (D10, D34, D35).
9. Spill hook (option A).
10. Loop compaction: engine history-replace, compactor, `EventCompaction`
    (D24, D25).

**Phase 3 — surface**
11. File access as an explicit narrowing-only port; Pydantic grant model
    (D16–D18).
12. Discovery: `search_tools`, per-tool deferral, auto-load (D4–D6).
13. MCP Node + Aggregator (D19–D22).
14. Task Node with explicit plan identity (D23).

**Later:** nested budgets (D26); BM25 or its removal (G2); the unwired-event
dispositions (D15).

---

## 16. Gaps this closes

| Item | Closed by |
|---|---|
| G1 — approval gate is a no-op | §7 |
| G3 / T2 — unwired hook events | §8 (partial: error family wired, rest reviewed) |
| G4 — no test suite | §14 |
| G13 — `max_rounds` error silently dropped | §5 (`outcome`) |
| G14 / T8 — compaction absent | §12 |
| T1 — approval state design, reuse vs parallel | §7 (D30–D31: one inline gate hook, no state to design; only durable grants persist) |
| T3 — multi-agent budgeting | §13 (specified, not built) |
| T4 — plan discovery policy | §11 |
| G2 — BM25 is a keyword alias | §6 (forced into scope by discovery) |

Untouched by this spec: G5 (dependency declaration), G6 (pool recovery), G7
(`EventUsageLimit` granularity), G8 (mid-batch stops), G9 (type coverage),
G10 (`EventStart.system_prompt` — noted in §5), G11 (`OpenAIClientMock`
name), G12 (version metadata), T5 (delegation depth), T6 (HTML floor), T7
(durable event sink).

---

## 17. Open questions

1. The grant record schema and the revocation *surface* -- where a user sees
   and withdraws what they have granted (§7; location settled by D35).
1b. Who answers an approval when no consumer is wired, and what a gated
   *subagent* does (§7, D30 item 4).
2. Disposition of the five `WRAP_*` unwired events (§8).
3. Whether `EventStart.system_prompt` is populated or dropped (§5).
4. Spill-file cleanup policy and its lifetime (§12).
5. Whether the hook-node question returns once per-tool binding exists —
   D12 is a "not yet", not a "never".
