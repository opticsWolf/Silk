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

**D51. A node cannot be the answerer for a human decision.** Recorded here
because the question has now been re-derived three times -- D12 (no Hooks
node), D32 (no Sign-Off node), and again while specifying the decision
backchannel. D48 settles *where* the seam lives; this records *why the node
form is not an option*, so the next pass does not start over.

*The blocking objection is the evaluation model, not a preference.* Weave
gathers a node's inputs **once, before `compute()` runs**. The Agent node
blocks *inside* `compute()`, on a worker thread, mid-generator. A value
arriving on an input port has nowhere to land. Two ways around that, both
rejected:

- **A graph cycle** (`Agent.request -> Approval.in`,
  `Approval.out -> Agent.decision`). Even granting the cycle, the second edge
  is read at the start of the Agent's *next* compute -- i.e. the next run.
  That is park-and-resume, which D30 removed.
- **Pass the seam object downstream** and let the node call `resolve()` on it
  directly. This works, and is roughly where the deleted sign-off design was
  heading, but the port then carries a **mutable handle rather than data**.
  The wire transports nothing; it exists only to introduce two nodes to each
  other. Values on wires stop being inspectable, serializable or replayable,
  and the real communication is out-of-band.

*And the hard part does not move.* Under either variant the answering widget
is still on the main thread and the blocked gate is still on the Agent's
worker thread, so `DecisionSeam` (D49) is required unchanged -- lock-guarded
slot, `threading.Event`, write-under-lock-then-wake, idempotent `resolve`,
direct `cancel`. **The node form is D49 plus a node.** It removes nothing.

*What it would additionally cost:*

| | Node answerer | D48 (node-local) |
|---|---|---|
| Safety depends on | graph topology -- a forgotten wire denies every gated call with no visible cause | nothing; the asker owns the UI |
| Lifetime | node is graph-scoped, the seam is run-scoped -- needs demultiplexing by `run_id` + correlation id | matched by construction |
| Two agents / subagents | one node renders N concurrent requests from N runs | each answers for itself; no multiplexing |
| D36's first bullet | returns as a real failure mode | withdrawn |

The topology point is the same argument that settled D34: a gate that only
works when something else happens to be wired up is not a gate.

There is also a structural oddity that gives the game away: such a node would
**never evaluate**. It must be live and interactive *while an upstream node
is mid-compute*, which is outside evaluation order entirely, and its output
would flow nowhere. It is a dock panel wearing a node costume.

*What the node form would genuinely buy* -- stated so the trade is on record
rather than dismissed: a central review point across several agents; approval
policy that is composable in the graph (auto-approve, log-then-approve,
route-elsewhere) and **visible in the saved file** rather than implicit; and
a headless story that does not depend on durable grants (the consequence D48
accepts, and open question 1d).

*Where the idea goes instead.* None of those benefits requires an inbound
mid-compute channel. The gate already takes a **policy snapshot at run
start** (D38), so a future *Approval Policy* node can feed a policy in as an
ordinary input -- data, read before compute, at the normal time. That yields
composability and file-visibility with none of the above. Same shape as D12:
the node returns as a **configuration** surface, never as a runtime
backchannel.

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

**I10. Guard middleware is monotonic.** A hook that enforces owner policy --
the approval gate, sandbox enforcement -- runs outermost in the
`wrap_tool_execute` chain and no later registration may wrap it. Corollary,
and the half that actually bites: **no middleware may produce a success
result without delegating to `handler()`**. Denying without delegating is
legal; fabricating a result is not. Borrowed from dsh's *monotonic guards*
("owner policy that must not be reordered"). Note this is a different
property from I7: essential means *cannot be dropped*; monotonic means
*cannot be bypassed by something registered around it*.

**I11. The model-visible prefix grows only at the tail.** Between two
requests in one run, the earlier request's token sequence is a prefix of the
later one -- except at a compaction, which is the single deliberate
invalidation. This is pi's append-only cache invariant, and on a local
backend it is a latency invariant, not a cost one (see D41).

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
- `EventDecisionRequest` / `EventDecisionResponse` — see §7. One pair, not
  one pair per question: the request carries a `kind`
  (`approval | acknowledge | release`), so tool approval, "acknowledge before
  I compact" and "release this paused step" reuse a single seam rather than
  growing one each (D48). Both are emitted from hook callbacks rather than
  yielded by the loop generator, so the vocabulary must span both emission
  paths (D30).
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
- **Auto-load interacts with I11.** If loading a tool mid-run re-advertises
  the schemas in the system prompt, the prompt's *head* changes and the KV
  prefix is invalidated for every remaining round (D41). Either the newly
  loaded tool is dispatchable without re-advertising it (the model already
  knows the name from `search_tools`), or the re-advertisement is a
  deliberate, counted invalidation. Silent recomposition is the one option
  that must not happen.

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

**D36. Every failure path denies.** One rule covers the four ways a decision
can fail to arrive, and closes the no-answerer question:

- The run has no answering UI at all — a headless or batch evaluation of the
  graph, where the node's widget never renders.
- The widget is destroyed mid-run: the graph is closed, or the node deleted,
  while a request is outstanding.
- The decision transport raises.
- The timeout expires (see D38).
- The call is raised inside a subagent, which has no node UI of its own and
  whose stream the orchestrator runs under its own `asyncio.run` on its own
  thread (`functions/orchestrator.py:360`).

*(An earlier draft of this list opened with "nothing is wired to the `events`
port". D48 withdraws it: the answerer is the Agent node's own UI, not a
downstream graph node, so an unwired `events` port costs observability and
nothing else. The failure modes above replace it.)*

All four resolve to **deny**, returning the same structured refusal the model
sees for an explicit rejection. Taken from dsh, which states it flatly for
the same seam -- *"If no approval seam → deny"* -- and from pi, whose
`before_tool` hook fails closed when a handler throws. This composes with
D35: an absent grant store means nothing is granted, an absent answerer means
nothing is approved. Every degradation in the subsystem points the same way.

A gated subagent is therefore not an error at build time; it simply cannot
call gated tools unless a durable grant already covers them. That is a usable
configuration, not a broken one.

**D37. The gate is a monotonic guard (I10).** `emit_middleware` runs handlers
in registration order with the first registered outermost
(`functions/hooks.py`, `_chain(0)` over a stable list), and today
`attach_catalog_hooks` registers the catalog middleware *before*
`attach_signoff_gate` (`functions/hook_catalog.py:492` vs `:497`) -- so the
gate is currently **inside** `tool_budget` and `redact_secrets`, not outside
them. Since a middleware may return without calling `handler()`, anything
registered ahead of the gate can answer a call the gate never sees. Harmless
for the shipped hooks, which only deny; not harmless as a rule. The gate must
be forced outermost, and I10's no-fabricated-success corollary enforced.

**D38. One decision seam, not a bespoke wait.** The block is a single named
object the gate awaits -- pi's `Effects` reasoning: *"this closed method set
is the complete crash-site catalog."* Concentrating it means the four hard
parts are solved once rather than per call site:

- **Correlation.** A request id pairs the emitted prompt with the answer.
- **Cancel ordering.** The cancellation reason is recorded **before** the
  waiter is woken, so a wake is never observable without knowing why it
  happened. Pi's rule -- *abort commits control before pulling the signal* --
  which is what lets it separate user aborts from transport failures. Without
  it, Stop, the timeout and a real approval race into one indistinguishable
  wakeup. Note the consumer loop that polls `is_compute_cancelled()`
  (`nodes/agent.py:477`) is *not running* while the gate blocks, so Stop must
  reach the seam directly.
- **Timeout.** A blocked gate holds the worker thread *and* the exclusive
  `RoleBinding` on the toolset, so no other Agent node can use that toolset
  meanwhile. Expiry denies (D36).
- **Policy snapshot.** The resolved policy and grants are captured at run
  start and not re-read mid-run -- pi's *inline capture over references*, so
  editing a Role or hook config mid-run affects the next run, not the one in
  flight.

**D48. The seam is node-local: the Agent node's own stream output UI asks
and answers.** No answerer node, no inbound graph port. The request is
rendered inline in the same widget that shows the streaming response, and the
human replies there.

*Why node-local is the only coherent option.* `events` is an **output** port.
In a dataflow graph a downstream consumer has no return path to the upstream
node that is blocked, so "the decision comes back on the same channel" could
never have meant a graph round trip. It means the same *conversation view* --
the node that asked is the node that is answered. The node-shaped alternative
is worked through and rejected in full at **D51** (§3), including what it
would have bought and where that idea should go instead.

*The two directions are not symmetric, and only one of them is new.*

- **Out (worker -> UI):** already built and already used. `emit_stream`
  (`nodes/agent.py:359`, `:484`) is a Qt queued signal emitted from the
  `ThreadedNode` worker while the generator sits mid-`next()`; `_stream_event`
  uses it with `throttle_ms=0`. The request goes out this way, and the UI
  renders it with `push_display` on the main thread like every other display
  write (WV401).
- **In (UI -> worker):** new, and **not** a Qt signal. The worker is blocked
  inside `next()` with no event loop, so there is nothing to deliver a queued
  signal *into*. It must be a plain threading primitive the blocked thread is
  already waiting on.

**D49. One run-scoped `DecisionSeam` object, and the ordering rule that makes
it testable.** Created at run start, held by the node, closed over by the
gate. Its whole surface:

| Called by | Method | Thread |
|---|---|---|
| gate (blocked) | `await_decision(request_id, timeout)` | worker |
| UI widget | `resolve(request_id, response)` | main |
| Stop handler | `cancel(reason)` | main |

Implementation shape: a `threading.Lock` guarding
`{request_id, response, reason}` plus a `threading.Event` the worker waits
on. The rule, generalising D38's cancel-before-wake to all four wake causes:

> **Write the outcome under the lock, then set the event; the waiter re-reads
> under the lock before acting.**

That is what makes approve, deny, Stop and timeout four *distinguishable*
wakeups rather than one ambiguous one, and it is what turns D42's ten
orderings into deterministic tests instead of a flaky suite. `resolve()` is
idempotent per request id -- a second decision for a resolved id is a no-op
reporting "already resolved", which is D42's fifth race.

Stop must call `cancel()` **directly**, not rely on the consumer loop: that
loop is not running while the gate blocks (D38, G8).

**D50. One seam serves acknowledge and release too.** The request carries a
`kind` (D2). `approval` is approve/deny on a held tool call; `acknowledge` is
a continue/abort checkpoint the agent must pass (a compaction about to fire,
per D24, is the obvious first user); `release` resumes a step the human
paused. All three are the same block, the same correlation id, the same
timeout, the same fail-closed rule -- only the response payload differs. This
is the point of D38: the second human-in-the-loop question must not build a
second waiter.

*Consequence to accept deliberately.* Because the answerer is the node's UI,
**a graph that works interactively degrades to deny when run headless** --
batch evaluation, CI, a subagent. That is the correct default under D36, but
it is a real behavioural difference between the two ways of running the same
graph, and the way to make a headless run useful is a durable grant (D34,
D35) that covers the gated tools in advance, not a weakening of the gate.

**D39. Stamp the schema version even though nothing migrates.** D33 stays
forward-only, but the store writes `PRAGMA user_version` and checks it on
open. Cost is one line; the benefit is that the next schema change fails with
"this plan file is from an older Silk, delete it" instead of an
`OperationalError` raised from inside `_load_con`'s column list. dsh does
exactly this -- `SESSION_FORMAT_VERSION = 0`, no compatibility promise, but
the number is *there*. A declared pre-release stance beats an undeclared
one.

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

**D40. Classify the model-request error before compacting.** D24's second
trigger reacts to `EventError(context="stream_response")`, but that context
covers *every* stream failure, including a dead server -- and G6 records that
the pool has no liveness check and no restart. As written, a crashed
`llama_cpp.server` would be answered by spending a summarization request
against it and retrying. Pi reduces transport noise to three orthogonal
predicates before anything upstream reacts -- *retryable? / overflow? /
recoverable-length?* -- with an explicit `isContextOverflow(message,
contextWindow)` separating genuine overflow from provider-limit errors.

Silk has the tool-side half of this already (`is_retryable_tool_error` in
`functions/reflection.py`) and nothing equivalent for model requests. A
model-request error classifier is therefore a **precondition of D24**, not a
follow-up: compact only on classified overflow, never on a generic stream
error.

### Prefix reuse — the local-inference constraint

**D41. Compaction is a prefill event, and prefill is the dominant local
cost.** T8 recorded a cache note claiming Silk's pool "does not depend on
cross-request prompt caching, so no equivalent protection is needed". That
was inherited from the hosted-API framing, where cache misses cost money. It
is wrong here, and the code says so.

*Verified mechanics* (llama-cpp-python 0.3.34, `llama_cpp/llama.py`,
`Llama.generate`): before evaluating a prompt the runtime scans for the
longest common prefix between the new tokens and `self._input_ids`, then
calls `kv_cache_seq_rm(-1, reuse_prefix, -1)` and evaluates **only the
suffix**. Three properties follow, and all three matter:

1. **There is one resident context per instance,** not one per conversation.
   `_input_ids` holds *the last prompt evaluated*, whoever sent it.
2. **Reuse therefore only survives consecutive requests from the same
   conversation.** Two Agent nodes alternating, or an orchestrator fan-out
   (`delegate_parallel`), clobber each other: A, B, A, B reuses nothing
   beyond the shared system prompt, while A, A, B, B reuses almost
   everything. Silk shares one server across all agents by design
   (`functions/model_pool.py`: "a shared client, not a slot").
3. **The multi-state cache is off.** `LlamaCache` -- the prefix-keyed store
   that would survive interleaving -- is opt-in through the server's `cache`
   / `cache_type` / `cache_size` settings
   (`llama_cpp/server/settings.py:143`), and `_SERVER_MODEL_KEYS`
   (`functions/model_pool.py:92`) does not forward them. Server defaults
   apply, so it is disabled.

*Why this collides with D24.* Compaction rewrites the **head** of the
context. After it, the longest common prefix with the previous request
collapses to roughly the system prompt, so the entire surviving context is
re-prefilled -- and by construction that context is near the ceiling, i.e.
the most expensive prefill the run will ever pay. Worse, D25 has the agent's
own model produce the summary: that nested request has a completely different
prompt, so it clobbers `_input_ids` on its way through. **A compaction costs
two full prefills, not one** -- the summarization request, then the rebuilt
context. On a hosted API this is a billing line; on a local GPU it is dead
wall-clock in the middle of a run, at the moment the user is already waiting.

**Consequences for the design:**

- **Spill (option A) is prefix-preserving; summarization (option C) is
  prefix-destroying.** The spill hook rewrites a tool result *before it is
  appended*, so history stays append-only and I11 holds. That is a stronger
  argument for A than "it is cheap and lands first": A is the mechanism that
  does not fight the cache, and it should carry as much of the load as it
  can before C is triggered at all.
- **Compaction must be rare and decisive.** Hysteresis on the trigger and a
  generous keep-recent, so a run compacts once rather than repeatedly --
  every repeat is another double prefill.
- **`EventCompaction` reports the prefill cost,** not only tokens dropped.
  Otherwise the most expensive thing compaction does is invisible.
- **Prefix stability becomes a rule (I11).** The system prompt must render
  byte-identical across a run: no timestamps, no volatile context. dsh states
  the same constraint for the same reason (§4.7: prefix-stable while
  identity, persona, variables, section text and order render identically).
- **Measurement comes first, and is nearly free.** `verbose` is already
  forwarded in `_SERVER_MODEL_KEYS`, and `generate()` prints
  `"<n> prefix-match hit, remaining <m> prompt tokens to eval"`. Before any
  of the above is tuned, capture that line against a real multi-round run and
  against a fan-out. The reuse rate is the number the whole design hangs on
  and nobody has looked at it.
- **Enabling `LlamaCache` is available but is not free** -- see D44. The
  forwarding is a one-line change; the cost model is not, so it is gated on
  the measurement above rather than switched on by default.

Note this reaches beyond compaction: item 2 means **`delegate_parallel` is
considerably more expensive than it looks today**, with each worker paying a
full prefill every round. That is a live cost in shipped code, not a
consequence of anything in this spec.

**D43. The shared llama server truncates in-flight streams, and that is a
correctness bug, not a performance one.** `llama_cpp/server/app.py`
serializes every request through `llama_outer_lock` / `llama_inner_lock`, so
the "parallel" in `delegate_parallel` never reaches the model -- requests
queue. Worse, the streaming publisher checks, per chunk:

```python
if interrupt_requests and llama_outer_lock.locked():
    await inner_send_chan.send(dict(data="[DONE]"))
    raise anyio.get_cancelled_exc_class()()
```

`ServerSettings.interrupt_requests` **defaults to `True`**
(`llama_cpp/server/settings.py:223`), and Silk's generated config sets only
`host`, `port` and `models` (`functions/model_pool.py:217`), so the default
stands. The effect: **while agent A is streaming, a request from agent B
truncates A's response.** A gets a well-formed `[DONE]`, and
`OpenAIClientMock.generator()` (`model_pool.py:153`) cannot distinguish that
from a natural stop -- it simply ends the generator. The agent then reasons
over, and may act on, a silently cut-off assistant turn.

And the detection is currently defeated by a default. `stream_response`
initialises `finish_reason = "stop"` (`functions/graph_engine.py:202`) and
only overwrites it when a chunk carries one (`:222`). A truncated stream
carries none -- so `last_stats` reports `finish_reason: "stop"`, the same
value a clean completion produces. The one signal that would expose the
truncation is pre-set to the answer that hides it.

*Required, three parts:* forward `interrupt_requests: false` in the server
config; initialise `finish_reason` to `None` and treat a stream that ends
with it unset as `EventError(context="stream_response")` rather than a
completed turn -- exactly the classification D40 already requires; and keep
that check even after the setting is forwarded, since a remote backend
(D45) can truncate for its own reasons. Until this lands, any concurrent
multi-agent graph is unsound, independent of cache behaviour.

**D44. `LlamaCache` is forwardable, but its cost model must be measured
before it is enabled.** `cache`, `cache_type` and `cache_size` are
`ModelSettings` fields, so enabling them means adding three names to
`_SERVER_MODEL_KEYS` (`model_pool.py:91`) -- the existing JSON config path
carries them, and `llama_cpp/server/model.py:334` constructs a
`LlamaRAMCache` or `LlamaDiskCache` and calls `set_cache`. No new machinery.

What the switch actually buys and costs:

- *Buys:* a prefix-keyed multi-state store. `_create_completion`
  (`llama.py:1363`) looks up the longest-prefix entry and loads it **only if
  its prefix beats the resident context's** -- so interleaved conversations
  stop clobbering each other, which is the mechanism item 2 is missing.
- *Costs:* every completion ends with
  `self.cache[prompt + completion] = self.save_state()`
  (`llama.py:1700`). `save_state` (`:2199`) allocates and memcpies the full
  serialized context blob (`llama_state_get_size`, scaling with `n_ctx` and
  KV quant) **plus** a copy of the scores array (at most
  `n_batch x n_vocab` float32 -- tens to hundreds of MB). Default
  `cache_size` is `2 << 30` = 2 GiB, so with a large context the LRU can
  evict on nearly every insert: pay the copy, keep nothing.

So this is a genuine trade, not an oversight to correct. Sizing, backing
(RAM vs disk) and whether `save_state` overhead is tolerable at the project's
context sizes are settled **by the D41 measurement**, not by argument.

**D45. The model pool is multi-backend; a single local server is one case of
it, not the shape.** `GGUFModelPool` today hardcodes one `_process`, one
`_port` and one `_client`; `checkout(session_id)` ignores its argument and
returns `self._client`, and `n_instances` is explicitly display-only
(`model_pool.py:196`). The pool must instead hold **N named backends**, each
either a spawned local `llama_cpp.server` or a remote OpenAI-compatible
endpoint (litellm, vLLM, a hosted provider), with the Agent/Role selecting
one.

Three things this needs, all small and all currently absent:

1. **`checkout()` is already the routing seam** -- it takes a `session_id`
   and discards it. Backend selection, and the request affinity D41 leaves
   open, both live there.
2. **`OpenAIClientMock` is already the full client surface**
   (`create_chat_completion`, `tokenize`, `reset`) and is constructed from a
   bare `base_url`. A remote backend is that class with a different URL and
   no subprocess -- but it sends only `Content-Type`
   (`model_pool.py:128`), so **there is no way to pass an API key today**.
   Adding an `Authorization` header is a precondition for litellm, and the
   key must follow D22: a credential *name* resolved at connect time, never
   persisted in the graph or a preset.
3. **`snapshot()` returns a single flat dict** (`model_pool.py:323`) with
   `total_instances: 1` and zeroed KV fields. It becomes per-backend, and it
   is the natural place to surface the D41 prefix-reuse rate.

Consequence for D41: prefix reuse is a *per-backend* property. Routing two
agents to two backends is itself a cache strategy -- and, given D43, the only
one that makes concurrent agents both correct and fast until
`interrupt_requests` is fixed.

**Option A (spill hook) ships alongside** -- and, per D41, carries the load
first: `spill_large_results(max_chars, spill_dir)` in `hook_catalog`, beside
`redact_secrets` — above threshold, write the full tool result to a file and
replace the model-visible content with a head/tail preview plus the path.
Deterministic, model-free, no extra model calls, append-only, covers the
dominant growth term. Needs a cleanup policy tied to the run/plan root.

### Session identity, and the three ways to restore prefix reuse

**D46. Per-agent session identity already exists end to end; the pool is the
only thing that discards it.** Nothing needs inventing here:

| Where | What | Line |
|---|---|---|
| Agent node | `self._session_id = str(uuid.uuid4())` -- one per node, persisted with node state | `nodes/agent.py:105` |
| Sub-agent | `session_id or str(uuid.uuid4())` -- a fresh key per sub-agent run | `functions/subagent.py:177` |
| GraphEngine | carries it and passes it on every request | `functions/graph_engine.py:57`, `:260` |
| Pool | **ignores it** -- `checkout()` increments a counter and returns the one shared client | `functions/model_pool.py:308` |

So the answer to "shouldn't each agent have its own session id" is that each
already does; the identity survives three layers and is dropped at the
fourth. Every mechanism below is a policy applied at that one seam, which is
why none of them requires new plumbing above the pool.

Two corrections fall out of the same reading:

- `Clear Context` reaches into `pool._session_instances`
  (`nodes/agent.py:258`), an attribute of the **old multi-`Llama` pool** that
  `GGUFModelPool` does not have. The access sits inside
  `except Exception: log.debug(...)`, so it fails silently: the session is
  never released and `_active_sessions` only ever grows. Fix with the pool
  work, not separately.
- `snapshot()` reports `bound_sessions` from that same counter, so the loader
  display is wrong for the same reason.

**D47. Three mechanisms restore prefix reuse; they attack different terms,
and the choice between them is a measurement, not a preference.**

First the shape of the problem. Reuse is lost two ways, independently:

- **(i) Contention** -- another conversation's prompt is resident, so the
  match collapses to the shared system prompt. Caused by interleaving.
- **(ii) Rewriting** -- compaction changes the *head* of the context, so the
  match collapses even with no other agent present (D41).

I11 and the spill hook address (ii). All three mechanisms below address
**(i) only** -- they are not alternatives to prefix stability, and none of
them makes compaction cheap.

**Mechanism A -- affinity: group the queue by session.** Do not interleave
one conversation's rounds with another's; run A's round to completion, then
B's. The session id from D46 is the grouping key.

*Cost: close to zero, and this is the non-obvious part.* Per D43 the server
already serializes every request through `llama_outer_lock` --
`delegate_parallel` never had model-level concurrency to lose. A therefore
does not trade throughput for reuse; it only changes the order of a queue
that already exists, from arrival order to session-grouped order. What it
does cost is *fairness*: a long agent turn delays the others, and that
becomes visible latency in an orchestrator fan-out.

*Where it lives:* the pool, at `checkout()`. Not the Agent node -- a node
cannot see the other conversations.

**Mechanism B -- `LlamaCache`: keep more than one resident state.** D44. Buys
correctness under interleaving without changing scheduling. Pays
`save_state()` on every completion whether or not the entry is ever read, and
can thrash against `cache_size` (D44).

**Mechanism C -- multiple backends: give each conversation its own resident
context.** D45. The only mechanism that removes the contention rather than
managing it, and the only one that also restores *real* concurrency, since
one `llama_cpp.server` gives none (D43). Costs a full weight load per local
backend -- so in practice it means either enough VRAM for N models, or remote
backends (litellm and similar), which is why D45 and this are the same work.

**How they compose.**

- **A and C compose and are the natural pair:** route sessions to backends,
  and group by session within each backend. C handles as many concurrent
  conversations as there are backends; A handles the overflow.
- **A largely subsumes B.** If scheduling can be grouped, the multi-state
  cache has little left to do -- B exists for the case where rounds
  *genuinely must* interleave and waiting is unacceptable.
- **B and C compose** but rarely need to: B's per-completion copy cost is
  paid per backend.
- **None of them substitutes for I11.** A prefix that is not byte-stable
  across a run defeats all three.

**How to decide -- the measurement, then the rule.** Phase 1 captures
`generate()`'s `"<n> prefix-match hit, remaining <m> prompt tokens to eval"`
line (D41). Tag each request with its session id at the pool seam and three
numbers fall out:

| Metric | Definition | What it decides |
|---|---|---|
| **Reuse rate** | matched tokens / prompt tokens, per request | whether reuse is being lost at all |
| **Contention rate** | fraction of requests whose immediate predecessor came from a *different* session | whether the loss is (i) or (ii) |
| **Prefill share** | prefill time / total request wall-clock | whether any of this is worth building |

Applied in order, and the first rule that matches wins:

1. **Prefill share is small (say under ~15%).** Do nothing. None of A, B or C
   pays for itself, and this is the outcome that must stay reachable -- the
   original T8 note assumed it without measuring, and the correct response to
   confirming it is to stop, not to build anyway.
2. **Contention rate is near zero** (single-agent graphs dominate). The loss
   is (ii). Ship I11 and the spill hook; skip A, B and C entirely.
3. **Contention is high, one local backend, VRAM-bound.** **A.** It is free
   in throughput terms, needs no new setting, and is a change to one function.
4. **Contention is high and A's latency penalty is unacceptable** -- an
   orchestrator whose workers must genuinely progress in lockstep. **B**, and
   only if the inequality holds: measured `save_state` copy cost per
   completion < measured prefill saved per hit, at this project's `n_ctx`.
   D44 exists because that is not obviously true.
5. **VRAM or a remote provider is available.** **C**, and prefer it -- it is
   the only option that also fixes the concurrency loss in D43, and the only
   one whose benefit does not degrade as the number of concurrent agents
   grows.

Rules 3 and 5 are not exclusive: implementing C does not retire A, because
the moment concurrent conversations outnumber backends, contention returns
and A is what handles it.

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

**D42. A manual-drive gate for the approval seam.** D30 gives Silk its first
real concurrency surface: a parked worker thread, a Qt thread resolving the
decision, and Stop and the timeout racing that resolution. Invariant fixtures
do not test races. Pi's answer is a test-mode gate that parks before each
effect and exposes step/peek/run-to-completion, with the enforced property
**zero writes and zero effects while parked** — which is what turns a race
catalog into deterministic tests rather than a flaky suite. Both reviews name
it the single most transferable idea in pi, and D30 is exactly the situation
it was built for.

Minimum catalog to drive in both orders: approve-vs-Stop, approve-vs-timeout,
deny-vs-Stop, decision-arrives-after-timeout, and a second decision for an id
already resolved. Five races, ten orderings — small, and impossible to
exercise reliably any other way.

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
6. **Measure prompt-prefix reuse** (D41, D47): `verbose` is already
   forwarded, so capture `generate()`'s prefix-match line across a
   multi-round run and an orchestrator fan-out, tagging each request with the
   session id the pool already receives (D46). Report reuse rate, contention
   rate and prefill share -- the three numbers D47's rule consumes. Nothing
   in mechanisms A/B/C is built before this exists, and "do nothing" is one
   of its permitted outcomes.
7. Model-request error classifier — overflow vs retryable vs terminal (D40).
   A precondition of D24, not a follow-up.
8. **Disable `interrupt_requests` on the spawned server** and treat a stream
   that ends without a terminal `finish_reason` as an error (D43). Two-line
   fix for silent response truncation whenever two agents run at once; it
   gates every concurrent graph, so it precedes everything in Phase 2.

**Phase 2 — safety and context**
1. Per-tool hook binding and the essential tier (D13, D14).
2. Delete the parked-state machinery (D31–D33): `awaiting_signoff`,
   `pending_goal`, the `signoff_*` columns, the park/apply store methods,
   `nodes/signoff_node.py`, and the Agent node's plan-shape pause inference.
   Forward-only — no migration.
3. Inline approval gate (D30, D11): the run-scoped `DecisionSeam` with
   correlation, write-under-lock-then-wake ordering, timeout and policy
   snapshot (D38, D49); the request/response UI built into the Agent node's
   stream output widget, with `emit_stream` outbound and a threading
   primitive inbound (D48); one `kind` field so acknowledge and release reuse
   the same seam (D50); fail-closed on every missing-answer path (D36); the
   gate forced outermost as a monotonic guard (D37, I10); run-scoped grants
   in the gate closure, durable per-tool grants in `~/.weave/silk/grants.json`
   (D10, D34, D35); `PRAGMA user_version` on the plan store (D39). Driven by
   the manual-drive race catalog (D42).
4. Spill hook (option A) — prefix-preserving, so it carries the load before
   compaction is triggered at all (D41).
5. **Whichever of D47's mechanisms the Phase 1 numbers select**, plus the
   prefix-stability rules (I11) which are unconditional:
   - *A — session affinity* (D46, D47): honour the session id at
     `checkout()`, group the queue by conversation, and fix the two things
     that reading exposed — the dead `pool._session_instances` access in
     `Clear Context` and the `bound_sessions` count it corrupts.
   - *C — multi-backend pool* (D45): N named backends behind `checkout()`,
     an `Authorization` header on `OpenAIClientMock` so remote/litellm
     endpoints work at all, per-backend `snapshot()`.
   - *B — `LlamaCache`* (D44) only if rule 4 fires and the copy-vs-prefill
     inequality holds.
6. Loop compaction: engine history-replace, compactor, `EventCompaction`
    carrying prefill cost (D24, D25, D40, D41).

**Phase 3 — surface**
1. File access as an explicit narrowing-only port; Pydantic grant model
    (D16–D18).
2. Discovery: `search_tools`, per-tool deferral, auto-load (D4–D6).
3. MCP Node + Aggregator (D19–D22).
4. Task Node with explicit plan identity (D23).

**Later:** nested budgets (D26); BM25 or its removal (G2); the unwired-event
dispositions (D15); the D47 mechanisms not selected by the measurement --
kept described rather than deleted, since the rule that skips one today
selects it as soon as the graph shape changes.

---

## 16. Gaps this closes

| Item | Closed by |
|---|---|
| G1 — approval gate is a no-op | §7 |
| G3 / T2 — unwired hook events | §8 (partial: error family wired, rest reviewed) |
| G4 — no test suite | §14 |
| G13 — `max_rounds` error silently dropped | §5 (`outcome`) |
| G14 / T8 — compaction absent | §12 |
| G15 — prompt-prefix reuse unconfigured/unmeasured | §12 (D41; measurement is Phase 1) |
| T1 — approval state design, reuse vs parallel | §7 (D30–D31: one inline gate hook, no state to design; only durable grants persist) |
| T3 — multi-agent budgeting | §13 (specified, not built) |
| T4 — plan discovery policy | §11 |
| G2 — BM25 is a keyword alias | §6 (forced into scope by discovery) |

G6 is no longer fully untouched: D40 makes a model-request error classifier a
precondition of D24, and that classifier is what would let a future supervisor
tell "the server died" from "the context overflowed". The restart itself stays
out of scope.

Untouched by this spec: G5 (dependency declaration), G7
(`EventUsageLimit` granularity), G8 (mid-batch stops), G9 (type coverage),
G10 (`EventStart.system_prompt` — noted in §5), G11 (`OpenAIClientMock`
name), G12 (version metadata), T5 (delegation depth), T6 (HTML floor), T7
(durable event sink).

---

## 17. Open questions

1. The grant record schema and the revocation *surface* -- where a user sees
   and withdraws what they have granted (§7; location settled by D35).
   *(The former question 1b -- who answers when nobody is listening -- is
   closed by D36: every missing-answer path denies.)*
1b. Which of D47's three mechanisms to build — decided by the rule in §12,
   not by argument, and blocked only on the Phase 1 measurement. What the
   rule does *not* settle: if B is selected, `LlamaCache` size and backing
   (RAM vs disk); if C, whether the backend is chosen by the Role, by the
   Agent node, or by a pool-side rule keyed on session, and what happens
   when a named backend is down.
1c. Whether session affinity (mechanism A) is the pool's business alone or
   needs a visible surface — an orchestrator fan-out that silently serializes
   is correct but looks hung, and D43 means it already does this today with
   no indication.
1d. Whether a *headless* run should fail loudly rather than deny quietly
   (D48): denying is correct, but a batch evaluation whose every gated tool
   is refused should probably say so once, not once per call.
2. Disposition of the five `WRAP_*` unwired events (§8).
3. Whether `EventStart.system_prompt` is populated or dropped (§5).
4. Spill-file cleanup policy and its lifetime (§12).
5. Whether the hook-node question returns once per-tool binding exists —
   D12 is a "not yet", not a "never".
