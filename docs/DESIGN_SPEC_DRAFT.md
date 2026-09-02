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

**I12. A human decision surface may be a node iff the decision happens at a
turn boundary.** Mid-run decisions are node-local (D48) or *mirrors* of the
node-local surface (D59) -- never nodes. This is the rule that reconciles the
shipped Sign-Off node with D51's rejection of an approval node: sign-off parks
the task, **ends the turn**, and re-triggers by pulse -- everything happens
where the evaluation model permits it -- while the approval gate blocks inside
`compute()`, where no graph channel can reach it. One sentence, and it is the
decision procedure for every future "should X be a node?"
(ARCHITECTURE_REVIEW.md, R6.)

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

**D67. Concurrent file writes are serialised by a two-tier lock in the
sandbox. The first tier exists; the second closes the subprocess hole.**

*Tier 1 -- per-path locks (already shipped, now a spec-level guarantee).*
`functions/tools/file_locks.py`: a **process-wide** registry of
`threading.Lock`s keyed by resolved path, deliberately module-global so the
guarantee spans FileToolSandbox instances -- **and therefore spans agents**:
two agents writing the same file through the file tools are already
serialised today. Every mutating file tool holds it (`write_file`,
`append_file`, `create_directory`, `edit_file`, `insert_text`, `copy_file`,
`move_file`, `delete_file`); two-path operations acquire in sorted canonical
order, so they cannot deadlock. Combined with `_atomic_write`'s
`os.replace`, a lost update between file tools is impossible in one
process, for any number of agents. Also on record: `edit_file` /
`insert_text` verify their anchor text before writing, which is **optimistic
concurrency at the semantic level** -- an agent editing against a stale read
fails cleanly with a mismatch instead of clobbering the other agent's
change. `write_file` alone is a blind overwrite (see D68).

*The hole (G19).* Toolchain subprocesses that rewrite files -- `ruff
format`, `cargo fmt`, `run_python` (which can write anything) -- never
touch `lock_paths`, and their `sequential=True` flag orders execution only
**within one agent's batch** (`tool_box.py:669`). Across agents nothing
serialises them: agent A's formatter can interleave with agent B's
`edit_file` on the same tree, and neither is told.

*Tier 2 -- a per-root readers-writer gate (new).* Same registry pattern as
tier 1, keyed by resolved sandbox root, process-global. Rules:

- **File tools**: shared(root containing the target) + exclusive(path) --
  their behaviour among themselves is unchanged.
- **Subprocess tools that may write** -- a `writes_files` flag on
  `CommandSpec` (`ruff_format`, `cargo fmt`, every `run_*`): exclusive(root)
  for the subprocess's duration, because nothing can know which files a
  subprocess will touch. Coarse on purpose: correctness first, and a
  formatter run is short.
- **Read-only subprocesses** (checks, `--no-fix` lints, mypy, radon): no
  gate.

Ordering rule, extending tier 1's: **root gates before path locks, both in
sorted canonical order.** Where registered roots from different ToolBoxes
nest, a writer takes the gates of every registered root in an
ancestor/descendant relation with its own -- the registry is small, so this
is cheap. An exclusive gate held by a long subprocess is a *visible* wait:
the blocked tool call emits its usual `tool_call` event and the agent's
status line says what it is waiting on, so a gated fan-out reads as queued,
not hung (same legibility rule as D53/1c).

**D68. Scope ruling: the lock is advisory and per-process; *ownership* is
the ledger's job, not the lock's.**

- *Advisory, per-process* -- the same boundary as D62, stated once: all
  agents are threads in one Weave process, and a lock protects cooperating
  tools. It cannot bind an external editor, another Weave instance, or an
  MCP server with its own file access. OS-level advisory file locks are
  deliberately not built: they cannot bind non-cooperating writers either,
  and the multiprocess caveat is already on record.
- *Duration* -- locks are held per operation (milliseconds, up to one
  subprocess run), **never per turn**. "This file is mine for the task" is
  *ownership*, and ownership is a **claim in the task ledger** (D63/D64):
  claim `file:<path>`, adjudicated by earliest `recorded_at`, visible and
  auditable -- exactly the shape task claims already have. A sandbox hook
  that consults claims as *dynamic* write policy (deny a write to a path
  claimed by another agent, alongside the static `file_permissions`
  narrowing) is recorded as an option, not built (§22 q8).
- *Lost updates at the reasoning level* -- `edit_file`'s anchors already
  catch the common case. If blind `write_file` overwrites ever bite in
  practice, the remedy is a CAS precondition (optional expected-digest
  argument), not longer lock holds.

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

## 15. Orchestration

The orchestrator is **already built and the design is right.** An orchestrator
*is* a Silk Agent (`nodes/orchestrator.py`) with one extra `workers` input;
delegation is registered as an ordinary tool on the agent's own toolset
(`functions/orchestrator.py`), which is why it inherits hooks, role
enforcement, `tool_events` observability and gate-ability for free rather than
needing a parallel plumbing stack. `run_subagent` reuses the whole
loop/toolset/role/reflection stack for the child run. Depth caps and a
delegation-chain cycle guard are in place.

Nothing below changes that shape. What follows is the gap between the design
and what `delegate_parallel` actually does once N > 1 -- which is where
"direct numerous subagents" lives.

**D52. `delegate_parallel` is unsound with N > 1 today: four independent
defects, all silent.**

1. **A same-worker fan-out fails.** `_run_one` writes the depth and chain onto
   `spec.toolset` (`functions/orchestrator.py:231-234`) -- a **shared, live**
   object owned by the graph. Two assignments naming the same worker race on
   those attributes, and then `RoleBinding.activate` refuses the second
   (`functions/subagent.py:165`) with "toolset already bound". The model gets
   `ok=False` for an entirely reasonable request -- *run `researcher` on topic
   A and on topic B*. The module docstring's "workers with distinct toolsets
   don't clash" is a precondition that nothing checks and the model cannot
   see.
2. **The depth and chain leak.** Those attributes are set on the child toolset
   and never cleared -- there is no `finally`. A worker's toolset keeps
   `_delegation_depth = 1` after the run, so its next use starts pre-charged,
   and a worker later run as a top-level orchestrator can refuse to delegate
   at all. Run-scoped state on a graph-scoped object, which is the same
   lifetime error D49 avoids for the decision seam.
3. **The assignment list is silently truncated.** `items = [...][:_MAX_PARALLEL]`
   with `_MAX_PARALLEL = 8` (`:78`, `:356`). Twelve assignments become eight;
   the reply is `ok=True` with "8/8 delegations succeeded". The model is told
   everything ran. This is D43's failure shape exactly -- a cap that discards
   instead of reporting -- and the fix is the same: refuse or report, never
   drop.
4. **The shared budget is not thread-safe.** `UsageLimits`
   (`functions/usage_limits.py`) is a plain dataclass with `+=` counters and
   *separate* `check_*` and `record_*` calls; it imports no lock. One instance
   is threaded into N concurrent workers, so check-then-record is a TOCTOU
   race: several workers pass the same check and collectively overrun the cap.
   The "one global cap for a fan-out" the docstring promises fails in the only
   case it exists for. Note T3 assumed this part worked and asked only for
   *sub*-budgets; it is a correctness gap, not an ergonomics one.

**D53. Sequence `delegate_parallel` behind D43; until then it is a sequential
loop.** A fan-out is the worst case for the stream-truncation bug: N workers
issue N interleaved requests against one shared server, where
`interrupt_requests=True` means each new worker's request truncates the
previous worker's in-flight stream, and the truncation reports as a clean stop
(D43). Every worker also gets a fresh `session_id`
(`functions/subagent.py:177`), so contention is 100% and prefix reuse is zero
(D47).

The irony is the point: **the tool that most needs concurrency is the one
place Silk has none.** `llama_outer_lock` serializes the requests anyway
(D43), so running the assignments in a `ThreadPoolExecutor` buys no model
throughput -- it buys only the interleaving that corrupts them.

*Ruling:* until D43 lands, `delegate_parallel` runs its assignments
**sequentially**. Identical results, no truncation, no measurable cost,
because the server was serializing them regardless. It becomes genuinely
concurrent when there is more than one backend to be concurrent *across*
(D45, D47 mechanism C) -- at which point the fan-out width should be bounded
by backend count, not by a constant.

**D54. The orchestrator throws away the two hooks `run_subagent` already
offers.** `run_subagent` accepts `on_event` and `should_stop`
(`functions/subagent.py:125-126`, polled at `:199-202`). `_run_one` passes
neither (`:236`). Two consequences:

- **No live observability.** Worker events are dropped on the floor. During a
  long fan-out the orchestrator's `tool_events` shows one `delegate` call and
  then nothing, for minutes; the trace is reconstructed only at the end from
  `tools_used`. For *numerous* subagents this is the difference between a
  progress view and a hang.
- **Stop does not propagate.** Stop sets the *orchestrator's* engine flag; the
  workers keep running to completion inside `pool.map`. This is **G8's most
  severe instance** -- a fan-out of eight long workers is uninterruptible, and
  no timeout bounds it.

*Required:* pass `on_event`, re-emitting worker events onto the orchestrator's
`_stream_event` path tagged with the worker name and the delegation's
`correlation_id` (which `DelegateResult` already carries); and pass
`should_stop` bound to the orchestrator's cancel check. Both parameters exist
and are already honoured by the runner -- this is wiring, not design. The
typed vocabulary (D2) gains a `worker` field so a nested event is
attributable.

**D55. One delegation-depth default (closes T5).** The runtime defaults
`max_depth=1` (`attach_orchestrator_tools`) while the node ships
`DELEGATION_MAX_DEPTH = 2` -- two defaults for one concept. The node's value
wins, becomes an **editable port** so the graph shows it, and the runtime
default follows it. Low-stakes and reversible; it is recorded only so the
divergence stops being re-discovered.

**D56. The approval seam already lands in the right place for delegation --
by construction.** `delegate` is `risk="medium"` and passes through
`wrap_tool_execute` like any other tool, on the **orchestrator's** toolbox. So
the gate fires on the orchestrator's own node UI, *before* the fan-out starts,
and asks the question a human can actually answer: approve this delegation,
not each of a worker's internal calls. Meanwhile a worker's own gated tools
deny (D36, D48: a subagent has no node UI), reachable only through a durable
grant.

This is a property of "delegation is a tool", not something to build -- worth
recording because the obvious alternative, plumbing approval down into
workers, would be strictly worse and looks superficially more thorough.

**D57. Fan-out results are the dominant context-growth term in an orchestrator
run, so the spill hook must cover them first.** `delegate_parallel` returns
every worker's full answer inline into the orchestrator's context: eight
workers times a long answer is the single largest tool result Silk can
produce, and it arrives in one round. Per D41 that makes it the highest-value
target for the spill hook (option A) -- write the full answers to files,
leave a per-worker head/tail preview plus paths in the model-visible result --
and it is prefix-preserving, so it defers compaction rather than triggering
it. An orchestrator that compacts is an orchestrator paying the double prefill
of D41 at the worst possible moment: mid-fan-out.

---

## 16. Observability & HITL surfaces

Three distinct problems hide inside "how do I see what my agents are doing":
**progress** (task planning and its advance, across N agents), **turn-boundary
decisions** (sign-off), and **mid-run decisions** (the D48 gate). Three rules
assign each its surface, and none of the three surfaces is interchangeable:

1. *Truth lives in the store.* Streams are lossy hints (`emit_stream` is
   "previews, never truth"; delivery is gated and never replayed), so any
   progress view must be a **projection of the plan store**, with events used
   only as refresh triggers. The Plan Viewer already works this way; nothing
   new may work otherwise. (ARCHITECTURE_REVIEW.md, R1/P1.)
2. *The turn-boundary rule* (I12) decides node vs. not-node.
3. *Weave's topology*: input ports are single-source
   (`engine/dataflow.py:393`), so N agents **cannot fan into one hub input**
   -- a hub must pull from storage, not collect wires. Output ports fan out
   freely (pulse dispatch walks every trace), so one hub can wake N agents.

**D58. The Task Hub node -- the multi-agent progress and sign-off surface,
and it is legal by I12.** A store-scanning projection node:

- **Inputs:** `roots` (`dirpath_list`) -- wired from `ToolBox.root_paths`,
  which is *already* the graph-wide aggregation point for sandbox roots, so
  the whole graph's plans arrive on **one wire**; `refresh` (`exec`) -- a
  timer pulse or any agent's `done`.
- **Body:** scans **all** `plan-*.db` under the roots (the store's discovery
  is newest-only today; a `scan_all()` is a small addition -- D60) and renders
  a kanban-style projection: one lane per plan, `claimed_by` as the per-task
  agent badge (the schema has carried it all along, unused by any view),
  `awaiting_signoff` highlighted. The store was *designed* for this --
  "multiple agents may share one plan", `BEGIN IMMEDIATE` + WAL -- the hub is
  the view that finally exercises it.
- **Sign-off actions:** because sign-off is a turn-boundary decision, the hub
  may host Approve/Reject for every listed plan (I12). A decision writes to
  the store with the human as actor (the revision log already records
  actors), then pulses `signed` -- one output port, fanned out to every
  agent's `run`. For the N-agent case this **absorbs the Sign-Off node**;
  whether the single-agent node remains as a convenience is open
  (§22 q6).
- **Outputs:** `plans_json` (digest -- the port's evaluated value is honest
  per rule 1), `pending` (count of open sign-offs *and* outstanding mid-run
  decision requests, from the D2 stream -- countable here, answerable only at
  the source per D59), `signed` (`exec`).

The hub never talks to a model, holds no run state, and works identically
however many agents run: it is a database viewer wearing a node costume --
which, unlike D51's dock-in-a-costume, is exactly what a node *is*: evaluate
inputs, produce values.

**Implemented (2026-09-02), minus the sign-off half.** `functions/task_board.py`
+ `nodes/task_hub.py`: `scan_roots` / `board` / `render_board` and a
`PendingDecisions` counter, with `roots`, `event`, `refresh` in and
`plans_json`, `pending` out. What is *not* built is the Approve/Reject
action and the `signed` port: D31-D33 deleted parked sign-off, so no
change waits in a row for a click -- a gated task change is decided
during the turn on the run's decision seam, and D59 reserves answering to
the asking node or its mirror. The hub counts (`pending`); it cannot
answer. That also settles §22 q6: the single-agent Sign-Off node does not
survive the hub because D32 had already deleted it.

**D59. Mid-run centralization is a dock, never a node.** With N agents, D48's
answering widgets are scattered across N node bodies. Centralizing them must
not reintroduce the rejected answerer node (D51), and does not have to --
Weave already ships the machinery:

- **Attention on the canvas.** An agent blocked on its `DecisionSeam` enters
  a visible waiting state and uses the existing pulse-glow animation
  (`weave/node/pulse_mixin.py` -- `heartbeat` exists, fittingly). The canvas
  itself is the "who needs me" dashboard. Zero new architecture.
- **A Decision Inbox dock.** Weave's NodePanel mirror system
  (`panel/mirror_contracts.py`) clones node widgets into docks **with
  action-signal forwarding -- mirrored buttons work**. The inbox is a dock
  listing mirrors of each waiting agent's D48 widget; clicking Approve in the
  dock *is* clicking it in the node -- same widget binding, same
  `seam.resolve()`. D51 is untouched: no wires, no graph channel, no
  rendezvous handle; a dock is main-thread UI like the log pane.
- **A session-scoped `DecisionRegistry`** feeds the dock: run-scoped seams
  register on `await_decision`, unregister on resolve/cancel/timeout. The
  registry holds weak references and run ids only -- it is a directory, not
  an owner, so seam lifetime (D49) is unchanged.

*Boundary restated:* the hub node (D58) may **count** mid-run requests; only
the asking node's UI -- or its mirror -- may **answer** one.

**D60. Identity plumbing -- what the surfaces above actually require.**

1. **The event envelope gains an `agent` field** (node title + node uuid).
   Today events carry `run_id` only; N independent agents' streams are
   indistinguishable once merged anywhere. D54's `worker` field covers
   *nested* attribution (orchestrator→worker); `agent` covers *top-level*
   attribution. Both, not either.
2. **`SqliteTaskStore.scan_all(roots)`** -- enumerate every `plan-*.db`
   under a root set, returning (path, plan) pairs; the existing discovery
   (newest-only, `_locate_db`) stays as the agent-side behaviour.
3. **Surface `claimed_by`** in projections (hub lanes, Plan Viewer badges).
   Schema support exists; no store change.

**D61. Orchestrator trees need no hub -- a wire suffices.** Once D54 lands
(worker events re-emitted on the orchestrator's own `tool_events`, tagged
`worker` + `correlation_id`), the orchestrator's event port *is* the
aggregated stream for its whole tree: one wire to one Plan Viewer / Hook
Monitor shows every worker. And workers sharing the orchestrator's working
directory share its plan DB (`claimed_by` distinguishes them). Recorded so
nobody builds a hub where a wire suffices: the Task Hub is for **independent
top-level agents**, whose streams share no port and never will (rule 3).

---

## 17. Persistence -- the Macrame ledger

Adopted for **task management** and **agent history**: `macrame-db`
(github.com/opticsWolf/Macrame) -- an embedded, single-file bitemporal graph
ledger on libSQL. Concepts (id/title/content) linked by typed, weighted
edges; two independent clocks per row (*valid time* -- when a fact held;
*transaction time* -- when the ledger learned it), addressable separately
(`as_of_valid`, `as_of_recorded`, `reconstruct`); branching with **merge
refused by doctrine**; DiskANN vector search + FTS5 + hybrid RRF; one Write
Actor serialising all writes; typed error taxonomy. The Python surface is
**synchronous with the GIL released** -- which is exactly what Silk needs,
since agents run on `ThreadedNode` workers with no event loop -- and writes
are sub-millisecond (assert_edge ~220 us, concept upsert ~193 us), so an
agent run writing a few hundred rows is noise.

The verdict is asymmetric: for history the fit is structural; for tasks the
ledger subsumes the existing store's semantics but forces one architectural
renegotiation (the sole-writer rule, D62).

### Why history fits structurally

- **I11 and Macrame's Doctrine III are the same invariant.** Silk: the
  model-visible prefix grows only at the tail; compaction is the single
  deliberate invalidation. Macrame: assertions are never rewritten, only
  *superseded*. So compaction (§12, D24/D25) stops being destructive: the
  engine's history-replace becomes a **supersession event in the ledger**,
  and the pre-compaction conversation stays addressable --
  `as_of_recorded(t)` answers *"what did the model actually see at round
  7"*, which is what the D41 measurement and the D42 tests want and cannot
  have while history lives in `self._history` on the node and dies with it.
- **History becomes memory, not storage.** `hybrid_search` (FTS5 + vectors,
  RRF) over past turns/runs is a `recall` tool on the ToolBox -- long-term
  agent memory across sessions, graph-linked: turn -> run -> task -> files
  touched -> agent. FTS5 works day one with no embedding model; vectors
  arrive when something produces embeddings (the GGUF pool can --
  llama.cpp serves embeddings). Side effect: G2's fake-BM25 gains a real
  ranked search to delegate to or be measured against.
- **Identity maps 1:1 onto D46/D60.** `agent:<uuid>`, `session:<id>`,
  `run:<run_id>`, `turn:<id>` as concepts; `IN_RUN`, `DELEGATED_TO`
  (run -> run, carrying D54's correlation), `CLAIMED_BY`, `TOUCHED`
  (run -> file) as edges. The identity plumbing D60 mandates for
  observability is the same plumbing the ledger wants as keys.

*Boundary:* the raw `tool_events` firehose does **not** go into the ledger
-- events are a log, not belief. T7's JSONL sink remains their home. The
ledger gets the distilled layer: turns, runs, task transitions, sign-offs,
compaction events.

### Why tasks fit -- and what it collides with

`SqliteTaskStore`'s semantics are a hand-rolled subset of what the ledger
gives natively: the revision log -> transaction time; `claimed_by` -> an
edge; parent pointers -> `SUBTASK_OF` / `DEPENDS_ON` as a real graph;
sign-off -> an assertion with the human as actor; status transitions ->
superseded `HAS_STATUS` edges, so *"what did the plan look like at 14:00"*
(`as_of_valid`) and *"what did we believe then"* (`as_of_recorded`) are one
call each. And branching maps onto fan-out exactly (D63).

The collision: the old store's sharing model is *"any process finds
`plan-*.db` by newest-file; SQLite file locking arbitrates"* -- explicitly
cross-process. Macrame is **one Write Actor, one process**; two processes
opening one ledger is outside its contract. For Silk today this is fine --
all agents are threads in one Weave process -- but it changes how D58's hub
reads (D62), and it resurfaces if Weave's multiprocess nodes ever host
agents. Stated once, here: **the ledger is a per-process resource.**

*Constraint sharpened (ruling):* the limit is **not** one agent per turn.
The Write Actor serialises writes *within* the process -- N agent threads
call write methods on the shared handle concurrently and the actor queues
them, sub-millisecond each. Concurrent agents writing is already safe;
what the architecture must manage is not write *access* but write
*discipline*: who asserts what, on which lineage, and how read-modify-write
races settle. That is D62-D66.

**D62. `LedgerRegistry` -- one handle per file, process-wide.** Qt-free, in
`functions/ledger.py`: a process-singleton mapping path -> open `Database`
handle, refcounted. Opening goes **only** through the registry -- nodes,
tools, the hub, the compactor never call `Database.open` themselves --
so double-open of a live ledger is impossible by construction: the
sole-writer rule enforced at the only place it can be. Filesystem
discovery (T4, D58) still *finds* ledger files; the registry answers "is
this one already open here" and hands back the shared handle. The registry
owns `close()` ordering (Macrame: close has exactly one owner): plugin
unload / graph close closes; no node or run ever does. A dead handle
surfaces as a typed `MacrameError` on the next call -- the same failure
surface as G6.

*Amendment to D58/D60(2):* with the ledger backend active, the Task Hub's
"scan" is **discovery of files + registry lookup of handles**, never a
second open of a live ledger. `scan_all()` as specified remains the
mechanism for the SQLite fallback backend (D66).

**D63. Lineage discipline -- main is durable belief; workers get branches.**

| Writer | Writes | Lineage |
|---|---|---|
| Top-level agent (own run) | its turns, run/session concepts, status transitions on tasks it claimed | main |
| Orchestrator | delegation edges; fork/abandon of worker branches | main + branch admin |
| Fan-out worker | everything it does | **its own branch** (`worker/<correlation_id>`, via `on_branch`) |
| Human (Task Hub / Sign-Off / decision UI) | sign-off assertions, goal revisions | main, high priority |
| Compactor | supersession events | main |
| Nobody | deletion -- archive path only (Macrame Doctrine V) | -- |

A worker forks its branch, works entirely on it, and its *result* returns
as an `AgentMessage` -- exactly as §15 already rules. What survives is
**promoted by re-assertion on main** (by the orchestrator, or by the human
via sign-off), never merged. This is not a workaround: Macrame *refuses*
merge by doctrine, and Silk independently decided worker results come back
as messages, never as merged state. The branch discipline wires two
independently-made identical calls together. An abandoned branch is one
transaction, full audit retained.

**D64. Read-modify-write: a lock for prevention, the ledger for
adjudication.** Both cheap because single-process is guaranteed by D62.

- *Prevention:* compound operations that must be atomic **as a decision**
  -- claim a task, park for sign-off, advance a status with a precondition
  -- go through the `TaskLedger` adapter, which holds a plain
  `threading.Lock` around read-check-assert. One process means one lock is
  *complete* correctness -- it replaces what `BEGIN IMMEDIATE` did
  cross-process for the old store. Multi-row writes use
  `write_bulk_atomic`.
- *Adjudication:* if a race slips past (or two claims arrive by design),
  do not prevent -- **record both and let the ledger arbitrate**: winner =
  earliest `recorded_at`; the loser's claim stays in history as an audited
  near-miss. Under Doctrine III a conflict is evidence, not corruption.

**D65. Turn-shaped writes.** Per run: assert the `run` concept at start;
stream single asserts for genuinely discrete facts (status transition,
claim, sign-off request -- the things `plan_changed_event` fires on); batch
the turn's bookkeeping (turn concept, `IN_RUN` / `TOUCHED` / `USED` edges)
into **one `write_bulk_atomic` at turn end**, from the same hook that emits
`run_finished`. Keeps the hot loop at ~2-3 ledger calls per turn; keeps a
turn atomic in transaction time (a turn either happened or did not -- which
is what makes `as_of_recorded` round-replay clean); keeps the event
firehose out (T7).

**D66. Seam and fallback.** Nodes and tools never import `macrame` -- only
`functions/ledger.py` does. `TaskLedger` implements the **existing
task-store protocol** (`Plan` / `Task` / ops, `plan_to_json`,
`plan_changed_event`), so the Plan Viewer, Sign-Off flow and the D58 hub do
not change; `HistoryLedger` adds turns/runs and the `recall` search used by
the memory tool. `macrame-db` is a **declared optional extra** -- Silk's
first declared binary dependency, which forces the G5 fix as a
precondition rather than a lingering gap. Absent, `SqliteTaskStore` remains
the backend and history stays in-node: the graph degrades to today's
behaviour, loudly (one log line), never silently.

```
agents / orchestrator / hub / human UI
        |  (task-store protocol . recall API)
   TaskLedger . HistoryLedger        <- RMW lock, turn batching, identity stamps
        |
   LedgerRegistry (1 handle/file)    <- sole-writer rule lives here
        |
   macrame Write Actor               <- Macrame's own serialisation, priority queues
```

Net: Silk adds **no locking for write throughput** (Macrame's actor is the
serialiser), one small lock for *decisions*, one registry for *ownership*
-- and gets branch-isolated workers, bitemporal plan audit, promoted-by-
assertion sign-off, and cross-session searchable memory in exchange.

*Placement (default, revisable):* one **task ledger per sandbox root**
(working dir), preserving T4/D58 discovery. Whether *history* shares that
file or lives in a per-user memory ledger (`~/.weave/silk/memory.db`) for
cross-project recall is open (§22 q7).

---

## 18. Graph authoring -- the agent places nodes

A tool family that lets an agent **build graph** -- place nodes on the canvas
and connect them -- from a user-editable whitelist. This is the first Silk
tool whose effect is on *Weave itself* rather than on files, a model, or a
task store, and that changes what has to be true before it runs.

**D69. Six tools, two of them read-only, all on one whitelist.** Placement
without inspection is blind: an agent that cannot see the graph cannot place
a node *relative* to it, cannot reuse an existing node, and cannot know which
ports are free. So the family is:

| Tool | Risk | What it does |
|---|---|---|
| `list_placeable_nodes` | low | the whitelist, rendered as the model needs it: class name, display name, description, category, input/output ports with datatypes -- all of it already in `NODE_REGISTRY` metadata (`registry/metadata.py`) and each node's port list |
| `describe_graph` | low | current nodes (id, class, title, position, ports, which are connected) and edges as `(src_id, src_port, dst_id, dst_port)` -- the same tuple shape the undo commands speak |
| `place_node` | medium | instantiate a whitelisted class at a position, optional title; returns the new node id |
| `connect` | medium | one edge, validated by the port type system before it is attempted |
| `disconnect` | high | remove one edge |
| `remove_node` | high | remove a node **created in this run** (D73) |

Two design notes. *The port type system does the hard part for free*:
`PortType.can_connect_from` consults `PortRegistry`'s converter registry
(`node/port_registry.py`), so an illegal connection is refused with a typed
reason the model can act on ("`silk_toolbox` does not connect to a
`silk_toolset` input") rather than producing a broken graph. And *the
descriptions the model reads already exist* -- `node_description`,
`node_tags`, port `datatype` and `port_description` are written for humans in
the node UI; nothing new has to be authored for the model.

Deliberately **not** in v1: setting widget values, moving/resizing nodes,
saving or loading graph files, creating nodes *not* on the whitelist,
and anything touching another graph. Setting widget values is the obvious
next request and is left out on purpose -- it is how a placed node becomes
*configured*, which is a much larger surface (every widget type) and wants
its own decision.

**D70. The tool runs on a worker thread and the canvas is main-thread-only,
so it needs the same seam as D49 -- generalised.** A tool executes inside
`ToolBox.execute_tool_calls_async` on the agent's `ThreadedNode` worker
(under `asyncio.to_thread`); every canvas mutation must happen on the Qt main
thread. The existing worker->main channels are one-way and return nothing
(`emit_stream`, `pulse`), which is exactly the gap D49 filled for human
decisions.

*Ruling:* generalise `DecisionSeam` into a **`MainThreadCall` seam** with the
same mechanics and the same ordering rule -- a queued signal carries the
request to the main thread, the main thread performs the mutation and writes
the outcome **under the lock before setting the event**, the blocked worker
re-reads under the lock. The human-decision seam (D48-D50) and this one are
then the same object with a different resolver: a person, or the main-thread
canvas. Same timeout discipline, same cancel path, same fail-closed default
(D36) -- a request that cannot be delivered (no canvas, headless run, graph
closing) **denies**, it does not hang.

This is the second user of D49's machinery, and it is the reason D49 was
specified as a general waiter rather than an approval-specific one. Build the
seam once, in `functions/`; the Qt resolver lives in the node layer.

**D71. The whitelist is a node-level, user-editable allow-list, and the
default is empty.** The tool pack mounts on the **ToolBox node** (which is
where sandbox roots and toolchains already live -- the graph-wide capability
surface), with a checkable tree of registered node classes grouped by
category, the same widget the tool tree already uses. Rules:

- **Default-deny.** An empty whitelist means the tools register but every
  placement is refused, and `list_placeable_nodes` returns nothing. There is
  no "allow all" checkbox; selecting every entry is possible but must be a
  deliberate act, not a default. Same reasoning as I6's monotone narrowing:
  the safe state is the one you get by doing nothing.
- **Whitelisting is by class name**, resolved against `NODE_REGISTRY` at
  build time, and a whitelisted class that is no longer registered is
  reported at ToolBox evaluation -- visibly, in the node -- rather than at
  agent run time.
- **The whitelist narrows like everything else** (I6): a ToolSet or Role may
  remove entries, never add. The Role's existing tool selector already gates
  the six tools as a group; per-class narrowing rides on the same grant model.
- **It travels as part of the ToolBox recipe**, so it is visible in the saved
  graph and shareable in a preset -- unlike grants (D35) it carries no secret
  and no filesystem authority.

**D72. Every mutation is one undoable command, and that is the primary safety
property.** Placements and connections go through the existing undo
commands -- `AddNodeCommand`, `AddConnectionCommand`, `RemoveNodesCommand`,
`RemoveConnectionsCommand` (`canvas/undo_commands.py`) -- pushed onto the
canvas's own `UndoManager`, never through raw scene manipulation. A tool call
that creates several items pushes **one `CompoundCommand`**, so one Ctrl+Z
undoes one *tool call*, not one primitive.

The consequence is the point: **the human can undo the agent's graph edits
with the same gesture they undo their own**, and the agent's work appears in
the same history. An agent whose edits could not be undone would be a
different, much more dangerous tool -- and would also be *invisible*, since
the undo stack is where a user looks to see what happened.

**D73. Approval, and the self-modification guard.**

- `place_node` and `connect` are `risk="medium"`; `disconnect` and
  `remove_node` are `risk="high"` and `requires_approval=True`. The approval
  gate (D30-D38, D48) already covers them -- the request renders in the
  Agent node's own stream UI, where the human can see the graph the change
  applies to. No new approval machinery.
- **Destructive calls are scoped to the run.** `remove_node` and `disconnect`
  operate only on nodes and edges **created by this agent, in this run** --
  tracked in the run's own record of what it placed. An agent may clean up
  after itself; it may not prune the user's graph. Widening that is a
  separate decision, and would need a much stronger approval story.
- **The agent may not modify its own execution path.** Refuse any mutation
  touching the Agent node itself, its ToolBox / ToolSet / Role / model chain,
  or any node upstream of it. The graph that *is running the agent* is not
  material the agent edits mid-run: the evaluation model gives no coherent
  meaning to rewiring a node's own inputs while it sits inside `compute()`,
  and this is the same class of objection as D51's. The check is a cheap
  upstream walk from the Agent node at request time.

**D74. What this composes with.** The tools are ordinary `ToolBox`
registrations, so they inherit hooks, `tool_events`, role enforcement and the
gate exactly like every other tool (the D56 property). Consequences worth
naming: a graph-authoring call appears in the Hook Monitor like any tool
call; a delegated worker inherits the whitelist through the toolset it was
given, and its placements are gated by D36's deny-without-a-UI rule; and the
Macrame history ledger (§17) records placements as ordinary run facts, so
*"what did the agent build, and when"* is answerable after the fact.

Interaction with Weave's hot-load work (`docs/HOT_RELOAD_PLAN.md`): the
whitelist is resolved through `NODE_REGISTRY`, which that plan gives a
`generation` counter and change notification. The ToolBox node should
re-resolve its whitelist on registry change, so a hot-loaded plugin's nodes
become placeable without rebuilding the graph.

---

## 19. Self-modification -- the agent extends Weave

§18 let the agent *use* Weave's parts. This section is the agent *making*
new ones: writing node and widget code, verifying it, loading it into the
running session -- and, when the change is one no reload can absorb, asking
for Weave to be restarted. It is a small addition to the tool surface and a
large addition to the risk surface, which is why the two are specified
together. Weave's half is `docs/HOT_RELOAD_PLAN.md` §3.10-§3.11; this is
Silk's.

**D75. The loop already exists except for one verb.** *Write* is the file
tools under the sandbox (D14-D18). *Verify* is the toolchain runner -- ruff,
mypy, pytest -- already there and already sequenced. *Observe* is `§18`'s
`describe_graph`. The only missing step is **load**: `list_suites`,
`load_suite`, `reload_suite`, registered as ordinary `ToolBox` tools so they
inherit hooks, `tool_events`, role enforcement and the approval gate for free
(the D56 property). Do not build a "self-improvement subsystem"; build the
missing verb and let composition do the rest. The composition *is* the
feature: an agent that writes a node can then place it and wire it (§18)
without a single new mechanism between the two.

**D76. The agent writes plugins into its own root, and never into the code
that is running it.** A dedicated user plugin directory
(`~/.weave/plugins/<name>/`, Weave's open question 5) is registered as a
sandbox root and is the *only* place `load_suite` will look for
agent-authored code. `weave/` core, `weave/plugins/silk/` and the virtualenv
are not writable, enforced by the existing static `file_permissions`
narrowing rather than by a new mechanism.

This is **D73's self-modification guard moved from the graph to the
filesystem**: the same rule -- the agent does not edit its own execution path
-- for the same reason, one layer down. The consequence is worth stating
plainly rather than discovering later: **Silk improving Silk is out of scope
in v1.** That needs review-then-relaunch and has a bootstrapping problem the
graph case does not (the code that would review the change is the code being
changed). Recorded as T10.

**D77. `import` is an execution boundary the sandbox does not cross -- so
loading is always high-risk, always approved, and never narrowable away.**

The blunt version: every file tool Silk has is sandboxed; **`import` is not
sandboxable**. Module-level code in an agent-authored file runs with the full
authority of the Weave process -- the network, the entire filesystem, the
user's keys -- no matter how narrow the sandbox was while that file was being
written. Write authority over a directory on the import path *is* process
authority, deferred by exactly one tool call. Everything else in this section
follows from that sentence.

- `load_suite` / `reload_suite` are `risk="high"` and
  `requires_approval=True` **always**, and no Role, preset or grant may
  pre-authorise them. Note this is not I6: I6 makes narrowing monotone, a
  ceiling nobody may raise. This is a **floor** nobody may lower. An
  "always approve" that a preset can switch off is not a control.
- **The request shows the code, not the name.** The approval renders the file
  list with sizes and mtimes, plus the diff for every file this run touched.
  A human approving "load `my_nodes`" has approved nothing; a human approving
  a 40-line diff has approved something.
- **Validation is execution too.** Weave's dry-run import (§3.6 there) runs
  top-level code, so for machine-authored code it happens *after* approval and
  in a **subprocess** (Weave §3.10) -- a segfault, a hang or a stray thread in
  generated code then costs one tool call instead of the session.
- Denial is cheap and normal: a refused load returns `denied`, the files stay
  on disk, and the agent can report what it built. Fail-closed (D36) applies
  unchanged -- no UI, no load.

**D78. Version discipline belongs in the verify step, and the linter is what
enforces it.** An agent editing an existing node class changes the shape of
state held in *users' saved graphs*. Weave's WV520 / WV521 / WV522 and the
committed state manifest (`HOT_RELOAD_PLAN.md` §5) are precisely that check,
so `weave_lint` joins ruff and mypy as a `CommandSpec` in the default
toolchain, and a WV521 finding is a **hard stop before load**: the agent
either bumps `node_state_api` and writes `migrate_state`, or leaves the class
alone and ships a new one with `node_supersedes`. This is where those rules
earn their keep -- a human author gets code review; an agent author gets the
linter, and nothing else.

**D79. Relaunch is a request at a turn boundary, never an action mid-run
(I12).** `request_relaunch(reason)` returns `queued` immediately; the run
finishes normally -- final message, ledger flush, task-store write -- and only
then is the human asked, with the reason and the pending changes in view. Two
consequences:

- **The agent cannot observe the far side.** The new process is a new session,
  a new run, empty memory. Anything the agent wants to survive the restart
  must already be in the ledger or the task store *before* it asks (§17).
  "Continue after the restart" is a task record with a claim, not a promise
  the runtime keeps -- and this is the sharpest argument yet for the ledger
  being durable belief (D63) rather than a cache.
- **A relaunch requested while other agents run is a queue, not a kill.**
  Orchestrated workers finish; the request waits for the last turn boundary in
  the graph, or it is refused with the list of what is still running. Either
  way the wait is visible (D53's legibility rule) -- a restart that silently
  waits looks like a restart that silently failed.

**D80. Silk's share of the release protocol** (Weave §3.11 step 5): what must
be let go **before** the child process spawns, and why overlap is not
survivable.

| Resource | Why it cannot overlap |
|---|---|
| `GGUFModelPool` | it is a `python -m llama_cpp.server` **child process** holding model weights and a port (`functions/model_pool.py`); two live pools mean double VRAM and an orphan that outlives its parent. `cleanup()` already exists and is the release hook. |
| Macrame ledger handles (D62) | one write actor per process is the entire concurrency model; two processes on one file breaks D64's earliest-`recorded_at` adjudication and Doctrine III's supersession order. Close the `LedgerRegistry` handles explicitly -- never rely on GC during interpreter teardown. |
| MCP server subprocesses (§10) | started per connect and re-resolving credentials at connect (D22), so the child reconnects cleanly -- provided the parent's servers are actually stopped. |
| File write locks (D67) | advisory and per-process, so they simply do not span the handoff. That is not a defect: it is the same reason durable ownership is a **ledger claim** and not a lock (D68). |

Distinct from **G6**: that is restarting a *dead model server* mid-run, and it
stays out of scope. This is releasing a healthy one on the way out.

**D81. The failure mode is a boot loop, and the guard needs a second half.**
Auto-load plus agent-authored code is an application that can fail to start.
Weave's loop guard (§3.11) covers the first half: start clean, quarantine the
suspect suite, name it in a visible report. Silk owes the other half -- **the
quarantine outcome is written to the task store as a fact**, so the next run's
agent is told *"your plugin `x` was quarantined after crashing on load, here
is the traceback"* instead of silently discovering that its work evaporated. A
self-improving loop with no feedback on failure does not improve; it repeats.

---

## 20. Phasing

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
9. **Make `delegate_parallel` honest** (D52, D53): run assignments
   sequentially until D43 lands, refuse-or-report instead of truncating the
   assignment list, clear the depth/chain attributes in a `finally`, reject a
   same-worker fan-out with an error the model can act on, and put a lock
   around `UsageLimits`. Four small fixes to shipped code that is currently
   silently wrong; none of them waits on anything else.
10. Thread `on_event` and `should_stop` through `_run_one` (D54) — the
    parameters already exist, so this is what makes a fan-out observable and
    interruptible.
11. The `agent` identity field on the event envelope (D60(1)) — stamped
    where `run_id` is stamped today; lands with D54's `worker` field so the
    vocabulary changes once, not twice.
12. The per-root write gate + `writes_files` flag on `CommandSpec` (D67
    tier 2, closes G19): small, and it is the only thing standing between
    two concurrent agents and an unserialised formatter-vs-edit race.

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
   compaction is triggered at all (D41); covering `delegate_parallel` results
   first, since they are the largest single result Silk produces (D57).
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
0. Delegation depth as an editable port, one default (D55, closes T5).
1. File access as an explicit narrowing-only port; Pydantic grant model
    (D16–D18).
2. Discovery: `search_tools`, per-tool deferral, auto-load (D4–D6).
3. MCP Node + Aggregator (D19–D22).
4. Task Node with explicit plan identity (D23).
5. Task Hub node (D58): `scan_all` on the store (D60(2)), the kanban
    projection with `claimed_by` lanes, sign-off actions, `signed` fan-out.
6. Decision Inbox dock + `DecisionRegistry` + blocked-on-decision canvas
    state (D59) — after the D48/D49 seam exists (Phase 2), since it mirrors
    that seam's widget.
7. `functions/ledger.py` (§17, D62–D66): `LedgerRegistry`, `TaskLedger`
    behind the existing task-store protocol, `HistoryLedger` + `recall`
    tool (FTS5 first). Preceded by declaring dependencies (G5) — the
    ledger is Silk's first declared binary dependency.
8. Graph authoring (§18, D69–D74): the `MainThreadCall` seam generalised
    from D49, the whitelist widget on the ToolBox node, the six tools, the
    self-modification guard. Strictly after the Phase 2 seam — it is the
    seam's second user, not its first.
9. Self-modification (§19, D75–D81): the three load verbs, the user plugin
    root, the always-approve floor with its diff-carrying request,
    `weave_lint` in the toolchain, `request_relaunch` + the release
    participants, the quarantine fact. Gated on Weave shipping
    `HOT_RELOAD_PLAN.md` Phases 1-2 and 6 -- until a load is lossless and
    reportable there is nothing safe to call. Everything except the load verb
    exists already (D75), so the Silk-side cost is mostly the approval
    surface, not the plumbing.

**Later:** embeddings for `recall` (vector half of §17 — needs an
embedding producer; the GGUF pool can serve one); nested budgets (D26);
BM25 or its removal (G2); the unwired-event
dispositions (D15); the D47 mechanisms not selected by the measurement --
kept described rather than deleted, since the rule that skips one today
selects it as soon as the graph shape changes.

---

## 21. Gaps this closes

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
name), T5 (delegation depth), T6 (HTML floor), T7 (durable event sink).

**G12 (version metadata) gains a second reason and stays cheap.** §19 makes
suite identity operational rather than cosmetic: a reload report, a
quarantine record and a "which build wrote this ledger assertion" question all
need a `__version__` to name. Still trivial; now load-bearing.

---

## 22. Open questions

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
5. Whether the hook-node question returns once per-tool binding exists --
   D12 is a "not yet", not a "never".
6. ~~Whether the single-agent Sign-Off node survives once the Task Hub
   (D58) absorbs its role for N agents.~~ **Answered (2026-09-02)** by the
   hub being built: there is no node to keep. D32 deleted `signoff_node.py`
   with the rest of the parked-state machinery, and the hub inherited the
   *viewing* half only -- nothing is parked, so nothing is signed off
   outside the turn that asks.
7. Ledger placement for *history* (§17): share the per-root task ledger, or
   a per-user memory ledger (`~/.weave/silk/memory.db`) so `recall` spans
   projects. Task ledgers are per sandbox root either way. Related: which
   embedding model stamps turn vectors, and whether embedding versioning
   (Macrame stores per-model tables) tracks the pool's loaded model.
8. Whether the sandbox consults ledger claims as *dynamic* write policy
   (D68) -- deny writes to paths another agent has claimed -- and whether a
   claim then needs a release path and a timeout, which is approval-gate
   territory (D38) rather than lock territory.
9. Whether graph authoring (§18) ever gains **widget configuration** -- an
   agent that can place a node but not set its values builds skeletons a
   human must finish. The surface is every widget type, so it needs its own
   decision rather than an extension of D69; the likely shape is a narrow
   typed setter over `WidgetCore` bindings, whitelisted per node class the
   way the classes themselves are.
10. Whether a suite the agent wrote should **auto-load at the next start**
   once a human has approved it once, or require approval every session. Once
   is the usable answer and the one that makes D81's quarantine necessary;
   every-session is the safe answer and makes the loop tedious enough that
   nobody will use it. The middle -- approve once, pin the file digests, and
   re-ask when they change -- is probably right and needs a place to store the
   pins that is not a preset (D35's reasoning applies: this is authority, not
   configuration).
11. Whether an agent may read the **quarantine traceback** (D81) and attempt a
   fix unprompted, or whether a crashed load ends the loop until a human says
   continue. Related to q10: an agent that can auto-load *and* auto-retry is
   the configuration in which a self-improving loop runs unattended, which is
   exactly when it should not.
