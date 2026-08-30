# Silk × Weave — Architecture Review

*2026-08-31. Scope: the shipped Silk codebase (`functions/`, `nodes/`, `widgets/`),
`docs/DESIGN_SPEC_DRAFT.md` (D1–D57), `docs/OPEN_TOPICS.md` (G1–G18, T1–T8), and the
Weave substrate Silk stands on (`weave/node/`, `weave/engine/`, `weave/widgetcore/`).
Method: code-verified — every claim cites a file, and nothing below is repeated from
the spec unless the review adds a judgement about it.*

This is a review of **fit**: does Silk use Weave the way Weave is built to be used,
where does it bend the substrate, and which bends are load-bearing design versus
accumulated drift. The output is a set of findings (R1–R14) and a short list of
candidate rules (§6) that would make the combined architecture consistent enough
that the next node is written by pattern-matching, not archaeology.

---

## 1. The substrate: what Weave actually provides

Silk's design space is defined by five Weave facts. Everything else in this review
derives from them.

**1. One evaluation model.** Lazy pull with dirty propagation, cycle-tolerant via a
per-request visited set, atomic cache swaps (`engine/dataflow.py`). Inputs are
gathered **once, before `compute()`** — an unconnected port falls back to its bound
widget's value (`dataflow.py:404`). Nothing can arrive at a node mid-compute
through the graph. This single fact decided D30, D48, D51, and it decides R6 below.

**2. Three channels, three semantics.** Weave is not one wire type wearing three hats
— it is three genuinely different transports:

| Channel | Direction | Semantics | Contract |
|---|---|---|---|
| **Evaluated data** | pull, upstream→down | cached truth | deterministic, replayable, serialized shape |
| **Stream** (`emit_stream`) | push, worker→main→down | *"previews, never truth"* (`node/threaded.py` module doc) | throttleable, droppable, gated by `_is_computing`/`_is_restoring`, canonical value arrives later via evaluation |
| **Pulse** (`pulse`) | push, edge event | trigger + optional payload | never throttled, main-thread, wakes manual nodes (`node/base.py:812`) |

**3. A real port type system.** `PortRegistry` (`node/port_registry.py`) gives types
identity (stable IDs, persisted for customs ≥200), visuals (theme palette index),
validation, and **casting with inheritance** — a directed converter registry that
canvas connect-validation consults (`PortType.can_connect_from`). Types are cheap
to add and Silk added nine.

**4. Widget bindings with roles.** `PortRole` (`widgetcore/port_models.py`):
INPUT / OUTPUT / **BIDIRECTIONAL** (shows incoming data, provides the default when
unconnected) / DISPLAY (never a port) / INTERNAL (persisted, never a port). The
BIDIRECTIONAL fallback is wired directly into input gathering (`dataflow.py:359`).

**5. Headless parity.** `DataFlowCore` is Qt-free; `HeadlessNode`
(`engine/headless.py`) runs the identical compute semantics with no Qt at all.
Parity exists *by construction* at the engine layer — whether a plugin benefits
depends entirely on how much of its logic lives above that line.

---

## 2. How Silk maps onto it

```
Qt shell        nodes/ (15)  +  widgets/ (5)
                 │  ports, WidgetCore bindings, emit_stream/pulse plumbing
────────────────┼──────────────────────────────────────────────────────
Qt-free core    functions/ (29 modules + tools/)
                 ToolBox → ToolSet → RoleBinding → AgentLoop
                 model_pool / graph_engine │ hooks │ task_store │ orchestrator
```

Verified: **no module under `functions/` imports PySide6** (grep over real import
statements, not docstrings). The layer boundary the architecture docs claim is the
layer boundary the code has. This is the plugin's single most valuable structural
property — every finding below is written so as not to erode it.

The node suite splits cleanly into four families:

- **Assembly** (config → handle): GGUF Loader, ToolBox, ToolSet, Toolchain, Role,
  Inference Settings, Agent Spec. All emit live or plain-data handles on custom
  port types; all `ActiveNode` or manual-threaded; presets everywhere
  (JSON + pydantic + shared `PresetBarWidget`) — a genuinely consistent idiom.
- **Execution**: Agent, Orchestrator (subclass of Agent — delegation arrives as
  ordinary tools, D56).
- **Observability sinks**: Hook Monitor, Pool Monitor, Plan Viewer, Chat Display —
  all consume the stream/pulse channels, none feeds truth back into evaluation.
- **Human surface**: Sign-Off node (turn-boundary), plus — per D48 — the Agent
  node's own stream UI (mid-run). See R6.

Nine custom port types (`nodes/silk_ports.py`): `gguf_model`, `silk_toolbox`,
`silk_toolset`, `silk_role`, `file_permissions`, `dirpath_list`, `toolchains`,
`agent_message`, `silk_agents`.

---

## 3. What is right and must be preserved

Stated first so the findings don't read as "rewrite it".

**S1. The type system encodes a security boundary.** `silk_toolbox` and
`silk_toolset` are *different port types*, and the Agent accepts only the latter.
"The full registry can't reach a model by accident" is enforced by canvas
connect-validation, before any code runs. This is the best kind of safety: the
illegal graph cannot be drawn. Same pattern: sandbox roots are the hard ceiling at
the ToolBox; ToolSet/Role can only narrow (`role_denied` at dispatch). A monotone
capability lattice, visible as wires.

**S2. Delegation is a tool, not a subsystem.** The Orchestrator is an Agent plus a
`workers` input; `delegate`/`delegate_parallel`/`list_workers` are ordinary tools on
its own toolset, so hooks, role enforcement, `tool_events`, and the approval gate
(D56) apply for free. The *implementation* of the fan-out is unsound today (D52),
but the shape is exactly right.

**S3. `AgentMessage` is plain data on the wire.** The A2A envelope
(`functions/messaging.py`) rides edges as a dict — inspectable, serializable,
correlation-carrying. This is the standard that R4 measures the live-handle ports
against.

**S4. Chain-accumulator idiom.** Toolchain → toolchains, Agent Spec → agents:
build a list down a chain of same-typed ports. One idiom, used twice, documented
in both docstrings, mirroring `weave.library` conventions. Cheap to learn.

**S5. Dedup discipline on streams.** Every streamed event carries
`run_id` + monotonic `seq` (`nodes/agent.py:352-364`) — the node authors already
treat stream delivery as at-least-once/lossy and gave consumers a dedup key. The
substrate's honesty is matched by the plugin's.

**S6. Qt-free core, verified.** See §2. Also the guarded, idempotent port
registration in `silk_ports.py` (correct instinct — see R9 for the caveat).

---

## 4. Findings

Each finding: observation → evidence → consequence → recommendation. Ordered by
how much architecture hangs on them, not by severity of any single bug.

### R1. Silk's event ports are a fourth channel that pretends to be the first

**Observation.** `tool_events` and `plan_events` are declared as ordinary `dict`
output ports (`nodes/agent.py:128-131`), but their traffic flows **only** on the
stream channel (`emit_stream(..., throttle_ms=0)` at `nodes/agent.py:360`), and the
canonical value never exists: `compute()`'s return dict contains `response`,
`chat_turn`, `outbox` — no `tool_events`, no `plan_events` (`nodes/agent.py:528`).
Consumers know this: Hook Monitor, Plan Viewer, and Sign-Off all override
`on_upstream_stream` and their `compute()` bodies note "tool_events itself arrives
via on_upstream_stream above" (`hook_monitor.py:172`).

**Consequence.** Three separate costs:

1. *The port lies to the evaluator.* Pull-evaluating `tool_events` yields nothing.
   A user who wires it into any non-stream-aware node (a Write node, a dict
   display) gets `None` and no explanation. The wire looks like a data wire, is
   colored like a data wire, and isn't one.
2. *Delivery is lossy by design for anyone not present.* Streams are gated by
   `_is_computing`, dropped on teardown, and never replayed. A monitor added
   mid-run, a node briefly disabled, a panel not yet mirrored — all silently miss
   events. `run_id`+`seq` lets a consumer *detect* a gap; nothing lets it *fill*
   one.
3. *The idiom is unteachable.* "Previews, never truth" is the substrate's contract;
   Silk uses the channel for facts that have no later truth. A plugin author
   reading `threaded.py`'s module doc and `agent.py` side by side gets two
   incompatible lessons.

**Recommendation.** Name the pattern and make it first-class instead of implicit:

- Short term (Silk-local): register a distinct `silk_event` port datatype for
  stream-only ports, visually distinct, with a validator that rejects nothing but
  a formatter that says `<event stream>`. Document the convention in NODES.md: an
  event port's evaluated value is a *digest* — and actually return one
  (`{"count": n, "last": …, "run_id": …}`) from `compute()` so pull-evaluation is
  honest and cheap graph reactions ("how many denials?") work without a
  stream-aware node.
- Medium term (Weave-level): a `PortKind.STREAM` flag in `PortRegistry` so the
  canvas can render live-wires differently and connect-validation can warn when a
  stream port feeds a pull-only consumer. Silk is the first plugin to need it; it
  will not be the last.
- The durable JSONL sink (T7) stops being optional the moment anyone builds on
  events: it is the replay/backfill story for cost 2. This review upgrades T7 from
  "nice to have" to *the* missing piece of the event architecture.

### R2. The three-channel trichotomy is the plugin-author contract — write it down

**Observation.** Silk uses all three channels correctly in the large:
truth on evaluated ports (`response`, `chat_turn`, `outbox`), previews and events
on streams, control on pulses (`run`, `done`, `signed`). But this mapping exists
nowhere as doctrine — it is recoverable only by reading `threaded.py`,
`base.py:812`, and three Silk nodes.

**Recommendation.** One page, in Weave's docs (not Silk's): *"Which channel does my
value ride?"* — truth → port; live progress → stream; happened-once → pulse
(payload allowed); happened-once-and-must-not-be-lost → pulse + durable store.
With the R1 digest convention as the worked example. This is the highest
consistency-per-line documentation available anywhere in the project.

### R3. Live handles on wires are fine — but only under a rule nobody has stated

**Observation.** Four Silk port types carry live Python objects: `silk_toolbox`,
`silk_toolset`, `silk_role`, `gguf_model` (pool handle). Meanwhile D51 rejected
passing a seam handle over a wire on the grounds that "the wire transports
nothing; values on wires stop being inspectable, serializable or replayable."
Both are right — but the spec never says *why* both are right, so every future
handle-shaped proposal will relitigate it.

The distinction that does the work:

> **A handle may ride a wire when (a) it is a product of upstream evaluation,
> rebuilt when upstream re-evaluates, and (b) the receiver only calls *into* it
> during its own compute.** A handle that exists to deliver calls *back against
> the flow direction*, or outside evaluation, is a rendezvous — and rendezvous
> objects are what D51 rejects.

ToolBox/ToolSet/Role handles satisfy (a)+(b): the ToolSet is rebuilt from the
ToolBox's *recipe* on evaluation (`toolset.py` docstring — "rebuilt from the
toolbox's recipe as an independent ToolBox instance"), and agents call it only
inside their run. The rejected seam handle violates both.

**Recommendation.** Promote this to a spec invariant (the natural I12). It
retroactively justifies the existing wiring, closes the door D51 closed *as a
rule* rather than as a case, and gives the MCP nodes (§10 of the spec) their
handle policy for free. Long-term direction stays "recipe on the wire, instance
at the consumer" — the ToolSet already does this internally; it is the correct
answer wherever handle-sharing bites (see R5).

### R4. Run-scoped state on graph-scoped objects is the plugin's one recurring bug

**Observation.** The same lifetime error, independently, in three places:

- The orchestrator writes `_delegation_depth`/`_delegation_chain` onto the shared
  live `spec.toolset` and never clears them (D52(2), `functions/orchestrator.py:231`).
- `RoleBinding` exclusive activation makes a fanned-out toolset fail at *run time*
  — the second agent gets "Another agent node holds this toolset right now"
  (`nodes/agent.py:344-347`) — because a per-run binding lives on a graph-lifetime
  object.
- The decision seam had to be explicitly specified as *run-scoped* (D49) to avoid
  becoming the third instance.

**Consequence.** Each instance was found separately and fixed (or specified)
separately; nothing prevents the fourth.

**Recommendation.** State it once as an invariant: **run-scoped state never lives
on graph-scoped objects.** Everything a run needs travels in the run's own context
— `RunContext` (`functions/run_context.py`) already exists and is the right
vehicle: delegation depth and chain belong in it, not in `setattr` on a toolset.
This turns D52(2)'s fix from "add a `finally`" into "move two fields where they
always belonged."

### R5. The type system can't say "exclusive" — so exclusivity fails at run time

**Observation.** `silk_toolset` is single-consumer in practice (RoleBinding), but
the canvas will happily fan one toolset wire out to three agents; the graph looks
legal, draws legal, and the second agent errors mid-run. Connection cardinality is
not expressible in `PortRegistry` — validators check values, not topology.

**Consequence.** The one place Silk's "illegal graphs cannot be drawn" story (S1)
breaks. The failure is also *timing-dependent*: two agents run sequentially work;
run concurrently, one fails.

**Recommendation.** Two options, not exclusive:

1. *Weave-level:* an `exclusive_consumer` flag on port registration; canvas
   refuses (or warns on) the second connection. Small, generic, benefits any
   future stateful-handle plugin.
2. *Silk-level (deeper fix, aligned with R3):* stop sharing the instance — the
   ToolSet node emits the recipe, each Agent builds its private instance at run
   start. RoleBinding exclusivity then guards nothing and can be deleted, and
   D52(1) (same-worker fan-out) dies as a side effect, because workers stop
   sharing live toolsets at all.

Option 2 is the architecturally honest one and the review's recommendation; option
1 is the cheap interim guard.

### R6. There are two human-in-the-loop tiers, and the evaluation model is the border

**Observation.** The codebase contains both a Sign-Off *node*
(`nodes/signoff_node.py` — parks tasks `awaiting_signoff`, human approves, pulses
`signed` → Agent `run`) and the spec's ruling that an approval *node* is
impossible (D51). These look contradictory. They are not — and the resolution is
the cleanest consistency rule in the whole design:

- The sign-off flow acts **at a turn boundary**: parking a task *ends the turn*
  (`signoff_hold` → the run completes), the human decides between runs, and a
  pulse starts the next run. Everything happens where the evaluation model
  permits it: no node is mid-compute while waiting. Pulse cycles are legal
  (edge events, not data edges; the engine tolerates them).
- The approval gate acts **mid-run**, while the Agent blocks inside `compute()` —
  where a node answerer is structurally impossible (D51) and the node-local seam
  (D48/D49) is the only coherent form.

**Consequence of not stating it.** The next reviewer reads D51, finds
`signoff_node.py`, and files a drift report (this reviewer nearly did). Worse,
the next human-interaction feature gets designed without knowing which tier it
belongs to — and the tier decides *everything* about its shape.

**Recommendation.** Add the rule to the spec explicitly:

> **A human decision may be a node if and only if it happens at a turn boundary.
> Mid-run decisions are node-local (D48/D49).**

One sentence reconciles D30/D48/D51 with the shipped Sign-Off node, and it is the
decision procedure for every future "should X be a node?" question. (It also
cleanly hosts the D51 postscript: the *Approval Policy* node — configuration read
at run start — is a turn-boundary artifact, hence legal.)

### R7. The Agent node is the Qt shell's thickest wall — and it holds Qt-free logic

**Observation.** `SilkAgentNode.compute()` is ~250 lines (`nodes/agent.py:330+`):
role binding, run-id/seq allocation, the whole hook-callback set
(`_on_run_started` … `_on_denied`), plan-change dedup, sign-off hold detection,
outbox assembly. Almost none of it touches Qt except through two seams
(`emit_stream`, `status_changed.emit`). Meanwhile `run_subagent`
(`functions/subagent.py`) rebuilds a parallel version of the same run-assembly
logic Qt-free, and `HeadlessNode` (engine parity, §1.5) can run graphs with no Qt
at all — but not *this* node, because its orchestration body is welded to the Qt
class.

**Consequence.** Three copies of "assemble and observe a run" drift apart (Agent
node, subagent runner, and whatever the test suite ends up scripting — G4), and
the headless story for agent graphs (spec open question 1d, CI evaluation) is
blocked not by the engine but by 250 lines standing on the wrong side of the
layer line.

**Recommendation.** Extract a Qt-free `AgentRun` builder into `functions/` taking
two callables — `emit(event_dict)` and `set_status(str)` — and returning the
result bundle (`response`/`chat_turn`/`outbox`). The Qt node's `compute()` becomes
~30 lines of adapter; `run_subagent` and a future `HeadlessAgentNode` consume the
same builder. This is the single highest-leverage refactor available: it converts
the verified layer discipline (§2) from "the imports are clean" into "the logic is
reusable," and it is a precondition for testing the D42 race catalog without a
running canvas.

### R8. Three ways to feed a task, precedence documented only in a comment

**Observation.** An Agent's task can arrive as (1) the `user_prompt` port,
(2) the BIDIRECTIONAL widget fallback when the port is unconnected, or (3) the
`inbox` AgentMessage ("becomes the task when no direct prompt is given" —
comment at `nodes/agent.py:113`). Wire (1) and (3) simultaneously and the inbox
is silently outranked.

**Recommendation.** Precedence belongs in the node description and NODES.md, and
the losing input deserves a status-line note at run start ("inbox message ignored:
direct prompt present"). Cheap, and it converts a silent rule into a visible one.
The BIDIRECTIONAL idiom itself is exactly right — this is the substrate's
designed behavior (§1.4) and the D48 decision UI extends the same
"node body is the conversation surface" pattern; no change wanted there.

### R9. Silk registers port types through the registry's private guts

**Observation.** Every registration in `silk_ports.py` is guarded with
`if "name" not in PortRegistry._by_name:` and the dirpath cast is installed by
writing `PortRegistry._cast_registry[key]` directly. The registry has a
persistence layer for custom types (stable IDs/colors across restarts,
`load_persistence`/`save_persistence`) that a bypass can silently fight.

**Consequence.** Works today; breaks the day the registry's internals change or
the persistence pass reassigns an ID Silk pinned by hand. Also unteachable — the
next plugin copies the private-attribute idiom.

**Recommendation.** Weave grows two public calls — `register_if_absent(...)`
(idempotent, returns the existing type) and `register_cast(src, dst, fn)` — and
Silk's file shrinks to declarative calls. The *instinct* in `silk_ports.py`
(idempotence under re-import, R2.5.6) is correct; only the access path is wrong.

### R10. Port colors are scattered where they could carry meaning

**Observation.** The nine Silk types use palette indices 99, 141, 150, 201, 205,
214, 226, 232, 237 — no discernible family structure. Weave resolves colors
through the theme palette precisely so types can be visually grouped.

**Recommendation.** Reserve a contiguous band per family: capability lattice
(toolbox/toolset/role) in adjacent hues, model/backend handles in another,
messaging in a third, stream-event ports (R1) in a deliberately "live"-looking
one. Users read a Silk graph by its wires; make the wire colors teach the
§2 family structure. One-file change, zero code risk.

### R11. The worker↔main contract is exactly three primitives — say so

**Observation.** Everything that crosses the Agent's thread boundary reduces to:
queued signal out (`emit_stream`, `status_changed`), plain threading primitive in
(`DecisionSeam`'s lock+event, D49 — explicitly *not* a Qt signal, since the worker
blocks mid-`next()` with no event loop), cancel token sideways
(`CancellationToken`, honored between rounds; G8 covers the gaps *within* a
batch). The orchestrator's fan-out broke precisely by inventing a fourth path
(shared mutable objects across threads, D52(4) `UsageLimits` TOCTOU).

**Recommendation.** Write the trichotomy into the spec's thread-model section as
the plugin-author rule: *out = queued signal; in = threading primitive owned by a
run-scoped object; control = cancel token; nothing else crosses.* R4's invariant
plus this rule would have prevented every threading defect found this month
(D52(2), D52(4), G8's seam half) at design time.

### R12. Node base-class selection is folk knowledge

**Observation.** The suite uses `ActiveNode` (cheap sync config: ToolSet, Role,
Inference Settings, Agent Spec…), `ThreadedNode` (streaming/long compute: Agent,
Chat Display, monitors' sinks, Toolchain — for version probing), and
`ThreadedManualNode` (user/pulse-triggered: GGUF Loader, Pool Monitor). The
choices are all *correct*, but the decision procedure lives in nobody's head but
the original author's.

**Recommendation.** A six-line table in NODES.md: sync + cheap → ActiveNode;
long/streaming → ThreadedNode; runs-when-told (button or pulse) →
ThreadedManualNode; wants engine-parity tests → keep the body in `functions/` and
adapt (R7). Costs a paragraph, saves every future contributor an afternoon.

### R13. The spec and the code disagree in three places the spec doesn't know about

Beyond the already-recorded G16/G17/G18:

1. **Sign-Off node vs D51** — resolved by R6's rule, but the spec must say so
   (today it reads as a contradiction).
2. **`silk_agents` validation is shape-only** — `hasattr(s, "model_handle")` —
   while the Orchestrator's real preconditions (distinct toolsets per worker,
   D52(1)) are checked nowhere. If R5 option 2 lands, this resolves itself;
   until then the validator is the only place a bad worker list could be caught
   before run time.
3. **Hook vocabulary honesty** (G3: 11 of 19 events never emitted) is the same
   disease as R1's port honesty — surfaces that advertise more than flows. One
   pruning pass should fix both under one principle: *a declared surface is a
   promise; don't declare what nothing emits.*

### R14. User-friendliness: the graph shows structure, not runtime

**Observation.** The assembly UX is genuinely good — the minimal agent is two
nodes (Loader → Agent), presets are uniform, docstrings/NODES.md are current, and
error states reach the node UI via `compute_error`. What the canvas cannot show
is *runtime occupancy*: which agent holds which toolset binding, which session
owns the resident KV context (D46/D47), whether a fan-out is serialized behind
`llama_outer_lock` (D53 — "correct but looks hung", spec open question 1c).

**Recommendation.** The observability family (§2) is the right home and mostly
exists: Pool Monitor already snapshots the pool; extend the snapshot with
per-session backend/affinity rows (D45's `snapshot()`-per-backend work provides
the data), and give the Agent's status line one word of truth during contention
("queued behind <agent>…"). No new node kind needed — this is the existing
pattern finishing its job. Combined with R1's digest ports, simple graph-native
dashboards ("denials this run") become wireable by end users without stream-aware
custom nodes.

---

## 5. What this adds up to

The architecture is in better shape than a defect list suggests. The three
serious structural risks are:

1. **The event channel has no honest contract** (R1) — everything observability
   builds on it, and it is currently a preview channel carrying facts.
2. **Shared live instances with per-run state** (R4+R5) — the one bug class that
   has now recurred three times and will recur again until the instance-sharing
   stops or the invariant is written.
3. **The Agent node's Qt-welded run assembly** (R7) — the wall between "clean
   imports" and "testable, headless-capable, single-sourced agent runs."

None requires a redesign; all three are *namings* followed by mechanical work.
That is the signature of an architecture whose bones are right.

## 6. Proposed rules (candidate invariants)

For adoption into DESIGN_SPEC_DRAFT §4 — each one paragraph, numbered here for
reference:

- **P1 (channels).** Truth rides evaluated ports; previews and events ride
  streams; discrete control rides pulses. An event port's evaluated value is its
  digest, never `None`. (R1, R2)
- **P2 (handles).** A handle may ride a wire iff it is rebuilt by upstream
  evaluation and only called into during the receiver's compute. Rendezvous
  handles are prohibited (D51 generalized). (R3)
- **P3 (lifetime).** Run-scoped state never lives on graph-scoped objects; it
  travels in `RunContext` or a run-scoped seam. (R4)
- **P4 (HITL tiers).** A human decision may be a node iff it happens at a turn
  boundary; mid-run decisions are node-local. (R6)
- **P5 (threads).** Worker→main: queued signal. Main→worker: threading primitive
  on a run-scoped object. Control: cancel token. Nothing else crosses. (R11)
- **P6 (surfaces).** Don't declare what nothing emits — ports, hook events, and
  tool schemas are promises. (R13.3, G3)

## 7. Priority map

| Finding | Action | Layer | Effort | Blocks / unblocks |
|---|---|---|---|---|
| R4/P3 | invariant + move depth/chain into `RunContext` | silk | S | D52(2) fix becomes trivial |
| R6/P4 | one-sentence rule into spec | docs | S | resolves apparent D51 contradiction |
| R1 (short) | `silk_event` type + digest returns | silk | M | honest graphs; simple dashboards (R14) |
| R5 (interim) | connect-time guard or clear pre-run error | silk | S | stops mid-run fan-out surprise |
| R7 | extract Qt-free `AgentRun` builder | silk | L | G4 tests, headless runs, de-dupes subagent path |
| R5 (real) | recipe-on-wire, instance-per-agent | silk | M–L | deletes RoleBinding exclusivity, kills D52(1) |
| R9 | public `register_if_absent`/`register_cast` | weave | S | plugin hygiene |
| R1 (real) | `PortKind.STREAM` in registry + canvas | weave | M | first-class event wires |
| T7 | durable JSONL sink | silk | M | replay/backfill for the event channel |
| R10 | color families | silk | S | graph legibility |
| R12 | base-class table in NODES.md | docs | S | contributor onboarding |
| R14 | pool snapshot rows + status-line contention word | silk | M | open question 1c |

*Cross-references: D30, D36, D43–D57 (spec); G1–G18, T3, T5, T7 (open topics);
S1–S6 preserved properties (§3).*
