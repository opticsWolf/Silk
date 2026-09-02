## Task system and sign-off

Silk has a first-class planning/audit subsystem, fully headless.

### `functions/task_store.py` — `SqliteTaskStore`

A SQLite-backed store with **optimistic concurrency** and a full audit
trail. `SqliteTaskStore(root, direct_write=True)` resolves a writable
database location (trying candidate directories in order) and opens the
schema:

- **`plan`** — one row per plan: `plan_id`, goal text + original text +
  acceptance criteria, `revised`, `revision`.
- **`task`** — `(plan_id, id)`-keyed rows: `title`, `status`, `parent`,
  `ord`, `note`, `origin`, `added_by`/`claimed_by`/`done_by` actors, and
  timestamps.
- **`revision`** — an append-only audit log (AUTOINCREMENT id, `at`,
  `actor`, `op_kind`, `op_json`, `rationale`) — every mutation is recorded
  with who did it and why.
- **`deviation`** — the from/to values of any change that deviated from the
  original plan, keyed to the revision that made it.

**Data model** (dataclasses): `Goal`, `Task`, `Deviation`, `Plan`, and
`Conflict`. A `Conflict` is a *genuine* collision (same task touched
twice, double-complete, goal race) — the store returns it instead of
guessing, and the model is told that retrying the identical operation will
not help.

**Operations:** `start`, `add_task`, `update_task`, `complete_task`,
`rescope_task`, `revise_goal`, `claim_task`, plus reads (`load`,
`history`). Each mutation bumps the plan `revision` and writes an
audit row. `plan_changed_event(store, last_revision)` returns a
`plan_summary` event **only if** the revision advanced past
`last_revision` — reads never bump the revision, so an unchanged plan never
re-streams. The Agent node calls this after each tool batch to push live
updates to a `Plan Viewer`.

### Which plan: explicit identity (D23)

The store answers "which plan is this?" two ways, and the difference is
the whole of T4.

Given an explicit `db_path`, `SqliteTaskStore(root, db_path=...)` opens
that file and no other — including when the file does not exist yet, which
is how a graph says *this is where the next plan goes*. Without one, it
falls back to the newest `plan-*.db` across `root` and `root/.silk/plan`.

That fallback is kept on purpose: it is also the mechanism by which
several agents rooted in one directory share one plan, and an agent with
no Task node wired in must still be able to plan. What it cannot do is
tell two *unrelated* plans in one root apart — whichever was written last
wins, so which plan an agent lands on depends on file timestamps.

`PlanRef` is the identity that removes the guess:

```python
@dataclass(frozen=True)
class PlanRef:
    root: str = ""       # where to look
    db_path: str = ""    # which file  (empty -> newest under root)
    plan_id: str = ""    # which plan  (label/diagnostics)
    label: str = ""
```

`coerce()` accepts a `PlanRef`, a plain dict (so it survives graph
save/load), a path string (root only), or `None`; `store()` builds the
store it names; `is_explicit` distinguishes a named plan from shared
discovery. It travels the graph as the `silk_plan` port type: the
`Silk Task` node emits it, the ToolBox node forwards it to
`attach_task_tools(plan=ref)` **through its build recipe** — so a derived
ToolSet replays the same reference rather than re-discovering — and the
Plan Viewer's `plan_ref` input outranks its `root`.

`SqliteTaskStore.scan_all(root)` is the read-only complement: every plan
under a root as plain rows (`db_path`, `label`, `plan_id`, `goal`,
`updated_at`, `tasks`, `open_tasks`, `mtime`), newest first, an unreadable
file reported as a row with an `error` rather than raised. It is what the
Task node's dropdown offers and what the Task Hub scans under D58/D60;
agents without a reference keep using newest-only discovery.

### `functions/ledger.py` — one handle per ledger file (D62)

The optional Macrame backend (§17) is a bitemporal graph ledger with
**one Write Actor per open handle**. Two handles on one file is outside
its contract — and the library does not refuse it: opening the same path
twice succeeds and leaves two writers racing. So the rule is enforced at
the only place that can be complete, the single place that opens.

`LedgerRegistry` maps a canonical path (resolved, case-folded on Windows)
to one refcounted handle. `acquire()` opens once and shares thereafter;
`release()` gives up a reference but deliberately does *not* close, since
several agents in one sandbox root share one ledger and Macrame's close
writes a final snapshot — cheap once, wasteful per run. `close_all()` is
the owner's call at plugin unload or graph close, which is exactly what
Macrame means by "close has one owner". A handle closed behind the
registry's back is reopened rather than handed out dead.

`get()` is the read side, and it is what the D62 amendment to D58/D60(2)
needs: the Task Hub finds ledger *files* on disk and asks the registry
whether one is already open here, so a scan never becomes a second open.

`macrame-db` is an optional extra (D66). Absent, `available()` is False,
`acquire()` raises `LedgerUnavailable` carrying a sentence that names the
distribution, and the caller falls back to `SqliteTaskStore` — loudly,
never silently. Edge types are validated by Macrame against `[A-Z0-9]+`,
so the vocabulary is `CLAIMEDBY` / `SUBTASKOF` / `INRUN`, not the
underscored names §17 writes prose in.

### `TaskLedger` — the task store on the ledger (D63–D66)

`TaskLedger` answers `SqliteTaskStore`'s protocol exactly: same methods,
same keyword-only signatures, same `Plan` / `Task` / `Goal` / `Deviation`
objects, same `Conflict` on a refusal, same `history()` shape (entries
numbered by position in the log, which is what the SQLite store's rowid
was doing). `tests/test_silk_task_ledger.py` runs one identical sequence
through both backends and compares the results field by field, including
the wording of nine refusals — because a tool that special-cases a
message must not have to ask which backend answered.

What changes is underneath. A status transition **supersedes** rather
than overwrites (Doctrine III), so `load(as_of=<datetime>)` answers *what
did the plan look like at 14:00* as a read: the plan header, its tasks
and its revision log all travel to the same instant, because Macrame
refuses `as_of` without an attribute mode and returning the past's
topology under the present's titles is the mistake that refusal exists to
prevent. Nothing is ever deleted (Doctrine V): a dropped task is a
superseded status with the revision that dropped it still readable.

The graph shape is the point. A plan owns its tasks (`HASTASK`) and its
revision log (`HASREVISION`); a subtask points at its parent
(`SUBTASKOF`); a claim is an edge to an `agent:<name>` concept
(`CLAIMEDBY`, `DONEBY`), so *what did this agent touch* is a traversal
rather than a scan. Actors are upserted before they are linked — an edge
needs both ends to exist, and an actor deserves to be a node anyway.
Re-asserting an edge that is already open is refused by Macrame (it would
be two claims of one fact), so the idempotent cases — an agent retrying
its own claim — go through `_link`, which asserts only what is not
already there.

**The lock is for decisions, not throughput (D64).** Macrame's Write
Actor already serialises writes within the process; N agent threads may
call write methods concurrently. What needs protecting is
read-check-assert: claim-a-task, advance-a-status-with-a-precondition.
Those run under one `threading.RLock` **per ledger file** — two roots are
two decision domains, and serialising them together would make one
agent's claim wait on an unrelated plan. Eight threads racing for one
task produce exactly one winner and seven `Conflict`s that name it. A
refused decision writes nothing at all, not even a bumped revision.

`start()` writes its edges in one `write_bulk_atomic` and the head
pointer last (D65): a plan either started or did not, and a half-started
one would read as a plan with missing tasks rather than as no plan.

### `HistoryLedger` and `recall` — memory instead of scrollback (§17, D66)

History is the half of §17 that fits structurally: an append-only
assertion log is what Silk's history already is (I11 — the prefix grows
only at the tail). Moving it off the node buys two things the node
cannot.

**Compaction stops being destructive.** `compacted(run_id, dropped=…,
kept=…, rationale=…)` records a supersession event rather than a
deletion, so `turns(run_id, include_superseded=False)` is what the model
sees now and `turns(run_id)` is still the whole conversation. *What did
the model actually see at round 7* survives the squeeze — the question
D41's measurement and D42's tests want and could not have while history
lived in `self._history` and died with the node (D24/D25).

**Memory outlives the run.** `recall(query, top_k=…, kinds=…)` is FTS5
keyword search over remembered turns and runs, so it reaches into earlier
runs and earlier sessions, and into rounds compaction dropped. §17's plan
is hybrid search (FTS5 + vectors, fused by RRF); FTS5 needs no embedding
model, so it ships first and the result shape does not change when the
vector half arrives.

Writes are turn-shaped (D65): the run concept and its identity edges at
start (`BYAGENT`, `INSESSION` — D60's observability plumbing and the
ledger's keys are the same plumbing), then per turn one concept for the
discrete fact and **one** `write_bulk_atomic` for the bookkeeping
(`INRUN`, `USED`, `TOUCHED`). A crash between the two leaves an orphan
turn, which reads as *a turn happened and we do not know what it
touched* — the honest answer. A turn recorded without a `start_run` still
lands, and gets a run to hang on.

History lives in its own file (`history.macrame`) beside the task ledger
(`ledger.macrame`): one Write Actor each keeps a chatty turn writer from
queueing behind a plan read, and lets a graph keep its plan while
dropping its memory.

`recall` reaches the agent as a tool
(`functions/tools/recall_tool.py`, the `Recall (memory)` checkbox on the
ToolBox node). It imports `functions/ledger.py` and nothing else — D66's
seam is that no node and no tool ever imports `macrame` — so a missing
extra is a tool that says why, not an ImportError at load time.

### Choosing a backend, loudly (D66)

`open_task_store(root, plan=…)` hands back whichever backend the
environment asks for via `SILK_TASK_BACKEND` (`ledger` / `sqlite`), and
`attach_task_tools` goes through it, so the tools never choose. If the
ledger is asked for and `macrame-db` is absent, that is **one warning
line and the SQLite store** — the graph degrades to today's behaviour,
never silently and never by crashing a run that was only going to write a
task list. `open_history()` has nothing to degrade to (before D66 history
died with the node), so it answers `None` and logs once, rather than
returning an empty search that reads like *nothing happened*.

**The default is still `sqlite`, deliberately.** Not because the ledger
is unfinished — the parity suite says otherwise — but because plan
*discovery* is still file-shaped: `PlanRef`, `scan_all` and the D58 hub
all look for `plan-*.db`. Flipping the default is a separate,
discovery-shaped change; until it happens the ledger is opt-in per
process, and everything above the store is already backend-blind.

### `functions/task_board.py` — the multi-agent projection (D58)

N independent top-level agents share no event port and never will, so
there is no wire that shows all of them — except the one place they all
write to. `scan_roots(roots)` gathers every plan under every root (the
same plan reached through two overlapping roots is one lane, not two),
`board(rows)` turns them into plain data, `render_board(data)` into
markdown.

`claimed_by` is what the projection exists for. The schema has carried it
since it was written and no view has ever shown it, which is why "who is
doing what" was unanswerable in a graph running four agents. It appears
per task and, aggregated, as the plan's actor list (D60(3)).

A plan that will not open still becomes a lane, carrying its error and no
tasks: a board that hides the file it could not read lies about how many
plans there are.

`PendingDecisions` folds the event stream into one number — how many
agents are blocked on a human right now. It holds correlation ids and
nothing that could resolve them, because D58 lets the hub **count**
mid-run requests and D59 reserves answering them to the asking node or
its dock mirror. A run that finishes clears whatever it never answered,
so a timeout does not pin the count above zero forever.

**The sign-off half of D58 is not implemented, deliberately.** D58 gave
the hub Approve/Reject buttons that write a sign-off to the store and a
`signed` pulse. D31–D33 then deleted parked sign-off entirely: there is
no `awaiting_signoff` row to approve later, because a gated task change
is decided *during* the turn on the run's decision seam. So the hub is
read-only, `signed` would have had nothing to pulse, and §22 q6 ("does
the single-agent Sign-Off node survive the hub?") is answered by D32
having already deleted that node.

### `functions/signoff.py` — the user sign-off gate

A **policy** maps each *change type* to who may sign it:

- **`agent`** — the agent self-signs; the change applies immediately
  (audited with the agent as actor).
- **`human`** — the change needs the user: the call blocks on the run's
  decision seam and applies only if the answer is yes. With no way to ask
  — a headless run, a destroyed widget, a timeout, a Stop — it is
  **refused**, with `approval_required`, the `change_type` and the target
  in the result, so the model can say what is waiting on the user instead
  of retrying (D36).

Change types: `add`, `complete`, `complete_final` (the completion that
closes the plan — resolved dynamically), `rescope`, `goal`. Plain progress
(`task_update` / `claim`) is never gated.

`signoff.py` owns this *vocabulary* — levels, change types, presets, and
the tool-to-change-type map — and nothing else. The gate that enforces it
lives in `functions/approval.py`, because D31 makes task changes and tool
calls two policy **domains of one middleware** rather than two
subsystems. `attach_signoff_gate(...)` is still the entry point the
`signoff` catalog hook selects and the recipe replays; it is a thin call
into `attach_approval_gate(task_policy=...)`. `SIGNOFF_MODES` presets
(`auto`/`requested`/`completions`/`final`/`strict`) expand to policies;
`custom` uses per-type levels.

**Nothing is parked (spec D31–D33).** There used to be a second approval
subsystem here: an `awaiting_signoff` status, four `signoff_*` columns, a
`pending_goal` on the plan, `request_signoff` / `request_goal_signoff` /
`sign_off`, a `Sign-Off` node, and an Agent node that inferred "the user
must decide something" from the *shape* of the plan and ended the turn.
All of it is deleted, not migrated — forward-only (D33): early
development, `plan-*.db` files are recreated rather than upgraded, so an
in-flight plan does not survive the change. Audit was never in those
columns anyway; it lives in `revision` and `deviation`, which already
record who did what, when and why.

### `functions/approval.py` — the approval gate

One `HOOK_WRAP_TOOL_EXECUTE` middleware, two policy domains:

- **task changes** — `{change_type: level}` over the five change types
  above (`task_policy`);
- **tool calls** — `{tool_or_risk: level}` over tool names and the `risk`
  band every tool already declares at registration (`tool_policy`).

The four plan tools are reachable from both, and where both have something
to say the **stricter** answer wins: naming a tool in one domain is never
undone by the other's default. A tool name always beats its risk band. The
tool domain has no defaults — an unnamed tool is ungated — because it is
open-ended, and "gate everything" would make every newly registered tool a
prompt; the task domain has five known types and is filled in with `agent`.

Because the gate must read plan state to tell a `complete` from a
`complete_final`, it is attached **with a handle to the ToolBox** so the
model can't bypass it. It is registered **essential** (D11/D13/D14, I7), so
a Role cannot deactivate it and a derived ToolSet carries it, and it is
**forced outermost** in the middleware chain (`HookRegistry.make_outermost`,
D37/I10): a middleware may return without calling `handler()`, so anything
ahead of the gate could answer a call the gate never sees. The corollary is
the half that bites — a denial produces `applied: false`, never a
fabricated success.

Everything the gate needs is snapshotted **at attach time** (D38): editing
a Role or a hook config mid-run affects the next run, not the one already
in flight.

Order of consultation, cheapest first: a run-scoped grant, a durable grant,
the policy, then the human. Grants can only *skip* the question, never
create one, so consulting them first cannot make the gate stricter than the
policy says.

### `functions/decision_seam.py` — asking, from a worker thread

The gate runs on the run's worker thread, inside a generator that is
mid-`next()` and cannot yield (D38). So the request goes **out** through the
hook emission path — the Agent node emits it into its own stream widget
(D48) — and the answer comes **back** through a threading primitive. One
run-scoped `DecisionSeam` per run; one `kind` field, so acknowledge and
release reuse the same block rather than growing a second waiter (D50).

The ordering rule is one sentence: **write the outcome under the lock, then
set the event; the waiter re-reads under the lock.** The event only says
that *something* happened; what happened is the state the writer committed
before setting it, and that is what makes approve, deny, Stop and timeout
four distinguishable wakeups rather than one ambiguous one.

Five causes, and four of them deny (D36): `answered`, `cancelled` (Stop
calls `seam.cancel()` directly — the consumer loop is not polling
anything), `timeout`, `no_answerer` (which denies *without* waiting out the
deadline), and `transport`. `await_decision` returns rather than raises on
every path, because its caller's job is to produce a tool result.

`DriveGate` is the test seam: named checkpoints a test parks the seam at,
so the races in D42's catalog are driven deterministically in both orders
rather than with sleeps. The property it exists to pin is **zero effects
while parked** — a held tool call that ran anyway would satisfy every
assertion about the decision and still be a disaster.

### `functions/decision_registry.py` + the Decision Inbox dock (D59)

The seam above is per node, which is right — the question is asked from
inside *that* node's `compute()`. With ten agents on a canvas it is also
ten places to look. The registry is the directory that fixes finding
without touching answering:

- A request registers when the Agent node shows it and unregisters when
  it settles; a run that ends releases whatever it never answered, so a
  timeout or a Stop does not leave a button behind.
- Entries hold a **weak** reference to the asking node and nothing that
  could resolve a decision. Seam lifetime (D49) is unchanged: the
  registry can never be why a run stays alive, and a deleted node's row
  disappears the next time anyone looks.
- `subscribe()` is how the dock hears about changes; a listener that
  raises is ignored, because a blocked run must not depend on a dock
  being well behaved.

`widgets/decision_inbox.py` is the surface: a `QDockWidget` with one row
per waiting agent, the same four answers the node offers (deny first),
and a *Show node* button that selects the asker on the canvas. Clicking
an answer calls that node's own `_answer_decision` — same method, same
seam — which is what a mirrored button does (`wire_action_proxy`); the
node's prompt is a composite container that NodePanel's clone strategies
do not cover, so the row is built rather than cloned. Closing the dock
takes away a shortcut, not the answer.

The canvas carries the same information without any panel open: a node
blocked on a decision switches its pulse to `heartbeat` and back
afterwards, so "who needs me" is visible in the graph itself.

Silk has no plugin-side hook into the host window, so
`DecisionInboxDock.attach(main_window)` is how it gets added.

**Why a dock and not a node.** D51 rejected an approval node: a node
cannot answer a question asked from inside `compute()`, because inputs
are gathered before compute runs. A dock is main-thread UI like the log
pane — no wires, no graph channel, no rendezvous handle. I12 in one
sentence: a decision surface may be a node only if the decision happens
at a turn boundary, and this one does not.

A run with no seam at all refuses every gated call -- and says so **once**:
the gate logs the first such refusal and counts the rest
(`headless_refusals(toolbox)`), which the Agent node reports at the end of
the run. A headless batch that denies forty times in silence is behaving
correctly and looks hung (§22 q1d).

### `functions/grants.py` — "don't ask again"

A decision may carry a scope: `once` (no grant at all), `run`, or `always`.
Run-scoped grants live in the gate's closure and die with the run. Durable
grants live in `~/.weave/silk/grants.json`, keyed by the **resolved project
root** (D10/D34/D35):

- **not the plan database** — it exists only when the task tools are
  mounted, and an agent with file tools and no planning tools is exactly
  the configuration that most needs a gate;
- **not under the sandbox root** — a durable "always allow `write_file`"
  record inside a tree the agent can write to is a permission the agent can
  award itself;
- **allow-only** — revoking is deleting, there are no deny records, so a
  missing, corrupt or unreadable store degrades to *nothing is granted*.
  Every failure path leads to more prompting, never less. That is also why
  the store is read-modify-write, last writer wins: a lost grant costs one
  extra prompt.

### `widgets/grant_manager.py` — taking it back (§22 q1)

D10 made "don't ask again" durable and D35 put it outside the graph; the
half that faces the person is a dock, because an allowance you cannot find
is an allowance you cannot take back. `GrantManagerDock` groups by project,
one row per grant, with **Revoke** on the row and **Revoke all** on the
project header. `GrantStore` grew the two readers it needs: `projects()`
and `all()` (newest first — a user starts from *what did I allow*, not from
a project root they have to remember).

Three properties, and they are the whole design:

- **It shows what is on disk.** Every refresh calls `reload()`, so a grant
  another window made appears here, and a revocation here is seen by the
  next gated call everywhere — the gate consults the store, not a cached
  set.
- **Revocation is deletion**, inheriting the allow-only store: nothing in
  this surface can create authority or a permanent refusal. It moves in the
  safe direction only, which is why it needs no approval of its own.
- **Run-scoped grants are not listed.** They live in the gate's closure and
  die with the run; showing them would invite revoking something already
  gone.

A dock and not a node, for D51/I12's reason: process-wide state, no inputs,
no outputs, nothing a graph could wire to it. Like the Decision Inbox it is
mounted by the host via `GrantManagerDock.attach(main_window)`, since Silk
has no plugin-side hook into the window.
