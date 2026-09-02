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
