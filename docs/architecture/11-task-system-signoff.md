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
- **`human`** — the change needs the user. Until the inline decision seam
  lands (spec D30) it is **refused**, with `approval_required`, the
  `change_type` and the target in the result, so the model can say what is
  waiting on the user instead of retrying.

Change types: `add`, `complete`, `complete_final` (the completion that
closes the plan — resolved dynamically), `rescope`, `goal`. Plain progress
(`task_update` / `claim`) is never gated.

Because the gate must read plan state (which task, is it the last one) it
is attached **with a handle to the ToolBox** so the model can't bypass it.
It runs as a `HOOK_WRAP_TOOL_EXECUTE` middleware, **bound** to the four
plan tools and registered **essential** (D11/D13/D14), so a Role cannot
deactivate it and a derived ToolSet carries it. It is exposed as the
configurable `signoff` catalog hook. `SIGNOFF_MODES` presets
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

