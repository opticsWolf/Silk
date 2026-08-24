## Task system and sign-off

Silk has a first-class planning/audit subsystem, fully headless.

### `functions/task_store.py` — `SqliteTaskStore`

A SQLite-backed store with **optimistic concurrency** and a full audit
trail. `SqliteTaskStore(root, direct_write=True)` resolves a writable
database location (trying candidate directories in order) and opens the
schema:

- **`plan`** — one row per plan: `plan_id`, goal text + original text +
  acceptance criteria, `revised`, `revision`, and any *pending goal*
  revision held for sign-off.
- **`task`** — `(plan_id, id)`-keyed rows: `title`, `status`, `parent`,
  `ord`, `note`, `origin`, `added_by`/`claimed_by`/`done_by` actors,
  timestamps, and the sign-off fields (`signoff_summary`, `signoff_by`,
  `signoff_note`, and `signoff_action` — the *held-and-applied* action,
  e.g. a deviation rescope that only lands on approval).
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
`rescope_task`, `revise_goal`, `claim_task`, `request_signoff`,
`request_goal_signoff`, `sign_off`, plus reads (`load`, `history`,
`pending_signoffs`). Each mutation bumps the plan `revision` and writes an
audit row. `plan_changed_event(store, last_revision)` returns a
`plan_summary` event **only if** the revision advanced past
`last_revision` — reads never bump the revision, so an unchanged plan never
re-streams. The Agent node calls this after each tool batch to push live
updates to a `Plan Viewer`.

### `functions/signoff.py` — the user sign-off gate

A **policy** maps each *change type* to who may sign it:

- **`agent`** — the agent self-signs; the change applies immediately
  (audited with the agent as actor).
- **`human`** — the change is *parked* for the user; only
  `SqliteTaskStore.sign_off` can apply it. Deviations (rescope / goal
  revision) are **held and applied on approval** — the `signoff_action`
  stores the pending action and it lands only when the user approves.

Change types: `add`, `complete`, `complete_final` (the completion that
closes the plan — resolved dynamically), `rescope`, `goal`. Plain progress
(`task_update` / `claim`) is never gated.

Because the gate must read plan state (which task, is it the last one) and
park the item itself, it is attached **with a handle to the ToolBox** so
the model can't bypass it. It runs as a `HOOK_WRAP_TOOL_EXECUTE` middleware
and is exposed as the configurable `signoff` catalog hook. `SIGNOFF_MODES`
presets (`auto`/`requested`/`completions`/`final`/`strict`) expand to
policies; `custom` uses per-type levels.

**Turn-boundary pause:** parking flips a task to `awaiting_signoff` (or sets
`pending_goal`); the Agent node then *ends the run* so control returns to
the user. The user's approve/reject goes through the `Sign-Off` node →
`sign_off(...)`, which applies or rejects the held action (recording
`signoff_by` / `signoff_note`).

