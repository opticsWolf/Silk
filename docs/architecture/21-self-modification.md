
## Self-modification — the agent extends Weave

§18 let the agent *use* Weave's parts; this is the agent making new ones
(spec §19, D75–D81). Weave's half is `docs/HOT_RELOAD_PLAN.md` §3.10–§3.11,
all six phases of which have shipped; Silk's half is a small addition to the
tool surface and a large addition to the risk surface, which is why they are
built together.

### The loop already existed except for one verb (D75)

*Write* is the file tools under the sandbox. *Verify* is the toolchain runner
— ruff, mypy, and now `weave_lint`. *Observe* is §18's `describe_graph`. Only
**load** was missing, so that is all that was built: four tools in
`functions/tools/suite_tools.py` (`list_suites`, `load_suite`,
`reload_suite`, `request_relaunch`), registered as ordinary `ToolBox` tools so
they inherit hooks, `tool_events`, role enforcement and the gate for free (the
D56 property). The composition is the feature: an agent that writes a node can
load it and then place it (§18) with no new mechanism in between.

### The one sentence everything follows from (D77)

> Every file tool Silk has is sandboxed; `import` is not sandboxable.

Module-level code in an agent-authored file runs with the full authority of
the Weave process — the network, the whole filesystem, the user's keys — no
matter how narrow the sandbox was while that file was being written. Write
authority over a directory on the import path *is* process authority, deferred
by exactly one tool call.

### `functions/self_modify.py` — the policy, Qt-free

- **The agent's own root** (D76). `user_plugin_root()` is Weave's
  `~/.weave/plugins` (or `$WEAVE_USER_PLUGIN_DIR`), and `check_suite()`
  refuses anything whose origin is not `user`: core, Silk and the virtualenv
  are not material the agent replaces. This is D73's self-modification guard
  moved one layer down, from the graph to the filesystem, for the same reason.
  `capability_for(name)` issues Weave's `SuiteCapability` scoped to exactly
  that one suite, so the refusal is enforced at Weave's verb and not only by
  Silk's good intentions.
- **What this run wrote** — `ChangeSet` plus `attach_change_tracking()`, a
  middleware around the write tools that snapshots each target before and
  after. The first snapshot of a file is the baseline, so two edits in one run
  still diff against what the run *found*.
- **The version check** — `lint_suite()` runs `weave_lint --format json` over
  the suite and treats `WV520`/`WV521`/`WV522` as blocking (D78). A human
  author gets code review; an agent author gets the linter, and nothing else.
  It fails **closed**: a linter that is missing, crashes or hangs reports
  `ran=False` and the load is refused, because a check that could not run is
  not a check that passed.
- **The quarantine fact** (D81) — `record_quarantine()` writes the failure
  where the *next* agent will meet it (the suite listing it reads before
  loading anything) and, when a plan is open, as a task. Weave's loop guard
  covers starting clean; this is the other half, because a self-improving loop
  with no feedback on failure does not improve, it repeats.

### `functions/load_floor.py` — the floor nobody may lower (D77)

Every other gated tool in Silk is gated *by policy*: a Role, a risk band, a
hook config, and a grant can pre-authorise it. `attach_load_floor()` installs
a second, unconditional middleware over `load_suite` / `reload_suite` that

- asks **every time**, whatever the policy says, consulting no grant — an
  "always approve" a preset can switch off is not a control (and note this is
  *not* I6: I6 is a ceiling nobody may raise, this is a floor nobody may
  lower);
- runs the version check first, so a `WV521` finding stops the load before a
  person is asked at all;
- carries **the code**, not the name: the file listing with sizes and mtimes
  plus the diff of everything this run touched. A human approving "load
  `my_nodes`" has approved nothing;
- refuses when there is no seam (D36), and takes a denial as ordinary — the
  files stay on disk and the agent can still report what it built.

The policy gate in `approval.py` steps aside for these two names rather than
putting a second dialog in front of one call; the stricter of the two is not
the policy one. The floor is installed by `attach_suite_tools` itself, because
registering the verbs without it would be registering a way to run arbitrary
code with no human in it.

### Where a load actually happens

1. `check_suite` — is it agent-authored (D76)?
2. the floor — lint, then the human, with the diff (D77, D78);
3. `weave.engine.validation.validate_suite` — a **subprocess** import, because
   a dry run executes the candidate's top-level code and a validation step
   that can segfault the session is not validation. A crash or a hang costs
   one tool call, and the traceback that comes back is the feedback the
   generator needs anyway;
4. the main thread, across §18's `MainThreadCall` seam — `CanvasAuthor`
   dispatches `load_suite` / `reload_suite` / `request_relaunch` before its
   canvas check, since a registry-only load is meaningful in a headless graph.

### Relaunch is a request, never an action (D79)

`request_relaunch(reason)` queues Weave's request and returns immediately; the
run finishes normally and a human is asked at a turn boundary (I12). The agent
never sees the far side — a new process is a new session with empty memory —
so anything that must survive has to be in the plan or the history ledger
*before* it asks. "Continue after the restart" is a task with a claim, not a
promise the runtime keeps, and that is the sharpest argument for the ledger
being durable belief (D63) rather than a cache.

### Letting go before the handoff (D80)

A relaunch spawns a child before this process exits, so anything exclusive
must be released first. The model pool and the task store were already
shutdown participants; `LedgerRegistry` and each `MCPSession` now register
too. Both do it explicitly rather than leaning on GC: interpreter teardown is
not a guarantee, a Macrame handle closed late loses its final snapshot, and
two processes on one ledger file breaks the one-write-actor model D64's
adjudication rests on. The ledger's participant also reports what it holds, so
a relaunch that waits can say why.

### The ToolBox node (D76, D77)

One checkbox, **Plugin authoring**. Ticking it mounts the four verbs, adds the
user plugin root to the sandbox, and — if nothing else made the box writable —
makes *only that root* writable: plugin authoring is not a reason to make the
user's project writable. Weave core and Silk sit outside every root, so the
existing static narrowing is what keeps them read-only. No new mechanism.

**Out of scope in v1, deliberately: Silk improving Silk** (T10). That needs
review-then-relaunch and has a bootstrapping problem the graph case does not —
the code that would review the change is the code being changed.

### Approve once, pinned to the bytes (§22 q10, q11)

The floor asks every time an agent asks. Question 10 was what a *start* may
do with an approval it already has, and the two obvious answers are both
wrong: asking every session makes "write a node today, use it tomorrow"
not happen, and approving a *directory* is approving whatever the agent
puts in it next.

`functions/suite_pins.py` takes the third: an approved load records a
SHA-256 per importable file, and `weave/bootstrap.py` loads a pinned suite
at the next start only when every digest still matches. One edited
character and the pin no longer applies, so the next load goes back
through the floor with a human looking at the diff. The pin file lives at
`~/.weave/silk/suite_pins.json`, beside the grants and for D35's exact
reason — a pin the agent could write is an agent approving its own code —
and it fails closed like the grant store: unreadable means nothing is
pinned, which means asking.

Three details the tests pin down. A pin covers only what an import can
execute (`.py`, `.pyi`, `.pth`, `.pyd`, `.so`, `.dll`, `.json`), so a
README does not invalidate it and a new module does. A suite over
`MAX_PINNED_FILES` is not pinned at all rather than pinned expensively — a
start is not a disk scan. And a shipped suite that appears under a pinned
name does not inherit the approval (D76): the pin was for agent-authored
code under the plugin root, not for a name.

**Quarantine breaks the pin** — that is question 11's answer.
`record_quarantine` unpins, so a suite that took the process down never
comes back by itself, whatever a human approved before it crashed. The
agent may still read the traceback out of `list_suites` and try a fix; it
simply cannot take the last step alone. Auto-load *and* auto-retry is
precisely the configuration in which a self-improving loop runs
unattended, which is exactly when it should not.

Both approvals are withdrawable in one place: the Grant Manager dock
(`widgets/grant_manager.py`) lists pinned suites under the durable grants,
and revoking one deletes no code — it puts that suite back in front of the
floor.
