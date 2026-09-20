# Post-Spec Planning

**Written 2026-09-20.** Every numbered item in the design spec's three
implementation phases ([DESIGN_SPEC_DRAFT.md §20](DESIGN_SPEC_DRAFT.md))
has a *Landed in* entry. The phased plan is finished, so there is no next
line to pick up — and "what is left" is three unlike things that have been
travelling together under one word.

This document separates them and lays out the options, so each can be
worked through and closed on its own terms.

| | Category | Count | What it needs |
|---|---|---|---|
| **A** | Buildable now | 1 | An engineering decision, then work |
| **B** | Design decisions | 3 | A judgement call; no code until it is made |
| **C** | Accepted, or belonging elsewhere | 3 | Naming the residue and living with it — or one of them moving to Weave |
| **D** | Documentation debt | 2 | Ten minutes |

**How to use this.** Each item states what is true today, the options with
their real costs, a recommendation, and a blank **Decision** line. Fill the
line, act, and delete the item — [OPEN_TOPICS.md](OPEN_TOPICS.md)'s standing
rule applies here too: the commit history is the archive.

---

## A. Buildable now

### A1. The model pool holds one backend, so concurrent agents queue

**Spec:** D45 (the remaining half), D47 mechanism C, §22 q1b/q1c.

**What is true today.** Silk talks to exactly one model server. Four agents
running at once do not run at once — they queue and go one at a time. The
September measurement put a number on it: contention 100%.

Session affinity (mechanism A, built 2026-09-19) made the queueing
*cheaper* — each conversation keeps its prompt warm, so waiting costs less
prefill. It did not make the queueing go away, and it cannot: the moment
concurrent conversations outnumber backends, something waits.

`checkout(session_id)` is already the routing seam and already takes the
session argument — it just discards it. `_process`, `_port` and `_client`
are singular. `snapshot()` returns one flat dict.

**Why this is no longer blocked.** The spec still says C "waits on the
measurement." It does not. The measurement ran 2026-09-19 and landed on
rule 3, selecting A. Rule 5 prefers C where hardware allows, and the two
compose — the spec is explicit that building C does not retire A.

#### Options

**A1-a — Leave it.** One backend, affinity softening the queue.

*Impact:* Orchestrator fan-out stays sequential. It looks parallel on the
canvas and is not. D53's legibility work means it at least *says* it is
serialising rather than looking hung, so the cost is honest latency rather
than confusion. Costs nothing, forecloses nothing.

**A1-b — N named backends, routing at `checkout()`.** The full D45 remainder.

*What it buys:* Fan-out that is actually parallel. It is the only option
whose benefit does not degrade as the number of concurrent agents grows.

*What it costs:*

- Hardware or money. Enough VRAM for N local servers, or remote endpoints
  with a bill. Without somewhere to put them this buys nothing at all.
- Real work in the pool: `checkout()` learns to route, `snapshot()` and the
  Pool Monitor learn to report per-backend instead of one flat dict.
- **A failure mode that does not exist today.** Right now "the server is
  dead" is unambiguous. With N backends one can be down while the rest are
  fine, and that needs a policy — see the sub-question below. This also
  interacts with D89's fallback chain, which already answers "this model
  failed terminally, walk to the next one." A down *backend* is a different
  question from a failed *model*, and the two must not be made to fight.

**A1-c — Build it only when a second backend exists.** Same as A1-b, deferred
behind an actual multi-backend setup.

*Impact:* Avoids building routing for a workload nobody is running. The risk
is the usual one — the seam is cheaper to build now, while the reasoning is
fresh, than in six months.

#### The sub-question, if A1-b or A1-c

**Who chooses the backend?** These feel entirely different to use, and the
choice is not cheap to reverse:

| Chooser | Meaning | Cost |
|---|---|---|
| **Role** | "Reviewers use the big model" becomes a property of the role | Backends become part of role definitions; a role stops being portable between machines with different hardware |
| **Agent node** | Picked per instance, on the canvas | Most explicit and most visible; most tedious for wide fan-out |
| **Pool-side rule** | Automatic, keyed on session | Nothing to configure and nothing to see; scheduling stays the pool's business, consistent with q1c's answer |

And: **what happens when a named backend is down?** Fail the request, fall
back to another backend, or mark it out and retry later. Falling back is the
friendliest, and the one that risks silently running a small model where a
large one was asked for — which D53's legibility rule says must at minimum
be *said*.

**Recommendation.** A1-c. This is infrastructure for concurrent fan-out; if
the common case is one agent against one local server it changes nothing. If
fan-out is where Silk is going, it is the highest-value item left, because
today fan-out is a promise the pool cannot keep. Pick the chooser before
writing any of it — retrofitting that is worse than deciding it.

**Decision:** _______________

---

## B. Design decisions

Each of these is marked *deliberately unsettled* in the spec. That is a
boundary somebody drew on purpose, not a backlog item that was forgotten.

### B1. May the agent edit a graph the user already drew?

**Spec:** T9, D69–D74, D73.

**What is true today.** The agent can add nodes, and can delete only what it
added this run. Nodes the user drew are untouchable, as is the agent's own
execution path.

#### Options

**B1-a — Keep "no" (v1).**

*Impact:* The agent builds new subgraphs but cannot fix or improve existing
ones. An assistant that adds, not a collaborator that refactors. For a user
with a large hand-built graph, that is most of the value left on the table.

**B1-b — Allow it behind per-call approval.**

*Impact:* **The option that looks reasonable and is not.** By the time a
human has clicked through forty individual "move this wire" prompts they
have approved nothing meaningful — approval fatigue converts a gate into a
rubber stamp. D48's per-call model was designed for tool calls that are
individually consequential, and graph mutations are not.

**B1-c — Allow it behind a diff review.** The human sees the whole proposed
change at once and accepts or rejects it, the way code review works.

*Impact:* The only shape that actually works, and a genuinely new UI — not
an extension of the approval gate. Note that it converges with B2-b: once a
change is reviewed whole before application, incremental mutation has
already been abandoned.

**Recommendation.** Keep B1-a. If this is wanted later it arrives as B1-c,
and almost certainly alongside B2-b, because they are the same insight
reached from two directions.

**Decision:** _______________

### B2. Should a built graph be proposed rather than applied call by call?

**Spec:** T9 second bullet, D70, D73.

**What is true today.** The agent places nodes one at a time and the user
watches them appear. D70 exists to get those mutations onto the main thread
from a worker.

#### Options

**B2-a — Keep incremental placement (v1).**

*Impact:* Interactive and watchable; the user can interrupt. Requires the
D70 main-thread seam and the D73 guard rails, and every new kind of mutation
has to be safe to apply mid-build.

**B2-b — Propose a subgraph; the human applies it in one gesture.**

*What it buys:* **Strictly safer.** Nothing is applied from a worker thread,
so the D70 seam is not needed at all — an entire category of threading
concern disappears rather than being managed.

*What it costs:* The interactive feel. It becomes batch: no watching, no
course-correcting mid-build. The spec rejected it for v1 for exactly this
reason, and that reason has not changed.

**The trigger is already written down.** If D73's guard rails start
accumulating exceptions, that is the signal the incremental model is
straining and B2-b is the answer. So this is a symptom to watch for, not a
decision to force now.

**Recommendation.** B2-a, with the trigger recorded. Revisit on the symptom,
not on a schedule.

**Decision:** _______________

### B3. May the agent edit Silk itself, or Weave core?

**Spec:** T10, D76, D77.

**What is true today.** The agent writes plugins into its own root; loading
is always human-approved with the diff in view; Silk and Weave core are
refused outright.

#### Options

**B3-a — Keep the refusal.**

*Impact:* No self-improvement of the harness. The agent can extend Weave
with new suites but cannot improve the thing running it.

**B3-b — Allow it, patch-and-restart.** The agent writes a patch; the
process restarts to apply it; never a live reload.

*Impact:* The spec already identifies this as the only defensible shape if
it ever happens — and identifies the problem the graph case does not have:
**the code that would review the change is the code being changed.** A
weakness introduced into the approval gate could be introduced *by* the
thing the gate exists to restrain, and a system compromised that way cannot
be relied on to report it. This is a different risk class from everything
else in this document.

**B3-c — Allow it with a live reload.** Already rejected; recorded so the
question is not re-derived a fourth time.

**Recommendation.** B3-a. This is not a feature gap, it is a line worth
keeping. If it is ever crossed it should be its own project with its own
review, not an increment on D76.

**Decision:** _______________

---

## C. Accepted, or belonging elsewhere

### C1. A dead model server is not restarted (G6)

**What is true today.** If the server dies mid-run, the in-flight request
fails, the loop turns it into an `EventError`, and nothing restarts the
pool. The user re-pulses the graph.

**Why that is tolerable.** Silk runs are atomic and graph-pulsed — there is
no session state to recover, so a dead run is simply re-run. The spec
declines the supervisor in as many words: *"The restart itself stays out of
scope."*

**What was genuinely dangerous is already fixed.** Compaction's second
trigger reacts to a stream error, which is also what a dead server produces.
Without classification, a crash would have been answered by spending a
summarization request against the corpse and retrying.
`functions/model_errors.py` (D40) tells the cases apart, and it is also what
any future supervisor would key off.

#### Options

**C1-a — Accept.** Remaining cost is one manual re-pulse after a rare,
recoverable event.

**C1-b — Build a supervisor.** Liveness check plus restart, keyed on the D40
classifier. Moderate work, guarding an event that is rare locally and
already non-destructive. It becomes materially more attractive under A1-b,
where "one of N backends died" is a routine condition rather than a total
outage.

**Recommendation.** C1-a, and revisit only if A1-b lands.

**Decision:** _______________

### C2. Weave's internals have no version contract (G20)

**What is true today.** Silk reaches into Weave internals; Weave promises
nothing about them. `functions/weave_contract.py` is a hand-written list of
every seam Silk depends on, with the reason for each, checked once at plugin
import. A finding is a named line in the load log and a failing test in this
tree the same day something is renamed — it never blocks the load.

**What is missing.** It is a list Silk maintains, not a promise Weave makes.
It can drift the moment someone adds a dependency without recording it.

**This one is the user's on both sides** — Weave is the same codebase. That
makes it the only C item that can be *closed* rather than accepted.

#### Options

**C2-a — Keep the maintained list.** Works today. Needs discipline: a new
reach into Weave must be added to the contract, and nothing enforces that
except review.

**C2-b — Enforce the list from the Weave side.** A lint rule or test that
fails when Silk imports a Weave internal the contract does not name. Closes
the drift without Weave promising anything, and is cheap.

**C2-c — Give Weave's engine-facing modules a declared API version.** Real
work, and the benefit is not Silk-specific: it is what any future
third-party plugin would need. Turns "a list Silk maintains" into "a promise
Weave makes", which is what G20 actually asks for.

**Recommendation.** C2-b is the cheap 80%. C2-c is worth its own
conversation, scoped as a Weave change with plugin authors in mind, rather
than as Silk maintenance.

**Decision:** _______________

### C3. Writing to an importable directory grants process authority (G21)

**What is true today, stated plainly.** Every file tool is sandboxed.
`import` is not. If the agent writes a file somewhere Python can import
from, then whenever anything imports it, that file's module-level code runs
with the **full authority of the Weave process** — network, the whole
filesystem, the user's keys. However narrow the sandbox was when the file
was written is irrelevant by then.

That is not a bug. That is what importing means. A sandbox root on the
import path is a *deferred* grant of process authority, redeemable with one
`load_suite` call.

**What is built.** Both halves report rather than refuse. `import_reach.py`
names, at ToolBox evaluation, which writable roots Python will import from
and why. `mcp_reach.py` names filesystem-shaped tools on a mounted MCP
server, which sits outside Silk's sandbox entirely — a different process, so
Silk's grants and locks do not apply to it at all.

#### Options

**C3-a — Accept and keep reporting.** The current state.

**C3-b — Refuse importable roots by default, with an explicit override.**

*Impact:* **This breaks the legitimate case.** An agent authoring its own
plugin writes into an importable tree *on purpose* — that is the entire
point of D76. The override would be ticked immediately and permanently, at
which point it is a warning with extra steps.

**C3-c — Narrow what an imported file may do.** Not available. Silk cannot
sandbox `import`, and nothing short of a separate process would.

**Recommendation.** C3-a, stated honestly. The right framing is *a residual
risk that has been chosen and made visible*, not a gap that is closed. The
MCP residue is inherent for a further reason worth keeping in view: the
classifier reads someone else's tool names, so a server hiding a write
behind `sync_state` is not reported — which is why the notice says "look
like" rather than claiming a complete list.

**Decision:** _______________

---

## D. Documentation debt

Not work, but a reference that contradicts itself stops being usable as one.

### D1. D45 says mechanism C "still waits on the measurement"

It does not. The measurement ran 2026-09-19 and its outcome is recorded in
§12, three sections away, selecting A and leaving C open as additive. The
stale clause is the last sentence of D45 item 2.

**Decision:** _______________

### D2. T10 lists auto-load and auto-retry as unsettled

Both were answered 2026-09-02 — §22 q10 by the pin store
(`suite_pins.json`, digest-matched, re-approved on any edit), and q11 by
*read the traceback yes, reload no* (`record_quarantine` unpins). The
bullet's own instinct — "probably not affirmatively" — is what got built.
Only the cross-reference is wrong.

**Decision:** _______________

---

## Suggested order

1. **D1, D2** — ten minutes, and everything below then reads against a spec
   that is not contradicting itself.
2. **A1's sub-question** — decide who picks the backend, even if A1 itself
   is deferred. It is the part that is expensive to retrofit.
3. **C2** — the only item here that can be *closed* rather than accepted,
   and the one whose value extends past Silk.
4. **A1** — on the trigger: an actual second backend, or fan-out becoming
   the common case.
5. **B1, B2, B3** — no action. Record the triggers; revisit on symptoms.
6. **C1, C3** — accept, and say so in OPEN_TOPICS.md so they stop reading as
   debt.
