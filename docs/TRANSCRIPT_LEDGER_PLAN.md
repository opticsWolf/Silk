# Transcript ledger: the full context as a tree, on Macrame

Status: design, 2026-11-17. Nothing built. Decision numbers are to be
assigned when this lands in [DESIGN_SPEC_DRAFT.md](DESIGN_SPEC_DRAFT.md)
§17 (Persistence -- the Macrame ledger).

## 1. Problem

Silk keeps no complete, durable record of what a model was actually shown.
What exists today:

| Where | What | Gap |
|---|---|---|
| Agent node state (`nodes/agent.py`, `self._history`, saved as `silk_agent.history`) | The live message list, tool calls and results included | Only the current window: compaction rewrites its head (`GraphEngine.replace_history_prefix`), and *Clear context* empties it |
| `<root>/.silk/spill/*.md` (`functions/spill.py`) | Large tool results; the compacted transcript when a spill writer is wired | Rendered text, not messages; swept after 14 days / 256 MB |
| `~/.weave/silk/runs/*.jsonl` (`functions/event_sink.py`) | Every event of a run, in order | Metadata only by design: text becomes its length, tool args their key names and size; newest runs only |
| `<root>/history.macrame` (`functions/remember.py`, opt-in hook) | User and assistant turn text, tool names, touched paths | No tool args or results, no system prompt, no message structure; filtered by `min_chars`; keyed by run, not by conversation |

So a compacted run cannot be replayed, a conversation cannot be rewound
to an earlier point, and there is no way to try two continuations of the
same context. This plan adds a **transcript ledger** that records the
context verbatim as a tree, and makes the Agent node able to rebuild its
history from it.

## 2. Goals and non-goals

Goals:

- **Verbatim**: every message the model is sent or sends -- system prompt
  per request, user, assistant (with native `tool_calls`), tool results,
  compaction summaries -- reconstructible exactly.
- **Tree**: rewind and fork are pointer moves, never copies or deletes.
- **Rebuildable**: the Agent node's history can be rebuilt from the
  ledger at any node of the tree, so rewind/fork change what the model
  sees next, not only what a viewer shows.
- **Time travel**: "the context as it stood at 14:00" is one read
  (Macrame's bitemporal `as_of_valid` / `as_of_recorded`).
- **Never fails a run**: like `remember`, a broken transcript disables
  itself for the run and says so once.

Non-goals:

- Replacing `history.macrame` / `recall`. Memory search stays on the
  turns `remember` writes; the transcript is not embedded (§8).
- Cross-machine sync or sharing. Files stay under the sandbox root.
- Changing how compaction decides *what* to drop (`functions/compaction.py`
  keeps its planner, I9 cut rule and summary prompt).

## 3. Where the record is taken

Not from hooks. The hook events (`after_model_response(text, …)`,
`after_tool_execute`, `after_compaction(summary_ref, …)`) carry text or
metadata, not the message dicts, and they say nothing about the system
prompt or a history cleared from the UI.

The engine's history list is the single source of truth, and it changes
in exactly three places in `functions/graph_engine.py`:

| Mutation | Method | Recorded as |
|---|---|---|
| append a user/assistant turn (assistant carries `tool_calls` on the native path) | `append_message` | one `message` entry |
| append a native `tool` turn | `append_tool_result` | one `message` entry, child of the assistant turn holding the call |
| replace the head with one summary turn | `replace_history_prefix` | one `compaction` entry (§5.4) |

Plus two in the Agent node: *Clear context* (`nodes/agent.py`, the
`self._history.clear()` path) and `restore_state`.

The engine gets an optional **history observer** (a small protocol in
`functions/protocols.py`, beside `AgentEngine`), called after each of the
three mutations with the entry (or the cut count and summary). The
transcript writer implements it. No observer, no cost: the default is
`None`, exactly as the engine behaves today.

The **request** is recorded too: before each model call the loop builds
the outgoing messages (`GraphEngine.build_messages`). The writer stores a `request`
entry naming the leaf it was built from, the system prompt (by hash, §6),
the model handle and the generation settings. That is what makes "what
was the model actually shown" exact rather than inferred.

## 4. Data model

One file per sandbox root: `<root>/transcript.macrame`, opened through
the existing `LedgerRegistry` (one handle per file per process, D62).
Plan discovery already skips `history.macrame` by stem; it must skip
`transcript` the same way (`functions/plan_discovery.py`, `HISTORY_STEM`).

Concepts (ids are stable strings, like the task ledger's):

| Concept | Id | Payload |
|---|---|---|
| conversation | `conv:<uuid>` | title, created_at, agent node uuid, parent conversation + fork point (for forks across conversations) |
| head | `head:<conv>` | the current leaf id. **Superseded** on every append, rewind or fork -- never edited, so its history is the conversation's navigation log |
| message | `msg:<conv>:<seq>` | `role`, `content` (or a blob ref), `tool_calls`, `tool_call_id`, `name`, `stats`, `run_id`, `at` |
| request | `req:<conv>:<seq>` | leaf it was built from, system-prompt hash, model id, generation settings, token counts |
| compaction | `cmp:<conv>:<seq>` | summary text, `first_kept` message id, tokens before/after, reason, file lists |
| branch summary | `bsum:<conv>:<seq>` | summary of an abandoned branch, from-leaf, common ancestor |
| blob | `blob:<sha256>` | large content (§6), referenced by hash |

Edges (Macrame types are `[A-Z0-9]+`, as in `ledger.py`):

| Edge | From → to | Meaning |
|---|---|---|
| `PARENT` | message/compaction/branch summary → its parent | the tree; the root's first message has none |
| `INCONV` | entry → conversation | membership, for listing |
| `SENT` | request → leaf message | the context it was built from |
| `REPLACES` | compaction → each message it stands in for | lets a viewer show what a summary covers without walking |
| `INRUN` | message → `run:<run_id>` | joins `history.macrame` / the events port (D60 identity) |
| `TOUCHED` | message → `file:<path>` | same file keys `remember` uses, so "which runs touched this file" works across both ledgers |

`seq` is a per-conversation counter, so ids sort in write order and two
branches never collide.

## 5. Operations

### 5.1 Append

Each observed mutation writes its concept, then one atomic batch of its
edges (`write_bulk_atomic`, as the task ledger does, D65), then
supersedes `head:<conv>`. A crash between the two leaves an orphan
message that reads as "written, not yet on the path" -- the same honest
failure `record_turn` documents.

### 5.2 Build context (the read the Agent node needs)

Walk `PARENT` from the head to the root, reverse, then apply compactions:
the newest `compaction` on the path stands for every message before its
`first_kept`; older compactions are already inside its summary (the
compactor carries previous summaries forward, `functions/compaction.py`).
The result is exactly the list `GraphEngine.history` held, so
`restore_state`-style loading can take it unchanged.

With `as_of`, the same walk reads the head *as it was* -- rewind in time,
not only in the tree.

### 5.3 Rewind

Supersede `head:<conv>` with an earlier message id. Nothing is deleted;
the abandoned branch stays readable. Optionally write a `branch_summary`
of the abandoned part, as a child of the new head, so the next run is not
blind to work that was done there (pi does this on `/tree`; see §9).

Rewinding to a **user** message puts its text back in the prompt box and
moves the head to its parent, so the user can edit and resend -- the edit
then forks naturally.

### 5.4 Compaction

The compactor is unchanged; the observer sees `replace_history_prefix`
and writes a `compaction` concept whose `PARENT` is the last dropped
message and whose `REPLACES` edges name the dropped range. The original
messages stay in the ledger. Today's `.silk/spill` archive of the
dropped transcript becomes redundant when the transcript ledger is on;
keep it only when it is off.

### 5.5 Fork

Two shapes, both without copying messages:

- **In place**: a new head for a new conversation whose first entry's
  `PARENT` is the fork point in the source conversation. Building its
  context walks across the conversation boundary.
- **Detached** (export): copy the active path into a new file, for
  sharing or archiving. Explicit only, because it duplicates content
  (and any secret in it).

### 5.6 Restore into the Agent node

Three entry points on the Agent node, all rebuilding `self._history`
from §5.2:

- **Resume**: a saved scene reopens on its conversation's head.
- **Rewind to …**: pick a message in a history view; §5.3, then rebuild.
- **Fork from …**: §5.5 in place, then rebuild; the node now writes to
  the new conversation.

The node's saved `silk_agent.history` stays as it is today: it is what a
scene restores when the transcript ledger is off or unavailable.

## 6. Size

Tool results and file reads are the bulk. Content above a threshold
(the spill threshold, `DEFAULT_THRESHOLD = 4000` chars, is the natural
default) goes to a **blob**: stored once by SHA-256 under
`<root>/.silk/transcript/blobs/<aa>/<sha>`, referenced from the message.
Identical content (the same file read twice) is stored once. System
prompts are blobs too, so a request records a hash, not 10 kB of
repeated text.

Blobs are not swept by age the way spill files are: a blob a message
references is part of the record. They go only when their conversation
is purged (§7).

## 7. Secrets and deletion

A verbatim record holds whatever a tool read: keys in a `.env`, tokens in
a command's output. The event sink keeps content out for exactly this
reason (`event_sink.py`: "Metadata only, always"), and Macrame never
deletes (Doctrine V). So:

- **Opt-in per Agent node**, off by default, like the `remember` hook.
  The node's Info says when it is on and where the file is.
- **Under the sandbox root**, read through the same grants as
  `history.macrame` (I6): an agent that cannot read a project's files
  cannot read its transcript.
- **Redaction without deletion**: a message's content can be superseded
  by a redacted version. Because content above the threshold lives in a
  blob, deleting the blob file removes the plain text from disk while the
  ledger keeps the fact that a message existed and was redacted. Small
  inline content is the open question (§11 q2).
- **Purge a conversation**: close it, delete its blobs, and supersede its
  concepts with tombstones. The ledger file itself still holds the old
  versions until Macrame offers compaction of history; say so in the UI
  rather than promise more than is true.
- Never written to `~/.weave` -- user-wide storage would cross the
  per-project boundary that memory already respects (`recall_tool.py`).

## 8. Relation to remember and recall

Independent. `remember` keeps writing filtered turns to `history.macrame`
and `recall` keeps searching them (FTS5, plus vectors when an embedder is
wired). The transcript is not embedded: one or two ledger writes per
message are cheap, an embedding per message is not.

The join is by key: transcript messages carry the same `run:<run_id>`
and `file:<path>` keys, so a recall hit can open the exact transcript
around it.

Later, `remember` could become a projection of the transcript (one
writer, two views) instead of a second hook. Not in this plan.

## 9. Comparison: pi

pi (the coding agent from `pi-mono`) keeps one JSONL file per session
under `~/.pi/agent/sessions/`, every entry with `id` / `parentId`:
messages, tool calls with full arguments, tool results, model and
thinking-level changes, compaction and branch-summary entries, labels.
The context is rebuilt by walking from the leaf to the root; `/tree`
moves the leaf (optionally writing a summary of the abandoned branch);
`/fork` and `/clone` copy the path into a new file. Compaction writes a
summary entry and keeps the originals in the file. Long-term memory is
`AGENTS.md` files; there is no memory search and no embeddings.

What this plan takes from it: the tree, the moving leaf, compaction and
branch summaries as ordinary entries, and rebuilding context from the
record rather than from a side copy.

What Silk already has: pi's cut rule (never cut at a tool result) is
Silk's I9 (`round_boundaries`, enforced again in
`replace_history_prefix`), and both carry the previous summary and the
read/modified file lists forward.

What Macrame adds over JSONL: time-travel reads, typed edges shared with
the task and history ledgers, and one write actor per file (D64).

What pi leaves open that this plan must not: per-project placement and
secrets (pi's docs cover neither, and its share commands upload sessions
as they are).

## 10. Phases

1. **Observer and writer.** The `HistoryObserver` protocol, the three
   engine call sites, `TranscriptLedger` in `functions/ledger.py` (append,
   head, build context), blob store. Agent node opt-in row. Tests:
   build-context equals `engine.history` after every mutation, across
   native and fenced tool calls and across compactions.
2. **Restore and rewind.** Resume a scene from the head; rewind to a
   message (with and without branch summary); *Clear context* as a new
   root rather than a deletion.
3. **Fork.** In-place fork across conversations; detached export.
4. **Viewer.** A tree view (Plan Viewer–style dock or a node) showing
   branches, compactions with what they replaced, and redactions.
5. **Redaction and purge.** §7, with the UI wording that says what is
   and is not removed.

Each phase ends green on Silk's suite. Phase 1 is useful on its own:
a verbatim, compaction-proof record of every run.

## 11. Open questions

1. **Threading.** The ledger hang (`test_only_one_agent_wins_a_contested_claim`,
   eight threads writing one task ledger) is not explained yet. A
   per-message writer runs on agents' worker threads, several agents per
   root. Phase 1 needs that hang understood, or the writer funnels its
   writes through one thread per file.
2. **Small secrets.** Content under the blob threshold is inline in the
   ledger and cannot be removed from the file. Lower the threshold for
   tool results (always blob them), or accept it and say so?
3. **Streaming.** Record the assistant turn only when it completes (as
   the engine appends it), or also partial text from a stopped stream?
   Today a stopped turn is appended with what arrived; recording that is
   the honest default.
4. **One file or two.** `transcript.macrame` beside `history.macrame`,
   or new kinds inside `history.macrame`? Separate keeps purge and size
   independent of memory; one file keeps edges between them local.
5. **Multi-agent.** A conversation per Agent node is clear. A worker
   spawned by an orchestrator: its own conversation with a `PARENT` link
   to the orchestrator's message that spawned it, or a branch of it?
