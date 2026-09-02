
## Graph authoring — the agent places nodes

The first Silk tool family whose effect is on **Weave itself** rather than on
files, a model, or a task store (spec §18, D69–D74). An agent may place nodes
from a user-editable whitelist, wire them, and clean up after itself — and
what is engineered here is mostly what it *cannot* do.

### `functions/blocking_seam.py` — the waiter both seams are made of

`BlockingSeam` is D49's mechanism, lifted out of `DecisionSeam` so its second
user is not a second implementation of the ordering rule:

> Write the outcome under the lock, then set the event; the waiter re-reads
> under the lock before acting.

Without that rule Stop, a timeout and a real answer collapse into one
"something woke me" and the seam cannot say which happened. The class owns
correlation (one `threading.Event` per request id, never a broadcast),
cancellation (`cancel`, `close`), the timeout, and `DriveGate` — the test-mode
gate that parks the seam at `ask` / `wait` / `resolve` / `wake` so a race can
be driven in order rather than hoped for.

Every failure path fails closed (D36) with a **named cause**:
`CAUSE_ANSWERED`, `CAUSE_CANCELLED`, `CAUSE_TIMEOUT`, `CAUSE_NO_ANSWERER`,
`CAUSE_TRANSPORT`. "Denied by the user" and "nobody was there to ask" are
different facts even when they have the same effect, and the seam's owner
turns the name into a refusal the model can read.

`DecisionSeam` is now a subclass: `await_decision` is `submit(...)` with a
`failed` builder, `resolve` is `commit`. The 58 existing gate and seam tests
pass unchanged, which is the claim `test_both_seams_are_the_same_waiter`
pins by identity (`DecisionSeam.submit is MainThreadCall.submit`).

### `functions/main_thread_call.py` — the canvas seam (D70)

The second user. A tool runs on the agent's `ThreadedNode` worker (under
`asyncio.to_thread`); Qt says the canvas may only be touched on the main
thread, and the existing worker→main channels (`emit_stream`, `pulse`) are
one-way and return nothing.

`MainThreadCall.call(op, **args)` blocks the worker and returns a
`CallResult`. Two facts the result keeps apart:

- `ok` — the main thread ran and the op succeeded;
- `performed` — the main thread ran *at all*. A handler that refuses ("no
  such port") is a different thing from a request that never arrived, and
  `failure_text()` says which.

`serve(request, handler)` runs on the main thread and turns a raising handler
into a refusal — this executes inside a Qt slot, where an escaping exception
would take out the event loop and leave the worker to time out for no reason.

### `nodes/graph_canvas.py` — the Qt resolver

`CanvasAuthor(QObject)` is the only piece that touches the scene. The seam
delivers into a `Signal(object)` with `Qt.QueuedConnection`, so the request
crosses to the main thread by Qt's own rules, and the six ops dispatch to
methods that go through the canvas's **own undo commands** — never raw scene
manipulation:

| Op | Command |
|---|---|
| `place_node` | `AddNodeCommand` (after `canvas.add_node`, the path `Canvas.spawn_node` takes) |
| `connect` | `PortUtils.are_compatible` → `ConnectionFactory.create` → `AddConnectionCommand` |
| `disconnect` | `RemoveConnectionsCommand` |
| `remove_node` | `RemoveNodesCommand` with `capture_node_snapshot` + `capture_node_connections` |

That is D72, and it is the primary safety property: **the human undoes the
agent's edit with the same gesture they undo their own**, and the edit shows
up where a user already looks to see what happened. `UndoManager.push` opens
a macro and closes it a tick later once the evaluation fence is clear and the
macro has stopped growing, so a call that creates several items lands as one
`CompoundCommand` — one Ctrl+Z per *tool call*, not per primitive. A node with
no scene refuses (`no canvas`) instead of guessing.

### `functions/graph_author.py` — the policy half, Qt-free

Everything that decides *whether* an edit may happen lives here, importable
without Qt and unit-testable against a fake resolver:

- **`Whitelist`** (D71) — default-deny. An empty list means the tools register
  and every placement is refused; `list_placeable_nodes` returns nothing.
  `narrowed()` composes like every other grant (I6: a ToolSet or Role may
  remove entries, never add), and `missing()` names whitelisted classes that
  are no longer registered — surfaced at the node, not as a refusal an agent
  hits halfway through building something.
- **`RunScope`** (D73) — what *this run* placed and connected. `remove_node`
  and `disconnect` touch nothing else: an agent may clean up after itself, it
  may not prune the user's graph. Removing a node forgets its edges with it.
- **`check_self_modification`** (D73) — an upstream walk from the Agent node
  at request time (`upstream_of`, `protected_nodes`, cycle-safe). The agent
  may not rewire itself, its ToolBox / ToolSet / Role / model chain, or
  anything feeding it: the evaluation model gives no coherent meaning to
  editing a node's own inputs while it sits inside `compute()`.
- **`CanvasBinding`** / `bind_canvas` / `canvas_binding` — the per-run seam,
  scope and agent uid, bound onto the ToolBox for the run and unbound at
  teardown. A binding left behind would point at a canvas nobody is driving,
  and would let the next run delete this one's nodes.

### `functions/tools/graph_authoring.py` — the six tools (D69)

`attach_graph_tools(toolbox, sandbox, whitelist=())` registers
`list_placeable_nodes` and `describe_graph` (`risk="low"`), `place_node` and
`connect` (`medium`), `disconnect` and `remove_node` (`high`,
`requires_approval=True` — the existing gate covers them, no new approval
machinery). Reads are what make placement possible at all: an agent that
cannot see the graph cannot place a node *relative* to it or know which ports
are free. The descriptions the model reads are the ones already written for
the node UI — `node_description`, `node_tags`, port `datatype` — so nothing
had to be authored twice.

Because they are ordinary `ToolBox` registrations they inherit hooks,
`tool_events`, role enforcement and the gate exactly like every other tool
(D74/D56): a graph-authoring call appears in the Hook Monitor, a delegated
worker inherits the whitelist through its toolset, and the history ledger
(§17) records placements as ordinary run facts.

### `widgets/node_whitelist.py` + the ToolBox node (D71)

`NodeWhitelistWidget` is a checkable tree of registered node classes grouped
by category — the same shape as the tool tree — bound to the ToolBox node's
`placeable_nodes` port. Its value is a sorted list of class names, so the
grant travels in the saved graph and in a preset; unlike file grants (D35) it
carries no secret and no filesystem authority. There is no "allow all"
checkbox: selecting everything is possible, but it has to be a deliberate act.

Deliberately not in v1: setting widget values, moving or resizing nodes,
saving or loading graph files, and anything touching another graph.
