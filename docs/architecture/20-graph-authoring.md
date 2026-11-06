
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
crosses to the main thread by Qt's own rules — which hold only because the
author moves itself to the main thread when built: the Agent node builds it
inside `compute`, on a worker, and a queued slot runs where its object lives,
so an author left there queued every request to the thread blocked waiting
for it. The six ops dispatch to
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
- **Node access** (D73, widened) — each node has an agent access level,
  weakest first: `none` (read it, wire *from* its outputs), `connect` (also
  wire into or out of its inputs), `configure` (also `set_node_value`),
  `full` (also `remove_node`). `ACCESS_NEEDED` maps each write to the level
  it needs on the node it edits — for a wire, the destination; reading a
  node's outputs needs nothing. The level lives in the node's plugin data
  under `silk` (Weave's `Node.plugin_data`, saved, undone and duplicated with
  the node): `owner` is the uid of the Agent node that placed it, `access`
  a level the user chose. Unchosen, a node is `full` for its owner — in this
  run and every later one — and `none` for everyone else, so the user's
  graph stays out of reach until they open it up. A chosen level applies to
  every agent, the owner included, so the user can take back a node an agent
  placed. The canvas reports each node's level on `describe_graph`
  (`access`), and the tools refuse below it with the level they saw and the
  one they needed.
- **Agent access** (`widgets/agent_access_menu.py`) — the user's half: a
  submenu in every node's right-click menu, through Weave's
  `register_node_menu_extension`. It says who placed the node and offers
  Default plus the four levels, for one node or a selection; each change is
  one `NodePropertyCommand` on `set_plugin_data`, so Ctrl+Z undoes it.
- **`RunScope`** — the fallback for a seam whose nodes carry no marks (the
  control API): there a node is the agent's in full for the run that placed
  it and nobody's after. A headless graph has no menu to grant access from,
  so nothing else is reachable there.
- **`check_self_modification`** (D73) — an upstream walk from the Agent node
  at request time (`upstream_of`, `protected_nodes`, cycle-safe). The agent
  may not rewire itself, its ToolBox / ToolSet / Role / model chain, or
  anything feeding it: the evaluation model gives no coherent meaning to
  editing a node's own inputs while it sits inside `compute()`.
- **`CanvasBinding`** / `bind_canvas` / `canvas_binding` — the per-run seam,
  scope and agent uid, bound onto the ToolBox for the run and unbound at
  teardown. A binding left behind would point at a canvas nobody is driving,
  and would let the next run delete this one's nodes.

### `functions/tools/graph_authoring.py` — the tools (D69, §22 q9)

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

`NodeWhitelistWidget` is a checkable tree of registered node classes in the
levels of the canvas's *Browse Nodes* menu — category, the nodes placed
directly in it, then its subcategories — bound to the ToolBox node's
`placeable_nodes` port and shown in a dialog behind the gear on the `graph`
tool category. A node row is ticked or not; a level row is ticked, unticked
or partly ticked, counts what is ticked beneath it, and ticks or unticks all
of it. A filter narrows the tree, and the right-click menu selects all, none
or the inverse — over the whole tree or the level under the cursor, and only
over the nodes the filter shows. Its value is a sorted list of class names,
so the grant travels in the saved graph and in a preset; unlike file grants
(D35) it carries no secret and no filesystem authority. There is no "allow
all" value: *Select All* names every class shown, one by one, which has to be
a deliberate act and does not reach a class registered later. A ticked class
the registry no longer has is listed under *Not registered*, where unticking
it drops it.

The tree subscribes to `NODE_REGISTRY`'s listener and coalesces a burst of
registrations into one rebuild (D74), so a suite hot-loaded into the running
session becomes tickable there — the tools already resolve the registry live,
and without this the widget would be the only place that had not noticed.

### Configuration: `list_node_settings` / `set_node_value` (§22 q9)

A node an agent can place but not set up is a skeleton a human has to finish,
which was the question: the surface is *every widget type*, so it needed its
own decision rather than an extension of D69. The answer is a narrow typed
setter over `WidgetCore` bindings, and the narrowing is where the safety is:

- **Values only.** `SETTABLE_TYPES` is string, int, float, bool and the
  path-like strings. Everything else on a node — a model, a toolset, a
  permissions object, a list of roots — comes from a connection, and the
  refusal says so. That keeps the authority-bearing ports (`file_permissions`,
  `dirpath_list`) unwritable *by construction* rather than by a rule someone
  has to remember.
- **INPUT and BIDIRECTIONAL roles, and INTERNAL where no authority is at
  stake.** DISPLAY is computed output and never a setting. INTERNAL is
  node-local state, and it is two things: the text box on a String Input or
  the number field on an Int Input — a value node's whole point, and the
  agent's to type into — and the permission switches — a ToolBox's **Plugin
  authoring** checkbox is an INTERNAL widget, so an agent that could write it
  could grant itself authority by ticking a box. `grants_authority` tells
  them apart by the node rather than the widget: a node from the Silk
  package, or one with an output in `AUTHORITY_TYPES` (a toolbox, toolset,
  role, plan, model, permissions, roots, MCP servers, toolchains, knowledge
  graph), keeps its
  INTERNAL widgets locked. Both resolvers (the canvas and the control seam)
  put the answer on each setting row, and a row without one is refused, so
  a resolver that forgets it fails closed. Both roles stay *readable*: `list_node_settings` returns every widget
  with its current value, and an agent reading a display port is reading a
  result, not editing one.
- **A wired port is not written behind the wire.** The upstream value wins
  over the widget, so writing one would be a change the graph then discards.
- **`configure` access** (D73, widened): every node the agent placed, and a
  user's node only once the user grants it — never the agent's own execution
  path, which no level opens.
- **One call, one undoable command.** `WidgetValueCommand`, applied then
  pushed — the order `_place` already uses, because the undo manager records
  what happened rather than performing it.

The result reports what the widget *holds afterwards*, not what was asked for.
A combo box takes the items it has; an agent that believes it set a value the
user never sees is the failure this avoids, so a rejected or clamped write
comes back `ok` with the real value and a note saying it changed.

Deliberately still not in v1: moving or resizing nodes, saving or loading
graph files, and anything touching another graph.
