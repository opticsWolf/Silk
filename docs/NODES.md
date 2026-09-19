# Silk Node Reference

Every Silk node searches under the **Silk AI** category, in one of four
submenus: **Loaders**, **Agents**, **Display** and **Configuration**. Ports
are listed as they are registered; `exec` ports are trigger pulses.

*(The category was plain "AI" until 2026-09-19, and the read-only surfaces
— Hook Monitor, Plan Viewer, Task Hub, Chat Log Display — sat under Weave's
own **Display** category. Both were changed so a Silk install adds one
menu, not entries scattered through a host's.)*

## Model

### GGUF Loader — `nodes/gguf_loader.py` (`GGUFLNode`)
Loads a GGUF model via llama.cpp into the shared pool (thread-safe; ejects
the previous model when re-run).

| Direction | Port | Type |
|---|---|---|
| in | `model_path` | `filepath` |
| in | `prompt_cache` | `filepath` |
| out | `model_obj` | `model_handle` |
| out | `pool_info` | `dict` (live pool stats) |

The handle carries `supports_tools` when the GGUF's own chat template was
written for structured tool calling — the probe reads the template and
looks for it being handed a `tools` list *and* rendering `tool_calls`.
That is what puts the agent on the native protocol instead of the text
fence. A file with no chat template at all (an embedding model, a vision
projector) is "unknown" rather than "no", and unknown uses fences.

**Group Requests by Agent** (advanced mode, on by default) is D47
mechanism A. One server holds one resident KV context, so two agents
taking turns overwrite each other's prompt cache and each round
re-prefills everything past the system prompt: measured at 74.8% prefix
reuse for one conversation against **0.4%** for two running at once, and
the same work in 1.1 s against 4.5 s. With grouping on, two conversations
reuse exactly what one does. It changes only the *order* of a queue that
already exists -- the server serialises every request anyway (D43) -- so
it costs no throughput, only a short wait at a switch, and that wait is
capped by what a lost prefix was measured to cost. Turn it off if your
prompts are small enough that prefill does not matter; the Pool Monitor's
*Affinity* row reports the wait it costs beside the reuse it buys.

### Model Endpoint — `nodes/model_endpoint.py` (`SilkModelEndpointNode`)
A model that runs somewhere else, wired where the loader would go (D45).
Emits the same `model_handle`, so Agent, Worker and the ToolBox's
embedding input take it without knowing the difference.

| Direction | Port | Type |
|---|---|---|
| in | `provider` | `string` (preset key) |
| in | `base_url` | `string` |
| in | `model` | `string` |
| in | `credential` | `string` (a **name**, never a value) |
| in | `context_length` | `int` (0 = unknown) |
| in | `supports_tools` | `bool` (native tool calling, off) |
| out | `model_obj` | `model_handle` |

Presets: LM Studio, llama.cpp server, Ollama, vLLM, LiteLLM proxy,
OpenRouter, and **Custom** for anything else that serves
`/v1/chat/completions` — a hosted gateway, your own proxy, a colleague's
box. A preset only fills *empty* fields, so "this provider, my host" is
one edit rather than a re-type.

**What the endpoint already knows, it fills in.** The `/models` request
that lists the models also carries, on a gateway that says so, each
model's price, its context window and whether it takes a `tools` field.
The node reads all three from that one response: the price goes on the
handle (and is what a `cost=` budget is measured against — see D88), and
the other two fill the fields *if they are empty*. A value you typed
always wins, because you can see a proxy in the way that a catalogue
cannot. LM Studio and llama.cpp quote no price, which the status line
says plainly rather than implying the run is free.

**Native tool calling** is a checkbox, and it is off. On, the agent's
tools go in the request's `tools` field; off, they go in a text fence.
The two failure modes are not symmetric — a server that does not accept
`tools` refuses the whole request, while fences merely cost accuracy on a
model that could have done better — so the protocol that works everywhere
is the one you get without asking. Most hosted gateways support native;
small local models often do not. Verified live against LM Studio.

Three behaviours worth knowing:

- **It asks before it answers.** Connecting probes `/models`, so a wrong
  URL or an unset key shows up here rather than as a failed agent run
  three nodes downstream. One model advertised and none chosen means that
  one is used; several means it asks, because picking one of nine on a
  metered gateway is picking someone's bill. A model the endpoint does not
  list is still used — gateways route unlisted aliases — and logged.
- **The credential is a name** (D22). The field holds the name of an
  environment variable or of an entry in `~/.weave/silk/secrets.json`,
  resolved at connect time into the request headers. The status line says
  the name resolved; it never shows the value, and neither does the handle,
  so a saved graph stays shareable.
- **An unknown context window stays unknown.** Most endpoints do not
  advertise one. Compaction (D25) needs a real denominator, so 0 means
  "off" rather than a guess that would summarise too early or overflow.

**Why there is no litellm dependency.** OpenRouter, LM Studio, vLLM,
Ollama and litellm's own proxy all speak the OpenAI chat API, so one HTTP
client reaches all of them. For providers that do not (Anthropic, Gemini,
Bedrock), run the **litellm proxy** and point this node at it: the
provider-specific translation stays in a process that specialises in it,
and Silk's model layer stays one wire format wide. An Unsloth fine-tune is
an ordinary model once vLLM or llama.cpp is serving it.

### Model Fallback — `nodes/model_fallback.py` (`SilkModelFallbackNode`)
One model behind another (D89). When the primary fails for good — the
server is gone, the key was revoked, the model was retired, or three
spaced retries all came back 503 — the run continues on the fallback
instead of ending.

| Direction | Port | Type |
|---|---|---|
| in | `primary` | `model_handle` (tried first) |
| in | `fallback` | `model_handle` (tried when the primary fails) |
| out | `model_obj` | `model_handle` (the chain, as one handle) |

What it emits *is* a model handle, so the Agent downstream sees one model
and runs the way it always has. Wire two of these in series for a chain
three deep; the flattening is done here, so nothing downstream recurses.
The same model wired into both ports is deduplicated — falling back to
the model that just failed cannot help.

**A chain is as capable as its weakest link**, and the status line says
so, because the trade is real:

| | rule | why |
|---|---|---|
| native tool calling | only if **every** member has it | the transport is chosen once, before the first request, and the system prompt is written to match it |
| context window | the **smallest** known one | compaction plans its cuts against it, and a conversation grown to fit the primary must still fit whatever catches it |
| price | only if **every** member quotes one | a cost cap that stops binding when the cheap fallback takes over is not a cap (D88) |

So a fence-only local model behind a native-tools gateway puts the whole
chain on fences. That is a trade, not a mistake — the status line states
it and lets you decide, rather than making it quietly.

**What a switch does not do.** It does not spend a round (`max_rounds`
bounds the model's reasoning, and being handed a dead server is not a
thought), it does not restart the conversation, and it does not happen
after tokens have already reached you — a second model continuing over a
partial answer would splice two voices into one turn. It also never
happens on a context overflow, which is compaction's (D40): the next
model's window is no larger. The fallback starts with a retry budget of
its own.

**What it does do** is yield an `EventModelSwitch` on the `events` port,
and the Agent node's status line says which model took over. The answer
from that point comes from a different model at a different price; a run
that finished on the cheap fallback should not look identical to one that
finished on the paid primary.

## Tool assembly

### Silk ToolBox — `nodes/toolbox.py`
The registry of **all** tools an agent network may use: sandbox roots (hard
ceiling), toolchain packs, per-tool selection and details.

| Direction | Port | Type |
|---|---|---|
| in | `sandbox_roots` | `dirpath_list` |
| in | `toolchains` | `toolchains` |
| in | `mcp` | `mcp_servers` |
| in | `plan` | `silk_plan` |
| in | `embedding_model` | `model_handle` |
| out | `toolbox` | `silk_toolbox` |
| out | `root_paths` | `dirpath_list` |

**Sandbox roots are wired, never typed.** There is no picker on the node:
a `Folder` node for one root, a `Folder List` for several — the
`dirpath` → `dirpath_list` cast wraps the single case on connection, so
both fit the one port. The roots are the hard ceiling of the whole graph,
and a ceiling that can also be set inside the node is a ceiling you
cannot read off the canvas. Unwired, the node builds nothing and says so.

**The tool tree is the only selector.** There used to be a row of group
checkboxes above it — File Read, File Write, Ripgrep, Task Planning and
the rest — and they asked the same question the ticks already answered,
in a way that could disagree with them. They are gone. A group is
attached because something in it is ticked, and that is the whole rule
(`groups_for`). Everything else falls out of the same selection:

* the sandbox is **writable** because a write tool is ticked
  (`WRITING_GROUPS`), not because a separate switch says so;
* **plugin authoring** is on because a `suite_tools` tool is ticked, and
  that is what adds `~/.weave/plugins` to the sandbox as a writable root;
* **graph authoring** mounts because a graph tool is ticked *and* the
  whitelist behind its gear names at least one class.

**The tree is populated before anything is wired.** It is seeded from a
static catalog (`functions/tool_groups.py`) built once by running every
attacher against a throwaway sandbox and recording what it registered —
nothing is executed, only described. The root is usually the thing being
wired *because* tools are wanted, so a tree that filled in only after the
root arrived was a tree that could not be set up first. `file_read` and
`ripgrep` start ticked; writing, planning, memory, graph authoring and
plugin authoring are each a different request, and start off.

**Unticking is reversible.** The tree lists what is *offered*, which is
the static catalog plus whatever the last build added dynamically — not
what survived the narrowing. Reading it off the finished box is what used
to make an unticked tool vanish from the tree altogether: the box no
longer had it, so the catalog no longer mentioned it, so there was
nothing left to tick back on. For the same reason a group with nothing
ticked keeps its row.

The narrowing is the **last entry of the build recipe**, not something
the node applies to its own output, so every ToolSet derived from this
box replays it. Without that a derived set would come back holding tools
its ToolBox had been told to drop.

A tool the tree has never shown cannot have been unticked, so the node
remembers which ones it has offered (`seen_tools`, saved with the graph).
The static tools are seen at construction; a toolchain pack or an MCP
server's tools are seen when they first arrive, and arrive **ticked** —
otherwise they would be pruned on the same evaluation that created them,
and wiring a toolchain would appear to do nothing.

Toolchain tools are ordinary tools: the packs register on the box like
every other attacher and appear in the same tree under their own
categories (`code`, `lint`, `build`), so they are ticked, filtered and
inherited exactly like the file tools.

**Some groups carry settings, and they sit on the group's own row.** A
category row with a ⚙ opens its configuration on double-click, or from
the row's context menu — the same affordance the Hooks list uses. Only
graph authoring has one today (*Allowed node classes…*), and it is there
rather than in a field of its own precisely because it is not a
preference: it is the grant those tools run under, and a grant belongs
beside the tools it governs.

**Hooks start on.** Every hook in the catalog is ticked by default except
the four that can refuse a call or stop to ask — `tool_approval`,
`signoff`, `tool_budget`, `task_audit` (`GATING_HOOKS`). Observation is
what people wish they had switched on *after* a run, and a hook that was
never ticked leaves nothing to go back to; a gate you did not choose is
how people learn to switch hooks off wholesale. Each default degrades on
its own when a dependency is missing — `remember` without the `ledger`
extra logs that it did not attach rather than failing the build.

Wiring a `Silk Task` node into `plan` names the plan the task tools work
on. Left unwired, they discover the newest plan under the sandbox root —
which is how several agents share one plan, and why two unrelated plans in
one root used to find each other (D23).

Ticking **`recall`** mounts memory search: keyword search
over the turns and runs remembered in this sandbox root's history ledger,
including ones from earlier sessions and ones compaction dropped (§17,
D66). It needs the `ledger` extra (`pip install macrame-db`); without it
the tool registers and says so rather than quietly returning nothing.

**Placeable nodes** is the graph-authoring grant (§18, D71), and it lives
behind the ⚙ on the tree's `graph` row. Tick the graph tools
(`list_placeable_nodes`, `describe_graph`, `list_node_settings`,
`place_node`, `connect`, `set_node_value`, `disconnect`, `remove_node`)
*and* name at least one node class there, and the pack mounts. Leave the
whitelist empty — the default — and the pack stays out of the prompt
entirely rather than mounting eight tools that refuse everything; the
status line says so, because ticked-but-ungranted is a half-finished
setup rather than a safe one. No agent fed by this ToolBox can build
graph at all until a class is named. Every edit an agent makes
goes onto the canvas's own undo stack, so one Ctrl+Z takes back one tool
call; destructive calls reach only what that run itself placed, and no
mutation may touch the agent, its tool chain, or anything upstream of it
(D72, D73). The list travels in the saved graph and in presets: it carries
no secret and no filesystem authority.

**Plugin authoring** (§19) is the `plugins` category of the tree: ticking
`list_suites`, `load_suite`, `reload_suite` or `request_relaunch` lets
the agent write node suites into `~/.weave/plugins` and load them into
the running session. It adds
that directory to the sandbox as the only writable root (unless file
writing is already on). Every load asks you, every time,
and shows you the diff of what this run wrote — no Role, preset or
grant can pre-approve it, because importing runs that code with the
full authority of the Weave process. The tools mount, and the load verb
is what asks; unlike graph authoring there is no separate grant to name
first, because the answer is given per load and never in advance. A state-version finding
(WV520–WV522) stops the load before you are even asked: it means saved
graphs would not survive it. Weave core and Silk stay read-only.

Which backend stores the plan is the environment's business, not the
node's: `SILK_TASK_BACKEND=ledger` puts it on the Macrame ledger, the
default keeps the SQLite store, and both answer the same protocol — the
Plan Viewer, the Task Hub and the sign-off flow cannot tell which
answered.

### Silk ToolSet — `nodes/toolset.py`
Selects a subset of ToolBox tools for an agent, with optional per-toolset
sandbox permissions and named presets.

| Direction | Port | Type |
|---|---|---|
| in | `toolbox` | `silk_toolbox` |
| in | `permissions` | `file_permissions` |
| out | `toolset` | `silk_toolset` |
| out | `permissions` | `file_permissions` |

### Silk MCP Server — `nodes/mcp_server.py`
Connects to one MCP server and shares the **live session** with every
downstream agent — one handshake per server, not per agent (D19). The
credential field holds the *name* of an environment variable or an entry in
`~/.weave/silk/secrets.json`; no secret is ever stored in the graph (D22).
Servers chain through `mcp_in` like toolchains, and every tool is prefixed
with the server id (D21).

| Direction | Port | Type |
|---|---|---|
| in | `mcp_in` | `mcp_servers` |
| in | `server_id` | `string` |
| in | `transport` | `string` |
| in | `command` | `string` |
| in | `args` | `string` |
| in | `url` | `string` |
| in | `credential` | `string` |
| out | `mcp` | `mcp_servers` |

### Silk MCP Aggregator — `nodes/mcp_aggregator.py`
Checkbox tree over every server on the wire: a category row is a server, a
leaf is one tool (D20). Unchecking records an exclusion — it never closes a
session, because the sessions belong to the MCP nodes and a toggle should
not cost a handshake.

| Direction | Port | Type |
|---|---|---|
| in | `mcp_in` | `mcp_servers` |
| out | `mcp` | `mcp_servers` |

### Silk Task — `nodes/task.py` *(Silk AI / Agents)*
Names the plan agents work on, so the store never has to guess which plan a
root means (D23). Lists the plans that already exist under the root with
their goal and open-task count; `(new plan)` plus a name creates one at a
path you can find again, `(newest under root)` keeps shared discovery.

| Direction | Port | Type |
|---|---|---|
| in | `root` | `string` |
| in | `root_paths` | `dirpath_list` (the ToolBox's sandbox ceiling) |
| in | `plan_choice` | `string` |
| in | `plan_name` | `string` |
| out | `plan` | `silk_plan` |

### Toolchain — `nodes/toolchain.py`
Configures a set of external toolchains — Python interpreters/venvs, ruff,
mypy, radon, maturin, cargo — as structured tools for the agent.

| Direction | Port | Type |
|---|---|---|
| in | `toolchains` | `toolchains` |
| out | `toolchains` | `toolchains` |

## Agent configuration

### Silk Role — `nodes/role.py`
Declarative agent configuration: persona instructions plus a
**hard-enforced** tool selection, downselected from the connected toolset.

| Direction | Port | Type |
|---|---|---|
| in | `toolset` | `silk_toolset` |
| in | `instructions` | `string` |
| in | `permissions` | `file_permissions` |
| out | `role` | `silk_role` |
| out | `permissions` | `file_permissions` |

### Inference Settings — `nodes/inference_settings.py`
Builds a `gen_params` dict (sampling / generation parameters) from UI
controls; save and recall configurations as named presets.

| Direction | Port | Type |
|---|---|---|
| out | `gen_params` | `dict` |

## Execution

### Silk Agent — `nodes/agent.py`
The autonomous tool-calling agent: wires model + toolset + role into the
Qt-free `AgentLoop`. Exec `run`/`done` ports let agents chain into networks.
This is the node that **runs an agent where it sits**; to describe an agent
for an Orchestrator to delegate to instead, use **Silk Worker** below.

| Direction | Port | Type |
|---|---|---|
| in | `model_obj` | `model_handle` |
| in | `toolset` | `silk_toolset` |
| in | `role` | `silk_role` |
| in | `system_prompt` | `string` |
| in | `user_prompt` | `string` |
| in | `inbox` | `agent_message` |
| in | `run` | `exec` |
| in | `inference_settings` | `dict` |
| in | `permissions` | `file_permissions` |
| out | `response` | `string` |
| out | `outbox` | `agent_message` |
| out | `events` | `dict` |
| out | `done` | `exec` |

`events` is the one typed stream (spec D2/D3): run lifecycle, model rounds,
tool calls and results, denials, plan snapshots, chat turns and decisions,
each carrying `type`, `ts`, `run_id`, `seq` and the agent identity.
Consumers filter by `type`.

**File access is a port, not a hidden handle** (spec D16-D18). Whatever
reaches `permissions` — wired straight here, or inherited down the
ToolSet → Role → Agent chain — narrows the toolset's sandbox for this run
and only this run, and is restored afterwards. The two sources compose by
narrowing, so adding a wire can only reduce access (I6); a malformed grant
grants nothing rather than falling back to the wider one.

**A blocked agent also shows on the canvas and in the Decision Inbox**
(D59): the node switches its pulse to a heartbeat while waiting, and
`DecisionInboxDock.attach(main_window)` opens a dock listing every waiting
agent with the same four answers (a Lace dock: the host window needs its
`dock_manager`, which a Weave host sets up). Answering there calls this node's own
handler — the dock owns no seam, and closing it strands nothing.

**The approval prompt is part of this node** (spec D48/I12). When a gated
tool call blocks, the question appears in the node itself — Deny, Allow
once, Allow this run, Always allow — and the held call resumes with the
answer. It is not a separate node because the run is inside `compute()`,
where no graph channel can reach it; the `decision.request` /
`decision.response` pair on `events` is a *mirror* for a monitor, never
the way the answer arrives (D59). Stop cancels the seam directly, so a run
waiting on a decision unwinds at once instead of waiting out the timeout.

**A placed node can also be configured** (§22 q9). `list_node_settings`
reads a node's widgets — name, type, current value, and why a port is not
writable — and `set_node_value` writes one, as one undo step. Values only
(text, number, boolean): a model, a toolset or a permissions object comes
from a connection, and display and internal widgets are readable but never
writable, which is what keeps a `Plugin authoring` checkbox out of an
agent's reach. Only nodes the run placed itself.

**An approved plugin comes back by itself, until it is edited** (§22 q10).
Approving a `load_suite` pins the SHA-256 of every importable file in that
suite, and the next start loads it without asking while the digests match.
Any edit, and the next load asks again with the diff; a suite that crashed
a start loses its pin entirely and must be approved by hand (§22 q11). The
pins are listed and revocable in the same dock as the grants.

**"Always allow" can be taken back** (§22 q1). A durable grant lives in
`~/.weave/silk/grants.json`, keyed by resolved project root, and
`GrantManagerDock.attach(main_window)` opens a dock that lists every one of
them with Revoke per grant and Revoke all per project (a Lace dock, like the
Decision Inbox, so the host window needs its `dock_manager`). It re-reads the file
each refresh, so a revocation takes effect on the next gated call in every
window; it can only remove, never grant. Run-scoped grants are not listed —
they end with the run.

**A priced run leaves a line in the ledger** (D90). When the endpoint
quoted a price, the run appends one line to `~/.weave/silk/spend.jsonl`
— when, how long, which models, tokens each way, how much, and how it
ended — attributed *per model*, so a run that fell back (D89) says which
model owns which part of the bill. There is no checkbox: a run against a
model that quotes nothing writes nothing, so a purely local graph never
acquires the file. Content never appears in it, a failed run is recorded
too (it was still prefilled), and a ledger that cannot be written is
logged once and ignored rather than failing the run.

`functions/cost_ledger.summarize(days=7)` answers "what did this week
cost me", with the per-model breakdown that makes the number actionable:
a total says you spent eleven dollars, the breakdown says nine of them
went to one model you could have put a cheaper one behind.

### Silk Worker — `nodes/worker.py`
Describes a named worker (model + toolset + role) that a Silk Orchestrator
can delegate to. Chain several to build a `silk_workers` roster.

| Direction | Port | Type |
|---|---|---|
| in | `model_obj` | `model_handle` |
| in | `toolset` | `silk_toolset` |
| in | `role` | `silk_role` |
| in | `description` | `string` |
| in | `budget` | `string` |
| in | `workers_in` | `silk_workers` |
| out | `workers` | `silk_workers` |

**This node runs nothing on its own.** Its `workers` output must reach a
Silk Orchestrator's `workers` input; nothing else consumes the type, so a
Worker with no Orchestrator downstream does exactly nothing.

That is the whole difference from the **Silk Agent** node above, which runs
where it sits — hence `run`/`done` and a `response` output, none of which a
Worker has. What a Worker has instead is a **name** (an orchestrator's model
addresses it by name, not by wire) and a **speciality** (advertised through
`list_workers` so the model can choose it). Neither means anything for an
Agent node.

It could not be a node that runs, either: `delegate` spawns workers inside
a single tool call, on a worker thread, possibly several at once, so a
worker has to be data the orchestrator carries rather than a box the engine
schedules.

*Renamed from "Silk Agent Spec"* (class `SilkAgentSpecNode`, ports
`agents_in`/`agents`, type `silk_agents`), which read like configuration
for an Agent node — the one thing it is not. Saved graphs using the old
node are not migrated.

### Silk Orchestrator — `nodes/orchestrator.py`
A Silk Agent that delegates self-contained sub-tasks to a roster of worker
agents (`delegate` / `delegate_parallel`). Takes the standard agent inputs
plus:

| Direction | Port | Type |
|---|---|---|
| in | `workers` | `silk_workers` |
| in | `max_depth` | `int` (spin box, default 2) |

`max_depth` is how deep delegation may nest: `1` lets the orchestrator call
workers but stops a worker from sub-delegating, `2` allows one further hop.
Cycles are refused at any depth. An upstream connection overrides the spin
box.

## Observability & human gates

### Hook Monitor — `nodes/hook_monitor.py` *(Silk AI / Display)*
Graph-native observability sink for the Agent's `events` stream: rolling log
of everything a run says, with per-type and per-tool counters.

| Direction | Port | Type |
|---|---|---|
| in | `event` | `dict` |
| out | `counts` | `dict` |

### Plan Viewer — `nodes/plan_viewer.py` *(Silk AI / Display)*
Display **and** graph-composition surface for the agent task tracker: shows
the current plan (goal, task tree, course corrections) rendered in the app's
markdown style.

| Direction | Port | Type |
|---|---|---|
| in | `root` | `string` |
| in | `plan_ref` | `silk_plan` |
| in | `plan` | `dict` |
| in | `event` | `dict` |
| out | `plan_json` | `dict` |
| out | `plan_text` | `string` |
| out | `plan_html` | `string` |

Sources are tried in order: an explicit `plan` snapshot, then `plan_ref`,
then `root`. The reference outranks the root because a root only says
*where* to look, and looking picks the newest plan there.

### Task Hub — `nodes/task_hub.py` *(Silk AI / Display)*
The multi-agent progress board (D58). Scans **every** `plan-*.db` under the
graph's sandbox roots and renders one section per plan, tasks grouped by
lane, with `claimed_by` as the per-task agent badge — the field the store
has always recorded and no view has ever shown.

| Direction | Port | Type |
|---|---|---|
| in | `roots` | `dirpath_list` (wire `Silk ToolBox.root_paths`) |
| in | `event` | `dict` (any agent's `events`; counted, never answered) |
| in | `refresh` | `exec` (a timer pulse or any agent's `done`) |
| out | `plans_json` | `dict` |
| out | `pending` | `int` |

`pending` is how many agents are blocked on a decision right now. The hub
may **count** those; only the asking node — or its dock mirror — may answer
one (D59). There are no Approve/Reject buttons here: D31–D33 deleted parked
sign-off, so a task change is decided during the turn, not held in a row.

### Chat Log Display — `nodes/chat_display.py` *(Silk AI / Display)*
Sink that continuously appends chat turns to a running log, rendering the
thread as markdown/HTML.

| Direction | Port | Type |
|---|---|---|
| in | `event` | `dict` | (the Agent's `events` stream; keeps `chat.turn`) |

### Pool Monitor — `nodes/pool_monitor.py` *(Silk AI / Display)*
Live snapshot of GGUF pool state (active/idle instances, capacity). Wire any
agent's `done` port to `refresh` for updates without polling.

Two rows are worth reading together, because the second exists for the
first:

- **Prefix reuse** — `reuse · contention · prefill · n measured` (D41,
  D47). Before any request it says *no requests measured yet* rather than
  0%, because those lead to opposite decisions. Reuse at 0% in every
  shape usually means the backend refuses partial KV removal; see
  [docs/prefix_reuse_measurement.md](prefix_reuse_measurement.md).
- **Affinity** — what the request queue is doing about it: the share of
  requests that followed one from the same conversation, the average wait
  that grouping cost, and how many hold windows paid off against how many
  elapsed unused. If the hold stops paying it suppresses itself and says
  so.

| Direction | Port | Type |
|---|---|---|
| in | `model_obj` | `model_handle` |
| in | `refresh` | `exec` |
| out | `pool_status` | `dict` |

## Example graph

```
[Silk MCP Server] ─mcp─▶ [Silk MCP Aggregator] ─mcp─┐
                                                    ▼
[Silk Task] ─plan──────────────────────────────────┐
                                                    ▼
[Toolchain] ─toolchains─▶ [Silk ToolBox] ◀──sandbox_roots (dirpath_list)
                              │ silk_toolbox
                              ▼
                         [Silk ToolSet] ◀──permissions (optional)
                              │ silk_toolset
                     ┌────────┴────────┐
                     ▼                 ▼
              [Silk Role]         [Silk Worker] ─agents─▶ [Silk Orchestrator]
                     │ silk_role              ▲ (worker bundles)
                     ▼                        │
  [GGUF Loader] ─model_handle─▶ [Silk Agent] ───┘
       │                        │
       └─ pool_info             ├─ events ─┬─▶ [Hook Monitor]
                                │          ├─▶ [Chat Log Display]
                                │          ├─▶ [Task Hub] ◀─root_paths─ [Silk ToolBox]
                                │          └─▶ [Plan Viewer]
                                └─ done (exec) ─▶ [Pool Monitor] .refresh
```
