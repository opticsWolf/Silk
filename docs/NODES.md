# Silk Node Reference

All nodes search under the **AI** category unless noted. Ports are listed as
they are registered; `exec` ports are trigger pulses.

## Model

### GGUF Loader — `nodes/gguf_loader.py` (`GGUFLNode`)
Loads a GGUF model via llama.cpp into the shared pool (thread-safe; ejects
the previous model when re-run).

| Direction | Port | Type |
|---|---|---|
| in | `model_path` | `filepath` |
| in | `prompt_cache` | `filepath` |
| out | `model_obj` | `gguf_model` |
| out | `pool_info` | `dict` (live pool stats) |

## Tool assembly

### Silk ToolBox — `nodes/toolbox.py`
The registry of **all** tools an agent network may use: sandbox roots (hard
ceiling), toolchain packs, category overview and per-tool details.

| Direction | Port | Type |
|---|---|---|
| in | `sandbox_roots` | `dirpath_list` |
| in | `toolchains` | `toolchains` |
| in | `mcp` | `mcp_servers` |
| in | `plan` | `silk_plan` |
| out | `toolbox` | `silk_toolbox` |
| out | `root_paths` | `dirpath_list` |

Wiring a `Silk Task` node into `plan` names the plan the task tools work
on. Left unwired, they discover the newest plan under the sandbox root —
which is how several agents share one plan, and why two unrelated plans in
one root used to find each other (D23).

The **Recall (memory)** checkbox mounts the `recall` tool: keyword search
over the turns and runs remembered in this sandbox root's history ledger,
including ones from earlier sessions and ones compaction dropped (§17,
D66). It needs the `ledger` extra (`pip install macrame-db`); without it
the tool registers and says so rather than quietly returning nothing.

The **Placeable Nodes** tree is the graph-authoring grant (§18, D71):
tick the node classes an agent may place, and the six graph tools
(`list_placeable_nodes`, `describe_graph`, `place_node`, `connect`,
`disconnect`, `remove_node`) mount. Leave it empty — the default — and no
agent fed by this ToolBox can build graph at all. Every edit an agent makes
goes onto the canvas's own undo stack, so one Ctrl+Z takes back one tool
call; destructive calls reach only what that run itself placed, and no
mutation may touch the agent, its tool chain, or anything upstream of it
(D72, D73). The list travels in the saved graph and in presets: it carries
no secret and no filesystem authority.

The **Plugin authoring** checkbox (§19) lets the agent write node suites
into `~/.weave/plugins` and load them into the running session. It adds
that directory to the sandbox as the only writable root (unless file
writing is already on), and mounts `list_suites`, `load_suite`,
`reload_suite` and `request_relaunch`. Every load asks you, every time,
and shows you the diff of what this run wrote — no Role, preset or
grant can pre-approve it, because importing runs that code with the
full authority of the Weave process. A state-version finding
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

### Silk Task — `nodes/task.py` *(AI / Agents)*
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

| Direction | Port | Type |
|---|---|---|
| in | `model_obj` | `gguf_model` |
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
agent with the same four answers. Answering there calls this node's own
handler — the dock owns no seam, and closing it strands nothing.

**The approval prompt is part of this node** (spec D48/I12). When a gated
tool call blocks, the question appears in the node itself — Deny, Allow
once, Allow this run, Always allow — and the held call resumes with the
answer. It is not a separate node because the run is inside `compute()`,
where no graph channel can reach it; the `decision.request` /
`decision.response` pair on `events` is a *mirror* for a monitor, never
the way the answer arrives (D59). Stop cancels the seam directly, so a run
waiting on a decision unwinds at once instead of waiting out the timeout.

**"Always allow" can be taken back** (§22 q1). A durable grant lives in
`~/.weave/silk/grants.json`, keyed by resolved project root, and
`GrantManagerDock.attach(main_window)` opens a dock that lists every one of
them with Revoke per grant and Revoke all per project. It re-reads the file
each refresh, so a revocation takes effect on the next gated call in every
window; it can only remove, never grant. Run-scoped grants are not listed —
they end with the run.

### Silk Agent Spec — `nodes/agent_spec.py`
A named worker bundle (model + toolset + role) for the Orchestrator; chain
specs to build a `silk_agents` roster.

| Direction | Port | Type |
|---|---|---|
| in | `model_obj` | `gguf_model` |
| in | `toolset` | `silk_toolset` |
| in | `role` | `silk_role` |
| in | `description` | `string` |
| in | `agents_in` | `silk_agents` |
| out | `agents` | `silk_agents` |

### Silk Orchestrator — `nodes/orchestrator.py`
A Silk Agent that delegates self-contained sub-tasks to a roster of worker
agents (`delegate` / `delegate_parallel`). Takes the standard agent inputs
plus:

| Direction | Port | Type |
|---|---|---|
| in | `workers` | `silk_agents` |
| in | `max_depth` | `int` (spin box, default 2) |

`max_depth` is how deep delegation may nest: `1` lets the orchestrator call
workers but stops a worker from sub-delegating, `2` allows one further hop.
Cycles are refused at any depth. An upstream connection overrides the spin
box.

## Observability & human gates

### Hook Monitor — `nodes/hook_monitor.py` *(Display / Agents)*
Graph-native observability sink for the Agent's `events` stream: rolling log
of everything a run says, with per-type and per-tool counters.

| Direction | Port | Type |
|---|---|---|
| in | `event` | `dict` |
| out | `counts` | `dict` |

### Plan Viewer — `nodes/plan_viewer.py` *(Display / Agents)*
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

### Task Hub — `nodes/task_hub.py` *(Display / Agents)*
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

### Chat Log Display — `nodes/chat_display.py` *(Display / Chat)*
Sink that continuously appends chat turns to a running log, rendering the
thread as markdown/HTML.

| Direction | Port | Type |
|---|---|---|
| in | `event` | `dict` | (the Agent's `events` stream; keeps `chat.turn`) |

### Pool Monitor — `nodes/pool_monitor.py` *(AI / Monitor)*
Live snapshot of GGUF pool state (active/idle instances, capacity). Wire any
agent's `done` port to `refresh` for updates without polling.

| Direction | Port | Type |
|---|---|---|
| in | `model_obj` | `gguf_model` |
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
              [Silk Role]         [Silk Agent Spec] ─agents─▶ [Silk Orchestrator]
                     │ silk_role              ▲ (worker bundles)
                     ▼                        │
  [GGUF Loader] ─gguf_model─▶ [Silk Agent] ───┘
       │                        │
       └─ pool_info             ├─ events ─┬─▶ [Hook Monitor]
                                │          ├─▶ [Chat Log Display]
                                │          ├─▶ [Task Hub] ◀─root_paths─ [Silk ToolBox]
                                │          └─▶ [Plan Viewer]
                                └─ done (exec) ─▶ [Pool Monitor] .refresh
```
