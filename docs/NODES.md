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
| out | `toolbox` | `silk_toolbox` |
| out | `root_paths` | `dirpath_list` |

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

**The approval prompt is part of this node** (spec D48/I12). When a gated
tool call blocks, the question appears in the node itself — Deny, Allow
once, Allow this run, Always allow — and the held call resumes with the
answer. It is not a separate node because the run is inside `compute()`,
where no graph channel can reach it; the `decision.request` /
`decision.response` pair on `events` is a *mirror* for a monitor, never
the way the answer arrives (D59). Stop cancels the seam directly, so a run
waiting on a decision unwinds at once instead of waiting out the timeout.

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
| in | `plan` | `dict` |
| in | `event` | `dict` |
| out | `plan_json` | `dict` |
| out | `plan_text` | `string` |
| out | `plan_html` | `string` |

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
                                │          └─▶ [Plan Viewer]
                                └─ done (exec) ─▶ [Pool Monitor] .refresh
```
