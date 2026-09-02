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
| out | `role` | `silk_role` |

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
| out | `response` | `string` |
| out | `outbox` | `agent_message` |
| out | `events` | `dict` |
| out | `done` | `exec` |

`events` is the one typed stream (spec D2/D3): run lifecycle, model rounds,
tool calls and results, denials, plan snapshots, chat turns and decisions,
each carrying `type`, `ts`, `run_id`, `seq` and the agent identity.
Consumers filter by `type`.

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
