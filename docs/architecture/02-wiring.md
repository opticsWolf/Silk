## Wiring at a glance

The tool pipeline is linear:

```
Silk ToolBox ─(silk_toolbox)─► Silk ToolSet ─(silk_toolset)─► Silk Role ─(silk_role)─► Silk Agent
    ▲                              ▲
(sandbox roots: dirpath_list)   (file_permissions)
+ toolchains: toolchains
```

and the agent's other connections:

```
GGUF Loader ─(gguf_model)─────────────► Silk Agent
Inference Settings ─(dict)─────────────► Silk Agent
Silk Agent.outbox (agent_message) ───► (another) Silk Agent.inbox
Silk Agent.events (dict) ────────────► Hook Monitor / Plan Viewer /
                                       Chat Log / Sign-Off
```

Full per-node port tables are in [NODES.md](../NODES.md); the custom port
datatypes (`gguf_model`, `silk_toolbox`, `silk_toolset`, `silk_role`,
`silk_agents`, `agent_message`, `file_permissions`, `dirpath_list`,
`toolchains`) are declared once in `nodes/silk_ports.py`. The load-bearing
pieces:

- `Silk ToolBox` (`nodes/toolbox.py`) builds the `ToolBox` (registry +
  executor). Inputs: `sandbox_roots` (`dirpath_list` — the hard ceiling for
  file tools, enforced by `FileToolSandbox` in
  `functions/tools/file_sandbox.py`; concurrent access to the same path is
  serialised by `file_locks.py`) and `toolchains` (`toolchains` — named packs of external
  tooling: Python venvs, ruff, mypy, cargo, …, attached with numbered name
  disambiguation). On evaluation it assembles the node's configuration
  (task tracker if planning is enabled, toolchain packs, catalog hooks)
  into a `build_recipe`, applies each attacher with tool attribution, and
  stamps `build_recipe` + `base_sandbox` onto the box so downstream
  `Silk ToolSet` nodes can rebuild it. Output: `toolbox` (`silk_toolbox`).
- `Silk ToolSet` (`nodes/toolset.py`) derives an **independent** ToolBox
  from a source: it re-runs the recorded recipe (not a view-wrap),
  restricts to the selected tools, and may re-root the sandbox from
  `permissions` (`file_permissions`) — see `toolset_build` below. Output:
  `toolset` (`silk_toolset`).
- `Silk Role` (`nodes/role.py`) binds a toolset selection to instructions,
  gen-params, and hooks, emitting a `Role` handle (`silk_role`).
- `Silk Agent` (`nodes/agent.py`) runs the `AgentLoop` on a worker thread.
  Inputs: `model_obj` (`gguf_model`), `toolset` (`silk_toolset`), `role`
  (`silk_role`), `system_prompt`, `user_prompt`, `inbox`
  (`agent_message`), `run` (`exec` pulse), `inference_settings`
  (`dict`). Outputs: `response` (`string`), `outbox` (`agent_message`),
  `events` (`dict` — the one typed event stream: run lifecycle, model
  rounds, tool calls and results, denials, plan snapshots, chat turns,
  decisions; every line carries `type`, `ts`, `run_id`, `seq` and the
  agent identity), and `done` (`exec`). The vocabulary is
  `functions/stream_events.py`; consumers filter by `type` rather than by
  port, and a monitor that does not recognise a type still logs it.
- `GGUF Loader` (`nodes/gguf_loader.py`) loads a `.gguf` into the shared
  model pool (thread-safe; ejects the previous model on re-run) and emits a
  `gguf_model` handle (`{"backend": "gguf", "pool": <pool>}`) plus live
  `pool_info` (`dict`).
- `Inference Settings` (`nodes/inference_settings.py`) emits a `gen_params`
  `dict` (sampling/decoding knobs).
- `Silk Orchestrator` (`nodes/orchestrator.py`) **is a `Silk Agent` with an
  extra `workers` input** (`silk_agents` — a chain of `Silk Agent Spec`
  nodes): on evaluation it mounts the delegation tools on the agent's
  toolset (`attach_orchestrator_tools`, or an in-place roster refresh via
  `set_orchestrator_workers` if already mounted), passing its default
  `max_depth=2` (`DELEGATION_MAX_DEPTH`), then runs as a plain agent. See
  [Multi-agent](12-multi-agent.md#multi-agent).
- `Toolchain` (`nodes/toolchain.py`) composes and validates `toolchains`
  data (nodes chain `toolchains` → `toolchains` to accumulate; every
  enabled entry is version-probed on evaluation so a broken path fails
  visibly here, not inside a run).

