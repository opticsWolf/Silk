# Silk Architecture

Silk is a local-first agentic runtime expressed as Weave nodes. It has two
strict layers:

| Layer | Where | Qt? | Role |
|---|---|---|---|
| **Node layer** | `nodes/`, `widgets/` | yes (PySide6) | thin graph/Qt wiring: ports, UI, node lifecycle |
| **Runtime layer** | `functions/` | **no** | the entire agent machinery; importable and unit-testable headless (no `QApplication`) |

No module under `functions/` imports PySide6 or touches Qt state. Nodes are
thin shells over it. (Importing the suite as `weave.plugins.silk` runs node
discovery, which imports the Qt modules — standard for every Weave suite, and
still needs no `QApplication`.)

## Canonical wiring

```
GGUF Loader ──gguf_model──▶ Silk Agent ◀──silk_toolset── Silk ToolSet ◀──silk_toolbox── Silk ToolBox
                                ▲                                              │
                                └──silk_role── Silk Role ◀─────────────────────┘
```

- **Model**: the GGUF Loader node loads a model into the shared pool and emits
  a `gguf_model` handle.
- **Tools**: the Silk ToolBox node is the registry of *all* tools (sandbox
  roots are the hard ceiling). The Silk ToolSet node selects a subset with
  optional per-set sandbox permissions. The Silk Role node layers persona
  instructions plus a *hard-enforced* tool selection on top.
- **Agent**: the Silk Agent node takes model + toolset + role (+ prompts,
  inference settings) and runs the autonomous tool-calling loop. It has
  `run` / `done` exec ports so agents chain into networks.
- **Observability sinks** (all consume event dicts):
  - `tool_events` → **Hook Monitor** (rolling event log + per-kind/per-tool counters)
  - `plan_events` → **Plan Viewer** (live plan; emits json/markdown/html) and **Sign-Off** (human gate)
  - `chat_turn` → **Chat Log Display** (running markdown chat log)
  - `done` → **Pool Monitor** `refresh` (live pool usage)
- **Multi-agent**: Silk Agent Spec bundles (model + toolset + role) chain into
  a `silk_agents` roster that the Silk Orchestrator node delegates to
  (`delegate` / `delegate_parallel`).

## Model layer

- `functions/model_pool.py` — a **server-based GGUF pool**: one background
  `llama_cpp.server` process (LM Studio-like) instead of in-memory
  `Llama` instances. Many agents share the pool; models load and eject on
  demand.
- `nodes/gguf_loader.py` — loads a GGUF file into the pool, thread-safe, and
  emits the `gguf_model` handle: `{"backend": "gguf", "model": <Llama> | "pool": <pool>}`.
- `functions/graph_engine.py` — the Qt-free `AgentEngine` over that handle;
  the only code that actually talks to the model. Sampling parameters arrive
  from the Inference Settings node's `gen_params` dict (only keys the engine
  knows are applied).
- `functions/gguf_meta.py` — header-only GGUF metadata probe; never touches
  the weight tensors.

## Tool layer

- **`ToolBox`** (`functions/tool_box.py`) — the single registry of all tools
  an agent network may use. Tools register with
  `register(name, description, args_model (pydantic BaseModel), procedure,
  replaces (BashHint), timeout, requires_approval, sequential, tags,
  category, risk, ...)`.
  - **Role enforcement**: `set_role_filter(predicate)` installs a hard,
    data-driven gate — a tool the connected role does not permit cannot run,
    whatever the model asks for.
  - **Sandbox**: file tools bind to a `FileToolSandbox`; the ToolBox node's
    sandbox roots are the ceiling, ToolSet nodes may add per-set
    `file_permissions` grants.
  - **Lifecycle**: hooks (below) and capabilities (deferred loading via tool
    search).
- **`ToolSet`** (`functions/toolset.py`) — composable, lifecycle-managed
  views of a ToolBox: `StaticToolSet`, `FilteredToolSet`, `PrefixedToolSet`,
  `CombinedToolSet`. `functions/toolset_build.py` rebuilds a *real* ToolBox
  from the source toolbox's **recipe**, which is how a ToolSet crosses
  graph/process boundaries.
- **Capabilities** (`functions/capabilities.py`) — reusable bundles of tools
  + instructions + settings; `DeferredCapability` loads on demand.
- **Tool discovery** (`functions/tools/tool_loader.py`) — any `*.py` in a
  tools directory that exposes an `attach_<name>_tools(toolbox, sandbox)`
  function is picked up. `ToolLoader.sync` applies changes incrementally to a
  *live* ToolBox (add/refresh while the harness runs); `ToolLoader.discover`
  enumerates statelessly. Import failures are reported, never fatal.

## Agent loop

- `functions/agent_loop.py` — the **single, Qt-free autonomous run loop**:
  stream one model response → parse tool calls → execute them through the
  ToolBox (parallel or per-tool sequential) → feed results back → repeat
  until the stop condition. Multi-turn behaviour lives *only* here.
- `functions/tool_transport.py` / `functions/tool_calling.py` — how calls are
  extracted from a model turn and results re-injected; a general
  (tool-agnostic) tool-calling protocol for local GGUF chat models, replacing
  hard-coded code-fence scraping.
- **Hooks** (`functions/hooks.py`) — middleware lifecycle with before/after
  and `wrap_*` hooks for model request/response, tool validate/execute,
  run start/end, tool denied, and `on_*_error` handlers.
  `functions/hook_catalog.py` bundles vetted implementations selectable by
  name — nodes offer them as checkboxes, and per-hook pydantic configs are
  editable via `widgets/config_dialog.py`.
- **Reflection & output** — `functions/reflection.py` retries on validation
  errors; `functions/output_schema.py` validates the model's final output
  against a pydantic model or JSON schema.
- **Guardrails** — `functions/usage.py` / `functions/usage_limits.py` track
  tokens, requests, and tool calls and enforce hard caps against runaway
  runs.
- **Streaming** — `functions/stream_events.py` replaces ad-hoc token deltas
  with typed events; `functions/event_format.py` shapes them for the Agent
  node's `tool_events` stream (Qt-free, so the Hook Monitor logic tests
  headless).
- **Multi-agent** — `functions/messaging.py` (`agent_message` inbox/outbox
  payloads), `functions/subagent.py`, and `functions/orchestrator.py`
  (delegation logic behind the Orchestrator node).

## Task system (the agent's plan)

- `functions/task_store.py` — a `SqliteTaskStore` keyed by a `root` path,
  with `Goal` / `Task` / `Plan` / `Deviation` / `Conflict` models, JSON
  round-tripping, `plan_changed_event`, and markdown rendering.
- The agent drives its own plan through the task tools
  (`functions/tools/task_tracker.py`): set a goal, grow the task tree,
  progress tasks, and **park** tasks as `awaiting_signoff` when they need a
  human.
- **Sign-off gate** — the Sign-Off node lists parked tasks (with the agent's
  summary); approving/rejecting pulses `signed` (exec) to resume the agent.
- **Plan Viewer** shows the live plan and re-emits it as json / markdown /
  html for downstream use.

## Port types

`nodes/silk_ports.py` registers the suite's custom port types once at import
(guarded against duplicate registration on re-import):

| Type | Carries |
|---|---|
| `gguf_model` | dict handle `{"backend": "gguf", "model": Llama \| "pool": pool}` |
| `silk_toolbox` | a live ToolBox registry (the full catalog) |
| `silk_toolset` | a ToolBox restricted to a selection — the *only* tool surface an Agent accepts |
| `silk_role` | a Role (declarative agent configuration) |
| `file_permissions` | `{"root", "roots", "entries": [{"path", "mode"}]}` per-path read / read_write grants |
| `dirpath_list` | ordered list of directory paths (sandbox roots) |
| `toolchains` | list of `ToolchainEnv` handles (configured executables) |

Event payloads (`chat_turn`, `tool_events`, `plan_events`) ride the plain
`dict` port type.

## Threading & concurrency

- File tools run synchronously inside `asyncio.to_thread`; when the ToolBox
  fires parallel tool calls, `functions/tools/file_locks.py` (process-wide
  per-path locks) keeps concurrent calls to the same file from clobbering
  each other.
- Model-running nodes are `ThreadedManualNode` subclasses: manual run/stop,
  with cross-thread signals for high-fidelity UI updates.

## Design rules

1. **No Qt below `functions/`** — anything that needs a test without a
   `QApplication` belongs in `functions/`.
2. **Nodes are thin** — wiring + UI only; behaviour lives in the runtime.
3. **Tools are data + a procedure** — pydantic `args_model`, a `procedure`
   docstring, declared risk and bash replacements; the ToolBox/role/sandbox
   stack does the enforcing.
4. **Enforcement is hard** — role filter and sandbox are gates in the
   execution path, not prompt suggestions.
