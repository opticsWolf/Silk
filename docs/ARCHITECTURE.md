# Silk — Architecture

Silk is a Weave plugin that embeds a GGUF local-LLM agent into a visual
graph. This document describes how the pieces fit together, module by
module, based on the code in `functions/`, `nodes/`, and `widgets/`.

It is the companion to [NODES.md](NODES.md) (what the graph nodes do) and
[TOOLS.md](TOOLS.md) (the built-in tools and how to add your own).

## Contents

- [Layers](#layers)
- [Wiring at a glance](#wiring-at-a-glance)
- [The two protocol contracts](#the-two-protocol-contracts)
- [The agent loop](#the-agent-loop)
- [Model layer](#model-layer)
- [Tool transport](#tool-transport)
- [The tool system](#the-tool-system)
  - [ToolBox](#toolbox)
  - [Capabilities](#capabilities)
  - [ToolSet layer](#toolset-layer)
  - [Roles](#roles)
  - [Hooks and middleware](#hooks-and-middleware)
  - [Hook catalog](#hook-catalog)
- [Usage, reflection, and output validation](#usage-reflection-and-output-validation)
- [Task system and sign-off](#task-system-and-sign-off)
- [Multi-agent](#multi-agent)
- [Tool discovery and search](#tool-discovery-and-search)
- [Event streams](#event-streams)
- [Thread model](#thread-model)
- [Design rules](#design-rules)

## Layers

Silk splits cleanly into two layers:

| Layer | Contents | Qt? |
|---|---|---|
| **Graph layer** | `nodes/` — the node classes that appear in the Weave node palette; `widgets/` — Qt helper widgets those nodes embed (`tool_tree.py`, `config_dialog.py`, `preset_bar.py`, `hook_select.py`, `toolchain_list.py`) | yes (PySide6) |
| **Runtime layer** | `functions/` — everything the agent actually runs on | **no** |

Every `functions/` module is importable and usable headless — you can drive
a full agent run from a plain script without a Weave canvas, without
PySide6 installed. The Qt layer is a thin shell: nodes instantiate the
runtime, wire its streams to ports, and render its events.

> **Where to start?** For "how do I use Silk in a graph", read
> [NODES.md](NODES.md). For "I want to add a tool / change behaviour", read
> [TOOLS.md](TOOLS.md) and the hooks section below. For "how does this work
> internally", keep reading.

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
Silk Agent.plan_events (dict) ───────► Plan Viewer / Sign-Off
Silk Agent.tool_events (dict) ───────► Hook Monitor
```

Full per-node port tables are in [NODES.md](NODES.md); the load-bearing
pieces:

- `Silk ToolBox` (`nodes/toolbox.py`) builds the `ToolBox` (registry +
  executor). Inputs: `sandbox_roots` (`dirpath_list` — the hard ceiling for
  file tools) and `toolchains` (`toolchains` — named packs of external
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
  (`dict`). Outputs: `response` (`string`), `chat_turn` (`dict`),
  `outbox` (`agent_message`), `tool_events` (`dict` — one per tool
  call/result/denial, fed by toolbox hooks), `plan_events` (`dict` —
  `plan_summary` snapshots), and `done` (`exec`).
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
  [Multi-agent](#multi-agent).
- `Toolchain` (`nodes/toolchain.py`) composes and validates `toolchains`
  data (nodes chain `toolchains` → `toolchains` to accumulate; every
  enabled entry is version-probed on evaluation so a broken path fails
  visibly here, not inside a run).

## The two protocol contracts

`functions/protocols.py` pins what were previously duck-typed seams into two
`@runtime_checkable` `Protocol`s. The `AgentLoop` binds to these, not to
concrete classes — so tests substitute fakes and alternative engines (remote
APIs, mock models) drop in without touching the loop.

```python
@runtime_checkable
class AgentEngine(Protocol):
    """One model request per stream_response() call. Owns the history.
    Never executes tools, never loops — multi-turn belongs to the AgentLoop."""
    usage_limits: Any
    reflection_config: Any
    history: list[dict[str, Any]]
    last_stats: dict[str, Any]
    def stream_response(self, gen_params: dict[str, Any]) -> Iterator[str]: ...
    def append_message(self, role: str, content: str, **stats: Any) -> None: ...
    def count_prompt_tokens(self) -> int: ...
    def request_stop(self) -> None: ...
    def stop_requested(self) -> bool: ...

@runtime_checkable
class ToolRegistry(Protocol):
    """What the loop needs from a tool registry (ToolBox satisfies it)."""
    tools: dict[str, dict[str, Any]]
    async def execute_tool_calls_async(self, tool_calls: list[Any]) -> list[dict]: ...
```

`GraphEngine` (see [Model layer](#model-layer)) is the production
`AgentEngine`; `ToolBox` is the production `ToolRegistry`.

## The agent loop

`functions/agent_loop.py` — `AgentLoop` is the single autonomous
multi-turn runtime. It is a generator, not a coroutine: `run(...)` yields a
stream of typed events (see [Event streams](#event-streams)) and you consume
it at your own pace.

```python
loop = AgentLoop(
    engine,                 # an AgentEngine (e.g. GraphEngine)
    toolbox=None,           # a ToolRegistry (a ToolBox); None → pure chat
    output_validator=None,  # object with validate_with_reflection(text, max_retries=...)
    max_rounds=16,          # DEFAULT_MAX_ROUNDS hard ceiling
)
for event in loop.run(user_input, gen_params=None):
    ...
```

Note what is *not* a constructor argument: the system prompt, usage limits,
and reflection config all live on the **engine** (the `AgentEngine` protocol
exposes `usage_limits`, `reflection_config`, and `history`), and hooks are
read from the **toolbox** (`toolbox.hooks`). The loop itself carries only
the round ceiling and an optional output validator.

One **round** is: one model request → extract any tool calls → dispatch them
→ feed the results back. The verified flow of `run(...)`:

1. `select_transport(engine, toolbox)` — chosen **before the first request**
   so native tool schemas are advertised up front (see
   [Tool transport](#tool-transport)).
2. Append the user message (if any), yield `EventStart(settings,
   input_tokens)`, emit `HOOK_BEFORE_RUN`.
3. `_run_rounds(...)`, wrapped so `HOOK_AFTER_RUN` fires **exactly once on
   every exit path** (normal completion, usage limit, stream error, early
   generator close) via `finally`, carrying `final_text`, `rounds`, and
   `elapsed_s`.

   Each round (up to `max_rounds`):

   1. **Usage gates before the request**: `usage_limits.check_request()` and
      `check_input_tokens(...)`; a breach yields `EventUsageLimit` +
      `EventError(recoverable=False)` and ends the run.
   2. Emit `HOOK_BEFORE_MODEL_REQUEST`; stream **exactly one** response from
      `engine.stream_response(gen_params)`, yielding an `EventDelta`
      (`delta`, `total_tokens`, `cumulative_text`, `tps`) per chunk. A
      mid-stream `UsageLimitExceeded` or any exception ends the run with an
      `EventError`.
   3. Emit `HOOK_AFTER_MODEL_RESPONSE` (with `finish_reason`); persist the
      assistant turn via `engine.append_message("assistant", ...)` with the
      request's token stats — tool fences stay inside that message.
   4. `calls = transport.extract_calls(engine, full_text)` — from fenced
      `tool_call` blocks, or the engine's structured `tool_calls` when the
      model supports native tool calling.
   5. **No calls** (or a stop was requested, or `toolbox is None` — pure
      chat) → the model produced a final answer; break out of the rounds.
   6. **Calls present**: `usage_limits.check_tool_calls(len(calls))`, yield
      one `EventToolCall` per call, then
      `results = asyncio.run(toolbox.execute_tool_calls_async(calls))` —
      the loop is a *synchronous* generator, so each tool batch runs its own
      asyncio event loop. A batch-level exception becomes per-call error
      results, never a raised error.
   7. Per result: yield `EventToolResult` (including structured error
      payloads — the model sees its own failures), and
      `transport.append_tool_result(engine, ...)` writes the result into the
      engine history in the transport's own format.
   8. **Reflection**: if any result is a *retryable* tool error, the loop
      collects notes (pulling the error's `correct_schema` out of the
      structured payload, if present) and — within
      `reflection_config.max_retries` — emits **one consolidated**
      `EventReflection` per round (not one per failing call) and appends the
      nudge via `transport.append_retry_nudge(...)`. The loop then continues
      so the model can read the outputs / retry.

4. If the rounds are exhausted without a final answer →
   `EventError(context="agent_loop", recoverable=True)`.
5. **Output validation**: if `output_validator` is set,
   `validate_with_reflection(full_text, max_retries=reflection_config.max_output_retries)`
   runs; a failure ends the run with an `EventError`, a pass may rewrite
   `full_text` (e.g. the validated model dumped back to JSON).
6. Yield `EventFinalResult` then `EventRunResult` (text, token counts,
   `tps`, `finish_reason`, the `tool_calls`/`tool_results` run trace, and a
   `usage_stats` snapshot).

**The loop never executes tools itself** — it only dispatches batches to the
`ToolRegistry` and interprets the results. That is the whole division of
labour: the loop owns *turns*, the engine owns *one request*, the ToolBox
owns *one tool batch*.

`DEFAULT_MAX_ROUNDS = 16` is a hard ceiling on autonomy; `usage_limits`
(request / input-token / output-token / tool-call caps) are the second,
independent brake. `loop.stop()` requests a graceful stop, honoured at the
next token boundary via the engine.

### Engine-side config

Because the loop consults the engine's `usage_limits` and
`reflection_config`, those knobs are set where the engine is built (the
`Silk Agent` node / `GraphEngine` construction), not on the loop. The
`ReflectionConfig` fields the loop uses: `max_retries` (tool-error nudges),
`max_output_retries` (output validation), and `tool_error_prompt` (the nudge
text).

## Model layer

### `functions/graph_engine.py` — `GraphEngine`

The production `AgentEngine`. It adapts a Weave `gguf_model` handle to the
protocol:

- Owns the conversation history (a list of role/content dicts) and appends
  a turn per `append_message(...)`.
- `stream_response(gen_params)` performs **exactly one** model request,
  yielding incremental text deltas. It checks out a model from the pool (if
  the handle is pool-backed) and checks it back in on completion.
- Tracks `last_stats` (token counts, request metadata) and
  `count_prompt_tokens()` (best-effort input-token count) — these feed
  `UsageLimits`.
- Captures native structured tool calls if the backend exposes them (see
  transport), so the loop doesn't have to regex the text.
- `request_stop()` / `stop_requested()` cooperate with an in-flight
  generation: the stop is honoured at the next token boundary.

### `functions/model_pool.py`

A server-based model pool so several agents can share loaded GGUF models:

- Spawns **one** background `llama_cpp.server` process and talks to it over
  its OpenAI-compatible HTTP/SSE API — no in-process `Llama` object per
  agent.
- Generates a JSON server config on the fly (paths, context, etc.).
- `OpenAIClientMock` mimics the `llama_cpp.Llama` API on top of the HTTP
  client, so code written against the in-process API works unchanged against
  a pooled model.
- A pool-backed `gguf_model` handle carries `"pool": pool` instead of
  `"model": Llama`; the engine checks a model out/in around a request.

### `functions/gguf_meta.py`

A small binary GGUF probe. `GGUFMeta(context_length, block_count)` reads just
the two integer values the loader UI needs (to clamp its spinboxes). It
implements the GGUF v1 vs v2+ length-encoding difference and *skips*
non-integer KV values by seeking — the probe only cares about integers but
must advance the stream past everything else.

## Tool transport

`functions/tool_transport.py` — models disagree on how to express a tool
call. Silk supports both and picks per run:

- **`FenceTransport`** — the model emits a fenced JSON block
  (```tool_call ... ```). Parsed from the streamed text. Works with any
  instruct model, any backend.
- **`NativeTransport`** — the backend returns structured tool calls
  (e.g. llama.cpp's native tool calling). No text parsing; exact types.

`select_transport(engine, toolbox)` chooses based on the handle/backend
capabilities and **degrades safely** — if native isn't available it falls
back to fenced text rather than failing. The loop calls it once per run,
before the first request, so native schemas are advertised to the model up
front.

The transport also owns the *format* of the round-trip, which is what keeps
the loop backend-agnostic:

- `extract_calls(engine, full_text)` → the `tool_calls` list (parsed from
  fences, or the engine's structured calls);
- `append_tool_result(engine, name, call_id, body)` → writes each result
  back into the engine history in the transport's native format;
- `append_retry_nudge(engine, text)` → delivers the reflection nudge in a
  template-safe role.

The `AgentLoop` only consumes `tool_calls` and results either way.

## The tool system

Silk's tooling is a four-part system: **ToolBox** (registry + executor),
**Capabilities** (packaged tool bundles), **ToolSets** (composable
recipes), and **Roles** (per-agent policy). They layer on each other.

### ToolBox

`functions/tool_box.py` — the central registry and executor.

**Registration.** `register(...)` stores a `meta` dict per tool name. The
fields that matter:

| Field | Meaning |
|---|---|
| `name`, `description` | identity + what goes to the model |
| `definition` | the LLM-facing tool definition (JSON schema of args) |
| `args_model` | a Pydantic model — strict validation at dispatch |
| `executable` | the callable that does the work |
| `is_async` | whether `executable` is a coroutine |
| `procedure` | optional prose appended to the system prompt |
| `source` | provenance tag (`'core'`, `'mcp'`, `'plugin'`, ...) |
| `timeout` | max execution seconds |
| `requires_approval` | reserved approval gate — **currently a no-op placeholder** in `_safe_execute` (the approval-status check is not yet implemented) |
| `sequential` | must not run concurrently with other sequential tools |
| `tags`, `category`, `risk` | used by role selectors and search |

**Execution** — `execute_tool_calls_async(tool_calls)` is the single entry
point the loop uses. It:

1. Splits the batch into **parallel** and **sequential** groups; the
   parallel group runs under `asyncio.gather`, sequential tools one at a
   time. Sync tools are pushed off the event loop via `asyncio.to_thread`
   (they never block it); timeouts wrap both async (`wait_for`) and sync
   (`to_thread` + `wait_for`) execution.
2. For each call, in order:
   - **Unknown tool** → an error result `"Tool 'x' is not registered."`
     (returned, not raised — the model can see and recover).
   - **Role gate** — if a role is active, `role_permits(name)` is
     checked *at dispatch*. A denial emits `HOOK_TOOL_DENIED` and returns a
     structured error with `error_type="role_denied"` plus a suggestion.
     Role enforcement is a hard boundary, not a prompt hint — it happens
     both at schema-advertisement time (`get_tool_schemas()` skips
     non-permitted tools, so the model only *sees* what it may use) and
     here (so a model that hallucinates a denied tool is still
     blocked).
   - **Argument parsing** — `args_model.model_validate_json(...)` does JSON
     parse *and* strict validation in one step. A failure returns a
     structured error whose payload includes the correct JSON schema
     (`correct_schema`) so the model can self-correct on retry.
   - **`_safe_execute`** — emits `HOOK_BEFORE_TOOL_EXECUTE`, then runs the
     tool inside the `HOOK_WRAP_TOOL_EXECUTE` middleware chain (which can
     short-circuit, transform, or retry), then emits
     `HOOK_AFTER_TOOL_EXECUTE`. Exceptions become structured error results.
3. **Every outcome is a result.** Failures (timeout, validation, execution,
   role denial, unknown) are returned as model-visible `content`, never
   raised. The loop's contract is that a tool batch always returns a list
   of results of the same shape.

**Structured error payload** (`_error`): the `content` of a failed call is a
JSON object:

```json
{
  "error": "...",
  "error_type": "role_denied",          // machine-readable; reflection treats some as non-retryable
  "details": [...],                      // validation error details
  "suggestion": "...",                  // human/model-readable next step
  "correct_schema": { ... }             // the JSON schema, on validation errors
}
```

**Output normalisation** (`_to_content`): plain strings pass through
unchanged (so human-readable output isn't double-encoded), Pydantic models
are dumped to JSON, and everything else is `json.dumps(..., default=str)` so
a stray non-serialisable object can't crash the run.

**Combined toolsets.** `await self._enter_combined()` merges externally
attached toolsets (e.g. an MCP server) into the live registry for the run.

### Capabilities

`functions/capabilities.py` — a **capability** is a packaged bundle of
tools, instructions, model settings, hooks, and ordering constraints,
registered as a unit on a ToolBox. This is how bigger features (task
tracking, code tools, MCP) plug into an agent without a node per tool.

- **`BaseCapability`** (ABC): `id`, `get_tools() -> list[dict]`,
  `get_instructions()`, `get_model_settings()`, `get_hooks()`,
  `get_ordering() -> CapabilityOrdering | None`, and
  `get_description()` (a string, or a `Callable[[RunContext], str]` for
  dynamic descriptions).
- **`Capability`** — the decorator-based base: `@tool`, `@tool_plain`, and
  `@instructions` collect members into the bundle.
- **`ToolSet(BaseCapability)`** — a `ToolSet` *is* a capability, which is
  how a whole toolset drops into a recipe.
- **`DeferredCapability`** — a capability that is *not* loaded upfront; it
  is advertised to the model and loaded on demand via the `load_capability`
  tool. This keeps the system prompt small for large tool suites.
- **`CapabilityOrdering`** — `position` (`'outermost'` / `'innermost'`) and
  `requires` (a list of capability ids that must be present). Activation
  validates `requires` and **fails loudly** (ValueError) if a prerequisite
  is missing — better to refuse to start than to run a capability without
  its dependency. Ordering is stable-sorted: outermost first, innermost
  last, the rest in caller order.

### ToolSet layer

`functions/toolset.py` — the composable recipe layer that sits *above*
raw tools and *below* a ToolBox. A `ToolSet` is a declarative description
of a tool bundle; a small set of immutable operations each return a new
`ToolSet`, so recipes are easy to build and test.

- **`ToolMeta`** (dataclass) — the per-tool record: `toolset`, `definition`,
  `args_model`, `executable`, `is_async`, `procedure`, `source`, `timeout`,
  `requires_approval`, `sequential`.
- **`ToolSet`** (ABC) — materialises via `async get_tools() -> dict` and
  `get_instructions()`. The operations:

| Operation | Returns | Effect |
|---|---|---|
| `combined(other)` | `CombinedToolSet` | union of two toolsets |
| `filtered(...)` | `FilteredToolSet` | keep/drop by predicate |
| `prefixed(prefix)` | `PrefixedToolSet` | rename all tools with a prefix (namespace isolation) |
| `with_metadata(**kw)` | `MetadataToolset` | overlay metadata on every tool |
| `prepared(...)` | `PreparedToolset` | prepare for a specific backend/context |
| `renamed(name_map)` | `RenamedToolset` | targeted renames |
| `approval_required(...)` | `ApprovalRequiredToolset` | mark tools as gated |
| `defer_loading(tool_names)` | `DeferredLoadingToolset` | mark tools for on-demand load |
| `include_return_schemas()` | `IncludeReturnSchemasToolset` | advertise return types to the model |

  Plus `apply(visitor)` / `visit_and_replace(...)` for tree transforms.
- **`StaticToolSet`** — the concrete, hand-built toolset (a named dict of
  tools) you start from.

`functions/toolset_build.py` (Qt-free) derives ToolSets at the graph level.
The recipe protocol: whoever builds a ToolBox stamps it with `build_recipe`
(a tuple of `(source_name, attacher)` pairs, each attacher callable as
`attacher(toolbox, sandbox)`) and `base_sandbox`. `build_toolset(source,
selected_names, permissions=None)` **rebuilds** that recipe (not a view-wrap)
against the source's sandbox or one re-rooted from *permissions* — so a
derived ToolBox keeps every ToolBox guarantee (dispatch-time role gate,
hooks, schema generation) and each agent gets an independent instance that
never fights another over one `RoleBinding`. Tools not in *selected_names*
are dropped, but `INFRASTRUCTURE_TOOLS` (`load_capability`) always stay.
`tool_catalog(toolbox)` flattens a registry into plain-data entries
(`{name, description, parameters, category, tags, risk}`, infrastructure
excluded) that are safe to hand across threads to UI code.

### Roles

`functions/role.py` — a **role** is a named, declarative agent
configuration: *which tools this agent may use, what it's told, how the
model is tuned, and what behaviour it carries.* It is pure data (no live
tool references), built by the `Silk Role` node, and *activated* against a
ToolBox.

**`ToolSelector`** — the declarative tool-subset rule, evaluated live at
dispatch:

- `deny_names` **always wins**.
- Otherwise a tool is permitted if `allow_all` is set, or its name is in
  `allow_names`, or any of its `tags`/`category` match — **subject to the
  `max_risk` ceiling** (risk levels ordered by `RISK_ORDER`; a tool above
  the ceiling is denied).
- An all-empty selector permits **nothing** (deny-by-default) — the safe
  posture for a misconfigured role.
- `ALLOW_ALL` is the implicit default role's selector.

**`Role`** — fields: `id`, `name`, `description`, `instructions`,
`selector` (ToolSelector), `model_settings` (dict overlaid on gen params),
`capabilities` (instances), `hooks` (event → list of callables),
`max_rounds`. `system_prompt_block()` renders the `[ROLE: name]` block added
to the system prompt.

**`RoleBinding`** — a role *activated* against a ToolBox. Owns everything
reversible so deactivation leaves the toolbox pristine:

- `RoleBinding.activate(role, toolbox)` (or use as a context manager):
  1. Refuses if a binding is already active (no silently stacking filters).
  2. Registers the role's capabilities, honouring declared ordering and
     validating `requires`.
  3. Registers the behavioural hooks (`register_hook_map`; `wrap_*` events
     go to the middleware layer).
  4. Installs the **hard** enforcement predicate:
     `toolbox.set_role_filter(role.selector.permits)`.
- `deactivate()` reverses all of it — clears the filter, unregisters hooks,
  removes the role's capabilities and tools, and refreshes the deferred-load
  list.
- `effective_gen_params(base)` — overlay precedence is explicit
  **base (GUI/node) > role > defaults**: keys present in `base` win; the
  role only fills gaps it defines.

### Hooks and middleware

`functions/hooks.py` — the extension surface. Two distinct mechanisms live
in one `HookRegistry`:

**Plain events** (`register(event, callback)`, fired by `emit`):
- `before_*` events fire in **registration order (FIFO)**.
- `after_*` events fire in **reverse order (LIFO)** — matching Pydantic AI
  hook semantics (the last-registered `after` runs first, like unwinding a
  stack).
- Hook exceptions are swallowed so a buggy hook can never break the run.

**Middleware** (`register_middleware(event, handler)`, fired by
`emit_middleware`):
- A middleware handler receives a `handler` callable that invokes the next
  layer in the chain (the innermost being the actual operation).
- A handler can **short-circuit** (not call `handler` → e.g. deny a tool
  call), **post-process** its result (e.g. redact secrets), or **retry** by
  calling `handler` more than once — each invocation re-runs the *remaining*
  chain (the implementation is index-based recursion precisely so a second
  `handler()` call doesn't find an emptied list and silently skip layers).
- `register_ordered(capability_id, position, wraps, wrapped_by, requires)`
  records ordering constraints for capabilities.

The `Hooks` facade exposes ergonomic decorator registration:
`hooks.on.before_model_request(...)`, `.wrap_tool_execute(...)`,
`.after_run(...)`, etc., plus `get_tools()` / `get_instructions()` /
`emit(...)`.

**The event vocabulary** (19 constants in `hooks.py`; the first eight are
emitted today, the rest are defined but not yet wired):

| Event | Kind | Wired | Fires / intended |
|---|---|---|---|
| `HOOK_BEFORE_RUN` / `HOOK_AFTER_RUN` | event | `agent_loop` | around a whole agent run (`after` carries `final_text`, `rounds`, `elapsed_s`) |
| `HOOK_BEFORE_MODEL_REQUEST` | event | `agent_loop` | before each model request |
| `HOOK_AFTER_MODEL_RESPONSE` | event | `agent_loop` | after each response (carries `finish_reason`) |
| `HOOK_BEFORE_TOOL_EXECUTE` / `HOOK_AFTER_TOOL_EXECUTE` | event | `tool_box` | around each tool call |
| `HOOK_WRAP_TOOL_EXECUTE` | middleware | `tool_box` | wraps tool execution (deny/redact/retry) |
| `HOOK_TOOL_DENIED` | event | `tool_box` | when a role denies a tool |
| `HOOK_AFTER_MODEL_REQUEST` | event | — | after a request (distinct from the response) |
| `HOOK_WRAP_MODEL_REQUEST` | middleware | — | wrap a model request |
| `HOOK_ON_MODEL_REQUEST_ERROR` | event | — | on a model-request failure |
| `HOOK_WRAP_TOOL_VALIDATE` | middleware | — | wrap argument validation |
| `HOOK_ON_TOOL_VALIDATE_ERROR` | event | — | on a validation failure |
| `HOOK_ON_TOOL_EXECUTE_ERROR` | event | — | on a tool-execution failure |
| `HOOK_WRAP_OUTPUT_VALIDATE` / `HOOK_ON_OUTPUT_VALIDATE_ERROR` | middleware/event | — | wrap final-output schema validation |
| `HOOK_WRAP_OUTPUT_PROCESS` / `HOOK_ON_OUTPUT_PROCESS_ERROR` | middleware/event | — | wrap post-processing of the final output |
| `HOOK_WRAP_RUN_EVENT_STREAM` | middleware | — | wrap the run's event stream itself |

Rows marked `—` are defined and subscribable but nothing emits them yet —
see [Open Topics](OPEN_TOPICS.md).

### Hook catalog

`functions/hook_catalog.py` — ready-to-use, named, optionally configurable
hook bundles. Each **`HookSpec`** has a `name`, `description`, a
`factory(config) -> HookMap`, and an optional `config_model` (Pydantic).
`build_hooks(names, configs)` merges the selected entries into one
`{event: [callables]}` map (unknown names are skipped with a warning, not a
failure — forward-compatible with presets written by newer versions), and
`attach_catalog_hooks` wires them onto a live ToolBox.

Bundled hooks:

| Name | Config | What it does |
|---|---|---|
| `log_tool_calls` | — | logs every tool call, result and role denial via the Weave logger |
| `timing` | — | logs the wall-clock duration of each tool execution |
| `usage_meter` | — | counts tool calls / denials per run; summary at run end |
| `redact_secrets` | `RedactSecretsConfig` | masks secret-looking matches (API keys, tokens) in tool results before the model sees them |
| `tool_budget` | `ToolBudgetConfig` | hard-denies tool calls beyond a per-run budget (total and optional per-tool ceiling) |
| `task_audit` | `TaskAuditConfig` | holds the task plan's rationale to a quality bar (bounces trivial `n/a`-style reasons on add/complete/rescope/revise-goal) and logs a timestamped trail of plan changes |
| `signoff` | `SignoffConfig` | requires user sign-off before task changes take effect, per change type (agent self-signs vs human approval). Needs Task Planning; configured here, enforced on the ToolBox |

The `Silk ToolBox` node's hook selector edits these configs through the
standard `config_dialog.py`, so users tune behaviour without code.

## Usage, reflection, and output validation

Three independent guardrails on a run:

- **`functions/usage_limits.py`** — `UsageLimits` caps a *run-wide* budget:
  output tokens, input tokens, model requests, and tool calls. Exceeding any
  cap yields a `USAGE_LIMIT` event and ends the run cleanly (not an error —
  a controlled stop). `functions/usage.py`'s `UsageStats` is the plain
  accumulator that records the actuals.
- **`functions/reflection.py`** — `ReflectionConfig` drives self-correction:
  when a tool returns a *retryable* validation error (or output validation
  fails), the loop injects a reflection prompt and retries, within a bounded
  retry budget. Non-retryable error types (e.g. `role_denied`) are not
  retried — the model is told the denial is final.
- **`functions/output_schema.py`** — `OutputSchema` (built via
  `from_model(BaseModel)` or `from_dict`) validates the model's *final*
  answer against a schema and `build_instruction()` injects the required
  shape into the system prompt. `OutputValidator.validate_with_reflection`
  couples it to the reflection loop.

`functions/run_context.py`'s `RunContext` is the typed bag carried through a
run: `engine`, `deps` (e.g. `db_pool`, `user_session`), `usage`,
`usage_limits`, `model_settings`, `run_step`, and the loaded/available
capability + tool name sets — what a dynamic capability description or tool
can see about the current run.

## Task system and sign-off

Silk has a first-class planning/audit subsystem, fully headless.

### `functions/task_store.py` — `SqliteTaskStore`

A SQLite-backed store with **optimistic concurrency** and a full audit
trail. `SqliteTaskStore(root, direct_write=True)` resolves a writable
database location (trying candidate directories in order) and opens the
schema:

- **`plan`** — one row per plan: `plan_id`, goal text + original text +
  acceptance criteria, `revised`, `revision`, and any *pending goal*
  revision held for sign-off.
- **`task`** — `(plan_id, id)`-keyed rows: `title`, `status`, `parent`,
  `ord`, `note`, `origin`, `added_by`/`claimed_by`/`done_by` actors,
  timestamps, and the sign-off fields (`signoff_summary`, `signoff_by`,
  `signoff_note`, and `signoff_action` — the *held-and-applied* action,
  e.g. a deviation rescope that only lands on approval).
- **`revision`** — an append-only audit log (AUTOINCREMENT id, `at`,
  `actor`, `op_kind`, `op_json`, `rationale`) — every mutation is recorded
  with who did it and why.
- **`deviation`** — the from/to values of any change that deviated from the
  original plan, keyed to the revision that made it.

**Data model** (dataclasses): `Goal`, `Task`, `Deviation`, `Plan`, and
`Conflict`. A `Conflict` is a *genuine* collision (same task touched
twice, double-complete, goal race) — the store returns it instead of
guessing, and the model is told that retrying the identical operation will
not help.

**Operations:** `start`, `add_task`, `update_task`, `complete_task`,
`rescope_task`, `revise_goal`, `claim_task`, `request_signoff`,
`request_goal_signoff`, `sign_off`, plus reads (`load`, `history`,
`pending_signoffs`). Each mutation bumps the plan `revision` and writes an
audit row. `plan_changed_event(store, last_revision)` returns a
`plan_summary` event **only if** the revision advanced past
`last_revision` — reads never bump the revision, so an unchanged plan never
re-streams. The Agent node calls this after each tool batch to push live
updates to a `Plan Viewer`.

### `functions/signoff.py` — the user sign-off gate

A **policy** maps each *change type* to who may sign it:

- **`agent`** — the agent self-signs; the change applies immediately
  (audited with the agent as actor).
- **`human`** — the change is *parked* for the user; only
  `SqliteTaskStore.sign_off` can apply it. Deviations (rescope / goal
  revision) are **held and applied on approval** — the `signoff_action`
  stores the pending action and it lands only when the user approves.

Change types: `add`, `complete`, `complete_final` (the completion that
closes the plan — resolved dynamically), `rescope`, `goal`. Plain progress
(`task_update` / `claim`) is never gated.

Because the gate must read plan state (which task, is it the last one) and
park the item itself, it is attached **with a handle to the ToolBox** so
the model can't bypass it. It runs as a `HOOK_WRAP_TOOL_EXECUTE` middleware
and is exposed as the configurable `signoff` catalog hook. `SIGNOFF_MODES`
presets (`auto`/`requested`/`completions`/`final`/`strict`) expand to
policies; `custom` uses per-type levels.

**Turn-boundary pause:** parking flips a task to `awaiting_signoff` (or sets
`pending_goal`); the Agent node then *ends the run* so control returns to
the user. The user's approve/reject goes through the `Sign-Off` node →
`sign_off(...)`, which applies or rejects the held action (recording
`signoff_by` / `signoff_note`).

## Multi-agent

Silk can fan one agent out over several others.

### `functions/orchestrator.py`

An **orchestrator** is an agent whose toolset includes delegation tools
(`list_workers`, `delegate`, `delegate_parallel`). `set_orchestrator_workers`
refreshes the live roster on an already-wired toolbox without tearing it
down (only provided fields change), and `attach_orchestrator_tools`
registers the delegation tools.

`_run_one` (never raises — errors are packed in-band) guards three things
before delegating:

1. **Depth cap** — `depth >= max_depth` (default `1`) → the agent is told it
   may not delegate further and to do the work directly.
2. **Cycle detection** — if the target worker is already in the active
   delegation `chain`, the recursion is refused with the chain shown.
3. **Unknown worker** → an error listing the available workers.

On success it wraps the request in an `AgentMessage` (kind `task`), pushes
`depth + 1` and `chain + [worker]` onto the child's toolset (so a
worker-that-is-an-orchestrator sees them and stops in time), runs
`run_subagent(...)`, and returns a `DelegateResult` (answer, `tools_used`,
`correlation_id`, or an in-band error). Delegation requests and replies are
stamped with a shared `correlation_id` so a delegate ↔ result pair is
self-describing.

The request/response shapes are Pydantic models (`DelegateArgs`,
`DelegateParallelArgs`, `Assignment`, `DelegateResult`,
`DelegateParallelResult`, `ListWorkersResult`, `WorkerInfo`).

In the graph, the roster is supplied by `Silk Agent Spec` nodes (their
`silk_agents` output feeds the `Silk Orchestrator`'s `workers` input);
because the orchestrator node subclasses the agent node, an orchestrator is
also a fully working agent that can do the work itself when delegation
isn't warranted.

### `functions/subagent.py`

**`AgentSpec`** — a runnable agent bundle: a `model_handle` (a `gguf_model`
dict), an optional `toolset` (a `ToolBox` from `build_toolset`; `None` means
pure chat, where tool fences are treated as final output), a `role`,
`name`/`description` (what the orchestrator advertises to its model),
`system_prompt`, `max_rounds`, `gen_params`, and an optional **shared**
`usage_limits`. Sharing one `UsageLimits` across a fan-out means the whole
delegation respects a single global budget instead of each sub-agent getting
its own fresh allowance. `is_runnable()` validates the handle has a usable
GGUF model. **`SubagentResult`** carries the reply text, `ok`, `error`, and
the `tool_calls` trace.

### `functions/messaging.py` — `AgentMessage`

The typed envelope for agent-to-agent communication. A bare
`response -> user_prompt` string edge carries no provenance; `AgentMessage`
makes a hand-off self-describing: `content`, `sender`,
`recipient` (or `BROADCAST = "*"`), a `kind` (`task | result | error | status | handoff`), a
`correlation_id` (defaulting to its own id, so a first message opens a
thread replies can join), `parent_id`, an `artifacts` bag for structured
payloads, `id` (uuid), and `ts`. It is deliberately plain
(`to_dict`/`from_dict`) so it rides graph edges as a dict (the
`agent_message` port datatype), embeds in a tool result, or gets logged —
without pulling in PySide6.

## Tool discovery and search

`functions/tool_search.py` — **`ToolSearch`** backs deferred tool discovery
for `DeferredCapability`. It indexes registered tools and capabilities and
supports pluggable strategies:

- `keywords` (default) — local keyword-overlap ranking.
- `bm25` — BM25-based ranking.
- `regex` — regex match.
- a **custom callable** `(queries, tools) -> names`.

`max_results` caps output; `search_capabilities` returns matching
capabilities (so a deferred capability can be surfaced to the model as a
candidate to load).

## Event streams

`functions/stream_events.py` defines the canonical event types the
`AgentLoop` yields — an `EventType` enum (`start`, `delta`, `tool_call`,
`tool_result`, `final_result`, `run_result`, `error`, `usage_limit`,
`reflection`) plus one dataclass per type, and `EventBuilder` / `EventStream`
helpers. This is the one shape every layer speaks, and the one the Qt layer
renders:

| Event | Carries |
|---|---|
| `EventStart` | `settings` (gen params), `input_tokens` |
| `EventDelta` | `delta`, `total_tokens`, `cumulative_text`, `tps` |
| `EventToolCall` | `tool_name`, `tool_args` (parsed dict), `call_id` |
| `EventToolResult` | `tool_name`, `result`, `call_id`, `error`, `error_message` |
| `EventFinalResult` | `text`, `tokens`, `input_tokens`, `tps`, `finish_reason` |
| `EventRunResult` | the final-result fields **plus** `tool_calls`, `tool_results` (the run trace) and `usage_stats` (`total_tokens`, `elapsed_s`, and the `UsageLimits` snapshot) |
| `EventError` | `error`, `context`, `recoverable` |
| `EventUsageLimit` | `limit_type` (`request` — request-count or input-token cap — / `output_tokens` / `tool_calls`) |
| `EventReflection` | `retry_count`, `max_retries`, `error_type`, `error_message` |

`functions/event_format.py` is the human-facing side: `format_event(event)`
turns a tool-events dict into one log line (e.g. `▶ run started`,
`■ run finished — 4 round(s), 12.3s`, `· model round 2…`, `· response round
2 (1830 chars)`, tool call/result previews). Event dicts carry `event`
(kind), `ts`, `run_id`, and `seq` — a monotonic-per-run sequence that is the
dedup key for re-evaluations. `functions/plan_render.py` renders plan
markdown to HTML for the Plan Viewer via the *optional* `mordant` parser
(`highlighting_mode="Attribute"`, so Qt's limited HTML subset resolves the
styles); when `mordant` is absent it returns `None` and the caller falls
back to plain text — a missing optional dependency never breaks the graph.

## Thread model

- **The agent loop is a synchronous generator.** It is driven from the GUI
  thread (or any thread) and yields events as they happen; it is not
  itself async.
- **Tool batches get their own event loop.** Each round's tool batch runs
  via `asyncio.run(toolbox.execute_tool_calls_async(...))` — a fresh,
  short-lived loop per batch, which keeps the sync/async boundary simple.
  Inside that batch: sync executables are pushed off the loop via
  `asyncio.to_thread` and never block it; both sync and async execution is
  wrapped in `asyncio.wait_for` for timeouts; independent calls run under
  `asyncio.gather`, and tools flagged `sequential` run one at a time, so a
  mutation that can't tolerate concurrency is safe by declaration.
- **Model requests are streaming generators** — the loop pulls deltas as
  they arrive, so the UI stays live and a stop is honoured at the next token
  boundary.
- **The task store is its own concern** — SQLite with optimistic revision
  checks, so concurrent agent runs on one plan resolve conflicts by
  returning a `Conflict` rather than corrupting state.

## Design rules

- **`functions/` has no Qt.** Keep it that way — the runtime is testable
  headless and the Qt layer stays a thin shell.
- **Bind to the `AgentEngine` / `ToolRegistry` protocols, not concrete
  classes.** New engines and tool registries drop in without touching the
  loop.
- **Tools never raise across the loop boundary.** Every failure becomes a
  structured, model-visible result so the agent can recover. Reserve
  exceptions for programmer errors, not run-time conditions.
- **Enforce policy at dispatch, not just in the prompt.** Role gates run in
  `execute_tool_calls_async`; the prompt only shapes what the model *tries*.
- **Errors carry the fix.** Validation errors include the correct JSON
  schema (`correct_schema`); denials carry a suggestion; unknown tools
  carry the roster. Give the model what it needs to self-correct.
- **Capabilities are units of packaging.** Group related tools +
  instructions + hooks + ordering into a capability rather than scattering
  raw `register` calls.
- **Everything observable is an event.** The loop yields a typed stream;
  nodes render it. Don't reach around the stream with side channels.
- **Concurrency is declared, not assumed.** Mark a tool `sequential` if it
  can't run in parallel; share one `UsageLimits` across a fan-out if the
  budget is global.
- **Deviations are held, not applied.** Anything that changes the plan in a
  way the user should see (rescope, goal revision, final completion under a
  human policy) is parked and applied only on sign-off.
