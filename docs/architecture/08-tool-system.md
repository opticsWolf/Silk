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
attached toolsets into the live registry for the run — the main consumer is
`functions/mcp_toolset.py`, which exposes a local MCP server (SSE /
Streamable-HTTP / stdio transports, auth tokens, resources, prompts,
sampling) as a combined toolset via `load_mcp_toolsets(...)`. MCP tools are
attached at the *registry* level, not through an engine-style seam — an MCP
server extends an existing ToolBox rather than replacing the engine.

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
- `make_outermost(event, entry)` moves one entry to the front of the chain.
  The chain runs first-registered outermost, and a middleware may return
  without calling `handler()` — so anything ahead of a guard can answer a
  call the guard never sees. A guard that must be monotonic (the approval
  gate, D37/I10) says so here instead of relying on the order a config file
  happened to produce.

**Binding and tiers** (spec D13/D14) — a registration is a `HookEntry`, not
a bare callable, and it carries two declarations the registry acts on:

- `tools` / `categories` scope the hook to particular tool calls. Empty
  means every tool, which is what every hook used to mean; a hook that
  cares about one tool now *says so* in the registry instead of opening
  with `if tool_name != ...: return` in a body nothing outside it can read.
  Categories are resolved through `ToolBox.tool_category`, which the box
  lends the registry at construction (`bind_categories`) — a bare registry
  with no index matches no category, so a category-bound hook stays quiet
  rather than firing for everything. A **bound** hook does not fire on the
  tool-less events (`before_run`): it declared it was about a tool.
- `essential=True` marks the infrastructure tier. `unregister` raises
  `EssentialHookError`, `clear()` keeps it unless you pass
  `keep_essential=False`, and `carry_essential_hooks` copies it onto a
  derived ToolSet — which is what makes invariant **I7** true for a hook
  installed *outside* the build recipe (the recipe replay already covers
  the ones inside it). The approval gate is the first hook that genuinely
  must not be droppable — and it is the worked example of the two tiers
  being different properties: `essential` means *cannot be dropped* (I7),
  `make_outermost` means *cannot be bypassed by something registered around
  it* (I10).

Both declarations can travel a `HookMap` — a plain `{event: [callables]}`
dict with no room for keywords — either as a `HookEntry` value or via the
`@bind_tools(...)` / `@essential` decorators on the callable itself.

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

Rows marked `—` are defined but nothing emits them yet, so registering on
one **raises `UnwiredHookEvent`** rather than registering cleanly and never
firing (D15) — see [Open Topics](../OPEN_TOPICS.md).

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

