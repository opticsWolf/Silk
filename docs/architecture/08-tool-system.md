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
| `requires_approval` | asks the user before every call, whatever the tool policy says — enforced by the floor in `functions/approval_floor.py` (D82), installed on the first flagged registration; a flagged tool in a toolbox with no floor is refused in `_safe_execute` rather than run |
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

### MCP servers as node-owned sessions (D19–D22)

`functions/mcp_session.py`. The combined-toolset path above is the wrong
shape for a graph: recipes are replayed per derived ToolSet, per agent, per
evaluation, and external toolsets are entered and exited around each
dispatched batch. An MCP server attached that way is re-handshaked
constantly — a stdio server respawned, a remote one re-authenticated, for
every batch of tool calls.

So the **node owns the session** (D19), the way the GGUF Loader owns a
model. `MCPSession` runs one connection on its own thread with its own
event loop — necessary because the dispatcher runs each batch in a fresh
`asyncio.run`, and a session belongs to the loop that opened it. `call()`
is a blocking bridge onto that loop with a timeout, so a wedged server
costs one bounded wait rather than a stuck agent.

What lands on a ToolBox is therefore not a ToolSet but ordinary registered
tools whose executables talk to an already-open session (`attach_mcp_tools`,
and `attach_bundle` as a recipe entry). Replaying the recipe copies dicts
and touches no server. It also means MCP tools get the rest of Silk for
free: the role gate, the approval hook, spill and discovery all work on
them without knowing what MCP is.

- **`MCPServerSpec`** — how to reach one server, as plain data.
  `credential` is a *name*, never a value (D22): resolved at connect time
  from the environment or `~/.weave/silk/secrets.json`, which is outside the
  graph, so saved graphs and presets stay shareable by construction. An
  unset credential refuses to connect and says where to put it.
- **`MCPBundle`** — the live sessions travelling one `mcp_servers` wire,
  plus the servers and tools the Aggregator switched off. Exclusions are
  recorded rather than filtered out, because unticking a tool must not cost
  a handshake when it is ticked again. `with_session` replaces a server of
  the same id rather than appending, so re-evaluation cannot double a
  server's tools.
- **Namespacing (D21)** — every tool is prefixed with its server id, so two
  servers offering `search` cannot collide and the model can see where a
  tool comes from. The prefix is Silk's: it is stripped again before the
  call goes out.

`load_mcp_toolsets(...)` and the ToolSet-level path remain for non-graph
use.

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
are dropped, but `INFRASTRUCTURE_TOOLS` (`load_capability`,
`search_tools`) always stay -- a selection that could drop discovery
would leave the agent unable to find the tools it was not given.
`tool_catalog(toolbox)` flattens a registry into plain-data entries
(`{name, description, parameters, category, tags, risk}`, infrastructure
excluded) that are safe to hand across threads to UI code.

### File access as a port (D16-D18)

`functions/file_grants.py` is the grant structure and the only place that
knows how two grants combine. `FileGrant` is one `(path, mode)` pair, mode
being `blocked` / `read` / `read_write`; `FileGrants` is a root (or roots)
plus a list of them, with `mode_for(path)` resolving by **nearest ancestor**,
defaulting to `blocked`. Both are Pydantic models, so `file_permissions` is
validated at the port boundary rather than trusted as a dict (D17) — the
`silk_ports.py` registration uses `FileGrants.is_valid` as its validator and
renders a short summary as the wire label.

`FileGrants.coerce` accepts a model, a plain dict, or `None`, so every
consumer takes the same argument and old dict-shaped graphs keep working.
`None` is not "no access" — it means *nothing was said*, which leaves the
sandbox exactly as it was. That distinction is what makes an unwired
`permissions` port harmless.

The grant travels **ToolSet → Role → Agent** as a visible port (D16):

- **`Silk ToolSet`** emits `permissions`: the grant its sandbox actually
  ended up with, after `split_by_ceiling` dropped everything outside the
  ToolBox's own roots. What comes out is the *effective* set, not what was
  asked for.
- **`Silk Role`** takes it in and hands it on unchanged, carrying it as data
  on `Role.file_grants`. The role never holds a live sandbox, so a role
  reused under a narrower toolset cannot smuggle a wider grant with it.
- **`Silk Agent`** composes the inherited grant with one wired straight to
  its own `permissions` port via `resolve_grants`, and applies the result
  for the run.

Every combination is `FileGrants.narrow`, and narrowing is asymmetric on
purpose: each entry becomes the **lesser** of what the upstream grant
allowed and what was asked for, a path the upstream never covered is *not*
added, and explicit `blocked` entries survive as holes. So adding a wire can
only ever reduce access (I6). An unparseable grant yields an **empty** grant
rather than the wider one it failed to parse — silently widening because a
structure was malformed is the failure this port was made explicit to
prevent.

The Agent applies it with `FileToolSandbox.restrict(path_modes)`, a context
manager that narrows `path_modes` / `write_enabled` **in place** and restores
them in a `finally`. In place, because the file tools closed over that
sandbox when they were registered: a replacement object would be built and
then ignored, and rebuilding the ToolBox would drop whatever is attached to
it live — the orchestrator's delegation tools, the run's approval gate. The
sandbox is a graph object shared with the next run, hence the restore.

`restrict` cannot widen and cannot re-enable confinement: `enabled` is a
deliberate ToolBox-level choice ("Enable sandbox"), and no grant travelling
down the chain can turn it back on (D18).

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

**The event vocabulary** (15 constants in `hooks.py`) — every one of them
fires. `UNWIRED_EVENTS` is empty, which is where §8's review table ended up
(§22 q2):

| Event | Kind | Emitted by | Fires |
|---|---|---|---|
| `HOOK_BEFORE_RUN` / `HOOK_AFTER_RUN` | event | `agent_loop` | around a whole agent run (`after` carries `final_text`, `rounds`, `elapsed_s`) |
| `HOOK_BEFORE_MODEL_REQUEST` | event | `agent_loop` | before each model request |
| `HOOK_AFTER_MODEL_REQUEST` | event | `agent_loop` | after a request, distinct from the response |
| `HOOK_AFTER_MODEL_RESPONSE` | event | `agent_loop` | after each response (carries `finish_reason`) |
| `HOOK_ON_MODEL_REQUEST_ERROR` | event | `agent_loop` | a model request failed (carries the classification) |
| `HOOK_WRAP_TOOL_VALIDATE` | middleware | `tool_box` | wraps argument parsing — repair or refuse before the tool is reached |
| `HOOK_ON_TOOL_VALIDATE_ERROR` | event | `tool_box` | arguments did not survive validation |
| `HOOK_BEFORE_TOOL_EXECUTE` / `HOOK_AFTER_TOOL_EXECUTE` | event | `tool_box` | around each tool call |
| `HOOK_WRAP_TOOL_EXECUTE` | middleware | `tool_box` | wraps tool execution (deny/redact/retry) |
| `HOOK_ON_TOOL_EXECUTE_ERROR` | event | `tool_box` | a tool raised or timed out |
| `HOOK_TOOL_DENIED` | event | `tool_box` | when a role denies a tool |
| `HOOK_ON_OUTPUT_VALIDATE_ERROR` | event | `agent_loop` | the final answer failed its schema |
| `HOOK_ON_OUTPUT_PROCESS_ERROR` | event | `agent_loop` | post-processing the final answer raised |

Registering on a name that is not in this list **raises
`UnwiredHookEvent`** rather than registering cleanly and never firing (D15).
That check used to guard a backlog; now it only guards typos.

**Why only one `wrap_*` around the model side survived.** Four names —
`wrap_model_request`, `wrap_output_validate`, `wrap_output_process`,
`wrap_run_event_stream` — were deleted rather than wired (§22 q2). They sat
in `agent_loop.py`'s synchronous generator and streaming paths, which async
middleware cannot express without turning the loop inside out, and
`Hooks.wrap_run_event_stream` had shipped as a stub that accepted
registrations and ignored them: the precise failure D15 exists to prevent.
`wrap_tool_validate` was kept because it could be honoured honestly — it
sits at an `await` in `ToolBox.execute_tool_calls_async`, it knows the tool
name (so `tools=` / `categories=` filtering means something), and it wraps
a call that returns a value. A middleware may hand the inner handler
different `raw_args` (repairing a quoted number the model got slightly
wrong, instead of spending a round on a validation error) or raise, which
ends that one call as an ordinary tool-result error the model can read —
never the run.

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
| `log_tool_calls` | `LogToolCallsConfig` | logs every tool call, result and role denial via the Weave logger |
| `timing` | `TimingConfig` | logs the wall-clock duration of each tool execution |
| `usage_meter` | `UsageMeterConfig` | counts tool calls / denials per run; summary at run end |
| `redact_secrets` | `RedactSecretsConfig` | masks secret-looking matches (API keys, tokens) in tool results before the model sees them |
| `tool_budget` | `ToolBudgetConfig` | hard-denies tool calls beyond a per-run budget (total and optional per-tool ceiling) |
| `task_audit` | `TaskAuditConfig` | holds the task plan's rationale to a quality bar (bounces trivial `n/a`-style reasons on add/complete/rescope/revise-goal) and logs a timestamped trail of plan changes |
| `signoff` | `SignoffConfig` | requires user sign-off before task changes take effect, per change type (agent self-signs vs human approval). Needs Task Planning; configured here, enforced on the ToolBox |
| `tool_approval` | `ToolApprovalConfig` | requires the user to approve tool calls before they run, by declared risk band or by name. Shares one gate with `signoff` — selecting both installs one middleware, not two (D31) |
| `spill` | `SpillConfig` | writes an oversized tool result to a file in the sandbox and leaves the model a head/tail preview plus the path |

The `Silk ToolBox` node's hook selector edits these configs through the
standard `config_dialog.py`, so users tune behaviour without code.

**Binding from the graph (§22 q5).** The observing hooks' configs derive
from `BoundHookConfig`, which adds `bind_tools` and `bind_categories`:
"log the file tools only" is expressible without touching code, and
`build_hooks` turns the field into the same `HookEntry` binding D13
defined. This is where the Hooks-node question D12 parked ended up — a
binding is a property of an entry, not a new place to compose hooks, so it
belongs in the config the two existing selectors already edit rather than
in a third node that would hide the ToolBox/Role split.

Two limits. A configured binding may only **narrow** what the code
declared (the I6 rule applied to hooks); one that shares nothing with the
code's binding raises rather than producing a hook that fires on nothing.
And only hooks that observe carry the field: `signoff`, `tool_approval`
and `task_audit` bind in code, out of a preset's reach, because a guard a
preset can narrow to nothing is not a guard (D77).

### `functions/spill.py` — keeping a big result out of the context

A tool result the model does not need in full still costs the whole run:
it is appended to history and re-sent, in full, on every later request.
The spill hook rewrites the result **before it is appended** — the text
goes to a file under `.silk/spill/` inside the sandbox, and what the model
sees is a head, a tail, and the path.

That ordering is the whole point (D41). Compaction rewrites the *head* of
the context, which collapses the longest common prefix with the previous
request to roughly the system prompt and forces a full re-prefill of a
context that is by construction near the ceiling — twice, since the
summary is itself a model request with a different prompt. Spill touches
only the newest message, so history stays append-only, invariant I11
holds, and compaction is *deferred* rather than caused.

Fan-out comes first (D57): `delegate_parallel` returns every worker's full
answer inline in one round, which is the largest single result Silk can
produce. So the hook understands the delegation result shapes structurally
and spills each worker's answer separately — the per-worker framing (who
answered, whether it worked) is small and worth keeping.

Every failure leaves the result inline and whole: no sandbox root means
the hook is not attached at all, because a path the agent cannot open is
not a reference, and a failed write costs a preview, never a result.

**Lifetime (§22 q4).** A spill file outlives its run, and the cleanup runs
on the way *in*. The preview left in history names the path and that
history is persisted with the Agent node, so deleting at run end would
turn every reference the model was handed into a dangling one — and an
orchestrator's workers may still be reading files while the run finishes.
`sweep()` is therefore called from `attach_spill_hook`, before the run has
written a single name: nothing live can be holding a path to what it
removes. Two bounds, `retain_days` (14) then `retain_bytes` (256 MB),
oldest deleted first, both settable on `SpillConfig`. It deletes only
names matching the pattern `write()` produces, only in the spill directory
itself, never recursively — the directory sits in the user's project, and
a file they put there is not ours to remove. A housekeeping failure is
logged and ignored; it never stops a run.

