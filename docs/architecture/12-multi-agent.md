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

1. **Depth cap** — `depth >= max_depth` → the agent is told it may not
   delegate further and to do the work directly. One default,
   `DEFAULT_MAX_DEPTH = 2`, shared by the runtime and the node (D55): the
   runtime used to fall back to `1` while the node shipped `2`, and a
   concept with two defaults is re-discovered as a bug rather than noticed.
   The `Silk Orchestrator` node exposes it as an editable `max_depth` port,
   so the number a graph runs at is visible in the graph.
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
`system_prompt`, `max_rounds`, `gen_params`, and an optional
`usage_limits`. `is_runnable()` validates the handle has a usable GGUF
model.

**Budgets nest (D26, closes T3).** Sharing one `UsageLimits` across a
fan-out makes the whole delegation respect a single global cap — and
nothing else, so one greedy worker could spend it all and leave the rest
of the fan-out with `USAGE_LIMIT` events. A spec's own `usage_limits` is
therefore not an *alternative* to the shared cap: `run_subagent` calls
`nest(shared, own)`, and with both present the worker's caps become a
`SubBudget` **inside** the shared one. It claims from itself first (it is
uncontended), then from the parent, refunding itself if the parent
refuses — so a worker is never billed for a request it was not allowed to
make, the shared counter other threads read is never transiently
over-charged, and no sub-budget can raise the global ceiling. Either half
may be absent, and nesting is idempotent, so a re-entered run does not
stack sub-budget on sub-budget. **`SubagentResult`** carries the reply text, `ok`, `error`, and
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

