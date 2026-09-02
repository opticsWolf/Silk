## Where new behaviour goes

| You want to… | Use |
|---|---|
| Add a core tool | implement it in `functions/tools/` and register it on a `ToolBox` (or as a `Capability`) — see [TOOLS.md](../TOOLS.md) |
| Add a behaviour without a node | a `hook_catalog` entry (`HookSpec`), or an ad-hoc hook via the `Silk ToolBox` node's hook selector |
| Constrain what one agent may do / tell it how | a `ToolSelector` + `Role` from the `Silk Role` node, activated by a `RoleBinding` |
| Package tools + instructions + hooks together | a `Capability` (`@tool` / `@instructions` members) |
| Keep the prompt small with a large tool suite | `DeferredCapability` + the `load_capability` tool (backed by `ToolSearch`) |
| Swap the model backend | implement the `AgentEngine` protocol (reference: `GraphEngine`) — the loop is unchanged |
| Add a tool-call transport | a transport beside `FenceTransport` / `NativeTransport` in `functions/tool_transport.py` |
| Cap or shape a run | `UsageLimits` caps, `ReflectionConfig` retries, or an `OutputSchema` + validator on the final answer |
| Add planning / audit behaviour | `SqliteTaskStore` operations + a `signoff` policy; surface it via the Plan Viewer node |
| Require a human before a tool runs | the `tool_approval` catalog hook (risk band or tool names) — the same middleware the `signoff` policy uses |
| Let a user say "don't ask again" | a `remember` scope on the decision: run-scoped in the gate closure, or a durable grant in `~/.weave/silk/grants.json` |
| Delegate work across agents | `Silk Agent Spec` nodes feeding the `Silk Orchestrator` (`delegate` / `delegate_parallel`) |
| Persist a node's configuration | `functions/presets.py` (`PresetStore`, `~/.weave/presets/`) |
| Pass structured data between agents | `AgentMessage` on the `outbox` → `inbox` ports |
