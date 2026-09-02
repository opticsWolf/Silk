## Where new behaviour goes

| You want to… | Use |
|---|---|
| Add a core tool | implement it in `functions/tools/` and register it on a `ToolBox` (or as a `Capability`) — see [TOOLS.md](../TOOLS.md) |
| Add a behaviour without a node | a `hook_catalog` entry (`HookSpec`), or an ad-hoc hook via the `Silk ToolBox` node's hook selector |
| Give one agent narrower file access than its toolset | a `FileGrants` on the `permissions` port (ToolSet → Role → Agent); it composes by narrowing, never widening (`functions/file_grants.py`, I6) |
| Constrain what one agent may do / tell it how | a `ToolSelector` + `Role` from the `Silk Role` node, activated by a `RoleBinding` |
| Package tools + instructions + hooks together | a `Capability` (`@tool` / `@instructions` members) |
| Let an agent find a tool it was not given | it already can — `search_tools` (D4/D5) is on every ToolBox and survives every derived ToolSet; a hit carries the schema, and calling an unloaded tool auto-loads it (D6) |
| Keep a large tool suite out of the prompt without hiding it | `ToolBox.defer_tools(names)` — registered and dispatchable, not advertised; discovery is how the model reaches it |
| Keep the prompt small with a large tool suite | `DeferredCapability` + the `load_capability` tool (backed by `ToolSearch`) |
| Swap the model backend | implement the `AgentEngine` protocol (reference: `GraphEngine`) — the loop is unchanged |
| Add a tool-call transport | a transport beside `FenceTransport` / `NativeTransport` in `functions/tool_transport.py` |
| Keep a long run inside the context window | the spill hook first (prefix-preserving, `functions/spill.py`), then a `Compactor` on the `AgentLoop` (`functions/compaction.py`) — in that order, because compaction costs two prefills (D41) |
| Rewrite what the model sees | `GraphEngine.replace_history_prefix` — the one non-appending history operation, and it refuses cuts that orphan a tool result (I9) |
| Cap or shape a run | `UsageLimits` caps, `ReflectionConfig` retries, or an `OutputSchema` + validator on the final answer |
| Add planning / audit behaviour | `SqliteTaskStore` operations + a `signoff` policy; surface it via the Plan Viewer node |
| Open a Macrame ledger | `LedgerRegistry.acquire()` in `functions/ledger.py` — never `Database.open`, anywhere else (D62) |
| Put the task plan on the ledger instead of SQLite | `SILK_TASK_BACKEND=ledger` — `open_task_store()` picks, the tools never do; a missing extra falls back to SQLite with one warning line (D66) |
| Ask what the plan looked like at some past instant | `TaskLedger.load(as_of=<datetime>)` — a read, not an archaeology project (D63) |
| Make a compound task decision atomic | put it behind `TaskLedger._commit()` — one RLock per ledger file around read-check-assert (D64); never a second lock elsewhere |
| Remember a turn across runs and sessions | `HistoryLedger.record_turn()` in `functions/ledger.py`; the agent reads it back with the `recall` tool (§17, D66) |
| Compact without losing what was compacted | `HistoryLedger.compacted()` — a supersession event, so the dropped rounds stay readable (D24/D25, I11) |
| Let an approved suite load at the next start | it already does — `functions/suite_pins.py` pins the SHA-256 of every importable file at approval and `weave/bootstrap.py` loads what still matches; an edit sends it back through the floor (§22 q10) |
| Make a crashed suite loadable again | fix it and load it through the floor: `record_quarantine` unpins, and no auto-retry exists on purpose (§22 q11) |
| Show or withdraw a durable grant | the Grant Manager dock (`widgets/grant_manager.py`) over `GrantStore.projects()` / `.all()` — read the file every refresh, and only ever delete (§22 q1) |
| Centralise answering N agents' approval prompts | the Decision Inbox dock (`widgets/decision_inbox.py`) over `functions/decision_registry.py` — never a node (D59, D51, I12) |
| See what N independent agents are doing | a `Silk Task Hub` node fed from `Silk ToolBox.root_paths`; it reads the plan stores, never the agents (D58) |
| Point agents at a *specific* plan | a `Silk Task` node into the ToolBox node's `plan` input (and the Plan Viewer's `plan_ref`); leave both unwired to keep newest-under-root shared discovery (D23) |
| Repair or refuse a model's tool arguments before the tool runs | a `wrap_tool_validate` middleware — the one surviving `wrap_*` on the model side; it may re-supply `raw_args` or raise, and raising ends the call, not the run (§22 q2) |
| Require a human before a tool runs | the `tool_approval` catalog hook (risk band or tool names) — the same middleware the `signoff` policy uses |
| Let a user say "don't ask again" | a `remember` scope on the decision: run-scoped in the gate closure, or a durable grant in `~/.weave/silk/grants.json` |
| Give agents the tools of an MCP server | a `Silk MCP Server` node per server (it owns the session), optionally a `Silk MCP Aggregator` to switch individual tools off, into the ToolBox node's `mcp` input (D19–D22) |
| Delegate work across agents | `Silk Agent Spec` nodes feeding the `Silk Orchestrator` (`delegate` / `delegate_parallel`) |
| Persist a node's configuration | `functions/presets.py` (`PresetStore`, `~/.weave/presets/`) |
| Pass structured data between agents | `AgentMessage` on the `outbox` → `inbox` ports |
| Let an agent build graph | tick classes in the `Silk ToolBox` node's **Placeable Nodes** tree; the six tools mount only when the list is non-empty, and default-deny is the empty state (D69, D71) |
| Let an agent configure a node it placed | `set_node_value` — values only, INPUT/BIDIRECTIONAL only, nodes this run placed only, through `WidgetValueCommand` (§22 q9) |
| Make an object port settable by an agent | don't: object datatypes come from a connection, which is what keeps `file_permissions` and `dirpath_list` unwritable by construction (§22 q9) |
| Reach the Qt main thread from a tool | `MainThreadCall.call()` (`functions/main_thread_call.py`) — the same waiter as the decision seam, resolved by the event loop instead of a person (D70) |
| Add a canvas op an agent may perform | a method on `CanvasAuthor` (`nodes/graph_canvas.py`) that goes through an undo command, plus a policy check in `functions/graph_author.py` — never raw scene manipulation (D72) |
| Add a blocking worker→elsewhere channel of your own | subclass `BlockingSeam` (`functions/blocking_seam.py`); never re-implement the ordering rule (D49) |
| Let an agent write and load its own nodes | tick **Plugin authoring** on the `Silk ToolBox` node; it mounts the four load verbs, adds `~/.weave/plugins` to the sandbox, and installs the approval floor with them (D75, D76) |
| Add a verb that runs agent-authored code | put the ask in `functions/load_floor.py`'s floor, never behind a policy — a gate a preset can switch off is not a control (D77) |
| Check that a node change survives saved graphs | `weave_lint` — as the `weave_lint_check` toolchain tool, and as the hard stop `lint_suite()` applies before any load (D78) |
| Restart Weave from inside a run | `request_relaunch(reason)` — it queues; a human confirms at a turn boundary, and the agent never sees the far side (D79) |
| Tell the next run why its plugin vanished | `record_quarantine()` in `functions/self_modify.py`; `list_suites` reads it back (D81) |
| Stop an agent editing the graph that runs it | nothing — `check_self_modification` already refuses the agent, its tool chain, and everything upstream of it (D73) |
