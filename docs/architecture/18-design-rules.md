## Design rules

Decisions this codebase has made, each with its rationale — so a refactor
doesn't silently undo one. The first eleven are standing rules; the rest
are structural decisions.

| Decision | Rationale |
|---|---|
| **`functions/` has no Qt** | The runtime stays testable headless and the Qt layer a thin shell. |
| **Bind to the `AgentEngine` / `ToolRegistry` protocols, not concrete classes** | New engines and tool registries drop in without touching the loop. |
| **Tools never raise across the loop boundary** | Every failure becomes a structured, model-visible result so the agent can recover; exceptions are reserved for programmer errors, not run-time conditions. |
| **Enforce policy at dispatch, not just in the prompt** | Role gates run in `execute_tool_calls_async`; the prompt only shapes what the model *tries* — a model that hallucinates a denied tool is still blocked. |
| **Errors carry the fix** | Validation errors include the correct JSON schema (`correct_schema`); denials carry a suggestion; unknown tools carry the roster — the model gets what it needs to self-correct. |
| **Capabilities are units of packaging** | Group related tools + instructions + hooks + ordering into a capability rather than scattering raw `register` calls. |
| **Everything observable is an event** | The loop yields a typed stream; nodes render it. Don't reach around the stream with side channels. |
| **Concurrency is declared, not assumed** | Mark a tool `sequential` if it can't run in parallel; share one `UsageLimits` across a fan-out if the budget is global. |
| **Deviations are held, not applied** | Anything that changes the plan in a way the user should see (rescope, goal revision, final completion under a human policy) is parked and applied only on sign-off. |
| **Observability is content-free** | Sinks, logs, and aggregations carry counts, durations, names, and statuses — never prompts, completions, or tool args/results. Stated now so a future event sink (OPEN_TOPICS T7) doesn't need a privacy retrofit. |
| **Derived state is rebuildable and zero-authority** | Anything indexed or cached (tool-search rankings, UI summaries) may be deleted and rebuilt from its source; it is never the source of truth. `tool_search.py` already follows this — now stated as a rule. |
| **The loop is a synchronous generator, not a coroutine** | It runs from any thread (including the GUI) with no event loop of its own; each tool batch gets its own short-lived `asyncio.run`, keeping the sync/async boundary in one place. |
| **ToolSets are derived by recipe rebuild, not view-wrap** | Each agent gets an independent ToolBox that keeps every ToolBox guarantee (role gate, hooks, schema generation) and never fights another agent over one `RoleBinding`. |
| **Runs are atomic (no mid-run input)** | Interactivity at run boundaries matches the sign-off turn-boundary pause and keeps the loop free of an inbox mechanism. |

