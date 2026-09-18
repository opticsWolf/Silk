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
| **A missing optional dependency is a log line, not a widget — and its level is set by the fallback** | An install fact is not a property of the user's graph, so it goes to the log once per process, never onto a node where it would face someone who may not be able to act on it every time they open the canvas. **A fallback exists → `warning`** (nothing is broken, but the surface is quietly worse than it should be: `mordant` → plain markdown, T6; the `gguf` package → unclamped spinbox defaults; `macrame-db` → the SQLite task store). **No fallback → `error`** (the capability is simply gone: no ledger means no `recall`, and a search that returns nothing must not read as "nothing happened"). An *installed* dependency that raises is a third case and stays a warning — that one is a bug, not a choice made at install time. |
| **A dock is placed by the host's dock manager, or not at all** | Silk's two docks are Lace docks (the docking system Weave's chrome moved to), so `attach()` takes the host's `dock_manager` -- the same attribute Weave's panel commands read -- and **raises** when there is none. No fallback to `QMainWindow.addDockWidget`: a Lace dock is a plain `QWidget`, so the fallback would construct, wire and subscribe a dock nobody can see while telling the caller it worked -- the failure G9 found in graph authoring, where a node was placed with a title the canvas never rendered. A dock's `objectName` is its identity across restarts, because that is the key Lace restores a saved layout by. |
| **A subclass does not name its attributes as if the base class had none** | `DockWidget` keeps the dock's own layout in `self._layout` and `set_widget` adds the content to it. Both Silk docks had used that name for their *body* layout -- harmless under `QDockWidget`, which has no such attribute, and a hang under Lace: the dock added the scroll area to the body inside that scroll area. It is the one name the two classes share, which is why the check is cheap and worth doing when a base class changes. |
| **Runs are atomic (no mid-run input)** | Interactivity at run boundaries matches the sign-off turn-boundary pause and keeps the loop free of an inbox mechanism. |

