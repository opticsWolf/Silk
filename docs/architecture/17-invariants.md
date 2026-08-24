## Invariants

The load-bearing guarantees of the runtime, stated as invariants (not style
rules) so a future test suite can pin them first — OPEN_TOPICS G4 is the
test plan, and these map onto it 1:1.

1. **A tool batch always returns one result per call, same shape, failures
   included.** Unknown tool, role denial, validation error, timeout, and
   exception all become model-visible results; nothing raises across the
   loop boundary. Enforced in `ToolBox.execute_tool_calls_async` /
   `_safe_execute` (`functions/tool_box.py`); the loop relies on it.
2. **`HOOK_AFTER_RUN` fires exactly once on every exit path** — normal
   completion, usage limit, stream error, early generator close. Enforced by
   the `finally` in `AgentLoop.run` (`functions/agent_loop.py`).
3. **The loop never executes tools.** It dispatches batches to the
   `ToolRegistry` and interprets the results; the engine never loops.
   Division of labour: the loop owns *turns*, the engine owns *one
   request*, the ToolBox owns *one tool batch*.
4. **A role-denied tool is both invisible and refused.**
   `get_tool_schemas()` skips non-permitted tools at advertisement time and
   `role_permits(...)` refuses them at dispatch (`functions/tool_box.py`).
   Removing either half breaks the other: the model would see tools it
   cannot call, or a hallucination of a denied tool would go uncaught.
5. **Store reads never mutate.** `plan_changed_event(...)` re-streams a
   `plan_summary` only when the plan's revision advanced past the last one
   seen; plain reads never bump the revision, so an unchanged plan never
   re-streams to the Plan Viewer. Enforced in `functions/task_store.py`.

