## System prompt assembly

The system prompt is assembled by **one function**:
`compose_system_prompt(base, role, toolset)` in `functions/subagent.py` —
shared by the `Silk Agent` node and the sub-agent runner (`run_subagent`),
so a node and its children build the same prompt shape. It joins these
sections with blank lines, in a fixed order:

1. **`base`** — the node's `system_prompt` input (or a sub-agent's
   `AgentSpec.system_prompt`). The engine stores it and prepends it as the
   system message on every request (`GraphEngine`).
2. **`role.system_prompt_block()`** — the `[ROLE: name]` block (omitted when
   empty).
3. **`toolset.build_system_prompt("")`** — the capability and `procedure`
   instructions of the activated tools.
4. **`tool_call_instructions(toolset)`** (`functions/tool_calling.py`) — the
   fenced tool-protocol section.

The **order is load-bearing, not cosmetic**: the function is called *after*
role activation so sections 3 and 4 see the role filter — a role-denied tool
is never *advertised* in the prompt. That is the schema-side half of
dispatch-time role enforcement (the other half is the `role_permits` check
at dispatch; see [ToolBox](08-tool-system.md#toolbox)).

The prompt is composed once per run when the engine is built (the Agent node
composes before constructing the `GraphEngine`); it is not re-derived
mid-run.

