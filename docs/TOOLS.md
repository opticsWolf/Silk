# Silk Tool Reference & Authoring

Tools are the only surface an agent can act on. Every tool is a plain
function registered on a `ToolBox` with a pydantic `args_model`, a
`procedure` (the docstring the model sees), declared risk, and optional bash
replacements. Enforcement (role filter + sandbox) happens in the execution
path, not in the prompt.

## Built-in tools

| Module | Tools | Purpose |
|---|---|---|
| `tools/file_read.py` | `read_file`, `view_file`, `list_directory`, `find_files`, `search_files`, `file_info` | read-only, sandboxed file access |
| `tools/file_write.py` | `write_file`, `append_file`, `create_directory`, `edit_file`, `insert_text` | sandboxed file mutation; `write_file` takes an optional `expected_sha256` precondition (`absent` = create only), compared inside the write's own lock, so a blind overwrite of another agent's change can be refused rather than discovered later (§22 q8) |
| `tools/file_manipulate.py` | `copy_file`, `move_file`, `delete_file`, `diff_files` | file operations |
| `tools/ripgrep_tool.py` | `ripgrep_search` | fast recursive content search via the in-process `pyripgrep` binding (`pip install ripgrep-python`); `.gitignore`-aware; `content` / `files_with_matches` / `count` output modes; structured JSON results |
| `tools/task_tracker.py` | `plan_start`, `plan_view`, `plan_history`, `task_add`, `task_update`, `task_complete`, `task_rescope`, `goal_revise`, `task_claim`, `request_signoff` | the agent's own task system: set a goal, grow/progress the task tree, park tasks for human sign-off |
| `tools/toolchains.py` | `run_python`, `ruff_check`, `ruff_format`, `mypy_check`, `radon_cc`, `radon_mi`, `maturin_develop`, `maturin_build`, `cargo_check`, `cargo_build` | configured external toolchains (Python venvs, linters, build tools) as structured tools |

Support modules (not tools themselves):

- `tools/file_sandbox.py` — sandbox configuration and path-safety helpers
- `tools/file_locks.py` — process-wide per-path locks (tools run in
  `asyncio.to_thread`; parallel calls to the same file serialize correctly)
- `tools/command_advice.py` — the bash→native-tool advice index
- `tools/tool_loader.py` — directory-based tool discovery
- `tools/tool_preview.py` — assembles the *exact* context a session ToolBox
  would hand the LLM (system prompt + sandbox policy + selected tools)

## Registering a tool

Tools live in modules that expose an `attach_*` function; the loader
discovers any `*.py` in the tools directory with one
(`__`-prefixed names are skipped; import failures are reported, never fatal):

```python
# my_tools.py
from pydantic import BaseModel

class GreetArgs(BaseModel):
    name: str

def attach_my_tools(toolbox, sandbox) -> None:
    @toolbox.register(
        name="greet",
        description="Greets a person by name.",          # what the model sees
        args_model=GreetArgs,
        procedure="Call with the target name.",           # how to use it
        category="demo",
        risk="low",
        replaces=[BashHint("hello", "greet(name=...)")],  # optional bash advice
        # optional: timeout=, requires_approval=, sequential=, tags=
    )
    def _greet(db_pool, user_session, name: str) -> str:
        return f"Hello, {name}!"
```

Rules of the shape:

- **Call convention** — every tool function is invoked as
  `func(db_pool, user_session, **validated_args)`: the first two parameters
  receive the ToolBox's injected context (a fresh ToolBox is created per
  user session), the rest are the *validated* fields of `args_model`.
- **`args_model`** — a pydantic `BaseModel`; validation errors surface to the
  model as tool errors (the reflection loop can then retry).
- **`procedure`** — the model-facing usage text; keep it imperative and
  concrete.
- **`risk`** — declared risk level; a tool policy may gate a whole band.
- **`requires_approval=True`** — this tool always asks the user first. It
  needs no policy and no hook config: the flag installs its own floor
  (D82), which asks on every call unless a grant pre-authorises the tool,
  and refuses when the run has no way to ask.
- **`replaces`** — `BashHint(command, native_call)` entries teach the model
  to prefer the native tool over a shell one-liner (e.g. `rg` →
  `ripgrep_search(...)`).

Two loader entry points (`tools/tool_loader.py`):

- `ToolLoader.sync` — incremental load/refresh into a **live** ToolBox (new
  files attached, changed files pruned-and-reattached, deleted files pruned)
  — powers add/refresh while the harness runs.
- `ToolLoader.discover` — stateless enumeration into throwaway ToolBoxes, for
  UIs that want to list tools without committing them.

## The sandbox

- The ToolBox node's **sandbox roots** are the hard ceiling — a per-path
  grant table (`file_permissions`: `{"root", "roots", "entries":
  [{"path", "mode"}]}`) where `mode` is read or read_write; blocked paths are
  simply absent.
- ToolSet nodes may add **per-toolset** grants on top of the ceiling.
- Every file tool checks `sandbox.is_allowed(path)` before touching anything;
  denied paths are reported to the model as tool errors.

## Optional: MCP tools

`functions/mcp_toolset.py` can attach a local **MCP (Model Context Protocol)**
server as a tool source — SSE, Streamable HTTP, or stdio transport. MCP
support is optional (guarded import); the rest of the toolbox works without
it.
