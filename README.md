# Silk

A local-first agentic runtime for the [Weave](https://github.com/opticsWolf/Weave)
visual workflow framework — exposed as Weave nodes.

Silk brings autonomous tool-calling agents into the Weave canvas:

- **GGUF model loader** — local inference via a `llama.cpp` model pool
- **Silk ToolBox** — sandboxed file/search tools with hard role enforcement
- **Silk Role** — declarative agent configuration
- **Silk Agent** — the autonomous tool-calling loop, with Exec trigger/done
  ports for chaining agent networks
- **Orchestrator, hooks, monitors** — multi-agent coordination and observability
- **Chat display** — node-rendered conversation log

The Qt-free runtime (ToolSet/ToolBox, capabilities, hooks, Role/RoleBinding,
AgentLoop, GraphEngine) lives under `functions/` and is importable
without PySide6.

## Repository layout

The repo root **is** the `silk` package (this layout is what allows the whole
repo to be synced into Weave as `weave/plugins/silk` via `git subtree`):

```
.                        (repo root = the `silk` package)
├── __init__.py          # registers all nodes with NODE_REGISTRY on import
├── functions/           # Qt-free agent runtime (tools, hooks, orchestrator…)
│   └── tools/           # sandboxed file/search tools
├── nodes/               # Weave node wrappers (agent, role, toolbox, monitors…)
├── widgets/             # Qt widgets (config dialog, tool tree, preset bar…)
├── README.md, pyproject.toml, LICENSE   # dev files (harmless in the subtree)
```

## Deployment / syncing with Weave

This plugin is **run in place** inside a Weave checkout — the package must
live at `weave/plugins/silk/` so Weave's submodule discovery can import it
and register its nodes. The two repos are linked with `git subtree`
(Weave-side remote name: `silk` → this repository):

- develop in **Weave** (`weave/plugins/silk/`), then push back here:
  ```
  git subtree push --prefix=weave/plugins/silk silk main
  ```
- develop in **this repo**, then pull into a Weave checkout:
  ```
  git subtree pull --prefix=weave/plugins/silk silk main
  ```

Sync in only one direction at a time; if you work in both, pull first,
then push, so the histories stay fast-forwardable.

Once Weave ships to PyPI and gains entry-point-based plugin discovery,
this will become installable via `pip install`.

Note: `functions/tools/task_tracker.py` intentionally imports
`weave.plugins.silk.functions.task_store` by absolute name — it is also
loadable as a top-level module by the dynamic tool loader, where a relative
import would fail.

## Dependencies

- **Weave** (the framework — provides `weave.node`, `weave.widgetcore`,
  `weave.widgets`, `weave.registry`, `weave._discovery`, …) and its PySide6
  stack
- `llama-cpp-python` (optional) — required for inference; the nodes import
  without it, but the GGUF loader logs an error if it is missing

## Development tools

This repository is run in place (no build system); `pyproject.toml` only
configures dev tools:

```
pip install ruff mypy
ruff check .
mypy
```

The linters from the Weave repo (`weave_lint` node/widget rule domains) apply
to this codebase as well.

## License

Apache-2.0 — see [LICENSE](LICENSE).
