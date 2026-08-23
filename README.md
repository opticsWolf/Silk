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
AgentLoop, GraphEngine) lives under `silk/functions/` and is importable
without PySide6.

## Repository layout

```
silk/
├── __init__.py        # registers all nodes with NODE_REGISTRY on import
├── functions/         # Qt-free agent runtime (tools, hooks, orchestrator…)
│   └── tools/         # sandboxed file/search tools
├── nodes/             # Weave node wrappers (agent, role, toolbox, monitors…)
└── widgets/           # Qt widgets (config dialog, tool tree, preset bar…)
```

## Deployment

This plugin is **run in place** inside a Weave checkout — the `silk/` package
must live at `weave/plugins/silk/` so Weave's submodule discovery can import
it and register its nodes:

```
<weave-checkout>/weave/plugins/silk/   ←  this repository's silk/ package
```

Copy (or `git subtree`/submodule) the `silk/` folder into a Weave checkout to
deploy. Once Weave ships to PyPI and gains entry-point-based plugin discovery,
this will become installable via `pip install`.

Note: `silk/functions/tools/task_tracker.py` intentionally imports
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
