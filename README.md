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

This plugin is **run in place** inside a Weave checkout. Weave links to
this repository directly as a **git submodule**: the path
`weave/plugins/silk/` in a Weave checkout *is* a checkout of this repo
(pinned to a specific commit in Weave's tree, tracking branch `main`).

Clone a Weave checkout with the plugin included:

    git clone --recurse-submodules <weave-url>
    # or, after a plain clone:
    git submodule update --init

### Workflow

- **Develop in the Weave subfolder** (`weave/plugins/silk/`):
  it is a normal checkout of this repo, so commit & push right there:

      cd weave/plugins/silk
      git checkout main        # first time only (submodules start detached)
      git add -A && git commit -m "..." && git push
      # then in the Weave root, pin the new commit:
      git add weave/plugins/silk && git commit -m "Bump Silk" && git push

- **Follow upstream** (e.g., after pushing from this standalone checkout):

      git submodule update --remote weave/plugins/silk   # in the Weave root
      git add weave/plugins/silk && git commit -m "Bump Silk" && git push

Each change is two small commits: one here (the actual change), one in
Weave (the pinned commit pointer).

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
