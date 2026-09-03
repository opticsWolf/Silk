## Layers

Silk splits cleanly into two layers:

| Layer | Contents | Qt? |
|---|---|---|
| **Graph layer** | `nodes/` — the node classes that appear in the Weave node palette; `widgets/` — Qt helper widgets those nodes embed (`tool_tree.py`, `config_dialog.py`, `preset_bar.py`, `hook_select.py`, `toolchain_list.py`) | yes (PySide6) |
| **Runtime layer** | `functions/` — everything the agent actually runs on | **no** |

Every `functions/` module is importable and usable headless — you can drive
a full agent run from a plain script without a Weave canvas, without
PySide6 installed. The Qt layer is a thin shell: nodes instantiate the
runtime, wire its streams to ports, and render its events.

Two small modules sit under everything and belong to neither concern:

- `functions/version.py` — `__version__` and `commit()`, the latter read
  from the files git writes rather than by calling git, so it is safe
  during a graph load and answers `""` in a source tree with no `.git`
  (G12). `weave.plugins.silk` logs `version_string()` once at import.
- `functions/credentials.py` — the D22 rule in one place: a graph stores a
  credential *name*, and the value is resolved at connect time from the
  environment or `~/.weave/silk/secrets.json`. MCP sessions and model
  backends both go through it; neither owns a copy.

> **Where to start?** For "how do I use Silk in a graph", read
> [NODES.md](../NODES.md). For "I want to add a tool / change behaviour", read
> [TOOLS.md](../TOOLS.md) and the hooks section below. For "how does this work
> internally", keep reading.

