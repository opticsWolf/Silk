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

> **Where to start?** For "how do I use Silk in a graph", read
> [NODES.md](../NODES.md). For "I want to add a tool / change behaviour", read
> [TOOLS.md](../TOOLS.md) and the hooks section below. For "how does this work
> internally", keep reading.

