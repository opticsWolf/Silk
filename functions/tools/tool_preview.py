"""
Qt-free assembly of the *exact* context a session ToolBox would hand to the LLM:
the combined system prompt (base prompt + sandbox policy + every selected tool's
``procedure`` block) and the tool-call schemas (the function definitions).

Kept free of any UI import so it can be unit-tested and reused outside the dock.
:class:`tools.tool_dock.ToolDock` calls :func:`compose_preview` to render its
live preview panes.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..tool_box import ToolBox

DEFAULT_BASE_PROMPT = "You are an AI assistant with access to the following tools."


def compose_preview(
    box: "ToolBox",
    base_prompt: str = DEFAULT_BASE_PROMPT,
    policy_block: Optional[str] = None,
) -> dict:
    """
    Render the preview for *box* (a ToolBox with the chosen tools attached).

    Returns a dict with:
      * ``system_prompt``  â€” base prompt, optional sandbox-policy block, and each
        tool's procedure, exactly as :meth:`ToolBox.build_system_prompt` emits.
      * ``tool_schemas``   â€” pretty-printed JSON of :meth:`ToolBox.get_tool_schemas`.
      * ``names``          â€” list of registered tool names.
      * ``headline``       â€” one-line "N tools: â€¦" summary.
    """
    base = (base_prompt or "").strip()
    if policy_block:
        base = f"{base}\n\n[SANDBOX POLICY]\n{policy_block.strip()}"

    system_prompt = box.build_system_prompt(base)

    schemas = box.get_tool_schemas()
    tool_schemas = (
        json.dumps(schemas, indent=2, ensure_ascii=False)
        if schemas else "// No tools selected â€” check a tool to preview its schema."
    )

    names = [s["function"]["name"] for s in schemas]
    headline = (
        f"{len(names)} tool{'s' if len(names) != 1 else ''}: " + ", ".join(names)
        if names else "No tools selected."
    )
    return {
        "system_prompt": system_prompt,
        "tool_schemas": tool_schemas,
        "names": names,
        "headline": headline,
    }
