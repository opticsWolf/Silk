# tool_calling.py
"""General (tool-agnostic) tool-calling protocol for local GGUF chat models.

This replaces the old hard-coded nushell code-fence scraping. The model signals
a tool call by emitting a fenced ``tool_call`` block containing JSON:

    ```tool_call
    {"name": "read_file", "arguments": {"path": "notes.txt"}}
    ```

``parse_tool_calls`` turns those blocks into lightweight objects shaped exactly
like the ones :meth:`ToolBox.execute_tool_calls_async` consumes (``.id`` and
``.function.name`` / ``.function.arguments``), so any tool registered in the
ToolBox â€” file tools, ripgrep, directory_tree, nushell â€” is callable by the same
path. Results are persisted with :func:`tool_result_content`, the canonical
``{"name", "content"}`` JSON shape the transcript and chat-log dock already parse.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

# A fenced ```tool_call â€¦ ``` block. Tolerant of an optional language tag case
# and trailing whitespace; DOTALL so the JSON body may span lines.
_TOOL_CALL_BLOCK = re.compile(r"```tool_call\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


# â”€â”€ Call objects (duck-typed to what ToolBox.execute_tool_calls_async expects) â”€â”€

@dataclass(frozen=True)
class _Function:
    name: str
    arguments: str            # JSON string (ToolBox validates it via the args model)


@dataclass(frozen=True)
class ToolCall:
    id: str
    function: _Function


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract every ``tool_call`` block from *text* into ToolCall objects.

    Malformed blocks (bad JSON, missing name) are skipped rather than raised, so
    a model that fumbles the format degrades gracefully instead of crashing the
    generation loop. ``arguments`` is always normalised to a JSON **string**.
    """
    calls: list[ToolCall] = []
    for raw in _TOOL_CALL_BLOCK.findall(text):
        block = raw.strip()
        if not block:
            continue
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("name") or data.get("tool")
        if not name or not isinstance(name, str):
            continue
        args = data.get("arguments", data.get("args", {}))
        if isinstance(args, str):
            arguments = args                      # already a JSON string
        else:
            try:
                arguments = json.dumps(args)
            except (TypeError, ValueError):
                arguments = "{}"
        calls.append(ToolCall(id=f"call_{uuid4().hex[:12]}", function=_Function(name, arguments)))
    return calls


def has_tool_call(text: str) -> bool:
    return _TOOL_CALL_BLOCK.search(text) is not None


# â”€â”€ Canonical persisted shape â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def tool_result_content(name: str, content: Any) -> str:
    """The JSON a ``tool`` message stores: ``{"name", "content"}``.

    This is the exact shape ``transcript_view`` and ``chat_log_dock`` already
    parse via ``_parse_tool_content``, so persisted tool turns render with the
    right label/icon and body everywhere.
    """
    if not isinstance(content, str):
        try:
            content = json.dumps(content, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            content = str(content)
    return json.dumps({"name": name, "content": content}, ensure_ascii=False)


def parse_tool_result(content: str) -> tuple[str, str]:
    """Inverse of :func:`tool_result_content`; mirrors the UI parser.

    Returns ``(tool_name, body)``. Falls back to ``("Tool", raw)`` for legacy
    or non-JSON payloads, matching the dock/transcript behaviour.
    """
    if not content:
        return "Tool", ""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "name" in data and "content" in data:
            body = data["content"]
            return data["name"], body if isinstance(body, str) else json.dumps(body, indent=2)
    except (json.JSONDecodeError, TypeError):
        pass
    return "Tool", str(content)


# â”€â”€ System-prompt block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def tool_call_instructions(toolbox: Any) -> str:
    """Build the tool-calling instruction block for the system prompt.

    Lists the tools currently registered in *toolbox* and explains the
    ``tool_call`` protocol. Tool-agnostic: whatever is mounted (file tools,
    ripgrep, directory_tree, nushell, â€¦) is advertised automatically.
    """
    tools = getattr(toolbox, "tools", {}) or {}
    # Respect the active role's filter so denied tools are never advertised.
    permits = getattr(toolbox, "role_permits", None)
    if permits is not None:
        tools = {name: meta for name, meta in tools.items() if permits(name)}
    if not tools:
        return ""

    lines = ["\n\nTOOLS:", "You can call the following tools:"]
    for name, meta in tools.items():
        try:
            desc = meta["definition"]["function"]["description"]
        except (KeyError, TypeError):
            desc = ""
        first = (desc or "").strip().splitlines()[0] if desc else ""
        lines.append(f"- {name}: {first}" if first else f"- {name}")

    lines.append(
        "\nTo call a tool, emit EXACTLY one fenced block and then stop and wait:\n"
        "```tool_call\n"
        '{"name": "<tool_name>", "arguments": {<json arguments>}}\n'
        "```\n"
        "The result is returned to you as a Tool Output message; you may then "
        "call another tool or answer normally. Only emit a tool_call block when "
        "you actually want to run a tool."
    )
    return "\n".join(lines)
