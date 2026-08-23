# -*- coding: utf-8 -*-
"""Tool-call transport for the AgentLoop.

How the loop (a) extracts tool calls from a model turn and (b) feeds results
back into the conversation, abstracted behind a strategy so the same loop
drives two representations:

* :class:`FenceTransport` — the universal fallback. Tool calls are parsed
  from ``` ```tool_call ``` fenced JSON in the assistant's text
  (:mod:`tool_calling`); results are persisted as ``tool`` turns that the
  GGUF engine renders as ``Tool Output:`` user messages. Works with **any**
  chat model because it is pure text.

* :class:`NativeTransport` — used only when the engine advertises structured
  tool-calling for the loaded model (``supports_native_tools()``). Calls come
  from the model's own chat-template + grammar (parsed by the engine into
  structured ``tool_calls``); results feed back through the native ``tool``
  role with ``tool_call_id`` pairing.

:func:`select_transport` chooses per run and degrades safely: the native path
is picked only when a toolbox is present *and* the engine says the model
supports it, so plain chat models keep working unchanged.
"""
from __future__ import annotations

from typing import Any, List

from . import tool_calling


class ToolTransport:
    """Strategy for moving tool calls/results between model and ToolBox."""

    def extract_calls(self, engine: Any, full_text: str) -> List[Any]:
        """Return the tool calls the model emitted this turn (possibly empty).

        Calls are duck-typed to what ``ToolBox.execute_tool_calls_async``
        consumes: ``.id`` and ``.function.name`` / ``.function.arguments``.
        """
        raise NotImplementedError

    def append_tool_result(
        self, engine: Any, name: str, call_id: str, body: str
    ) -> None:
        """Persist a tool result into the conversation history."""
        raise NotImplementedError

    def append_retry_nudge(self, engine: Any, text: str) -> None:
        """Persist a reflection retry nudge in a template-safe role."""
        raise NotImplementedError


class FenceTransport(ToolTransport):
    """Text-fence protocol: universal, model-agnostic."""

    def extract_calls(self, engine: Any, full_text: str) -> List[Any]:
        return tool_calling.parse_tool_calls(full_text)

    def append_tool_result(
        self, engine: Any, name: str, call_id: str, body: str
    ) -> None:
        engine.append_message(
            "tool", tool_calling.tool_result_content(name, body)
        )

    def append_retry_nudge(self, engine: Any, text: str) -> None:
        # ``user`` (not ``system``): local chat templates assume a single
        # leading system message, so a mid-conversation system turn is
        # non-standard — some templates drop or reject it. A user turn is
        # always template-safe and lands in the same slot the engine already
        # uses for tool output.
        engine.append_message("user", text)


class NativeTransport(ToolTransport):
    """Structured tool_calls rendered by the model's own chat template."""

    def extract_calls(self, engine: Any, full_text: str) -> List[Any]:
        # The engine captured structured calls while streaming this turn.
        return engine.pull_tool_calls()

    def append_tool_result(
        self, engine: Any, name: str, call_id: str, body: str
    ) -> None:
        engine.append_tool_result(call_id, name, body)

    def append_retry_nudge(self, engine: Any, text: str) -> None:
        # A user turn is the portable, template-safe choice here too, and
        # mirrors the fence path so reflection behaves identically.
        engine.append_message("user", text)


def select_transport(engine: Any, toolbox: Any) -> ToolTransport:
    """Pick the tool transport for a run.

    Prefers the native path when a toolbox is present *and* the engine
    advertises native tool support for the loaded model; otherwise the fence
    fallback. The native branch also hands the engine the tool schemas so it
    can advertise them to the model (``tools=[...]``) for template + grammar
    use. Any engine lacking the native hooks (e.g. test fakes) transparently
    gets the fence transport.
    """
    if toolbox is None:
        return FenceTransport()
    supports = getattr(engine, "supports_native_tools", None)
    if callable(supports) and supports():
        get_schemas = getattr(toolbox, "get_tool_schemas", None)
        schemas = get_schemas() if callable(get_schemas) else []
        engine.enable_native_tools(schemas)
        return NativeTransport()
    return FenceTransport()
