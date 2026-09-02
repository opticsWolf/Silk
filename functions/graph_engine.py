# -*- coding: utf-8 -*-
"""GraphEngine — a Qt-free AgentEngine over a Weave ``gguf_model`` handle.

The Weave graph passes models around as the ``gguf_model`` port payload:
``{"backend": "gguf", "model": Llama}`` or ``{"backend": "gguf",
"pool": GGUFModelPool}``. GraphEngine adapts that handle to the
:class:`~.protocols.AgentEngine` contract the AgentLoop consumes: it owns
the in-memory conversation history, performs exactly **one** model request
per :meth:`stream_response` call, and tracks usage. It never executes tools
and never loops — that is the AgentLoop's job.

Pool semantics: an instance is checked out for the duration of one stream
and checked back in when the stream ends (also on error/stop), so parallel
agent nodes can share one loaded model pool.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any, Optional

from .reflection import ReflectionConfig
from .usage_limits import UsageLimits

#: gen_params keys forwarded verbatim to llama.cpp's create_chat_completion.
_GEN_PARAM_KEYS = (
    "max_tokens", "temperature", "top_p", "top_k", "min_p",
    "repeat_penalty", "presence_penalty", "frequency_penalty",
    "seed", "stop",
)


class GraphEngine:
    """Single-turn chat engine over a ``gguf_model`` graph handle."""

    def __init__(
        self,
        model_handle: dict[str, Any],
        system_prompt: str = "",
        history: Optional[list[dict[str, Any]]] = None,
        usage_limits: Optional[UsageLimits] = None,
        reflection_config: Optional[ReflectionConfig] = None,
        session_id: str = "default",
    ) -> None:
        if not isinstance(model_handle, dict) or model_handle.get("backend") != "gguf":
            raise ValueError("GraphEngine needs a gguf_model handle dict.")
        self._handle = model_handle
        self.system_prompt = system_prompt
        # History is caller-owned state (the Agent node persists it); we
        # mutate the same list so the node sees every appended turn.
        self.history: list[dict[str, Any]] = history if history is not None else []
        self.usage_limits = usage_limits or UsageLimits()
        self.reflection_config = reflection_config or ReflectionConfig()
        self.last_stats: dict[str, Any] = {}
        self._stop = threading.Event()
        self.session_id = session_id

        # Native structured tool calling (opt-in; see supports_native_tools).
        # When armed by the AgentLoop, stream_response passes ``tools=`` to
        # the model and captures the structured tool_calls it emits.
        self._native_tools_enabled = False
        self._tool_schemas: list[dict[str, Any]] = []
        self._pending_tool_calls: list[dict[str, Any]] = []

    # -- AgentEngine: control ------------------------------------------------

    def request_stop(self) -> None:
        self._stop.set()

    def clear_stop(self) -> None:
        self._stop.clear()

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    # -- AgentEngine: history --------------------------------------------------

    def append_message(self, role: str, content: str, **stats: Any) -> None:
        entry: dict[str, Any] = {"role": role, "content": content}
        if stats:
            entry["stats"] = {k: v for k, v in stats.items() if v is not None}
        # Native path: attach the structured tool_calls captured during this
        # turn's stream to the assistant entry so build_messages can render
        # them (and the model sees a well-formed call→result pairing). A copy
        # is stored so pull_tool_calls() can still clear the pending buffer.
        if (
            role == "assistant"
            and self._native_tools_enabled
            and self._pending_tool_calls
        ):
            entry["tool_calls"] = list(self._pending_tool_calls)
        self.history.append(entry)

    # -- AgentEngine: native tool calling (optional capability) --------------

    def supports_native_tools(self) -> bool:
        """Whether the loaded model advertises structured tool calling.

        Opt-in via the model handle (``supports_tools``), set by the loader
        from the GGUF's chat-template metadata. Defaults to False so any model
        without a tool-aware template keeps using the fence protocol. This is
        the gate ``select_transport`` consults to choose the native path.
        """
        return bool(self._handle.get("supports_tools", False))

    def enable_native_tools(self, schemas: list[dict[str, Any]]) -> None:
        """Arm structured tool calling and advertise *schemas* to the model."""
        self._tool_schemas = list(schemas or [])
        self._native_tools_enabled = bool(self._tool_schemas)

    def pull_tool_calls(self) -> list[Any]:
        """Structured calls captured during the last stream (consumed once).

        Returned as :class:`~.tool_calling.ToolCall` objects so the ToolBox
        dispatch path is identical to the fence protocol.
        """
        from .tool_calling import ToolCall, _Function

        calls = [
            ToolCall(
                id=d["id"],
                function=_Function(
                    d["function"]["name"], d["function"]["arguments"]
                ),
            )
            for d in self._pending_tool_calls
        ]
        self._pending_tool_calls = []
        return calls

    def append_tool_result(self, call_id: str, name: str, content: str) -> None:
        """Persist a tool result as a native ``tool`` turn (id-paired)."""
        self.history.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content if isinstance(content, str) else str(content),
        })

    def build_messages(self) -> list[dict[str, str]]:
        """Chat-completion messages for the current state.

        ``tool`` turns are mapped to ``user`` messages with a ``Tool Output:``
        wrapper — local GGUF models have no native tool role, and this is the
        same convention the silk tool_call protocol documents.
        """
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for entry in self.history:
            role = entry.get("role", "user")
            content = str(entry.get("content", ""))
            if self._native_tools_enabled:
                # Native rendering: the model's chat template handles the
                # ``tool`` role and assistant ``tool_calls`` directly.
                if role == "tool":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": entry.get("tool_call_id", ""),
                        "name": entry.get("name", ""),
                        "content": content,
                    })
                elif role == "assistant" and entry.get("tool_calls"):
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": entry["tool_calls"],
                    })
                else:
                    messages.append({"role": role, "content": content})
            elif role == "tool":
                # Fence path: GGUF models have no native tool role, so map it
                # to a user turn with a Tool Output wrapper.
                messages.append({"role": "user", "content": f"Tool Output:\n{content}"})
            else:
                messages.append({"role": role, "content": content})
        return messages

    def count_prompt_tokens(self) -> int:
        """Best-effort input-token estimate for the current prompt state."""
        text = "\n".join(m["content"] for m in self.build_messages())
        model = self._handle.get("model")
        if model is not None and hasattr(model, "tokenize"):
            try:
                return len(model.tokenize(text.encode("utf-8", errors="replace")))
            except Exception:
                pass
        # Pool instances are not checked out just to count; approximate.
        return max(1, len(text) // 4)

    # -- AgentEngine: generation ----------------------------------------------

    def stream_response(self, gen_params: dict[str, Any]) -> Iterator[str]:
        """Yield text deltas for exactly one model request."""
        self.last_stats = {}
        # Claimed, not merely checked: one budget is shared by every worker
        # of a fan-out, so check-then-record lets several of them pass the
        # same check and collectively overrun the cap (spec D52.4).
        self.usage_limits.reserve_request()

        model, pool = self._checkout()
        full_text = ""
        token_count = 0
        # Deliberately None, not "stop": a stream truncated by the server
        # (llama_cpp.server's ``interrupt_requests``, spec D43) carries no
        # finish_reason at all, and a default of "stop" is precisely the
        # value that hides it. Unset at the end of a stream that did not
        # raise means the answer was cut off.
        finish_reason: Optional[str] = None
        errored = False
        start = time.time()
        try:
            params = {k: gen_params[k] for k in _GEN_PARAM_KEYS if k in gen_params}
            if self._native_tools_enabled and self._tool_schemas:
                params["tools"] = self._tool_schemas
                params["tool_choice"] = "auto"
            stream = model.create_chat_completion(
                messages=self.build_messages(),
                stream=True,
                **params,
            )

            tool_frags: dict[int, dict[str, str]] = {}
            for chunk in stream:
                if self._stop.is_set():
                    finish_reason = "stopped"
                    break
                choice = (chunk.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                # Native tool-call fragments arrive across chunks keyed by
                # index; accumulate id + name + argument text.
                if self._native_tools_enabled:
                    for frag in (delta.get("tool_calls") or []):
                        _accumulate_tool_call(tool_frags, frag)
                text = delta.get("content")
                if not text:
                    continue
                token_count += 1
                self.usage_limits.reserve_output_tokens(1)
                full_text += text
                yield text

            if self._native_tools_enabled:
                self._pending_tool_calls = _finalize_tool_calls(tool_frags)
        except Exception as exc:
            errored = True
            self.last_stats = {"error": str(exc), "text": full_text}
            raise
        finally:
            elapsed = time.time() - start
            self.last_stats.setdefault("text", full_text)
            self.last_stats.update({
                "tokens": token_count,
                "input_tokens": self.count_prompt_tokens() if not pool else 0,
                "tps": (token_count / elapsed) if elapsed > 0 else 0.0,
                "finish_reason": finish_reason,
                "truncated": finish_reason is None and not errored,
            })
            self._checkin(model, pool, session_id=self.session_id)

    # -- pool handling -----------------------------------------------------

    def _checkout(self) -> tuple[Any, Any]:
        pool = self._handle.get("pool")
        if pool is not None:
            model = pool.checkout(session_id=self.session_id)
            if model is None:
                raise RuntimeError(
                    "GGUF model pool exhausted — increase the pool size or "
                    "wait for other agents to finish."
                )
            return model, pool
        model = self._handle.get("model")
        if model is None:
            raise RuntimeError("gguf_model handle has neither 'model' nor 'pool'.")
        return model, None

    @staticmethod
    def _checkin(model: Any, pool: Any, session_id: str = "default") -> None:
        if pool is not None and model is not None:
            pool.checkin(model, session_id=session_id)


# -- native tool-call streaming helpers ------------------------------------

def _accumulate_tool_call(
    frags: dict[int, dict[str, str]], frag: dict[str, Any]
) -> None:
    """Merge one streamed ``tool_calls`` delta fragment into the accumulator.

    OpenAI-style streaming splits a single call across chunks: the first
    fragment carries ``id`` and ``function.name``; later fragments append
    ``function.arguments`` text. Fragments are keyed by ``index`` so parallel
    calls in one turn stay separate.
    """
    idx = int(frag.get("index", 0) or 0)
    acc = frags.setdefault(idx, {"id": "", "name": "", "arguments": ""})
    if frag.get("id"):
        acc["id"] = frag["id"]
    fn = frag.get("function") or {}
    if fn.get("name"):
        acc["name"] = fn["name"]
    if fn.get("arguments"):
        acc["arguments"] += fn["arguments"]


def _finalize_tool_calls(
    frags: dict[int, dict[str, str]]
) -> list[dict[str, Any]]:
    """Turn accumulated fragments into OpenAI-shaped tool_call dicts."""
    calls: list[dict[str, Any]] = []
    for i, (_idx, acc) in enumerate(sorted(frags.items())):
        if not acc["name"]:
            continue  # a fragment with no name is unusable — skip it
        calls.append({
            "id": acc["id"] or f"call_{i}",
            "type": "function",
            "function": {
                "name": acc["name"],
                "arguments": acc["arguments"] or "{}",
            },
        })
    return calls
