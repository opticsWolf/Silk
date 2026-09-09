# -*- coding: utf-8 -*-
"""Native (structured tool_calls) transport + GraphEngine capability (Qt-free).

Covers the OpenAI-style tool-calling path that runs when the loaded model
advertises support, and the fence fallback that keeps plain chat models
working. No real GGUF model — a stub ``create_chat_completion`` drives the
engine's streaming tool-call accumulation and message rendering.
"""

from __future__ import annotations

import json
import sys


from silk.functions.agent_loop import AgentLoop
from silk.functions.reflection import ReflectionConfig
from silk.functions.stream_events import (
    EventRunResult,
    EventToolCall,
)
from silk.functions.tool_calling import ToolCall, _Function
from silk.functions.tool_transport import (
    FenceTransport,
    NativeTransport,
    select_transport,
)
from silk.functions.usage_limits import UsageLimits


# ── transport selection ───────────────────────────────────────────────────


class _SupportsNative:
    def __init__(self, supports: bool):
        self._supports = supports
        self.enabled_with = None

    def supports_native_tools(self):
        return self._supports

    def enable_native_tools(self, schemas):
        self.enabled_with = schemas


class _SchemaToolbox:
    def get_tool_schemas(self):
        return [{"type": "function", "function": {"name": "t", "parameters": {}}}]


def test_select_transport_prefers_native_and_advertises_schemas():
    engine = _SupportsNative(True)
    toolbox = _SchemaToolbox()
    transport = select_transport(engine, toolbox)
    assert isinstance(transport, NativeTransport)
    assert engine.enabled_with == toolbox.get_tool_schemas()


def test_select_transport_falls_back_when_unsupported_or_no_toolbox():
    assert isinstance(select_transport(_SupportsNative(False), _SchemaToolbox()),
                      FenceTransport)
    # No native hooks at all (a plain fake) → fence.
    assert isinstance(select_transport(object(), _SchemaToolbox()), FenceTransport)
    # No toolbox → fence.
    assert isinstance(select_transport(_SupportsNative(True), None), FenceTransport)


# ── native loop end-to-end (fake engine) ──────────────────────────────────


class NativeFakeEngine:
    """Scripted engine that advertises native tool calling.

    script: list of ``(text, [(tool_name, args_dict), ...])`` per turn.
    """

    def __init__(self, script):
        self._script = list(script)
        self.usage_limits = UsageLimits()
        self.reflection_config = ReflectionConfig(max_retries=2)
        self.history: list = []
        self.last_stats: dict = {}
        self._stopped = False
        self._enabled = False
        self.schemas = None
        self._pending: list = []

    # native capability
    def supports_native_tools(self):
        return True

    def enable_native_tools(self, schemas):
        self._enabled = True
        self.schemas = schemas

    def pull_tool_calls(self):
        calls, self._pending = self._pending, []
        return calls

    def append_tool_result(self, call_id, name, content):
        self.history.append({"role": "tool", "tool_call_id": call_id,
                             "name": name, "content": content})

    # engine basics
    def stream_response(self, gen_params):
        text, calls = self._script.pop(0) if self._script else ("(done)", [])
        self.last_stats = {"tokens": 1, "input_tokens": 1, "tps": 1.0,
                           "finish_reason": "tool_calls" if calls else "stop"}
        self._pending = [
            ToolCall(id=f"call_{i}",
                     function=_Function(name, json.dumps(args)))
            for i, (name, args) in enumerate(calls)
        ]
        for word in (text.split(" ") if text else []):
            yield word + " "

    def append_message(self, role, content, **kw):
        entry = {"role": role, "content": content}
        if role == "assistant" and self._enabled and self._pending:
            entry["tool_calls"] = list(self._pending)
        self.history.append(entry)

    def count_prompt_tokens(self):
        return 1

    def request_stop(self):
        self._stopped = True

    def stop_requested(self):
        return self._stopped


class NativeFakeToolbox:
    def __init__(self, results):
        self.tools = {name: {} for name in results}
        self._results = results
        self.executed: list = []

    def get_tool_schemas(self):
        return [{"type": "function", "function": {"name": n, "parameters": {}}}
                for n in self.tools]

    async def execute_tool_calls_async(self, tool_calls):
        out = []
        for tc in tool_calls:
            self.executed.append(tc.function.name)
            body = self._results.get(tc.function.name, json.dumps({"error": "?"}))
            out.append({"tool_call_id": tc.id, "name": tc.function.name,
                        "content": body})
        return out


def test_native_loop_dispatches_structured_calls_and_pairs_results():
    engine = NativeFakeEngine([("", [("echo", {"text": "hi"})]),
                              ("final answer", [])])
    toolbox = NativeFakeToolbox({"echo": "echo says hi"})
    events = list(AgentLoop(engine, toolbox).run("go", {}))

    # The structured call was dispatched (not a text fence).
    assert toolbox.executed == ["echo"]
    assert engine.schemas is not None, "schemas advertised to the engine"
    assert [e.tool_name for e in events if isinstance(e, EventToolCall)] == ["echo"]

    # Result fed back via the native tool role, id-paired.
    tool_turns = [h for h in engine.history if h["role"] == "tool"]
    assert len(tool_turns) == 1
    assert tool_turns[0]["tool_call_id"] == "call_0"
    assert tool_turns[0]["name"] == "echo"

    # The assistant turn carries the structured tool_calls (for rendering).
    asst = [h for h in engine.history if h["role"] == "assistant"]
    assert asst[0].get("tool_calls"), "assistant turn carries structured tool_calls"

    run_result = [e for e in events if isinstance(e, EventRunResult)][0]
    assert run_result.text.strip() == "final answer"


# ── GraphEngine native streaming + rendering (stub model) ──────────────────


class _StubModel:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_messages = None
        self.last_kw = None

    def create_chat_completion(self, messages, stream, **kw):
        self.last_messages = messages
        self.last_kw = kw
        return iter(self._chunks)


def test_graph_engine_accumulates_streamed_tool_calls_and_renders_native():
    from silk.functions.graph_engine import GraphEngine

    # One tool call, arguments split across two chunks (OpenAI streaming shape).
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_x",
             "function": {"name": "read_file", "arguments": '{"pa'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'th": "a.txt"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    model = _StubModel(chunks)
    eng = GraphEngine({"backend": "gguf", "model": model, "supports_tools": True})

    assert eng.supports_native_tools()
    eng.enable_native_tools(
        [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    )

    list(eng.stream_response({}))  # drain the (empty) text stream
    assert "tools" in model.last_kw and model.last_kw["tool_choice"] == "auto"

    # Mirror the loop's order: append assistant (attaches pending) → pull.
    eng.append_message("assistant", "")
    calls = eng.pull_tool_calls()
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert json.loads(calls[0].function.arguments) == {"path": "a.txt"}

    eng.append_tool_result("call_x", "read_file", "file body")
    msgs = eng.build_messages()
    asst = [m for m in msgs if m["role"] == "assistant"][0]
    assert asst["tool_calls"][0]["function"]["name"] == "read_file"
    tool_msg = [m for m in msgs if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "call_x"
    assert tool_msg["content"] == "file body"


def test_graph_engine_fence_path_unaffected_without_support_flag():
    from silk.functions.graph_engine import GraphEngine

    model = _StubModel([{"choices": [{"delta": {"content": "hi"}}]}])
    eng = GraphEngine({"backend": "gguf", "model": model})  # no supports_tools
    assert not eng.supports_native_tools()

    list(eng.stream_response({}))
    # No tools advertised on the fence path.
    assert "tools" not in (model.last_kw or {})
    # tool turns render as Tool Output user messages.
    eng.append_message("tool", '{"name": "t", "content": "body"}')
    msgs = eng.build_messages()
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"].startswith("Tool Output:")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
