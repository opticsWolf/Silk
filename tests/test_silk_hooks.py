# -*- coding: utf-8 -*-
"""Tests for the configurable hook system (Qt-free).

Covers the named hook catalog (names → callables, unknown names
skipped), recipe-style attachment, capability-carried hooks now being
registered on load and removed on RoleBinding deactivation, and the new
run/model-level emit points in AgentLoop.
"""

from __future__ import annotations



from silk.functions.capabilities import HooksCapability
from silk.functions.agent_loop import AgentLoop
from silk.functions.hook_catalog import (
    HOOK_CATALOG,
    attach_catalog_hooks,
    build_hooks,
    catalog_names,
)
from silk.functions.hooks import (
    HOOK_AFTER_MODEL_RESPONSE,
    HOOK_AFTER_RUN,
    HOOK_AFTER_TOOL_EXECUTE,
    HOOK_BEFORE_MODEL_REQUEST,
    HOOK_BEFORE_RUN,
    HOOK_BEFORE_TOOL_EXECUTE,
    HOOK_TOOL_DENIED,
    HookRegistry,
)
from silk.functions.presets import PresetStore, RolePreset
from silk.functions.role import ALLOW_ALL, Role, RoleBinding
from silk.functions.tool_box import ToolBox


# ── catalog ──────────────────────────────────────────────────────────────


def test_catalog_ships_starter_hooks():
    assert {"log_tool_calls", "timing", "usage_meter"} <= set(catalog_names())


def test_build_hooks_merges_and_skips_unknown():
    hooks = build_hooks(["log_tool_calls", "timing", "no_such_hook"])
    # Both known hooks contribute to before/after tool execute.
    assert len(hooks[HOOK_BEFORE_TOOL_EXECUTE]) == 2
    assert len(hooks[HOOK_AFTER_TOOL_EXECUTE]) == 2
    # log_tool_calls alone handles denials.
    assert len(hooks[HOOK_TOOL_DENIED]) == 1
    assert all(callable(cb) for cbs in hooks.values() for cb in cbs)


def test_factories_produce_independent_closures():
    a = HOOK_CATALOG["usage_meter"].factory()
    b = HOOK_CATALOG["usage_meter"].factory()
    assert a[HOOK_BEFORE_TOOL_EXECUTE][0] is not b[HOOK_BEFORE_TOOL_EXECUTE][0]


def test_attach_catalog_hooks_registers_on_toolbox():
    box = ToolBox()
    attach_catalog_hooks(box, None, names=("log_tool_calls",))
    assert box.hooks._hooks[HOOK_BEFORE_TOOL_EXECUTE]
    assert box.hooks._hooks[HOOK_TOOL_DENIED]
    # Catalog hooks must survive being emitted with real kwargs.
    box.hooks.emit(HOOK_BEFORE_TOOL_EXECUTE, tool_name="t", tool_args={"a": 1})
    box.hooks.emit(HOOK_TOOL_DENIED, tool_name="t")
    # Transparency stamp: attached names are recorded (accumulating).
    assert box.catalog_hook_names == ("log_tool_calls",)
    attach_catalog_hooks(box, None, names=("timing", "log_tool_calls"))
    assert box.catalog_hook_names == ("log_tool_calls", "timing")


def test_catalog_hook_names_survive_toolset_derivation(tmp_path):
    """The Role node reads the stamp off the *toolset* — recipe replay
    must re-create it on every derived instance."""
    from functools import partial

    from silk.functions.tools.file_sandbox import FileToolSandbox
    from silk.functions.tools.file_read import attach_file_read_tools
    from silk.functions.toolset_build import build_toolset

    sandbox = FileToolSandbox(root_dir=tmp_path)
    recipe = (
        ("file_read", attach_file_read_tools),
        ("hooks", partial(attach_catalog_hooks, names=("usage_meter",))),
    )
    box = ToolBox()
    for _name, attacher in recipe:
        attacher(box, sandbox)
    box.build_recipe = recipe
    box.base_sandbox = sandbox

    toolset = build_toolset(box, ["read_file"])
    assert toolset.catalog_hook_names == ("usage_meter",)
    assert toolset.hooks._hooks[HOOK_BEFORE_TOOL_EXECUTE], \
        "infrastructure hooks must be live on the derived toolset"


# ── capability-carried hooks ─────────────────────────────────────────────


def test_capability_hooks_registered_on_activation_and_removed():
    seen: list[str] = []

    def probe(tool_name: str = "", **_kw) -> None:
        seen.append(tool_name)

    box = ToolBox()
    cap = HooksCapability(
        id="probe-cap", description="probe",
        hooks={HOOK_BEFORE_TOOL_EXECUTE: probe},  # bare callable, not a list
    )
    role = Role(id="r", selector=ALLOW_ALL, capabilities=[cap])

    binding = RoleBinding.activate(role, box)
    box.hooks.emit(HOOK_BEFORE_TOOL_EXECUTE, tool_name="alpha", tool_args={})
    assert seen == ["alpha"], "capability hook must fire while active"

    binding.deactivate()
    box.hooks.emit(HOOK_BEFORE_TOOL_EXECUTE, tool_name="beta", tool_args={})
    assert seen == ["alpha"], "capability hook must be gone after deactivation"
    assert "probe-cap" not in box._capability_hooks


# ── AgentLoop run/model-level emit points ────────────────────────────────


class _NoLimits:
    def check_request(self): pass
    def check_input_tokens(self, n): pass
    def check_tool_calls(self, n): pass
    def record_tool_calls(self, n): pass
    def snapshot(self): return {}


class _Reflection:
    max_retries = 0
    max_output_retries = 0
    tool_error_prompt = ""


class _MiniEngine:
    """Just enough AgentEngine for a single plain-text round."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.history: list = []
        self.last_stats: dict = {}
        self.usage_limits = _NoLimits()
        self.reflection_config = _Reflection()

    def append_message(self, role, content, **kw):
        self.history.append((role, content))

    def count_prompt_tokens(self):
        return 0

    def stream_response(self, gen_params):
        self.last_stats = {"tokens": 1, "finish_reason": "stop"}
        yield self._text

    def request_stop(self): pass
    def stop_requested(self): return False


class _HooksOnlyToolbox:
    """Registry stand-in: carries hooks; never reached for execution."""

    def __init__(self) -> None:
        self.hooks = HookRegistry()
        self.tools: dict = {}

    async def execute_tool_calls_async(self, calls):
        return []


def test_agent_loop_emits_run_and_model_events_in_order():
    toolbox = _HooksOnlyToolbox()
    order: list[str] = []
    for event in (HOOK_BEFORE_RUN, HOOK_BEFORE_MODEL_REQUEST,
                  HOOK_AFTER_MODEL_RESPONSE, HOOK_AFTER_RUN):
        toolbox.hooks.register(event, lambda _e=event, **kw: order.append(_e))

    loop = AgentLoop(_MiniEngine("plain answer"), toolbox)
    events = list(loop.run("hi", {}))

    assert events[-1].text == "plain answer"
    assert order == [
        HOOK_BEFORE_RUN,
        HOOK_BEFORE_MODEL_REQUEST,
        HOOK_AFTER_MODEL_RESPONSE,
        HOOK_AFTER_RUN,
    ]


def test_agent_loop_after_run_carries_final_text_and_rounds():
    toolbox = _HooksOnlyToolbox()
    captured: dict = {}
    toolbox.hooks.register(
        HOOK_AFTER_RUN, lambda **kw: captured.update(kw)
    )

    loop = AgentLoop(_MiniEngine("done!"), toolbox)
    list(loop.run("hi", {}))

    assert captured["final_text"] == "done!"
    assert captured["rounds"] == 1
    assert captured["elapsed_s"] >= 0.0


def test_agent_loop_without_toolbox_emits_nothing_and_still_runs():
    loop = AgentLoop(_MiniEngine("no tools here"), toolbox=None)
    events = list(loop.run("hi", {}))
    assert events[-1].text == "no tools here"


# ── configurable middleware hooks (redaction, budget) ────────────────────


def _register_echo_tool(box: ToolBox, reply: str):
    """A dummy tool returning *reply*; returns a call-count spy list."""
    calls: list[int] = []

    @box.register("echo", "returns a fixed string")
    def echo(db_pool, user_session):
        calls.append(1)
        return reply

    return calls


def _call(name: str = "echo", call_id: str = "c1"):
    import json
    from types import SimpleNamespace
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps({}))
    )


def test_redact_secrets_rewrites_tool_results():
    import asyncio

    box = ToolBox()
    _register_echo_tool(box, "the key is AKIAABCDEFGHIJKLMNOP, use it")
    attach_catalog_hooks(box, None, names=("redact_secrets",))

    results = asyncio.run(box.execute_tool_calls_async([_call()]))
    content = results[0]["content"]
    assert "AKIA" not in content
    assert "[REDACTED]" in content


def test_redact_secrets_custom_pattern_and_replacement():
    import asyncio

    box = ToolBox()
    _register_echo_tool(box, "internal codename: bluebird")
    attach_catalog_hooks(
        box, None,
        names=("redact_secrets",),
        configs={"redact_secrets": {
            "patterns": [r"bluebird"], "replacement": "███",
        }},
    )

    results = asyncio.run(box.execute_tool_calls_async([_call()]))
    assert "bluebird" not in results[0]["content"]
    assert "███" in results[0]["content"]


def test_tool_budget_denies_beyond_limit_and_is_not_retryable():
    import asyncio
    import json

    from silk.functions.reflection import is_retryable_tool_error

    box = ToolBox()
    calls = _register_echo_tool(box, "ok")
    attach_catalog_hooks(
        box, None,
        names=("tool_budget",),
        configs={"tool_budget": {"max_calls": 2}},
    )

    r1 = asyncio.run(box.execute_tool_calls_async([_call(call_id="1")]))
    r2 = asyncio.run(box.execute_tool_calls_async([_call(call_id="2")]))
    r3 = asyncio.run(box.execute_tool_calls_async([_call(call_id="3")]))

    assert r1[0]["content"] == "ok" and r2[0]["content"] == "ok"
    payload = json.loads(r3[0]["content"])
    assert payload["error_type"] == "budget_exceeded"
    assert len(calls) == 2, "the denied call must never reach the executable"
    assert not is_retryable_tool_error(r3[0]["content"]), \
        "budget denials must not burn reflection retries"


def test_tool_budget_resets_on_new_run():
    import asyncio

    box = ToolBox()
    calls = _register_echo_tool(box, "ok")
    attach_catalog_hooks(
        box, None,
        names=("tool_budget",),
        configs={"tool_budget": {"max_calls": 1}},
    )

    asyncio.run(box.execute_tool_calls_async([_call(call_id="1")]))
    box.hooks.emit(HOOK_BEFORE_RUN)  # AgentLoop emits this at run start
    asyncio.run(box.execute_tool_calls_async([_call(call_id="2")]))
    assert len(calls) == 2, "a new run must reset the budget"


def _register_spy_tool(box: ToolBox, name: str, reply: str = "ok"):
    """A tool that records the kwargs it was called with (for hook tests)."""
    calls: list[dict] = []

    @box.register(name, "spy")
    def _spy(db_pool, user_session, **kw):
        calls.append(kw)
        return reply

    return calls


def _call_with(name: str, args: dict, call_id: str = "c1"):
    import json
    from types import SimpleNamespace
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args))
    )


def test_task_audit_in_catalog():
    assert "task_audit" in catalog_names()


def test_task_audit_bounces_trivial_rationale():
    import asyncio
    import json

    box = ToolBox()
    calls = _register_spy_tool(box, "task_add")
    attach_catalog_hooks(box, None, names=("task_audit",))

    # trivial rationale -> short-circuited, store never reached
    r = asyncio.run(box.execute_tool_calls_async(
        [_call_with("task_add", {"title": "x", "rationale": "n/a"})]))
    payload = json.loads(r[0]["content"])
    assert "vague" in payload["error"].lower()
    assert calls == [], "a trivial rationale must not reach the executable"

    # a real rationale passes through
    r = asyncio.run(box.execute_tool_calls_async(
        [_call_with("task_add", {"title": "x",
                                 "rationale": "cover the exporter edge cases"})]))
    assert r[0]["content"] == "ok" and len(calls) == 1


def test_task_audit_leaves_progress_and_other_tools_alone():
    import asyncio

    box = ToolBox()
    upd = _register_spy_tool(box, "task_update")     # plain progress, not guarded
    other = _register_spy_tool(box, "echo2")         # unrelated tool
    attach_catalog_hooks(box, None, names=("task_audit",))

    asyncio.run(box.execute_tool_calls_async(
        [_call_with("task_update", {"id": "t1", "status": "in_progress"})]))
    asyncio.run(box.execute_tool_calls_async([_call_with("echo2", {})]))
    assert len(upd) == 1 and len(other) == 1


def test_task_audit_strict_off_allows_trivial():
    import asyncio

    box = ToolBox()
    calls = _register_spy_tool(box, "task_complete")
    attach_catalog_hooks(box, None, names=("task_audit",),
                         configs={"task_audit": {"strict": False}})
    asyncio.run(box.execute_tool_calls_async(
        [_call_with("task_complete", {"id": "t1", "rationale": "ok"})]))
    assert len(calls) == 1, "strict=False must let a trivial rationale through"


def test_role_can_carry_middleware_hooks_and_remove_them():
    from silk.functions.hooks import HOOK_WRAP_TOOL_EXECUTE

    role = Role(
        id="budgeted", selector=ALLOW_ALL,
        hooks=build_hooks(["tool_budget"], {"tool_budget": {"max_calls": 5}}),
    )
    box = ToolBox()
    binding = RoleBinding.activate(role, box)
    assert box.hooks._middleware[HOOK_WRAP_TOOL_EXECUTE], \
        "wrap_* role hooks must land in the middleware registry"
    binding.deactivate()
    assert not box.hooks._middleware.get(HOOK_WRAP_TOOL_EXECUTE), \
        "middleware must be removed on deactivation"


def test_middleware_handler_can_reinvoke_the_chain():
    """A wrap_* middleware may call ``handler()`` more than once (retry /
    error-recovery). Each call must re-run the *remaining* chain — the old
    implementation popped from a shared list, so the second invocation found
    it emptied and silently skipped every middleware below it."""
    import asyncio
    from silk.functions.hooks import (
        HookRegistry, HOOK_WRAP_TOOL_EXECUTE,
    )

    reg = HookRegistry()
    seen_by_inner_mw: list[int] = []
    innermost_runs: list[int] = []

    async def retry_mw(handler, **kw):
        first = await handler(**kw)     # run the downstream chain twice
        second = await handler(**kw)
        return (first, second)

    async def counting_mw(handler, **kw):
        seen_by_inner_mw.append(1)
        return await handler(**kw)

    reg.register_middleware(HOOK_WRAP_TOOL_EXECUTE, retry_mw)
    reg.register_middleware(HOOK_WRAP_TOOL_EXECUTE, counting_mw)

    async def innermost(**kw):
        innermost_runs.append(1)
        return "ran"

    result = asyncio.run(
        reg.emit_middleware(HOOK_WRAP_TOOL_EXECUTE, innermost=innermost)
    )
    assert result == ("ran", "ran")
    assert len(seen_by_inner_mw) == 2, "inner middleware must run on every handler() call"
    assert len(innermost_runs) == 2, "innermost must run on every handler() call"


def test_invalid_hook_config_falls_back_to_defaults():
    hooks = build_hooks(
        ["tool_budget"], {"tool_budget": {"max_calls": "not-a-number"}}
    )
    # Falls back to defaults instead of raising — the hook still exists.
    from silk.functions.hooks import HOOK_WRAP_TOOL_EXECUTE
    assert hooks[HOOK_WRAP_TOOL_EXECUTE]


# ── event formatting / counting (Hook Monitor logic) ─────────────────────


def test_format_event_covers_every_type():
    """One vocabulary, one formatter: every EventType renders (spec D2)."""
    from silk.functions.event_format import format_event
    from silk.functions.stream_events import EventType

    samples = {
        EventType.RUN_START: {"context_length": 8192},
        EventType.RUN_FINISHED: {"rounds": 2, "elapsed_s": 1.25},
        EventType.RUN_RESULT: {"outcome": "completed"},
        EventType.MODEL_REQUEST: {"round": 1},
        EventType.MODEL_RESPONSE: {"round": 1, "chars": 42},
        EventType.TOOL_CALL: {"tool_name": "read_file",
                              "tool_args": {"path": "a.txt"}},
        EventType.TOOL_RESULT: {"tool_name": "read_file", "chars": 120},
        EventType.TOOL_DENIED: {"tool_name": "write_file"},
        EventType.PLAN: {"revision": 4},
        EventType.COMPACTION: {"turns_dropped": 3, "tokens_before": 900,
                               "tokens_after": 300},
        EventType.DECISION_REQUEST: {"kind": "approval", "prompt": "ok?"},
        EventType.DECISION_RESPONSE: {"kind": "approval", "approved": True},
        EventType.ERROR: {"context": "stream_response", "error": "boom"},
        EventType.USAGE_LIMIT: {"limit_type": "requests", "limit_value": 4,
                                "current_value": 5},
        EventType.REFLECTION: {"retry_count": 0, "max_retries": 2,
                               "error_type": "validation"},
        EventType.WORKER: {"worker": "researcher", "event_type": "EventToolCall",
                           "digest": "read_file"},
        EventType.CHAT_TURN: {"ai": "hello"},
        EventType.DELTA: {"chars": 3},
        EventType.FINAL_RESULT: {},
    }
    assert set(samples) == set(EventType), "a new event type needs a log line"

    lines = {member: format_event({"type": member.value, **fields})
             for member, fields in samples.items()}
    assert "read_file" in lines[EventType.TOOL_CALL]
    assert "a.txt" in lines[EventType.TOOL_CALL]
    assert "DENIED" in lines[EventType.TOOL_DENIED]
    assert "2 round(s)" in lines[EventType.RUN_FINISHED]
    assert "revision 4" in lines[EventType.PLAN]
    assert "researcher" in lines[EventType.WORKER]
    assert all(line.startswith("[") for line in lines.values()),         "every line is timestamped"


def test_format_event_names_the_agent_when_streams_are_merged():
    from silk.functions.event_format import format_event
    from silk.functions.stream_events import EventType

    line = format_event({"type": EventType.MODEL_REQUEST.value, "round": 1,
                         "agent": "Coordinator"})
    assert "Coordinator" in line


def test_event_counter_and_dedup_key():
    from silk.functions.event_format import EventCounter, event_key
    from silk.functions.stream_events import EventType

    call = EventType.TOOL_CALL.value
    counter = EventCounter()
    counter.record({"type": call, "tool_name": "read_file"})
    counter.record({"type": call, "tool_name": "read_file"})
    counter.record({"type": EventType.TOOL_DENIED.value, "tool_name": "write_file"})
    counter.record({"type": EventType.RUN_FINISHED.value})

    assert counter.kinds[call] == 2
    assert counter.tools["read_file"] == 2
    assert "denied: 1" in counter.summary()
    assert counter.as_dict()["tools"] == {"read_file": 2}

    assert event_key({"run_id": "r", "seq": 3}) == ("r", 3)
    assert event_key({"type": call}) is None, "no key → no dedup"

    counter.clear()
    assert not counter.kinds and not counter.tools


# ── presets carry hook names ─────────────────────────────────────────────


def test_role_preset_round_trips_hook_names(tmp_path):
    store = PresetStore("roles_hooks_test", RolePreset, directory=tmp_path)
    store.upsert(RolePreset(name="observed", hooks=["log_tool_calls", "timing"]))

    reloaded = PresetStore("roles_hooks_test", RolePreset, directory=tmp_path)
    assert reloaded.get("observed").hooks == ["log_tool_calls", "timing"]
