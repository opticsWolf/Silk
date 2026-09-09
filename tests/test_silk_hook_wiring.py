# -*- coding: utf-8 -*-
"""The error family fires, and a hook that could not fire is refused.

Spec D15 / gap G3. Eleven of nineteen declared hook events were never
emitted, and the consequence was not the missing events themselves: it was
that registering on one succeeded. A hook that looks installed and never
runs is worse than an error, because what you have to notice is an
*absence*. So the error family is wired, and registration on anything
nothing emits raises.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


from pydantic import BaseModel, Field

from silk.functions.agent_loop import AgentLoop
from silk.functions.hooks import (
    HOOK_AFTER_MODEL_REQUEST,
    HOOK_ON_MODEL_REQUEST_ERROR,
    HOOK_ON_TOOL_EXECUTE_ERROR,
    HOOK_ON_TOOL_VALIDATE_ERROR,
    HOOK_WRAP_TOOL_VALIDATE,
    KNOWN_EVENTS,
    UNWIRED_EVENTS,
    WIRED_EVENTS,
    HookRegistry,
    UnwiredHookEvent,
    register_hook_map,
)
from silk.functions.reflection import ReflectionConfig
from silk.functions.tool_box import ToolBox
from silk.functions.usage_limits import UsageLimits


# -- registration refuses what it cannot honour ---------------------------

def test_every_declared_event_is_one_that_fires():
    """§22 q2 closed the five open WRAP_* names: one wired, four deleted.

    The guard that refuses an unwired registration stays -- it is what
    catches a typo -- but there is nothing left for it to refuse except
    misspellings, which is the state it was built to reach.
    """
    assert UNWIRED_EVENTS == frozenset()
    assert WIRED_EVENTS == KNOWN_EVENTS


def test_registering_on_a_misspelled_event_raises_too():
    registry = HookRegistry()
    with pytest.raises(UnwiredHookEvent) as exc:
        registry.register("after_toool_execute", lambda **kw: None)
    assert "no such hook event" in str(exc.value)


def test_a_wired_event_still_registers_normally():
    registry = HookRegistry()
    seen = []
    for event in sorted(WIRED_EVENTS):
        if event.startswith("wrap_"):
            continue
        registry.register(event, lambda **kw: seen.append(1))
    registry.emit(HOOK_AFTER_MODEL_REQUEST, round_index=0)
    assert seen


def test_a_hook_map_naming_an_event_that_does_not_exist_registers_nothing():
    """Half-installed is a state nobody described."""
    registry = HookRegistry()
    good = lambda **kw: None                              # noqa: E731
    with pytest.raises(UnwiredHookEvent):
        register_hook_map(registry, {
            "before_run": good,
            "wrap_output_validate": good,     # deleted by §22 q2
        })
    assert not registry._hooks.get("before_run")


def test_the_deleted_wrap_events_are_gone_from_the_vocabulary():
    """Deleted, not merely unwired: the name itself is a misspelling now."""
    for name in ("wrap_model_request", "wrap_output_validate",
                 "wrap_output_process", "wrap_run_event_stream"):
        assert name not in KNOWN_EVENTS
        with pytest.raises(UnwiredHookEvent) as exc:
            HookRegistry().register_middleware(name, lambda **kw: None)
        assert "no such hook event" in str(exc.value)


def test_the_one_that_survived_is_registrable():
    registry = HookRegistry()
    registry.register_middleware(HOOK_WRAP_TOOL_VALIDATE,
                                 lambda handler, **kw: handler(**kw))
    assert registry._middleware[HOOK_WRAP_TOOL_VALIDATE]


# -- the model-request half fires -----------------------------------------

class _Engine:
    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.usage_limits = UsageLimits()
        self.reflection_config = ReflectionConfig()
        self.history: list[dict] = []
        self.last_stats: dict = {}

    def stream_response(self, gen_params):
        if self._behaviour == "raise":
            raise RuntimeError("Requested tokens exceed context window")
        yield "hello"
        self.last_stats = {"finish_reason": "stop", "tokens": 1}

    def append_message(self, role, content, **stats):
        self.history.append({"role": role, "content": content})

    def count_prompt_tokens(self):
        return 1

    def request_stop(self):
        pass

    def stop_requested(self):
        return False


class _Box:
    def __init__(self):
        self.hooks = HookRegistry()
        self.tools: dict = {}

    async def execute_tool_calls_async(self, tool_calls):
        return []


def test_after_model_request_fires_once_per_request():
    box = _Box()
    seen = []
    box.hooks.register(HOOK_AFTER_MODEL_REQUEST,
                       lambda **kw: seen.append(kw.get("ok")))
    list(AgentLoop(_Engine("ok"), box).run("go"))
    assert seen == [True]


def test_a_failed_request_reaches_the_error_hook_with_its_classification():
    box = _Box()
    seen = []
    box.hooks.register(HOOK_ON_MODEL_REQUEST_ERROR,
                       lambda **kw: seen.append((kw.get("kind"), kw.get("round_index"))))
    list(AgentLoop(_Engine("raise"), box).run("go"))
    assert seen == [("overflow", 0)]


# -- and the tool half ----------------------------------------------------

class _Args(BaseModel):
    text: str = Field(..., description="Text.")


def _box_with_tools():
    box = ToolBox(db_pool=None, user_session={})

    @box.register(name="ok_tool", description="Echo.", args_model=_Args,
                  category="util")
    def _ok(db_pool, user_session, text: str):
        return {"echoed": text}

    @box.register(name="boom", description="Fail.", args_model=_Args,
                  category="util")
    def _boom(db_pool, user_session, text: str):
        raise RuntimeError("nope")

    return box


def _call(box, name, arguments):
    tc = SimpleNamespace(id="c1",
                         function=SimpleNamespace(name=name, arguments=arguments))
    return asyncio.run(box.execute_tool_calls_async([tc]))


def test_bad_arguments_reach_the_validate_error_hook():
    box = _box_with_tools()
    seen = []
    box.hooks.register(HOOK_ON_TOOL_VALIDATE_ERROR,
                       lambda **kw: seen.append(kw.get("tool_name")))
    _call(box, "ok_tool", json.dumps({"wrong": 1}))
    assert seen == ["ok_tool"]


def test_unparseable_arguments_reach_it_too():
    box = _box_with_tools()
    seen = []
    box.hooks.register(HOOK_ON_TOOL_VALIDATE_ERROR,
                       lambda **kw: seen.append(kw.get("tool_name")))
    _call(box, "ok_tool", "{not json")
    assert seen == ["ok_tool"]


def test_a_raising_tool_reaches_the_execute_error_hook():
    box = _box_with_tools()
    seen = []
    box.hooks.register(HOOK_ON_TOOL_EXECUTE_ERROR,
                       lambda **kw: seen.append(kw.get("error")))
    out = _call(box, "boom", json.dumps({"text": "x"}))
    assert seen == ["nope"]
    assert "Execution failed" in out[0]["content"]


# -- the one that got wired (§22 q2) --------------------------------------


class _CountArgs(BaseModel):
    count: int = Field(..., description="How many.")


def _validating_box(middleware=None, *, tools=None):
    box = ToolBox()

    @box.register(name="counter", tags=("t",), category="test", risk="low",
                  description="Takes a number.", args_model=_CountArgs)
    def _counter(db_pool, user_session, count: int):
        return {"count": count}

    if middleware is not None:
        box.hooks.register_middleware("wrap_tool_validate", middleware,
                                      tools=tools)
    return box


def _validate_call(box, raw):
    import asyncio
    import json
    from types import SimpleNamespace

    request = SimpleNamespace(id="c1", function=SimpleNamespace(
        name="counter", arguments=raw))
    out = asyncio.run(box.execute_tool_calls_async([request]))
    return json.loads(out[0]["content"])


def test_validation_runs_unwrapped_when_nothing_is_registered():
    assert _validate_call(_validating_box(), '{"count": 2}')["count"] == 2


def test_a_middleware_can_repair_arguments_the_model_got_wrong():
    """The use that justifies wiring this one: a fixable mistake.

    Without it the model gets a validation error and spends a round
    correcting a quoted number.
    """
    seen = []

    async def repair(handler=None, tool_name="", raw_args="", **_kw):
        seen.append((tool_name, raw_args))
        return await handler(raw_args=raw_args.replace('"3"', "3"))

    assert _validate_call(_validating_box(repair), '{"count": "3"}')["count"] == 3
    assert seen == [("counter", '{"count": "3"}')]


def test_a_middleware_can_refuse_before_the_tool_is_reached():
    """A refusal here ends one call, not the run.

    The model reads why and can try something else -- the same shape
    every other validation outcome has.
    """
    ran = []

    async def refuse(handler=None, **_kw):
        raise ValueError("not this one")

    box = _validating_box(refuse)
    box.tools["counter"]["procedure"] = lambda *a, **k: ran.append(1)

    result = _validate_call(box, '{"count": 1}')
    assert "refused before the tool ran" in result["error"]
    assert "not this one" in result["error"]
    assert ran == [], "the procedure never saw the call"


def test_a_middleware_bound_to_other_tools_does_not_fire():
    """`tools=` filtering works here because the wrap knows the name."""
    fired = []

    async def other(handler=None, **kw):
        fired.append(1)
        return await handler(**kw)

    box = _validating_box(other, tools=("something_else",))
    assert _validate_call(box, '{"count": 4}')["count"] == 4
    assert fired == []
