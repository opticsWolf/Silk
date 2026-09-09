# -*- coding: utf-8 -*-
"""Per-tool hook binding and the essential tier (spec D13/D14, Phase 2 item 1).

Two things used to be invisible. A hook that only cared about one tool said
so inside its own body, so no config, UI or test could see which hooks
touched which tools; and "infrastructure hooks: part of the recipe" was a
comment in ``nodes/toolbox.py`` rather than something the registry enforced.
These tests pin both as declarations the registry acts on.
"""

from __future__ import annotations

import asyncio

import pytest


from silk.functions.hooks import (
    HOOK_AFTER_TOOL_EXECUTE,
    HOOK_BEFORE_RUN,
    HOOK_BEFORE_TOOL_EXECUTE,
    HOOK_WRAP_TOOL_EXECUTE,
    EssentialHookError,
    HookRegistry,
    bind_tools,
    essential,
    register_hook_map,
    unregister_hook_map,
)
from silk.functions.tool_box import ToolBox
from silk.functions.toolset_build import carry_essential_hooks


def _recorder():
    seen: list[str] = []

    def hook(tool_name: str = "", **_kw) -> None:
        seen.append(tool_name)

    return hook, seen


# -- D13: the registry does the filtering ---------------------------------


def test_an_unbound_hook_fires_for_every_tool():
    registry = HookRegistry()
    hook, seen = _recorder()
    registry.register(HOOK_BEFORE_TOOL_EXECUTE, hook)

    for name in ("read_file", "write_file"):
        registry.emit(HOOK_BEFORE_TOOL_EXECUTE, tool_name=name)

    assert seen == ["read_file", "write_file"]


def test_a_tool_bound_hook_fires_only_for_its_tools():
    registry = HookRegistry()
    hook, seen = _recorder()
    registry.register(HOOK_BEFORE_TOOL_EXECUTE, hook, tools=["write_file"])

    for name in ("read_file", "write_file", "run_python"):
        registry.emit(HOOK_BEFORE_TOOL_EXECUTE, tool_name=name)

    assert seen == ["write_file"]


def test_a_bound_hook_stays_quiet_on_a_tool_less_event():
    """It declared it was about a tool; no tool, nothing to say."""
    registry = HookRegistry()
    fired: list[int] = []
    registry.register(HOOK_BEFORE_RUN, lambda **_kw: fired.append(1),
                      tools=["write_file"])
    registry.emit(HOOK_BEFORE_RUN)
    assert fired == []


def test_a_category_bound_hook_uses_the_boxs_index():
    """The registry has no tool index of its own; the ToolBox lends it one."""
    box = ToolBox()

    @box.register("peek", "reads", category="filesystem")
    def peek() -> str:
        return "ok"

    @box.register("ponder", "thinks", category="reasoning")
    def ponder() -> str:
        return "ok"

    hook, seen = _recorder()
    box.hooks.register(HOOK_BEFORE_TOOL_EXECUTE, hook, categories=["filesystem"])

    box.hooks.emit(HOOK_BEFORE_TOOL_EXECUTE, tool_name="peek")
    box.hooks.emit(HOOK_BEFORE_TOOL_EXECUTE, tool_name="ponder")

    assert seen == ["peek"]


def test_an_unbound_registry_matches_no_category():
    """Unknown category -> quiet, not universal. The safe direction."""
    registry = HookRegistry()  # nothing lent it an index
    hook, seen = _recorder()
    registry.register(HOOK_BEFORE_TOOL_EXECUTE, hook, categories=["filesystem"])
    registry.emit(HOOK_BEFORE_TOOL_EXECUTE, tool_name="peek")
    assert seen == []


def test_a_binding_can_ride_a_hook_map_as_a_decorator():
    """A hook map is a plain dict, so the declaration goes on the function."""
    seen: list[str] = []

    @bind_tools("write_file")
    def only_writes(tool_name: str = "", **_kw) -> None:
        seen.append(tool_name)

    registry = HookRegistry()
    register_hook_map(registry, {HOOK_AFTER_TOOL_EXECUTE: [only_writes]})

    registry.emit(HOOK_AFTER_TOOL_EXECUTE, tool_name="read_file")
    registry.emit(HOOK_AFTER_TOOL_EXECUTE, tool_name="write_file")
    assert seen == ["write_file"]


def test_middleware_is_bound_too():
    registry = HookRegistry()
    wrapped: list[str] = []

    async def guard(handler=None, tool_name: str = "", **_kw):
        wrapped.append(tool_name)
        return await handler()

    async def innermost(**_kw):
        return "ran"

    registry.register_middleware(HOOK_WRAP_TOOL_EXECUTE, guard,
                                 tools=["write_file"])

    async def drive():
        return [
            await registry.emit_middleware(
                HOOK_WRAP_TOOL_EXECUTE, innermost=innermost, tool_name=name,
            )
            for name in ("read_file", "write_file")
        ]

    assert asyncio.run(drive()) == ["ran", "ran"]
    assert wrapped == ["write_file"], "the guard skipped the tool it disclaimed"


def test_has_middleware_answers_per_tool():
    registry = HookRegistry()

    async def guard(handler=None, **_kw):
        return await handler()

    registry.register_middleware(HOOK_WRAP_TOOL_EXECUTE, guard,
                                 tools=["write_file"])
    assert registry.has_middleware(HOOK_WRAP_TOOL_EXECUTE, "write_file")
    assert not registry.has_middleware(HOOK_WRAP_TOOL_EXECUTE, "read_file")
    assert registry.has_middleware(HOOK_WRAP_TOOL_EXECUTE), "any at all"


def test_the_task_audit_hook_declares_its_tools_instead_of_testing_them():
    """D13's motivating case, converted: the set is readable from outside."""
    from silk.functions.hook_catalog import (
        _AUDIT_GUARDED,
        build_hooks,
    )

    entries = build_hooks(["task_audit"])[HOOK_WRAP_TOOL_EXECUTE]
    assert [e.tools for e in entries] == [_AUDIT_GUARDED]


# -- §22 q5: the binding a graph can express -------------------------------


def test_a_configured_binding_reaches_the_registry():
    """The answer to "does the Hooks node come back": no, a field does.

    D12 asked for a concrete scenario the two selectors cannot express.
    "Log the file tools only" is one, and it needs a config field on the
    hook -- a binding is a property of an entry -- not a third node to
    compose hooks in.
    """
    from silk.functions.hook_catalog import build_hooks

    entries = build_hooks(
        ["log_tool_calls"], {"log_tool_calls": {"bind_tools": "read_file, write_file"}},
    )[HOOK_BEFORE_TOOL_EXECUTE]
    assert [e.tools for e in entries] == [frozenset({"read_file", "write_file"})]


def test_no_binding_leaves_the_hook_exactly_as_it_was():
    from silk.functions.hook_catalog import build_hooks

    entries = build_hooks(["log_tool_calls"])[HOOK_BEFORE_TOOL_EXECUTE]
    assert all(not getattr(e, "bound", False) for e in entries), (
        "empty means every tool, which is what a hook meant before D13"
    )


def test_a_configured_binding_can_only_narrow_what_the_code_declared():
    """The I6 rule, applied to hooks: config makes a hook quieter, never louder."""
    from silk.functions.hook_catalog import (
        _AUDIT_GUARDED,
        _entry_of,
        _narrow,
    )

    entry = _entry_of(lambda **kw: None)
    entry = type(entry)(callback=entry.callback, tools=_AUDIT_GUARDED)
    narrowed = _narrow(entry, frozenset({sorted(_AUDIT_GUARDED)[0]}),
                       frozenset(), "task_audit")
    assert narrowed.tools == frozenset({sorted(_AUDIT_GUARDED)[0]})
    assert narrowed.tools < _AUDIT_GUARDED


def test_a_binding_that_shares_nothing_with_the_code_is_refused():
    """A hook that fires on nothing is the failure this field prevents."""
    from silk.functions.hook_catalog import _entry_of, _narrow

    entry = _entry_of(lambda **kw: None)
    entry = type(entry)(callback=entry.callback, tools=frozenset({"write_file"}))
    with pytest.raises(ValueError, match="fires on nothing"):
        _narrow(entry, frozenset({"read_file"}), frozenset(), "somehook")


def test_a_guard_has_no_binding_field_to_narrow():
    """§22 q5's limit: observers bind from the graph, gates do not.

    A guard a preset can narrow to nothing is not a guard (D77), so the
    approval and sign-off gates and the plan audit keep their binding in
    code where nothing configurable can reach it.
    """
    from silk.functions.hook_catalog import (
        HOOK_CATALOG, BoundHookConfig,
    )

    for name in ("signoff", "tool_approval", "task_audit"):
        model = HOOK_CATALOG[name].config_model
        assert model is None or not issubclass(model, BoundHookConfig), (
            f"'{name}' guards something; it must not be narrowable from a preset"
        )
    for name in ("log_tool_calls", "timing", "usage_meter", "redact_secrets"):
        assert issubclass(HOOK_CATALOG[name].config_model, BoundHookConfig)


# -- D14: the essential tier ----------------------------------------------


def test_an_essential_hook_refuses_to_be_unregistered():
    registry = HookRegistry()
    hook, _ = _recorder()
    registry.register(HOOK_BEFORE_TOOL_EXECUTE, hook, essential=True)

    with pytest.raises(EssentialHookError):
        registry.unregister(HOOK_BEFORE_TOOL_EXECUTE, hook)
    assert registry.callbacks(HOOK_BEFORE_TOOL_EXECUTE) == [hook]


def test_deactivating_a_role_cannot_strip_the_infrastructure_tier():
    registry = HookRegistry()

    @essential
    def floor(**_kw) -> None:
        pass

    registered = register_hook_map(registry, {HOOK_BEFORE_TOOL_EXECUTE: [floor]})
    with pytest.raises(EssentialHookError):
        unregister_hook_map(registry, registered)
    assert registry.callbacks(HOOK_BEFORE_TOOL_EXECUTE) == [floor]


def test_clearing_keeps_the_essential_tier_and_can_be_forced():
    registry = HookRegistry()
    keeper, _ = _recorder()
    passing, _ = _recorder()
    registry.register(HOOK_BEFORE_TOOL_EXECUTE, keeper, essential=True)
    registry.register(HOOK_BEFORE_TOOL_EXECUTE, passing)

    registry.clear()
    assert registry.callbacks(HOOK_BEFORE_TOOL_EXECUTE) == [keeper]

    registry.clear(keep_essential=False)
    assert registry.callbacks(HOOK_BEFORE_TOOL_EXECUTE) == []


def test_a_non_essential_hook_registered_twice_is_removed_once():
    """The entry handle is what tells two registrations of one callable apart."""
    registry = HookRegistry()
    hook, _ = _recorder()
    first = registry.register(HOOK_BEFORE_TOOL_EXECUTE, hook, tools=["a"])
    registry.register(HOOK_BEFORE_TOOL_EXECUTE, hook, tools=["b"])

    registry.unregister(HOOK_BEFORE_TOOL_EXECUTE, first)
    remaining = registry.entries(HOOK_BEFORE_TOOL_EXECUTE)
    assert [e.tools for e in remaining] == [frozenset({"b"})]


def test_essential_entries_lists_both_stores():
    registry = HookRegistry()

    async def guard(handler=None, **_kw):
        return await handler()

    hook, _ = _recorder()
    registry.register(HOOK_BEFORE_TOOL_EXECUTE, hook, essential=True)
    registry.register_middleware(HOOK_WRAP_TOOL_EXECUTE, guard, essential=True)
    registry.register(HOOK_AFTER_TOOL_EXECUTE, _recorder()[0])

    pairs = registry.essential_entries()
    assert {event for event, _ in pairs} == {
        HOOK_BEFORE_TOOL_EXECUTE, HOOK_WRAP_TOOL_EXECUTE,
    }


# -- I7: essential hooks survive derivation -------------------------------


def test_an_essential_hook_outside_the_recipe_survives_derivation():
    """The half the recipe cannot do: a hook installed on a live toolbox."""
    source, derived = ToolBox(), ToolBox()
    hook, seen = _recorder()
    source.hooks.register(HOOK_BEFORE_TOOL_EXECUTE, hook, essential=True)

    assert carry_essential_hooks(source, derived) == 1
    derived.hooks.emit(HOOK_BEFORE_TOOL_EXECUTE, tool_name="peek")
    assert seen == ["peek"]


def test_carrying_does_not_duplicate_what_the_recipe_replayed():
    source, derived = ToolBox(), ToolBox()
    hook, seen = _recorder()
    source.hooks.register(HOOK_BEFORE_TOOL_EXECUTE, hook, essential=True)
    derived.hooks.register(HOOK_BEFORE_TOOL_EXECUTE, hook, essential=True)

    assert carry_essential_hooks(source, derived) == 0
    derived.hooks.emit(HOOK_BEFORE_TOOL_EXECUTE, tool_name="peek")
    assert seen == ["peek"], "fired once, not twice"


def test_a_non_essential_hook_is_not_carried():
    source, derived = ToolBox(), ToolBox()
    source.hooks.register(HOOK_BEFORE_TOOL_EXECUTE, _recorder()[0])
    assert carry_essential_hooks(source, derived) == 0
    assert derived.hooks.callbacks(HOOK_BEFORE_TOOL_EXECUTE) == []
