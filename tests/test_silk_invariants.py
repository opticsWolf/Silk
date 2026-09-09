# -*- coding: utf-8 -*-
"""Executable invariant fixtures (spec D27, Phase 1 item 1; G4).

The catalog in ``silk_invariant_catalog.py`` is the data: one record per
invariant, one per violation class. This module is the executable half, plus
three meta-tests that keep the two from drifting:

* every catalog record has a check, and every check has a record;
* every invariant written in the architecture doc and in the spec appears in
  the catalog, with the statement re-read from the document rather than
  paraphrased here.

A record marked PENDING describes an invariant that is specified but not yet
implemented. Its check is written anyway and run ``xfail(strict=True)``, so
the suite fails the day the behaviour lands and the catalog is not updated.
A pending fixture that starts passing is drift too.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
from pydantic import BaseModel


from silk_invariant_catalog import (  # noqa: E402
    ALL_FIXTURES,
    ARCHITECTURE_DOC,
    INVARIANTS,
    PENDING,
    SPEC_DOC,
)
from silk.functions.hooks import (  # noqa: E402
    HOOK_AFTER_RUN,
    HOOK_TOOL_DENIED,
    HOOK_WRAP_TOOL_EXECUTE,
)
from silk.functions.role import (  # noqa: E402
    Role,
    RoleBinding,
    ToolSelector,
)
from silk.functions.approval import (  # noqa: E402
    LEVEL_HUMAN,
    attach_approval_gate,
    attach_signoff_gate,
)
from silk.functions.decision_seam import (  # noqa: E402
    DecisionSeam,
)
from silk.functions.prefix_guard import (  # noqa: E402
    KIND_HISTORY,
    KIND_SYSTEM,
    PrefixGuard,
)
from silk.functions.signoff import (  # noqa: E402
    CHANGE_TYPES,
    SIGNOFF_MODES,
    normalize_policy,
    preset_policy,
)
from silk.functions.task_store import (  # noqa: E402
    SqliteTaskStore,
    plan_changed_event,
)
from silk.functions.tool_box import ToolBox  # noqa: E402
from silk.functions.tools.file_sandbox import (  # noqa: E402
    FileToolSandbox,
)
from silk.functions.tools.task_tracker import (  # noqa: E402
    attach_task_tools,
)
from silk.functions.toolset_build import (  # noqa: E402
    sandbox_from_permissions,
)

# The AgentLoop fixtures reuse the scripted engine/toolbox the loop tests
# already have. Sharing them is deliberate: an invariant fixture that needs
# its own private notion of "an engine" is testing its own mock.
from test_silk_agent_loop import (  # noqa: E402
    TOOL_FENCE,
    FakeEngine,
    FakeToolbox,
    run_loop,
)

# ── the check registry ────────────────────────────────────────────────────

CHECKS: dict[tuple[str, str], Callable[[], None]] = {}


def check(invariant: str, case: str) -> Callable:
    """Bind a check to a catalog record."""
    def wrap(fn: Callable[[], None]) -> Callable[[], None]:
        key = (invariant, case)
        assert key not in CHECKS, f"duplicate check for {invariant}:{case}"
        CHECKS[key] = fn
        return fn
    return wrap


def _tmp() -> str:
    return tempfile.mkdtemp(prefix="silk-inv-")


# ── I1. one result per call, failures included ────────────────────────────


class _Args(BaseModel):
    n: int = 0


def _batch_box() -> ToolBox:
    box = ToolBox()

    @box.register(name="fine", description="works", args_model=_Args,
                  tags=("read",))
    def _fine(db, session, n: int = 0) -> str:
        return f"ok:{n}"

    @box.register(name="explodes", description="raises", args_model=_Args,
                  tags=("read",))
    def _explodes(db, session, n: int = 0) -> str:
        raise RuntimeError("intentional")

    @box.register(name="dawdles", description="outruns its timeout",
                  args_model=_Args, tags=("read",), timeout=0.05)
    def _dawdles(db, session, n: int = 0) -> str:
        time.sleep(0.6)
        return "too late"

    # The description carries a distinctive word so the discovery fixture
    # (I8) can search for something that unambiguously hits this tool.
    @box.register(name="forbidden", description="quarantined widget wrangler",
                  args_model=_Args, tags=("write",))
    def _forbidden(db, session, n: int = 0) -> str:
        return "should never run"

    return box


def _call(name: str, **args) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"call-{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _run(box: ToolBox, *calls) -> list[dict]:
    return asyncio.run(box.execute_tool_calls_async(list(calls)))


def _assert_one_result_per_call(calls, results) -> None:
    """The invariant itself: same count, same ids, same shape, no raise.

    Order is deliberately not asserted -- the batch runs parallel tools
    before sequential ones, and the loop matches results by
    ``tool_call_id``, which is what the invariant is about.
    """
    assert len(results) == len(calls)
    assert {r["tool_call_id"] for r in results} == {c.id for c in calls}
    for result in results:
        assert set(result) >= {"tool_call_id", "name", "content"}
        assert isinstance(result["content"], str)


@check("I1", "unknown_tool")
def _i1_unknown() -> None:
    calls = [_call("fine"), _call("no_such_tool")]
    results = _run(_batch_box(), *calls)
    _assert_one_result_per_call(calls, results)
    body = next(r for r in results if r["name"] == "no_such_tool")["content"]
    assert "not registered" in body


@check("I1", "role_denied")
def _i1_denied() -> None:
    box = _batch_box()
    role = Role(id="reader", selector=ToolSelector(allow_tags=frozenset({"read"})))
    binding = RoleBinding.activate(role, box)
    try:
        calls = [_call("fine"), _call("forbidden")]
        results = _run(box, *calls)
    finally:
        binding.deactivate()
    _assert_one_result_per_call(calls, results)
    body = json.loads(next(r for r in results if r["name"] == "forbidden")["content"])
    assert body.get("error_type") == "role_denied"


@check("I1", "validation_error")
def _i1_validation() -> None:
    calls = [_call("fine", n="not-a-number")]
    results = _run(_batch_box(), *calls)
    _assert_one_result_per_call(calls, results)
    assert "Validation error" in results[0]["content"]


@check("I1", "timeout")
def _i1_timeout() -> None:
    calls = [_call("dawdles"), _call("fine")]
    results = _run(_batch_box(), *calls)
    _assert_one_result_per_call(calls, results)
    body = next(r for r in results if r["name"] == "dawdles")["content"]
    assert "timed out" in body.lower()


@check("I1", "exception")
def _i1_exception() -> None:
    calls = [_call("explodes"), _call("fine")]
    results = _run(_batch_box(), *calls)
    _assert_one_result_per_call(calls, results)
    body = next(r for r in results if r["name"] == "explodes")["content"]
    assert "intentional" in body or "error" in body.lower()


# ── I2. HOOK_AFTER_RUN exactly once on every exit path ────────────────────


class _CountingHooks:
    """Minimal hook registry: counts emissions, swallows nothing."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def emit(self, event: str, **_kw) -> None:
        self.counts[event] = self.counts.get(event, 0) + 1

    def emit_middleware(self, event, handler, **_kw):   # pragma: no cover
        return handler()

    def register(self, *_a, **_kw) -> None:             # pragma: no cover
        pass


def _loop_with_hooks(engine, toolbox=None, **kwargs):
    """A loop whose hook emissions are counted.

    The loop finds its registry on the toolbox (``_emit``: no toolbox means
    pure chat and no hooks), so counting requires one even for a run with no
    tool calls in it.
    """
    from silk.functions.agent_loop import AgentLoop

    hooks = _CountingHooks()
    toolbox = toolbox if toolbox is not None else FakeToolbox({})
    toolbox.hooks = hooks
    loop = AgentLoop(engine, toolbox, **kwargs)
    return loop, hooks


@check("I2", "normal_completion")
def _i2_normal() -> None:
    loop, hooks = _loop_with_hooks(FakeEngine(["done"]))
    list(loop.run("go", {"max_tokens": 8}))
    assert hooks.counts.get(HOOK_AFTER_RUN) == 1


@check("I2", "usage_limit")
def _i2_usage_limit() -> None:
    from silk.functions.usage_limits import UsageLimits

    limits = UsageLimits(request_limit=1)
    engine = FakeEngine([TOOL_FENCE, "never reached"], usage_limits=limits)
    loop, hooks = _loop_with_hooks(engine, FakeToolbox({"echo": "hi"}))
    list(loop.run("go", {"max_tokens": 8}))
    assert hooks.counts.get(HOOK_AFTER_RUN) == 1


@check("I2", "stream_error")
def _i2_stream_error() -> None:
    class Exploding(FakeEngine):
        def stream_response(self, gen_params):
            raise RuntimeError("backend fell over")
            yield  # pragma: no cover - makes it a generator

    loop, hooks = _loop_with_hooks(Exploding(["unused"]))
    list(loop.run("go", {"max_tokens": 8}))
    assert hooks.counts.get(HOOK_AFTER_RUN) == 1


@check("I2", "early_close")
def _i2_early_close() -> None:
    loop, hooks = _loop_with_hooks(FakeEngine(["a b c d e f g"]))
    events = loop.run("go", {"max_tokens": 8})
    next(events)          # consume only the first event
    events.close()        # the caller walks away mid-run
    assert hooks.counts.get(HOOK_AFTER_RUN) == 1


# ── I3. the loop never executes tools ─────────────────────────────────────


class _StrictEngine(FakeEngine):
    """An engine that refuses anything outside the AgentEngine protocol.

    If the loop ever tried to run a tool through the engine, it would have
    to reach for an attribute that is not part of one request's worth of
    work -- and this raises instead.
    """

    #: The AgentEngine protocol, plus the optional extensions the loop is
    #: allowed to probe: a context window it may not have (G14c), and the
    #: native tool-call surface, which is about how *one request* advertises
    #: tools -- not about running them.
    _PROTOCOL = {
        "stream_response", "append_message", "count_prompt_tokens",
        "request_stop", "stop_requested", "usage_limits",
        "reflection_config", "history", "last_stats", "context_length",
        "supports_native_tools", "set_tool_schemas", "take_tool_calls",
        "pending_tool_calls", "_responses", "_stopped",
    }

    def __getattr__(self, name: str):
        if name.startswith("__") or name in self._PROTOCOL:
            raise AttributeError(name)
        raise AssertionError(
            f"the loop reached for engine.{name}: tool work does not belong "
            f"to the engine (invariant I3)"
        )


@check("I3", "engine_is_never_asked_to_run_a_tool")
def _i3_engine_untouched() -> None:
    engine = _StrictEngine([TOOL_FENCE, "final"])
    events = run_loop(engine, FakeToolbox({"echo": "echoed"}))
    assert events, "the run must complete without touching the engine for tools"


@check("I3", "every_call_goes_through_the_toolbox")
def _i3_via_toolbox() -> None:
    toolbox = FakeToolbox({"echo": "echoed"})
    run_loop(FakeEngine([TOOL_FENCE, "final"]), toolbox)
    assert toolbox.executed == ["echo"], "the batch is the toolbox's, once"


# ── I4. a role-denied tool is invisible and refused ────────────────────────


@check("I4", "not_advertised")
def _i4_invisible() -> None:
    box = _batch_box()
    role = Role(id="reader", selector=ToolSelector(allow_tags=frozenset({"read"})))
    binding = RoleBinding.activate(role, box)
    try:
        names = {
            schema.get("function", schema).get("name")
            for schema in box.get_tool_schemas()
        }
    finally:
        binding.deactivate()
    assert "fine" in names
    assert "forbidden" not in names


@check("I4", "refused_at_dispatch")
def _i4_refused() -> None:
    box = _batch_box()
    ran: list[str] = []
    box.hooks.register(HOOK_TOOL_DENIED, lambda **kw: ran.append(kw["tool_name"]))
    role = Role(id="reader", selector=ToolSelector(allow_tags=frozenset({"read"})))
    binding = RoleBinding.activate(role, box)
    try:
        results = _run(box, _call("forbidden"))
    finally:
        binding.deactivate()
    body = json.loads(results[0]["content"])
    assert body.get("error_type") == "role_denied"
    assert ran == ["forbidden"], "the denial must be announced, not just returned"


# ── I5. store reads never mutate ──────────────────────────────────────────


def _seeded_store() -> SqliteTaskStore:
    store = SqliteTaskStore(_tmp())
    store.start(goal="g", acceptance=["a"],
                tasks=[{"title": "one"}, {"title": "two"}], actor="ag")
    return store


@check("I5", "read_does_not_bump_the_revision")
def _i5_no_bump() -> None:
    store = _seeded_store()
    before = store.load().revision
    for _ in range(3):
        store.load()
    assert store.load().revision == before


@check("I5", "unchanged_plan_does_not_restream")
def _i5_no_restream() -> None:
    store = _seeded_store()
    first = plan_changed_event(store, None)
    assert first is not None, "the first read of a plan is a change"
    assert plan_changed_event(store, first["revision"]) is None
    store.complete_task(task_id="t1", actor="ag", rationale="done")
    assert plan_changed_event(store, first["revision"]) is not None


# ── I6. file access narrows monotonically ─────────────────────────────────


def _ceiling() -> tuple[FileToolSandbox, str, str]:
    root = _tmp()
    outside = _tmp()
    return FileToolSandbox(root_dir=root, allowed_paths=[root]), root, outside


@check("I6", "entry_outside_the_ceiling_is_dropped")
def _i6_entry_dropped() -> None:
    base, root, outside = _ceiling()
    derived = sandbox_from_permissions(
        {"root": root, "entries": [
            {"path": root, "mode": "read"},
            {"path": outside, "mode": "read_write"},
        ]},
        base,
    )
    assert derived.is_allowed(Path(root) / "a.txt")
    assert not derived.is_allowed(Path(outside) / "a.txt"), (
        "a permission naming a path outside the ceiling must grant nothing"
    )


@check("I6", "root_outside_the_ceiling_is_ignored")
def _i6_root_ignored() -> None:
    base, root, outside = _ceiling()
    derived = sandbox_from_permissions(
        {"roots": [outside], "entries": [{"path": root, "mode": "read"}]},
        base,
    )
    assert str(derived.root_dir).startswith(str(Path(root).resolve()))


@check("I6", "confinement_cannot_be_switched_off")
def _i6_no_escape() -> None:
    base, root, outside = _ceiling()
    # Whatever the permission structure claims, a derived sandbox comes back
    # confined: the escape hatch is not reachable from downstream data.
    for permissions in (
        {"root": outside, "enabled": False, "entries": []},
        {"root": root, "enabled": False,
         "entries": [{"path": root, "mode": "read_write"}]},
    ):
        derived = sandbox_from_permissions(permissions, base)
        assert derived.enabled is True
        assert not derived.is_allowed(Path(outside) / "secret")


@check("I6", "a_downstream_grant_cannot_widen_an_upstream_one")
def _i6_chain_narrows() -> None:
    from silk.functions.file_grants import FileGrants, resolve_grants

    root = _tmp()
    inner = str(Path(root) / "src")
    outside = _tmp()

    upstream = FileGrants(root=root, entries=[{"path": inner, "mode": "read"}])
    # The downstream port asks for more of everything: write on the subtree
    # it was given read on, plus a path it was never given at all.
    asked = FileGrants(root=root, entries=[
        {"path": inner, "mode": "read_write"},
        {"path": outside, "mode": "read_write"},
    ])
    combined = resolve_grants(asked, upstream)
    assert combined.mode_for(inner) == "read", (
        "asking for write on a read grant must stay read"
    )
    assert combined.mode_for(outside) == "blocked", (
        "asking for a path the upstream never covered must grant nothing"
    )


@check("I6", "a_run_scoped_restriction_cannot_widen_the_sandbox")
def _i6_restrict_narrows() -> None:
    root = _tmp()
    inner = Path(root) / "src"
    inner.mkdir(parents=True, exist_ok=True)
    outside = _tmp()
    sandbox = FileToolSandbox(root_dir=root, allowed_paths=[root],
                              path_modes={str(inner): "read"})

    with sandbox.restrict({str(inner): "read_write", outside: "read_write"}):
        assert sandbox.is_allowed(inner / "a.txt")
        assert not sandbox.is_writable(inner / "a.txt"), (
            "a run-scoped grant cannot turn a read policy into a write one"
        )
        assert not sandbox.is_allowed(Path(outside) / "a.txt"), (
            "nor add a path the sandbox never covered"
        )
    # And the live object the next run shares is handed back untouched.
    assert sandbox.is_allowed(inner / "a.txt")
    assert not sandbox.is_writable(inner / "a.txt")


# ── I7. essential hooks survive derivation ──────────────────────────


@check("I7", "essential_hook_survives_a_toolset")
def _i7_survives() -> None:
    from silk.functions.hooks import (
        HOOK_BEFORE_TOOL_EXECUTE, HookRegistry,
    )
    from silk.functions.tool_box import ToolBox
    from silk.functions.toolset_build import carry_essential_hooks

    fired: list[str] = []

    def hook(tool_name: str = "", **_kw) -> None:
        fired.append(tool_name)

    source, derived = ToolBox(), ToolBox()
    assert isinstance(source.hooks, HookRegistry)
    source.hooks.register(HOOK_BEFORE_TOOL_EXECUTE, hook, essential=True)

    carry_essential_hooks(source, derived)
    derived.hooks.emit(HOOK_BEFORE_TOOL_EXECUTE, tool_name="anything")
    assert fired == ["anything"], "the essential hook did not survive derivation"


@check("I7", "essential_hook_cannot_be_dropped")
def _i7_undroppable() -> None:
    from silk.functions.hooks import (
        HOOK_BEFORE_TOOL_EXECUTE, EssentialHookError, HookRegistry,
    )

    registry = HookRegistry()

    def hook(**_kw):
        return None

    registry.register(HOOK_BEFORE_TOOL_EXECUTE, hook, essential=True)
    try:
        registry.unregister(HOOK_BEFORE_TOOL_EXECUTE, hook)
    except EssentialHookError:
        pass
    # Refusing loudly and refusing silently would both satisfy the
    # invariant; surviving the attempt is what it actually says.
    assert hook in registry.callbacks(HOOK_BEFORE_TOOL_EXECUTE)
    registry.clear()
    assert hook in registry.callbacks(HOOK_BEFORE_TOOL_EXECUTE), (
        "a wholesale clear dropped it, which is the same loss by another route"
    )


# ── I8. discovery obeys the role gate (PENDING) ───────────────────────────


@check("I8", "search_hides_a_denied_tool")
def _i8_search_gated() -> None:
    from silk.functions.capabilities import Capability

    # Discovery runs over the tools a capability contributes -- that is the
    # surface the model searches, and the one the role gate does not reach.
    box = _batch_box()
    box.register_capability(Capability(
        id="quarantine",
        description="a capability the reader role cannot use",
        tools=[{"type": "function", "function": {
            "name": "wrangle_widgets",
            "description": "quarantined widget wrangler",
            "parameters": {"type": "object", "properties": {}},
        }}],
    ))
    role = Role(id="reader", selector=ToolSelector(allow_tags=frozenset({"read"})))
    binding = RoleBinding.activate(role, box)
    try:
        assert not box.role_permits("wrangle_widgets"), (
            "the fixture is only meaningful if dispatch would refuse this tool"
        )
        found = {hit.get("function", hit).get("name")
                 for hit in box.tool_search.search("wrangler")}
    finally:
        binding.deactivate()
    assert "wrangle_widgets" not in found, (
        "discovery advertised a tool dispatch will refuse"
    )


@check("I8", "search_hides_an_unusable_capability")
def _i8_capability_hidden() -> None:
    from silk.functions.capabilities import Capability
    from silk.functions.role import Role, RoleBinding, ToolSelector
    from silk.functions.tool_box import ToolBox

    box = ToolBox()
    box.register_capability(Capability(
        id="wrangling",
        description="quarantined widget wrangling",
        defer_loading=True,
        tools=[{"type": "function", "function": {
            "name": "wrangle_widgets",
            "description": "quarantined widget wrangler",
            "parameters": {"type": "object", "properties": {}},
        }}],
    ))
    role = Role(id="reader", selector=ToolSelector(allow_tags=frozenset({"read"})))
    binding = RoleBinding.activate(role, box)
    try:
        found = {cap.id for cap in box.tool_search.search_capabilities("wrangling")}
    finally:
        binding.deactivate()
    assert "wrangling" not in found, (
        "discovery offered a capability whose every tool dispatch will refuse"
    )


@check("I8", "auto_load_does_not_widen_the_role")
def _i8_autoload_gated() -> None:
    from silk.functions.capabilities import Capability
    from silk.functions.role import Role, RoleBinding, ToolSelector
    from silk.functions.tool_box import ToolBox

    box = ToolBox()
    box.register_capability(Capability(
        id="wrangling",
        description="quarantined widget wrangling",
        defer_loading=True,
        tools=[{"type": "function", "function": {
            "name": "wrangle_widgets",
            "description": "quarantined widget wrangler",
            "parameters": {"type": "object", "properties": {}},
        }}],
    ))
    role = Role(id="reader", selector=ToolSelector(allow_tags=frozenset({"read"})))
    binding = RoleBinding.activate(role, box)
    try:
        results = _run(box, _call("wrangle_widgets"))
    finally:
        binding.deactivate()

    assert len(results) == 1
    assert "role_denied" in results[0]["content"], (
        "auto-load must not hand the model a tool the role forbids"
    )


# ── I9. compaction cuts on whole-round boundaries ────────────────────────


def _compacted(history, **kw):
    """Run a real compaction over *history* and return what survived."""
    from silk.functions.compaction import Compactor

    engine = _CompactionEngine(history)
    Compactor(summarizer=lambda _text: "summary", **kw).maybe_compact(
        engine, force=True)
    return engine.history


class _CompactionEngine:
    """The rewrite operation under the rule, over a plain history list."""

    def __init__(self, history):
        self.history = history

    def replace_history_prefix(self, count, summary, **_kw):
        assert self.history[count].get("role") != "tool", (
            "the compactor planned a cut the engine has to refuse (I9)"
        )
        self.history[:count] = [{"role": "user", "content": summary}]
        return count

    def count_prompt_tokens(self):
        return sum(len(str(m.get("content", ""))) for m in self.history) // 4


@check("I9", "an_assistant_turn_and_its_results_move_together")
def _i9_pairs_move() -> None:
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2"}]},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
    ]
    kept = _compacted(history, keep_recent=2, min_dropped=2)
    ids = [m.get("tool_call_id") for m in kept if m["role"] == "tool"]
    called = [c["id"] for m in kept if m["role"] == "assistant"
              for c in (m.get("tool_calls") or ())]
    assert set(ids) == set(called), (
        "a call and its result must be dropped together or kept together"
    )


@check("I9", "a_tool_result_is_never_orphaned")
def _i9_no_orphans() -> None:
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2"}]},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        {"role": "assistant", "content": "done"},
    ]
    for keep in range(0, len(history) + 1):
        kept = _compacted(list(history), keep_recent=keep, min_dropped=1)
        for index, message in enumerate(kept):
            if message["role"] != "tool":
                continue
            assert any(m["role"] == "assistant" for m in kept[:index]), (
                "a tool result kept without its assistant turn corrupts the "
                f"next request (keep_recent={keep})"
            )


# ── D31 survivors: the sign-off gate's resolution rules ───────────────────


@check("D31", "preset_expands_to_a_full_policy")
def _d31_presets() -> None:
    for mode in SIGNOFF_MODES:
        policy = preset_policy(mode)
        assert set(policy) == set(CHANGE_TYPES), mode
        assert all(level in ("agent", "human") for level in policy.values()), mode
    assert preset_policy("final")["complete_final"] == "human"
    assert preset_policy("final")["complete"] == "agent"


@check("D31", "unknown_preset_falls_back_to_auto")
def _d31_unknown_preset() -> None:
    assert preset_policy("no-such-mode") == preset_policy("auto")


@check("D31", "an_explicit_policy_beats_the_preset")
def _d31_normalize() -> None:
    policy = normalize_policy({"goal": "human", "bogus": "human",
                               "complete": "sometimes"})
    assert policy["goal"] == "human"
    assert "bogus" not in policy
    assert policy["complete"] == "agent", "an unknown level is not a gate"


def _gated_box(tasks: list[dict]) -> tuple[ToolBox, str]:
    root = _tmp()
    box = ToolBox(None, {"agent_id": "ag"})
    sandbox = SimpleNamespace(root_dir=root)
    attach_task_tools(box, sandbox)
    # 'final' gates only the plan-closing completion, so the two fixtures
    # below differ in exactly the resolution under test.
    attach_signoff_gate(box, sandbox, mode="final")
    _run(box, _call("plan_start", goal="g", tasks=tasks))
    return box, root


@check("D31", "complete_resolves_to_complete_final")
def _d31_final() -> None:
    box, root = _gated_box([{"title": "only"}])
    body = json.loads(_run(box, _call("task_complete", id="t1",
                                      rationale="done"))[0]["content"])
    assert body.get("approval_required") is True, (
        "completing the last open task is a plan-closing completion"
    )
    assert body.get("change_type") == "complete_final"
    task = next(t for t in SqliteTaskStore(root).load().tasks if t.id == "t1")
    assert task.status != "done", "a refused completion must not have applied"


@check("D31", "complete_stays_complete_with_work_left")
def _d31_not_final() -> None:
    box, root = _gated_box([{"title": "one"}, {"title": "two"}])
    body = json.loads(_run(box, _call("task_complete", id="t1",
                                      rationale="done"))[0]["content"])
    assert not body.get("approval_required"), (
        "an ordinary completion is not gated by the 'final' preset"
    )
    task = next(t for t in SqliteTaskStore(root).load().tasks if t.id == "t1")
    assert task.status == "done"


# I10. guard middleware is monotonic
#
# Two ways to break it and one corollary. The first two are about *order*
# -- something ahead of the gate, or something that arrives later and
# expects to wrap it -- and the third is the half the spec says actually
# bites: a guard that denies must not hand back a result shaped like
# success.


def _guarded_box(ask=None) -> tuple[ToolBox, list, object]:
    """A box whose one gated tool is a high-risk write, plus a run log."""
    box = ToolBox(None, {"agent_id": "ag"})
    ran: list[str] = []

    @box.register("write_file", "writes a file", risk="high")
    def _write(_pool, _session, **_kw):
        ran.append("write_file")
        return "wrote"

    seam = DecisionSeam(ask, timeout_s=2.0) if ask is not None else None
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam)
    return box, ran, seam


@check("I10", "the_gate_outranks_an_earlier_registration")
def _i10_earlier() -> None:
    box = ToolBox(None, {"agent_id": "ag"})
    ran: list[str] = []

    @box.register("write_file", "writes a file", risk="high")
    def _write(_pool, _session, **_kw):
        ran.append("write_file")
        return "wrote"

    async def bypass(handler=None, **_kw):
        return "hijacked"           # a success, without delegating

    box.hooks.register_middleware(HOOK_WRAP_TOOL_EXECUTE, bypass)
    entry = attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN})

    chain = box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE)
    assert chain[0] is entry, "the gate must be outermost, not merely present"
    body = json.loads(_run(box, _call("write_file"))[0]["content"])
    assert body.get("approval_required") is True and ran == [], (
        "a middleware registered first would otherwise answer a call the "
        "gate never sees"
    )


@check("I10", "a_later_registration_does_not_wrap_the_gate")
def _i10_later() -> None:
    box, _ran, _seam = _guarded_box()
    entry = box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE)[0]

    async def latecomer(handler=None, **_kw):
        return await handler()

    box.hooks.register_middleware(HOOK_WRAP_TOOL_EXECUTE, latecomer)
    assert box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE)[0] is entry


@check("I10", "a_denial_never_fabricates_a_result")
def _i10_no_fabrication() -> None:
    holder: list = []

    def deny(request):
        holder[0].deny(request.decision_id, reason="no")

    box, ran, seam = _guarded_box(ask=deny)
    holder.append(seam)

    content = _run(box, _call("write_file"))[0]["content"]
    body = json.loads(content)
    assert body["applied"] is False and body["approval_required"] is True
    assert body["error"] is None, "a policy refusal is not an error to retry"
    assert ran == [], "the denied call must not have executed"


# I11. the model-visible prefix grows only at the tail
#
# The rule is invisible at the call site: nothing breaks if the system
# prompt starts rendering the time of day. What breaks is the KV cache,
# silently, in proportion to how far back the change happened (D41).


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


@check("I11", "appending_is_not_a_break")
def _i11_append() -> None:
    guard = PrefixGuard()
    history = [_msg("user", "do the thing")]
    assert guard.observe(history, system_prompt="S") is None
    for reply in ("a", "b", "c"):
        history = [*history, _msg("assistant", reply)]
        assert guard.observe(history, system_prompt="S") is None
    assert guard.clean, guard.report()


@check("I11", "a_volatile_system_prompt_is_caught")
def _i11_system() -> None:
    guard = PrefixGuard()
    history = [_msg("user", "hi")]
    guard.observe(history, system_prompt="You are an agent. It is 10:04.")
    broken = guard.observe(history, system_prompt="You are an agent. It is 10:05.")
    assert broken is not None and broken.kind == KIND_SYSTEM


@check("I11", "a_rewritten_message_is_caught")
def _i11_history() -> None:
    guard = PrefixGuard()
    guard.observe([_msg("user", "hi"), _msg("assistant", "hello")],
                  system_prompt="S")
    broken = guard.observe([_msg("user", "hi"), _msg("assistant", "HELLO")],
                           system_prompt="S")
    assert broken is not None and broken.kind == KIND_HISTORY
    assert broken.position == 1, "the guard names where it stopped agreeing"


@check("I11", "compaction_is_the_one_forgiven_break")
def _i11_compaction() -> None:
    guard = PrefixGuard()
    guard.observe([_msg("user", "a"), _msg("assistant", "b")],
                  system_prompt="S")
    guard.note_compaction()
    assert guard.observe([_msg("assistant", "summary")],
                         system_prompt="S") is None
    assert guard.clean
    # ... and exactly once: a run that compacts must not go quiet after it.
    broken = guard.observe([_msg("assistant", "rewritten")], system_prompt="S")
    assert broken is not None and not guard.clean


# ── I12. a decision surface is a node only at a turn boundary ─────────────


@check("I12", "a_counting_surface_cannot_answer")
def _i12_counting_only() -> None:
    """The hub sees every blocked agent and can reach none of them.

    D58 lets the hub count mid-run requests; D59 reserves answering to the
    node that asked, or a dock mirror of its widget. The line between the
    two is that the counter never holds anything resolvable -- an id is
    not a handle.
    """
    from silk.functions.decision_seam import DecisionSeam
    from silk.functions.stream_events import EventType
    from silk.functions.task_board import PendingDecisions

    pending = PendingDecisions()
    pending.record({"type": EventType.DECISION_REQUEST.value,
                    "decision_id": "d1", "run_id": "run-a"})
    assert pending.count == 1, "the fixture needs one outstanding request"
    assert pending.waiting() == ["d1"], "ids, which is all a count needs"

    resolvers = [name for name in dir(pending)
                 if any(word in name.lower()
                        for word in ("resolve", "approve", "answer", "deny"))]
    assert not resolvers, (
        f"the counter exposes {resolvers}; only the asking node's surface "
        f"may answer a live request (D59)"
    )
    assert hasattr(DecisionSeam, "resolve"), (
        "and the seam is where answering lives -- if this moved, the "
        "check above stopped meaning anything"
    )


@check("I12", "nothing_is_parked_for_a_later_decision")
def _i12_nothing_parked() -> None:
    """No state survives the turn, so no node can sign off after it.

    This is what makes the read-only hub the *complete* answer rather than
    a first half: there is no held change for an Approve button to apply
    (D31-D33). A resurrected parked state would silently reopen the
    question I12 settles.
    """
    from pathlib import Path

    from silk.functions import task_store

    assert not any("signoff" in status or "await" in status
                   for status in task_store.STATUSES), (
        f"a parked status is back in {sorted(task_store.STATUSES)}"
    )
    schema = Path(task_store.__file__).read_text(encoding="utf-8")
    for parked in ("awaiting_signoff", "signoff_actor", "pending_goal"):
        assert parked not in schema, (
            f"{parked} is back in the store; a change that outlives its "
            f"turn needs a decision surface I12 does not permit"
        )


# ── the fixtures ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=str)
def test_invariant_fixture(fixture, request):
    if fixture.status == PENDING:
        request.node.add_marker(
            pytest.mark.xfail(strict=True, reason=fixture.pending)
        )
    CHECKS[fixture.key]()


# ── the anti-drift meta-tests ─────────────────────────────────────────────


def test_every_catalog_record_has_a_check():
    missing = [str(fx) for fx in ALL_FIXTURES if fx.key not in CHECKS]
    assert not missing, f"catalog records with no executable check: {missing}"


def test_every_check_has_a_catalog_record():
    known = {fx.key for fx in ALL_FIXTURES}
    extra = [f"{inv}:{case}" for (inv, case) in CHECKS if (inv, case) not in known]
    assert not extra, f"checks that no catalog record describes: {extra}"


def test_pending_records_say_why():
    for fx in ALL_FIXTURES:
        if fx.status == PENDING:
            assert fx.pending.strip(), f"{fx} is pending with no reason"


def _read(relative: str) -> str:
    root = Path(__file__).resolve().parent.parent
    return (root / relative).read_text(encoding="utf-8")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def test_the_catalog_quotes_the_documents_verbatim():
    """The one rule that keeps the document and the suite together (D27)."""
    architecture = _normalize(_read(ARCHITECTURE_DOC))
    spec = _normalize(_read(SPEC_DOC))

    numbered = {
        f"I{n}": _normalize(title)
        for n, title in re.findall(r"(\d+)\.\s+\*\*(.+?)\*\*", architecture)
    }
    lettered = {
        ident: _normalize(title)
        for ident, title in re.findall(r"\*\*(I\d+)\.\s+(.+?)\*\*", spec)
    }

    for invariant in INVARIANTS:
        documented = numbered.get(invariant.id) or lettered.get(invariant.id)
        assert documented, (
            f"{invariant.id} is in the catalog but not in {invariant.source}"
        )
        assert _normalize(invariant.title) == documented, (
            f"{invariant.id} drifted:\n  catalog: {invariant.title}\n"
            f"  document: {documented}"
        )


def test_the_catalog_covers_every_documented_invariant():
    """Every documented invariant is in the catalog."""
    covered = {inv.id for inv in INVARIANTS}
    assert covered == {f"I{n}" for n in range(1, 13)}, (
        "Phase 1 item 1 covered the five existing invariants plus I6-I9; I10 "
        "joined them with the approval gate and I11 with the prefix guard "
        "(Phase 2 items 3 and 5), each being the code that gave its rule "
        "something to be true of. I12 was review-only until the Task Hub "
        "(Phase 3 item 5) became the first surface that had to obey it."
    )
