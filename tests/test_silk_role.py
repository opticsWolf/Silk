# -*- coding: utf-8 -*-
"""Silk Role model tests: selector semantics, hard dispatch enforcement,
RoleBinding lifecycle (hooks + capabilities), and serialisation.

Qt-free — exercises silk.functions only.
"""

from __future__ import annotations

import asyncio
import json
import sys


from pydantic import BaseModel

from silk.functions.capabilities import BaseCapability, CapabilityOrdering
from silk.functions.hooks import (
    HOOK_BEFORE_TOOL_EXECUTE,
    HOOK_TOOL_DENIED,
)
from silk.functions.reflection import is_retryable_tool_error
from silk.functions.role import (
    ALLOW_ALL,
    DEFAULT_ROLE,
    Role,
    RoleBinding,
    ToolSelector,
)
from silk.functions.tool_box import ToolBox
from silk.functions.tool_calling import ToolCall, _Function


# ── helpers ─────────────────────────────────────────────────────────────


class _EchoArgs(BaseModel):
    text: str = ""


def make_toolbox() -> tuple[ToolBox, dict]:
    """ToolBox with three tools of different tags/risk; records executions."""
    called: dict[str, int] = {}
    box = ToolBox()

    @box.register(name="echo_read", description="reads", args_model=_EchoArgs,
                  tags=("read",), category="file", risk="low")
    def _echo_read(db, session, text: str = "") -> str:
        called["echo_read"] = called.get("echo_read", 0) + 1
        return f"read:{text}"

    @box.register(name="echo_write", description="writes", args_model=_EchoArgs,
                  tags=("write",), category="file", risk="medium")
    def _echo_write(db, session, text: str = "") -> str:
        called["echo_write"] = called.get("echo_write", 0) + 1
        return f"write:{text}"

    @box.register(name="danger", description="dangerous", args_model=_EchoArgs,
                  tags=("write",), category="system", risk="high")
    def _danger(db, session, text: str = "") -> str:
        called["danger"] = called.get("danger", 0) + 1
        return "boom"

    return box, called


def call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"call_{name}", function=_Function(name, json.dumps(args)))


def run_calls(box: ToolBox, *calls: ToolCall) -> list[dict]:
    return asyncio.run(box.execute_tool_calls_async(list(calls)))


# ── ToolSelector semantics ───────────────────────────────────────────────


def test_empty_selector_denies_everything():
    sel = ToolSelector()
    assert not sel.permits("anything", None)


def test_allow_all_and_deny_precedence():
    sel = ToolSelector(allow_all=True, deny_names=frozenset({"danger"}))
    assert sel.permits("echo_read", None)
    assert not sel.permits("danger", {"tags": ()})


def test_name_tag_category_matching():
    meta_read = {"tags": frozenset({"read"}), "category": "file", "risk": "low"}
    meta_sys = {"tags": frozenset({"exotic"}), "category": "system", "risk": "low"}
    by_name = ToolSelector(allow_names=frozenset({"x"}))
    assert by_name.permits("x", None)
    assert not by_name.permits("y", meta_read)
    by_tag = ToolSelector(allow_tags=frozenset({"read"}))
    assert by_tag.permits("any", meta_read)
    assert not by_tag.permits("any", meta_sys)
    by_cat = ToolSelector(allow_categories=frozenset({"system"}))
    assert by_cat.permits("any", meta_sys)
    assert not by_cat.permits("any", meta_read)


def test_risk_ceiling():
    sel = ToolSelector(allow_all=True, max_risk="medium")
    assert sel.permits("a", {"risk": "low"})
    assert sel.permits("a", {"risk": "medium"})
    assert not sel.permits("a", {"risk": "high"})
    # Untagged tools count as low risk.
    assert sel.permits("a", None)


def test_selector_round_trip():
    sel = ToolSelector(
        allow_names=frozenset({"a"}), allow_tags=frozenset({"t"}),
        max_risk="medium", deny_names=frozenset({"d"}),
    )
    assert ToolSelector.from_dict(sel.to_dict()) == sel


def test_role_round_trip():
    role = Role(id="r", name="R", instructions="be careful",
                selector=ToolSelector(allow_tags=frozenset({"read"})),
                model_settings={"temperature": 0.2}, max_rounds=4)
    restored = Role.from_dict(role.to_dict())
    assert restored.id == role.id
    assert restored.selector == role.selector
    assert restored.model_settings == role.model_settings
    assert restored.max_rounds == 4


# ── hard enforcement at dispatch ────────────────────────────────────────


def test_denied_tool_never_executes_and_returns_role_denied():
    box, called = make_toolbox()
    denials: list[str] = []
    box.hooks.register(HOOK_TOOL_DENIED, lambda **kw: denials.append(kw["tool_name"]))

    role = Role(id="reader", selector=ToolSelector(allow_tags=frozenset({"read"})))
    binding = RoleBinding.activate(role, box)
    try:
        results = run_calls(box, call("echo_read", text="hi"), call("echo_write", text="no"))
    finally:
        binding.deactivate()

    by_name = {r["name"]: r for r in results}
    assert "read:hi" in by_name["echo_read"]["content"]
    assert called == {"echo_read": 1}, "denied executable must never run"

    payload = json.loads(by_name["echo_write"]["content"])
    assert payload["error_type"] == "role_denied"
    assert denials == ["echo_write"]
    # Reflection must not burn retries on a permanent denial.
    assert not is_retryable_tool_error(by_name["echo_write"]["content"])


def test_schemas_and_prompt_match_enforcement():
    box, _ = make_toolbox()
    role = Role(id="reader", selector=ToolSelector(allow_tags=frozenset({"read"})))
    with RoleBinding(role, box):
        names = {s["function"]["name"] for s in box.get_tool_schemas()}
        assert "echo_read" in names
        assert "echo_write" not in names and "danger" not in names
    # Deactivated → everything advertised again.
    names = {s["function"]["name"] for s in box.get_tool_schemas()}
    assert {"echo_read", "echo_write", "danger"} <= names


def test_default_role_changes_nothing():
    box, called = make_toolbox()
    with RoleBinding(DEFAULT_ROLE, box):
        results = run_calls(box, call("danger"))
    assert called == {"danger": 1}
    assert results[0]["content"] == "boom"


def test_double_activation_raises():
    box, _ = make_toolbox()
    with RoleBinding(DEFAULT_ROLE, box):
        try:
            RoleBinding.activate(Role(id="other", selector=ALLOW_ALL), box)
        except RuntimeError:
            pass
        else:
            raise AssertionError("second activation must raise")


# ── RoleBinding lifecycle: hooks + capabilities ──────────────────────────


def test_role_hooks_registered_and_removed():
    box, _ = make_toolbox()
    order: list[str] = []
    box.hooks.register(HOOK_BEFORE_TOOL_EXECUTE, lambda **kw: order.append("toolbox"))

    role = Role(id="r", selector=ALLOW_ALL,
                hooks={HOOK_BEFORE_TOOL_EXECUTE: [lambda **kw: order.append("role")]})
    with RoleBinding(role, box):
        run_calls(box, call("echo_read"))
    # Toolbox (infrastructure) layer fires before the role (behaviour) layer.
    assert order == ["toolbox", "role"]

    order.clear()
    run_calls(box, call("echo_read"))
    assert order == ["toolbox"], "role hook must be gone after deactivation"


class _ToyCapability(BaseCapability):
    def __init__(self, id: str = "toy", requires: list[str] | None = None):
        super().__init__(id=id, description="toy capability")
        self._requires = requires or []

    def get_tools(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {"name": f"{self.id}_tool", "description": "toy",
                         "parameters": {"type": "object", "properties": {}}},
        }]

    def get_instructions(self) -> str:
        return f"[{self.id}] instructions"

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(requires=self._requires) if self._requires else None


def test_role_activates_and_removes_capabilities():
    box, _ = make_toolbox()
    role = Role(id="r", selector=ALLOW_ALL, capabilities=[_ToyCapability()])
    with RoleBinding(role, box):
        assert "toy_tool" in box.tools
        assert "toy" in box.get_loaded_capability_ids()
    assert "toy_tool" not in box.tools
    assert "toy" not in box.get_loaded_capability_ids()


def test_capability_missing_requirement_fails_activation():
    box, _ = make_toolbox()
    role = Role(id="r", selector=ALLOW_ALL,
                capabilities=[_ToyCapability(requires=["absent"])])
    try:
        RoleBinding.activate(role, box)
    except ValueError as exc:
        assert "absent" in str(exc)
    else:
        raise AssertionError("missing capability requirement must fail activation")


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
