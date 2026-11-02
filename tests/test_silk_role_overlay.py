# -*- coding: utf-8 -*-
"""A Role's word on the ToolBox-wired hooks: tighten, adjust, never loosen.

The ToolBox owns the approval gate, sign-off, compaction, spill, memory and
the run log. A Role may layer on three of them for its own runs: its gate
asks only where it is stricter than the box's (and never twice for one
call), and its compaction settings replace the box's. It can loosen
nothing, and the three that are properties of the box cannot be touched.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from silk.functions.approval import LEVEL_HUMAN, attach_approval_gate, bind_run_seam
from silk.functions.compaction import compactor_for
from silk.functions.decision_seam import DecisionSeam
from silk.functions.hook_catalog import attach_catalog_hooks
from silk.functions.hooks import HOOK_WRAP_TOOL_EXECUTE
from silk.functions.role import Role, RoleBinding
from silk.functions.tool_box import ToolBox


class _Answerer:
    def __init__(self, holder, approve=True):
        self.holder, self.approve, self.asked = holder, approve, []

    def __call__(self, request):
        self.asked.append(request.tool_name)
        seam = self.holder[0]
        if self.approve:
            seam.approve(request.decision_id, actor="w")
        else:
            seam.deny(request.decision_id, actor="w", reason="no")


def _box(*, box_policy=None):
    box = ToolBox(None, {"agent_id": "ag"})
    ran: list[str] = []

    @box.register("peek", "reads", risk="low")
    def _peek(_p, _s, **_kw):
        ran.append("peek")
        return "peeked"

    @box.register("write_file", "writes", risk="high")
    def _write(_p, _s, **_kw):
        ran.append("write_file")
        return "wrote"

    if box_policy is not None:
        attach_approval_gate(box, tool_policy=box_policy)
    holder: list = [None]
    answerer = _Answerer(holder)
    holder[0] = DecisionSeam(answerer, timeout_s=5.0)
    bind_run_seam(box, holder[0])
    return box, ran, answerer


def _call(box, name):
    tc = SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments="{}"))
    return asyncio.run(box.execute_tool_calls_async([tc]))[0]["content"]


def _role(names, configs=None):
    return Role(id="r", hook_settings={"names": list(names), "configs": configs or {}})


# -- the gate ---------------------------------------------------------------


def test_a_role_can_gate_a_tool_the_box_lets_through():
    box, ran, answerer = _box(box_policy={"write_file": LEVEL_HUMAN})
    role = _role(["tool_approval"], {"tool_approval": {"tools": "peek"}})

    with RoleBinding.activate(role, box):
        assert _call(box, "peek") == "peeked"
    assert answerer.asked == ["peek"], "the role asked about its own tool"

    answerer.asked.clear()
    assert _call(box, "peek") == "peeked"
    assert answerer.asked == [], "only for this role's runs"


def test_a_call_both_gate_is_asked_about_once():
    box, ran, answerer = _box(box_policy={"write_file": LEVEL_HUMAN})
    role = _role(["tool_approval"], {"tool_approval": {"preset": "everything"}})

    with RoleBinding.activate(role, box):
        assert _call(box, "write_file") == "wrote"
        assert _call(box, "peek") == "peeked"
    assert answerer.asked == ["write_file", "peek"], (
        "write_file once (the box asked; the role is no stricter there), "
        "peek once (only the role gates it)"
    )


def test_a_role_cannot_loosen_the_box():
    box, ran, answerer = _box(box_policy={"write_file": LEVEL_HUMAN})
    role = _role(["tool_approval"], {"tool_approval": {"preset": "off"}})

    with RoleBinding.activate(role, box):
        assert _call(box, "write_file") == "wrote"
    assert answerer.asked == ["write_file"], "the box's gate still asked"


def test_a_role_gates_on_its_own_where_the_box_has_no_gate():
    box, ran, answerer = _box()
    role = _role(["tool_approval"], {"tool_approval": {"tools": "write_file"}})

    with RoleBinding.activate(role, box):
        assert _call(box, "write_file") == "wrote"
        assert _call(box, "peek") == "peeked"
    assert answerer.asked == ["write_file"]


def test_the_box_gate_stays_outermost_and_the_role_gate_is_removed():
    box, ran, answerer = _box(box_policy={"write_file": LEVEL_HUMAN})
    base = box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE)[0]
    role = _role(["tool_approval"], {"tool_approval": {"tools": "peek"}})

    binding = RoleBinding.activate(role, box)
    entries = box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE)
    assert entries[0] is base and len(entries) == 2
    binding.deactivate()
    assert box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE) == [base]


def test_a_denial_from_the_role_gate_refuses_the_call():
    box, ran, answerer = _box()
    answerer.approve = False
    role = _role(["tool_approval"], {"tool_approval": {"tools": "peek"}})

    with RoleBinding.activate(role, box):
        payload = json.loads(_call(box, "peek"))
    assert payload["applied"] is False and ran == []


def test_the_role_gate_tightens_the_box_built_by_its_hooks():
    """The real wiring: the box's gate comes from attach_catalog_hooks."""
    box, ran, answerer = _box()
    attach_catalog_hooks(box, None, names=["tool_approval"],
                         configs={"tool_approval": {"tools": "write_file"}})
    role = _role(["tool_approval"], {"tool_approval": {"tools": "peek,write_file"}})

    with RoleBinding.activate(role, box):
        _call(box, "write_file")
        _call(box, "peek")
    assert answerer.asked == ["write_file", "peek"]


# -- compaction -------------------------------------------------------------


def _compacting_box(**cfg):
    box = ToolBox()
    attach_catalog_hooks(box, None, names=["auto_compact"],
                         configs={"auto_compact": cfg})
    return box


def test_a_role_that_ticks_auto_compact_replaces_the_box_settings():
    box = _compacting_box(keep_recent_pct=40, focus="box")
    c = compactor_for(box, _role(["auto_compact"], {"auto_compact": {
        "keep_recent_pct": 20, "focus": "role"}}))
    assert c.auto and c.keep_recent_ratio == 0.20 and c.focus == "role"


def test_a_role_that_does_not_tick_it_has_no_say():
    box = _compacting_box(keep_recent_pct=40)
    c = compactor_for(box, _role(["timing"]))
    assert c.auto and c.keep_recent_ratio == 0.40


def test_a_role_can_switch_compaction_on_where_the_box_left_it_off():
    box = ToolBox()
    attach_catalog_hooks(box, None, names=["timing"])
    assert compactor_for(box).auto is False
    assert compactor_for(box, _role(["auto_compact"])).auto is True


# -- what a Role cannot touch -----------------------------------------------


def test_the_box_only_hooks_are_dropped_from_a_role_and_said_so(caplog):
    from silk.nodes.role import _role_hook_settings

    out = _role_hook_settings("r", {"names": ["spill", "timing", "run_log",
                                              "tool_approval"], "configs": {}})
    assert out["names"] == ["timing", "tool_approval"]


def test_hook_settings_round_trip():
    role = _role(["tool_approval"], {"tool_approval": {"tools": "peek"}})
    assert Role.from_dict(role.to_dict()).hook_settings == role.hook_settings
    assert Role.from_dict({"id": "old"}).hook_settings == {}
