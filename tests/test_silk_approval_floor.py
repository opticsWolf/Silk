# -*- coding: utf-8 -*-
"""The floor under ``requires_approval=True`` (spec D81; closes G1).

The flag was stored at registration and read by nothing: `_safe_execute`
held a `pass` with a TODO, and the only thing that ever asked was the
*policy* gate -- which exists only when a Role or hook config configures
one, and then sees only the tools that policy named. D73 leans on the flag
for `remove_node` and `disconnect`, so in a graph with no tool policy the
two destructive graph-authoring tools ran unasked.

What these tests defend:

* the flag asks on its own, with no policy anywhere;
* a grant may skip the question (unlike the load floor, which no grant
  may pre-authorise);
* every missing-answer path denies -- including the one where the toolbox
  has no floor at all, which is refused at the execution site;
* one flagged call produces one dialog, not two.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


from silk.functions.approval import (  # noqa: E402
    LEVEL_HUMAN,
    attach_approval_gate,
    bind_run_seam,
    headless_refusals,
)
from silk.functions.approval_floor import (  # noqa: E402
    APPROVAL_FLOOR_ATTR,
    attach_approval_floor,
    flagged,
)
from silk.functions.decision_seam import (  # noqa: E402
    DecisionSeam,
)
from silk.functions.grants import (  # noqa: E402
    SCOPE_RUN,
    GrantStore,
)
from silk.functions.hooks import (  # noqa: E402
    HOOK_WRAP_TOOL_EXECUTE,
    EssentialHookError,
)
from silk.functions.tool_box import ToolBox  # noqa: E402


# -- scaffolding -----------------------------------------------------------


class _Answerer:
    def __init__(self, holder, *, approve=True, remember=""):
        self.holder = holder
        self.approve = approve
        self.remember = remember
        self.asked = []

    def __call__(self, request):
        self.asked.append(request)
        seam = self.holder[0]
        if self.approve:
            seam.approve(request.decision_id, actor="frank",
                         remember=self.remember)
        else:
            seam.deny(request.decision_id, actor="frank", reason="not that one")

    @property
    def tools_asked_about(self):
        return [r.tool_name for r in self.asked]


def _seam(**kw):
    holder: list = [None]
    answerer = _Answerer(holder, **kw)
    holder[0] = DecisionSeam(answerer, timeout_s=5.0)
    return holder[0], answerer


def _box():
    """A ToolBox with one flagged tool and one ordinary one."""
    box = ToolBox(None, {"agent_id": "ag"})
    ran: list[str] = []

    @box.register("remove_node", "removes a node", risk="high",
                  requires_approval=True)
    def _remove(_pool, _session, **_kw):
        ran.append("remove_node")
        return "removed"

    @box.register("describe_graph", "reads the graph", risk="low")
    def _describe(_pool, _session, **_kw):
        ran.append("describe_graph")
        return "described"

    return box, ran


def _call(box, name, args=None):
    tc = SimpleNamespace(id="c1", function=SimpleNamespace(
        name=name, arguments=json.dumps(args or {})))
    return asyncio.run(box.execute_tool_calls_async([tc]))[0]["content"]


def _refusal(content):
    payload = json.loads(content)
    assert payload["applied"] is False and payload["approval_required"] is True
    assert payload["error"] is None, "a refusal is not an error to retry"
    return payload


# -- the flag carries its own gate ----------------------------------------


def test_the_flag_installs_the_floor_at_registration():
    box, _ran = _box()
    assert getattr(box, APPROVAL_FLOOR_ATTR, None) is not None, (
        "declaring requires_approval is the whole of what a tool author does"
    )
    assert flagged(box, "remove_node") and not flagged(box, "describe_graph")


def test_a_flagged_tool_asks_with_no_policy_anywhere():
    """G1 itself: this call used to run unasked in any run without a policy."""
    box, ran = _box()
    seam, answerer = _seam(approve=True)
    bind_run_seam(box, seam)

    assert _call(box, "remove_node", {"id": "n7"}) == "removed"
    assert ran == ["remove_node"]
    assert answerer.tools_asked_about == ["remove_node"]
    assert answerer.asked[0].tool_args == {"id": "n7"}
    assert answerer.asked[0].detail["requires_approval"] is True


def test_an_unflagged_tool_is_never_asked_about():
    box, ran = _box()
    seam, answerer = _seam()
    bind_run_seam(box, seam)

    assert _call(box, "describe_graph") == "described"
    assert ran == ["describe_graph"] and answerer.asked == []


def test_a_denial_refuses_and_runs_nothing():
    box, ran = _box()
    seam, _answerer = _seam(approve=False)
    bind_run_seam(box, seam)

    payload = _refusal(_call(box, "remove_node", {"id": "n7"}))
    assert "not that one" in payload["message"]
    assert ran == [], "a denial never fabricates success (I10)"


# -- missing answers deny (D36) -------------------------------------------


def test_no_seam_is_a_refusal_not_a_run():
    box, ran = _box()
    payload = _refusal(_call(box, "remove_node", {"id": "n7"}))
    assert "no way to ask" in payload["message"]
    assert ran == [] and headless_refusals(box) == 1


def test_a_flagged_tool_with_no_floor_at_all_is_refused():
    """The fail-closed half, at the site where G1's TODO sat.

    A capability writes the flag onto the definition it hands over, which
    lands in `ToolBox.tools` without going through `register`, so the
    registration-time install is not the only way a flagged tool arrives.
    """
    box = ToolBox(None, {"agent_id": "ag"})
    ran: list[str] = []

    @box.register("prune", "prunes", risk="high")
    def _prune(_pool, _session, **_kw):
        ran.append("prune")
        return "pruned"

    box.tools["prune"]["requires_approval"] = True   # as a capability does
    assert getattr(box, APPROVAL_FLOOR_ATTR, None) is None

    payload = json.loads(_call(box, "prune"))
    assert payload["error_type"] == "approval_required"
    assert ran == []


# -- grants ---------------------------------------------------------------


def test_a_run_grant_skips_the_second_question():
    box, ran = _box()
    seam, answerer = _seam(approve=True, remember=SCOPE_RUN)
    bind_run_seam(box, seam)

    assert _call(box, "remove_node", {"id": "n1"}) == "removed"
    assert _call(box, "remove_node", {"id": "n2"}) == "removed"
    assert answerer.tools_asked_about == ["remove_node"], (
        "'and don't ask again' is a grant, not a one-off"
    )
    assert ran == ["remove_node", "remove_node"]


def test_a_durable_grant_pre_authorises_without_asking(tmp_path):
    box, ran = _box()
    store = GrantStore(directory=tmp_path)
    store.grant(str(tmp_path), "remove_node", granted_by="frank")
    seam, answerer = _seam()
    bind_run_seam(box, seam)
    attach_approval_floor(box, grants=store, project_root=str(tmp_path))

    assert _call(box, "remove_node", {"id": "n1"}) == "removed"
    assert answerer.asked == [], (
        "unlike the load verbs (D77), a flagged tool may be pre-authorised"
    )


def test_re_attaching_replaces_the_floor_rather_than_stacking_it():
    box, _ran = _box()
    seam, answerer = _seam(approve=True)
    bind_run_seam(box, seam)
    attach_approval_floor(box)

    assert _call(box, "remove_node", {"id": "n1"}) == "removed"
    assert answerer.tools_asked_about == ["remove_node"], "one floor, one ask"


# -- one call, one dialog --------------------------------------------------


def test_a_policy_that_also_names_the_tool_does_not_ask_twice():
    box, ran = _box()
    seam, answerer = _seam(approve=True)
    bind_run_seam(box, seam)
    attach_approval_gate(box, tool_policy={"remove_node": LEVEL_HUMAN})

    assert _call(box, "remove_node", {"id": "n1"}) == "removed"
    assert answerer.tools_asked_about == ["remove_node"]
    assert ran == ["remove_node"]


# -- the floor cannot be removed ------------------------------------------


def test_the_floor_is_essential_and_outermost():
    box, _ran = _box()
    entry = getattr(box, APPROVAL_FLOOR_ATTR)
    assert box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE)[0] is entry, (
        "D37: middleware ahead of it could answer the call it must gate"
    )
    with pytest.raises(EssentialHookError):
        box.hooks.unregister_middleware(HOOK_WRAP_TOOL_EXECUTE, entry.callback)
