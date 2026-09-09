# -*- coding: utf-8 -*-
"""The unified approval gate (spec D30, D31, D34-D37; invariants I7, I10).

One ``wrap_tool_execute`` middleware resolves two policy domains -- task
changes and tool calls -- because D31 makes them two *policies*, not two
subsystems. The tests are grouped by the claim each one defends:

* the policy resolves the way the spec says (name beats risk band, and the
  snapshot is taken once at attach time);
* grants can only skip the question, never create one;
* every missing-answer path denies, and a denial never fabricates success
  (D36, I10);
* the gate is outermost and essential, so nothing downstream can either
  answer around it or remove it (D37, I7).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from types import SimpleNamespace

import pytest


from silk.functions.approval import (  # noqa: E402
    LEVEL_AGENT,
    LEVEL_HUMAN,
    RISK_KEYS,
    TOOL_PRESETS,
    attach_approval_gate,
    attach_signoff_gate,
    bind_run_seam,
    headless_refusals,
    normalize_tool_policy,
    run_seam,
    tool_preset_policy,
)
from silk.functions.decision_seam import (  # noqa: E402
    DecisionSeam,
)
from silk.functions.hook_catalog import (  # noqa: E402
    ToolApprovalConfig,
    attach_catalog_hooks,
    tool_policy_from_config,
)
from silk.functions.grants import (  # noqa: E402
    SCOPE_ALWAYS,
    SCOPE_RUN,
    GrantStore,
    RunGrants,
)
from silk.functions.hooks import (  # noqa: E402
    HOOK_WRAP_TOOL_EXECUTE,
    EssentialHookError,
)
from silk.functions.tool_box import ToolBox  # noqa: E402
from silk.functions.tools.task_tracker import (  # noqa: E402
    attach_task_tools,
)


# -- scaffolding -----------------------------------------------------------


class _Answerer:
    """A seam answerer that decides inline, the way a UI eventually will."""

    def __init__(self, seam_holder, *, approve=True, remember=""):
        self.holder = seam_holder
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
    """A seam plus the answerer that serves it (they reference each other)."""
    holder: list = [None]
    answerer = _Answerer(holder, **kw)
    holder[0] = DecisionSeam(answerer, timeout_s=5.0)
    return holder[0], answerer


def _box():
    """A ToolBox with one tool per risk band and a record of what ran."""
    box = ToolBox(None, {"agent_id": "ag"})
    ran: list[str] = []

    @box.register("peek", "reads a file", risk="low")
    def _peek(_pool, _session, **_kw):
        ran.append("peek")
        return "peeked"

    @box.register("write_file", "writes a file", risk="high")
    def _write(_pool, _session, **_kw):
        ran.append("write_file")
        return "wrote"

    @box.register("edit_file", "edits a file", risk="medium")
    def _edit(_pool, _session, **_kw):
        ran.append("edit_file")
        return "edited"

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


# -- the policy vocabulary -------------------------------------------------


def test_tool_policy_normalization_drops_junk_and_adds_nothing():
    assert normalize_tool_policy({"write_file": LEVEL_HUMAN, "x": "maybe",
                                  "": LEVEL_HUMAN}) == {
        "write_file": LEVEL_HUMAN,
    }
    assert normalize_tool_policy(None) == {}, (
        "an unnamed tool is ungated -- the tool domain is open-ended, so a "
        "default of 'gate everything' would make every new tool a prompt"
    )


def test_the_presets_climb_the_risk_bands():
    assert tool_preset_policy("off") == {}
    assert tool_preset_policy("high_risk") == {"high": LEVEL_HUMAN}
    assert set(tool_preset_policy("everything")) == set(RISK_KEYS)
    assert tool_preset_policy("nonsense") == TOOL_PRESETS["off"], "unknown → off"


def test_a_gate_with_nothing_to_gate_is_not_installed():
    box, _ran = _box()
    assert attach_approval_gate(box, task_policy={}, tool_policy={}) is None
    assert box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE) == []


# -- resolution ------------------------------------------------------------


def test_an_ungated_tool_is_never_asked_about():
    box, ran = _box()
    seam, answerer = _seam()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam)

    assert _call(box, "peek") == "peeked"
    assert ran == ["peek"] and answerer.asked == []


def test_a_gated_tool_runs_only_after_the_human_says_yes():
    box, ran = _box()
    seam, answerer = _seam(approve=True)
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam)

    assert _call(box, "write_file", {"path": "a.txt"}) == "wrote"
    assert ran == ["write_file"]
    assert answerer.tools_asked_about == ["write_file"]
    assert answerer.asked[0].tool_args == {"path": "a.txt"}, (
        "the human is shown the arguments, not just the tool name"
    )


def test_a_denial_refuses_and_the_tool_does_not_run():
    """I10's corollary: a denial never fabricates success."""
    box, ran = _box()
    seam, _answerer = _seam(approve=False)
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam)

    payload = _refusal(_call(box, "write_file"))
    assert ran == [], "the held call must not have run"
    assert "not that one" in payload["message"]
    assert payload["target"] == "write_file"


def test_a_risk_band_gates_the_tools_in_it():
    box, ran = _box()
    seam, answerer = _seam()
    attach_approval_gate(box, tool_policy=tool_preset_policy("writes"),
                         seam=seam)

    _call(box, "peek")
    _call(box, "edit_file")
    _call(box, "write_file")
    assert answerer.tools_asked_about == ["edit_file", "write_file"]
    assert ran == ["peek", "edit_file", "write_file"]


def test_a_tool_name_beats_its_risk_band():
    """The specific beats the general, in both directions."""
    box, _ran = _box()
    seam, answerer = _seam()
    attach_approval_gate(box, seam=seam, tool_policy={
        "high": LEVEL_HUMAN,          # would gate write_file ...
        "write_file": LEVEL_AGENT,    # ... but this one is named
        "peek": LEVEL_HUMAN,          # a low-risk tool, gated by name
    })

    _call(box, "write_file")
    _call(box, "peek")
    assert answerer.tools_asked_about == ["peek"]


def test_the_policy_is_snapshotted_at_attach_time():
    """D38: editing a Role mid-run affects the next run, not this one."""
    box, ran = _box()
    seam, answerer = _seam()
    policy = {"write_file": LEVEL_HUMAN}
    attach_approval_gate(box, tool_policy=policy, seam=seam)

    policy["peek"] = LEVEL_HUMAN          # a live edit, after the fact
    policy.pop("write_file")

    _call(box, "peek")
    _call(box, "write_file")
    assert answerer.tools_asked_about == ["write_file"]
    assert ran == ["peek", "write_file"]


# -- grants ----------------------------------------------------------------


def test_a_run_grant_skips_the_question(tmp_path):
    box, ran = _box()
    seam, answerer = _seam()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam, run_grants=RunGrants(["write_file"]),
                         project_root=str(tmp_path))

    assert _call(box, "write_file") == "wrote"
    assert answerer.asked == [] and ran == ["write_file"]


def test_a_durable_grant_skips_the_question(tmp_path):
    store = GrantStore(tmp_path)
    store.grant(tmp_path, "write_file", granted_by="frank")
    box, ran = _box()
    seam, answerer = _seam()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam, grants=store, project_root=str(tmp_path))

    assert _call(box, "write_file") == "wrote"
    assert answerer.asked == [] and ran == ["write_file"]


def test_a_grant_for_another_project_does_not_skip_it(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    store = GrantStore(tmp_path)
    store.grant(other, "write_file")
    box, _ran = _box()
    seam, answerer = _seam()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam, grants=store, project_root=str(tmp_path))

    _call(box, "write_file")
    assert answerer.tools_asked_about == ["write_file"]


def test_a_grant_cannot_create_a_question_it_only_removes_one(tmp_path):
    """Consulting grants first cannot make the gate stricter than the policy."""
    store = GrantStore(tmp_path)
    store.grant(tmp_path, "peek")
    box, ran = _box()
    seam, answerer = _seam()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam, grants=store, project_root=str(tmp_path))

    assert _call(box, "peek") == "peeked"
    assert answerer.asked == [] and ran == ["peek"]


def test_remember_for_the_run_stops_asking_again():
    box, ran = _box()
    seam, answerer = _seam(remember=SCOPE_RUN)
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam)

    _call(box, "write_file")
    _call(box, "write_file")
    assert len(answerer.asked) == 1, "'don't ask again this run' means once"
    assert ran == ["write_file", "write_file"]


def test_remember_always_writes_a_durable_grant(tmp_path):
    store = GrantStore(tmp_path)
    box, _ran = _box()
    seam, answerer = _seam(remember=SCOPE_ALWAYS)
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam, grants=store, project_root=str(tmp_path))

    _call(box, "write_file")
    assert len(answerer.asked) == 1
    assert GrantStore(tmp_path).allows(tmp_path, "write_file") is True, (
        "the grant outlives the run that made it"
    )


def test_remember_always_without_a_store_still_approves_this_call():
    box, ran = _box()
    seam, _answerer = _seam(remember=SCOPE_ALWAYS)
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam)          # no durable store configured
    assert _call(box, "write_file") == "wrote" and ran == ["write_file"]


# -- the seam is bound per run (D38) ---------------------------------------


def test_a_seam_bound_at_run_time_is_the_one_the_gate_asks():
    """The gate is installed when the box is built; who to ask arrives later."""
    box, ran = _box()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN})
    seam, answerer = _seam()

    bind_run_seam(box, seam)
    assert _call(box, "write_file") == "wrote"
    assert answerer.tools_asked_about == ["write_file"] and ran == ["write_file"]


def test_unbinding_leaves_a_gate_with_nothing_to_ask():
    box, ran = _box()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN})
    seam, _answerer = _seam()

    with run_seam(box, seam):
        assert _call(box, "write_file") == "wrote"
    _refusal(_call(box, "write_file"))
    assert ran == ["write_file"], (
        "a seam left bound after its run is a widget nobody is watching"
    )


def test_an_attached_seam_wins_over_a_bound_one():
    """A headless embedder with exactly one seam is not overridden by a run."""
    box, _ran = _box()
    attached, attached_answerer = _seam()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=attached)
    bound, bound_answerer = _seam()

    bind_run_seam(box, bound)
    _call(box, "write_file")
    assert attached_answerer.asked and bound_answerer.asked == []


# -- fail closed (D36) -----------------------------------------------------


def test_a_gate_with_no_seam_refuses():
    """A run configured to ask, with nothing to ask with."""
    box, ran = _box()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN})
    payload = _refusal(_call(box, "write_file"))
    assert ran == []
    assert "durable grant" in payload["message"]


def test_a_headless_run_says_so_once_and_counts_the_rest(caplog):
    """Denying is right; denying silently forty times looks like a hang.

    §22 q1d: a batch evaluation whose every gated tool is refused should
    say so once, not once per call -- and the count is what lets the node
    say it at the end (D53's legibility rule).
    """
    box, ran = _box()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN})

    with caplog.at_level("WARNING"):
        for _ in range(3):
            _refusal(_call(box, "write_file"))

    assert ran == []
    said = [r for r in caplog.records if "no way to ask" in r.getMessage()]
    assert len(said) == 1, "once per run, not once per call"
    assert headless_refusals(box) == 3


def test_binding_a_seam_starts_a_fresh_count():
    box, _ran = _box()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN})
    _refusal(_call(box, "write_file"))
    assert headless_refusals(box) == 1

    bind_run_seam(box, DecisionSeam(None, timeout_s=5.0))
    assert headless_refusals(box) == 0, (
        "the count is a fact about this run, not about the toolbox"
    )


def test_a_seam_with_no_answerer_refuses():
    box, ran = _box()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=DecisionSeam(None, timeout_s=5.0))
    _refusal(_call(box, "write_file"))
    assert ran == []


def test_a_transport_that_raises_refuses():
    def _broken(_request):
        raise RuntimeError("the widget is gone")

    box, ran = _box()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=DecisionSeam(_broken, timeout_s=5.0))
    _refusal(_call(box, "write_file"))
    assert ran == []


def test_a_cancelled_seam_refuses_every_later_call():
    """Stop, mid-run: what is already held denies and so does what follows."""
    box, ran = _box()
    seam, _answerer = _seam()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam)
    seam.cancel("user pressed Stop")

    payload = _refusal(_call(box, "write_file"))
    assert ran == [] and "stopped" in payload["message"]


# -- the gate as a monotonic guard (D37, I7, I10) --------------------------


def _middleware_names(box):
    return [e.callback.__name__
            for e in box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE)]


def test_the_gate_is_forced_outermost():
    box, ran = _box()

    async def bypass(handler=None, **_kw):
        return "hijacked"                     # never calls handler()

    box.hooks.register_middleware(HOOK_WRAP_TOOL_EXECUTE, bypass)
    seam, answerer = _seam(approve=False)
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam)

    assert _middleware_names(box)[0] == "gate", (
        "registration order put the gate second; D37 says it must not stay there"
    )
    _refusal(_call(box, "write_file"))
    assert ran == [] and len(answerer.asked) == 1, (
        "a middleware ahead of the gate could otherwise answer a call the "
        "gate never sees"
    )


def test_a_middleware_registered_later_does_not_displace_the_gate():
    box, _ran = _box()
    seam, _answerer = _seam()
    attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                         seam=seam)

    async def latecomer(handler=None, **_kw):
        return await handler()

    box.hooks.register_middleware(HOOK_WRAP_TOOL_EXECUTE, latecomer)
    assert _middleware_names(box) == ["gate", "latecomer"]


def test_forcing_an_already_outermost_gate_is_a_no_op():
    box, _ran = _box()
    seam, _answerer = _seam()
    entry = attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                                 seam=seam)
    box.hooks.make_outermost(HOOK_WRAP_TOOL_EXECUTE, entry)
    assert box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE) == [entry]


def test_the_gate_is_essential_and_cannot_be_removed():
    """I7: a gate a downstream layer can drop is not a gate."""
    box, _ran = _box()
    seam, _answerer = _seam()
    entry = attach_approval_gate(box, tool_policy={"write_file": LEVEL_HUMAN},
                                 seam=seam)
    assert entry.essential is True

    with pytest.raises(EssentialHookError):
        box.hooks.unregister_middleware(HOOK_WRAP_TOOL_EXECUTE, entry)
    box.hooks.clear()
    assert box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE) == [entry]


# -- one gate, two domains (D31) -------------------------------------------


def _task_box(tmp, **kw):
    box = ToolBox(None, {"agent_id": "ag"})
    sandbox = SimpleNamespace(root_dir=tmp)
    attach_task_tools(box, sandbox)
    entry = attach_approval_gate(box, sandbox, **kw)
    _call(box, "plan_start", {"goal": "g", "tasks": [{"title": "A"},
                                                     {"title": "B"}]})
    return box, entry


def test_both_domains_share_a_single_middleware():
    tmp = tempfile.mkdtemp()
    seam, answerer = _seam()
    box, _entry = _task_box(
        tmp, task_policy={"goal": LEVEL_HUMAN}, seam=seam,
        tool_policy={"task_rescope": LEVEL_HUMAN},
    )
    assert len(box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE)) == 1, (
        "D31 unifies the two approval domains in the gate, not in a store"
    )

    _call(box, "goal_revise", {"new_text": "G2", "rationale": "pivot"})
    _call(box, "task_rescope", {"id": "t1", "new_title": "A2",
                                "rationale": "narrower"})
    assert answerer.tools_asked_about == ["goal_revise", "task_rescope"]


def test_a_task_change_names_its_change_type_not_its_tool():
    tmp = tempfile.mkdtemp()
    seam, answerer = _seam(approve=False)
    box, _entry = _task_box(tmp, task_policy={"goal": LEVEL_HUMAN}, seam=seam)

    payload = _refusal(_call(box, "goal_revise", {"new_text": "G2",
                                                  "rationale": "pivot"}))
    assert payload["change_type"] == "goal" and payload["target"] == "goal"
    assert answerer.asked[0].detail["change_type"] == "goal"


def test_the_plan_closing_completion_is_its_own_change_type():
    """complete_final is resolved from plan state, which is why the gate
    holds a toolbox handle rather than being a free function."""
    tmp = tempfile.mkdtemp()
    seam, answerer = _seam()
    box, _entry = _task_box(
        tmp, task_policy={"complete_final": LEVEL_HUMAN}, seam=seam)

    _call(box, "task_complete", {"id": "t1", "rationale": "done"})
    assert answerer.asked == [], "one task still open; not the final completion"
    _call(box, "task_complete", {"id": "t2", "rationale": "done"})
    assert [r.detail["change_type"] for r in answerer.asked] == ["complete_final"]


def test_the_signoff_entry_point_is_the_same_gate():
    tmp = tempfile.mkdtemp()
    box = ToolBox(None, {"agent_id": "ag"})
    sandbox = SimpleNamespace(root_dir=tmp)
    attach_task_tools(box, sandbox)
    seam, answerer = _seam(approve=False)
    entry = attach_signoff_gate(box, sandbox, mode="strict", seam=seam)

    assert entry is not None and entry.essential is True
    _call(box, "plan_start", {"goal": "g", "tasks": [{"title": "A"}]})
    _refusal(_call(box, "goal_revise", {"new_text": "G2", "rationale": "p"}))
    assert answerer.tools_asked_about == ["goal_revise"]


def test_an_all_agent_signoff_policy_installs_nothing():
    tmp = tempfile.mkdtemp()
    box = ToolBox(None, {"agent_id": "ag"})
    sandbox = SimpleNamespace(root_dir=tmp)
    attach_task_tools(box, sandbox)
    assert attach_signoff_gate(box, sandbox, mode="auto") is None


# -- the catalog hook (the ToolBox node's surface) -------------------------


def test_the_tool_approval_config_adds_names_on_top_of_a_band():
    policy = tool_policy_from_config(ToolApprovalConfig(
        preset="high_risk", tools="run_command, , send_email"))
    assert policy == {"high": LEVEL_HUMAN, "run_command": LEVEL_HUMAN,
                      "send_email": LEVEL_HUMAN}, (
        "'ask before run_command' is the common case and fits no risk band"
    )
    assert tool_policy_from_config(ToolApprovalConfig()) == {}


def test_selecting_both_catalog_hooks_installs_one_gate():
    """D31: two policy domains, one middleware -- not two."""
    tmp = tempfile.mkdtemp()
    box = ToolBox(None, {"agent_id": "ag"})
    sandbox = SimpleNamespace(root_dir=tmp)
    attach_task_tools(box, sandbox)

    @box.register("write_file", "writes a file", risk="high")
    def _write(_pool, _session, **_kw):
        return "wrote"

    attach_catalog_hooks(box, sandbox, names=("signoff", "tool_approval"),
                         configs={"signoff": {"preset": "strict"},
                                  "tool_approval": {"preset": "high_risk",
                                                    "durable_grants": False}})
    assert len(box.hooks.middleware_entries(HOOK_WRAP_TOOL_EXECUTE)) == 1

    seam, answerer = _seam(approve=False)
    with run_seam(box, seam):
        _call(box, "plan_start", {"goal": "g", "tasks": [{"title": "A"}]})
        _refusal(_call(box, "write_file"))
        _refusal(_call(box, "goal_revise", {"new_text": "G2",
                                            "rationale": "pivot"}))
    assert answerer.tools_asked_about == ["write_file", "goal_revise"]


def test_tool_approval_alone_needs_no_task_store():
    """The configuration that most needs a gate: file tools, no planning."""
    tmp = tempfile.mkdtemp()
    box, ran = _box()
    attach_catalog_hooks(box, SimpleNamespace(root_dir=tmp),
                         names=("tool_approval",),
                         configs={"tool_approval": {"tools": "write_file",
                                                    "durable_grants": False}})
    seam, answerer = _seam()
    with run_seam(box, seam):
        assert _call(box, "write_file") == "wrote"
        assert _call(box, "peek") == "peeked"
    assert answerer.tools_asked_about == ["write_file"] and len(ran) == 2
