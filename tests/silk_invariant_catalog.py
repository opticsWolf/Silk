# -*- coding: utf-8 -*-
"""The invariant catalog: one record per invariant, one per violation class.

Spec D27, Phase 1 item 1. This is *data*. The executable side lives in
``test_silk_invariants.py``, which refuses to run unless every record here
has a check and every check has a record -- the whole point of encoding the
invariants as fixtures is that the document and the suite cannot drift apart.

Each record names a violation class, because an invariant is only pinned by
the ways it can break. "One result per call" is not tested by a happy path;
it is tested by the five failure modes that must still produce a result.

Records carry a status:

``ENFORCED``
    The runtime upholds this today. The check runs, and a regression fails
    the suite.
``PENDING``
    The invariant is specified but not yet implemented. The check is written
    anyway and marked ``xfail(strict=True)``, so it fails the suite the day
    the behaviour lands and the record is not updated. A pending fixture that
    starts passing is as much a drift as one that starts failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ENFORCED = "enforced"
PENDING = "pending"


@dataclass(frozen=True)
class Fixture:
    """One violation class of one invariant."""

    invariant: str
    case: str
    describes: str
    status: str = ENFORCED
    #: Why it cannot be enforced yet, and what will change that. Required
    #: for PENDING, meaningless otherwise.
    pending: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.invariant, self.case)

    def __str__(self) -> str:      # pragma: no cover - test ids only
        return f"{self.invariant}:{self.case}"


@dataclass(frozen=True)
class Invariant:
    """An invariant, its wording, and where the runtime upholds it."""

    id: str
    #: The bolded statement, verbatim from the document it comes from. A
    #: meta-test re-reads the document and fails if these diverge.
    title: str
    source: str
    enforced_by: str
    fixtures: tuple[Fixture, ...] = field(default_factory=tuple)


def _fx(invariant: str, *cases: tuple) -> tuple[Fixture, ...]:
    return tuple(Fixture(invariant, *case) for case in cases)


#: Where each half of the catalog is written down, relative to the silk repo
#: root. Re-read by the meta-tests.
ARCHITECTURE_DOC = "docs/architecture/17-invariants.md"
SPEC_DOC = "docs/DESIGN_SPEC_DRAFT.md"


INVARIANTS: tuple[Invariant, ...] = (
    Invariant(
        id="I1",
        title=(
            "A tool batch always returns one result per call, same shape, "
            "failures included."
        ),
        source=ARCHITECTURE_DOC,
        enforced_by="ToolBox.execute_tool_calls_async / _safe_execute",
        fixtures=_fx(
            "I1",
            ("unknown_tool", "a call naming no registered tool still returns a result"),
            ("role_denied", "a call the role forbids still returns a result"),
            ("validation_error", "arguments that fail the model still return a result"),
            ("timeout", "a tool that outruns its timeout still returns a result"),
            ("exception", "a tool that raises still returns a result"),
        ),
    ),
    Invariant(
        id="I2",
        title="`HOOK_AFTER_RUN` fires exactly once on every exit path",
        source=ARCHITECTURE_DOC,
        enforced_by="the finally in AgentLoop.run",
        fixtures=_fx(
            "I2",
            ("normal_completion", "the ordinary path fires it once"),
            ("usage_limit", "a cap that stops the run still fires it once"),
            ("stream_error", "an engine that raises still fires it once"),
            ("early_close", "a generator closed before exhaustion still fires it once"),
        ),
    ),
    Invariant(
        id="I3",
        title="The loop never executes tools.",
        source=ARCHITECTURE_DOC,
        enforced_by="AgentLoop dispatches batches; the engine owns one request",
        fixtures=_fx(
            "I3",
            ("engine_is_never_asked_to_run_a_tool",
             "an engine that refuses any non-protocol access still drives a tool round"),
            ("every_call_goes_through_the_toolbox",
             "the toolbox sees exactly the calls the model made"),
        ),
    ),
    Invariant(
        id="I4",
        title="A role-denied tool is both invisible and refused.",
        source=ARCHITECTURE_DOC,
        enforced_by="get_tool_schemas() at advertisement, role_permits() at dispatch",
        fixtures=_fx(
            "I4",
            ("not_advertised", "the denied tool is absent from the schemas"),
            ("refused_at_dispatch", "calling it anyway is refused and never executes"),
        ),
    ),
    Invariant(
        id="I5",
        title="Store reads never mutate.",
        source=ARCHITECTURE_DOC,
        enforced_by="plan_changed_event / task_store revisions",
        fixtures=_fx(
            "I5",
            ("read_does_not_bump_the_revision", "loading a plan leaves its revision alone"),
            ("unchanged_plan_does_not_restream", "a second read emits no plan_summary"),
        ),
    ),
    Invariant(
        id="I6",
        title="File access narrows monotonically.",
        source=SPEC_DOC,
        enforced_by="functions/file_grants.py, functions/toolset_build.py, "
                    "functions/tools/file_sandbox.py",
        fixtures=_fx(
            "I6",
            ("entry_outside_the_ceiling_is_dropped",
             "a permission naming a path outside the roots grants nothing"),
            ("root_outside_the_ceiling_is_ignored",
             "a declared root outside the ceiling does not become the root"),
            ("confinement_cannot_be_switched_off",
             "no derived sandbox comes back with the escape hatch open"),
            ("a_downstream_grant_cannot_widen_an_upstream_one",
             "composing grants down the ToolSet -> Role -> Agent chain only "
             "ever narrows, whatever the downstream one asks for"),
            ("a_run_scoped_restriction_cannot_widen_the_sandbox",
             "applying a grant in place narrows the live sandbox and never "
             "adds a path it did not already cover"),
        ),
    ),
    Invariant(
        id="I7",
        title="Essential hooks survive derivation.",
        source=SPEC_DOC,
        enforced_by="functions/hooks.py, functions/toolset_build.py",
        fixtures=_fx(
            "I7",
            ("essential_hook_survives_a_toolset",
             "a hook declared essential rides the recipe into a derived box",
             ENFORCED),
            ("essential_hook_cannot_be_dropped",
             "a derived box cannot unregister an essential hook",
             ENFORCED),
        ),
    ),
    Invariant(
        id="I8",
        title="Discovery obeys the role gate.",
        source=SPEC_DOC,
        enforced_by="functions/tool_search.py, functions/tool_discovery.py",
        fixtures=_fx(
            "I8",
            ("search_hides_a_denied_tool",
             "tool search does not return a tool the active role forbids"),
            ("search_hides_an_unusable_capability",
             "nor a capability every one of whose tools the role forbids"),
            ("auto_load_does_not_widen_the_role",
             "loading a tool at dispatch does not make a denied tool "
             "callable -- the gate runs on the loaded tool too"),
        ),
    ),
    Invariant(
        id="I9",
        title="Compaction cuts on whole-round boundaries.",
        source=SPEC_DOC,
        enforced_by="functions/compaction.py, functions/graph_engine.py",
        fixtures=_fx(
            "I9",
            ("an_assistant_turn_and_its_results_move_together",
             "compacting never separates tool_calls from their tool results",
             ENFORCED),
            ("a_tool_result_is_never_orphaned",
             "no surviving tool-role message lacks its assistant turn",
             ENFORCED),
        ),
    ),
    Invariant(
        id="I10",
        title="Guard middleware is monotonic.",
        source=SPEC_DOC,
        enforced_by="functions/hooks.py, functions/approval.py",
        fixtures=_fx(
            "I10",
            ("the_gate_outranks_an_earlier_registration",
             "a middleware registered before the gate cannot answer around it",
             ENFORCED),
            ("a_later_registration_does_not_wrap_the_gate",
             "registering after the gate does not displace it from position 0",
             ENFORCED),
            ("a_denial_never_fabricates_a_result",
             "a refused call produces applied:false, never a success payload",
             ENFORCED),
        ),
    ),
    Invariant(
        id="I11",
        title="The model-visible prefix grows only at the tail.",
        source=SPEC_DOC,
        enforced_by="functions/prefix_guard.py, functions/graph_engine.py",
        fixtures=_fx(
            "I11",
            ("appending_is_not_a_break",
             "a run that only appends messages reports nothing",
             ENFORCED),
            ("a_volatile_system_prompt_is_caught",
             "a system prompt that does not render byte-identically is named",
             ENFORCED),
            ("a_rewritten_message_is_caught",
             "an already-sent message that changed is named, with its index",
             ENFORCED),
            ("compaction_is_the_one_forgiven_break",
             "a declared compaction is forgiven exactly once, not forever",
             ENFORCED),
        ),
    ),
    Invariant(
        id="I12",
        title="A human decision surface may be a node iff the decision "
              "happens at a turn boundary.",
        source=SPEC_DOC,
        enforced_by="functions/task_board.py, nodes/task_hub.py, "
                    "functions/decision_seam.py, functions/task_store.py",
        fixtures=_fx(
            "I12",
            ("a_counting_surface_cannot_answer",
             "the Task Hub's pending counter holds ids only -- no path from "
             "a board to resolving a live request (D58 counts, D59 answers)",
             ENFORCED),
            ("nothing_is_parked_for_a_later_decision",
             "the store has no awaiting-sign-off state, so no node can "
             "approve a change after the turn that asked for it ended",
             ENFORCED),
        ),
    ),
)


#: The parts of the shipped sign-off gate that survive into D31 and are
#: therefore worth pinning now. The park/hold/apply path is deleted by D31,
#: so it is deliberately absent -- pinning behaviour that is scheduled for
#: removal buys nothing and costs a rewrite.
GATE_FIXTURES: tuple[Fixture, ...] = _fx(
    "D31",
    ("preset_expands_to_a_full_policy",
     "every named mode resolves to a level for every change type"),
    ("unknown_preset_falls_back_to_auto",
     "an unrecognised mode is the permissive one, not a crash"),
    ("an_explicit_policy_beats_the_preset",
     "normalize_policy drops unknown types and unknown levels"),
    ("complete_resolves_to_complete_final",
     "completing the last open task is a plan-closing completion"),
    ("complete_stays_complete_with_work_left",
     "completing one of several open tasks is an ordinary completion"),
)


ALL_FIXTURES: tuple[Fixture, ...] = tuple(
    fx for inv in INVARIANTS for fx in inv.fixtures
) + GATE_FIXTURES


def by_key() -> dict[tuple[str, str], Fixture]:
    return {fx.key: fx for fx in ALL_FIXTURES}
