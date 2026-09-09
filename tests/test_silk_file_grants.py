# -*- coding: utf-8 -*-
"""File grants as data, and the chain that may only narrow (spec D16-D18, I6).

The model is the easy half. What these tests are actually defending is the
direction of every operation: `narrow` is not symmetric, a malformed grant
grants nothing rather than falling back to the wider one it failed to
parse, and no value of this model turns confinement off.

That direction is the whole feature. A permission structure that widens
under composition is worse than none at all, because the graph then shows a
narrowing that is not one -- and the failure is invisible until an agent
writes somewhere nobody expected it to reach.
"""

from __future__ import annotations


import pytest

from pydantic import ValidationError


from silk.functions.file_grants import (  # noqa: E402
    MODE_BLOCKED,
    MODE_READ,
    MODE_READ_WRITE,
    FileGrant,
    FileGrants,
    grants_from_paths,
    lesser_mode,
    resolve_grants,
)
from silk.functions.tools.file_sandbox import (  # noqa: E402
    FileToolSandbox,
)
from silk.functions.toolset_build import (  # noqa: E402
    sandbox_from_permissions,
    split_by_ceiling,
)

ROOT = "/project"
SRC = "/project/src"
FILE = "/project/src/main.py"
SECRETS = "/project/src/secrets.env"
OUTSIDE = "/etc"


def _grants(*entries, root: str = ROOT) -> FileGrants:
    return FileGrants(root=root, roots=[root],
                      entries=[FileGrant(path=p, mode=m) for p, m in entries])


# -- the model (D17) --------------------------------------------------------


def test_a_dict_is_accepted_and_a_string_is_not():
    assert FileGrants.coerce({"root": ROOT, "entries": [{"path": SRC}]}) is not None
    assert FileGrants.coerce(None) is None, "no grant is not an empty grant"
    with pytest.raises(TypeError):
        FileGrants.coerce("/project")


@pytest.mark.parametrize("bad", [
    {"entries": [{"path": "   "}]},          # a grant needs a path
    {"entries": [{"path": SRC, "mode": "delete_everything"}]},
    {"entries": "not a list"},
])
def test_a_malformed_grant_is_refused_at_the_boundary(bad):
    assert FileGrants.is_valid(bad) is False
    with pytest.raises(ValidationError):
        FileGrants.coerce(bad)


def test_the_default_mode_is_the_lesser_one():
    assert FileGrant(path=SRC).mode == MODE_READ, (
        "an unstated mode must not mean write access"
    )


def test_lesser_mode_treats_anything_unknown_as_blocked():
    assert lesser_mode(MODE_READ_WRITE, MODE_READ) == MODE_READ
    assert lesser_mode(MODE_READ, MODE_BLOCKED) == MODE_BLOCKED
    assert lesser_mode(MODE_READ, "nonsense") == "nonsense", "unknown reads as 0"


# -- the hierarchy ----------------------------------------------------------


def test_a_directory_grant_covers_its_subtree():
    grants = _grants((SRC, MODE_READ_WRITE))
    assert grants.mode_for(FILE) == MODE_READ_WRITE
    assert grants.mode_for("/project/docs/x.md") == MODE_BLOCKED


def test_the_nearest_entry_wins_so_a_block_carves_a_hole():
    grants = _grants((SRC, MODE_READ_WRITE), (SECRETS, MODE_BLOCKED))
    assert grants.mode_for(FILE) == MODE_READ_WRITE
    assert grants.mode_for(SECRETS) == MODE_BLOCKED, (
        "a per-path entry beats the directory it sits in"
    )


def test_an_uncovered_path_is_blocked_not_defaulted():
    assert _grants((SRC, MODE_READ)).mode_for(OUTSIDE) == MODE_BLOCKED
    assert FileGrants().mode_for(FILE) == MODE_BLOCKED


# -- narrowing is the point, and it is not symmetric (D16, I6) --------------


def test_narrowing_takes_the_lesser_mode():
    wide = _grants((SRC, MODE_READ_WRITE))
    narrowed = wide.narrow(_grants((SRC, MODE_READ)))
    assert narrowed.mode_for(FILE) == MODE_READ


def test_narrowing_cannot_widen():
    narrow = _grants((SRC, MODE_READ))
    result = narrow.narrow(_grants((SRC, MODE_READ_WRITE)))
    assert result.mode_for(FILE) == MODE_READ, (
        "asking for write against a read grant yields read, not write"
    )


def test_a_path_the_upstream_grant_never_covered_is_not_added():
    result = _grants((SRC, MODE_READ_WRITE)).narrow(
        _grants((OUTSIDE, MODE_READ_WRITE)))
    assert result.mode_for(OUTSIDE) == MODE_BLOCKED
    assert not result.grants_anything


def test_a_downstream_block_survives_narrowing():
    """Otherwise a granted ancestor covers the hole again."""
    result = _grants((SRC, MODE_READ_WRITE)).narrow(
        _grants((SRC, MODE_READ_WRITE), (SECRETS, MODE_BLOCKED)))
    assert result.mode_for(FILE) == MODE_READ_WRITE
    assert result.mode_for(SECRETS) == MODE_BLOCKED


def test_narrowing_by_nothing_changes_nothing():
    """An unwired port must be harmless, not an empty grant."""
    wide = _grants((SRC, MODE_READ_WRITE))
    assert wide.narrow(None).to_dict() == wide.to_dict()


def test_a_root_outside_the_upstream_roots_is_dropped():
    result = _grants((SRC, MODE_READ)).narrow(
        FileGrants(root=OUTSIDE, roots=[OUTSIDE], entries=[]))
    assert OUTSIDE not in result.effective_roots()
    assert result.effective_roots() == [ROOT], "it falls back, never escapes"


def test_a_chain_of_narrowings_only_ever_shrinks():
    """ToolSet → Role → Agent, however the middle links are written."""
    toolset = _grants((ROOT, MODE_READ_WRITE))
    role = toolset.narrow(_grants((SRC, MODE_READ_WRITE)))
    agent = role.narrow(_grants((SRC, MODE_READ), (ROOT, MODE_READ_WRITE)))

    assert agent.mode_for(FILE) == MODE_READ
    assert agent.mode_for("/project/README.md") == MODE_BLOCKED, (
        "the role narrowed to src; the agent cannot re-widen to the root"
    )


# -- interop ---------------------------------------------------------------


def test_the_dict_shape_round_trips():
    grants = _grants((SRC, MODE_READ_WRITE))
    assert FileGrants.coerce(grants.to_dict()).to_dict() == grants.to_dict()


def test_grants_from_paths_is_read_only_by_default():
    grants = grants_from_paths([SRC, FILE], root=ROOT)
    assert [e.mode for e in grants.entries] == [MODE_READ, MODE_READ]
    assert grants.effective_roots() == [ROOT]


def test_the_summary_says_what_a_status_line_needs():
    text = _grants((SRC, MODE_READ_WRITE), (SECRETS, MODE_BLOCKED)).summary()
    assert "read/write" in text and "blocked" in text
    assert FileGrants().summary() == "no paths granted"


# -- the sandbox builder speaks the model ----------------------------------


def test_the_builder_accepts_a_model_and_a_dict(tmp_path):
    (tmp_path / "src").mkdir()
    grants = _grants((str(tmp_path / "src"), MODE_READ_WRITE), root=str(tmp_path))

    from_model = sandbox_from_permissions(grants, None)
    from_dict = sandbox_from_permissions(grants.to_dict(), None)
    for sandbox in (from_model, from_dict):
        assert sandbox.is_allowed(tmp_path / "src" / "a.py")
        assert sandbox.is_writable(tmp_path / "src" / "a.py")


def test_the_ceiling_is_checked_against_a_live_sandbox(tmp_path):
    """I6: the ToolBox's roots are the hard ceiling, not another grant."""
    inside = tmp_path / "project"
    inside.mkdir()
    base = FileToolSandbox(root_dir=str(inside))
    grants = _grants((str(inside), MODE_READ), (str(tmp_path / "elsewhere"),
                                                MODE_READ_WRITE),
                     root=str(inside))

    kept, dropped = split_by_ceiling(grants, base)
    assert [e["path"] for e in kept] == [str(inside)]
    assert len(dropped) == 1


# -- narrowing a live sandbox in place (D16/D18) ---------------------------


def test_restrict_narrows_and_restores(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    sandbox = FileToolSandbox(root_dir=str(tmp_path))
    assert sandbox.is_writable(tmp_path / "other.txt")

    with sandbox.restrict({str(src): MODE_READ}):
        assert sandbox.is_allowed(src / "a.py")
        assert not sandbox.is_writable(src / "a.py"), "read means read"
        assert not sandbox.is_allowed(tmp_path / "other.txt")

    assert sandbox.is_writable(tmp_path / "other.txt"), (
        "the sandbox is a live graph object; the next run gets it back whole"
    )


def test_restrict_cannot_widen(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    sandbox = FileToolSandbox(root_dir=str(tmp_path), write_enabled=False)

    with sandbox.restrict({str(src): MODE_READ_WRITE}):
        assert sandbox.is_allowed(src / "a.py")
        assert not sandbox.is_writable(src / "a.py"), (
            "a read-only sandbox stays read-only however the grant is written"
        )


def test_restrict_cannot_reopen_a_narrower_path_policy(tmp_path):
    src = tmp_path / "src"
    (src / "deep").mkdir(parents=True)
    sandbox = FileToolSandbox(root_dir=str(tmp_path),
                              path_modes={str(src): MODE_READ})

    # Asking for the root, which the policy never granted, yields nothing --
    # not the root.
    with sandbox.restrict({str(tmp_path): MODE_READ_WRITE}):
        assert not sandbox.is_allowed(tmp_path / "other.txt"), (
            "a path the sandbox did not cover is not added by narrowing"
        )
        assert not sandbox.is_allowed(src / "deep" / "a.py"), (
            "and the ask named nothing the policy did grant"
        )

    # Asking within the granted subtree keeps it, at the lesser mode.
    with sandbox.restrict({str(src / "deep"): MODE_READ_WRITE}):
        assert sandbox.is_allowed(src / "deep" / "a.py")
        assert not sandbox.is_writable(src / "deep" / "a.py")


def test_restrict_never_switches_confinement_back_on(tmp_path):
    """D18: the escape hatch is a ToolBox choice and is not inheritable."""
    sandbox = FileToolSandbox(root_dir=str(tmp_path), enabled=False)
    with sandbox.restrict({str(tmp_path): MODE_READ}):
        assert sandbox.enabled is False, "a grant cannot turn confinement on"


def test_restrict_restores_after_an_exception(tmp_path):
    sandbox = FileToolSandbox(root_dir=str(tmp_path))
    before = sandbox.path_modes
    with pytest.raises(RuntimeError):
        with sandbox.restrict({str(tmp_path): MODE_READ}):
            raise RuntimeError("the run failed")
    assert sandbox.path_modes is before


# -- the Agent's end of the chain ------------------------------------------


class _Role:
    def __init__(self, grants):
        self.file_grants = grants


def _agent_grants(inputs, role):
    """What the Agent node does with the two grants that can reach it."""
    return resolve_grants(inputs.get("permissions"),
                          getattr(role, "file_grants", None))


def test_the_agent_composes_the_role_grant_with_its_own_port():
    role = _Role(_grants((ROOT, MODE_READ_WRITE)))
    result = _agent_grants({"permissions": _grants((SRC, MODE_READ))}, role)
    assert result.mode_for(FILE) == MODE_READ
    assert result.mode_for("/project/README.md") == MODE_BLOCKED


def test_the_agent_says_nothing_when_nothing_is_wired():
    assert _agent_grants({}, _Role(None)) is None, (
        "no grant anywhere must leave the toolset's own sandbox alone"
    )


def test_either_side_alone_is_used_as_given():
    only_port = _agent_grants({"permissions": _grants((SRC, MODE_READ))}, _Role(None))
    only_role = _agent_grants({}, _Role(_grants((SRC, MODE_READ))))
    assert only_port.mode_for(FILE) == MODE_READ
    assert only_role.mode_for(FILE) == MODE_READ


def test_an_invalid_grant_grants_nothing_rather_than_falling_back():
    role = _Role(_grants((ROOT, MODE_READ_WRITE)))
    result = _agent_grants({"permissions": "/project"}, role)
    assert result is not None and not result.grants_anything, (
        "widening access because a structure was malformed is the failure "
        "this port was made explicit to prevent"
    )
