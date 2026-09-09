# -*- coding: utf-8 -*-
"""Durable and run-scoped approval grants (spec D10, D34, D35).

Almost every test here is really one claim in two directions: **a grant can
only ever remove a prompt, never add one**. That is what lets the store be
lossy -- unreadable file, invalid record, a racing writer -- without being
unsafe, and it is why "degrades to empty" appears so often below.

The other half is scope. A grant is keyed by the *resolved* project root and
lives outside the sandbox, because a durable "always allow write_file" record
inside a tree the agent can write to is a permission the agent can award
itself.
"""

from __future__ import annotations

import json


from silk.functions.grants import (  # noqa: E402
    FORMAT_VERSION,
    GRANT_DIR,
    SCOPE_ALWAYS,
    SCOPE_ONCE,
    SCOPE_RUN,
    SCOPES,
    Grant,
    GrantStore,
    RunGrants,
)


def _store(tmp_path) -> GrantStore:
    return GrantStore(tmp_path)


# -- the basics ------------------------------------------------------------


def test_an_empty_store_grants_nothing(tmp_path):
    store = _store(tmp_path)
    assert store.allows(tmp_path, "write_file") is False
    assert store.allows(tmp_path, "") is False
    assert store.for_project(tmp_path) == []


def test_a_grant_is_durable(tmp_path):
    _store(tmp_path).grant(tmp_path, "write_file", granted_by="frank")

    fresh = _store(tmp_path)          # a different run, a different object
    assert fresh.allows(tmp_path, "write_file") is True
    assert fresh.tools(tmp_path) == frozenset({"write_file"})
    assert fresh.for_project(tmp_path)[0].granted_by == "frank"


def test_the_file_names_its_format_version(tmp_path):
    _store(tmp_path).grant(tmp_path, "write_file")
    document = json.loads((tmp_path / "grants.json").read_text(encoding="utf-8"))
    assert document["version"] == FORMAT_VERSION
    assert document["grants"][0]["tool_name"] == "write_file"


def test_the_default_location_is_outside_any_project(tmp_path):
    """D34: not the plan database, and not under the sandbox root."""
    assert GRANT_DIR.name == "silk" and GRANT_DIR.parent.name == ".weave"
    assert GRANT_DIR.parent.parent == GRANT_DIR.parent.parent.home()


# -- scope -----------------------------------------------------------------


def test_a_grant_does_not_travel_between_projects(tmp_path):
    scratch, sensitive = tmp_path / "scratch", tmp_path / "sensitive"
    scratch.mkdir()
    sensitive.mkdir()
    store = _store(tmp_path)
    store.grant(scratch, "run_command")

    assert store.allows(scratch, "run_command") is True
    assert store.allows(sensitive, "run_command") is False, (
        "consent given in one project is not consent in another"
    )


def test_the_key_is_the_resolved_root(tmp_path):
    store = _store(tmp_path)
    store.grant(tmp_path, "write_file")
    indirect = tmp_path / "sub" / ".."
    (tmp_path / "sub").mkdir()
    assert store.allows(indirect, "write_file") is True, (
        "the same directory reached by a different path is the same project"
    )


def test_an_unnamed_root_is_its_own_key_not_a_wildcard(tmp_path):
    store = _store(tmp_path)
    store.grant("", "write_file")
    assert store.allows("", "write_file") is True
    assert store.allows(tmp_path, "write_file") is False, (
        "there is no such thing as a grant that applies everywhere"
    )


# -- revocation ------------------------------------------------------------


def test_revoking_deletes_the_record(tmp_path):
    store = _store(tmp_path)
    store.grant(tmp_path, "write_file")

    assert store.revoke(tmp_path, "write_file") is True
    assert store.allows(tmp_path, "write_file") is False
    assert store.revoke(tmp_path, "write_file") is False, "nothing left to revoke"
    # Deletion, not a deny record: the file holds no trace to resurrect.
    document = json.loads((tmp_path / "grants.json").read_text(encoding="utf-8"))
    assert document["grants"] == []


def test_revoking_a_project_clears_only_that_project(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    store = _store(tmp_path)
    for tool in ("write_file", "run_command", "delete_file"):
        store.grant(a, tool)
    store.grant(b, "write_file")

    assert store.revoke_project(a) == 3
    assert store.tools(a) == frozenset()
    assert store.tools(b) == frozenset({"write_file"})


# -- every failure degrades to "ask" ---------------------------------------


def test_an_unparseable_store_reads_as_empty(tmp_path):
    (tmp_path / "grants.json").write_text("{not json at all", encoding="utf-8")
    store = _store(tmp_path)
    assert store.allows(tmp_path, "write_file") is False


def test_a_store_of_the_wrong_shape_reads_as_empty(tmp_path):
    (tmp_path / "grants.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert _store(tmp_path).for_project(tmp_path) == []


def test_an_invalid_record_is_skipped_and_the_rest_survive(tmp_path):
    (tmp_path / "grants.json").write_text(json.dumps({
        "version": FORMAT_VERSION,
        "grants": [
            {"tool_name": "write_file", "project": str(tmp_path.resolve())},
            {"nonsense": True},
        ],
    }), encoding="utf-8")
    store = _store(tmp_path)
    assert store.allows(tmp_path, "write_file") is True
    assert len(store.for_project(tmp_path)) == 1


def test_a_store_that_cannot_be_written_does_not_raise_into_a_run(tmp_path):
    """The cost of a lost grant is one more prompt, not a crashed tool call."""
    blocker = tmp_path / "grants.json"
    blocker.mkdir()                       # a directory where the file goes
    store = GrantStore(tmp_path)
    grant = store.grant(tmp_path, "write_file")
    assert grant.tool_name == "write_file"
    assert GrantStore(tmp_path).allows(tmp_path, "write_file") is False


# -- concurrency -----------------------------------------------------------


def test_a_concurrent_grant_is_not_clobbered(tmp_path):
    """Read-modify-write: the racing store keeps what it did not know about."""
    first, second = _store(tmp_path), _store(tmp_path)
    first.grant(tmp_path, "write_file")
    second.grant(tmp_path, "run_command")     # opened before the first write

    assert _store(tmp_path).tools(tmp_path) == {"write_file", "run_command"}


# -- run scope -------------------------------------------------------------


def test_the_scope_vocabulary():
    assert SCOPES == (SCOPE_ONCE, SCOPE_RUN, SCOPE_ALWAYS)


def test_run_grants_live_and_die_with_the_run():
    run = RunGrants()
    assert run.allows("write_file") is False
    run.allow("write_file")
    assert run.allows("write_file") is True and len(run) == 1
    assert list(run) == ["write_file"]
    assert RunGrants().allows("write_file") is False, "nothing persisted"


def test_run_grants_can_be_seeded():
    assert RunGrants(["a", "b"]).allows("a") is True
    assert len(RunGrants(["a", "a"])) == 1


def test_a_grant_record_stamps_when_it_was_made():
    grant = Grant(tool_name="write_file", project="/p")
    assert grant.granted_at > 0 and grant.granted_by == ""


# -- listing, for a revocation surface (§22 q1) -----------------------------


def test_the_store_lists_the_projects_it_holds(tmp_path):
    """You cannot withdraw what you cannot find."""
    store = _store(tmp_path)
    assert store.projects() == [], "nothing granted, nothing to show"

    a, b = tmp_path / "a", tmp_path / "b"
    store.grant(a, "write_file")
    store.grant(a, "run_command")
    store.grant(b, "write_file")

    assert store.projects() == sorted(
        [GrantStore.project_key(a), GrantStore.project_key(b)]), (
        "one entry per project, not per grant"
    )


def test_an_unnamed_root_is_listed_under_its_own_key(tmp_path):
    """The empty key is a project like any other, never 'everywhere'."""
    store = _store(tmp_path)
    store.grant(None, "write_file")
    assert store.projects() == [""]
    assert store.tools(None) == {"write_file"}
    assert store.tools(tmp_path) == frozenset(), "and it grants nothing else"


def test_the_store_lists_every_grant_newest_first(tmp_path):
    """The question a user starts from is *what did I allow*, lately."""
    store = _store(tmp_path)
    store.grant(tmp_path, "old_tool")
    store.grant(tmp_path, "new_tool")
    store._grants[(GrantStore.project_key(tmp_path), "old_tool")].granted_at = 1.0

    names = [g.tool_name for g in store.all()]
    assert names == ["new_tool", "old_tool"]
    assert all(isinstance(g, Grant) for g in store.all())


def test_listing_reflects_a_revocation(tmp_path):
    """The list is the file, so what a revocation removes stops showing."""
    store = _store(tmp_path)
    store.grant(tmp_path, "write_file")
    assert len(store.all()) == 1
    assert store.revoke(tmp_path, "write_file") is True
    assert store.all() == [] and store.projects() == []
