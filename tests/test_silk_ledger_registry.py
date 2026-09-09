# -*- coding: utf-8 -*-
"""One handle per ledger file, process-wide (D62).

Macrame has one Write Actor per open handle, and the library does *not*
refuse a second open — two handles on one file just race. The sole-writer
rule therefore has to be enforced at the only place that can be complete:
the registry every caller goes through. These tests pin that, the
refcount that lets several agents share one root's ledger, and the
fallback that keeps a missing extra from being a crash.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from silk.functions import ledger as ledger_mod
from silk.functions.ledger import (
    DISTRIBUTION, LedgerRegistry, LedgerUnavailable, ledger_path,
    unavailable_reason,
)

pytestmark = pytest.mark.skipif(
    not ledger_mod.available(),
    reason=f"the {DISTRIBUTION} extra is not installed",
)


@pytest.fixture
def registry():
    reg = LedgerRegistry()
    yield reg
    reg.close_all()


def _open(reg, path):
    return reg.acquire(path, snapshot_every_entries=None)


# ── the sole-writer rule ─────────────────────────────────────────────────


def test_one_file_is_one_handle(registry, tmp_path):
    path = ledger_path(tmp_path)
    assert _open(registry, path) is _open(registry, path), (
        "two handles on one ledger is two write actors racing; the "
        "library permits it, so the registry must not"
    )
    assert len(registry) == 1


def test_two_spellings_of_one_file_are_one_handle(registry, tmp_path):
    path = ledger_path(tmp_path)
    roundabout = Path(str(tmp_path)) / "sub" / ".." / path.name
    assert _open(registry, path) is _open(registry, roundabout), (
        "a rule enforced against spellings rather than files is no rule"
    )


def test_two_files_are_two_handles(registry, tmp_path):
    first = _open(registry, ledger_path(tmp_path, "a"))
    second = _open(registry, ledger_path(tmp_path, "b"))
    assert first is not second and len(registry) == 2


# ── sharing ──────────────────────────────────────────────────────────────


def test_a_second_agent_shares_the_handle_and_keeps_it_alive(registry, tmp_path):
    path = ledger_path(tmp_path)
    handle = _open(registry, path)
    _open(registry, path)
    assert registry.refs(path) == 2

    assert registry.release(path) == 1
    assert registry.get(path) is handle, (
        "one run ending must not take the write actor from the agents "
        "still using it"
    )


def test_releasing_the_last_reference_still_leaves_it_open(registry, tmp_path):
    path = ledger_path(tmp_path)
    _open(registry, path)
    assert registry.release(path) == 0
    assert registry.is_open(path), (
        "close writes a final snapshot; paying that per run would make a "
        "graph that runs in a loop pay for it every time"
    )


def test_releasing_something_never_opened_is_not_an_error(registry, tmp_path):
    assert registry.release(ledger_path(tmp_path)) == 0


# ── looking, without opening ─────────────────────────────────────────────


def test_lookup_answers_without_opening(registry, tmp_path):
    path = ledger_path(tmp_path)
    assert registry.get(path) is None and not registry.is_open(path)
    assert len(registry) == 0, "asking must not be a way to open"

    handle = _open(registry, path)
    assert registry.get(path) is handle, (
        "this is what turns the hub's file scan into discovery plus "
        "lookup rather than a second open (D62 amendment to D58)"
    )


def test_paths_lists_what_is_open(registry, tmp_path):
    _open(registry, ledger_path(tmp_path, "a"))
    _open(registry, ledger_path(tmp_path, "b"))
    assert len(registry.paths()) == 2
    assert all(name.endswith(".macrame") for name in registry.paths())


# ── closing is the registry's job ────────────────────────────────────────


def test_close_all_closes_everything(registry, tmp_path):
    _open(registry, ledger_path(tmp_path, "a"))
    _open(registry, ledger_path(tmp_path, "b"))
    assert registry.close_all() == 2 and len(registry) == 0
    assert registry.close_all() == 0, "closing twice is not an event"


def test_closing_one_leaves_the_others(registry, tmp_path):
    kept = _open(registry, ledger_path(tmp_path, "keep"))
    assert registry.close(ledger_path(tmp_path, "drop")) is False
    _open(registry, ledger_path(tmp_path, "drop"))
    assert registry.close(ledger_path(tmp_path, "drop")) is True
    assert registry.get(ledger_path(tmp_path, "keep")) is kept


def test_a_closed_handle_is_reopened_not_handed_out(registry, tmp_path):
    path = ledger_path(tmp_path)
    handle = _open(registry, path)
    handle.close()          # something closed it behind the registry's back
    assert registry.get(path) is None, "a dead handle is not an open ledger"

    replacement = _open(registry, path)
    assert replacement is not handle and registry.is_open(path)


# ── the extra is optional, and says so ───────────────────────────────────


def test_a_ledger_lives_under_its_sandbox_root(tmp_path):
    assert ledger_path(tmp_path).parent == tmp_path, (
        "one ledger per sandbox root keeps T4/D58 file discovery working"
    )


def test_the_missing_extra_is_a_refusal_with_a_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger_mod, "_macrame", None)
    monkeypatch.setattr(ledger_mod, "_IMPORT_ERROR", None)
    reason = unavailable_reason()
    assert DISTRIBUTION in reason and "SQLite task store" in reason, (
        "D66: absent, the graph degrades to today's behaviour loudly"
    )
    with pytest.raises(LedgerUnavailable):
        LedgerRegistry().acquire(ledger_path(tmp_path))


def test_the_extra_is_declared(tmp_path):
    """G5: 'install the extra' is not advice without a name and a floor."""
    pyproject = (Path(ledger_mod.__file__).parent.parent / "pyproject.toml")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert any(DISTRIBUTION in spec for spec in extras["ledger"])
    assert data["project"]["dependencies"], "the runtime floor is declared too"
