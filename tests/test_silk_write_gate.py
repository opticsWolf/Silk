# -*- coding: utf-8 -*-
"""Tests for the per-root write gate (spec D67 tier 2, closes G19).

Tier 1 -- the per-path locks -- covers the file tools and nothing else. A
toolchain subprocess that rewrites files never passes through them, so these
tests are about the second tier: a writer holds its whole root, a file write
under that root waits, and read-only commands wait for nothing.

Threads and a short timeout stand in for "did it actually block": a gate that
does not exclude shows up as an event set almost immediately, one that does
shows up as the wait timing out.
"""

from __future__ import annotations

import os
import threading

import pytest


from silk.functions.tools import file_locks
from silk.functions.tools.file_locks import (
    lock_paths,
    register_root,
    registered_roots,
    write_gate,
)
from silk.functions.tools.toolchains import SPEC_PACKS

# How long a thread is given to reach a gate it is *not* expected to be
# blocked by. Generous, because a false pass here would be a lock that never
# excludes anything; a false failure only shows up as a flake on a loaded
# machine.
GRAB = 5.0
# How long we wait to conclude that a thread really is blocked. Short,
# because the whole point is that it never gets through.
BLOCKED = 0.3


@pytest.fixture(autouse=True)
def _clean_registry():
    """The root registry is process-wide; keep tests from leaking into it."""
    roots = set(file_locks._roots)
    gates = dict(file_locks._root_gates)
    yield
    file_locks._roots.clear()
    file_locks._roots.update(roots)
    file_locks._root_gates.clear()
    file_locks._root_gates.update(gates)


def _spec(name: str):
    for pack in SPEC_PACKS.values():
        for spec in pack:
            if spec.tool_name == name:
                return spec
    raise AssertionError(f"no spec named {name}")


# ── registration ──────────────────────────────────────────────────────────


def test_register_root_is_idempotent_and_canonical(tmp_path):
    first = register_root(tmp_path)
    second = register_root(str(tmp_path) + os.sep + "." + os.sep)
    assert first == second == str(tmp_path.resolve())
    assert first in registered_roots()


def test_the_sandbox_registers_its_own_root(tmp_path):
    from silk.functions.tools.file_sandbox import FileToolSandbox

    FileToolSandbox(root_dir=tmp_path)
    assert str(tmp_path.resolve()) in registered_roots()


# ── the gate excludes ─────────────────────────────────────────────────────


def _run_while_gate_held(root, body, *, expect_blocked: bool) -> bool:
    """Hold *root* exclusively, run *body* in a thread, report whether it ran."""
    done = threading.Event()

    def worker():
        body()
        done.set()

    with write_gate(root):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        ran = done.wait(BLOCKED if expect_blocked else GRAB)
    thread.join(GRAB)
    return ran


def test_a_writer_excludes_a_file_write_under_the_same_root(tmp_path):
    register_root(tmp_path)
    target = tmp_path / "a.txt"

    def write():
        with lock_paths(target):
            target.write_text("written")

    assert not _run_while_gate_held(tmp_path, write, expect_blocked=True)
    # And it goes through the moment the gate is dropped.
    with lock_paths(target):
        target.write_text("written")
    assert target.read_text() == "written"


def test_a_writer_ignores_a_path_outside_its_root(tmp_path):
    inside = tmp_path / "root"
    inside.mkdir()
    register_root(inside)
    outside = tmp_path / "elsewhere.txt"

    def write():
        with lock_paths(outside):
            outside.write_text("x")

    assert _run_while_gate_held(inside, write, expect_blocked=False)


def test_a_writer_excludes_a_writer_on_a_nested_root(tmp_path):
    outer = tmp_path / "project"
    inner = outer / "sub"
    inner.mkdir(parents=True)
    register_root(outer)
    register_root(inner)

    def inner_write():
        with write_gate(inner):
            pass

    assert not _run_while_gate_held(outer, inner_write, expect_blocked=True)


def test_two_file_writes_share_the_root(tmp_path):
    """Shared never excludes shared: tier 1 behaviour is unchanged."""
    register_root(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def hold():
        with lock_paths(tmp_path / "a.txt"):
            entered.set()
            release.wait(GRAB)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert entered.wait(GRAB)
    try:
        done = threading.Event()

        def other():
            with lock_paths(tmp_path / "b.txt"):
                done.set()

        second = threading.Thread(target=other, daemon=True)
        second.start()
        assert done.wait(GRAB)
        second.join(GRAB)
    finally:
        release.set()
        thread.join(GRAB)


def test_write_gate_on_none_is_a_no_op():
    with write_gate(None):
        pass


def test_the_gate_is_released_when_the_body_raises(tmp_path):
    register_root(tmp_path)
    with pytest.raises(RuntimeError):
        with write_gate(tmp_path):
            raise RuntimeError("boom")
    # Not deadlocked: a second acquisition still succeeds.
    with write_gate(tmp_path):
        pass


# ── which commands declare themselves writers ─────────────────────────────


@pytest.mark.parametrize(
    "name", ["run_python", "ruff_format", "cargo_build", "cargo_fmt"]
)
def test_writing_commands_are_declared(name):
    assert _spec(name).writes_files is True


@pytest.mark.parametrize("name", ["ruff_check", "mypy_check", "radon_cc"])
def test_read_only_commands_take_no_gate(name):
    assert _spec(name).writes_files is False


# -- the precondition, where ownership was rejected (§22 q8) ---------------
#
# q8 asked whether the sandbox should consult ledger *claims* as dynamic
# write policy -- deny a write to a path another agent claimed. It does
# not: a permission that depends on another agent's runtime state is one
# no human configured, none can see in the node UI, and an agent could
# create by claiming. The lost-update case it was meant to answer is
# handled here instead, the way D68 already sanctioned: an optimistic
# precondition the caller opts into.


def _sandbox(tmp_path):
    from silk.functions.tools.file_sandbox import FileToolSandbox

    return FileToolSandbox(root_dir=str(tmp_path), write_enabled=True)


def _write(tmp_path, name, content, expected=""):
    from silk.functions.tools.file_write import _write_file_impl

    return _write_file_impl(_sandbox(tmp_path), name, content, expected)


def _sha(text):
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_a_write_with_no_precondition_still_just_writes(tmp_path):
    assert "Successfully wrote" in _write(tmp_path, "a.txt", "one")
    assert "Successfully wrote" in _write(tmp_path, "a.txt", "two")
    assert (tmp_path / "a.txt").read_text() == "two"


def test_a_matching_digest_lets_the_write_through(tmp_path):
    _write(tmp_path, "a.txt", "one")
    out = _write(tmp_path, "a.txt", "two", _sha("one"))
    assert "Successfully wrote" in out
    assert (tmp_path / "a.txt").read_text() == "two"


def test_a_changed_file_refuses_the_write_and_says_so(tmp_path):
    """The blind-overwrite hole D68 left open, closed at the caller's option."""
    _write(tmp_path, "a.txt", "one")
    stale = _sha("one")
    _write(tmp_path, "a.txt", "someone else got here first")

    out = _write(tmp_path, "a.txt", "mine", stale)
    assert "does not match the precondition" in out
    assert "Read it again" in out, "a refusal the model can act on"
    assert (tmp_path / "a.txt").read_text() == "someone else got here first", (
        "the other agent's work is still there"
    )


def test_absent_means_create_only(tmp_path):
    assert "Successfully wrote" in _write(tmp_path, "new.txt", "one", "absent")
    out = _write(tmp_path, "new.txt", "again", "absent")
    assert "already exists" in out
    assert (tmp_path / "new.txt").read_text() == "one"


def test_a_precondition_on_a_file_that_vanished_says_which_way_it_failed(tmp_path):
    out = _write(tmp_path, "gone.txt", "content", _sha("one"))
    assert "does not exist" in out
    assert not (tmp_path / "gone.txt").exists(), (
        "a failed precondition writes nothing, including a new file"
    )


def test_the_check_happens_inside_the_lock_the_write_takes(tmp_path):
    """Compare-and-swap, not compare-then-hope.

    If the digest were read before the lock was taken, another writer
    could land between the two and the precondition would pass over a
    file it had never seen.
    """
    import contextlib

    import silk.functions.tools.file_write as fw
    from silk.functions.tools.file_write import _write_file_impl

    _write(tmp_path, "a.txt", "one")
    sandbox = _sandbox(tmp_path)
    order = []

    real_lock = sandbox.lock_paths

    @contextlib.contextmanager
    def spy_lock(*paths):
        order.append("lock")
        with real_lock(*paths):
            yield
        order.append("unlock")

    sandbox.lock_paths = spy_lock

    real_digest, real_atomic = fw._digest, fw._atomic_write

    def spy_digest(path):
        order.append("digest")
        return real_digest(path)

    def spy_atomic(path, data):
        order.append("write")
        return real_atomic(path, data)

    fw._digest, fw._atomic_write = spy_digest, spy_atomic
    try:
        _write_file_impl(sandbox, "a.txt", "two", _sha("one"))
    finally:
        fw._digest, fw._atomic_write = real_digest, real_atomic

    assert order == ["lock", "digest", "write", "unlock"]
    assert (tmp_path / "a.txt").read_text() == "two"
