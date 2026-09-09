# -*- coding: utf-8 -*-
"""A writable root that Python can import from (G21 residue, D77).

Every file tool is sandboxed; `import` is not. Module-level code in a file
the agent wrote runs with the whole Weave process's authority — network,
whole filesystem, the user's keys — however narrow the sandbox was when
the file was written. So pointing the sandbox at a directory inside the
venv, inside Weave, or anywhere on `sys.path` grants far more than the
file-permissions UI suggests, and until now nothing said so.

What is pinned here is that the check *reports* and never refuses. The
legitimate case is real: an agent authoring its own Weave plugin (D76)
writes into an importable tree on purpose. The defect was silence, not
permissiveness.
"""

from __future__ import annotations

import os
import sys


from silk.functions import import_reach as ir


def test_an_ordinary_project_directory_is_not_importable(tmp_path):
    assert ir.importable_reason(tmp_path) == ""
    assert ir.import_reach_warning([tmp_path]) == ""


def test_a_directory_inside_weave_is_named_as_such():
    """The sharpest thing to say about a root: writing there edits the app."""
    weave = os.path.dirname(os.path.dirname(os.path.abspath(
        ir.__file__)))            # .../silk/functions -> .../silk

    assert ir.importable_reason(weave) == ir.IN_WEAVE


def test_a_directory_on_sys_path_is_caught(tmp_path, monkeypatch):
    root = tmp_path / "on_the_path"
    root.mkdir()
    monkeypatch.setattr(sys, "path", [str(root)])
    monkeypatch.setattr(ir, "_weave_root", lambda: None)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "elsewhere"))

    assert ir.importable_reason(root) == ir.ON_SYS_PATH


def test_a_root_containing_an_import_directory_counts_too(tmp_path, monkeypatch):
    """Not only "inside": a root *above* site-packages can create files in it."""
    inner = tmp_path / "env" / "lib" / "site-packages"
    inner.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", [str(inner)])
    monkeypatch.setattr(ir, "_weave_root", lambda: None)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "elsewhere"))

    assert ir.importable_reason(tmp_path / "env") == ir.ON_SYS_PATH


def test_the_environment_is_reported_before_the_generic_path_entry(
    tmp_path, monkeypatch,
):
    env = tmp_path / "venv"
    (env / "Lib" / "site-packages").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(env))
    monkeypatch.setattr(sys, "base_prefix", str(env))
    monkeypatch.setattr(sys, "path", [str(env / "Lib" / "site-packages")])
    monkeypatch.setattr(ir, "_weave_root", lambda: None)

    # Both reasons are true; the one that names the environment is the one
    # a user can act on.
    assert ir.importable_reason(env / "Lib") == ir.IN_ENVIRONMENT


def test_a_relative_or_empty_root_is_not_a_finding():
    """Resolving it would answer about the process cwd, not about the root."""
    assert ir.importable_reason('') == ''
    assert ir.importable_reason('   ') == ''
    assert ir.importable_reason('some/relative/dir') == ''


def test_the_warning_names_every_offending_root_and_why(tmp_path, monkeypatch):
    a = tmp_path / "a"
    b = tmp_path / "b"
    for path in (a, b):
        path.mkdir()
    monkeypatch.setattr(sys, "path", [str(a), str(b)])
    monkeypatch.setattr(ir, "_weave_root", lambda: None)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "elsewhere"))

    warning = ir.import_reach_warning([a, b, tmp_path / "c"])

    assert str(a) in warning and str(b) in warning
    assert str(tmp_path / "c") not in warning
    # It says what the reach *is*, not merely that something is wrong.
    assert "authority" in warning
    assert "plugin authoring" in warning


def test_roots_are_reported_as_pairs_for_a_caller_that_wants_its_own_words(
    tmp_path, monkeypatch,
):
    root = tmp_path / "a"
    root.mkdir()
    monkeypatch.setattr(sys, "path", [str(root)])
    monkeypatch.setattr(ir, "_weave_root", lambda: None)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "elsewhere"))

    assert ir.importable_roots([root, tmp_path / "b"]) == [
        (str(root), ir.ON_SYS_PATH)
    ]
