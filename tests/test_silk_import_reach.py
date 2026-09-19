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


# ── the wiring ──────────────────────────────────────────────────────────
#
# Everything above tests the check in isolation. What the check is *for*
# is that the user is told, and that is a property of the ToolBox node:
# a warning computed in a worker thread, stashed on the node, and read
# back by on_evaluate_finished when it composes the status line. The
# status line has been reworked since (tool tree, categories, graph
# hint), so the seam is worth pinning rather than assuming.


def _toolbox_node():
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from silk.nodes.toolbox import SilkToolBoxNode

    return SilkToolBoxNode()


def _make_importable(monkeypatch, tmp_path, root):
    """Make *root* look like somewhere Python imports from, and nothing else."""
    from silk.functions import import_reach as ir

    monkeypatch.setattr(sys, "path", [str(root)])
    monkeypatch.setattr(ir, "_weave_root", lambda: None)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "elsewhere"))


def test_an_importable_writable_root_reaches_the_status_line(
    tmp_path, monkeypatch,
):
    """The whole point of the check: the user sees it, not just the log."""
    root = tmp_path / "on_the_path"
    root.mkdir()
    _make_importable(monkeypatch, tmp_path, root)

    node = _toolbox_node()
    # write_file is what *asks* for write access -- there is no second
    # switch -- so ticking it is what makes the root's reach matter.
    result = node.compute(
        {"sandbox_roots": [str(root)], "enabled_tools": ["write_file"]})

    assert node._import_reach, "a writable root on sys.path must be reported"

    shown = []
    monkeypatch.setattr(node._widget_core, "push_display",
                        lambda name, text: shown.append((name, text)))
    # The real compute output rather than a stand-in: the status line
    # asks the toolbox for its catalog, so a sentinel would test the
    # stand-in instead of the seam.
    monkeypatch.setattr(node, "_get_cached_value", result.get)
    node.on_evaluate_finished()

    status = [text for name, text in shown if name == "status"]
    assert status, "on_evaluate_finished must push a status line"
    assert node._import_reach in status[-1], (
        "the warning must survive the status line's rework -- a log line "
        "alone is a warning nobody reads (G21)")


def test_a_read_only_toolbox_says_nothing_about_import_reach(
    tmp_path, monkeypatch,
):
    """No write grant, no deferred authority, no warning.

    The pairing matters: a check that fired on every root would be
    noise, and noise is how the real case gets ignored.
    """
    root = tmp_path / "on_the_path"
    root.mkdir()
    _make_importable(monkeypatch, tmp_path, root)

    node = _toolbox_node()
    node.compute({"sandbox_roots": [str(root)], "enabled_tools": ["read_file"]})

    assert node._import_reach == "", (
        "reading an importable directory grants nothing")
