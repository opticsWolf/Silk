# -*- coding: utf-8 -*-
"""Seeing and withdrawing what was granted (real Qt, subprocess) — §22 q1.

D10 made "don't ask again" durable and D35 put it outside the graph; this
is the half that faces the person. An allowance you cannot find is an
allowance you cannot take back, so what is pinned here is that the dock
shows what is *on disk*, that revoking really deletes, and that nothing
in this surface can create authority.
"""
from __future__ import annotations

import os
import subprocess
import sys

import _qt_subprocess  # noqa: F401  (bootstraps silk + weave for the plain-script run)


def _run_checks() -> None:

    import tempfile
    from pathlib import Path

    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication.instance() or QApplication([])  # noqa: F841

    from silk.functions.grants import GrantStore
    from silk.functions.suite_pins import PinStore
    from silk.widgets.grant_manager import (
        EMPTY_TEXT, GrantManagerDock,
    )

    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp))
        pins = PinStore(Path(tmp))
        window = QMainWindow()
        dock = GrantManagerDock.attach(window, store=store, pins=pins)

        # ── nothing granted says so ──
        assert not dock._empty.isHidden()
        assert dock._empty.text() == EMPTY_TEXT
        assert len(dock._rows) == 1, "only the 'no plugin loads itself' line"
        print("PASS an empty store says every call will ask")

        project_a = str(Path(tmp) / "project_a")
        project_b = str(Path(tmp) / "project_b")
        store.grant(project_a, "write_file", granted_by="me")
        store.grant(project_a, "run_command", granted_by="me")
        store.grant(project_b, "write_file", granted_by="me")

        # ── the dock reads the file, not a cache ──
        dock.refresh()
        assert len(dock._rows) == 6, (
            "two project headers, three grants, one plugin header -- and "
            "none of it came "
            "from this process's memory: another window's grant is "
            "visible here"
        )
        print("PASS the dock lists what is on disk")

        # ── revoking one deletes one ──
        assert dock.revoke(project_a, "write_file")
        assert not dock.revoke(project_a, "write_file"), (
            "and revoking again is honest about there being nothing left"
        )
        assert GrantStore(Path(tmp)).tools(project_a) == frozenset(
            {"run_command"}), "the deletion reached the file"
        print("PASS revoking a grant removes it from disk")

        # ── revoking a project takes only that project ──
        counted = []
        dock.revoked.connect(counted.append)
        assert dock.revoke_project(project_a) == 1
        assert counted == [1]
        fresh = GrantStore(Path(tmp))
        assert fresh.tools(project_a) == frozenset()
        assert fresh.tools(project_b) == frozenset({"write_file"}), (
            "a grant made in one project never applied to another, and "
            "revocation respects the same boundary"
        )
        print("PASS revoking a project leaves the others alone")

        # ── the surface can only remove ──
        verbs = [name for name in dir(dock)
                 if not name.startswith("_") and callable(getattr(dock, name))]
        assert "grant" not in verbs, (
            "nothing in a revocation surface may create authority: the "
            "only way to grant is to answer a prompt (D10)"
        )
        assert dock.revoke_project(project_b) == 1
        dock.refresh()
        assert len(dock._rows) == 1 and not dock._empty.isHidden()
        print("PASS the dock ends empty, and can only ever remove")

        # ── the plugin approvals are here too, and can be taken back ──
        suite = Path(tmp) / "my_nodes"
        suite.mkdir()
        (suite / "__init__.py").write_text("VALUE = 1" + chr(10),
                                           encoding="utf-8")
        assert pins.pin("my_nodes", suite, pinned_by="approval") is not None

        dock.refresh()
        assert len(dock._rows) == 2, "the plugin header and its one row"
        assert dock.revoke_pin("my_nodes") is True
        assert not dock.revoke_pin("my_nodes")
        assert PinStore(Path(tmp)).names() == [], "the file, not a cache"
        assert suite.is_dir(), (
            "withdrawing an approval is not deleting the user's code: it "
            "puts the suite back in front of the load prompt"
        )
        dock.refresh()
        assert len(dock._rows) == 1
        print("PASS a plugin approved to self-load can be withdrawn")

        # ── run-scoped grants are not here to be shown ──
        from silk.functions.grants import RunGrants

        run = RunGrants(["write_file"])
        assert run.allows("write_file")
        assert store.all() == [], (
            "a run grant lives in a gate closure and dies with the run; "
            "listing it would invite revoking something already gone"
        )
        print("PASS run-scoped grants stay out of the durable surface")


def test_grant_manager():
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        f"Grant manager check failed (exit {proc.returncode}):\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


if __name__ == "__main__":
    _run_checks()
    print("PASS test_grant_manager")
