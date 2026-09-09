# -*- coding: utf-8 -*-
"""Which Silk is this (G12).

The package had no version and no commit, so a log line or a bug report
could say what went wrong but never in which build. Only the Weave
checkout knew — as a submodule pin, which is precisely the thing you do
not have when what you have is a log file.

The commit is read out of the files git writes rather than by calling
git: this is imported during a graph load, a subprocess per import is a
cost for nothing, and an exported source tree with no `.git` must still
import. So the cases worth pinning are the shapes on disk — a submodule's
`.git` *file*, a detached HEAD, a packed ref — and the one that returns
an empty string without complaining.
"""

from __future__ import annotations

import os


from silk.functions import version as v

HASH = "3770e2d8dcf0aaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _repo(root, git_dir_name=".git"):
    """A checkout whose git data sits in a directory, as a plain clone's does."""
    git = root / git_dir_name
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/dev\n", encoding="utf-8")
    (git / "refs" / "heads" / "dev").write_text(HASH + "\n", encoding="utf-8")
    return git


def test_a_plain_checkout_reports_its_head(tmp_path):
    _repo(tmp_path)

    assert v.commit(tmp_path) == HASH


def test_a_submodule_follows_the_gitdir_file(tmp_path):
    """Silk *is* a submodule, so this is the case that actually runs."""
    real = tmp_path / "parent" / ".git" / "modules" / "silk"
    real.mkdir(parents=True)
    _repo(real.parent, "silk")           # the git data, one level in
    work = tmp_path / "work"
    work.mkdir()
    (work / ".git").write_text(
        f"gitdir: {os.path.relpath(real.parent / 'silk', work)}\n",
        encoding="utf-8",
    )

    assert v.commit(work) == HASH


def test_a_detached_head_is_the_hash_itself(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text(HASH + "\n", encoding="utf-8")

    assert v.commit(tmp_path) == HASH


def test_a_packed_ref_is_found_when_the_loose_one_is_gone(tmp_path):
    """A gc'd repository keeps no file at `refs/heads/dev`."""
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/dev\n", encoding="utf-8")
    (git / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted \n"
        f"{HASH} refs/heads/dev\n",
        encoding="utf-8",
    )

    assert v.commit(tmp_path) == HASH


def test_no_git_at_all_is_an_answer_not_a_failure(tmp_path):
    """An exported source tree is a supported way to have Silk."""
    assert v.commit(tmp_path) == ""


def test_junk_where_a_hash_should_be_is_not_reported_as_one(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("not a hash\n", encoding="utf-8")

    assert v.commit(tmp_path) == ""


def test_the_version_string_survives_having_no_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "_ROOT", tmp_path)

    assert v.version_string() == f"silk {v.__version__}"


def test_the_version_string_carries_a_short_commit_when_there_is_one(
    tmp_path, monkeypatch,
):
    _repo(tmp_path)
    monkeypatch.setattr(v, "_ROOT", tmp_path)

    assert v.version_string() == f"silk {v.__version__} ({HASH[:12]})"


def test_the_declared_version_matches_pyproject():
    """Two hand-kept numbers, so the test is the thing keeping them equal."""
    from pathlib import Path

    text = (Path(v.__file__).parent.parent / "pyproject.toml").read_text(
        encoding="utf-8")
    declared = [line for line in text.splitlines()
                if line.startswith("version = ")]

    assert declared and declared[0] == f'version = "{v.__version__}"'
