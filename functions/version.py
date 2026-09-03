# -*- coding: utf-8 -*-
"""What Silk this is (G12).

A bug report that says "Silk misbehaved" is worth much less than one that
says which Silk. The version in ``pyproject.toml`` moves rarely and says
little; the commit says everything, and only the *Weave* checkout knew it
-- as a submodule pin, which is exactly the wrong place to read it from
when what you have is a log line.

So this reads the checkout's own HEAD, and it reads it the plain way: the
files git writes. No subprocess (a `git` call per import, on a plugin that
may be imported during a graph load, is a cost for nothing), no
dependency on git being installed at all, and no failure that matters --
a source tree with no `.git` is a perfectly good way to ship this, and it
simply has no commit to report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

#: Kept in step with ``pyproject.toml`` by hand. Silk runs in place as a
#: submodule and is never installed, so there is no package metadata to
#: read it back from -- ``importlib.metadata`` would find nothing.
__version__ = "0.1.0"

#: How much of a commit hash is worth showing. Enough to find it, short
#: enough to sit in a log line.
_SHORT = 12

_ROOT = Path(__file__).resolve().parent.parent


def _git_dir(root: Path) -> Optional[Path]:
    """Where this checkout's git data lives, submodules included.

    A submodule's ``.git`` is a *file* holding ``gitdir: <path>``, relative
    to the submodule. That indirection is the whole reason this helper
    exists rather than a ``root / ".git"``.
    """
    candidate = root / ".git"
    if candidate.is_dir():
        return candidate
    try:
        text = candidate.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    target = Path(text[len("gitdir:"):].strip())
    if not target.is_absolute():
        target = (root / target).resolve()
    return target if target.is_dir() else None


def commit(root: Optional[Path] = None) -> str:
    """The full commit hash of this checkout, or ``""`` if unknowable.

    ``""`` is an ordinary answer, not a failure: an exported source tree
    has no HEAD, and reporting the version without a commit is still more
    than reporting nothing.
    """
    git_dir = _git_dir(Path(root) if root is not None else _ROOT)
    if git_dir is None:
        return ""
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not head.startswith("ref:"):
        # Detached HEAD: the hash is right there.
        return head if _is_hash(head) else ""
    ref = head[len("ref:"):].strip()
    try:
        return (git_dir / ref).read_text(encoding="utf-8").strip()
    except OSError:
        pass
    # Packed refs: a repository that has been gc'd keeps no loose ref file.
    try:
        packed = (git_dir / "packed-refs").read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in packed.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref and _is_hash(parts[0]):
            return parts[0]
    return ""


def _is_hash(text: str) -> bool:
    return len(text) >= 7 and all(c in "0123456789abcdef" for c in text.lower())


def version_string() -> str:
    """``silk 0.1.0 (abc123def456)`` -- one line, for a log or a report."""
    head = commit()
    return f"silk {__version__}" + (f" ({head[:_SHORT]})" if head else "")
