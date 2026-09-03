# -*- coding: utf-8 -*-
"""Which writable roots are also *importable* (spec D77, G21 residue).

Every file tool is sandboxed. ``import`` is not. Module-level code in a
file the agent wrote runs with the full authority of the Weave process --
network, whole filesystem, the user's keys -- however narrow the sandbox
was when the file was written. So a writable sandbox root that sits
somewhere Python will import from is a *deferred* grant of process
authority, redeemable later by anything that imports.

D77 is the mitigation for the path Silk controls (`load_suite`: always
approve, a floor no preset can lower, validation in a subprocess after
approval). This module is the cheap check for the residue D77 does not
cover: a user who points the sandbox at a directory inside the venv, or
inside Weave, or anywhere on ``sys.path``, has granted more than the
file-permissions UI suggests -- and nothing tells them.

It only ever *reports*. Refusing such a root would break the legitimate
case (an agent developing its own Weave plugin, D76, writes into an
importable tree on purpose); what was missing is that the user is told
which reach they just handed out.
"""
from __future__ import annotations

import site
import sys
import sysconfig
from pathlib import Path
from typing import Iterable, Optional

#: Reason codes, so a caller can phrase the warning its own way.
ON_SYS_PATH = "on sys.path"
IN_ENVIRONMENT = "inside the Python environment"
IN_WEAVE = "inside Weave itself"


def _resolved(path: "str | Path") -> Optional[Path]:
    """*path* as an absolute path, or ``None`` if it is not one.

    Relative input is rejected rather than resolved: it would be resolved
    against the process's working directory, which is not where the user
    pointed the sandbox, and answering a question about the wrong
    directory is worse than not answering it.
    """
    text = str(path).strip()
    if not text:
        return None
    try:
        expanded = Path(text).expanduser()
        if not expanded.is_absolute():
            return None
        return expanded.resolve()
    except (OSError, ValueError, RuntimeError):
        return None


def _contains(parent: Path, child: Path) -> bool:
    """True when *child* is *parent* or lies under it."""
    if parent == child:
        return True
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _import_dirs() -> list[Path]:
    """Directories Python will import from, as far as this process knows.

    ``sys.path`` first, because that is the live answer, and the entries
    ``site``/``sysconfig`` add because a root can be importable *later* --
    a fresh interpreter, or the venv's own site-packages that this process
    happens not to have on its path.
    """
    raw: list[str] = [p for p in sys.path if p]
    for name in ("purelib", "platlib", "scripts", "data"):
        try:
            found = sysconfig.get_path(name)
        except (KeyError, OSError):
            found = None
        if found:
            raw.append(found)
    try:
        raw.extend(site.getsitepackages())
    except AttributeError:  # a virtualenv without it
        pass
    user_site = getattr(site, "getusersitepackages", None)
    if user_site is not None:
        try:
            raw.append(str(user_site()))
        except Exception:  # noqa: BLE001 - site is allowed to be odd
            pass
    out: list[Path] = []
    for entry in raw:
        resolved = _resolved(entry)
        # "" means the current directory, which is not a grant anyone made
        # here; a file on the path (a zip import) cannot be written into.
        if resolved is not None and resolved.is_dir() and resolved not in out:
            out.append(resolved)
    return out


def _weave_root() -> Optional[Path]:
    """Where Weave itself lives -- writing there is editing the app."""
    return _resolved(Path(__file__).resolve().parent.parent.parent.parent)


def importable_reason(root: "str | Path") -> str:
    """Why *root* is importable, or ``""`` if it is not.

    The first reason that applies, most specific first: being inside Weave
    is the sharpest thing to say about a root, and being inside the
    environment is sharper than the generic path entry that implies it.
    """
    resolved = _resolved(root)
    if resolved is None:
        return ""

    weave = _weave_root()
    if weave is not None and _contains(weave, resolved):
        return IN_WEAVE

    prefix = _resolved(sys.prefix)
    base = _resolved(getattr(sys, "base_prefix", sys.prefix))
    for env in (prefix, base):
        if env is not None and _contains(env, resolved):
            return IN_ENVIRONMENT

    for entry in _import_dirs():
        # Either direction counts: a root *under* site-packages is
        # importable as a package, and a root that *contains* an import
        # directory can create files inside it.
        if _contains(entry, resolved) or _contains(resolved, entry):
            return ON_SYS_PATH
    return ""


def importable_roots(roots: Iterable["str | Path"]) -> list[tuple[str, str]]:
    """``(root, reason)`` for each of *roots* Python can import from."""
    found: list[tuple[str, str]] = []
    for root in roots:
        reason = importable_reason(root)
        if reason:
            found.append((str(root), reason))
    return found


def import_reach_warning(roots: Iterable["str | Path"]) -> str:
    """One sentence naming the writable roots that reach the interpreter.

    Empty when there are none, so a caller can treat it as a flag.
    """
    found = importable_roots(roots)
    if not found:
        return ""
    listed = "; ".join(f"{root} ({reason})" for root, reason in found)
    return (
        f"Writable sandbox root {listed} is importable: code written there "
        f"runs with the whole Weave process's authority once anything "
        f"imports it, however narrow the write grant was. "
        if len(found) == 1 else
        f"Writable sandbox roots are importable — {listed}: code written "
        f"there runs with the whole Weave process's authority once anything "
        f"imports it, however narrow the write grant was. "
    ) + "Intended for plugin authoring; a surprise anywhere else."
