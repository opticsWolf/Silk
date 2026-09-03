# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

File grants: what an agent may read and write, as data (spec D16/D17).

`file_permissions` has been a dict described only in a docstring: ``{"root":
str, "roots": [...], "entries": [{"path", "mode"}]}``. Every consumer
re-derived the same three facts from it -- which roots survive, what mode a
path ends up with, whether anything is granted at all -- and a malformed
structure was discovered by the sandbox behaving oddly rather than by the
port refusing it. This module makes the structure a model and the
derivations methods (D17).

**Narrowing is the whole point (D16, I6).** A grant travels ToolSet → Role
→ Agent as a visible port, and each layer may only *narrow* what it
received. So the operation that matters is not "merge" or "override" but
:meth:`FileGrants.narrow`, and it is deliberately pessimistic in three
ways at once:

* a path the receiver does not already cover is **not** granted, however
  the narrowing set describes it;
* where both cover a path, the **lesser** mode wins;
* a root outside the receiver's roots is dropped rather than added.

None of those is symmetric, which is the point: `a.narrow(b)` is what `a`
is willing to hand on when `b` asks, never what `b` would like to have.
The hard ceiling one level further out -- the ToolBox's own sandbox roots
-- is applied separately in ``toolset_build.split_by_ceiling``, because it
is enforced against a live sandbox rather than against another grant.

**The escape hatch is not in here (D18).** Whether confinement is on at all
is a ToolBox-level toggle. A grant can only ever describe paths inside a
sandbox; there is no representable value of this model that turns
confinement off, which is what keeps I6 true by construction rather than by
review.
"""
from __future__ import annotations

from pathlib import PurePath
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from weave.logger import get_logger

log = get_logger("SilkGrants")

#: The three modes, in ascending order of access. ``blocked`` is
#: representable on purpose: an explicit block on a path beats a granted
#: ancestor, which is how a subtree is carved out of a larger grant.
MODE_BLOCKED = "blocked"
MODE_READ: Literal["read"] = "read"
MODE_READ_WRITE = "read_write"

MODES = (MODE_BLOCKED, MODE_READ, MODE_READ_WRITE)

#: Ascending access order; the shared definition of "lesser mode".
MODE_ORDER: dict[str, int] = {MODE_BLOCKED: 0, MODE_READ: 1, MODE_READ_WRITE: 2}


def lesser_mode(a: str, b: str) -> str:
    """The more restrictive of two modes (unknown modes read as blocked)."""
    return a if MODE_ORDER.get(a, 0) <= MODE_ORDER.get(b, 0) else b


def _covers(ancestor: str, path: str) -> bool:
    """Whether *ancestor* is *path* or one of its parents.

    String-normalised rather than resolved: a grant is data that may name a
    path this process cannot stat (a file that does not exist yet, a root on
    a machine the graph was authored on). Resolution belongs to the
    sandbox, which does it against the filesystem at call time.
    """
    try:
        a = PurePath(str(ancestor))
        p = PurePath(str(path))
    except (TypeError, ValueError):
        return False
    return a == p or a in p.parents


class FileGrant(BaseModel):
    """One path and what may be done to it."""

    path: str
    mode: Literal["blocked", "read", "read_write"] = MODE_READ

    model_config = {"extra": "ignore"}

    @field_validator("path")
    @classmethod
    def _path_is_not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("a file grant needs a path")
        return text

    @property
    def grants_access(self) -> bool:
        return self.mode in (MODE_READ, MODE_READ_WRITE)


class FileGrants(BaseModel):
    """A complete file-access grant: roots, plus per-path modes.

    Hierarchical, like the sandbox it builds: a directory entry covers its
    subtree, and the nearest entry to a path wins -- so a ``blocked`` entry
    carves a hole in a granted directory, and a file created later inherits
    the nearest granted directory rather than being invisible until someone
    re-selects it.
    """

    #: The primary root. Kept alongside ``roots`` because the port has
    #: always carried both and saved graphs use either.
    root: str = ""
    roots: list[str] = Field(default_factory=list)
    entries: list[FileGrant] = Field(default_factory=list)

    model_config = {"extra": "ignore"}

    # -- construction -------------------------------------------------------

    @classmethod
    def coerce(cls, value: Any) -> Optional["FileGrants"]:
        """Accept a model, a dict, or ``None``; reject anything else.

        The port boundary calls this (D17). ``None`` means "no grant
        travelled", which is different from an empty grant: no grant leaves
        the upstream sandbox alone, an empty one grants nothing.
        """
        if value is None:
            return None
        if isinstance(value, FileGrants):
            return value
        if isinstance(value, dict):
            return cls.model_validate(value)
        raise TypeError(
            f"file grants must be a dict or FileGrants, not {type(value).__name__}"
        )

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """Whether *value* is a usable grant (or the absence of one)."""
        try:
            cls.coerce(value)
        except Exception:      # noqa: BLE001 - a port validator answers yes/no
            return False
        return True

    # -- derivations --------------------------------------------------------

    def effective_roots(self) -> list[str]:
        """``roots`` if given, else ``root``, else nothing."""
        found = [str(r) for r in self.roots if str(r).strip()]
        if not found and str(self.root).strip():
            found = [str(self.root)]
        return found

    def mode_map(self) -> dict[str, str]:
        """``{path: mode}`` for every entry, last entry winning on a tie."""
        return {e.path: e.mode for e in self.entries}

    def mode_for(self, path: Any) -> str:
        """The mode this grant gives *path*, by nearest covering entry.

        Ties go to the longest match, which is what "nearest" means: an
        explicit entry beats its own directory, and a directory beats its
        parent. Nothing covering it at all is ``blocked``.
        """
        target = str(path)
        best: Optional[str] = None
        best_len = -1
        for entry in self.entries:
            if _covers(entry.path, target) and len(entry.path) > best_len:
                best, best_len = entry.mode, len(entry.path)
        return best if best is not None else MODE_BLOCKED

    @property
    def grants_anything(self) -> bool:
        """Whether any path is readable at all."""
        return any(e.grants_access for e in self.entries)

    # -- the operation this model exists for (D16, I6) ----------------------

    def narrow(self, other: Any) -> "FileGrants":
        """This grant restricted by *other*; never widened by it.

        ``other`` is what the downstream layer asks for. Every path it names
        is granted at most what *this* grant already gave it, and a path
        this grant does not cover is not granted at all -- so a chain of
        narrowings is monotonically decreasing however the middle links are
        written (I6).

        ``None`` means "asks for nothing in particular" and returns this
        grant unchanged, which is what makes an unwired port harmless.
        """
        requested = FileGrants.coerce(other)
        if requested is None:
            return self.model_copy(deep=True)

        entries: list[FileGrant] = []
        for entry in requested.entries:
            allowed = self.mode_for(entry.path)
            mode = lesser_mode(allowed, entry.mode)
            if mode == MODE_BLOCKED:
                continue                    # not covered upstream: no grant
            entries.append(FileGrant(path=entry.path, mode=mode))

        # An explicit block downstream stays a block: it is a narrowing, and
        # dropping it would let a granted ancestor cover the hole again.
        entries.extend(
            FileGrant(path=e.path, mode=MODE_BLOCKED)
            for e in requested.entries if e.mode == MODE_BLOCKED
        )

        mine = self.effective_roots()
        roots = [
            r for r in requested.effective_roots()
            if not mine or any(_covers(m, r) for m in mine)
        ] or mine
        return FileGrants(root=roots[0] if roots else "", roots=roots,
                          entries=entries)

    # -- interop ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The plain-dict shape the port has always carried."""
        return {
            "root": self.root,
            "roots": list(self.roots),
            "entries": [{"path": e.path, "mode": e.mode} for e in self.entries],
        }

    def summary(self) -> str:
        """One line for a node's status readout."""
        if not self.entries:
            return "no paths granted"
        writable = sum(1 for e in self.entries if e.mode == MODE_READ_WRITE)
        readable = sum(1 for e in self.entries if e.mode == MODE_READ)
        blocked = sum(1 for e in self.entries if e.mode == MODE_BLOCKED)
        parts = [f"{readable} read", f"{writable} read/write"]
        if blocked:
            parts.append(f"{blocked} blocked")
        return ", ".join(parts)


def resolve_grants(direct: Any, inherited: Any) -> Optional[FileGrants]:
    """Compose the two ways a grant reaches an agent (spec D16).

    ``inherited`` travelled down the ToolSet → Role chain; ``direct`` is
    wired straight to the node. They compose by narrowing rather than by one
    winning, so adding a wire can only ever reduce access (I6). ``None``
    from both means "nothing was said", which leaves the toolset's own
    sandbox exactly as it was -- an unwired port must be harmless.

    An unparseable grant yields an **empty** grant, not the wider one it
    failed to parse: silently widening access because a structure was
    malformed is the failure this port was made explicit to prevent.
    """
    try:
        asked = FileGrants.coerce(direct)
    except Exception as exc:      # noqa: BLE001 - a bad grant is data
        log.warning(f"Ignoring an invalid file grant: {exc}")
        return FileGrants()
    try:
        upstream = FileGrants.coerce(inherited)
    except Exception as exc:      # noqa: BLE001
        log.warning(f"Ignoring an invalid inherited file grant: {exc}")
        return FileGrants()

    if upstream is None:
        return asked
    if asked is None:
        return upstream
    return upstream.narrow(asked)


def grants_from_paths(
    paths: Iterable[Any], *, mode: str = MODE_READ, root: str = "",
) -> FileGrants:
    """A grant over *paths*, all at one mode -- the common simple case."""
    entries = [FileGrant(path=str(p), mode=mode) for p in paths if str(p).strip()]
    roots = [root] if root else []
    return FileGrants(root=root, roots=roots, entries=entries)
