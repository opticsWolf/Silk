# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Durable per-tool approval grants (spec D10, D34, D35).

"Always allow ``write_file`` in this project" has to survive the run that
granted it, and the three constraints on where it lives each come from a
distinct failure:

**Not the plan database.** The plan store exists only when the task tools
are mounted, and both readers of it fall through when it is absent. An
agent with file tools and *no* planning tools is exactly the configuration
that most needs a gate; a gate that is not there when planning is off is
not a gate. So the tool-call domain never consults the task store.

**Not under the sandbox root.** The plan store is rooted at the sandbox
root -- inside the tree the agent can write to. A durable "always allow
``write_file``" record is precisely the thing an agent must not be able to
author. Grants are user-scoped, with the resolved project root as the
lookup key: scoped *to* the project, stored *outside* it.

**Allow-only.** A grant is a record that exists; revoking one deletes it.
There are no deny records, so a missing, corrupt or unreadable store
degrades to *nothing is granted* -- ask every time -- which is the safe
direction. A store that held denials would resurrect revoked permissions
on data loss. Every failure path here leads to more prompting, never less.

The mechanics deliberately copy :mod:`.presets`, which already owns
``~/.weave/`` -- a ``version`` field, a Pydantic model per record,
``mkdir(parents=True)`` then ``write_text`` on flush, and a reload that
treats a missing or unparseable file as empty. What a grant must *not* do
is become a preset: presets are made to be shared and copied between
projects, and a grant that travels is consent nobody gave.

**Concurrency: read-modify-write, last writer wins.** Two Agent nodes
granting at once can clobber one another, and that is acceptable precisely
because the records are allow-only -- the lost grant costs one extra
prompt. It is the one place Silk does not follow the task store's
optimistic-concurrency habit, and the reason is that the failure is benign
in the safe direction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, Field, ValidationError

from weave.logger import get_logger

log = get_logger("SilkGrants")

#: Sibling of ``~/.weave/presets``; deliberately *not* inside it.
GRANT_DIR = Path.home() / ".weave" / "silk"

GRANT_FILE = "grants.json"

FORMAT_VERSION = 1

#: The scopes a decision can produce (D10). ``once`` is not a grant at all
#: -- it is the absence of one -- and is listed so a UI has a name for the
#: default button.
SCOPE_ONCE = "once"
SCOPE_RUN = "run"
SCOPE_ALWAYS = "always"
SCOPES = (SCOPE_ONCE, SCOPE_RUN, SCOPE_ALWAYS)


class Grant(BaseModel):
    """One durable allowance: this tool, in this project, from this moment."""

    tool_name: str
    #: Resolved project root the grant applies to.
    project: str
    granted_at: float = Field(default_factory=time.time)
    granted_by: str = ""
    note: str = ""


class GrantStore:
    """Allow-only grants keyed by resolved project root.

    Reads are cheap and total: :meth:`allows` answers False for anything it
    does not positively know about, including every way the file can fail.
    """

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.path = (directory or GRANT_DIR) / GRANT_FILE
        #: (project, tool_name) -> Grant
        self._grants: dict[tuple[str, str], Grant] = {}
        self.reload()

    # -- keys -------------------------------------------------------------

    @staticmethod
    def project_key(root: Optional[str | Path]) -> str:
        """The lookup key for a project root.

        ``Path(root).resolve()``, matching the task store, so a grant made
        in a scratch project never applies to a sensitive one. An unnamed
        root gets a key of its own rather than sharing one with every other
        unnamed root -- there is no such thing as a grant that applies
        everywhere.
        """
        if not root:
            return ""
        try:
            return str(Path(root).resolve())
        except (OSError, ValueError):
            return str(root)

    # -- persistence ------------------------------------------------------

    def reload(self) -> None:
        """Re-read from disk. Every failure means *nothing is granted*."""
        self._grants.clear()
        if not self.path.is_file():
            return
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                f"Grant store '{self.path}' unreadable ({exc}); treating it as "
                "empty -- every gated call will ask."
            )
            return
        if not isinstance(document, dict):
            return
        for raw in document.get("grants", []) or ():
            try:
                grant = Grant.model_validate(raw)
            except ValidationError as exc:
                log.warning(f"Skipping invalid grant {raw!r}: {exc}")
                continue
            self._grants[(grant.project, grant.tool_name)] = grant

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            document = {
                "version": FORMAT_VERSION,
                "grants": [
                    self._grants[key].model_dump()
                    for key in sorted(self._grants)
                ],
            }
            self.path.write_text(
                json.dumps(document, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            # A grant that could not be written is a grant that does not
            # exist. The user will be asked again, which is the whole
            # failure mode -- worth a warning, not an exception into a run.
            log.warning(f"Could not write the grant store '{self.path}': {exc}")

    # -- access -----------------------------------------------------------

    def allows(self, project: Optional[str | Path], tool_name: str) -> bool:
        """Whether *tool_name* is durably granted in *project*."""
        if not tool_name:
            return False
        return (self.project_key(project), str(tool_name)) in self._grants

    def grant(
        self,
        project: Optional[str | Path],
        tool_name: str,
        *,
        granted_by: str = "",
        note: str = "",
    ) -> Grant:
        """Record a durable allowance. Read-modify-write, last writer wins.

        The reload before the write is what makes concurrent granting lose
        only the *racing* grant rather than every grant written since this
        store was opened.
        """
        self.reload()
        grant = Grant(
            tool_name=str(tool_name), project=self.project_key(project),
            granted_by=granted_by, note=note,
        )
        self._grants[(grant.project, grant.tool_name)] = grant
        self._flush()
        return grant

    def revoke(self, project: Optional[str | Path], tool_name: str) -> bool:
        """Delete a grant. Revocation is deletion; there are no deny records."""
        self.reload()
        if self._grants.pop((self.project_key(project), str(tool_name)),
                            None) is None:
            return False
        self._flush()
        return True

    def revoke_project(self, project: Optional[str | Path]) -> int:
        """Delete every grant for one project; returns how many."""
        self.reload()
        key = self.project_key(project)
        doomed = [k for k in self._grants if k[0] == key]
        for k in doomed:
            del self._grants[k]
        if doomed:
            self._flush()
        return len(doomed)

    def for_project(self, project: Optional[str | Path]) -> list[Grant]:
        """Every grant in one project, for a UI that lists or revokes them."""
        key = self.project_key(project)
        return [g for (proj, _tool), g in sorted(self._grants.items())
                if proj == key]

    def projects(self) -> list[str]:
        """Every project that holds a grant, for a revocation surface.

        Sorted, and the empty key -- grants made with no project root --
        sorts first under its own name, because "granted everywhere" is
        not a thing that exists and a UI must not imply it does.
        """
        return sorted({project for project, _tool in self._grants})

    def all(self) -> list[Grant]:
        """Every grant this store holds, newest first.

        A user withdrawing authority wants to start from *what did I
        allow*, not from a project they have to remember the name of
        (§22 q1).
        """
        return sorted(self._grants.values(),
                      key=lambda g: (-g.granted_at, g.project, g.tool_name))

    def tools(self, project: Optional[str | Path]) -> frozenset[str]:
        """Just the granted tool names -- the shape the gate wants."""
        return frozenset(g.tool_name for g in self.for_project(project))


class RunGrants:
    """Run-scoped allowances: "approve this tool for the rest of the run".

    Lives in the gate's closure and dies with the run -- no persistence at
    all, which is the whole distinction from :class:`GrantStore` (D10). It
    is a class rather than a bare set so the gate has one object to consult
    and one to hand a decision to, whichever scope that decision names.
    """

    def __init__(self, tools: Iterable[str] = ()) -> None:
        self._tools: set[str] = {str(t) for t in tools}

    def allows(self, tool_name: str) -> bool:
        return str(tool_name) in self._tools

    def allow(self, tool_name: str) -> None:
        self._tools.add(str(tool_name))

    def __iter__(self):
        return iter(sorted(self._tools))

    def __len__(self) -> int:
        return len(self._tools)
