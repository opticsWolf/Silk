# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

What an MCP server can reach, said out loud (G21 residue 2, D84).

Every file tool Silk registers is sandboxed: a root list, a
narrowing-only grant model (D16-D18), locks across agents (D67). An MCP
server is a **separate process**. It writes wherever its own process can,
Silk neither sandboxes it nor sees it, and the file-permissions port on
the canvas says nothing about it. That was already true before §19; §19
made it consequential, because now something in this process is willing
to import what appears on disk (G21).

Silk cannot fix that -- the authority is the server's operating-system
authority, granted when the user launched it. What it can do is stop the
surface from *implying* otherwise. This module reads the tool list a
server advertises and names the tools that look like filesystem access,
so the node that mounted the server says so where the sandbox settings
are read.

Two honesty rules, because this is a heuristic over someone else's names:

- **Silence is not a promise.** A server whose tools do not look
  filesystem-shaped may still write files; the notice says "these look
  like", never "these are the only ones".
- **It reports, it never refuses.** A filesystem MCP server is a
  legitimate thing to want. Refusing to mount one would be Silk deciding
  a question the user already answered by configuring it.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

__all__ = [
    "WRITE_HINTS",
    "READ_HINTS",
    "file_tools",
    "reach_notice",
]

#: Verbs that suggest a tool *changes* something on a filesystem. Matched
#: against the tool name and its description, word-wise, so `write_file`
#: and "Writes the file at PATH" both hit and `rewrite_query` does not.
WRITE_HINTS = (
    "write", "create", "delete", "remove", "move", "rename", "copy",
    "edit", "patch", "append", "mkdir", "save", "unlink", "chmod",
)

#: Verbs that suggest a tool *reads* a filesystem. Reported separately:
#: reading the user's disk from a process Silk does not sandbox is a
#: smaller thing than writing to it, but it is not nothing.
READ_HINTS = ("read", "list", "glob", "search", "grep", "stat", "open", "cat")

#: Words that make a hint filesystem-flavoured rather than generic. A
#: server full of `create_issue` / `list_repos` tools is not filesystem
#: access, and reporting it as such is how a notice trains people to
#: ignore notices.
_SUBJECTS = ("file", "files", "directory", "directories", "dir", "dirs",
             "folder", "folders", "path", "paths", "filesystem", "fs",
             "disk")

_WORD = re.compile(r"[a-z0-9]+")


#: Endings stripped before matching, so "saves the file to disk" hits the
#: same `save` hint that `save_file` does. Servers write their
#: descriptions in prose and their names in verbs; both are evidence.
_ENDINGS = ("ing", "ies", "es", "ed", "s")


def _stems(word: str) -> set[str]:
    """Every plausible root of one word, not the single best guess.

    "saves" is `save` after dropping `s` and `sav` after dropping `es`,
    and a stemmer that had to choose would get one of them wrong. Nothing
    downstream is harmed by an extra non-word: it only matches if it
    matches a hint, and `sav` is not a hint.
    """
    return {word[: -len(e)] for e in _ENDINGS
            if len(word) > len(e) + 2 and word.endswith(e)}


def _words(entry: Any) -> set[str]:
    name = str((entry or {}).get("name", ""))
    description = str((entry or {}).get("description", ""))
    found = set(_WORD.findall(f"{name} {description}".lower()))
    for word in tuple(found):
        found |= _stems(word)
    return found


def _classify(entry: Any) -> str:
    """``"write"``, ``"read"`` or ``""`` for one advertised tool."""
    words = _words(entry)
    if not words & set(_SUBJECTS):
        return ""
    if words & set(WRITE_HINTS):
        return "write"
    if words & set(READ_HINTS):
        return "read"
    return ""


def file_tools(tools: Iterable[Any]) -> dict[str, list[str]]:
    """The advertised tools that look like filesystem access.

    Returns ``{"write": [...], "read": [...]}`` with the names sorted, so
    a caller renders a stable line and a test can compare one.
    """
    found: dict[str, list[str]] = {"write": [], "read": []}
    for entry in tools or ():
        kind = _classify(entry)
        if kind:
            found[kind].append(str((entry or {}).get("name", "")))
    return {k: sorted(v) for k, v in found.items()}


def reach_notice(server: str, tools: Iterable[Any]) -> str:
    """One sentence for the node's status line and the log, or ``""``.

    Empty when nothing looks filesystem-shaped -- which is *not* a
    guarantee that the server touches no files, and the notice text says
    so where it matters.
    """
    found = file_tools(tools)
    writes, reads = found["write"], found["read"]
    if not writes and not reads:
        return ""

    parts = []
    if writes:
        parts.append(f"writes ({', '.join(writes)})")
    if reads:
        parts.append(f"reads ({', '.join(reads)})")
    return (
        f"'{server}' advertises tools that look like filesystem access: "
        + "; ".join(parts)
        + ". An MCP server runs in its own process, so Silk's file "
          "sandbox, grants and locks do not apply to it -- what it may "
          "touch is what its process may touch."
    )
