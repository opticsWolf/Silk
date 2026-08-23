"""
Bash â†’ native-tool advice, assembled from what each tool declares at
registration time.

This module used to hard-code a table mapping shell commands ("cat", "grep", â€¦)
to native tools. That table is gone. Instead:

  * Each tool declares the shell commands it stands in for via
    ``register(..., replaces=[BashHint("cat", "read_file(path=...)")])``.
  * The :class:`ToolBox` aggregates those declarations into a
    :class:`BashHintIndex`.
  * :func:`hint_for_command` turns a raw command string into a one-line nudge,
    using only the index â€” it has no built-in knowledge of any specific tool.

Advice therefore reflects the tools that are *actually loaded*: drop a new tool
into the directory and its bash equivalents light up automatically; a tool that
isn't mounted is never suggested.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Iterable, Optional

# â”€â”€ Sentinel "commands" for shell constructs that aren't a single binary â”€â”€
REDIRECT = "<redirect>"   # ``>``, ``>>`` or ``tee``
# (in-place edits like ``sed -i`` are handled via BashHint.in_place, no sentinel)

# Prefixes that wrap the real command without changing what it does.
_WRAPPERS = {"sudo", "command", "env", "time", "nice"}


@dataclass(frozen=True)
class BashHint:
    """One "use the native tool instead of this shell command" suggestion.

    ``command`` is the shell command the declaring tool replaces (e.g. ``"cat"``)
    or the :data:`REDIRECT` sentinel. ``how`` is a short how-to that names the
    native call. ``in_place`` marks the hint as the handler for the ``-i``
    (edit-in-place) form of a stream editor, e.g. ``sed -i``. ``tool`` is filled
    in by the ToolBox at registration time and need not be set by callers.
    """
    command: str
    how: str
    in_place: bool = False
    tool: str = ""

    def with_tool(self, tool: str) -> "BashHint":
        return BashHint(self.command, self.how, self.in_place, tool)


class BashHintIndex:
    """Aggregates :class:`BashHint`\\ s contributed by tools; answers lookups.

    Hints are bucketed by command name (plus a separate bucket for the ``-i``
    in-place form). Multiple tools may claim the same command â€” e.g. both
    ``search_files`` and ``ripgrep_search`` replace ``grep`` â€” and all of them
    are surfaced in the resulting hint.
    """

    def __init__(self) -> None:
        self._by_command: dict[str, list[BashHint]] = {}
        self._in_place: dict[str, list[BashHint]] = {}

    def add(self, hint: BashHint) -> None:
        bucket = self._in_place if hint.in_place else self._by_command
        bucket.setdefault(hint.command, []).append(hint)

    def remove_tool(self, tool: str) -> None:
        """Drop every hint contributed by *tool* (used when a plugin reloads)."""
        for bucket in (self._by_command, self._in_place):
            for cmd in list(bucket):
                survivors = [h for h in bucket[cmd] if h.tool != tool]
                if survivors:
                    bucket[cmd] = survivors
                else:
                    del bucket[cmd]

    def command_hints(self, command: str) -> list[BashHint]:
        return self._by_command.get(command, [])

    def in_place_hints(self, command: str) -> list[BashHint]:
        return self._in_place.get(command, [])

    def redirect_hints(self) -> list[BashHint]:
        return self._by_command.get(REDIRECT, [])

    def is_empty(self) -> bool:
        return not self._by_command and not self._in_place


# â”€â”€ Formatting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _dedupe(items: Iterable[str]) -> list[str]:
    """Order-preserving de-duplication."""
    return list(dict.fromkeys(x for x in items if x))


def _render(lead: str, hints: list[BashHint]) -> str:
    tools = " / ".join(_dedupe(h.tool for h in hints))
    hows = "; ".join(_dedupe(h.how for h in hints))
    tool_part = f"prefer the {tools} tool â€” " if tools else ""
    return f"â„¹ï¸ Tip: {lead} {tool_part}{hows}".rstrip(" ;") + "."


# â”€â”€ Parsing / dispatch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _split_head(tokens: list[str]) -> tuple[Optional[str], list[str]]:
    """Return (command, args) skipping VAR=value assignments and wrapper prefixes."""
    for i, tok in enumerate(tokens):
        if "=" in tok and not tok.startswith(("-", "/")) and tok.split("=", 1)[0].isidentifier():
            continue  # leading environment assignment
        if tok in _WRAPPERS:
            continue  # sudo / env / time / â€¦
        return tok, tokens[i + 1:]
    return None, []


def hint_for_command(command: str, index: BashHintIndex) -> Optional[str]:
    """Return a one-line nudge if *command* has a native equivalent, else None.

    Pure function of the command string and the *index*; safe to call on every
    shell invocation. Returns ``None`` when nothing in the index matches.
    """
    cmd = command.strip()
    if not cmd or index.is_empty():
        return None

    try:
        tokens = shlex.split(cmd, comments=True)
    except ValueError:
        tokens = cmd.split()
    if not tokens:
        return None

    head, rest = _split_head(tokens)
    if head is None:
        return None
    base = head.rsplit("/", 1)[-1]  # strip any path prefix on the binary

    # In-place stream edit (e.g. `sed -i`, `awk -i`) â€” only triggers when a tool
    # actually registered an in_place hint for this command.
    if any(t == "-i" or t.startswith("-i") for t in rest):
        hints = index.in_place_hints(base)
        if hints:
            return _render(f"`{base} -i` edits in place â€”", hints)

    hints = index.command_hints(base)
    if hints:
        return _render(f"instead of `{base}`,", hints)

    # Output redirection anywhere implies a write.
    if any(tok in (">", ">>") for tok in tokens) or "tee" in tokens:
        hints = index.redirect_hints()
        if hints:
            return _render("instead of shell redirection,", hints)

    return None
