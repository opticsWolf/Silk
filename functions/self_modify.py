# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The rules an agent's *code* changes obey (spec §19, D76-D78, D81).

§18 let the agent use Weave's parts; this is the agent making new ones.
The loop already existed except for one verb (D75): **write** is the file
tools, **verify** is the toolchain runner, **observe** is `describe_graph`
-- only **load** was missing. So there is no self-improvement subsystem
here, just the missing verb and the three rules that make it safe to call.

The one sentence everything follows from (D77):

    Every file tool Silk has is sandboxed; ``import`` is not sandboxable.

Module-level code in an agent-authored file runs with the full authority
of the Weave process, no matter how narrow the sandbox was while that
file was being written. Write authority over a directory on the import
path *is* process authority, deferred by exactly one tool call. Hence:

1. **The agent writes into its own root and nowhere else** (D76) --
   `~/.weave/plugins/<name>/`, the user plugin dir, and never `weave/`,
   never `plugins/silk/`, never the virtualenv. This is D73's
   self-modification guard moved one layer down, from the graph to the
   filesystem, for the same reason.
2. **Loading is always approved** (D77) -- an unconditional floor no
   Role, preset or grant may lower, and the request carries the *diff*,
   because a human approving "load `my_nodes`" has approved nothing.
3. **The linter is the code review** (D78) -- a `WV521` finding is a hard
   stop before load, because a machine author gets the linter and nothing
   else. A human author gets a reviewer; this is the substitute.

Qt-free: everything here is a decision, and the decisions are testable
without a canvas. The main-thread half (the load itself) lives in the node
layer behind the D70 seam.
"""
from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from weave.logger import get_logger

from .graph_author import Refusal

log = get_logger("SilkSelfModify")

#: The load verbs (D75). Named once: the floor, the Role selector, the
#: registrations and the tests all read the same tuple.
OP_LIST_SUITES = "list_suites"
OP_LOAD_SUITE = "load_suite"
OP_RELOAD_SUITE = "reload_suite"
OP_RELAUNCH = "request_relaunch"

#: Loading is always approved and never narrowable away (D77). This is
#: **not** I6: I6 makes narrowing monotone -- a ceiling nobody may raise.
#: This is a floor nobody may lower. An "always approve" a preset can
#: switch off is not a control.
ALWAYS_APPROVE = (OP_LOAD_SUITE, OP_RELOAD_SUITE)

#: Suites the agent may load: only those it could have written (D76).
ORIGIN_USER = "user"

#: The linter findings that stop a load dead (D78). WV520-522 are the
#: state-versioning rules: an agent editing an existing node class changes
#: the shape of state held in *users' saved graphs*, and these are exactly
#: the check for that.
BLOCKING_CODES = ("WV520", "WV521", "WV522")

#: How long the version check may take before it is treated as not having
#: run at all.
LINT_TIMEOUT_S = 180.0

#: Where a quarantine fact is kept when there is no plan to write it to.
QUARANTINE_FILE = "quarantine.json"

#: Cap on what a diff may contribute to an approval request. A human
#: reading a 5000-line diff in a dialog is approving by scroll position.
MAX_DIFF_LINES = 400
MAX_DIFF_FILES = 40


# ── where the agent may write (D76) ──────────────────────────────────────


def user_plugin_root() -> Path:
    """`~/.weave/plugins`, or `$WEAVE_USER_PLUGIN_DIR`.

    Weave owns this path; Silk asks rather than reimplementing it, so a
    portable install or a test override moves both at once.
    """
    from weave.engine.suite_loader import user_plugin_dir

    return Path(user_plugin_dir())


def suites() -> list[dict]:
    """Every discoverable suite, JSON-shaped, with `writable` added.

    `writable` is Silk's judgement, not Weave's: a builtin suite is
    loadable by a human from the Plugin Manager and *not* material an
    agent may author (D76). Saying so in the listing is how the agent
    learns the rule before it hits it.
    """
    from weave.engine.suite_loader import list_suites

    rows = []
    for row in list_suites():
        entry = dict(row)
        entry["writable"] = entry.get("origin") == ORIGIN_USER
        rows.append(entry)
    return rows


def find(name: str) -> Optional[dict]:
    return next((s for s in suites() if s["name"] == name), None)


def check_suite(name: str, op: str = OP_LOAD_SUITE) -> Optional[Refusal]:
    """Whether *name* is a suite this agent may load at all (D76)."""
    clean = str(name or "").strip()
    if not clean:
        return Refusal("Name the suite to load.", op=op, subject=clean)
    info = find(clean)
    if info is None:
        root = user_plugin_root()
        return Refusal(
            f"No suite called '{clean}'. Agent-authored suites live in "
            f"{root} as a package directory ({clean}/__init__.py); write "
            f"one there and it becomes loadable.",
            op=op, subject=clean)
    if not info["writable"]:
        return Refusal(
            f"'{clean}' is a shipped suite ({info['origin']}), not "
            f"agent-authored code. Silk loads only what is under "
            f"{user_plugin_root()}: the code that is running this agent "
            f"is not material the agent replaces.",
            op=op, subject=clean)
    return None


def capability_for(name: str):
    """The `SuiteCapability` this agent gets: exactly one user suite.

    Weave's handle is scoped at issuance, and the issuer here is Silk --
    so it issues the narrowest thing that does the job. Never core, never
    another vendor's plugin, never unscoped.
    """
    from weave.engine.capability import for_suite

    return for_suite(name, holder="silk-agent")


# ── what this run wrote (D77: the request shows the code) ────────────────


#: The file tools whose calls are worth watching. A tool that cannot
#: change a file cannot contribute to a diff.
WRITE_TOOLS = ("write_file", "append_file", "edit_file", "insert_text",
               "create_directory", "move_file", "copy_file", "delete_file")

#: Argument names those tools use for the file they act on.
PATH_ARGS = ("path", "file_path", "destination", "dest", "target", "src",
             "source")


@dataclass
class FileChange:
    """One file, before and after this run touched it."""

    path: str
    before: Optional[str] = None
    after: Optional[str] = None

    @property
    def created(self) -> bool:
        return self.before is None and self.after is not None

    @property
    def deleted(self) -> bool:
        return self.before is not None and self.after is None

    def diff(self, max_lines: int = MAX_DIFF_LINES) -> str:
        """Unified diff, truncated with a line that says so."""
        before = (self.before or "").splitlines(keepends=True)
        after = (self.after or "").splitlines(keepends=True)
        lines = list(difflib.unified_diff(
            before, after,
            fromfile=f"a/{self.path}" if not self.created else "/dev/null",
            tofile=f"b/{self.path}" if not self.deleted else "/dev/null",
            n=3))
        if len(lines) > max_lines:
            kept = lines[:max_lines]
            kept.append(f"... {len(lines) - max_lines} more diff lines "
                        f"(the whole file is on disk)\n")
            lines = kept
        return "".join(lines)


class ChangeSet:
    """What this run wrote, kept so an approval can show it (D77).

    Snapshots are taken around each write tool call, so the diff is of
    *this run's* work rather than of whatever the file looked like when
    the session started. Reading is best-effort: an unreadable or binary
    file records `None` and shows up as "changed, not shown" rather than
    taking the run down.
    """

    #: Files larger than this are recorded as touched, not as content: a
    #: diff of a 4 MB file helps nobody and costs the whole prompt.
    MAX_BYTES = 256 * 1024

    def __init__(self) -> None:
        self._changes: dict[str, FileChange] = {}

    def __len__(self) -> int:
        return len(self._changes)

    @staticmethod
    def _read(path: Path) -> Optional[str]:
        try:
            if not path.is_file():
                return None
            if path.stat().st_size > ChangeSet.MAX_BYTES:
                return None
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def before(self, path: str | os.PathLike) -> None:
        """Record the pre-call content, once per file."""
        target = Path(path)
        key = str(target)
        if key in self._changes:
            return                      # the first snapshot is the baseline
        self._changes[key] = FileChange(path=key, before=self._read(target))

    def after(self, path: str | os.PathLike) -> None:
        target = Path(path)
        key = str(target)
        change = self._changes.get(key)
        if change is None:
            change = FileChange(path=key)
            self._changes[key] = change
        change.after = self._read(target)

    def within(self, root: str | os.PathLike) -> list[FileChange]:
        """Changes under *root*, oldest recorded first."""
        base = Path(root).resolve()
        out = []
        for change in self._changes.values():
            try:
                candidate = Path(change.path).resolve()
            except OSError:
                continue
            if candidate == base or base in candidate.parents:
                out.append(change)
        return out

    def diffs(self, root: str | os.PathLike,
              max_files: int = MAX_DIFF_FILES) -> list[dict]:
        """The diffs an approval request carries."""
        out = []
        for change in self.within(root)[:max_files]:
            text = change.diff()
            out.append({"path": change.path,
                        "created": change.created,
                        "deleted": change.deleted,
                        "diff": text or "(no textual change to show)"})
        return out


_CHANGES_ATTR = "_silk_changes"


def bind_changes(toolbox: Any, changes: Optional[ChangeSet]) -> None:
    """Bind this run's change record onto *toolbox*, or clear it."""
    if changes is None:
        if hasattr(toolbox, _CHANGES_ATTR):
            delattr(toolbox, _CHANGES_ATTR)
        return
    setattr(toolbox, _CHANGES_ATTR, changes)


def run_changes(toolbox: Any) -> Optional[ChangeSet]:
    return getattr(toolbox, _CHANGES_ATTR, None)


def _target_paths(args: dict) -> list[str]:
    return [str(args[key]) for key in PATH_ARGS
            if key in args and isinstance(args[key], (str, os.PathLike))]


def attach_change_tracking(toolbox: Any,
                           changes: Optional[ChangeSet] = None) -> Any:
    """Watch the write tools so a load request can show the diff.

    A middleware rather than a wrapper on each tool: the file tools are
    Silk's oldest surface and must not learn about §19, and a middleware
    also catches a write tool that arrives later from a capability.
    """
    from .hooks import HOOK_WRAP_TOOL_EXECUTE

    record = changes if changes is not None else ChangeSet()
    bind_changes(toolbox, record)

    async def watch(handler=None, tool_name: str = "",
                    tool_args: Optional[dict] = None, **_kw: Any) -> Any:
        paths = _target_paths(dict(tool_args or {}))
        for path in paths:
            record.before(path)
        try:
            return await handler()
        finally:
            for path in paths:
                record.after(path)

    entry = toolbox.hooks.register_middleware(
        HOOK_WRAP_TOOL_EXECUTE, watch, tools=WRITE_TOOLS)
    return entry


# ── the version check is the code review (D78) ───────────────────────────


@dataclass
class LintOutcome:
    """What `weave_lint` said about a candidate suite."""

    ran: bool = False
    ok: bool = False
    blocking: list[dict] = field(default_factory=list)
    errors: int = 0
    warnings: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {"ran": self.ran, "ok": self.ok, "errors": self.errors,
                "warnings": self.warnings, "note": self.note,
                "blocking": self.blocking}

    def refusal_text(self) -> str:
        if not self.ran:
            return (f"The state-version check could not run ({self.note}), "
                    f"so nothing is known about what this load would do to "
                    f"saved graphs. Loading is refused: a check that cannot "
                    f"run is not a check that passed.")
        lines = [f"{f['code']} {f['file']}:{f['line']} — {f['message']}"
                 for f in self.blocking[:20]]
        return (
            "The state-version check refuses this load. Either bump "
            "`node_state_api` and write `migrate_state`, or leave the class "
            "alone and ship a new one with `node_supersedes` — a saved graph "
            "holding the old state is the thing being protected here:\n"
            + "\n".join(lines)
        )


def lint_suite(path: str | os.PathLike, *,
               python: Optional[str] = None,
               timeout_s: float = LINT_TIMEOUT_S) -> LintOutcome:
    """Run `weave_lint` over *path* and read the JSON it prints.

    Fails **closed** (D36): a linter that is missing, crashes or hangs
    produces `ran=False`, and the caller refuses the load. The
    alternative -- loading because the check could not be made -- turns
    the one review a machine author gets into an optional one.
    """
    argv = [python or sys.executable, "-m", "weave_lint",
            "--format", "json", "--no-color", str(path)]
    try:
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return LintOutcome(note=f"it did not finish within {timeout_s:g}s")
    except OSError as exc:
        return LintOutcome(note=f"it could not be started: {exc}")

    payload = _last_json_object(done.stdout)
    if payload is None:
        tail = (done.stderr or done.stdout or "").strip()[-400:]
        return LintOutcome(note=f"it printed no report ({tail or 'no output'})")

    summary = payload.get("summary") or {}
    diagnostics = payload.get("diagnostics") or []
    blocking = [
        {"code": str(d.get("code", "")), "file": str(d.get("file", "")),
         "line": d.get("line", 0), "message": str(d.get("message", ""))}
        for d in diagnostics
        if str(d.get("code", "")).upper() in BLOCKING_CODES
    ]
    return LintOutcome(
        ran=True, ok=not blocking, blocking=blocking,
        errors=int(summary.get("error", 0) or 0),
        warnings=int(summary.get("warning", 0) or 0),
        note="clean" if not blocking else "state-version findings",
    )


def _last_json_object(text: str) -> Optional[dict]:
    """The JSON object in *text*, ignoring whatever was printed first.

    The linter prints a human line before its JSON; a caller that assumed
    the whole of stdout was JSON would break on the day it prints a
    warning, which is the day it matters most.
    """
    start = text.find("{")
    while start != -1:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    return None


# ── what the human is shown before a load (D77) ──────────────────────────


def suite_files(path: str | os.PathLike, limit: int = 200) -> list[dict]:
    """Every file in the suite, with size and mtime.

    The listing is the *other* half of the request: a diff shows what
    this run wrote, and the listing shows what else is in the directory
    -- including a file some earlier run left there.
    """
    root = Path(path)
    rows: list[dict] = []
    if not root.is_dir():
        return rows
    for item in sorted(root.rglob("*")):
        if item.is_dir() or "__pycache__" in item.parts:
            continue
        try:
            stat = item.stat()
        except OSError:
            continue
        rows.append({
            "path": str(item.relative_to(root)).replace("\\", "/"),
            "bytes": stat.st_size,
            "modified": time.strftime("%Y-%m-%d %H:%M:%S",
                                      time.localtime(stat.st_mtime)),
        })
        if len(rows) >= limit:
            rows.append({"path": f"... more than {limit} files", "bytes": 0,
                         "modified": ""})
            break
    return rows


def load_request_detail(name: str, path: str | os.PathLike, *,
                        changes: Optional[ChangeSet] = None,
                        lint: Optional[LintOutcome] = None,
                        reload: bool = False) -> dict:
    """The payload an approval renders. Code, not a name (D77)."""
    diffs = changes.diffs(path) if changes is not None else []
    return {
        "risk": "high",
        "kind": "suite_load",
        "suite": name,
        "path": str(path),
        "reload": reload,
        "files": suite_files(path),
        "diffs": diffs,
        "changed_files": len(diffs),
        "lint": (lint.as_dict() if lint is not None else {}),
        "why": ("Importing this code runs it with the full authority of "
                "this process: the network, the filesystem, your keys. "
                "The sandbox that constrained writing it does not apply "
                "to importing it."),
    }


def approval_prompt(name: str, changed: int, *, reload: bool = False) -> str:
    verb = "Reload" if reload else "Load"
    if changed:
        return (f"{verb} the suite '{name}'? This run wrote {changed} file"
                f"{'' if changed == 1 else 's'} in it; importing runs that "
                f"code with full process authority.")
    return (f"{verb} the suite '{name}'? Importing runs its code with full "
            f"process authority. This run wrote none of it.")


# ── the quarantine fact (D81) ────────────────────────────────────────────


def quarantine_path(root: Optional[str | os.PathLike] = None) -> Path:
    base = Path(root) if root is not None else user_plugin_root()
    return base / QUARANTINE_FILE


def record_quarantine(names: Sequence[str], *,
                      traceback_text: str = "",
                      root: Optional[str | os.PathLike] = None,
                      store: Any = None,
                      actor: str = "weave") -> list[str]:
    """Write down that a suite was quarantined, for the *next* run.

    Weave's loop guard covers the first half -- start clean, skip the
    suspect suite, say so in a report. This is the other half: a
    self-improving loop with no feedback on failure does not improve, it
    repeats. The fact lands where the next agent will actually meet it:
    in the suite listing it reads before loading anything, and -- when a
    plan is open -- as a task, so a human sees it in the same place they
    see the rest of the work.
    """
    written = [str(n) for n in names if str(n).strip()]
    if not written:
        return []

    target = quarantine_path(root)
    facts = read_quarantine(target.parent)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    for name in written:
        facts[name] = {"suite": name, "at": stamp,
                       "traceback": traceback_text[-4000:]}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    except OSError as exc:
        log.warning(f"could not record the quarantine fact: {exc}")

    if store is not None:
        for name in written:
            try:
                store.add_task(
                    title=f"Plugin '{name}' was quarantined after failing "
                          f"to load",
                    note=(traceback_text[-2000:] or
                          "Weave's loop guard skipped it on the last start."),
                    actor=actor,
                    rationale="a quarantined plugin is a fact the next run "
                              "needs, not a log line it will never read",
                )
            except Exception as exc:  # noqa: BLE001 -- never fail a start
                log.warning(f"could not add the quarantine task: {exc}")
    return written


def read_quarantine(root: Optional[str | os.PathLike] = None) -> dict:
    """Quarantine facts by suite name; `{}` when there are none."""
    target = quarantine_path(root)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def clear_quarantine(name: str,
                     root: Optional[str | os.PathLike] = None) -> bool:
    """Forget the fact once the suite loads again."""
    facts = read_quarantine(root)
    if name not in facts:
        return False
    facts.pop(name)
    target = quarantine_path(root)
    try:
        target.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    except OSError:
        return False
    return True


def annotate(rows: Iterable[dict],
             root: Optional[str | os.PathLike] = None) -> list[dict]:
    """Add the quarantine fact to a suite listing (D81)."""
    facts = read_quarantine(root)
    out = []
    for row in rows:
        entry = dict(row)
        fact = facts.get(entry.get("name", ""))
        if fact is not None:
            entry["quarantined"] = True
            entry["quarantine_note"] = fact.get("traceback", "")[-1000:]
            entry["quarantined_at"] = fact.get("at", "")
        out.append(entry)
    return out
