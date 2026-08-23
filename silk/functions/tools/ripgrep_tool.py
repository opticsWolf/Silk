"""
Ripgrep-backed search tools using the in-process `pyripgrep` native binding
(pip install ripgrep-python). No `rg` binary or subprocess is required Ã¢â‚¬â€ the
engine is linked into the wheel Ã¢â‚¬â€ so there is no shell/argv surface: the
pattern and paths are passed as plain function arguments.

Exposes one tool, `ripgrep_search`, mirroring ripgrep's three output modes
(content / files_with_matches / count). All paths are constrained to the
FileToolSandbox; results outside the sandbox are dropped as defence in depth.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .command_advice import BashHint

if TYPE_CHECKING:
    from pathlib import Path
    from ..tool_box import ToolBox
    from .file_sandbox import FileToolSandbox

# Guarded import so a deploy target without the native wheel doesn't break the
# whole tools package Ã¢â‚¬â€ the tool still registers and reports a clear error.
try:
    import pyripgrep  # type: ignore

    _GREP: Any = pyripgrep.Grep()
    _ENGINE_ERROR: str | None = None
except Exception as e:  # pragma: no cover - exercised via the unit test's monkeypatch
    _GREP = None
    _ENGINE_ERROR = f"{type(e).__name__}: {e}"


# Parses a ripgrep content line "abspath:lineno:text". Non-greedy path so the
# FIRST ":<digits>:" wins (filenames rarely contain a ":<digits>:" sequence).
_CONTENT_RE = re.compile(r"^(?P<file>.*?):(?P<no>\d+):(?P<text>.*)$", re.DOTALL)


# Ã¢â€â‚¬Ã¢â€â‚¬ Schemas Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

class RipgrepSearchArgs(BaseModel):
    pattern: str = Field(..., description="Regular expression to search for (Rust regex syntax).")
    path: str = Field(".", description="Directory or file to search (relative to sandbox root).")
    output_mode: Literal["content", "files_with_matches", "count"] = Field(
        "content",
        description=(
            "'content' = matching lines; 'files_with_matches' = just the file paths; "
            "'count' = match count per file."
        ),
    )
    glob: str | None = Field(
        None,
        description="Optional glob to include/exclude files, e.g. '*.py' or '!*.test.js'.",
    )
    file_type: str | None = Field(
        None,
        description="Optional ripgrep file type filter, e.g. 'py', 'rust', 'js'.",
    )
    ignore_case: bool = Field(False, description="Case-insensitive search.")
    multiline: bool = Field(False, description="Allow matches to span multiple lines.")
    max_results: int = Field(
        200,
        gt=0,
        description="Maximum number of results (lines or files) to return.",
    )


class RipgrepMatch(BaseModel):
    file: str = Field(..., description="Path relative to sandbox root.")
    line_number: int = Field(..., description="1-based line number (0 if unparsable).")
    line: str = Field(..., description="The matching line text.")


class RipgrepSearchResponse(BaseModel):
    pattern: str = Field(..., description="Pattern that was searched.")
    path: str = Field(..., description="Directory that was searched (relative to sandbox root).")
    output_mode: str = Field(..., description="The output mode used.")
    matches: list[RipgrepMatch] = Field(default_factory=list, description="Populated in 'content' mode.")
    files: list[str] = Field(default_factory=list, description="Populated in 'files_with_matches' mode.")
    counts: dict[str, int] = Field(default_factory=dict, description="Populated in 'count' mode (relpath -> count).")
    total: int = Field(0, description="Lines, files, or summed counts depending on mode.")
    truncated: bool = Field(False, description="True if results were capped by max_results.")
    engine_available: bool = Field(True, description="False if the pyripgrep native engine is not installed.")
    error: str | None = Field(None, description="Error message if the search failed.")


# Ã¢â€â‚¬Ã¢â€â‚¬ Implementation Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

def _rel_or_none(sandbox: "FileToolSandbox", abspath: str) -> str | None:
    """Absolute result path -> path relative to the sandbox root, or None if outside."""
    from pathlib import Path
    try:
        p = Path(abspath).resolve()
    except OSError:
        return None
    if not sandbox.is_allowed(p):
        return None  # denied paths are invisible
    try:
        return str(p.relative_to(sandbox.root_dir))
    except ValueError:
        return None


def _denied_under(sandbox: "FileToolSandbox", directory: "Path") -> list:
    """Denied paths that live under *directory* (so it can't be searched whole)."""
    out = []
    for d in getattr(sandbox, "denied_paths", []) or []:
        try:
            d.resolve().relative_to(directory)
            out.append(d)
        except (ValueError, OSError):
            continue
    return out


def _allowed_search_roots(sandbox: "FileToolSandbox", target: "Path") -> list:
    """
    Prune denied subtrees up front: return the set of paths to actually hand to
    the engine so it never descends into a denied directory (defence-in-depth +
    perf). A directory with no denied path beneath it is returned whole; one that
    straddles a denial is split into its visible children (recursively), with
    individual allowed files surfaced directly.

    Fast path: when nothing under *target* is denied (the common case), returns
    ``[target]`` Ã¢â‚¬â€ identical to a single plain search, no overhead.
    """
    if not getattr(sandbox, "enabled", True):
        return [target]
    if not target.is_dir() or not _denied_under(sandbox, target):
        return [target]

    roots: list = []

    def _visit(directory: "Path") -> None:
        if not _denied_under(sandbox, directory):
            roots.append(directory)  # clean subtree Ã¢â‚¬â€ search it whole
            return
        try:
            children = sorted(directory.iterdir())
        except (PermissionError, OSError):
            return
        for child in children:
            if not sandbox.is_allowed(child):
                continue  # denied/invisible Ã¢â‚¬â€ skip entirely (never scanned)
            if child.is_dir():
                _visit(child)
            else:
                roots.append(child)

    _visit(target)
    return roots


def _search_ripgrep_impl(
    sandbox: "FileToolSandbox",
    pattern: str,
    path: str,
    output_mode: str,
    glob: str | None,
    file_type: str | None,
    ignore_case: bool,
    multiline: bool,
    max_results: int,
) -> RipgrepSearchResponse:
    base = RipgrepSearchResponse(pattern=pattern, path=path, output_mode=output_mode)

    if _GREP is None:
        base.engine_available = False
        base.error = (
            "pyripgrep engine is not available "
            f"({_ENGINE_ERROR}). Install it with: pip install ripgrep-python"
        )
        return base

    try:
        sandbox.check_read()
        target = sandbox.resolve_path(path)
    except (ValueError, PermissionError) as e:
        base.error = str(e)
        return base

    if not target.exists():
        base.error = f"'{target}' does not exist."
        return base

    # Engine-level prefilter: compute the visible search roots, so the engine
    # never even descends into denied directories. The common case (no denial
    # under the target) yields a single root == target, i.e. one plain search.
    roots = _allowed_search_roots(sandbox, target)

    base_kwargs: dict[str, Any] = {
        "output_mode": output_mode,
        "head_limit": max_results,
    }
    if glob is not None:
        base_kwargs["glob"] = glob
    if file_type is not None:
        base_kwargs["type"] = file_type
    if ignore_case:
        base_kwargs["i"] = True
    if multiline:
        base_kwargs["multiline"] = True
    if output_mode == "content":
        base_kwargs["n"] = True  # need line numbers to parse

    def _run(root: "Path"):
        return _GREP.search(pattern, path=str(root), **base_kwargs)

    # Merge raw results across roots. Lists (content / files_with_matches) are
    # concatenated; counts (dict) are summed. Roots are disjoint, so no dedup.
    raw_list: list = []
    raw_counts: dict[str, int] = {}
    try:
        for root in roots:
            r = _run(root)
            if output_mode == "count":
                for abspath, n in r.items():
                    raw_counts[abspath] = raw_counts.get(abspath, 0) + n
            else:
                raw_list.extend(r)
    except ValueError as e:
        # pyripgrep raises ValueError for invalid patterns.
        base.error = f"Invalid pattern or search error: {e}"
        return base
    except Exception as e:  # pragma: no cover - defensive
        base.error = f"ripgrep search failed: {type(e).__name__}: {e}"
        return base

    rel_path = str(target.relative_to(sandbox.root_dir))
    base.path = rel_path

    # is_allowed post-filter is RETAINED as the correctness backstop, independent
    # of the root-narrowing prefilter above.
    if output_mode == "files_with_matches":
        files = [r for r in (_rel_or_none(sandbox, p) for p in raw_list) if r is not None]
        base.files = files[:max_results]
        base.total = len(base.files)
        base.truncated = len(files) >= max_results
        return base

    if output_mode == "count":
        counts: dict[str, int] = {}
        for abspath, n in raw_counts.items():
            rel = _rel_or_none(sandbox, abspath)
            if rel is not None:
                counts[rel] = n
        base.counts = counts
        base.total = sum(counts.values())
        return base

    # content mode -> list of "abspath:lineno:text"
    matches: list[RipgrepMatch] = []
    for entry in raw_list:
        m = _CONTENT_RE.match(entry)
        if m:
            rel = _rel_or_none(sandbox, m.group("file"))
            if rel is None:
                continue
            matches.append(RipgrepMatch(file=rel, line_number=int(m.group("no")), line=m.group("text")))
        else:
            matches.append(RipgrepMatch(file=rel_path, line_number=0, line=entry))
    base.matches = matches[:max_results]
    base.total = len(base.matches)
    base.truncated = len(matches) >= max_results
    return base


# Ã¢â€â‚¬Ã¢â€â‚¬ Registration Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

def attach_ripgrep_tools(toolbox: "ToolBox", sandbox: "FileToolSandbox") -> None:
    """Mount the ripgrep search tool onto a ToolBox instance."""
    toolbox._file_sandbox = sandbox  # type: ignore[attr-defined]

    procedure_base = sandbox.describe_policy()
    engine_note = (
        "" if _GREP is not None
        else "\n- NOTE: the pyripgrep engine is not installed; this tool will return an error until "
             "`pip install ripgrep-python` is run."
    )

    @toolbox.register(
        name="ripgrep_search",
        tags=("search", "read"), category="search", risk="low",
        description=(
            "Fast recursive content search powered by ripgrep (in-process, .gitignore-aware). "
            "Choose output_mode: 'content' for matching lines, 'files_with_matches' for file paths, "
            "'count' for per-file counts. Returns structured JSON."
        ),
        args_model=RipgrepSearchArgs,
        replaces=[
            BashHint("rg", "ripgrep_search(pattern=..., output_mode='content'|'files_with_matches'|'count')"),
            BashHint("ag", "ripgrep_search(pattern=..., path=...)"),
            BashHint("grep", "ripgrep_search(pattern=..., path=...) (fast)"),
            BashHint("egrep", "ripgrep_search(pattern=..., path=...)"),
        ],
        procedure=(
            "Fast code/content search (ripgrep engine).\n"
            "- output_mode='content' returns {file, line_number, line}; 'files_with_matches' returns paths; "
            "'count' returns {file: count}.\n"
            "- Respects .gitignore and skips binary files automatically.\n"
            "- Use glob ('*.py', '!*.min.js') or file_type ('py','rust') to scope the search.\n"
            "- Pattern is Rust-regex syntax; prefer this over search_files for large trees.\n"
            f"{engine_note}\n"
            f"\n{procedure_base}"
        ),
    )
    def _ripgrep_search(
        db_pool: Any, user_session: dict,
        pattern: str, path: str = ".", output_mode: str = "content",
        glob: str | None = None, file_type: str | None = None,
        ignore_case: bool = False, multiline: bool = False, max_results: int = 200,
    ) -> RipgrepSearchResponse:
        return _search_ripgrep_impl(
            sandbox, pattern, path, output_mode, glob, file_type, ignore_case, multiline, max_results
        )
