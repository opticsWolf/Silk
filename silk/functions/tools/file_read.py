"""Read-only file tools: read_file, view_file, list_directory, find_files, search_files, file_info."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .command_advice import BashHint

if TYPE_CHECKING:
    from pathlib import Path
    from ..tool_box import ToolBox
    from .file_sandbox import FileToolSandbox


# Per-file ceiling for content search so a single pathological file can't be
# slurped into memory. Generous enough never to bite normal source trees.
_SEARCH_MAX_FILE_BYTES = 20 * 1024 * 1024


# Ã¢â€â‚¬Ã¢â€â‚¬ Pydantic request schemas Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

class ReadFileArgs(BaseModel):
    path: str = Field(..., description="Path to the file (relative to sandbox root).")
    max_bytes: int = Field(
        512_000,
        gt=0,
        description="Maximum bytes to read. Capped by sandbox policy.",
    )
    offset: int = Field(
        0,
        ge=0,
        description="Byte offset to start reading from.",
    )


class ViewFileArgs(BaseModel):
    path: str = Field(..., description="Path to the file (relative to sandbox root).")
    start_line: int = Field(
        1,
        ge=1,
        description="1-based line number to start viewing from.",
    )
    end_line: int | None = Field(
        None,
        description="1-based inclusive line to stop at. None = to end of file (capped by max_lines).",
    )
    max_lines: int = Field(
        400,
        gt=0,
        le=5000,
        description="Maximum number of lines to return.",
    )


class ListDirectoryArgs(BaseModel):
    path: str = Field(".", description="Directory to list (relative to sandbox root).")
    recursive: bool = Field(False, description="Recurse into subdirectories.")
    depth: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum recursion depth (ignored if recursive=False).",
    )
    pattern: str = Field(
        "*",
        description="Glob pattern to filter entries (e.g. '*.py', 'test_*').",
    )


class DirectoryTreeArgs(BaseModel):
    path: str = Field(".", description="Directory to render (relative to sandbox root).")
    depth: int = Field(
        3,
        ge=1,
        le=12,
        description="Maximum tree depth.",
    )
    pattern: str = Field(
        "*",
        description="Glob pattern applied to file names (directories are always shown).",
    )
    show_files: bool = Field(True, description="Include files; set False for a directories-only tree.")
    max_entries: int = Field(
        500,
        gt=0,
        le=5000,
        description="Cap on rendered entries; the tree is truncated past this.",
    )


class FindFilesArgs(BaseModel):
    pattern: str = Field(
        "*.py",
        description="Glob pattern to match against file names (e.g. '*.txt', 'test_*').",
    )
    path: str = Field(".", description="Directory to search (relative to sandbox root).")
    max_results: int = Field(
        200,
        gt=0,
        description="Maximum number of results to return.",
    )


class SearchFilesArgs(BaseModel):
    pattern: str = Field(
        ...,
        description="Regular expression pattern to search for.",
    )
    path: str = Field(".", description="Directory to search (relative to sandbox root).")
    file_pattern: str = Field(
        "*",
        description="Glob pattern to select which files to search (e.g. '*.py').",
    )
    context_lines: int = Field(
        0,
        ge=0,
        le=10,
        description="Number of context lines before/after each match.",
    )
    max_results: int = Field(
        100,
        gt=0,
        description="Maximum number of matching lines to return.",
    )
    ignore_case: bool = Field(False, description="Case-insensitive search.")


class FileInfoArgs(BaseModel):
    path: str = Field(..., description="Path to the file or directory (relative to sandbox root).")


# Ã¢â€â‚¬Ã¢â€â‚¬ Pydantic response schemas Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

class ReadFileResponse(BaseModel):
    """Structured reply from read_file."""
    path: str = Field(..., description="Relative path of the file that was read.")
    content: str = Field(..., description="File contents (UTF-8, replacement chars for invalid bytes).")
    bytes_read: int = Field(..., description="Number of bytes actually read.")
    truncated: bool = Field(..., description="True if the file was cut short by max_bytes.")
    offset: int = Field(..., description="Byte offset where reading started.")
    error: str | None = Field(None, description="Error message if the read failed.")


class ViewFileResponse(BaseModel):
    """Structured reply from view_file (line-numbered slice)."""
    path: str = Field(..., description="Relative path of the file that was viewed.")
    content: str = Field("", description="Line-numbered text in the requested range.")
    start_line: int = Field(..., description="1-based line the view starts at.")
    end_line: int = Field(..., description="1-based line the view ends at (inclusive).")
    total_lines: int = Field(..., description="Total number of lines in the file.")
    truncated: bool = Field(False, description="True if the range was capped by max_lines.")
    error: str | None = Field(None, description="Error message if the view failed.")


class DirectoryEntry(BaseModel):
    """One entry inside a directory listing."""
    name: str = Field(..., description="File or directory name.")
    type: str = Field(..., description="'file' or 'directory'.")
    size_bytes: int | None = Field(None, description="Size in bytes (files only, None for directories).")
    relative_path: str = Field(..., description="Path relative to sandbox root.")


class ListDirectoryResponse(BaseModel):
    """Structured reply from list_directory."""
    path: str = Field(..., description="Relative path of the directory that was listed.")
    entries: list[DirectoryEntry] = Field(default_factory=list, description="Listed entries.")
    error: str | None = Field(None, description="Error message if the listing failed.")


class FindFilesResponse(BaseModel):
    """Structured reply from find_files."""
    pattern: str = Field(..., description="Glob pattern that was searched.")
    path: str = Field(..., description="Directory that was searched (relative to sandbox root).")
    results: list[str] = Field(default_factory=list, description="Relative paths of matching files.")
    total_found: int = Field(default=0, description="Number of files returned (more may exist if truncated).")
    truncated: bool = Field(default=False, description="True if results were capped by max_results.")
    error: str | None = Field(None, description="Error message if the search failed.")


class SearchMatch(BaseModel):
    """One matching line from search_files."""
    file: str = Field(..., description="Relative path of the file containing the match.")
    line_number: int = Field(..., description="1-based line number.")
    line: str = Field(..., description="The line text (trimmed).")
    is_context: bool = Field(False, description="True if this is a context line, not the match itself.")


class SearchFilesResponse(BaseModel):
    """Structured reply from search_files."""
    pattern: str = Field(..., description="Regex pattern that was searched.")
    path: str = Field(..., description="Directory that was searched (relative to sandbox root).")
    file_pattern: str = Field(..., description="Glob pattern used to select files.")
    matches: list[SearchMatch] = Field(default_factory=list, description="Matching lines with context.")
    total_matches: int = Field(default=0, description="Number of matching lines found (may exceed matches if truncated).")
    truncated: bool = Field(default=False, description="True if results were capped by max_results.")
    error: str | None = Field(None, description="Error message if the search failed (e.g. invalid regex).")


class FileInfoResponse(BaseModel):
    """Structured reply from file_info."""
    path: str = Field(..., description="Relative path of the file or directory.")
    absolute_path: str = Field(..., description="Absolute filesystem path.")
    type: str = Field(..., description="'file' or 'directory'.")
    size_bytes: int | None = Field(None, description="Size in bytes (files only, None for directories).")
    modified: str = Field(..., description="ISO-like modification timestamp (YYYY-MM-DD HH:MM:SS).")
    error: str | None = Field(None, description="Error message if metadata could not be read.")


# Ã¢â€â‚¬Ã¢â€â‚¬ Tool implementations Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

def _get_sandbox(toolbox: Any) -> "FileToolSandbox":
    """Retrieve the sandbox from the toolbox instance."""
    sandbox = getattr(toolbox, "_file_sandbox", None)
    if sandbox is None:
        raise RuntimeError(
            "No FileToolSandbox configured. Call attach_file_tools(sandbox=Ã¢â‚¬Â¦) first."
        )
    return sandbox


# -- read_file -----------------------------------------------------------

def _read_file_impl(sandbox: "FileToolSandbox", path: str, max_bytes: int, offset: int) -> ReadFileResponse:
    try:
        sandbox.check_read()
        target = sandbox.resolve_path(path)
        sandbox.check_extension(target)
    except (ValueError, PermissionError) as e:
        return ReadFileResponse(path=path, content="", bytes_read=0, truncated=False, offset=offset, error=str(e))

    if not target.is_file():
        return ReadFileResponse(path=path, content="", bytes_read=0, truncated=False, offset=offset,
                                error=f"'{target}' is not a file or does not exist.")

    cap = min(max_bytes, sandbox.max_read_bytes)
    truncated = False

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            content = f.read(cap)
            # Truncated only if there is genuinely more to read past the cap.
            if f.read(1):
                truncated = True
    except OSError as e:
        return ReadFileResponse(path=path, content="", bytes_read=0, truncated=False, offset=offset,
                                error=f"Error reading '{target}': {e}")

    total_read = len(content)
    rel = str(target.relative_to(sandbox.root_dir))
    if truncated:
        content += f"\n\n[Truncated Ã¢â‚¬â€ returned the first {cap} bytes from offset {offset}; increase max_bytes or use offset to read more]"

    return ReadFileResponse(
        path=rel,
        content=content,
        bytes_read=total_read,
        truncated=truncated,
        offset=offset,
    )


# -- view_file -----------------------------------------------------------

def _view_file_impl(
    sandbox: "FileToolSandbox", path: str, start_line: int, end_line: int | None, max_lines: int
) -> ViewFileResponse:
    try:
        sandbox.check_read()
        target = sandbox.resolve_path(path)
        sandbox.check_extension(target)
    except (ValueError, PermissionError) as e:
        return ViewFileResponse(path=path, start_line=start_line, end_line=start_line,
                                total_lines=0, error=str(e))

    if not target.is_file():
        return ViewFileResponse(path=path, start_line=start_line, end_line=start_line,
                                total_lines=0, error=f"'{target}' is not a file or does not exist.")

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ViewFileResponse(path=path, start_line=start_line, end_line=start_line,
                                total_lines=0, error=f"Error reading '{target}': {e}")

    rel = str(target.relative_to(sandbox.root_dir))
    lines = text.splitlines()
    total = len(lines)

    if total == 0:
        return ViewFileResponse(path=rel, content="", start_line=0, end_line=0, total_lines=0)

    start = max(1, min(start_line, total))
    end = total if end_line is None else max(start, min(end_line, total))

    truncated = False
    if end - start + 1 > max_lines:
        end = start + max_lines - 1
        truncated = True

    width = len(str(end))
    numbered = "\n".join(
        f"{n:>{width}}\t{lines[n - 1]}" for n in range(start, end + 1)
    )

    return ViewFileResponse(
        path=rel,
        content=numbered,
        start_line=start,
        end_line=end,
        total_lines=total,
        truncated=truncated,
    )


# -- list_directory ------------------------------------------------------

def _glob_match(name: str, pattern: str) -> bool:
    """Minimal glob match supporting * and ?."""
    import fnmatch
    return fnmatch.fnmatch(name, pattern)


def _list_directory_impl(
    sandbox: "FileToolSandbox", path: str, recursive: bool, depth: int, pattern: str
) -> ListDirectoryResponse:
    sandbox.check_read()
    try:
        target = sandbox.resolve_path(path)
    except (ValueError, PermissionError) as e:
        return ListDirectoryResponse(path=path, error=str(e))

    if not target.is_dir():
        return ListDirectoryResponse(path=path, error=f"'{target}' is not a directory.")

    entries: list[DirectoryEntry] = []

    def _walk(current: "Path", current_depth: int) -> None:
        if current_depth >= depth:
            return
        try:
            for entry in sorted(current.iterdir()):
                if not sandbox.is_allowed(entry):
                    continue  # denied paths are invisible to the LLM
                rel = str(entry.relative_to(sandbox.root_dir))
                if recursive and entry.is_dir():
                    entries.append(DirectoryEntry(name=entry.name, type="directory", size_bytes=None, relative_path=rel))
                    _walk(entry, current_depth + 1)
                else:
                    if pattern == "*" or _glob_match(entry.name, pattern):
                        kind = "directory" if entry.is_dir() else "file"
                        size = entry.stat().st_size if entry.is_file() else None
                        entries.append(DirectoryEntry(name=entry.name, type=kind, size_bytes=size, relative_path=rel))
        except PermissionError:
            entries.append(DirectoryEntry(name=str(current.relative_to(sandbox.root_dir)),
                                          type="directory", size_bytes=None,
                                          relative_path=str(current.relative_to(sandbox.root_dir))))

    _walk(target, 0)
    return ListDirectoryResponse(path=str(target.relative_to(sandbox.root_dir)), entries=entries)


# -- directory_tree ------------------------------------------------------

def _directory_tree_impl(
    sandbox: "FileToolSandbox",
    path: str,
    depth: int,
    pattern: str,
    show_files: bool,
    max_entries: int,
) -> str:
    """Render a compact ASCII tree, honouring the sandbox read policy.

    Every entry is gated by ``sandbox.is_allowed`` Ã¢â‚¬â€ files (and whole subtrees)
    the LLM may not read are simply absent from the tree, with no placeholder
    that would leak their existence. Returns a plain string (not JSON) so the
    tree renders human-readably for the model.
    """
    sandbox.check_read()
    try:
        target = sandbox.resolve_path(path)
    except (ValueError, PermissionError) as e:
        return f"Error: {e}"
    if not target.is_dir():
        return f"Error: '{target}' is not a directory."

    root_label = target.name or str(target)
    lines: list[str] = [f"{root_label}/"]
    rendered = 0
    truncated = False

    def _visible_children(directory: "Path") -> list["Path"]:
        try:
            entries = list(directory.iterdir())
        except (PermissionError, OSError):
            return []
        kept = []
        for entry in entries:
            if not sandbox.is_allowed(entry):
                continue  # invisible under the read policy
            if entry.is_dir():
                kept.append(entry)
            elif show_files and (pattern == "*" or _glob_match(entry.name, pattern)):
                kept.append(entry)
        # directories first, then files; each alphabetical
        kept.sort(key=lambda p: (p.is_file(), p.name.lower()))
        return kept

    def _walk(directory: "Path", prefix: str, current_depth: int) -> None:
        nonlocal rendered, truncated
        if current_depth >= depth:
            return
        children = _visible_children(directory)
        for i, entry in enumerate(children):
            if rendered >= max_entries:
                truncated = True
                return
            last = i == len(children) - 1
            connector = "Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ " if last else "Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ "
            name = entry.name + ("/" if entry.is_dir() else "")
            lines.append(f"{prefix}{connector}{name}")
            rendered += 1
            if entry.is_dir():
                extension = "    " if last else "Ã¢â€â€š   "
                _walk(entry, prefix + extension, current_depth + 1)
                if truncated:
                    return

    _walk(target, "", 0)
    if truncated:
        lines.append(f"Ã¢â‚¬Â¦ (truncated at {max_entries} entries)")
    return "\n".join(lines)


# -- find_files ----------------------------------------------------------

def _find_files_impl(
    sandbox: "FileToolSandbox", pattern: str, path: str, max_results: int
) -> FindFilesResponse:
    sandbox.check_read()
    try:
        target = sandbox.resolve_path(path)
    except (ValueError, PermissionError) as e:
        return FindFilesResponse(pattern=pattern, path=path, error=str(e))

    if not target.is_dir():
        return FindFilesResponse(pattern=pattern, path=path,
                                 error=f"'{target}' is not a directory.")

    results: list[str] = []
    truncated = False
    for entry in target.rglob(pattern):
        if entry.is_file() and sandbox.is_allowed(entry):
            rel = str(entry.relative_to(sandbox.root_dir))
            results.append(rel)
            if len(results) >= max_results:
                truncated = True
                break

    return FindFilesResponse(
        pattern=pattern,
        path=str(target.relative_to(sandbox.root_dir)),
        results=results,
        total_found=len(results),
        truncated=truncated,
    )


# -- search_files --------------------------------------------------------

def _search_files_impl(
    sandbox: "FileToolSandbox",
    pattern: str,
    path: str,
    file_pattern: str,
    context_lines: int,
    max_results: int,
    ignore_case: bool,
) -> SearchFilesResponse:
    sandbox.check_read()
    try:
        target = sandbox.resolve_path(path)
    except (ValueError, PermissionError) as e:
        return SearchFilesResponse(pattern=pattern, path=path, file_pattern=file_pattern, error=str(e))

    if not target.is_dir():
        return SearchFilesResponse(pattern=pattern, path=path, file_pattern=file_pattern,
                                   error=f"'{target}' is not a directory.")

    import re

    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return SearchFilesResponse(pattern=pattern, path=path, file_pattern=file_pattern,
                                   error=f"Invalid regex pattern '{pattern}': {e}")

    matches: list[SearchMatch] = []
    count = 0
    truncated = False

    for filepath in target.rglob(file_pattern):
        if not filepath.is_file() or not sandbox.is_allowed(filepath):
            continue
        try:
            if filepath.stat().st_size > _SEARCH_MAX_FILE_BYTES:
                continue
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            continue

        for i, line in enumerate(lines):
            if regex.search(line):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                rel = str(filepath.relative_to(sandbox.root_dir))
                for j in range(start, end):
                    matches.append(SearchMatch(
                        file=rel,
                        line_number=j + 1,
                        line=lines[j].rstrip(),
                        is_context=(j != i),
                    ))
                if context_lines > 0 and end < len(lines):
                    matches.append(SearchMatch(
                        file=rel,
                        line_number=end + 1,
                        line="...",
                        is_context=True,
                    ))
                count += 1
                if count >= max_results:
                    truncated = True
                    break
        if truncated:
            break

    return SearchFilesResponse(
        pattern=pattern,
        path=str(target.relative_to(sandbox.root_dir)),
        file_pattern=file_pattern,
        matches=matches,
        total_matches=count,
        truncated=truncated,
    )


# -- file_info -----------------------------------------------------------

def _file_info_impl(sandbox: "FileToolSandbox", path: str) -> FileInfoResponse:
    sandbox.check_read()
    try:
        target = sandbox.resolve_path(path)
    except (ValueError, PermissionError) as e:
        return FileInfoResponse(path=path, absolute_path="", type="unknown", modified="", error=str(e))

    if not target.exists():
        return FileInfoResponse(path=path, absolute_path=str(target), type="unknown", modified="",
                                error=f"'{target}' does not exist.")

    import time
    stat = target.stat()
    kind = "directory" if target.is_dir() else "file"
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
    size = stat.st_size if target.is_file() else None

    return FileInfoResponse(
        path=str(target.relative_to(sandbox.root_dir)),
        absolute_path=str(target),
        type=kind,
        size_bytes=size,
        modified=mtime,
    )


# Ã¢â€â‚¬Ã¢â€â‚¬ Registration Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

def attach_file_read_tools(toolbox: "ToolBox", sandbox: "FileToolSandbox") -> None:
    """Mount all read-only file tools onto a ToolBox instance."""
    toolbox._file_sandbox = sandbox  # type: ignore[attr-defined]

    procedure_base = sandbox.describe_policy()

    @toolbox.register(
        name="read_file",
        tags=("file", "read"), category="file", risk="low",
        description="Read the raw contents of a file. Supports byte offset and size limits. Returns structured JSON.",
        args_model=ReadFileArgs,
        replaces=[
            BashHint("cat", "read_file(path=...) for raw text"),
            BashHint("head", "read_file(path=..., max_bytes=...), or view_file(start_line=1, end_line=N)"),
            BashHint("tail", "read_file(path=..., offset=...) to skip ahead"),
            BashHint("wc", "read_file(path=...) then count, or file_info(path=...) for size"),
        ],
        procedure=(
            "Read a file's text contents verbatim.\n"
            "- Returns UTF-8 text with replacement characters for invalid bytes.\n"
            "- Use offset to skip the beginning of large files.\n"
            "- max_bytes caps the read; the sandbox enforces its own ceiling.\n"
            "- For making edits, prefer view_file to get exact line-numbered text.\n"
            "- Reply is structured JSON with fields: path, content, bytes_read, truncated, offset.\n"
            f"\n{procedure_base}"
        ),
    )
    def _read_file(db_pool: Any, user_session: dict, path: str, max_bytes: int = 512_000, offset: int = 0) -> ReadFileResponse:
        return _read_file_impl(sandbox, path, max_bytes, offset)

    @toolbox.register(
        name="view_file",
        tags=("file", "read"), category="file", risk="low",
        description=(
            "View a file with line numbers, optionally a [start_line, end_line] range. "
            "Use this before edit_file so you can copy exact text. Returns structured JSON."
        ),
        args_model=ViewFileArgs,
        replaces=[
            BashHint("cat", "view_file(path=...) for line-numbered text"),
            BashHint("bat", "view_file(path=...) for line-numbered text"),
            BashHint("less", "view_file(path=..., start_line=..., end_line=...) Ã¢â‚¬â€ pagers block the shell"),
            BashHint("more", "view_file(path=..., start_line=..., end_line=...)"),
            BashHint("head", "view_file(path=..., start_line=1, end_line=N)"),
        ],
        procedure=(
            "View line-numbered file contents.\n"
            "- Line numbers are 1-based and shown left of a tab; do NOT include them in edit_file's old_str.\n"
            "- Use start_line/end_line to focus on a region of a large file.\n"
            "- max_lines caps how many lines are returned (truncated=true if exceeded).\n"
            "- Reply is structured JSON with fields: path, content, start_line, end_line, total_lines, truncated.\n"
            f"\n{procedure_base}"
        ),
    )
    def _view_file(
        db_pool: Any, user_session: dict,
        path: str, start_line: int = 1, end_line: int | None = None, max_lines: int = 400,
    ) -> ViewFileResponse:
        return _view_file_impl(sandbox, path, start_line, end_line, max_lines)

    @toolbox.register(
        name="list_directory",
        tags=("file", "read", "search"), category="file", risk="low",
        description="List files and directories. Supports glob filtering and optional recursion. Returns structured JSON.",
        args_model=ListDirectoryArgs,
        replaces=[
            BashHint("ls", "list_directory(path=..., pattern='*.py', recursive=...)"),
            BashHint("ll", "list_directory(path=...)"),
            BashHint("dir", "list_directory(path=...)"),
            BashHint("tree", "list_directory(path=..., recursive=true, depth=...)"),
        ],
        procedure=(
            "List directory contents.\n"
            "- Use pattern='*.py' to filter by glob.\n"
            "- Set recursive=True to descend into subdirectories (respects depth).\n"
            "- Reply is structured JSON with fields: path, entries (array of {name, type, size_bytes, relative_path}).\n"
            f"\n{procedure_base}"
        ),
    )
    def _list_directory(
        db_pool: Any, user_session: dict,
        path: str = ".", recursive: bool = False, depth: int = 3, pattern: str = "*",
    ) -> ListDirectoryResponse:
        return _list_directory_impl(sandbox, path, recursive, depth, pattern)

    @toolbox.register(
        name="directory_tree",
        tags=("file", "read", "search"), category="file", risk="low",
        description=(
            "Render a directory as a compact ASCII tree (Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬/Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ connectors). "
            "Returns a plain text tree. Honours the sandbox read policy: files or "
            "subtrees you cannot read are omitted entirely."
        ),
        args_model=DirectoryTreeArgs,
        replaces=[
            BashHint("tree", "directory_tree(path=..., depth=..., show_files=...)"),
        ],
        procedure=(
            "Get a bird's-eye view of a directory's structure as an ASCII tree.\n"
            "- Cheaper than recursive list_directory when you just need the shape.\n"
            "- Directories are always shown; 'pattern' filters file names; "
            "'show_files=false' gives a directories-only outline.\n"
            "- Output is a text tree, already filtered to readable paths.\n"
            f"\n{procedure_base}"
        ),
    )
    def _directory_tree(
        db_pool: Any, user_session: dict,
        path: str = ".", depth: int = 3, pattern: str = "*",
        show_files: bool = True, max_entries: int = 500,
    ) -> str:
        return _directory_tree_impl(sandbox, path, depth, pattern, show_files, max_entries)

    @toolbox.register(
        name="find_files",
        tags=("file", "search"), category="file", risk="low",
        description="Search for files matching a glob pattern recursively. Returns structured JSON.",
        args_model=FindFilesArgs,
        replaces=[
            BashHint("find", "find_files(pattern='*.py', path=...)"),
            BashHint("fd", "find_files(pattern=..., path=...)"),
        ],
        procedure=(
            "Find files by glob pattern (e.g. '*.log', 'test_*').\n"
            "- Recurses into all subdirectories.\n"
            "- Results are relative to the sandbox root.\n"
            "- Reply is structured JSON with fields: pattern, path, results (array), total_found, truncated.\n"
            f"\n{procedure_base}"
        ),
    )
    def _find_files(db_pool: Any, user_session: dict, pattern: str, path: str = ".", max_results: int = 200) -> FindFilesResponse:
        return _find_files_impl(sandbox, pattern, path, max_results)

    @toolbox.register(
        name="search_files",
        tags=("file", "search"), category="file", risk="low",
        description="Search file contents with a regex pattern. Returns structured JSON with matches, file paths, and line numbers.",
        args_model=SearchFilesArgs,
        replaces=[
            BashHint("grep", "search_files(pattern=..., path=...)"),
            BashHint("egrep", "search_files(pattern=..., path=...)"),
            BashHint("awk", "search_files(pattern=...) to extract matching lines"),
        ],
        procedure=(
            "Search file contents using regular expressions.\n"
            "- Use file_pattern to restrict which files are searched.\n"
            "- context_lines adds surrounding lines around each match.\n"
            "- Reply is structured JSON with fields: pattern, path, file_pattern, matches (array of {file, line_number, line, is_context}), total_matches, truncated.\n"
            f"\n{procedure_base}"
        ),
    )
    def _search_files(
        db_pool: Any, user_session: dict,
        pattern: str, path: str = ".", file_pattern: str = "*",
        context_lines: int = 0, max_results: int = 100, ignore_case: bool = False,
    ) -> SearchFilesResponse:
        return _search_files_impl(sandbox, pattern, path, file_pattern, context_lines, max_results, ignore_case)

    @toolbox.register(
        name="file_info",
        tags=("file", "read"), category="file", risk="low",
        description="Get metadata about a file or directory (size, type, modification time). Returns structured JSON.",
        args_model=FileInfoArgs,
        replaces=[
            BashHint("stat", "file_info(path=...)"),
            BashHint("file", "file_info(path=...)"),
            BashHint("wc", "file_info(path=...) for size"),
        ],
        procedure=(
            "Return file metadata.\n"
            "- Includes relative path, absolute path, type, size, and modification time.\n"
            "- Reply is structured JSON with fields: path, absolute_path, type, size_bytes, modified.\n"
            f"\n{procedure_base}"
        ),
    )
    def _file_info(db_pool: Any, user_session: dict, path: str) -> FileInfoResponse:
        return _file_info_impl(sandbox, path)
