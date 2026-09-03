"""Write file tools: write_file, append_file, create_directory, edit_file, insert_text."""
from __future__ import annotations

import hashlib
import os
import tempfile
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .command_advice import BashHint, REDIRECT

if TYPE_CHECKING:
    from pathlib import Path
    from ..tool_box import ToolBox
    from .file_sandbox import FileToolSandbox


# Ã¢â€â‚¬Ã¢â€â‚¬ Pydantic schemas Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

class WriteFileArgs(BaseModel):
    path: str = Field(..., description="Path to write to (relative to sandbox root). Parent directories are created automatically.")
    content: str = Field(..., description="Content to write to the file.")
    expected_sha256: str = Field(
        "",
        description=(
            "Optional precondition: only write if the file's current "
            "SHA-256 is exactly this. Use 'absent' to require that the "
            "file does not exist yet. Leave empty for a blind overwrite."
        ),
    )


class AppendFileArgs(BaseModel):
    path: str = Field(..., description="Path of the file to append to (relative to sandbox root).")
    content: str = Field(..., description="Content to append to the end of the file.")


class CreateDirectoryArgs(BaseModel):
    path: str = Field(..., description="Path of the directory to create (relative to sandbox root). Parent directories are created automatically.")


class EditFileArgs(BaseModel):
    path: str = Field(..., description="Path to the file to edit (relative to sandbox root).")
    old_str: str = Field(
        ...,
        min_length=1,
        description=(
            "Exact text to find and replace. Must match a unique, non-overlapping "
            "region of the file unless replace_all is true. Include enough surrounding "
            "context (indentation, neighbouring lines) to disambiguate Ã¢â‚¬â€ a bare keyword "
            "or short phrase will likely match multiple times."
        ),
    )
    new_str: str = Field(..., description="Replacement text that will replace the matched old_str.")
    replace_all: bool = Field(
        False,
        description="If true, replace every occurrence of old_str instead of erroring on multiple matches.",
    )


class InsertTextArgs(BaseModel):
    path: str = Field(..., description="Path to the file to insert into (relative to sandbox root). Must already exist.")
    after_line: int = Field(
        ...,
        ge=0,
        description="Insert the text AFTER this 1-based line number. Use 0 to insert at the very beginning of the file.",
    )
    text: str = Field(..., description="Text to insert. A trailing newline is added if missing.")


# Ã¢â€â‚¬Ã¢â€â‚¬ Pydantic response schemas Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

class EditFileResponse(BaseModel):
    """Structured reply from edit_file."""
    path: str = Field(..., description="Relative path of the file that was edited (or attempted).")
    matches_found: int = Field(..., description="Number of times old_str was found in the file.")
    replaced: bool = Field(..., description="True if the replacement was performed.")
    replacements: int = Field(0, description="Number of occurrences actually replaced.")
    error: str | None = Field(None, description="Error message if replacement failed.")
    match_lines: list[int] = Field(
        default_factory=list,
        description="1-based line numbers where old_str was found (populated when matches_found > 1).",
    )
    preview: str = Field(
        "",
        description="First 5 lines of the file (populated when matches_found == 0, to help the model retry).",
    )


class InsertTextResponse(BaseModel):
    """Structured reply from insert_text."""
    path: str = Field(..., description="Relative path of the file that was modified (or attempted).")
    inserted: bool = Field(..., description="True if the text was inserted.")
    after_line: int = Field(..., description="The 1-based line the text was inserted after (0 = start).")
    lines_inserted: int = Field(0, description="Number of lines added.")
    error: str | None = Field(None, description="Error message if the insert failed.")


# Ã¢â€â‚¬Ã¢â€â‚¬ Internal helpers Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

def _atomic_write(target: "Path", data: str) -> None:
    """
    Write *data* to *target* atomically: write a sibling temp file, fsync, then
    os.replace() into place. A crash mid-write leaves the original untouched.
    newline="" preserves the caller's line endings exactly (no translation).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp_", suffix=target.suffix or ".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _find_all(text: str, pattern: str) -> list[int]:
    """Return all start indices of *pattern* in *text* (overlapping)."""
    indices = []
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx == -1:
            break
        indices.append(idx)
        start = idx + 1  # overlapping: a self-overlapping pattern is still ambiguous
    return indices


def _line_col(text: str, offset: int) -> tuple[int, int]:
    """Return (1-based line, 1-based column) for character offset *offset*."""
    line = text[:offset].count("\n") + 1
    col = offset - text.rfind("\n", 0, offset) - 1
    return line, col


def _too_big(sandbox: "FileToolSandbox", data: str) -> str | None:
    """Return an error string if *data* exceeds the sandbox write ceiling, else None."""
    size = len(data.encode("utf-8"))
    if size > sandbox.max_write_bytes:
        return (
            f"Error: Content ({size} bytes) exceeds maximum write size "
            f"({sandbox.max_write_bytes} bytes)."
        )
    return None


# Ã¢â€â‚¬Ã¢â€â‚¬ Tool implementations Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

#: What ``expected_sha256`` means when the file should not exist yet.
ABSENT = "absent"


def _digest(target: "Path") -> str:
    """The file's SHA-256, or :data:`ABSENT` when there is no file."""
    try:
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except FileNotFoundError:
        return ABSENT


def _write_file_impl(sandbox: "FileToolSandbox", path: str, content: str,
                     expected_sha256: str = "") -> str:
    try:
        sandbox.check_write()
        target = sandbox.resolve_path(path)
        sandbox.check_extension(target)
        sandbox.assert_writable(target)
    except (ValueError, PermissionError) as e:
        return str(e)

    if sandbox.dry_run:
        return f"[Dry-run] Would write {len(content)} bytes to '{target}'"

    size_err = _too_big(sandbox, content)
    if size_err:
        return size_err

    # The compare-and-swap precondition (D68, §22 q8). Checked *inside*
    # the lock the write already takes, so nothing can land between the
    # comparison and the replace. This is the answer to blind overwrites
    # between agents: an optimistic precondition the caller opts into,
    # not a claim that makes one agent's write policy depend on another
    # agent's runtime state.
    expected = str(expected_sha256 or "").strip().lower()
    try:
        with sandbox.lock_paths(target):
            if expected:
                actual = _digest(target)
                if actual != expected:
                    rel = target.relative_to(sandbox.root_dir)
                    if actual == ABSENT:
                        found = "the file does not exist"
                    elif expected == ABSENT:
                        found = f"it already exists (sha256 {actual})"
                    else:
                        found = f"it is now sha256 {actual}"
                    return (
                        f"Error: '{rel}' does not match the precondition -- "
                        f"{found}. Someone else changed it since you read "
                        "it. Read it again and re-apply your change on top "
                        "of what is there now."
                    )
            _atomic_write(target, content)
    except OSError as e:
        return f"Error writing '{target}': {e}"

    return f"Successfully wrote {len(content)} bytes to '{target.relative_to(sandbox.root_dir)}'."


def _append_file_impl(sandbox: "FileToolSandbox", path: str, content: str) -> str:
    try:
        sandbox.check_write()
        target = sandbox.resolve_path(path)
        sandbox.check_extension(target)
        sandbox.assert_writable(target)
    except (ValueError, PermissionError) as e:
        return str(e)

    if not target.exists():
        return f"Error: '{target}' does not exist. Use write_file to create it first."

    if sandbox.dry_run:
        return f"[Dry-run] Would append {len(content)} bytes to '{target}'"

    size_err = _too_big(sandbox, content)
    if size_err:
        return size_err

    try:
        with sandbox.lock_paths(target):
            with open(target, "a", encoding="utf-8", newline="") as f:
                f.write(content)
    except OSError as e:
        return f"Error appending to '{target}': {e}"

    return f"Successfully appended {len(content)} bytes to '{target.relative_to(sandbox.root_dir)}'."


def _create_directory_impl(sandbox: "FileToolSandbox", path: str) -> str:
    try:
        sandbox.check_write()
        target = sandbox.resolve_path(path)
        sandbox.assert_writable(target)
    except (ValueError, PermissionError) as e:
        return str(e)

    if sandbox.dry_run:
        return f"[Dry-run] Would create directory '{target}'"

    try:
        with sandbox.lock_paths(target):
            target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"Error creating directory '{target}': {e}"

    if target.is_dir():
        return f"Successfully created directory '{target.relative_to(sandbox.root_dir)}'."
    return f"Error: '{target.relative_to(sandbox.root_dir)}' exists but is not a directory."


def _edit_file_impl(
    sandbox: "FileToolSandbox", path: str, old_str: str, new_str: str, replace_all: bool = False
) -> EditFileResponse:
    """Replace old_str with new_str.

    Default: requires exactly one match; 0 or 2+ matches return error details so
    the LLM can retry. With replace_all=True, every occurrence is replaced.
    """
    try:
        sandbox.check_write()
        target = sandbox.resolve_path(path)
        sandbox.check_extension(target)
        sandbox.assert_writable(target)
    except (ValueError, PermissionError) as e:
        return EditFileResponse(path=path, matches_found=0, replaced=False, error=str(e))

    if not target.is_file():
        rel = str(target.relative_to(sandbox.root_dir)) if target.exists() else path
        return EditFileResponse(path=rel, matches_found=0, replaced=False,
                                error=f"'{target}' is not a file or does not exist.")

    with sandbox.lock_paths(target):
        try:
            with open(target, "r", encoding="utf-8", newline="") as f:
                content = f.read()
        except UnicodeDecodeError:
            rel = str(target.relative_to(sandbox.root_dir))
            return EditFileResponse(path=rel, matches_found=0, replaced=False,
                                    error=f"'{rel}' is not valid UTF-8 text and cannot be edited.")
        except OSError as e:
            rel = str(target.relative_to(sandbox.root_dir))
            return EditFileResponse(path=rel, matches_found=0, replaced=False,
                                    error=f"Error reading '{target}': {e}")

        rel = str(target.relative_to(sandbox.root_dir))

        # Guard: empty old_str would cause an infinite loop in _find_all.
        if old_str == "":
            return EditFileResponse(
                path=rel, matches_found=len(content) + 1, replaced=False,
                error="old_str is empty Ã¢â‚¬â€ would match at every position.",
            )

        # Normalise line endings so \r\n / \n / \r all behave identically.
        # The LLM sends \n; on Windows files may contain \r\n.
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        old_str = old_str.replace("\r\n", "\n").replace("\r", "\n")

        matches = _find_all(content, old_str)

        if len(matches) == 0:
            preview = "\n".join(content.split("\n")[:5])
            return EditFileResponse(
                path=rel, matches_found=0, replaced=False,
                error="old_str not found. The text to replace must match exactly (including whitespace and indentation).",
                preview=preview,
            )

        if len(matches) > 1 and not replace_all:
            line_nums = [_line_col(content, idx)[0] for idx in matches]
            return EditFileResponse(
                path=rel, matches_found=len(matches), replaced=False,
                error=(
                    f"old_str matches {len(matches)} times Ã¢â‚¬â€ replacement would be ambiguous. "
                    f"Provide more surrounding context so old_str matches once, or set replace_all=true."
                ),
                match_lines=line_nums[:20],
            )

        # Perform the replacement (one match, or all when replace_all).
        if replace_all:
            new_content = content.replace(old_str, new_str)
            replacements = len(matches)
        else:
            start = matches[0]
            new_content = content[:start] + new_str + content[start + len(old_str):]
            replacements = 1

        if sandbox.dry_run:
            return EditFileResponse(path=rel, matches_found=len(matches), replaced=False, replacements=0,
                                    error=f"[Dry-run] Would replace {replacements} occurrence(s) in '{rel}'")

        size_err = _too_big(sandbox, new_content)
        if size_err:
            return EditFileResponse(path=rel, matches_found=len(matches), replaced=False, error=size_err)

        try:
            _atomic_write(target, new_content)
        except OSError as e:
            return EditFileResponse(path=rel, matches_found=len(matches), replaced=False,
                                    error=f"Error writing '{target}': {e}")

        return EditFileResponse(path=rel, matches_found=len(matches), replaced=True, replacements=replacements)


def _insert_text_impl(sandbox: "FileToolSandbox", path: str, after_line: int, text: str) -> InsertTextResponse:
    try:
        sandbox.check_write()
        target = sandbox.resolve_path(path)
        sandbox.check_extension(target)
        sandbox.assert_writable(target)
    except (ValueError, PermissionError) as e:
        return InsertTextResponse(path=path, inserted=False, after_line=after_line, error=str(e))

    if not target.is_file():
        rel = str(target.relative_to(sandbox.root_dir)) if target.exists() else path
        return InsertTextResponse(path=rel, inserted=False, after_line=after_line,
                                  error=f"'{target}' is not a file or does not exist. Use write_file to create it first.")

    with sandbox.lock_paths(target):
        try:
            with open(target, "r", encoding="utf-8", newline="") as f:
                original = f.read()
        except UnicodeDecodeError:
            rel = str(target.relative_to(sandbox.root_dir))
            return InsertTextResponse(path=rel, inserted=False, after_line=after_line,
                                      error=f"'{rel}' is not valid UTF-8 text and cannot be edited.")
        except OSError as e:
            rel = str(target.relative_to(sandbox.root_dir))
            return InsertTextResponse(path=rel, inserted=False, after_line=after_line,
                                      error=f"Error reading '{target}': {e}")

        rel = str(target.relative_to(sandbox.root_dir))
        lines = original.splitlines(keepends=True)
        total = len(lines)

        if after_line > total:
            return InsertTextResponse(path=rel, inserted=False, after_line=after_line,
                                      error=f"after_line {after_line} is beyond end of file ({total} lines).")

        block = text if text.endswith("\n") else text + "\n"
        # Guard: if the preceding line lacks a trailing newline, add one so we don't
        # accidentally join the insert onto it.
        if 0 < after_line <= total and lines and not lines[after_line - 1].endswith("\n"):
            lines[after_line - 1] = lines[after_line - 1] + "\n"

        new_content = "".join(lines[:after_line]) + block + "".join(lines[after_line:])

        size_err = _too_big(sandbox, new_content)
        if size_err:
            return InsertTextResponse(path=rel, inserted=False, after_line=after_line, error=size_err)

        if sandbox.dry_run:
            return InsertTextResponse(path=rel, inserted=False, after_line=after_line,
                                      error=f"[Dry-run] Would insert text after line {after_line} in '{rel}'")

        try:
            _atomic_write(target, new_content)
        except OSError as e:
            return InsertTextResponse(path=rel, inserted=False, after_line=after_line,
                                      error=f"Error writing '{target}': {e}")

        return InsertTextResponse(path=rel, inserted=True, after_line=after_line,
                                  lines_inserted=len(block.splitlines()))


# Ã¢â€â‚¬Ã¢â€â‚¬ Registration Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

def attach_file_write_tools(toolbox: "ToolBox", sandbox: "FileToolSandbox") -> None:
    """Mount all write file tools onto a ToolBox instance."""
    toolbox._file_sandbox = sandbox  # type: ignore[attr-defined]

    procedure_base = sandbox.describe_policy()

    @toolbox.register(
        name="write_file",
        tags=("file", "write"), category="file", risk="medium",
        description="Write content to a file (atomic). Creates parent directories automatically. Overwrites existing files.",
        args_model=WriteFileArgs,
        replaces=[
            BashHint("touch", "write_file(path=..., content='')"),
            BashHint("nano", "write_file(path=..., content=...) Ã¢â‚¬â€ nano is interactive and blocks the shell"),
            BashHint("vi", "write_file(path=..., content=...) Ã¢â‚¬â€ vi is interactive and blocks the shell"),
            BashHint("vim", "write_file(path=..., content=...) Ã¢â‚¬â€ vim is interactive and blocks the shell"),
            BashHint(REDIRECT, "write_file(path=..., content=...) instead of `>` redirection"),
        ],
        procedure=(
            "Write content to a file.\n"
            "- Creates parent directories if they don't exist.\n"
            "- Overwrites the file if it already exists (written atomically).\n"
            "- Content size is capped by the sandbox's max_write_bytes policy.\n"
            "- expected_sha256: optional. Pass the digest you read to "
            "refuse the write if someone changed the file since -- use it "
            "when another agent may be working in the same tree, or "
            "'absent' to create a file only if it is not there yet.\n"
            f"\n{procedure_base}"
        ),
    )
    def _write_file(db_pool: Any, user_session: dict, path: str, content: str,
                    expected_sha256: str = "") -> str:
        return _write_file_impl(sandbox, path, content, expected_sha256)

    @toolbox.register(
        name="append_file",
        tags=("file", "write"), category="file", risk="medium",
        description="Append content to an existing file. The file must already exist.",
        args_model=AppendFileArgs,
        replaces=[
            BashHint(REDIRECT, "append_file(path=..., content=...) instead of `>>` redirection"),
        ],
        procedure=(
            "Append text to the end of an existing file.\n"
            "- The file must already exist; use write_file to create it first.\n"
            "- Content size is capped by the sandbox's max_write_bytes policy.\n"
            f"\n{procedure_base}"
        ),
    )
    def _append_file(db_pool: Any, user_session: dict, path: str, content: str) -> str:
        return _append_file_impl(sandbox, path, content)

    @toolbox.register(
        name="create_directory",
        tags=("file", "write"), category="file", risk="medium",
        description="Create a directory (and any missing parent directories).",
        args_model=CreateDirectoryArgs,
        replaces=[
            BashHint("mkdir", "create_directory(path=...)"),
        ],
        procedure=(
            "Create a directory, including all missing parent directories.\n"
            "- Succeeds silently if the directory already exists.\n"
            f"\n{procedure_base}"
        ),
    )
    def _create_directory(db_pool: Any, user_session: dict, path: str) -> str:
        return _create_directory_impl(sandbox, path)

    @toolbox.register(
        name="edit_file",
        tags=("file", "write"), category="file", risk="medium",
        description=(
            "Replace an exact text match in a file (atomic). old_str must match exactly once Ã¢â‚¬â€ "
            "include enough surrounding context to disambiguate Ã¢â‚¬â€ unless replace_all is set. "
            "Returns structured JSON with match count, replacement status, and error details."
        ),
        args_model=EditFileArgs,
        replaces=[
            BashHint("sed", "edit_file(path=..., old_str=..., new_str=...) for exact replacements"),
            BashHint("awk", "edit_file(path=..., old_str=..., new_str=...) to modify in place"),
            BashHint("nano", "edit_file(path=..., old_str=..., new_str=...) Ã¢â‚¬â€ nano blocks the shell"),
            BashHint("vi", "edit_file(path=..., old_str=..., new_str=...) Ã¢â‚¬â€ vi blocks the shell"),
            BashHint("vim", "edit_file(path=..., old_str=..., new_str=...) Ã¢â‚¬â€ vim blocks the shell"),
            BashHint("sed", "edit_file(path=..., old_str=..., new_str=...) Ã¢â‚¬â€ reliable, drift-free in-place edit", in_place=True),
            BashHint("awk", "edit_file(path=..., old_str=..., new_str=...) Ã¢â‚¬â€ reliable, drift-free in-place edit", in_place=True),
        ],
        procedure=(
            "Surgical text replacement in a file.\n"
            "- old_str must match a unique region of the file (or set replace_all=true).\n"
            "- Include surrounding lines and exact indentation to disambiguate.\n"
            "- If old_str matches 0 times: error + preview of first 5 lines Ã¢â‚¬â€ view_file for exact text.\n"
            "- If old_str matches 2+ times and replace_all is false: error + match line numbers.\n"
            "- Reply fields: path, matches_found, replaced, replacements, error, match_lines, preview.\n"
            "- Safer than line-number edits because it never drifts; the write is atomic.\n"
            f"\n{procedure_base}"
        ),
    )
    def _edit_file(
        db_pool: Any, user_session: dict, path: str, old_str: str, new_str: str, replace_all: bool = False
    ) -> EditFileResponse:
        return _edit_file_impl(sandbox, path, old_str, new_str, replace_all)

    @toolbox.register(
        name="insert_text",
        tags=("file", "write"), category="file", risk="medium",
        description=(
            "Insert text after a given 1-based line number (atomic). Use after_line=0 to insert "
            "at the start of the file. The file must already exist. Returns structured JSON."
        ),
        args_model=InsertTextArgs,
        procedure=(
            "Insert a block of text at a line boundary.\n"
            "- after_line=0 inserts at the very top; after_line=N inserts below line N.\n"
            "- after_line may not exceed the file's line count.\n"
            "- A trailing newline is added to the inserted text if missing.\n"
            "- Reply fields: path, inserted, after_line, lines_inserted, error.\n"
            f"\n{procedure_base}"
        ),
    )
    def _insert_text(db_pool: Any, user_session: dict, path: str, after_line: int, text: str) -> InsertTextResponse:
        return _insert_text_impl(sandbox, path, after_line, text)
