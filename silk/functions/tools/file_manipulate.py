"""Manipulation file tools: copy_file, move_file, delete_file, diff_files."""
from __future__ import annotations

import difflib
import shutil
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .command_advice import BashHint

if TYPE_CHECKING:
    from ..tool_box import ToolBox
    from .file_sandbox import FileToolSandbox


# Ã¢â€â‚¬Ã¢â€â‚¬ Pydantic schemas Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

class CopyFileArgs(BaseModel):
    source: str = Field(..., description="Source file or directory (relative to sandbox root).")
    destination: str = Field(..., description="Destination path (relative to sandbox root).")


class MoveFileArgs(BaseModel):
    source: str = Field(..., description="Source file or directory to move (relative to sandbox root).")
    destination: str = Field(..., description="Destination path (relative to sandbox root).")


class DeleteFileArgs(BaseModel):
    path: str = Field(..., description="File or directory to delete (relative to sandbox root).")
    recursive: bool = Field(
        False,
        description="If True, delete directories recursively. Only applies to directories.",
    )


class DiffFilesArgs(BaseModel):
    file_a: str = Field(..., description="First file (relative to sandbox root).")
    file_b: str = Field(..., description="Second file (relative to sandbox root).")
    max_lines: int = Field(
        200,
        gt=0,
        description="Maximum diff lines to return.",
    )


# Ã¢â€â‚¬Ã¢â€â‚¬ Tool implementations Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

def _copy_file_impl(sandbox: "FileToolSandbox", source: str, destination: str) -> str:
    try:
        sandbox.check_write()
        src = sandbox.resolve_path(source)
        dst = sandbox.resolve_path(destination)
        sandbox.check_extension(dst)
        sandbox.assert_writable(dst)
    except (ValueError, PermissionError) as e:
        return str(e)

    if not src.exists():
        return f"Error: Source '{src}' does not exist."

    if sandbox.dry_run:
        return f"[Dry-run] Would copy '{src}' to '{dst}'"


    try:
        with sandbox.lock_paths(src, dst):
            if src.is_dir():
                if dst.exists():
                    dst = dst / src.name  # copy into existing dir
                shutil.copytree(str(src), str(dst))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
    except OSError as e:
        return f"Error copying '{src}' to '{dst}': {e}"

    rel_src = src.relative_to(sandbox.root_dir)
    rel_dst = dst.relative_to(sandbox.root_dir)
    return f"Successfully copied '{rel_src}' to '{rel_dst}'."


def _move_file_impl(sandbox: "FileToolSandbox", source: str, destination: str) -> str:
    try:
        sandbox.check_write()
        src = sandbox.resolve_path(source)
        dst = sandbox.resolve_path(destination)
        sandbox.check_extension(dst)
        sandbox.assert_writable(dst)
    except (ValueError, PermissionError) as e:
        return str(e)

    if not src.exists():
        return f"Error: Source '{src}' does not exist."

    if sandbox.dry_run:
        return f"[Dry-run] Would move '{src}' to '{dst}'"


    try:
        with sandbox.lock_paths(src, dst):
            if dst.exists():
                return f"Error: Destination '{dst}' already exists."
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    except OSError as e:
        return f"Error moving '{src}' to '{dst}': {e}"

    rel_src = src.relative_to(sandbox.root_dir)
    rel_dst = dst.relative_to(sandbox.root_dir)
    return f"Successfully moved '{rel_src}' to '{rel_dst}'."


def _delete_file_impl(sandbox: "FileToolSandbox", path: str, recursive: bool) -> str:
    try:
        sandbox.check_delete()
        target = sandbox.resolve_path(path)
        sandbox.assert_writable(target)
    except (ValueError, PermissionError) as e:
        return str(e)

    if not target.exists():
        return f"Error: '{target}' does not exist."

    if sandbox.dry_run:
        return f"[Dry-run] Would delete '{target}' (recursive={recursive})"


    try:
        with sandbox.lock_paths(target):
            if target.is_dir():
                if recursive:
                    shutil.rmtree(str(target))
                else:
                    target.rmdir()  # only empty dirs without recursive
            else:
                target.unlink()
    except OSError as e:
        return f"Error deleting '{target}': {e}"

    return f"Successfully deleted '{target.relative_to(sandbox.root_dir)}'."


def _diff_files_impl(sandbox: "FileToolSandbox", file_a: str, file_b: str, max_lines: int) -> str:
    try:
        sandbox.check_read()
        a = sandbox.resolve_path(file_a)
        b = sandbox.resolve_path(file_b)
        sandbox.check_extension(a)
        sandbox.check_extension(b)
    except (ValueError, PermissionError) as e:
        return str(e)

    if not a.is_file():
        return f"Error: '{a}' is not a file or does not exist."
    if not b.is_file():
        return f"Error: '{b}' is not a file or does not exist."

    try:
        with open(a, "r", encoding="utf-8", errors="replace") as fa:
            lines_a = fa.readlines()
        with open(b, "r", encoding="utf-8", errors="replace") as fb:
            lines_b = fb.readlines()
    except OSError as e:
        return f"Error reading files: {e}"


    diff = difflib.unified_diff(
        lines_a,
        lines_b,
        fromfile=str(a.relative_to(sandbox.root_dir)),
        tofile=str(b.relative_to(sandbox.root_dir)),
        lineterm="",
    )

    diff_lines = []
    for line in diff:
        diff_lines.append(line)
        if len(diff_lines) >= max_lines:
            diff_lines.append(f"[Diff truncated at {max_lines} lines]")
            break

    if not diff_lines:
        rel_a = a.relative_to(sandbox.root_dir)
        rel_b = b.relative_to(sandbox.root_dir)
        return f"Files '{rel_a}' and '{rel_b}' are identical."

    return "\n".join(diff_lines)


# Ã¢â€â‚¬Ã¢â€â‚¬ Registration Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

def attach_file_manipulate_tools(toolbox: "ToolBox", sandbox: "FileToolSandbox") -> None:
    """Mount all manipulation file tools onto a ToolBox instance."""
    toolbox._file_sandbox = sandbox  # type: ignore[attr-defined]

    procedure_base = sandbox.describe_policy()

    @toolbox.register(
        name="copy_file",
        tags=("file", "write"), category="file", risk="medium",
        description="Copy a file or directory to a new location within the sandbox.",
        args_model=CopyFileArgs,
        replaces=[BashHint("cp", "copy_file(source=..., destination=...)")],
        procedure=(
            "Copy a file or directory.\n"
            "- Source must exist; destination parent dirs are created automatically.\n"
            "- For directories, copies recursively.\n"
            f"\n{procedure_base}"
        ),
    )
    def _copy_file(db_pool: Any, user_session: dict, source: str, destination: str) -> str:
        return _copy_file_impl(sandbox, source, destination)

    @toolbox.register(
        name="move_file",
        tags=("file", "write"), category="file", risk="medium",
        description="Move or rename a file or directory within the sandbox.",
        args_model=MoveFileArgs,
        replaces=[BashHint("mv", "move_file(source=..., destination=...)")],
        procedure=(
            "Move or rename a file or directory.\n"
            "- Destination must not already exist.\n"
            "- Destination parent dirs are created automatically.\n"
            f"\n{procedure_base}"
        ),
    )
    def _move_file(db_pool: Any, user_session: dict, source: str, destination: str) -> str:
        return _move_file_impl(sandbox, source, destination)

    @toolbox.register(
        name="delete_file",
        tags=("file", "write"), category="file", risk="high",
        description="Delete a file or directory. Directories require recursive=True. Delete is disabled by default in sandbox policy.",
        args_model=DeleteFileArgs,
        replaces=[
            BashHint("rm", "delete_file(path=..., recursive=...)"),
            BashHint("rmdir", "delete_file(path=...)"),
        ],
        procedure=(
            "Delete a file or directory.\n"
            "- For files: unlinks immediately.\n"
            "- For directories: requires recursive=True to remove non-empty dirs.\n"
            "- Ã¢Å¡Â Ã¯Â¸Â This operation is typically disabled by default Ã¢â‚¬â€ check sandbox policy.\n"
            f"\n{procedure_base}"
        ),
    )
    def _delete_file(db_pool: Any, user_session: dict, path: str, recursive: bool = False) -> str:
        return _delete_file_impl(sandbox, path, recursive)

    @toolbox.register(
        name="diff_files",
        tags=("file", "read"), category="file", risk="low",
        description="Show a unified diff between two text files.",
        args_model=DiffFilesArgs,
        replaces=[BashHint("diff", "diff_files(file_a=..., file_b=...)")],
        procedure=(
            "Compare two files and show their differences as a unified diff.\n"
            "- Both files must exist and be text files.\n"
            "- Returns identical message if files have no differences.\n"
            f"\n{procedure_base}"
        ),
    )
    def _diff_files(db_pool: Any, user_session: dict, file_a: str, file_b: str, max_lines: int = 200) -> str:
        return _diff_files_impl(sandbox, file_a, file_b, max_lines)
