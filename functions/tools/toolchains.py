# -*- coding: utf-8 -*-
"""Configured external toolchains as structured agent tools — Qt-free.

A **toolchain** is a configured executable (a Python interpreter/venv,
ruff, mypy, radon, maturin, cargo, …) probed and frozen into a
:class:`ToolchainEnv` by the Toolchain node. Tools are then generated
from declarative :class:`CommandSpec` packs: each spec becomes one
registered ToolBox tool with a pydantic argument schema, so the model
can only vary whitelisted fields — the executable and the argv skeleton
live in the registration closure where the model cannot reach them
(implementation space); path arguments are vetted through the sandbox
(argument space).

Containment honesty: these tools execute real processes. ``run_python``
and the build tools (`cargo`, `maturin` — build.rs / proc-macros /
setup.py run arbitrary project code) are ``risk="high"`` and are gated
by the capability/risk axis (ToolSet / Role), not by path policy.
"""
from __future__ import annotations

import builtins

import shutil
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Optional

from pydantic import BaseModel, Field, create_model

from .file_locks import write_gate

if TYPE_CHECKING:
    from ..tool_box import ToolBox
    from .file_sandbox import FileToolSandbox


class ToolchainError(RuntimeError):
    """Raised when a toolchain cannot be located or probed."""


@dataclass(frozen=True)
class ToolchainEnv:
    """A resolved, probed external executable."""

    id: str                      # "python", "ruff", "cargo", ...
    executable: str              # absolute path to the binary
    version: str = ""            # probed version line
    label: str = ""              # display label ("venvA python 3.13")
    env: Mapping[str, str] = field(default_factory=dict)  # extra env vars

    def describe(self) -> str:
        return f"{self.label or self.id} ({self.version or 'version unknown'})"


#: Known toolchains: candidate executable names + how to probe a version.
KNOWN_TOOLCHAINS: dict[str, dict[str, Any]] = {
    "python": {"names": ["python", "python3"], "version_args": ["--version"]},
    "ruff": {"names": ["ruff"], "version_args": ["--version"]},
    # The node linter (spec D78). A human author gets code review; an
    # agent author gets this, which is why it belongs in the toolchain
    # beside ruff and mypy rather than in a script nobody runs.
    "weave_lint": {"names": ["weave-lint", "weave_lint"],
                   "version_args": ["--help"]},
    "mypy": {"names": ["mypy"], "version_args": ["--version"]},
    "radon": {"names": ["radon"], "version_args": ["--version"]},
    "maturin": {"names": ["maturin"], "version_args": ["--version"]},
    "cargo": {"names": ["cargo"], "version_args": ["--version"]},
}


def probe_toolchain(
    kind: str,
    executable: Optional[str] = None,
    timeout: float = 10.0,
) -> ToolchainEnv:
    """Locate (PATH or explicit path) and version-probe a toolchain.

    Raises :class:`ToolchainError` with a actionable message when the
    executable is missing or does not answer the version probe.
    """
    known = KNOWN_TOOLCHAINS.get(kind)
    if known is None:
        raise ToolchainError(
            f"Unknown toolchain '{kind}'. Known: {', '.join(sorted(KNOWN_TOOLCHAINS))}"
        )

    exe = (executable or "").strip()
    if not exe:
        for name in known["names"]:
            found = shutil.which(name)
            if found:
                exe = found
                break
        if not exe:
            raise ToolchainError(
                f"'{kind}' not found on PATH — set an explicit executable."
            )
    else:
        resolved = shutil.which(exe)
        if resolved is None:
            raise ToolchainError(f"Executable not found: {exe}")
        exe = resolved

    try:
        proc = subprocess.run(
            [exe, *known["version_args"]],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ToolchainError(f"Probing '{exe}' failed: {e}") from e
    if proc.returncode != 0:
        raise ToolchainError(
            f"'{exe} {' '.join(known['version_args'])}' exited "
            f"{proc.returncode}: {(proc.stderr or proc.stdout).strip()[:200]}"
        )

    version = (proc.stdout or proc.stderr).strip().splitlines()[0]
    return ToolchainEnv(id=kind, executable=exe, version=version, label=kind)


# ── declarative command specs ─────────────────────────────────────────────


class ParamSpec(BaseModel):
    """One whitelisted, model-controllable parameter of a command."""

    name: str
    #: What the model may put here. Named ``type`` on the wire, so the
    #: annotation is quoted -- an unquoted `self.type` lookup reads as a
    #: type alias to a checker.
    type: "Literal['str', 'int', 'bool', 'path']" = "str"
    description: str = ""
    required: bool = False
    default: Any = None
    flag: Optional[str] = None   # None → positional; "--flag" → flagged

    def python_type(self) -> "builtins.type":
        """The Python type this parameter accepts.

        The return annotation is qualified because the class has a field
        called ``type``, and inside the class body that name is the field,
        not the builtin.
        """
        kinds: dict[str, type] = {"str": str, "int": int, "bool": bool,
                                  "path": str}
        return kinds[str(self.type)]


class CommandSpec(BaseModel):
    """One toolchain subcommand exposed as a structured tool."""

    tool_name: str
    toolchain_id: str
    description: str
    base_args: list[str] = Field(default_factory=list)
    params: list[ParamSpec] = Field(default_factory=list)
    category: str = "code"
    tags: list[str] = Field(default_factory=list)
    risk: str = "high"
    timeout_s: float = 120.0
    output_limit: int = 24_000   # chars of combined output fed to the model
    writes_files: bool = False
    """Whether this command may rewrite files in the sandbox root.

    True makes the tool take the root's write gate exclusively for the
    subprocess's duration, so a formatter cannot interleave with another
    agent's ``edit_file`` on the same tree (spec D67 tier 2, closes G19).
    Coarse on purpose: nothing can know which files a subprocess touches,
    and a run that only *might* write (``ruff format --check``) is still
    declared a writer -- the cost is a short wait, the alternative is a
    lost update nobody is told about.
    """


def build_argv(
    spec: CommandSpec,
    env: ToolchainEnv,
    args: dict[str, Any],
    sandbox: Optional["FileToolSandbox"] = None,
) -> list[str]:
    """Assemble the subprocess argv from a spec + validated arguments.

    Path-typed parameters are resolved through the sandbox (raising on
    escapes); bools become bare flags; everything else is stringified.
    """
    argv = [env.executable, *spec.base_args]
    for param in spec.params:
        value = args.get(param.name, param.default)
        if value is None or (param.type != "bool" and value == ""):
            continue
        if param.type == "bool":
            if value and param.flag:
                argv.append(param.flag)
            continue
        if param.type == "path" and sandbox is not None:
            value = str(sandbox.resolve_path(str(value)))
        if param.flag:
            argv.extend([param.flag, str(value)])
        else:
            argv.append(str(value))
    return argv


def _format_output(proc: subprocess.CompletedProcess, limit: int) -> str:
    parts = [f"exit code: {proc.returncode}"]
    if proc.stdout:
        parts.append(f"--- stdout ---\n{proc.stdout}")
    if proc.stderr:
        parts.append(f"--- stderr ---\n{proc.stderr}")
    text = "\n".join(parts)
    if len(text) > limit:
        text = text[:limit] + f"\n… (truncated at {limit} chars)"
    return text


# ── built-in spec packs ───────────────────────────────────────────────────

_PATH_PARAM = ParamSpec(
    name="path", type="path", default=".",
    description="File or directory to operate on, inside the sandbox.",
)

SPEC_PACKS: dict[str, list[CommandSpec]] = {
    "python": [
        CommandSpec(
            tool_name="run_python",
            writes_files=True,
            toolchain_id="python",
            description=(
                "Execute a Python code snippet with the configured "
                "interpreter (isolated mode, cwd = sandbox root). Use "
                "print() to produce output. The code runs as a real "
                "process — request this tool only when computation or "
                "library access is genuinely needed."
            ),
            base_args=["-I", "-c"],
            params=[ParamSpec(
                name="code", type="str", required=True,
                description="Python source code to execute.",
            )],
            category="code", tags=["code", "exec", "python"],
            risk="high", timeout_s=60.0,
        ),
    ],
    "ruff": [
        CommandSpec(
            tool_name="ruff_check",
            toolchain_id="ruff",
            description="Lint Python files with ruff (no fixes applied).",
            base_args=["check", "--no-fix", "--output-format=concise"],
            params=[_PATH_PARAM],
            category="lint", tags=["lint", "python", "ruff"],
            risk="low", timeout_s=120.0,
        ),
        CommandSpec(
            tool_name="ruff_format",
            writes_files=True,
            toolchain_id="ruff",
            description=(
                "Format Python files with ruff. With check=true only "
                "reports what would change."
            ),
            base_args=["format"],
            params=[
                _PATH_PARAM,
                ParamSpec(name="check", type="bool", default=False,
                          flag="--check",
                          description="Only check, do not rewrite files."),
            ],
            category="lint", tags=["format", "python", "ruff"],
            risk="medium", timeout_s=120.0,
        ),
    ],
    "weave_lint": [
        CommandSpec(
            tool_name="weave_lint_check",
            toolchain_id="weave_lint",
            description=(
                "Check Weave node and widget classes: port declarations, "
                "state versioning (WV520-WV522), and the node/widget "
                "contract. Run this before loading any suite you wrote -- "
                "a state-version finding means saved graphs would not "
                "survive the load, and the load will be refused."
            ),
            base_args=["--no-color"],
            params=[
                _PATH_PARAM,
                ParamSpec(name="select", type="str", default="",
                          flag="--select",
                          description="Only these codes, e.g. 'WV521'."),
                ParamSpec(name="fmt", type="str", default="line",
                          flag="--format",
                          description="human | line | json | github."),
            ],
            category="lint", tags=["lint", "nodes", "weave", "state"],
            risk="low", timeout_s=300.0,
        ),
    ],
    "mypy": [
        CommandSpec(
            tool_name="mypy_check",
            toolchain_id="mypy",
            description="Static type-check Python files with mypy.",
            base_args=[],
            params=[_PATH_PARAM],
            category="lint", tags=["types", "python", "mypy"],
            risk="medium",  # mypy plugins configured in the project can execute code
            timeout_s=300.0,
        ),
    ],
    "radon": [
        CommandSpec(
            tool_name="radon_cc",
            toolchain_id="radon",
            description="Cyclomatic complexity report (radon cc).",
            base_args=["cc", "-s", "-a"],
            params=[_PATH_PARAM],
            category="lint", tags=["metrics", "python", "radon"],
            risk="low", timeout_s=120.0,
        ),
        CommandSpec(
            tool_name="radon_mi",
            toolchain_id="radon",
            description="Maintainability index report (radon mi).",
            base_args=["mi", "-s"],
            params=[_PATH_PARAM],
            category="lint", tags=["metrics", "python", "radon"],
            risk="low", timeout_s=120.0,
        ),
    ],
    "maturin": [
        CommandSpec(
            tool_name="maturin_develop",
            writes_files=True,
            toolchain_id="maturin",
            description=(
                "Build the Rust crate and install it into the current "
                "Python environment (maturin develop)."
            ),
            base_args=["develop"],
            params=[ParamSpec(name="release", type="bool", default=False,
                              flag="--release",
                              description="Build with optimizations.")],
            category="build", tags=["build", "rust", "python", "maturin"],
            risk="high", timeout_s=900.0,
        ),
        CommandSpec(
            tool_name="maturin_build",
            writes_files=True,
            toolchain_id="maturin",
            description="Build release wheels (maturin build).",
            base_args=["build"],
            params=[ParamSpec(name="release", type="bool", default=True,
                              flag="--release",
                              description="Build with optimizations.")],
            category="build", tags=["build", "rust", "python", "maturin"],
            risk="high", timeout_s=900.0,
        ),
    ],
    "cargo": [
        CommandSpec(
            tool_name="cargo_check",
            writes_files=True,
            toolchain_id="cargo",
            description="Type-check the Rust crate (cargo check).",
            base_args=["check", "--message-format=short"],
            params=[],
            category="build", tags=["build", "rust", "cargo"],
            risk="high",  # build scripts / proc-macros execute during check
            timeout_s=600.0,
        ),
        CommandSpec(
            tool_name="cargo_build",
            writes_files=True,
            toolchain_id="cargo",
            description="Compile the Rust crate (cargo build).",
            base_args=["build", "--message-format=short"],
            params=[ParamSpec(name="release", type="bool", default=False,
                              flag="--release",
                              description="Build with optimizations.")],
            category="build", tags=["build", "rust", "cargo"],
            risk="high", timeout_s=900.0,
        ),
        CommandSpec(
            tool_name="cargo_test",
            writes_files=True,
            toolchain_id="cargo",
            description="Run the Rust test suite (cargo test).",
            base_args=["test"],
            params=[ParamSpec(name="test_filter", type="str", default="",
                              description="Optional test name filter.")],
            category="build", tags=["test", "rust", "cargo"],
            risk="high", timeout_s=900.0,
        ),
        CommandSpec(
            tool_name="cargo_clippy",
            writes_files=True,
            toolchain_id="cargo",
            description="Lint the Rust crate (cargo clippy).",
            base_args=["clippy", "--message-format=short"],
            params=[],
            category="lint", tags=["lint", "rust", "cargo"],
            risk="high", timeout_s=600.0,
        ),
        CommandSpec(
            tool_name="cargo_fmt",
            writes_files=True,
            toolchain_id="cargo",
            description=(
                "Format the Rust crate (cargo fmt). With check=true only "
                "reports what would change."
            ),
            base_args=["fmt"],
            params=[ParamSpec(name="check", type="bool", default=False,
                              flag="--check",
                              description="Only check, do not rewrite files.")],
            category="lint", tags=["format", "rust", "cargo"],
            risk="medium", timeout_s=120.0,
        ),
    ],
}


def _args_model(spec: CommandSpec) -> Optional[type[BaseModel]]:
    """Build the pydantic argument model driving schema + validation."""
    if not spec.params:
        return None
    fields: dict[str, Any] = {}
    for param in spec.params:
        default = ... if param.required else param.default
        fields[param.name] = (
            param.python_type(),
            Field(default, description=param.description),
        )
    return create_model(f"{spec.tool_name}_args", **fields)


def attach_toolchain_tools(
    toolbox: "ToolBox",
    sandbox: "FileToolSandbox",
    toolchains: Iterable[ToolchainEnv],
) -> None:
    """Register every built-in spec whose toolchain is present.

    Each tool runs its subprocess with cwd pinned to the sandbox root and
    the toolchain's extra env layered over the inherited environment.
    ``sequential=True`` keeps exec/build tools from interleaving with
    parallel file writes.

    Duplicate kinds (e.g. two Python venvs) are legal: the first instance
    keeps the plain tool name, later ones get a numbered suffix
    (``run_python``, ``run_python_2``, …) — each description carries its
    environment label so the model can tell them apart.
    """
    import os

    for env in toolchains:
        for spec in SPEC_PACKS.get(env.id, []):
            tool_name = spec.tool_name
            if tool_name in toolbox.tools:
                index = 2
                while f"{tool_name}_{index}" in toolbox.tools:
                    index += 1
                tool_name = f"{tool_name}_{index}"

            def make_executable(env: ToolchainEnv, spec: CommandSpec):
                def run_tool(db_pool, user_session, **args: Any) -> str:
                    argv = build_argv(spec, env, args, sandbox)
                    run_env = {**os.environ, **dict(env.env)}
                    # A writer holds the whole root for as long as it runs;
                    # a read-only command takes nothing (spec D67 tier 2).
                    # The wait is legible: before_tool_execute has already
                    # fired, so a queued formatter reads as a tool call in
                    # flight rather than as a hang.
                    gate = (
                        write_gate(sandbox.root_dir) if spec.writes_files
                        else nullcontext()
                    )
                    try:
                        with gate:
                            proc = subprocess.run(
                                argv,
                                cwd=str(sandbox.root_dir),
                                env=run_env,
                                capture_output=True,
                                text=True,
                                timeout=spec.timeout_s,
                            )
                    except subprocess.TimeoutExpired:
                        return (
                            f"error: '{spec.tool_name}' timed out after "
                            f"{spec.timeout_s:.0f}s"
                        )
                    return _format_output(proc, spec.output_limit)
                return run_tool

            toolbox.register(
                tool_name,
                f"{spec.description} [{env.describe()}]",
                args_model=_args_model(spec),
                timeout=spec.timeout_s + 10.0,  # outer guard beyond the subprocess timeout
                sequential=True,
                tags=tuple(spec.tags),
                category=spec.category,
                risk=spec.risk,
            )(make_executable(env, spec))
