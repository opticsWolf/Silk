# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The ``node-dev`` toolset: the agent writes a Rust compute node, proves it
against a Python reference, hot-loads and benchmarks it in Weave, and asks
for sign-off (Weave P6). The loop and its rules are in
``functions/node_dev.py``; this module mounts them as tools and installs
the floor that asks a person before code loads into Weave and before a
kind is promoted -- every time, with the code in the request (D77).

The Weave the tools drive is the one :func:`bind_weave` names (anything
with ``call(method, **params)``: ``LocalControl``, ``ControlClient``), or
else the control API behind the run's graph seam (``ControlSeam``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic import BaseModel, Field

from weave.logger import get_logger

from ..approval import _SEAM_ATTR, _refusal
from ..decision_seam import KIND_APPROVAL, DecisionRequest, new_decision_id
from ..hooks import HOOK_WRAP_TOOL_EXECUTE
from ..node_dev import (
    ASKS_EVERY_TIME, OP_BENCH, OP_BUILD, OP_DECLARE, OP_LOAD, OP_PROMOTE, OP_PROVE, OP_REFERENCE,
    OP_SCAFFOLD, OP_STATUS, OP_WRITE_KERNEL, SCRATCH_SET, NodeDevError, Workspace, ready,
)
from .. import node_dev as nd

if TYPE_CHECKING:
    from ..tool_box import ToolBox
    from .file_sandbox import FileToolSandbox

log = get_logger("SilkNodeDevTools")

__all__ = ["attach_node_dev_tools", "attach_node_dev_floor", "bind_weave", "bind_signoff_advisor",
           "node_dev_workspace"]

_WEAVE_ATTR = "_silk_node_dev_weave"
_WORKSPACE_ATTR = "_silk_node_dev_workspace"
_ADVISOR_ATTR = "_silk_node_dev_advisor"
_APPROVER_ATTR = "_silk_node_dev_approver"
_FLOOR_ATTR = "_silk_node_dev_floor"


def bind_weave(toolbox: Any, control: Any) -> None:
    """The Weave node-dev loads into and benchmarks on (``None`` unbinds)."""
    setattr(toolbox, _WEAVE_ATTR, control)


def bind_signoff_advisor(toolbox: Any, advisor: Optional[Callable[[dict], Any]]) -> None:
    """A classifier whose suggestion rides on a promotion request -- shown
    to the person beside the code, never taken as their answer (D-S8)."""
    setattr(toolbox, _ADVISOR_ATTR, advisor)


def node_dev_workspace(toolbox: Any) -> Optional[Workspace]:
    return getattr(toolbox, _WORKSPACE_ATTR, None)


def _weave(toolbox: Any) -> Any:
    control = getattr(toolbox, _WEAVE_ATTR, None)
    if control is not None:
        return control
    from ..graph_author import canvas_binding
    binding = canvas_binding(toolbox)
    seam = None if binding is None else binding.seam
    return getattr(seam, "control", None)


# ── the floor: asked every time ──────────────────────────────────────────

def attach_node_dev_floor(toolbox: Any, *, promote_root: Optional[os.PathLike] = None) -> Any:
    """Ask a person before every :data:`ASKS_EVERY_TIME` call, with the code.

    No grant, preset or Role pre-authorises these, and ``remember`` is not
    honoured: a "don't ask again" for loading a native library would be a
    standing grant of the Weave process's authority. No seam, no load
    (D36). Essential and outermost, like the load floor (D37)."""
    existing = getattr(toolbox, _FLOOR_ATTR, None)
    if existing is not None:
        return existing

    async def floor(handler=None, tool_name: str = "", tool_args: Optional[dict] = None, **_kw: Any) -> Any:
        args = dict(tool_args or {})
        kind = str(args.get("kind") or "")
        ws = node_dev_workspace(toolbox)
        if ws is None:
            return _refusal(target=kind, change_type=tool_name, text="node-dev has no workspace bound.")
        try:
            if tool_name == OP_PROMOTE:
                missing = ready(ws, kind)
                if missing:
                    # Nothing to ask about yet: say what is missing, as the linter
                    # speaks before the human is asked (D78).
                    return _refusal(target=kind, change_type=tool_name,
                                    text="Not ready to promote: " + "; ".join(missing) + ".")
                target = str(args.get("set_id") or "")
                detail = nd.signoff_detail(ws, kind, target, getattr(toolbox, _ADVISOR_ATTR, None))
                prompt = (f"Promote {kind} from the scratch set into '{target}'? It was written by an agent; "
                          f"review the kernel, the proof and the bench.")
            else:
                spec, state = ws.load(kind)
                if not state.get("proven"):
                    return _refusal(target=kind, change_type=tool_name,
                                    text=f"Prove the kernel first ({OP_PROVE}); an unproven build is not offered "
                                         f"to the user for loading.")
                detail = nd.load_detail(ws, kind)
                prompt = (f"Load {kind} (and its Python reference) into Weave's scratch set? Both run with the "
                          f"full authority of the Weave process.")
        except NodeDevError as exc:
            return _refusal(target=kind, change_type=tool_name, text=str(exc))
        asker = getattr(toolbox, _SEAM_ATTR, None)
        if asker is None:
            return _refusal(target=kind, change_type=tool_name, text=(
                "Loading or promoting agent-written code always needs the user's approval, and this run has no "
                "way to ask. No grant, preset or Role can pre-authorise it."))
        decision = asker.await_decision(DecisionRequest(
            decision_id=new_decision_id(), kind=KIND_APPROVAL, prompt=prompt, tool_name=tool_name,
            tool_args=args, detail=detail))
        if not decision.approved:
            return _refusal(target=kind, change_type=tool_name, text=decision.refusal_text())
        setattr(toolbox, _APPROVER_ATTR, decision.actor or "user")
        try:
            return await handler()
        finally:
            setattr(toolbox, _APPROVER_ATTR, None)

    entry = toolbox.hooks.register_middleware(HOOK_WRAP_TOOL_EXECUTE, floor, tools=ASKS_EVERY_TIME,
                                              essential=True)
    toolbox.hooks.make_outermost(HOOK_WRAP_TOOL_EXECUTE, entry)
    setattr(toolbox, _FLOOR_ATTR, entry)
    return entry


# ── schemas ──────────────────────────────────────────────────────────────

class PortArg(BaseModel):
    name: str = Field(..., description="An identifier.")
    datatype: str = Field(..., description="float, int, bool, string or ndarray.")
    description: str = ""


class ParamArg(BaseModel):
    name: str
    datatype: str = Field("float", description="float, int, bool or string.")
    default: Any = Field(..., description="Its default value.")
    description: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None


class DeclareArgs(BaseModel):
    kind: str = Field(..., description="CamelCase kind name, e.g. 'SumSquares'.")
    description: str = Field(..., description="What it computes, one sentence.")
    inputs: list[PortArg] = Field(default_factory=list)
    outputs: list[PortArg] = Field(..., min_length=1)
    params: list[ParamArg] = Field(default_factory=list)
    examples: list[dict] = Field(..., min_length=1, description=(
        "Cases with expected outputs: {'inputs': {...}, 'params': {...}, 'outputs': {...}}."))
    budget_ms: float = Field(0.0, description="Time budget per evaluation on the bench inputs, ms (0: none).")
    speedup: float = Field(1.0, description="How many times faster than the reference the kernel must be.")
    tolerance: float = Field(1e-9, description="Relative (absolute near zero) agreement with the reference.")
    generate: dict = Field(default_factory=dict, description=(
        "How property and bench inputs are made, by name: {'shape': [n], 'low': a, 'high': b} for an "
        "array, {'low': a, 'high': b} for a number, {'values': [...]} to pick from."))
    cases: int = Field(25, description="Generated property cases.")
    category: str = "Generated"


class KindArg(BaseModel):
    kind: str = Field(..., description="The declared kind.")


class ReferenceArgs(KindArg):
    body: str = Field(..., description=(
        "The Python function's body (numpy is imported as np). Its arguments are the inputs and params "
        "by name; return the output, or a dict of outputs. Slow and obvious beats clever: it is the oracle."))


class KernelArgs(KindArg):
    source: str = Field(..., description="The whole src/lib.rs, starting from the scaffold.")


class BuildArgs(KindArg):
    clippy: bool = Field(True, description="Run Clippy in tier 3.")


class ProveArgs(KindArg):
    cases: Optional[int] = Field(None, description="Generated cases (default: the spec's).")
    seed: int = 0


class BenchArgs(KindArg):
    runs: int = Field(20, ge=1, le=1000, description="Computes per node.")


class PromoteArgs(KindArg):
    set_id: str = Field(..., description="The plugin set to promote into (an identifier, not 'scratch').")


class StatusArgs(BaseModel):
    kind: str = Field("", description="One kind, or empty for all.")


class NodeDevResult(BaseModel):
    ok: bool
    op: str = ""
    result: dict = Field(default_factory=dict)
    message: str = ""


# ── the tools ────────────────────────────────────────────────────────────

def attach_node_dev_tools(toolbox: "ToolBox", sandbox: Optional["FileToolSandbox"] = None, *,
                          root: Optional[os.PathLike] = None,
                          promote_root: Optional[os.PathLike] = None) -> Workspace:
    """Mount the ten node-dev tools and the floor. *root* is the workspace
    (default ``<sandbox root>/.weave/node-dev``, else ``~/.weave/node-dev``);
    *promote_root* where promoted sets go (default the user plugin root)."""
    if root is None:
        base = Path(getattr(sandbox, "root_dir", "") or getattr(sandbox, "root", "") or Path.home())
        root = base / ".weave" / "node-dev"
    ws = Workspace(root)
    setattr(toolbox, _WORKSPACE_ATTR, ws)
    attach_node_dev_floor(toolbox)

    def _promote_root() -> Path:
        if promote_root is not None:
            return Path(promote_root)
        from ..self_modify import user_plugin_root
        return user_plugin_root()

    def run(op: str, fn: Callable[[], dict]) -> NodeDevResult:
        try:
            result = fn()
        except NodeDevError as exc:
            return NodeDevResult(ok=False, op=op, message=str(exc))
        except FileNotFoundError as exc:
            return NodeDevResult(ok=False, op=op, message=f"A tool is missing: {exc}")
        return NodeDevResult(ok=bool(result.get("ok", True)), op=op, result=result)

    def needs_weave(op: str, fn: Callable[[Any], dict]) -> NodeDevResult:
        control = _weave(toolbox)
        if control is None:
            return NodeDevResult(ok=False, op=op, message=(
                "No Weave is bound to this run through its control API, so nothing can be loaded or timed."))
        return run(op, lambda: fn(control))

    @toolbox.register(
        name=OP_DECLARE, tags=("node-dev", "write"), category="node-dev", risk="low",
        description="Declare a new compute node: ports, params, examples, budget, speedup, tolerance. "
                    "The declaration must lint clean before any code is written.",
        args_model=DeclareArgs,
        procedure=(
            "Step 1 of node-dev.\n"
            "- Examples need expected outputs: they become the conformance cases (tier 2).\n"
            "- speedup is against your own Python reference; budget_ms is per evaluation on the bench inputs.\n"
            "- Re-declaring a kind starts its later steps over."),
    )
    def _declare(db_pool: Any, user_session: dict, **spec: Any) -> NodeDevResult:
        return run(OP_DECLARE, lambda: nd.declare(ws, spec))

    @toolbox.register(
        name=OP_REFERENCE, tags=("node-dev", "write", "run"), category="node-dev", risk="high",
        description="Write the slow, obvious Python reference; it is checked against the examples in a "
                    "separate interpreter and becomes the oracle.",
        args_model=ReferenceArgs,
        procedure="Step 2. Give the function body only; the signature comes from the declaration.",
    )
    def _reference(db_pool: Any, user_session: dict, kind: str, body: str) -> NodeDevResult:
        return run(OP_REFERENCE, lambda: nd.write_reference(ws, kind, body))

    @toolbox.register(
        name=OP_SCAFFOLD, tags=("node-dev", "write"), category="node-dev", risk="low",
        description="Scaffold the kind's Rust crate from its declaration and return src/lib.rs.",
        args_model=KindArg,
        procedure="Step 3a. Keep the struct, its attributes and the export; fill in compute.",
    )
    def _scaffold(db_pool: Any, user_session: dict, kind: str) -> NodeDevResult:
        return run(OP_SCAFFOLD, lambda: nd.scaffold(ws, kind))

    @toolbox.register(
        name=OP_WRITE_KERNEL, tags=("node-dev", "write"), category="node-dev", risk="medium",
        description="Write the kernel's src/lib.rs (the whole file).",
        args_model=KernelArgs,
        procedure="Step 3b. Then build it.",
    )
    def _write_kernel(db_pool: Any, user_session: dict, kind: str, source: str) -> NodeDevResult:
        return run(OP_WRITE_KERNEL, lambda: nd.write_kernel(ws, kind, source))

    @toolbox.register(
        name=OP_BUILD, tags=("node-dev", "run"), category="node-dev", risk="high",
        description="Build the kernel (cargo, release) into the scratch set and run lint tiers 1 to 3; "
                    "returns the findings, errors first.",
        args_model=BuildArgs,
        procedure=(
            "Step 4.\n- A compile error comes back as a finding at its line.\n"
            "- Tier 2 runs your examples on the build; tier 3 is Weave's Rust rules and Clippy."),
    )
    def _build(db_pool: Any, user_session: dict, kind: str, clippy: bool = True) -> NodeDevResult:
        return run(OP_BUILD, lambda: nd.build(ws, kind, clippy=clippy))

    @toolbox.register(
        name=OP_PROVE, tags=("node-dev", "run"), category="node-dev", risk="high",
        description="Prove the kernel: the examples and generated cases, kernel against reference within "
                    "the tolerance, in a separate interpreter.",
        args_model=ProveArgs,
        procedure="Step 5. A failure shows the case's inputs (arrays by shape) and the error.",
    )
    def _prove(db_pool: Any, user_session: dict, kind: str, cases: Optional[int] = None,
               seed: int = 0) -> NodeDevResult:
        return run(OP_PROVE, lambda: nd.prove(ws, kind, cases, seed))

    @toolbox.register(
        name=OP_LOAD, tags=("node-dev", "plugins"), category="node-dev", risk="high",
        description="Hot-load the scratch set (the kernel and its reference) into Weave; the user is asked "
                    "every time, with the code.",
        args_model=KindArg,
        procedure=(
            "Step 6a.\n- Only a proven build is offered.\n"
            "- Its kinds load as unreviewed, in the 'scratch' set; a second load reloads in place."),
    )
    def _load(db_pool: Any, user_session: dict, kind: str) -> NodeDevResult:
        if getattr(toolbox, _APPROVER_ATTR, None) is None:
            return NodeDevResult(ok=False, op=OP_LOAD, message="Loading code into Weave needs the user's approval.")
        return needs_weave(OP_LOAD, lambda control: nd.load(control, ws, kind))

    @toolbox.register(
        name=OP_BENCH, tags=("node-dev", "run", "graph"), category="node-dev", risk="medium",
        description="Benchmark the loaded kernel against its reference in a test graph, with the engine's "
                    "per-node timing; says whether the speedup and the budget are met.",
        args_model=BenchArgs,
        procedure="Step 6b. The test graph is taken down afterwards. Not met: improve, build, prove, load, bench.",
    )
    def _bench(db_pool: Any, user_session: dict, kind: str, runs: int = 20) -> NodeDevResult:
        return needs_weave(OP_BENCH, lambda control: nd.bench(control, ws, kind, runs))

    @toolbox.register(
        name=OP_PROMOTE, tags=("node-dev", "plugins"), category="node-dev", risk="high",
        description="Ask the user to sign off and promote the kind from the scratch set into a real plugin set.",
        args_model=PromoteArgs,
        procedure=(
            "Step 7.\n- Only after build, proof and bench have passed on the same build.\n"
            "- The user decides; a classifier's view may be shown to them, never instead of them.\n"
            "- In a Weave that serves plugins.unload, the kind moves to the promoted set now; otherwise with "
            "Weave's next session."),
    )
    def _promote(db_pool: Any, user_session: dict, kind: str, set_id: str) -> NodeDevResult:
        approver = getattr(toolbox, _APPROVER_ATTR, None)
        if approver is None:
            # Reached without the floor: refuse rather than promote unasked.
            return NodeDevResult(ok=False, op=OP_PROMOTE, message="Promotion needs the user's sign-off.")
        return run(OP_PROMOTE, lambda: nd.promote(ws, kind, _promote_root(), set_id, approved_by=approver,
                                                  control=_weave(toolbox)))

    @toolbox.register(
        name=OP_STATUS, tags=("node-dev", "read"), category="node-dev", risk="low",
        description="Where each node-dev kind stands: its steps, its bench, and what is still missing.",
        args_model=StatusArgs,
    )
    def _status(db_pool: Any, user_session: dict, kind: str = "") -> NodeDevResult:
        return run(OP_STATUS, lambda: nd.status(ws, kind or None))

    return ws
