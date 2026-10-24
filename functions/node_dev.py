# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The ``node-dev`` loop: an agent writes a compute node in Rust, proves it
against a slow Python reference, hot-loads it into the project's scratch
plugin set, benchmarks it, and hands it to a person for sign-off (Weave
manifest, "LLM-built compute nodes, with Silk as the harness"; Weave P6).

The steps, each a tool (``functions/tools/node_dev_tools.py``):

1. **Declare** (:func:`declare`): a :class:`Spec` -- ports, types,
   parameters, examples with expected outputs, the time budget per
   evaluation, the factor the kernel must beat its reference by, the
   tolerance -- becomes the kind's declaration, which must lint clean
   (tier 1) before any code is written.
2. **Reference first** (:func:`write_reference`): a slow, obvious Python
   version (a ``weave.sdk`` ``@node``), checked against the examples. It
   is the oracle.
3. **Implement** (:func:`scaffold`, :func:`write_kernel`): the declaration
   scaffolds a Rust guest crate (``weave.sdk.rust``); the agent writes its
   ``src/lib.rs``.
4. **Build and lint** (:func:`build`): cargo, then lint tiers 1 to 3 on the
   build in the scratch set; the findings are the tool's result.
5. **Prove** (:func:`prove`): the examples, plus property cases -- inputs
   generated from the spec, the kernel's outputs against the reference's
   within the tolerance.
6. **Hot-load and bench** (:func:`load`, :func:`bench`): the scratch set
   loads into the Weave the agent drives (``plugins.load`` /
   ``plugins.reload``; its kinds are *unreviewed*), and a test graph built
   through the control API times the kernel and the reference with the
   engine's own per-node timing (``node.timing``). 3 to 6 repeat until
   the budget is met.
7. **Sign off** (:func:`promote`): a person approves moving the kind from
   ``scratch`` into a real plugin set, on the approval seam.

What runs where, and who is asked (D77, the toolchain rule):

* Building and proving run processes -- cargo, and a fresh interpreter for
  the proof that loads both the kernel and the reference -- so they are
  ``risk="high"`` tools, gated by the Role and risk axis as ``run_python``
  and ``cargo`` are. Nothing they run enters this process or Weave's.
* **Loading into Weave asks every time** (:data:`ASKS_EVERY_TIME`): a native
  library or a Python reference runs with the full authority of the Weave
  process, so this is D77's load floor, with the code in the request and
  no "don't ask again". A set whose components are all WASM is sandboxed
  by its grants and would not need it; this build environment has no
  ``wasm32-wasip2`` target, so kernels build native.
* **Promotion asks every time** and is a human decision: a classifier may
  attach a suggestion to the request (:func:`signoff_detail`), never an
  answer.

Qt-free: everything here is files, subprocesses and control-API calls.
"""
from __future__ import annotations

import difflib
import json
import keyword
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from weave.logger import get_logger

log = get_logger("SilkNodeDev")

# ── the verbs ────────────────────────────────────────────────────────────

OP_DECLARE = "node_dev_declare"
OP_REFERENCE = "node_dev_reference"
OP_SCAFFOLD = "node_dev_scaffold"
OP_WRITE_KERNEL = "node_dev_write_kernel"
OP_BUILD = "node_dev_build"
OP_PROVE = "node_dev_prove"
OP_LOAD = "node_dev_load"
OP_BENCH = "node_dev_bench"
OP_PROMOTE = "node_dev_promote"
OP_STATUS = "node_dev_status"

OPS = (OP_DECLARE, OP_REFERENCE, OP_SCAFFOLD, OP_WRITE_KERNEL, OP_BUILD, OP_PROVE, OP_LOAD, OP_BENCH,
       OP_PROMOTE, OP_STATUS)

#: Asked on the seam every time, whatever the policy or the grants (D77):
#: loading code into Weave, and promoting it out of scratch.
ASKS_EVERY_TIME = (OP_LOAD, OP_PROMOTE)

#: The project's scratch plugin set: where unreviewed kinds load.
SCRATCH_SET = "scratch"

#: Datatypes a spec may name, and what they are in Python.
DATATYPES = ("float", "int", "bool", "string", "ndarray")
_PY_TYPES = {"float": "float", "int": "int", "bool": "bool", "string": "str", "ndarray": "np.ndarray"}

#: Defaults for what a spec leaves out.
DEFAULT_CASES = 25
DEFAULT_TOLERANCE = 1e-9
DEFAULT_RUNS = 20
PROVE_TIMEOUT_S = 600.0

#: The cap on a source shown in an approval request (as self_modify's).
MAX_SOURCE_LINES = 400


class NodeDevError(Exception):
    """A step refused, with a sentence for the model."""


# ── the spec ─────────────────────────────────────────────────────────────

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KIND = re.compile(r"^[A-Z][A-Za-z0-9]*$")


@dataclass
class Port:
    name: str
    datatype: str
    description: str = ""


@dataclass
class Param:
    name: str
    datatype: str
    default: Any
    description: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class Spec:
    """What the agent declares (step 1)."""
    kind: str
    description: str
    inputs: List[Port]
    outputs: List[Port]
    params: List[Param] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    #: The time budget per evaluation, on the benchmark's inputs (ms).
    budget_ms: float = 0.0
    #: The kernel must be this many times faster than the reference.
    speedup: float = 1.0
    #: Outputs agree when within this (relative, and absolute near zero).
    tolerance: float = DEFAULT_TOLERANCE
    #: How property and benchmark inputs are made, by port or param name:
    #: ``{"shape": [n], "low": a, "high": b}`` for an array, ``{"low": a,
    #: "high": b}`` for a number, ``{"values": [...]}`` to pick from.
    generate: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cases: int = DEFAULT_CASES
    category: str = "Generated"

    @property
    def reference_kind(self) -> str:
        return f"{self.kind}Reference"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Spec":
        """A spec, or :class:`NodeDevError` naming every problem at once."""
        problems: List[str] = []
        kind = str(d.get("kind") or "")
        if not _KIND.match(kind) or keyword.iskeyword(kind):
            problems.append(f"kind {kind!r}: a CamelCase name, letters and digits (e.g. 'SumSquares')")

        def ports(key: str) -> List[Port]:
            out = []
            for i, p in enumerate(d.get(key) or []):
                name, dt = str(p.get("name") or ""), str(p.get("datatype") or "")
                if not _NAME.match(name) or keyword.iskeyword(name):
                    problems.append(f"{key}[{i}] name {name!r}: an identifier")
                if dt not in DATATYPES:
                    problems.append(f"{key}[{i}] ({name}) datatype {dt!r}: one of {', '.join(DATATYPES)}")
                out.append(Port(name, dt, str(p.get("description") or "")))
            return out

        inputs, outputs = ports("inputs"), ports("outputs")
        if not outputs:
            problems.append("outputs: at least one")
        params = []
        for i, p in enumerate(d.get("params") or []):
            name, dt = str(p.get("name") or ""), str(p.get("datatype") or "float")
            if not _NAME.match(name):
                problems.append(f"params[{i}] name {name!r}: an identifier")
            if dt not in ("float", "int", "bool", "string"):
                problems.append(f"params[{i}] ({name}) datatype {dt!r}: float, int, bool or string")
            if "default" not in p:
                problems.append(f"params[{i}] ({name}): a default")
            params.append(Param(name, dt, p.get("default"), str(p.get("description") or ""),
                                p.get("minimum"), p.get("maximum")))
        names = [p.name for p in inputs] + [p.name for p in params]
        if len(set(names)) != len(names) or set(names) & {p.name for p in outputs}:
            problems.append("port and param names: each once")
        examples = list(d.get("examples") or [])
        if not examples:
            problems.append("examples: at least one, with expected outputs (they are tier 2's cases)")
        for i, ex in enumerate(examples):
            if not isinstance(ex, dict) or not isinstance(ex.get("outputs"), dict):
                problems.append(f"examples[{i}]: {{'inputs': {{...}}, 'params': {{...}}, 'outputs': {{...}}}}")
        speedup = float(d.get("speedup") or 1.0)
        budget = float(d.get("budget_ms") or 0.0)
        tolerance = float(d.get("tolerance") or DEFAULT_TOLERANCE)
        if speedup < 1.0 or budget < 0 or tolerance <= 0:
            problems.append("speedup >= 1, budget_ms >= 0 (0: no budget), tolerance > 0")
        if problems:
            raise NodeDevError("The spec is not usable: " + "; ".join(problems) + ".")
        return cls(kind, str(d.get("description") or kind), inputs, outputs, params, examples, budget, speedup,
                   tolerance, dict(d.get("generate") or {}), int(d.get("cases") or DEFAULT_CASES),
                   str(d.get("category") or "Generated"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── the workspace ────────────────────────────────────────────────────────

class Workspace:
    """Where node-dev keeps its files, one folder per project::

        ROOT/specs/KIND.json        the spec and the step's state
        ROOT/reference/KIND.py      the reference (a weave.sdk @node file)
        ROOT/crates/KIND/           the kernel's crate
        ROOT/target/                cargo's (or a shared one)
        ROOT/sets/scratch/          the scratch plugin set
    """

    def __init__(self, root: os.PathLike, *, target_dir: Optional[os.PathLike] = None) -> None:
        self.root = Path(root)
        #: cargo's target folder: the workspace's own, or one shared with
        #: other workspaces (``$WEAVE_NODE_DEV_TARGET``), which builds the
        #: SDK and its dependencies once.
        shared = target_dir or os.environ.get("WEAVE_NODE_DEV_TARGET")
        self._target = Path(shared) if shared else None

    def spec_path(self, kind: str) -> Path:
        return self.root / "specs" / f"{kind}.json"

    def stub_path(self, kind: str) -> Path:
        return self.root / "specs" / f"{kind}_declared.py"

    def reference_path(self, kind: str) -> Path:
        return self.root / "reference" / f"{kind}.py"

    def crate_dir(self, kind: str) -> Path:
        return self.root / "crates" / kind

    @property
    def target_dir(self) -> Path:
        return self._target or self.root / "target"

    @property
    def sets_root(self) -> Path:
        return self.root / "sets"

    @property
    def scratch(self) -> Path:
        return self.sets_root / SCRATCH_SET

    def kinds(self) -> List[str]:
        d = self.root / "specs"
        return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []

    # -- the spec and its state --------------------------------------------

    def load(self, kind: str) -> Tuple[Spec, Dict[str, Any]]:
        p = self.spec_path(kind)
        if not p.is_file():
            known = ", ".join(self.kinds()) or "none yet"
            raise NodeDevError(f"No kind {kind!r} is declared (declared: {known}). Declare it first.")
        doc = json.loads(p.read_text())
        return Spec.from_dict(doc["spec"]), doc.get("state") or {}

    def save(self, spec: Spec, state: Dict[str, Any]) -> None:
        p = self.spec_path(spec.kind)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.new")
        tmp.write_text(json.dumps({"spec": spec.to_dict(), "state": state}, indent=2, default=str))
        os.replace(tmp, p)

    def update(self, kind: str, **changes: Any) -> Dict[str, Any]:
        spec, state = self.load(kind)
        state.update(changes)
        self.save(spec, state)
        return state


def _require(state: Mapping[str, Any], step: str, why: str) -> None:
    if not state.get(step):
        raise NodeDevError(why)


# ── step 1: declare ──────────────────────────────────────────────────────

def _signature(spec: Spec) -> str:
    args = [f"{p.name}: {_PY_TYPES[p.datatype]}" for p in spec.inputs]
    args += [f"{p.name}: {_PY_TYPES[p.datatype]} = {p.default!r}" for p in spec.params]
    return ", ".join(args)


def node_source(spec: Spec, *, name: str, body: str) -> str:
    """A ``weave.sdk`` node file for *spec*: the function *name* with *body*.

    Everything the agent wrote except *body* reaches the file as a Python
    literal (``repr``) or a checked identifier, so the declaration stub
    (*body* ``raise NotImplementedError``) is safe to import here."""
    outputs = {p.name: p.datatype for p in spec.outputs}
    examples = [{k: v for k, v in ex.items() if k in ("inputs", "params", "outputs")} for ex in spec.examples]
    return (
        "# Written by Silk's node-dev from the kind's spec.\n"
        "import numpy as np\n"
        "from weave.sdk import node\n\n\n"
        f"@node(outputs={outputs!r}, category={spec.category!r},\n"
        f"      examples={examples!r})\n"
        f"def {name}({_signature(spec)}):\n"
        f"    {spec.description!r}\n"
        + textwrap.indent(textwrap.dedent(body).strip("\n") + "\n", "    ")
    )


def _declarations_of(path: Path) -> List[Dict[str, Any]]:
    os.environ.setdefault("WEAVE_HEADLESS", "1")
    from weave.sdk import declarations, load
    return declarations(load(path), provider="rust")


def declare(ws: Workspace, spec_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Step 1: the spec, its declaration, and tier 1's findings on it. A
    re-declared kind starts its later steps over."""
    spec = Spec.from_dict(spec_dict)
    stub = ws.stub_path(spec.kind)
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(node_source(spec, name=spec.kind, body="raise NotImplementedError"))
    import weave_rs
    (decl,) = _declarations_of(stub)
    findings = [dict(f, tier=1) for f in weave_rs.lint(json.dumps({"declarations": [decl]}), SCRATCH_SET)]
    ok = not any(f.get("severity") == "error" for f in findings)
    ws.save(spec, {"declared": ok, "declaration": decl})
    return {"kind": spec.kind, "ok": ok, "declaration": decl, "findings": findings,
            "next": "write the reference" if ok else "fix the spec: the declaration must lint clean"}


# ── step 2: the reference ────────────────────────────────────────────────

#: The proof harness: run in a fresh interpreter (``python -I``), so the
#: agent's reference and kernel never enter this process.
_HARNESS = r'''
import json, math, os, sys
args = json.loads(sys.argv[1])
sys.path[:0] = args["path"]
os.environ["WEAVE_HEADLESS"] = "1"
import numpy as np

def plain(v):
    if isinstance(v, np.ndarray):
        return {"ndarray": v.tolist(), "shape": list(v.shape)}
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if hasattr(v, "to_numpy"):
        a = np.asarray(v.to_numpy())
        return {"ndarray": a.tolist(), "shape": list(a.shape)}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return v

def make(spec, gen, rng):
    out = {}
    for p in spec["inputs"] + [dict(q, param=True) for q in spec["params"]]:
        g = gen.get(p["name"])
        if g is None and p.get("param"):
            out[p["name"]] = p["default"]
            continue
        g = g or {}
        if "values" in g:
            out[p["name"]] = g["values"][int(rng.integers(len(g["values"])))]
        elif p["datatype"] == "ndarray":
            shape = g.get("shape") or [1000]
            out[p["name"]] = rng.uniform(g.get("low", -1.0), g.get("high", 1.0), size=shape)
        elif p["datatype"] == "int":
            out[p["name"]] = int(rng.integers(int(g.get("low", -100)), int(g.get("high", 100)) + 1))
        elif p["datatype"] == "bool":
            out[p["name"]] = bool(rng.integers(2))
        elif p["datatype"] == "float":
            out[p["name"]] = float(rng.uniform(g.get("low", -10.0), g.get("high", 10.0)))
        else:
            out[p["name"]] = p.get("default", "")
    return out

def close(a, b, tol):
    a, b = plain(a), plain(b)
    if isinstance(a, dict) and "ndarray" in a:
        a = np.asarray(a["ndarray"], dtype=float)
    if isinstance(b, dict) and "ndarray" in b:
        b = np.asarray(b["ndarray"], dtype=float)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) or isinstance(a, list) or isinstance(b, list):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape:
            return False, f"shape {list(a.shape)} against {list(b.shape)}"
        err = float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))) if a.size else 0.0
        return err <= tol, err
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return a == b, 0.0 if a == b else f"{a!r} against {b!r}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True, 0.0
        err = abs(a - b) / max(1.0, abs(b))
        return err <= tol, err
    return a == b, 0.0 if a == b else f"{a!r} against {b!r}"

def run(fn, values):
    try:
        return fn(values), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

spec = args["spec"]
from weave.sdk import definitions, load
ref_cls = definitions(load(args["reference"]))[0]
def reference(values):
    return dict(ref_cls().compute(dict(values), None))
kernel = None
if args.get("sets"):
    from weave.plugin import PluginHost, grant_all
    host = PluginHost(ask=grant_all, project=args["project"])
    report = host.load(args["sets"])
    if spec["kind"] not in [k.split(":")[-1] for k, _ in __import__("weave_rs").plugin_kinds()]:
        print(json.dumps({"error": "the kernel did not load: " + "; ".join(report.summary_lines())}))
        sys.exit(0)
    import weave_rs
    pk = weave_rs.PluginKernel(spec["kind"])
    def kernel(values):
        return dict(pk.compute(dict(values)))

def outputs_of(out):
    return {p["name"]: out.get(p["name"]) for p in spec["outputs"]}

cases, failures, worst = [], [], 0.0
for i, ex in enumerate(spec["examples"]):
    values = {**{k: (np.asarray(v, dtype=float) if isinstance(v, list) else v) for k, v in (ex.get("inputs") or {}).items()},
              **(ex.get("params") or {})}
    for p in spec["params"]:
        values.setdefault(p["name"], p["default"])
    cases.append(("example %d" % i, values, ex["outputs"]))
rng = np.random.default_rng(args.get("seed", 0))
for i in range(args.get("cases", 0)):
    cases.append(("case %d" % i, make(spec, spec.get("generate") or {}, rng), None))

for label, values, expected in cases:
    ref, ref_err = run(reference, values)
    if expected is not None:
        if ref_err:
            failures.append({"case": label, "side": "reference", "error": ref_err})
        else:
            for name, want in expected.items():
                ok, err = close(ref.get(name), want, spec["tolerance"])
                if not ok:
                    failures.append({"case": label, "side": "reference", "output": name,
                                     "got": plain(ref.get(name)), "expected": want, "error": err})
    if kernel is None:
        continue
    got, k_err = run(kernel, values)
    if k_err:
        failures.append({"case": label, "side": "kernel", "error": k_err})
        continue
    against = expected if expected is not None else (outputs_of(ref) if not ref_err else None)
    if against is None:
        continue
    for name, want in against.items():
        ok, err = close(got.get(name), want, spec["tolerance"])
        if isinstance(err, float):
            worst = max(worst, err)
        if not ok:
            failures.append({"case": label, "side": "kernel", "output": name, "error": err,
                             "inputs": {k: plain(v) if not isinstance(v, np.ndarray) or v.size <= 8
                                        else {"shape": list(v.shape)} for k, v in values.items()}})
print(json.dumps({"cases": len(cases), "failures": failures[:20], "failed": len(failures), "max_error": worst}))
'''


def _weave_path() -> List[str]:
    import weave
    return [str(Path(weave.__file__).resolve().parent.parent)]


def _harness(ws: Workspace, spec: Spec, *, sets: Optional[List[str]], cases: int, seed: int = 0) -> Dict[str, Any]:
    args = {"path": _weave_path(), "spec": spec.to_dict(), "reference": str(ws.reference_path(spec.kind)),
            "sets": sets, "project": str(ws.root / "project"), "cases": cases, "seed": seed}
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        out = subprocess.run([sys.executable, "-I", "-c", _HARNESS, json.dumps(args)], capture_output=True,
                             text=True, timeout=PROVE_TIMEOUT_S, env=env, cwd=str(ws.root))
    except subprocess.TimeoutExpired:
        raise NodeDevError(f"The proof took longer than {PROVE_TIMEOUT_S:.0f} s and was stopped.")
    line = next((l for l in reversed(out.stdout.splitlines()) if l.startswith("{")), None)
    if out.returncode != 0 or line is None:
        raise NodeDevError("The proof did not run: " + (out.stderr.strip()[-2000:] or "no output"))
    result = json.loads(line)
    if "error" in result:
        raise NodeDevError(result["error"])
    return result


def write_reference(ws: Workspace, kind: str, body: str) -> Dict[str, Any]:
    """Step 2: the reference -- *body* is the function's body; the inputs
    and params are its arguments, and it returns the output (or a dict of
    outputs). Checked against the spec's examples, in a fresh interpreter."""
    spec, state = ws.load(kind)
    _require(state, "declared", "The declaration does not lint clean yet: declare the kind first.")
    path = ws.reference_path(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    before = path.read_text() if path.is_file() else ""
    path.write_text(node_source(spec, name=spec.reference_kind, body=body))
    result = _harness(ws, spec, sets=None, cases=0)
    ok = result["failed"] == 0
    ws.update(kind, referenced=ok, reference_changed=before != path.read_text(), proven=False)
    return {"kind": kind, "ok": ok, "examples": result["cases"], "failures": result["failures"],
            "next": "scaffold the kernel" if ok else "fix the reference: it must reproduce every example"}


# ── step 3: the kernel ───────────────────────────────────────────────────

def scaffold(ws: Workspace, kind: str) -> Dict[str, Any]:
    """Step 3a: the crate, from the declaration. An existing ``lib.rs`` stays."""
    from weave.sdk import rust
    spec, state = ws.load(kind)
    _require(state, "referenced", "Write the reference first: it is the oracle the kernel is proven against.")
    crate = rust.scaffold(state["declaration"], ws.crate_dir(kind))
    return {"kind": kind, "crate": str(crate), "lib_rs": (crate / "src" / "lib.rs").read_text(),
            "next": f"write src/lib.rs with {OP_WRITE_KERNEL} (keep the struct and the export; fill compute)"}


def write_kernel(ws: Workspace, kind: str, source: str) -> Dict[str, Any]:
    """Step 3b: the agent's ``src/lib.rs``."""
    spec, state = ws.load(kind)
    lib = ws.crate_dir(kind) / "src" / "lib.rs"
    if not lib.parent.is_dir():
        raise NodeDevError(f"Scaffold the crate first ({OP_SCAFFOLD}).")
    if f'kind = "{kind}"' not in source:
        raise NodeDevError(f'The source must keep #[node(kind = "{kind}", ...)]: the kind is the declared one.')
    lib.write_text(source)
    ws.update(kind, built=False, linted=False, proven=False)
    return {"kind": kind, "bytes": len(source.encode()), "next": f"build it ({OP_BUILD})"}


# ── step 4: build and lint ───────────────────────────────────────────────

def component_id(kind: str) -> str:
    return f"{SCRATCH_SET}.{re.sub(r'(?<!^)(?=[A-Z])', '_', kind).lower()}"


def _reference_entry(spec: Spec) -> str:
    return (f'[[component]]\nid = "{component_id(spec.kind)}_reference"\nprovider = "process"\n'
            f'path = "reference/{spec.kind}.py"\ncontract = "weave-node 1.1"\nnodes = ["{spec.reference_kind}"]\n')


def _write_reference_component(ws: Workspace, spec: Spec) -> None:
    from weave.sdk import rust
    dest = ws.scratch / "reference" / f"{spec.kind}.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".py.new")
    shutil.copy2(ws.reference_path(spec.kind), tmp)
    os.replace(tmp, dest)
    manifest = ws.scratch / "weave-plugin.toml"
    text = manifest.read_text() if manifest.is_file() else rust.manifest_head(
        SCRATCH_SET, description="Unreviewed kinds built by Silk's node-dev; promoted kinds leave it.")
    manifest.write_text(rust.put_component(text, component_id(spec.kind) + "_reference", _reference_entry(spec)))


def _summary(findings: Sequence[Mapping[str, Any]], limit: int = 30) -> List[Dict[str, Any]]:
    keep = ("tier", "rule", "severity", "location", "message", "hint")
    order = {"error": 0, "warning": 1, "info": 2}
    rows = sorted(findings, key=lambda f: order.get(f.get("severity"), 3))
    return [{k: f.get(k) for k in keep if f.get(k)} for f in rows[:limit]]


def build(ws: Workspace, kind: str, *, clippy: bool = True) -> Dict[str, Any]:
    """Step 4: cargo, then lint tiers 1 to 3 on the build in the scratch set."""
    from weave.sdk import rust
    spec, state = ws.load(kind)
    crate = ws.crate_dir(kind)
    if not (crate / "Cargo.toml").is_file():
        raise NodeDevError(f"Scaffold the crate first ({OP_SCAFFOLD}).")
    started = time.monotonic()
    built = rust.build(crate, "native", release=True, target_dir=ws.target_dir)
    if not built.ok:
        ws.update(kind, built=False, linted=False)
        return {"kind": kind, "ok": False, "stage": "build", "findings": _summary(built.findings),
                "next": "fix the compile errors and build again"}
    declared = rust.declarations_of(built.artefact)
    mismatch = _declaration_mismatch(state["declaration"], declared, kind)
    rust.write_set(ws.scratch, SCRATCH_SET, built.artefact, component=component_id(kind),
                   description="Unreviewed kinds built by Silk's node-dev; promoted kinds leave it.")
    _write_reference_component(ws, spec)
    checked = rust.check(crate, built, ws.scratch, plugin=SCRATCH_SET, component=component_id(kind),
                         tolerance=spec.tolerance, clippy=clippy, target_dir=ws.target_dir)
    every = [f for fs in checked["tiers"].values() for f in fs] + [f for f in built.findings
                                                                   if f["severity"] == "error"] + mismatch
    ok = checked["ok"] and not mismatch
    from weave.sdk.rust import _package  # the crate's name, for the record
    ws.update(kind, built=True, linted=ok, proven=False, artefact=str(built.artefact),
              digest=_digest(built.artefact), crate=_package(crate))
    return {"kind": kind, "ok": ok, "stage": "lint", "seconds": round(time.monotonic() - started, 1),
            "errors": checked["errors"] + len(mismatch), "warnings": checked["warnings"],
            "findings": _summary(every), "artefact": str(built.artefact),
            "next": f"prove it ({OP_PROVE})" if ok else "fix the findings (errors first) and build again"}


def _declaration_mismatch(want: Mapping[str, Any], got: Sequence[Mapping[str, Any]], kind: str) -> List[Dict]:
    """Findings where the build's declaration is not the declared one."""
    mine = [d for d in got if d.get("kind") == kind]
    if not mine:
        return [{"tier": 1, "rule": "node-dev", "severity": "error", "location": "src/lib.rs",
                 "message": f"the build does not declare {kind} (it declares {[d.get('kind') for d in got]})"}]
    d = mine[0]
    out = []
    controls = {c["name"] for c in want.get("controls") or []}
    pairs = (("inputs", [p["name"] for p in want.get("inputs", []) if p["name"] not in controls],
              [p["name"] for p in d.get("inputs", [])]),
             ("outputs", [p["name"] for p in want.get("outputs", [])], [p["name"] for p in d.get("outputs", [])]),
             ("params", sorted(controls), sorted(c["name"] for c in d.get("controls") or [])))
    for what, a, b in pairs:
        if a != b:
            out.append({"tier": 1, "rule": "node-dev", "severity": "error", "location": "src/lib.rs",
                        "message": f"{what} are {b}, the declaration's are {a}"})
    return out


def _digest(path: os.PathLike) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── step 5: prove ────────────────────────────────────────────────────────

def prove(ws: Workspace, kind: str, cases: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    """Step 5: the examples and *cases* generated inputs, kernel against
    reference, in a fresh interpreter that loads the scratch set."""
    spec, state = ws.load(kind)
    _require(state, "linted", f"The build does not pass tiers 1 to 3 yet ({OP_BUILD}).")
    n = spec.cases if cases is None else int(cases)
    result = _harness(ws, spec, sets=[str(ws.sets_root)], cases=n, seed=seed)
    ok = result["failed"] == 0
    ws.update(kind, proven=ok, proof={"cases": result["cases"], "max_error": result["max_error"],
                                      "digest": state.get("digest")})
    return {"kind": kind, "ok": ok, "cases": result["cases"], "failed": result["failed"],
            "max_error": result["max_error"], "tolerance": spec.tolerance, "failures": result["failures"],
            "next": f"hot-load it ({OP_LOAD})" if ok else "fix the kernel (or the reference) and build again"}


# ── step 6: hot-load and bench, through the control API ─────────────────

def _numbered(text: str) -> str:
    lines = text.splitlines()
    shown = lines[:MAX_SOURCE_LINES]
    more = f"\n... {len(lines) - len(shown)} more lines" if len(lines) > len(shown) else ""
    return "\n".join(shown) + more


def load_detail(ws: Workspace, kind: str) -> Dict[str, Any]:
    """What a person is shown before the scratch set loads: the code."""
    spec, state = ws.load(kind)
    lib = ws.crate_dir(kind) / "src" / "lib.rs"
    ref = ws.reference_path(kind)
    last = state.get("loaded_sources") or {}
    detail: Dict[str, Any] = {"kind": kind, "set": SCRATCH_SET, "provider": "native",
                              "sandboxed": False, "proof": state.get("proof"),
                              "why": "A native library and a Python reference run with the full authority of the "
                                     "Weave process; nothing sandboxes them."}
    for label, path in (("kernel", lib), ("reference", ref)):
        text = path.read_text() if path.is_file() else ""
        before = last.get(label)
        detail[label] = {"path": str(path), "lines": len(text.splitlines()),
                         "diff" if before is not None else "source":
                             "".join(difflib.unified_diff(before.splitlines(True), text.splitlines(True),
                                                          f"{label} (loaded)", f"{label} (now)"))[:40000]
                             if before is not None else _numbered(text)}
    return detail


def load(control: Any, ws: Workspace, kind: str) -> Dict[str, Any]:
    """Step 6a: the scratch set into the Weave *control* serves (the gate
    has already asked). Loaded once, then reloaded in place."""
    spec, state = ws.load(kind)
    _require(state, "proven", f"Prove the kernel first ({OP_PROVE}).")
    listed = control.call("plugins.list")
    if any(s.get("id") == SCRATCH_SET for s in listed.get("sets", [])):
        reports = control.call("plugins.reload")["reports"]
        bad = [r for r in reports if r.get("outcome") != "loaded"]
        if bad:
            raise NodeDevError("The reload was refused: " + json.dumps(bad)[:2000])
        outcome = {"reloaded": [r.get("component") for r in reports]}
    else:
        report = control.call("plugins.load", roots=[str(ws.sets_root)])
        if SCRATCH_SET not in report.get("loaded", []):
            line = next((s for s in report.get("sets", []) if s.get("id") == SCRATCH_SET), {})
            raise NodeDevError("The scratch set did not load: " + json.dumps(line.get("findings") or report)[:2000])
        outcome = {"loaded": SCRATCH_SET}
    kinds = control.call("graph.kinds")
    names = {k.get("kind") or k.get("name") for k in (kinds.get("kinds") if isinstance(kinds, dict) else kinds)}
    names |= {n.split(":")[-1] for n in names if n}
    missing = [k for k in (kind, spec.reference_kind) if k not in names]
    if missing:
        raise NodeDevError(f"Loaded, but Weave does not serve {missing} (a kind new to a loaded scratch set "
                           f"loads with the next session).")
    sources = {"kernel": (ws.crate_dir(kind) / "src" / "lib.rs").read_text(),
               "reference": ws.reference_path(kind).read_text()}
    ws.update(kind, loaded=True, loaded_digest=state.get("digest"), loaded_sources=sources)
    return {"kind": kind, "ok": True, **outcome, "unreviewed": True,
            "next": f"benchmark it ({OP_BENCH})"}


def _bench_inputs(spec: Spec, seed: int = 1) -> Dict[str, Any]:
    import numpy as np
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {}
    for p in spec.inputs:
        g = spec.generate.get(p.name) or {}
        if "values" in g:
            out[p.name] = g["values"][0]
        elif p.datatype == "ndarray":
            out[p.name] = rng.uniform(g.get("low", -1.0), g.get("high", 1.0), size=g.get("shape") or [1000]).tolist()
        elif p.datatype == "int":
            out[p.name] = int(g.get("high", 100))
        elif p.datatype == "float":
            out[p.name] = float(rng.uniform(g.get("low", -10.0), g.get("high", 10.0)))
        elif p.datatype == "bool":
            out[p.name] = True
        else:
            out[p.name] = ""
    for p in spec.params:
        g = spec.generate.get(p.name) or {}
        out[p.name] = g["values"][0] if "values" in g else p.default
    return out


def bench(control: Any, ws: Workspace, kind: str, runs: int = DEFAULT_RUNS) -> Dict[str, Any]:
    """Step 6b: a test graph -- one constant per input feeding the kernel
    and the reference -- built through the control API; each node is run
    *runs* times (``node.trigger``) and timed by the engine (``node.timing``,
    the compute alone). The graph is taken down afterwards."""
    spec, state = ws.load(kind)
    _require(state, "loaded", f"Hot-load it first ({OP_LOAD}).")
    if state.get("loaded_digest") != state.get("digest"):
        raise NodeDevError(f"The loaded build is not the latest one: hot-load it again ({OP_LOAD}).")
    values = _bench_inputs(spec)
    placed: List[str] = []
    try:
        params = {p.name: values[p.name] for p in spec.params}
        k = control.call("node.add", kind=kind, pos=[600, 0], title=f"{kind} (bench)", params=params)["uid"]
        r = control.call("node.add", kind=spec.reference_kind, pos=[600, 200], title=f"{kind} reference (bench)",
                         params=params)["uid"]
        placed += [k, r]
        for i, p in enumerate(spec.inputs):
            c = control.call("node.add", kind="weave.core.constant", pos=[0, 120 * i],
                             params={"value": values[p.name]})["uid"]
            placed.append(c)
            for dst in (k, r):
                control.call("wire.connect", src=c, output="out", dst=dst, input=p.name, check_types=False)
        out = spec.outputs[0].name
        timing = {}
        for label, uid in (("kernel", k), ("reference", r)):
            first = control.call("node.request", uid=uid, port=out)
            if first.get("error"):
                raise NodeDevError(f"The {label} failed in the bench graph: {first['error']}")
            control.call("node.timing", uid=uid, reset=True)        # the warm-up is done
            for _ in range(max(1, int(runs))):
                control.call("node.trigger", uid=uid)
                control.call("node.request", uid=uid, port=out)
            timing[label] = control.call("node.timing", uid=uid)
    finally:
        for uid in reversed(placed):
            try:
                control.call("node.remove", uid=uid)
            except Exception:  # noqa: BLE001 - best effort: the bench graph is ours to take down
                pass
    kernel_ms, reference_ms = timing["kernel"]["min_ms"], timing["reference"]["min_ms"]
    factor = reference_ms / kernel_ms if kernel_ms > 0 else math.inf
    within = spec.budget_ms <= 0 or timing["kernel"]["mean_ms"] <= spec.budget_ms
    met = factor >= spec.speedup and within
    result = {"kernel_ms": round(kernel_ms, 4), "reference_ms": round(reference_ms, 4),
              "kernel_mean_ms": round(timing["kernel"]["mean_ms"], 4), "runs": timing["kernel"]["runs"],
              "speedup": round(factor, 2) if factor != math.inf else "inf", "required_speedup": spec.speedup,
              "budget_ms": spec.budget_ms, "within_budget": within, "met": met,
              "inputs": {k: ({"shape": [len(v)]} if isinstance(v, list) else v) for k, v in values.items()}}
    ws.update(kind, bench=result, benched=met, bench_digest=state.get("digest"))
    return {"kind": kind, "ok": met, **result,
            "next": "ask the user to promote it" if met else "make the kernel faster, then build, prove, load, bench"}


# ── step 7: sign-off and promotion ───────────────────────────────────────

def ready(ws: Workspace, kind: str) -> List[str]:
    """What still stands between *kind* and promotion (empty: nothing)."""
    spec, state = ws.load(kind)
    missing = []
    for step, why in (("declared", "the declaration lints clean"), ("referenced", "the reference reproduces "
                      "the examples"), ("linted", "the build passes tiers 1 to 3"), ("proven", "the kernel "
                      "agrees with the reference"), ("benched", "the bench met the speedup and the budget")):
        if not state.get(step):
            missing.append(why)
    if state.get("benched") and state.get("bench_digest") != state.get("digest"):
        missing.append("the bench ran on the latest build")
    return missing


def signoff_detail(ws: Workspace, kind: str, target: str,
                   suggest: Optional[Callable[[Dict[str, Any]], Any]] = None) -> Dict[str, Any]:
    """What a person reviews before promotion: the code, the proof, the
    bench, where it goes; and a classifier's suggestion, if one is bound
    -- shown as a suggestion, never taken as the answer."""
    spec, state = ws.load(kind)
    detail = load_detail(ws, kind)
    detail.update({"target_set": target, "spec": spec.to_dict(), "bench": state.get("bench"),
                   "proof": state.get("proof"), "ready": not ready(ws, kind), "missing": ready(ws, kind)})
    if suggest is not None:
        try:
            detail["suggestion"] = suggest(dict(detail))
        except Exception as exc:  # noqa: BLE001 - a broken advisor is no advice
            detail["suggestion"] = {"error": repr(exc)}
    return detail


def promote(ws: Workspace, kind: str, target_root: os.PathLike, set_id: str, *,
            approved_by: str = "") -> Dict[str, Any]:
    """Step 7 (after sign-off): the kind leaves ``scratch`` for the set
    *set_id* under *target_root* -- its crate (the source, without the
    build), the built library, and the manifest entry; the scratch set
    keeps its reference, not its kernel. The promoted set loads with the
    next session (one set owns a kind, and the scratch set holds it until
    then)."""
    from weave.sdk import rust
    missing = ready(ws, kind)
    if missing:
        raise NodeDevError("Not ready to promote: " + "; ".join(missing) + ".")
    if not _NAME.match(set_id) or set_id == SCRATCH_SET:
        raise NodeDevError(f"Promote into a named set other than {SCRATCH_SET!r} (an identifier).")
    spec, state = ws.load(kind)
    dest = Path(target_root) / set_id
    crate_dest = dest / "crates" / kind
    if crate_dest.exists():
        shutil.rmtree(crate_dest)
    shutil.copytree(ws.crate_dir(kind), crate_dest, ignore=shutil.ignore_patterns("target", "*.new"))
    component = f"{set_id}.{component_id(kind).split('.', 1)[1]}"
    rust.write_set(dest, set_id, Path(state["artefact"]), component=component,
                   description=f"Kinds promoted from Silk's node-dev scratch set.")
    # Out of scratch: its kernel entry and file go; the reference stays as the oracle.
    manifest = ws.scratch / "weave-plugin.toml"
    text = manifest.read_text()
    blocks = re.split(r"(?m)^(?=\[\[component\]\]\s*$)", text)
    keep = [b for b in blocks if f'id = "{component_id(kind)}"' not in b]
    manifest.write_text("".join(keep))
    record = {"set": set_id, "path": str(dest), "component": component, "approved_by": approved_by,
              "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "digest": state.get("digest")}
    ws.update(kind, promoted=record)
    (dest / "PROVENANCE.json").write_text(json.dumps({kind: {**record, "spec": spec.to_dict(),
                                                             "proof": state.get("proof"),
                                                             "bench": state.get("bench")}}, indent=2))
    return {"kind": kind, "ok": True, **record,
            "next": "the promoted set loads with the next session"}


def status(ws: Workspace, kind: Optional[str] = None) -> Dict[str, Any]:
    """Where each kind (or *kind*) stands, and what is next."""
    rows = []
    for k in ([kind] if kind else ws.kinds()):
        spec, state = ws.load(k)
        steps = {s: bool(state.get(s)) for s in ("declared", "referenced", "built", "linted", "proven",
                                                    "loaded", "benched")}
        rows.append({"kind": k, "steps": steps, "bench": state.get("bench"), "promoted": state.get("promoted"),
                     "missing": ready(ws, k), "unreviewed": not state.get("promoted")})
    return {"kinds": rows, "workspace": str(ws.root)}
