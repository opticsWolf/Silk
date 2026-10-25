# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The node-dev toolset (Weave P6): from a spec, the tools deliver a Rust node
that passes lint tiers 1 to 3, agrees with its Python reference, beats it
by the declared factor in Weave's own timing, and is promoted after a
person signs off. The calls go through tool dispatch, as the model's do;
the person is a scripted answerer on the run's decision seam; the Weave is
a headless one behind the control API (drafts on: ``plugins.*`` and
``node.timing`` are 1.1-draft). Builds use cargo.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from types import SimpleNamespace

import pytest

weave_rs = pytest.importorskip("weave_rs")
if not hasattr(weave_rs, "PluginKernel"):
    pytest.skip("Weave without plugin kernels (P9)", allow_module_level=True)
if shutil.which("cargo") is None:
    pytest.skip("no cargo", allow_module_level=True)

from weave.control import ControlClient  # noqa: E402
from weave.plugin import PluginHost, grant_all  # noqa: E402

from silk.functions import node_dev as nd  # noqa: E402
from silk.functions.approval import bind_run_seam  # noqa: E402
from silk.functions.decision_seam import DecisionSeam  # noqa: E402
from silk.functions.tool_box import ToolBox  # noqa: E402
from silk.functions.tools.node_dev_tools import (  # noqa: E402
    attach_node_dev_tools, bind_signoff_advisor, bind_weave,
)

SPEC = {
    "kind": "SumSquares",
    "description": "The sum of the squares of a signal, scaled.",
    "inputs": [{"name": "x", "datatype": "ndarray", "description": "The signal"}],
    "outputs": [{"name": "total", "datatype": "float"}],
    "params": [{"name": "scale", "datatype": "float", "default": 1.0}],
    "examples": [{"inputs": {"x": [1.0, 2.0]}, "outputs": {"total": 5.0}},
                 {"inputs": {"x": [3.0]}, "params": {"scale": 2.0}, "outputs": {"total": 18.0}}],
    "speedup": 5.0,
    "budget_ms": 50.0,
    "tolerance": 1e-9,
    "generate": {"x": {"shape": [100000], "low": -1.0, "high": 1.0}, "scale": {"low": 0.5, "high": 2.0}},
    "cases": 6,
    "category": "Dev",
}

REFERENCE = """\
total = 0.0
for v in list(x):
    total += float(v) * float(v)
return total * scale
"""

COMPUTE = """    fn compute(&mut self, _ctx: &mut Ctx) -> Result<(), String> {
        self.total = self.x.to_f64().iter().map(|v| v * v).sum::<f64>() * self.scale;
        Ok(())
    }"""

UNWRITTEN = """    fn compute(&mut self, _ctx: &mut Ctx) -> Result<(), String> {
        Err("SumSquares: compute is not written yet".into())
    }"""


class Person:
    """The human on the seam: approves or denies, and keeps what was asked."""

    def __init__(self, answer: bool = True):
        self.answer = answer
        self.asked = []
        self.seam = DecisionSeam(self._ask, timeout_s=30.0)

    def _ask(self, request):
        self.asked.append(request)
        if self.answer:
            self.seam.approve(request.decision_id, actor="frank", remember="run")
        else:
            self.seam.deny(request.decision_id, actor="frank", reason="not today")


def call(box, name, **args):
    request = SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    out = asyncio.run(box.execute_tool_calls_async([request]))
    return json.loads(out[0]["content"])


@pytest.fixture
def weave(tmp_path):
    """A headless Weave as a child process: the tools run on a worker
    thread, and reach Weave as Silk does, over the control API."""
    import os
    import weave as weave_pkg
    root = os.path.dirname(os.path.dirname(weave_pkg.__file__))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
    client = ControlClient.spawn(project=str(tmp_path / "project"), grant_new=True, drafts=True, env=env, cwd=root)
    yield client
    client.close()


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    """One cargo target for the module's crates: the SDK builds once."""
    d = tmp_path_factory.mktemp("node-dev-target")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def box(tmp_path, weave, target, monkeypatch):
    monkeypatch.setenv("WEAVE_NODE_DEV_TARGET", str(target))
    toolbox = ToolBox()
    attach_node_dev_tools(toolbox, root=tmp_path / "node-dev", promote_root=tmp_path / "plugins")
    bind_weave(toolbox, weave)
    person = Person()
    bind_run_seam(toolbox, person.seam)
    toolbox.person = person
    return toolbox


def box_root(box):
    from silk.functions.tools.node_dev_tools import node_dev_workspace
    return node_dev_workspace(box).root


def _to_proven(box, spec=None, reference=REFERENCE, compute=COMPUTE):
    spec = spec or SPEC
    kind = spec["kind"]
    declared = call(box, nd.OP_DECLARE, **spec)
    assert declared["ok"], declared
    assert declared["result"]["findings"] == [] or all(f["severity"] != "error"
                                                       for f in declared["result"]["findings"])
    ref = call(box, nd.OP_REFERENCE, kind=kind, body=reference)
    assert ref["ok"], ref
    scaffold = call(box, nd.OP_SCAFFOLD, kind=kind)
    unwritten = UNWRITTEN.replace("SumSquares", kind)
    assert scaffold["ok"] and unwritten in scaffold["result"]["lib_rs"], scaffold
    source = scaffold["result"]["lib_rs"].replace(unwritten, compute)
    assert call(box, nd.OP_WRITE_KERNEL, kind=kind, source=source)["ok"]
    built = call(box, nd.OP_BUILD, kind=kind)
    assert built["ok"], json.dumps(built, indent=1)
    assert built["result"]["errors"] == 0
    proven = call(box, nd.OP_PROVE, kind=kind)
    assert proven["ok"], json.dumps(proven, indent=1)
    assert proven["result"]["cases"] == len(spec["examples"]) + spec["cases"]
    return source


def test_from_a_spec_to_a_promoted_rust_node(box, weave, tmp_path):
    """The P6 gate."""
    _to_proven(box)
    person = box.person
    # Nothing has been asked yet: building and proving ran outside Weave,
    # in staging; the scratch set Weave loads holds nothing yet.
    assert person.asked == []
    assert not (tmp_path / "node-dev" / "sets" / "scratch").exists()
    loaded = call(box, nd.OP_LOAD, kind="SumSquares")
    assert loaded["ok"] and loaded["result"]["unreviewed"], loaded
    (asked,) = person.asked
    assert asked.tool_name == nd.OP_LOAD
    assert "self.x.to_f64()" in asked.detail["kernel"]["source"], "the person sees the code"
    assert "total += float(v)" in asked.detail["reference"]["source"]
    assert asked.detail["sandboxed"] is False
    sets = {s["id"].split("/")[-1] for s in weave.call("plugins.list")["sets"]}
    assert nd.SCRATCH_SET in sets

    bench = call(box, nd.OP_BENCH, kind="SumSquares", runs=8)
    assert bench["ok"], json.dumps(bench, indent=1)
    r = bench["result"]
    assert r["met"] and r["speedup"] >= SPEC["speedup"] and r["within_budget"], r
    assert r["runs"] == 8, "the engine counted the computes"
    assert weave.call("graph.describe")["nodes"] == [], "the bench graph is taken down"

    # "Don't ask again" was answered, and is not honoured: a second load asks again.
    again = call(box, nd.OP_LOAD, kind="SumSquares")
    assert again["ok"] and len(person.asked) == 2, "asked every time"

    # Sign-off: a classifier may suggest; the person decides.
    bind_signoff_advisor(box, lambda detail: {"suggest": "approve", "why": "proven and fast"})
    promoted = call(box, nd.OP_PROMOTE, kind="SumSquares", set_id="fast_math")
    assert promoted["ok"], promoted
    signoff = person.asked[-1]
    assert signoff.tool_name == nd.OP_PROMOTE and signoff.detail["suggestion"]["suggest"] == "approve"
    assert signoff.detail["bench"]["met"] and signoff.detail["ready"]
    assert promoted["result"]["approved_by"] == "frank"
    dest = tmp_path / "plugins" / "fast_math"
    manifest = (dest / "weave-plugin.toml").read_text()
    assert 'id = "fast_math"' in manifest and '"SumSquares"' in manifest
    assert (dest / "crates" / "SumSquares" / "src" / "lib.rs").is_file()
    assert not (dest / "crates" / "SumSquares" / "target").exists()
    assert json.loads((dest / "PROVENANCE.json").read_text())["SumSquares"]["approved_by"] == "frank"
    scratch = (tmp_path / "node-dev" / "sets" / "scratch" / "weave-plugin.toml").read_text()
    assert '"SumSquares"' not in scratch.replace('"SumSquaresReference"', ""), "it left the scratch set"
    # It moved in this session: the promoted set owns the kind now, and the
    # scratch set (its reference left) loaded again.
    assert promoted["result"]["moved"] is True, promoted
    listed = {s["id"].split("/")[-1]: s for s in weave.call("plugins.list")["sets"]}
    assert "local/fast_math:SumSquares" in listed["fast_math"]["kinds"] and not listed["fast_math"]["unreviewed"]
    assert listed[nd.SCRATCH_SET]["unreviewed"] and all("Reference" in k for k in listed[nd.SCRATCH_SET]["kinds"])
    decls = {d["kind"]: d for d in weave.call("graph.kinds")["kinds"]}
    assert "unreviewed" not in decls["SumSquares"]["meta"].get("tags", [])
    assert "unreviewed" in decls["SumSquaresReference"]["meta"]["tags"]
    # And in a new session (one set owns a kind).
    fresh = PluginHost(ask=grant_all)
    assert fresh.load([tmp_path / "plugins"]).loaded == ["local/fast_math"]


SUM_CUBES = dict(SPEC, kind="SumCubes", description="The sum of the cubes of a signal, scaled.",
                 examples=[{"inputs": {"x": [1.0, 2.0]}, "outputs": {"total": 9.0}}])
CUBES_REFERENCE = REFERENCE.replace("float(v) * float(v)", "float(v) ** 3")
CUBES = COMPUTE.replace("v * v", "v * v * v")


def test_a_kind_new_to_a_loaded_scratch_set_loads_in_the_same_session(box, weave, tmp_path):
    _to_proven(box)
    assert call(box, nd.OP_LOAD, kind="SumSquares")["ok"]
    uid = weave.call("node.add", kind="SumSquares")["uid"]
    _to_proven(box, SUM_CUBES, CUBES_REFERENCE, CUBES)
    staged = (tmp_path / "node-dev" / "sets" / "scratch" / "weave-plugin.toml").read_text()
    assert "SumCubes" not in staged, "built and proven, not yet let load"
    loaded = call(box, nd.OP_LOAD, kind="SumCubes")
    assert loaded["ok"] and loaded["result"]["set_reloaded"], loaded
    kinds = {d["kind"] for d in weave.call("graph.kinds")["kinds"]}
    assert {"SumSquares", "SumCubes", "SumCubesReference"} <= kinds
    assert weave.call("node.describe", uid=uid)["kind"] == "SumSquares", "the open node stayed"
    tags = {d["kind"]: d["meta"].get("tags", []) for d in weave.call("graph.kinds")["kinds"]}
    assert "unreviewed" in tags["SumCubes"] and "unreviewed" in tags["SumSquares"]


def test_a_kernel_that_disagrees_is_not_proven_and_not_offered(box):
    declared = call(box, nd.OP_DECLARE, **SPEC)
    assert declared["ok"]
    assert call(box, nd.OP_REFERENCE, kind="SumSquares", body=REFERENCE)["ok"]
    lib = call(box, nd.OP_SCAFFOLD, kind="SumSquares")["result"]["lib_rs"]
    # Right on the examples (which tier 2 checks), wrong on long signals.
    wrong = COMPUTE.replace("self.x.to_f64().iter()", "self.x.to_f64().iter().take(4)")
    assert call(box, nd.OP_WRITE_KERNEL, kind="SumSquares", source=lib.replace(UNWRITTEN, wrong))["ok"]
    assert call(box, nd.OP_BUILD, kind="SumSquares", clippy=False)["ok"]
    proven = call(box, nd.OP_PROVE, kind="SumSquares")
    assert not proven["ok"] and proven["result"]["failed"] > 0
    failure = proven["result"]["failures"][0]
    assert failure["side"] == "kernel" and failure["inputs"]["x"] == {"shape": [100000]}
    refused = call(box, nd.OP_LOAD, kind="SumSquares")
    assert refused["applied"] is False and "Prove" in refused["message"]
    assert box.person.asked == [], "an unproven build is not put to the person"


def test_a_compile_error_and_a_bad_spec_come_back_as_findings(box):
    bad = call(box, nd.OP_DECLARE, **{**SPEC, "kind": "sum squares", "examples": [{"inputs": {}}]})
    assert not bad["ok"] and "CamelCase" in bad["message"] and "examples[0]" in bad["message"]
    assert call(box, nd.OP_DECLARE, **SPEC)["ok"]
    early = call(box, nd.OP_SCAFFOLD, kind="SumSquares")
    assert not early["ok"] and "reference" in early["message"], "reference first"
    assert call(box, nd.OP_REFERENCE, kind="SumSquares", body=REFERENCE)["ok"]
    lib = call(box, nd.OP_SCAFFOLD, kind="SumSquares")["result"]["lib_rs"]
    typo = lib.replace(UNWRITTEN, COMPUTE.replace("self.scale", "self.scael"))
    assert call(box, nd.OP_WRITE_KERNEL, kind="SumSquares", source=typo)["ok"]
    built = call(box, nd.OP_BUILD, kind="SumSquares", clippy=False)
    assert not built["ok"] and built["result"]["stage"] == "build"
    (error, *_) = built["result"]["findings"]
    assert error["severity"] == "error" and "scael" in (error["message"] + error.get("hint", ""))
    assert ":" in error["location"]


def test_a_reference_that_misses_an_example_is_not_the_oracle(box):
    assert call(box, nd.OP_DECLARE, **SPEC)["ok"]
    wrong = call(box, nd.OP_REFERENCE, kind="SumSquares", body="return float(sum(x))")
    assert not wrong["ok"] and wrong["result"]["failures"][0]["side"] == "reference"


def test_denied_or_unasked_nothing_loads_and_nothing_is_promoted(box, weave):
    _to_proven(box)
    box.person.answer = False
    denied = call(box, nd.OP_LOAD, kind="SumSquares")
    assert denied["applied"] is False and "frank" in denied["message"]
    assert not (box_root(box) / "sets" / "scratch").exists(), "nothing reached the set Weave loads"
    assert nd.SCRATCH_SET not in {s["id"].split("/")[-1] for s in weave.call("plugins.list")["sets"]}
    early = call(box, nd.OP_PROMOTE, kind="SumSquares", set_id="fast_math")
    assert early["applied"] is False and "bench" in early["message"], "not ready: nobody is asked"
    assert len(box.person.asked) == 1
    bind_run_seam(box, None)
    headless = call(box, nd.OP_LOAD, kind="SumSquares")
    assert headless["applied"] is False and "no way to ask" in headless["message"]
