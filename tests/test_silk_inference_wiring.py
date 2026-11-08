# -*- coding: utf-8 -*-
"""Inference-settings wiring: the dict rides the model wire, not the run.

Qt-free helper tests run everywhere; node port-shape tests need the Weave
env (PySide6) and error here -- same convention as the other node suites.
"""
import ast as _ast
import pathlib

import pytest

from silk.functions.graph_engine import apply_inference_settings
from silk.functions.graph_engine import handle_gen_params
from silk.functions.model_fallback import build_chain, chain_of


def test_stamp_keeps_a_copy_of_the_wired_dict():
    handle: dict = {"backend": "gguf"}
    wired = {"temperature": 0.3, "top_p": 0.9}
    out = apply_inference_settings(handle, wired)
    assert out is handle
    assert handle["gen_params"] == wired
    assert handle["gen_params"] is not wired


def test_stamp_ignores_missing_empty_and_foreign_inputs():
    assert apply_inference_settings({}, None) == {}
    assert apply_inference_settings({}, {}) == {}
    assert apply_inference_settings({}, []) == {}
    assert apply_inference_settings({}, "temperature=0.3") == {}


def test_agent_reads_the_model_wire():
    assert handle_gen_params(None) == {}
    assert handle_gen_params({"backend": "gguf"}) == {}
    assert handle_gen_params({"gen_params": []}) == {}
    handle = apply_inference_settings({"backend": "gguf"},
                                      {"temperature": 0.3})
    assert handle_gen_params(handle) == {"temperature": 0.3}


def test_settings_survive_a_fallback_chain():
    primary = apply_inference_settings({"backend": "gguf", "pool": 1},
                                       {"temperature": 0.3,
                                        "top_p": 0.9})
    backup = {"backend": "gguf", "pool": 2}
    chained = build_chain(primary, backup)
    assert chained is not None
    # Flattened for the walk, or whole for the agent: same stamp.
    assert handle_gen_params(chain_of(chained)[0]) == {"temperature": 0.3,
                                                       "top_p": 0.9}
    assert handle_gen_params(chained) == {"temperature": 0.3,
                                          "top_p": 0.9}
    # The chain is a new dict carrying the primary's stamp by value.
    assert chained is not primary
    assert chained["gen_params"] == primary["gen_params"]


# ── Qt node shape (Weave env; imports local so collection works anywhere) ──
@pytest.fixture()
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_gguf_loader_offers_an_inference_inlet(qt_app):
    from silk.nodes.gguf_loader import GGUFLNode
    node = GGUFLNode()
    inlets = {p.name: p.datatype for p in node.inputs}
    assert inlets.get("inference_settings") == "dict"


def test_agent_no_longer_takes_run_time_settings(qt_app):
    from silk.nodes.agent import SilkAgentNode
    from silk.nodes.orchestrator import SilkOrchestratorNode
    for cls in (SilkAgentNode, SilkOrchestratorNode):
        assert "inference_settings" not in [p.name for p in cls().inputs]
    # 4 dropped the port; 5 (Task box gone, stop/clear inputs) kept it gone.
    assert SilkAgentNode.node_version >= 4


def test_no_dangling_self_calls_inference_wiring():
    # Framework/inherited members resolve outside the class body; the
    # guard only fails on names that are neither defined, inherited, nor
    # plain framework API.
    inherited = {
        "nodes/gguf_loader.py": {
            "_get_cached_value", "cancel_compute", "emit_stream",
            "is_compute_cancelled", "on_ui_change",
        },
        "nodes/agent.py": {
            "_fan_out_stream", "_get_cached_value", "_start_computing_pulse",
            "emit_stream", "get_pulse_waveform", "is_compute_cancelled", "pulse",
            "set_pulse_waveform",
        },
    }
    for path, cls_name in (
        ("nodes/gguf_loader.py", "GGUFLNode"),
        ("nodes/agent.py", "SilkAgentNode"),
    ):
        src = (pathlib.Path(__file__).resolve().parent.parent
               / path)
        tree = _ast.parse(src.read_text(encoding="utf-8"))
        cls = next(n for n in _ast.walk(tree)
                   if isinstance(n, _ast.ClassDef) and n.name == cls_name)
        defined = {n.name for n in cls.body
                   if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
        called = {n.func.attr for n in _ast.walk(cls)
                  if isinstance(n, _ast.Call)
                  and isinstance(n.func, _ast.Attribute)
                  and isinstance(n.func.value, _ast.Name)
                  and n.func.value.id == "self"}
        allow = {"add_input", "add_output", "set_content_widget",
                 "cancel_compute", "emit_stream", "report_progress"}
        allow |= inherited[path]
        assert not called - defined - allow, path
