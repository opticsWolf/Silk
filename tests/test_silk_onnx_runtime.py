# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

S5, Qt-free: ONNX Runtime status asked once, in a child, and read from
one place by every ONNX node.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from silk.functions import onnx_runtime

OK = {"embroider": "0.15.0", "onnxruntime": "1.31.1",
      "info": {"ort_dylib": "C:/ort/onnxruntime.dll",
               "ort_dylib_exists": True, "cuda_usable": True}}
CPU = {"embroider": "0.15.0", "onnxruntime": "1.31.1",
       "info": {"ort_dylib": "C:/ort/onnxruntime.dll",
                "ort_dylib_exists": True, "cuda_usable": False}}


@pytest.fixture(autouse=True)
def fresh():
    onnx_runtime.forget()
    yield
    onnx_runtime.forget()


def counting(answer):
    calls = []

    def runner(executable):
        calls.append(executable)
        return dict(answer)

    return runner, calls


def test_the_answer_is_asked_once_and_cached():
    runner, calls = counting(OK)
    assert onnx_runtime.problem(runner=runner) == ""
    assert onnx_runtime.describe(runner=runner) == \
        "embroider 0.15.0, ONNX Runtime 1.31.1, CUDA ready"
    assert calls == [None], "one child, however often a node asks"
    onnx_runtime.info(refresh=True, runner=runner)
    assert len(calls) == 2


def test_a_missing_embroider_is_one_sentence():
    answer = {"error": "ModuleNotFoundError: No module named 'embroider'"}
    text = onnx_runtime.problem(answer)
    assert "embroider is not installed" in text
    assert "okfgraph[cpu]" in text
    assert text == onnx_runtime.describe(answer)


def test_another_probe_failure_names_itself():
    text = onnx_runtime.problem({"error": "RuntimeError: boom"})
    assert "could not be asked" in text and "RuntimeError: boom" in text


def test_a_missing_ort_is_an_install_hint():
    answer = {"info": {"ort_dylib": "C:/x/onnxruntime.dll",
                       "ort_dylib_exists": False, "cuda_usable": False}}
    text = onnx_runtime.problem(answer)
    assert "No ONNX Runtime at C:/x/onnxruntime.dll" in text
    assert "okfgraph[cpu]" in text


def test_cuda_is_a_promise_the_runtime_must_keep():
    assert "pick CPU" in onnx_runtime.problem(CPU, device="cuda")
    assert onnx_runtime.problem(CPU, device="cpu") == ""
    assert onnx_runtime.problem(CPU, device="auto") == ""


def test_the_quiet_line_says_what_will_run():
    assert onnx_runtime.describe(CPU) == \
        "embroider 0.15.0, ONNX Runtime 1.31.1, CPU only"
    assert onnx_runtime.briefly(CPU) == "ONNX Runtime 1.31.1, CPU only"
    assert onnx_runtime.briefly(OK) == "ONNX Runtime 1.31.1, CUDA ready"


def test_a_tooltip_only_improves_after_someone_asked():
    assert onnx_runtime.cached_text() == "", "no child while building a node"
    onnx_runtime.info(runner=counting(OK)[0])
    assert "ONNX Runtime 1.31.1" in onnx_runtime.cached_text()


@pytest.mark.skipif(importlib.util.find_spec("embroider") is None,
                    reason="embroider is not installed here")
def test_the_real_probe_runs_in_a_child_and_stays_out_of_this_process():
    answer = onnx_runtime.info(refresh=True)
    assert "error" not in answer, answer
    assert answer["info"]["ort_dylib_exists"] is True
    assert onnx_runtime.problem(answer) == ""
    assert "onnxruntime" not in sys.modules, "never in Weave's process"
    assert "embroider" not in sys.modules
