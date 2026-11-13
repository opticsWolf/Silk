# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

S4, Qt-free: bobine on the shared vocabulary, and the warm host.

The Converter Settings rows speak the stack words -- ``auto|cpu|cuda``
and ``auto|fp32|fp16|int8``, the same tuples bobine 0.9 exports and the
ONNX Loader shows -- and the worker passes them through instead of
translating to a separate precision/quantization pair.

The warm host is ``functions/bobine_server.py`` (bobine.serve's handler
and pool, with Silk's settings dict translated) behind
``functions/bobine_pool.BobinePool``, itself a ``ChildServerPool``. The
live tests skip where bobine (0.9, the one with ``bobine.serve``) is not
installed.
"""
from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace as NS

import pytest

from silk.functions import bobine_convert as bc
from silk.functions.bobine_convert import ConversionCancelled
from silk.functions.bobine_pool import (
    SERVER_SCRIPT, pool_command, server_args,
)
from silk.functions.bobine_worker import config_kwargs

needs_host = pytest.mark.skipif(
    not bc.BOBINE_AVAILABLE or importlib.util.find_spec("bobine.serve") is None,
    reason="bobine 0.9 (with bobine.serve) is not installed")


def fake_bob():
    """Enough of bobine's module surface for config_kwargs, no import."""
    return NS(
        RoutingMode=NS(Auto="auto-enum", Surgical="surgical-enum"),
        TableBackend=NS(Slanet="slanet-enum", Potatr="potatr-enum"),
        ImageMode=NS(Link="link-enum", EmbedBase64="embed-enum"),
    )


# ── the vocabulary ──────────────────────────────────────────────────────

def test_the_rows_are_the_stack_tuples():
    assert bc.DEVICES == ("auto", "cpu", "cuda")
    assert bc.PRECISIONS == ("auto", "fp32", "fp16", "int8")
    assert bc.DEFAULT_SETTINGS["device"] == "auto"
    assert bc.DEFAULT_SETTINGS["precision"] == "auto"
    assert "quantization" not in bc.DEFAULT_SETTINGS


def test_clean_settings_maps_the_words_scenes_already_saved():
    assert bc.clean_settings({"device": "gpu"})["device"] == "cuda"
    assert bc.clean_settings({"device": "tpu"})["device"] == "auto"
    old = bc.clean_settings({"precision": "fp32", "quantization": "int8"})
    assert old["precision"] == "int8", "the old formula-weights row folded in"
    assert bc.clean_settings({"precision": "fp16", "quantization": "int8"})["precision"] == "fp16"
    assert "quantization" not in bc.clean_settings({"quantization": "int8", "precision": "fp16"})


def test_the_worker_passes_device_and_precision_through():
    kw = config_kwargs(fake_bob(), bc.clean_settings(
        {"routing": "surgical", "device": "cuda", "precision": "int8"}))
    assert kw["device"] == "cuda" and kw["precision"] == "int8"
    assert kw["routing_mode"] == "surgical-enum", "the enum, not its word"
    assert kw["table_backend"] == "slanet-enum"
    assert "model_precision" not in kw and "model_quantization" not in kw
    assert "ort_providers" not in kw, "bobine 0.9 picks providers from device"


def test_describe_uses_the_shared_words():
    assert bc.describe({}) == ("route auto, device auto, precision auto, "
                              "tables slanet")
    assert bc.describe({"routing": "never"}) == "route never, tables slanet"


# ── the host's command and settings -------------------------------------

def test_the_host_command_is_the_launcher_with_its_flags():
    argv = pool_command("C:/cache", 45)
    assert argv[0] == sys.executable
    assert argv[1] == SERVER_SCRIPT and argv[1].endswith("bobine_server.py")
    assert argv[2:4] == ["--idle-seconds", "45.0"]
    assert argv[4:6] == ["--cache-dir", "C:/cache"]
    assert argv[-4] == "--host" and argv[-3] == "127.0.0.1"
    assert argv[-2] == "--port"
    assert "--cache-dir" not in server_args(""), "no flag when there is no cache"


# ── the warm host, live --------------------------------------------------

@needs_host
def test_a_warm_host_converts_and_keeps_its_converter(tmp_path):
    pool = bc.warm_pool({"routing": "never", "device": "cpu", "precision": "fp32"},
                        idle_seconds=60)
    try:
        assert pool.snapshot()["started"] is True
        pages = []
        first = tmp_path / "one.txt"
        first.write_text("# One\n\nFirst body.\n", encoding="utf-8")
        settings = {"routing": "never", "device": "cpu", "precision": "fp32"}
        out = bc.convert_file(str(first), str(tmp_path / "out"), settings,
                              pool=pool, on_page=lambda i, n: pages.append((i, n)))
        assert "First body." in out["markdown"] and out["md_path"]
        second = tmp_path / "two.txt"
        second.write_text("Second body.\n", encoding="utf-8")
        out2 = bc.convert_file(str(second), str(tmp_path / "out2"),
                               settings, pool=pool)
        assert "Second body." in out2["markdown"]

        report = pool.models()
        warm = [row for row in report["data"] if row.get("object") == "warm"][0]["data"]
        assert len(warm) == 1, "one settings key, one converter"
        assert warm[0]["documents"] == 2, "the second file reused it"
        assert warm[0]["settings"]["device"] == "cpu"
        assert pages == [], "a text file reports no pages"
    finally:
        pool.cleanup()
    assert pool.snapshot()["started"] is False


@needs_host
def test_a_cancel_stops_the_host(tmp_path):
    pool = bc.warm_pool({"routing": "never"}, idle_seconds=60)
    note = tmp_path / "note.txt"
    note.write_text("Text.\n", encoding="utf-8")
    with pytest.raises(ConversionCancelled):
        pool.convert(str(note), str(tmp_path / "out"), {"routing": "never"},
                     should_continue=lambda: False)
    assert pool.snapshot()["started"] is False, "a shared host cannot abort one file"
