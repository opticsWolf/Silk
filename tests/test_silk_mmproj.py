# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Vision-projector (mmproj) support: server key, detection, tooltips.

Qt-free tests run everywhere; node tests need the Weave env (PySide6)
and error here -- same convention as the other node suites.
"""
import pytest

from silk.functions.model_pool import _SERVER_MODEL_KEYS


def test_server_config_forwards_mmproj():
    assert "mmproj" in _SERVER_MODEL_KEYS


# ── Qt node shape (Weave env; imports local so collection works anywhere) ──
@pytest.fixture()
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def node(qt_app):
    from silk.nodes.gguf_loader import GGUFLNode
    return GGUFLNode()


def _port(node, name):
    return next(p for p in node.inputs if p.name == name)


def test_mmproj_ports(node):
    # The switch is node-local config (INTERNAL); only the file is a port.
    assert "enable_mmproj" not in {p.name for p in node.inputs}
    assert node._widget_core._bindings["enable_mmproj"].widget is node.chk_mmproj
    assert node._widget_core.get_port_value("enable_mmproj") is False
    assert _port(node, "mmproj_path").datatype == "filepath"


def test_detection_fills_the_picker_once(node, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"0" * 64)
    (tmp_path / "mmproj-f16.gguf").write_bytes(b"0" * 32)
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    wc = node._widget_core
    wc.apply_port_value("model_path", str(tmp_path / "model.gguf"),
                        user_edit=False)
    wc.apply_port_value("enable_mmproj", True, user_edit=False)
    node._update_estimation()
    assert wc.get_port_value("mmproj_path").endswith("mmproj-f16.gguf")
    # A cleared picker stays cleared: manual text-only wins, no rescan.
    wc.apply_port_value("mmproj_path", "", user_edit=False)
    node._update_estimation()
    assert wc.get_port_value("mmproj_path") == ""


def test_every_widget_has_a_tooltip(node):
    missing = [name for name, b in node._widget_core._bindings.items()
               if not b.widget.toolTip()]
    assert missing == []
