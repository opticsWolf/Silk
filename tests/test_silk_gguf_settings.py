# -*- coding: utf-8 -*-
"""Per-model JSON memory, probe-driven sliders, MTP switch (GGUF Loader).

Qt-free tests run everywhere; node tests need the Weave env (PySide6)
and error here -- same convention as the other node suites.
"""

from silk.functions.model_pool import _SERVER_MODEL_KEYS
from silk.functions.presets import GGUFModelPreset, GGUFModelStore


def test_gguf_preset_defaults_match_loader():
    preset = GGUFModelPreset(name="x")
    assert (preset.gpu_layers, preset.n_ctx, preset.n_threads,
            preset.seed) == (-1, 4096, -1, -1)
    assert preset.enable_mmproj is False and preset.enable_mtp is False
    assert preset.mtp_draft_tokens == 2


def test_gguf_store_roundtrip_and_moved_fallback(tmp_path):
    store = GGUFModelStore(directory=tmp_path)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"0" * 64)
    assert store.lookup(str(model)) is None
    store.remember(str(model), gpu_layers=35, n_ctx=8192,
                   enable_mtp=True, mtp_draft_tokens=3)
    hit = store.lookup(str(model))
    assert hit is not None and hit.gpu_layers == 35
    assert hit.enable_mtp is True and hit.mtp_draft_tokens == 3
    moved = tmp_path / "sub"
    moved.mkdir()
    twin = moved / "m.gguf"
    twin.write_bytes(b"0" * 64)
    fallback = store.lookup(str(twin))
    assert fallback is not None and fallback.n_ctx == 8192


def test_server_keys_cover_vision_and_mtp():
    assert {"mmproj", "spec_type", "spec_draft_n_max"} <= set(
        _SERVER_MODEL_KEYS)


# ── Qt node shape (Weave env; imports local so collection works anywhere) ──
import pytest


@pytest.fixture()
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def node(qt_app):
    from silk.nodes.gguf_loader import GGUFLNode
    return GGUFLNode()


def test_mtp_ports(node):
    bindings = node._widget_core._bindings
    assert bindings["enable_mtp"].widget is node.chk_mtp
    assert node._widget_core.get_port_value("enable_mtp") is False
    assert node._widget_core.get_port_value("mtp_draft_tokens") == 2


def test_sliders_follow_probe_and_spinners(node):
    node._apply_probe("x", {"exists": True, "size_gb": 4.0,
                            "max_ctx": 8192, "layers": 32})
    assert node.spin_gpu.maximum() == 33
    assert node.slider_gpu.maximum() == 33
    assert node.spin_ctx.maximum() == 8192
    assert node.slider_ctx.maximum() == 8192
    node.slider_gpu.setValue(10)
    assert node.spin_gpu.value() == 10
    node.spin_ctx.setValue(4096)
    assert node.slider_ctx.value() == 4096
    node.spin_gpu.setValue(-1)
    assert node.slider_gpu.value() == node.slider_gpu.maximum()
    assert node._widget_core.get_port_value("gpu_layers_slider") == \
        node.slider_gpu.value()


def test_probe_shrink_cascades_to_spinners(node):
    node.slider_gpu.setValue(10)
    assert node.spin_gpu.value() == 10
    node._apply_probe("x", {"exists": True, "size_gb": 4.0,
                            "max_ctx": 4096, "layers": 4})
    assert node.spin_gpu.maximum() == 5
    assert node.slider_gpu.maximum() == 5
    assert node.spin_gpu.value() == 5
    assert node.slider_gpu.value() == 5
    assert node._widget_core.get_port_value("gpu_layers") == 5


def test_restore_kicks_detection(node, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"0" * 64)
    state = node.get_state()
    state.setdefault("widget_data", {})["model_path"] = str(model)
    node.restore_state(state)
    assert node._restored_for == str(model)


def test_settings_restore_on_reconnect(node, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"0" * 64)
    node._model_store.remember(str(model), gpu_layers=7, n_ctx=8192,
                               n_threads=4, enable_mtp=True,
                               mtp_draft_tokens=3)
    node._widget_core.apply_port_value("model_path", str(model),
                                       user_edit=False)
    node._update_estimation()
    assert node.spin_gpu.value() == 7
    assert node.spin_ctx.value() == 8192
    assert node.spin_threads.value() == 4
    assert node.chk_mtp.isChecked()
    assert node.spin_mtp_tokens.value() == 3
