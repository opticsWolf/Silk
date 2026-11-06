# -*- coding: utf-8 -*-
"""Preset bars mirror as working twins (classifier/ensemble/inference
settings/role/toolset).

Qt-free tests run everywhere; widget tests need the Weave env (PySide6)
and error here -- same convention as the other node suites.
"""
import pytest


def test_bar_contract_is_mirror_ready(qt_app):
    from silk.widgets.preset_bar import PresetBarWidget
    contract = PresetBarWidget.__mirror__
    assert contract.value_signal == "preset_chosen"
    assert contract.default_datatype == "str"
    assert callable(contract.read_value) and callable(contract.write_value)
    assert callable(contract.clone)


# ── Qt widget/node shape (Weave env; imports local so collection works ──
# anywhere) ───────────────────────────────────────────────────────────────
@pytest.fixture()
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def bar(qt_app, tmp_path):
    from silk.functions.presets import InferenceSettingsPreset, PresetStore
    from silk.widgets.preset_bar import PresetBarWidget
    store = PresetStore("inference_settings", InferenceSettingsPreset,
                        directory=tmp_path)
    applied = []
    return PresetBarWidget(store, collect=lambda: {"temperature": 0.5},
                           apply=applied.append)


def test_twin_shares_the_live_list(bar):
    from PySide6.QtTest import QSignalSpy
    twin = type(bar).clone_of(bar)
    assert twin._combo.model() is bar._combo.model()
    spy = QSignalSpy(bar.preset_chosen)
    payload = {"name": "calm", "temperature": 0.2}
    bar._store.upsert(bar._store.model.model_validate(payload))
    bar.refresh(select="calm")
    assert twin._combo.findText("calm") >= 0
    assert len(spy) == 1


def test_select_name_never_applies(qt_app, tmp_path):
    from PySide6.QtTest import QSignalSpy
    from silk.functions.presets import InferenceSettingsPreset, PresetStore
    from silk.widgets.preset_bar import PresetBarWidget
    applied = []
    bar = PresetBarWidget(
        PresetStore("inference_settings", InferenceSettingsPreset,
                    directory=tmp_path),
        collect=lambda: {"temperature": 0.5}, apply=applied.append)
    payload = {"name": "calm", "temperature": 0.2}
    bar._store.upsert(bar._store.model.model_validate(payload))
    bar.refresh()
    bar._combo.setCurrentIndex(0)
    spy = QSignalSpy(bar.preset_chosen)
    bar.select_name("calm")
    assert bar._combo.currentText() == "calm"
    assert len(spy) == 0
    assert applied == []


def test_five_nodes_register_the_bar(qt_app):
    from silk.nodes.classifier import SilkClassifierNode
    from silk.nodes.ensemble import SilkEnsembleNode
    from silk.nodes.inference_settings import AgentInferenceSettingsNode
    from silk.nodes.role import SilkRoleNode
    from silk.nodes.toolset import SilkToolSetNode
    for cls in (SilkClassifierNode, SilkEnsembleNode,
                AgentInferenceSettingsNode, SilkRoleNode, SilkToolSetNode):
        node = cls()
        assert "preset" in node._widget_core._bindings
