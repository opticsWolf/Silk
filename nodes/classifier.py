# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Silk Classifier Node (D-S8, C6).

A typed classifier as a graph resource: hosted Jev first, a local judge
model second, and -- when neither answers -- the human. Its output is a
``silk_classifier`` handle for the Agent node (and, through the
Orchestrator, its workers). What the handle may *do* is decided elsewhere:
the ToolBox's approval policy sets which calls are at ``triage``, the Role
selects classifier hooks. Wiring decides what is possible; policy decides
what is allowed.

Ports
    in   judge_model : model_handle      optional second model, the local judge
    out  classifier  : silk_classifier   ClassifierHandle
    out  stats       : dict              meter (calls, failures, cost, by purpose)

The handle is rebuilt only when the form changes, so the cache, breaker and
meter live across runs; each run snapshots the spec through ``for_run`` (D38).
"""
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLineEdit, QSpinBox

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import MODEL_HANDLE_TYPE, SILK_CLASSIFIER_TYPE  # noqa: F401
from ..functions.classifier import (
    BackendConfig, BandsConfig, ClassifierHandle, ClassifierSpec, build_classifier,
)
from ..functions.classifier_local import judge_client, pool_scorer
from ..functions.credentials import resolve_credential
from ..functions.presets import ClassifierPreset, PresetStore
from ..widgets.bands_table import BandsTableWidget
from ..widgets.preset_bar import PresetBarWidget

log = get_logger("SilkClassifier")

DEFAULT_JEV_MODEL = "jev/jev-1.13.0"
LOCAL_JUDGE_MODEL = "local/judge"

MODES = (
    ("Off - you decide everything", "off"),
    ("Shadow - score and record, never skip", "shadow"),
    ("Live - may skip triage-level questions", "live"),
)


def spec_from_inputs(inputs: Dict[str, Any]) -> ClassifierSpec:
    """The node's form as a :class:`ClassifierSpec`. Qt-free, tested."""
    jev_model = str(inputs.get("jev_model") or DEFAULT_JEV_MODEL).strip() or DEFAULT_JEV_MODEL
    backends = [BackendConfig(kind="jev", model=jev_model)]
    if bool(inputs.get("use_local", True)):
        backends.append(BackendConfig(kind="local", model=str(
            inputs.get("judge_model_id") or LOCAL_JUDGE_MODEL)))
    bands = {}
    for model, raw in dict(inputs.get("bands") or {}).items():
        try:
            bands[str(model)] = BandsConfig.model_validate(raw)
        except Exception:                                  # the table marks it; skip here
            log.warning(f"Ignoring invalid bands for {model!r}")
    return ClassifierSpec(
        mode=str(inputs.get("mode") or "shadow"),
        backends=backends,
        egress_allowed=bool(inputs.get("egress_allowed", False)),
        key_name=str(inputs.get("key_name") or "jev").strip() or "jev",
        tool_calls_per_run=int(inputs.get("tool_calls_per_run") or 0),
        bands=bands,
    )


def judge_model_id(judge_model: Any) -> str:
    """``local/<file stem>``: bands are calibrated per judge *file*."""
    from pathlib import Path
    pool = judge_model.get("pool") if isinstance(judge_model, dict) else judge_model
    path = getattr(pool, "model_path", "") or ""
    return f"local/{Path(path).stem}" if path else LOCAL_JUDGE_MODEL


@register_node
class SilkClassifierNode(ActiveNode):
    """Typed classifier resource: Jev, a local judge, or nobody (humans decide)."""

    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk Classifier"
    node_description: ClassVar[Optional[str]] = (
        "Typed classifier for approval triage, completion checks, classifier hooks and "
        "the classify tool. Hosted Jev, a local judge model, or unwired: you decide.")
    node_tags: ClassVar[Optional[List[str]]] = ["jev", "classifier", "triage", "judge", "silk"]
    node_icon: ClassVar[Optional[str]] = "scale-balanced"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1
    node_version = 1

    def __init__(self, title: str = "Silk Classifier", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        self.add_input("judge_model", datatype="model_handle")
        self.add_output("classifier", datatype="silk_classifier")
        self.add_output("stats", datatype="dict")

        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        # ── Presets ──
        self._preset_store: PresetStore[ClassifierPreset] = PresetStore(
            "silk_classifiers", ClassifierPreset)
        self._preset_bar = PresetBarWidget(
            self._preset_store, collect=self._collect_preset, apply=self._apply_preset)
        form.addRow("Preset:", self._preset_bar)

        # ── Widgets ──
        self.combo_mode = QComboBox()
        for label, value in MODES:
            self.combo_mode.addItem(label, userData=value)
        self.combo_mode.setCurrentIndex(1)
        form.addRow("Mode:", self.combo_mode)
        self._widget_core.register_widget(
            "mode", self.combo_mode, role=PortRole.INTERNAL, datatype="str",
            default="shadow", add_to_layout=False)

        self.edit_model = QLineEdit(DEFAULT_JEV_MODEL)
        self.edit_model.setToolTip("A pinned version: bands are calibrated per model.")
        form.addRow("Jev model:", self.edit_model)
        self._widget_core.register_widget(
            "jev_model", self.edit_model, role=PortRole.INTERNAL, datatype="string",
            default=DEFAULT_JEV_MODEL, policy=debounced(400), add_to_layout=False)

        self.edit_key = QLineEdit("jev")
        self.edit_key.setToolTip("The credential's *name* (D22); the key never enters the graph.")
        form.addRow("Credential:", self.edit_key)
        self._widget_core.register_widget(
            "key_name", self.edit_key, role=PortRole.INTERNAL, datatype="string",
            default="jev", policy=debounced(400), add_to_layout=False)

        self.chk_egress = QCheckBox("May send tool arguments to the hosted classifier (redacted)")
        form.addRow(self.chk_egress)
        self._widget_core.register_widget(
            "egress_allowed", self.chk_egress, role=PortRole.INTERNAL, datatype="bool",
            default=False, add_to_layout=False)

        self.chk_local = QCheckBox("Fall back to the judge model")
        self.chk_local.setChecked(True)
        form.addRow(self.chk_local)
        self._widget_core.register_widget(
            "use_local", self.chk_local, role=PortRole.INTERNAL, datatype="bool",
            default=True, add_to_layout=False)

        self.spin_budget = QSpinBox()
        self.spin_budget.setRange(0, 1000)
        self.spin_budget.setValue(10)
        form.addRow("classify calls / run:", self.spin_budget)
        self._widget_core.register_widget(
            "tool_calls_per_run", self.spin_budget, role=PortRole.INTERNAL, datatype="int",
            default=10, add_to_layout=False)

        self.bands_table = BandsTableWidget()
        form.addRow("Calibrated bands:", self.bands_table)
        self._widget_core.register_widget(
            "bands", self.bands_table, role=PortRole.INTERNAL, datatype="dict",
            default={}, add_to_layout=False)

        self._handle: Optional[ClassifierHandle] = None
        self._handle_key: Optional[str] = None

    # ── Presets ─────────────────────────────────────────────────────

    def _collect_preset(self) -> dict:
        get = self._widget_core.get_port_value
        spec = spec_from_inputs({k: get(k) for k in (
            "mode", "jev_model", "key_name", "egress_allowed", "use_local",
            "tool_calls_per_run", "bands")})
        return {"spec": spec.model_dump(mode="json")}

    def _apply_preset(self, preset: ClassifierPreset) -> None:
        spec = ClassifierSpec.model_validate(preset.spec or {})
        jev = next((b.model for b in spec.backends if b.kind == "jev"), DEFAULT_JEV_MODEL)
        apply = self._widget_core.apply_port_value
        apply("mode", spec.mode)
        apply("jev_model", jev)
        apply("key_name", spec.key_name)
        apply("egress_allowed", spec.egress_allowed)
        apply("use_local", any(b.kind == "local" for b in spec.backends))
        apply("tool_calls_per_run", spec.tool_calls_per_run)
        apply("bands", {m: b.model_dump() for m, b in spec.bands.items()})

    # ── Compute ─────────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        judge = inputs.get("judge_model")
        judge = judge if isinstance(judge, dict) and judge.get("backend") == "gguf" else None
        spec = spec_from_inputs(dict(inputs, judge_model_id=(
            judge_model_id(judge) if judge is not None else LOCAL_JUDGE_MODEL)))
        key = spec.model_dump_json() + f"|{id(judge.get('pool')) if judge else 0}"
        if self._handle is None or key != self._handle_key:
            scorer = None
            if judge is not None and spec.mode != "off":
                try:
                    scorer = pool_scorer(judge_client(judge))
                except ValueError as exc:
                    log.warning(f"Judge model unusable: {exc}")
            chain = build_classifier(spec, secrets=resolve_credential, local_scorer=scorer)
            self._handle = ClassifierHandle(spec, chain, judge)
            self._handle_key = key
        meter = self._handle.classifier.meter
        return {
            "classifier": self._handle,
            "stats": {
                "mode": spec.mode,
                "backends": [b.name for b in self._handle.classifier.backends],
                "calibrated": sorted(spec.bands),
                "calls": meter.calls, "failures": meter.failures,
                "cost_usd": round(meter.cost_usd, 6),
                "by_purpose": dict(meter.by_purpose),
            },
        }

    def cleanup(self) -> None:
        self._preset_bar.cleanup()
        super().cleanup()
