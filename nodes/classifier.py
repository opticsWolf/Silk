# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Silk Classifier Node (D-S8, C6).

A typed classifier as a graph resource: a wired classifier model first (Jev
Loader, Classifier Loader), a local judge model second, and -- when neither
answers --
the human. Its output is a
``silk_classifier`` handle for the Agent node (and, through the
Orchestrator, its workers). What the handle may *do* is decided elsewhere:
the ToolBox's approval policy sets which calls are at ``triage``, the Role
selects classifier hooks. Wiring decides what is possible; policy decides
what is allowed.

Ports
    in   classifier_model : classifier_model  first model, a wired decision backend:
                                        Laya-style encoder answering all typed questions
                                        in one CPU forward pass, or a hosted endpoint
                                        (Jev) with the same questions and answers
    in   judge_model : model_handle      optional second model, the local judge:
                                        generative chat GGUF scoring each option by
                                        log-probability (slower, GPU-hungry); must differ
                                        from the agent's own model (self-grading refused)
    out  classifier  : silk_classifier   ClassifierHandle
    out  stats       : dict              meter (calls, failures, cost, by purpose)

The handle is rebuilt only when the form changes, so the cache, breaker and
meter live across runs; each run snapshots the spec through ``for_run`` (D38).
"""
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QSpinBox

from weave.widgetcore import WidgetCore, PortRole
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import MODEL_HANDLE_TYPE, SILK_CLASSIFIER_TYPE  # noqa: F401
from ..functions.classifier import (
    BackendConfig, BandsConfig, ClassifierHandle, ClassifierSpec, build_classifier,
    decision_member,
)
from ..functions.classifier_local import (
    LOCAL_JUDGE_MODEL, judge_client, judge_model_id, pool_scorer,
)
from ..functions.credentials import resolve_credential
from ..functions.presets import ClassifierPreset, PresetStore
from ..widgets.bands_table import BandsTableWidget
from ..widgets.preset_bar import PresetBarWidget

log = get_logger("SilkClassifier")

LOCAL_DECISION_MODEL = "local/decision"

MODES = (
    ("Off - you decide everything", "off"),
    ("Shadow - score and record, never skip", "shadow"),
    ("Live - may skip triage-level questions", "live"),
)


def spec_from_inputs(inputs: Dict[str, Any]) -> ClassifierSpec:
    """The node's form as a :class:`ClassifierSpec`. Qt-free, tested.

    Both links are always admitted: a wired handle answers, an unwired
    port reads as absent and its link is skipped at answer time. Wiring
    decides what is possible; policy (mode, bands, egress) decides what
    is allowed. Hosted endpoints arrive by wire (a Jev Loader's
    classifier model), not from this form: the entry's URL, key name
    and egress come from the handle itself (see ``decision_member``).
    """
    backends = [
        BackendConfig(kind="local", model=str(
            inputs.get("judge_model_id") or LOCAL_JUDGE_MODEL)),
        BackendConfig(kind="systemone", model=str(
            inputs.get("decision_model_id") or LOCAL_DECISION_MODEL)),
    ]
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
        tool_calls_per_run=int(inputs.get("tool_calls_per_run") or 0),
        bands=bands,
    )


@register_node
class SilkClassifierNode(ActiveNode):
    """Typed classifier resource: a wired classifier model (Classifier Loader,
    Jev Loader), a local judge model, or nobody (humans decide)."""

    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk Classifier"
    node_description: ClassVar[Optional[str]] = (
        "Typed classifier for approval triage, completion checks, classifier hooks and "
        "the classify tool. A wired classifier model (Jev Loader, Classifier Loader), "
        "a local judge model, or unwired: you decide.")
    node_tags: ClassVar[Optional[List[str]]] = ["jev", "classifier", "triage", "judge", "silk"]
    node_icon: ClassVar[Optional[str]] = "a-b"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 5
    node_version = 1

    @classmethod
    def migrate_state(cls, state: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """api 1 → 2: the decision_model port and the use_systemone box are new.

        Both are additive: an old graph has no wire on the port (unwired
        reads as no decision pool) and no saved value for the box (which
        defaults to True, i.e. "use a wired pool when one appears").
        api 2 → 3: Jev left the node for its own loader. The obsolete
        ``jev_model``/``key_name`` widget values are dropped -- a custom
        credential name must be re-entered on the Jev Loader (the default
        JEV_API_KEY keeps working). Nothing else is remapped.
        api 3 → 4: the decision input is the ``classifier_model`` port
        type (first) and the judge stays a plain ``model_handle``
        (second). Wires off the old ``decision_model`` port need
        re-drawing onto ``classifier_model``; widgets are untouched.
        api 4 → 5: the "Fall back to …" boxes are gone. Wired =
        admitted: a wired handle answers, an unwired port is skipped.
        Old graphs saved both boxes checked (the only behavior that
        ever shipped as default), so nothing is remapped -- the
        obsolete widget values are dropped.
        """
        if from_version < 5:
            values = state.get("widget_data")
            if isinstance(values, dict):
                values.pop("use_local", None)
                values.pop("use_systemone", None)
        if from_version < 3:
            values = state.get("widget_data")
            if isinstance(values, dict):
                values.pop("jev_model", None)
                values.pop("key_name", None)
        return state

    def __init__(self, title: str = "Silk Classifier", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        # First the classifier model (typed decision backend), then the
        # judge (a plain chat handle): the port types say which is which,
        # so a chat model can never land where a decision backend belongs.
        self.add_input("classifier_model", datatype="classifier_model")
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
        self._widget_core.register_widget(
            "preset", self._preset_bar, role=PortRole.INTERNAL,
            datatype="str", default="", add_to_layout=False,
        )

        # ── Widgets ──
        self.combo_mode = QComboBox()
        for label, value in MODES:
            self.combo_mode.addItem(label, userData=value)
        self.combo_mode.setCurrentIndex(1)
        self.combo_mode.setToolTip(
            "off: the classifier is never consulted, every approval goes to a human.\n"
            "shadow: answers are scored and recorded but never skip a human "
            "(safe default; gathers calibration data).\n"
            "live: may skip triage-level questions, but only when the answering "
            "model's calibrated bands all pass.")
        form.addRow("Mode:", self.combo_mode)
        self._widget_core.register_widget(
            "mode", self.combo_mode, role=PortRole.INTERNAL, datatype="str",
            default="shadow", add_to_layout=False)

        self.chk_egress = QCheckBox("Allow state off-machine to hosted classifiers (redacted)")
        self.chk_egress.setToolTip(
            "Checked: wired hosted links (Jev, hosted systemone) may send redacted "
            "state off this machine -- redaction is best-effort, residual risk accepted. "
            "Unchecked: the classifier is air-gapped; hosted links are skipped and "
            "the human decides. Local links (judge, wired Laya pool) always stay on "
            "this machine. This switch covers the classifier only, not the rest "
            "of the graph.")
        form.addRow(self.chk_egress)
        self._widget_core.register_widget(
            "egress_allowed", self.chk_egress, role=PortRole.INTERNAL, datatype="bool",
            default=False, add_to_layout=False)

        self.spin_budget = QSpinBox()
        self.spin_budget.setRange(0, 1000)
        self.spin_budget.setValue(100)
        self.spin_budget.setToolTip(
            "How many times the LLM-callable classify tool may ask per run "
            "(purpose 'tool'); 0 disables the tool. The approval gate ('gate') "
            "and hooks are never budgeted.")
        form.addRow("Classify calls / run:", self.spin_budget)
        self._widget_core.register_widget(
            "tool_calls_per_run", self.spin_budget, role=PortRole.INTERNAL, datatype="int",
            default=100, add_to_layout=False)

        self.bands_table = BandsTableWidget()
        self.bands_table.setToolTip(
            "Per-model calibration: a model may only skip the human when every band "
            "passes (reversible, in-scope, consistent, risk, intent confidence). "
            "A model with no row here never skips -- it always asks.")
        form.addRow("Calibrated bands:", self.bands_table)
        self._widget_core.register_widget(
            "bands", self.bands_table, role=PortRole.INTERNAL, datatype="dict",
            default={}, add_to_layout=False)

        self.set_content_widget(self._widget_core)

        self._handle: Optional[ClassifierHandle] = None
        self._handle_key: Optional[str] = None
        self._pending_known: List[str] = []

    # ── Main thread ───────────────────────────────────────────────────
    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        known, self._pending_known = self._pending_known, []
        self.bands_table.set_known_models(known)

    # ── Presets ─────────────────────────────────────────────────────

    def _collect_preset(self) -> dict:
        get = self._widget_core.get_port_value
        spec = spec_from_inputs({k: get(k) for k in (
            "mode", "egress_allowed", "tool_calls_per_run", "bands")})
        return {"spec": spec.model_dump(mode="json")}

    def _apply_preset(self, preset: ClassifierPreset) -> None:
        spec = ClassifierSpec.model_validate(preset.spec or {})
        apply = self._widget_core.apply_port_value
        apply("mode", spec.mode)
        apply("egress_allowed", spec.egress_allowed)
        apply("tool_calls_per_run", spec.tool_calls_per_run)
        apply("bands", {m: b.model_dump() for m, b in spec.bands.items()})

    # ── Compute ─────────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        judge = inputs.get("judge_model")
        judge = judge if isinstance(judge, dict) and judge.get("backend") == "gguf" else None
        decision = inputs.get("classifier_model")
        decision = decision if isinstance(decision, dict) and decision else None
        try:
            member = decision_member(decision) if decision is not None else None
        except ValueError as exc:
            log.warning(f"Classifier model unusable: {exc}")
            member = None
        spec = spec_from_inputs(dict(inputs,
            judge_model_id=(judge_model_id(judge)
                            if judge is not None else LOCAL_JUDGE_MODEL),
            decision_model_id=(member[0] if member is not None else LOCAL_DECISION_MODEL)))
        if member is not None:
            # The wired handle is the entry: pool-backed or hosted, the
            # URL/key/egress travel in the entry, never as form values.
            _, entry, member_kwargs = member
            spec.backends = [entry if b.kind == "systemone" else b
                             for b in spec.backends]
        else:
            member_kwargs = {}
        key = (spec.model_dump_json()
               + f"|{id(judge.get('pool')) if judge else 0}"
               + f"|{id(decision.get('pool')) if decision is not None else 0}")
        if self._handle is None or key != self._handle_key:
            scorer = None
            if judge is not None and spec.mode != "off":
                try:
                    scorer = pool_scorer(judge_client(judge))
                except ValueError as exc:
                    log.warning(f"Judge model unusable: {exc}")
            systemone = member_kwargs.get("systemone") \
                if spec.mode != "off" else None
            chain = build_classifier(spec, secrets=resolve_credential,
                                     local_scorer=scorer, systemone=systemone)
            self._handle = ClassifierHandle(spec, chain, judge)
            self._handle_key = key
        known = ([member[0]] if member is not None else [])
        if judge is not None:
            known.append(judge_model_id(judge))
        self._pending_known = known
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
