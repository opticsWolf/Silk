# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Classifier Ensemble -- several classifier models wired in, one verdict out.

Up to three classifier models (a Classifier Loader's pool or a Jev
Loader's hosted endpoint) plus an optional judge fallback answer as one,
by policy:

* **first** -- the first member that answers wins (backups, not debate).
* **consensus** -- everybody answers (in threads) and the answers are
  combined with equal weights.
* **cascade** -- members are asked in input order and the walk stops
  early when every answer's confidence clears the agreement threshold;
  otherwise it continues and combines.

The judge is the fallback, never a voter: it is asked only when no
decision member answered (mirroring the Classifier node, and sparing
the GPU-heavy logprob run whenever the cheap encoders sufficed).

Early stops and single answers keep the member's verdict whole -- member
model id, member bands. Only a real combination mints the joined id
(``model-a+model-b``, sorted), which needs its own bands row on *this*
node. An ensemble id with no row never skips, exactly like any other
unmeasured model. With nobody wired the verdict is an error and the
human decides, as always.

Downstream changes nothing: the output is a ClassifierHandle (an
EnsembleClassifier behind ``for_run`` + ``spec``), so the gate, hooks
and tool take it as-is -- including the self-grading refusal, which
reads the carried judge.
"""
from itertools import combinations
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QSpinBox,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import CLASSIFIER_MODEL_TYPE  # noqa: F401
from .silk_ports import MODEL_HANDLE_TYPE  # noqa: F401
from .silk_ports import SILK_CLASSIFIER_TYPE  # noqa: F401
from ..functions.classifier import (
    BackendConfig, BandsConfig, ClassifierSpec, EnsembleHandle, EnsembleSpec,
    build_classifier, build_ensemble, decision_member,
)
from ..functions.classifier_local import (
    judge_client, judge_model_id, pool_scorer,
)
from ..functions.credentials import resolve_credential
from ..functions.presets import EnsemblePreset, PresetStore
from ..widgets.bands_table import BandsTableWidget
from ..widgets.preset_bar import PresetBarWidget

log = get_logger("SilkEnsembleNode")

MODES = (
    ("Off - you decide everything", "off"),
    ("Shadow - score and record, never skip", "shadow"),
    ("Live - may skip triage-level questions", "live"),
)

POLICIES = (
    ("First answer wins (backups)", "first"),
    ("Consensus - combine everybody", "consensus"),
    ("Cascade - stop when confident", "cascade"),
)

MEMBER_PORTS = (("a", "classifier_a"), ("b", "classifier_b"), ("c", "classifier_c"))


def spec_from_inputs(inputs: Dict[str, Any]) -> EnsembleSpec:
    """The node's form as an :class:`EnsembleSpec`. Qt-free, tested."""
    bands = {}
    for model, raw in dict(inputs.get("bands") or {}).items():
        try:
            bands[str(model)] = BandsConfig.model_validate(raw)
        except Exception:                                  # the table marks it; skip here
            log.warning(f"Ignoring invalid bands for {model!r}")
    return EnsembleSpec(
        policy=str(inputs.get("policy") or "cascade"),
        agreement=float(inputs.get("agreement", 0.7)),
        mode=str(inputs.get("mode") or "shadow"),
        bands=bands,
        tool_calls_per_run=int(inputs.get("tool_calls_per_run") or 100),
    )


@register_node
class SilkEnsembleNode(ActiveNode):
    """Up to three classifiers as one, by policy."""
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Classifier Ensemble"
    node_description: ClassVar[Optional[str]] = (
        "Asks up to three wired classifier models as one verdict -- first "
        "answer wins, consensus of everybody, or cascade that stops when "
        "confident -- with a judge fallback when none answered. Combined "
        "answers need their own bands row to skip.")
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "classifier", "ensemble", "triage", "consensus", "cascade",
    ]
    node_icon: ClassVar[Optional[str]] = "a-b"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 2
    node_version = 1

    @classmethod
    def migrate_state(cls, state: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """api 1 → 2: members are classifier models, not classifiers.

        The ``classifier_a/b/c`` ports changed type from ``silk_classifier``
        to ``classifier_model`` and the ``judge_model`` fallback port is new.
        Wires off the old ports need re-drawing; widgets are untouched.
        api 2 → 3: the "Classify calls / run" budget widget is new.
        Additive: old graphs get the 100 default, nothing is remapped.
        """
        return state

    def __init__(self, title: str = "Classifier Ensemble", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        # Three classifier models (typed decision backends), then the
        # judge fallback (a plain chat handle, asked only when no
        # decision member answered).
        for _, port in MEMBER_PORTS:
            self.add_input(port, datatype="classifier_model")
        self.add_input("judge_model", datatype="model_handle")
        self.add_output("classifier", datatype="silk_classifier")
        self.add_output("stats", datatype="dict")

        # ── Layout & WidgetCore ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        # ── Presets ──
        self._preset_store: PresetStore[EnsemblePreset] = PresetStore(
            "silk_ensembles", EnsemblePreset)
        self._preset_bar = PresetBarWidget(
            self._preset_store, collect=self._collect_preset, apply=self._apply_preset)
        form.addRow("Preset:", self._preset_bar)
        self._widget_core.register_widget(
            "preset", self._preset_bar, role=PortRole.INTERNAL,
            datatype="str", default="", add_to_layout=False,
        )

        self.combo_policy = QComboBox()
        for label, value in POLICIES:
            self.combo_policy.addItem(label, userData=value)
        self.combo_policy.setCurrentIndex(2)
        self.combo_policy.setToolTip(
            "first: the first member that answers wins (backups). "
            "consensus: everybody answers and the answers combine with "
            "equal weights. cascade: ask in input order, stop early when "
            "every answer's confidence clears the agreement threshold, "
            "else combine.")
        form.addRow("Policy:", self.combo_policy)
        self._widget_core.register_widget(
            "policy", self.combo_policy, role=PortRole.INPUT,
            datatype="string", default="cascade", add_to_layout=False,
        )
        self.add_input("policy", datatype="string")

        self.spin_agreement = QDoubleSpinBox()
        self.spin_agreement.setRange(0.0, 1.0)
        self.spin_agreement.setSingleStep(0.05)
        self.spin_agreement.setValue(0.7)
        self.spin_agreement.setToolTip(
            "Cascade stops early when every answer's confidence is at or "
            "above this. Higher asks more members more often; lower trusts "
            "the first confident answer.")
        form.addRow("Agreement:", self.spin_agreement)
        self._widget_core.register_widget(
            "agreement", self.spin_agreement, role=PortRole.INPUT,
            datatype="float", default=0.7, add_to_layout=False,
        )
        self.add_input("agreement", datatype="float")

        self.combo_mode = QComboBox()
        for label, value in MODES:
            self.combo_mode.addItem(label, userData=value)
        self.combo_mode.setCurrentIndex(1)
        self.combo_mode.setToolTip(
            "off: the ensemble is never consulted, every approval goes "
            "to a human. shadow: verdicts are scored and recorded but "
            "never skip a human (gathers the data the joined bands rows "
            "are fitted from). live: may skip triage-level questions, but "
            "only when a bands row for the answering id -- member or "
            "joined -- passes.")
        form.addRow("Mode:", self.combo_mode)
        self._widget_core.register_widget(
            "mode", self.combo_mode, role=PortRole.INPUT,
            datatype="string", default="shadow", add_to_layout=False,
        )
        self.add_input("mode", datatype="string")

        self.spin_budget = QSpinBox()
        self.spin_budget.setRange(0, 1000)
        self.spin_budget.setValue(100)
        self.spin_budget.setToolTip(
            "How many times the LLM-callable classify tool may ask per run "
            "(purpose 'tool'); 0 disables the tool. The approval gate ('gate') "
            "and hooks are never budgeted. Shared with delegated workers: "
            "one run, one budget.")
        form.addRow("Classify calls / run:", self.spin_budget)
        self._widget_core.register_widget(
            "tool_calls_per_run", self.spin_budget, role=PortRole.INPUT,
            datatype="int", default=100, add_to_layout=False,
        )
        self.add_input("tool_calls_per_run", datatype="int")

        self.bands_table = BandsTableWidget()
        self.bands_table.setToolTip(
            "Per-model calibration for the ids that answer here: member "
            "ids for early stops, joined ids (model-a+model-b, sorted) "
            "for combinations. An id with no row here never skips.")
        form.addRow("Bands:", self.bands_table)
        self._widget_core.register_widget(
            "bands", self.bands_table, role=PortRole.INTERNAL,
            datatype="dict", default={}, add_to_layout=False,
        )

        self._label_status = QLabel("No members wired.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        #: The last handle handed downstream, kept so an unrelated graph
        #: edit does not rebuild the ensemble pointlessly.
        self._handle: Optional[EnsembleHandle] = None
        self._handle_key: str = ""
        self._sync_status: str = "No members wired."
        self._pending_known: List[str] = []

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── Presets ─────────────────────────────────────────────────────

    def _collect_preset(self) -> dict:
        get = self._widget_core.get_port_value
        spec = spec_from_inputs({k: get(k) for k in (
            "policy", "agreement", "mode", "bands",
            "tool_calls_per_run")})
        return {"spec": spec.model_dump(mode="json")}

    def _apply_preset(self, preset: EnsemblePreset) -> None:
        spec = EnsembleSpec.model_validate(preset.spec or {})
        apply = self._widget_core.apply_port_value
        apply("policy", spec.policy)
        apply("agreement", spec.agreement)
        apply("mode", spec.mode)
        apply("tool_calls_per_run", spec.tool_calls_per_run)
        apply("bands", {m: b.model_dump() for m, b in spec.bands.items()})

    # ── Main thread ───────────────────────────────────────────────────

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        known, self._pending_known = self._pending_known, []
        self.bands_table.set_known_models(known)
        self._widget_core.push_display("status", self._sync_status)

    def cleanup(self) -> None:
        self._preset_bar.cleanup()
        super().cleanup()

    # ── Worker thread ─────────────────────────────────────────────────

    @staticmethod
    def _decision_chain(handle: Any) -> Optional[tuple[str, Any]]:
        """A wired classifier model as (bands id, one-link chain)."""
        try:
            member = decision_member(handle)
        except ValueError as exc:
            log.warning(f"Ensemble member unusable: {exc}")
            return None
        if member is None:
            return None
        model_id, entry, kwargs = member
        sub = ClassifierSpec(mode="live", backends=[entry])
        chain = build_classifier(sub, secrets=resolve_credential, **kwargs)
        return model_id, chain

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            spec = spec_from_inputs({k: inputs.get(k) for k in (
                "policy", "agreement", "mode", "bands",
                "tool_calls_per_run")})
        except ValueError as exc:
            self._handle = None
            self._handle_key = ""
            self._sync_status = f"Invalid input: {exc}."
            return {"classifier": None}
        members = []
        member_ids = []
        for label, port in MEMBER_PORTS:
            built = self._decision_chain(inputs.get(port))
            if built is not None:
                member_ids.append(built[0])
                members.append((label, built[1]))
        judge = inputs.get("judge_model")
        judge = judge if isinstance(judge, dict) and judge.get("backend") == "gguf" \
            else None
        fallback = None
        if judge is not None and spec.mode != "off":
            try:
                scorer = pool_scorer(judge_client(judge))
                judge_id = judge_model_id(judge)
                sub = ClassifierSpec(
                    mode="live",
                    backends=[BackendConfig(kind="local", model=judge_id)])
                fallback = build_classifier(sub, local_scorer=scorer)
            except ValueError as exc:
                log.warning(f"Ensemble judge unusable: {exc}")
        key = (spec.model_dump_json()
               + f"|{','.join(f'{label}={id(chain)}' for label, chain in members)}"
               + f"|judge={id(fallback) if fallback is not None else 0}")
        # What can answer here: member ids, their sorted joins (only real
        # combinations mint joined ids), and the judge fallback's own id.
        ids = sorted(set(member_ids))
        known = ids + ["+".join(c) for r in range(2, len(ids) + 1)
                       for c in combinations(ids, r)]
        if fallback is not None:
            known.append(judge_model_id(judge))
        self._pending_known = known
        if self._handle is None or key != self._handle_key:
            ensemble = build_ensemble(spec, members, fallback=fallback)
            self._handle = EnsembleHandle(
                spec=spec, classifier=ensemble,
                judge_model=judge,
                members=tuple(label for label, _ in members))
            self._handle_key = key
        labels = ", ".join(label for label, _ in members) or "nobody wired"
        if fallback is not None:
            labels += " + judge fallback"
        self._sync_status = (
            f"Ensemble ({spec.policy}, agreement {spec.agreement}): "
            f"{labels}  ·  {len(spec.bands)} calibrated id(s)")
        meter = self._handle.classifier.meter
        return {
            "classifier": self._handle,
            "stats": {
                "mode": spec.mode,
                "policy": spec.policy,
                "members": [label for label, _ in members],
                "backends": [b.name for b in self._handle.classifier.backends],
                "calibrated": sorted(spec.bands),
                "calls": meter.calls, "failures": meter.failures,
                "cost_usd": round(meter.cost_usd, 6),
                "by_purpose": dict(meter.by_purpose),
            },
        }
