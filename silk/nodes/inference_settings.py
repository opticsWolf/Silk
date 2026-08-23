# -*- coding: utf-8 -*-
"""Agent Inference Settings Node.

Constructs a ``gen_params`` dictionary from UI controls and emits it on an
output port.  Only keys that match ``GraphEngine._GEN_PARAM_KEYS`` are
included, so the dict flows directly through ``stream_response`` without
silent drops.

Checkbox-gated parameters are only included when the box is checked; an
unchecked box means "use llama.cpp's own default" (key absent from dict).

Configurations can be stored as named presets (JSON on disk,
pydantic-validated on load) and recalled from the dropdown.
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node

from ..functions.presets import PresetStore, InferenceSettingsPreset
from ..widgets.preset_bar import PresetBarWidget

# Keys that GraphEngine.stream_response forwards to create_chat_completion.
# Kept in sync with weave.plugins.silk.functions.graph_engine._GEN_PARAM_KEYS.
_VALID_GEN_KEYS = frozenset((
    "max_tokens", "temperature", "top_p", "top_k", "min_p",
    "repeat_penalty", "presence_penalty", "frequency_penalty",
    "seed", "stop",
))


@register_node
class AgentInferenceSettingsNode(ActiveNode):
    """Generates a gen_params dictionary for Agent nodes."""

    node_class: ClassVar[str] = "AI"
    node_subclass: ClassVar[str] = "Configuration"
    node_name: ClassVar[Optional[str]] = "Inference Settings"
    node_description: ClassVar[Optional[str]] = (
        "Configures sampling and generation parameters for an agent. "
        "Save and recall configurations as named presets."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "inference", "sampling", "settings", "gen_params",
    ]
    node_icon: ClassVar[Optional[str]] = "brackets-contain"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT

    def __init__(self, title: str = "Inference Settings", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        self.add_output("gen_params", datatype="dict")

        # ── Layout ──
        self._widget_core = WidgetCore()
        self._widget_core.set_node(self)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)

        # ── Presets ──
        self._preset_store: PresetStore[InferenceSettingsPreset] = PresetStore(
            "inference_settings", InferenceSettingsPreset,
        )
        self._preset_bar = PresetBarWidget(
            self._preset_store,
            collect=self._collect_preset,
            apply=self._apply_preset,
        )
        preset_row = QWidget()
        preset_row_layout = QVBoxLayout(preset_row)
        preset_row_layout.setContentsMargins(0, 0, 0, 0)
        preset_row_layout.addWidget(self._preset_bar)
        main_layout.addWidget(preset_row)

        # ── Settings Group ──
        grp_settings = QGroupBox("Settings")
        form_settings = QFormLayout(grp_settings)

        # Temperature (always included)
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(0.0, 2.0)
        self.spin_temp.setSingleStep(0.05)
        self.spin_temp.setValue(0.7)
        form_settings.addRow("Temperature:", self.spin_temp)
        self._widget_core.register_widget(
            "temperature", self.spin_temp,
            role=PortRole.INTERNAL, datatype="float", default=0.7,
            add_to_layout=False,
        )

        # Max Tokens (checkbox-gated, default off = no limit)
        self.chk_max_tokens = QCheckBox("Limit Response Length")
        self.chk_max_tokens.setChecked(False)
        self.spin_max_tokens = QSpinBox()
        self.spin_max_tokens.setRange(1, 131072)
        self.spin_max_tokens.setValue(1024)
        self.spin_max_tokens.setEnabled(False)
        self.chk_max_tokens.toggled.connect(self.spin_max_tokens.setEnabled)
        form_settings.addRow(self.chk_max_tokens, self.spin_max_tokens)
        self._widget_core.register_widget(
            "use_max_tokens", self.chk_max_tokens,
            role=PortRole.INTERNAL, datatype="bool", default=False,
            add_to_layout=False,
        )
        self._widget_core.register_widget(
            "max_tokens", self.spin_max_tokens,
            role=PortRole.INTERNAL, datatype="int", default=1024,
            add_to_layout=False,
        )

        # Stop Strings
        self.line_stop = QLineEdit()
        self.line_stop.setPlaceholderText("Enter comma-separated stop strings…")
        form_settings.addRow("Stop Strings:", self.line_stop)
        self._widget_core.register_widget(
            "stop_strings", self.line_stop,
            role=PortRole.INTERNAL, datatype="str", default="",
            add_to_layout=False, policy=debounced(300),
        )

        main_layout.addWidget(grp_settings)

        # ── Sampling Group ──
        grp_sampling = QGroupBox("Sampling")
        form_sampling = QFormLayout(grp_sampling)

        # Top K (always included)
        self.spin_top_k = QSpinBox()
        self.spin_top_k.setRange(0, 100)
        self.spin_top_k.setValue(40)
        form_sampling.addRow("Top K:", self.spin_top_k)
        self._widget_core.register_widget(
            "top_k", self.spin_top_k,
            role=PortRole.INTERNAL, datatype="int", default=40,
            add_to_layout=False,
        )

        # Top P (checkbox-gated)
        self.chk_top_p = QCheckBox("Top P")
        self.chk_top_p.setChecked(True)
        self.spin_top_p = QDoubleSpinBox()
        self.spin_top_p.setRange(0.0, 1.0)
        self.spin_top_p.setSingleStep(0.01)
        self.spin_top_p.setValue(0.95)
        self.chk_top_p.toggled.connect(self.spin_top_p.setEnabled)
        form_sampling.addRow(self.chk_top_p, self.spin_top_p)
        self._widget_core.register_widget(
            "use_top_p", self.chk_top_p,
            role=PortRole.INTERNAL, datatype="bool", default=True,
            add_to_layout=False,
        )
        self._widget_core.register_widget(
            "top_p", self.spin_top_p,
            role=PortRole.INTERNAL, datatype="float", default=0.95,
            add_to_layout=False,
        )

        # Min P (checkbox-gated)
        self.chk_min_p = QCheckBox("Min P")
        self.chk_min_p.setChecked(False)
        self.spin_min_p = QDoubleSpinBox()
        self.spin_min_p.setRange(0.0, 1.0)
        self.spin_min_p.setSingleStep(0.01)
        self.spin_min_p.setValue(0.0)
        self.spin_min_p.setEnabled(False)
        self.chk_min_p.toggled.connect(self.spin_min_p.setEnabled)
        form_sampling.addRow(self.chk_min_p, self.spin_min_p)
        self._widget_core.register_widget(
            "use_min_p", self.chk_min_p,
            role=PortRole.INTERNAL, datatype="bool", default=False,
            add_to_layout=False,
        )
        self._widget_core.register_widget(
            "min_p", self.spin_min_p,
            role=PortRole.INTERNAL, datatype="float", default=0.0,
            add_to_layout=False,
        )

        # Repeat Penalty (checkbox-gated)
        self.chk_repeat = QCheckBox("Repeat Penalty")
        self.chk_repeat.setChecked(False)
        self.spin_repeat = QDoubleSpinBox()
        self.spin_repeat.setRange(0.0, 2.0)
        self.spin_repeat.setSingleStep(0.01)
        self.spin_repeat.setValue(1.0)
        self.spin_repeat.setEnabled(False)
        self.chk_repeat.toggled.connect(self.spin_repeat.setEnabled)
        form_sampling.addRow(self.chk_repeat, self.spin_repeat)
        self._widget_core.register_widget(
            "use_repeat_penalty", self.chk_repeat,
            role=PortRole.INTERNAL, datatype="bool", default=False,
            add_to_layout=False,
        )
        self._widget_core.register_widget(
            "repeat_penalty", self.spin_repeat,
            role=PortRole.INTERNAL, datatype="float", default=1.0,
            add_to_layout=False,
        )

        # Presence Penalty (checkbox-gated)
        self.chk_presence = QCheckBox("Presence Penalty")
        self.chk_presence.setChecked(False)
        self.spin_presence = QDoubleSpinBox()
        self.spin_presence.setRange(-2.0, 2.0)
        self.spin_presence.setSingleStep(0.05)
        self.spin_presence.setValue(0.0)
        self.spin_presence.setEnabled(False)
        self.chk_presence.toggled.connect(self.spin_presence.setEnabled)
        form_sampling.addRow(self.chk_presence, self.spin_presence)
        self._widget_core.register_widget(
            "use_presence_penalty", self.chk_presence,
            role=PortRole.INTERNAL, datatype="bool", default=False,
            add_to_layout=False,
        )
        self._widget_core.register_widget(
            "presence_penalty", self.spin_presence,
            role=PortRole.INTERNAL, datatype="float", default=0.0,
            add_to_layout=False,
        )

        # Frequency Penalty (checkbox-gated)
        self.chk_freq = QCheckBox("Frequency Penalty")
        self.chk_freq.setChecked(False)
        self.spin_freq = QDoubleSpinBox()
        self.spin_freq.setRange(-2.0, 2.0)
        self.spin_freq.setSingleStep(0.05)
        self.spin_freq.setValue(0.0)
        self.spin_freq.setEnabled(False)
        self.chk_freq.toggled.connect(self.spin_freq.setEnabled)
        form_sampling.addRow(self.chk_freq, self.spin_freq)
        self._widget_core.register_widget(
            "use_frequency_penalty", self.chk_freq,
            role=PortRole.INTERNAL, datatype="bool", default=False,
            add_to_layout=False,
        )
        self._widget_core.register_widget(
            "frequency_penalty", self.spin_freq,
            role=PortRole.INTERNAL, datatype="float", default=0.0,
            add_to_layout=False,
        )

        main_layout.addWidget(grp_sampling)

        container = QWidget()
        container.setLayout(main_layout)
        self.set_content_widget(container)

        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── Presets (main thread) ────────────────────────────────────────

    def _collect_preset(self) -> dict:
        return {
            "temperature": float(self._widget_core.get_port_value("temperature") or 0.7),
            "use_max_tokens": bool(self._widget_core.get_port_value("use_max_tokens")),
            "max_tokens": int(self._widget_core.get_port_value("max_tokens") or 1024),
            "stop_strings": str(self._widget_core.get_port_value("stop_strings") or ""),
            "top_k": int(self._widget_core.get_port_value("top_k") or 40),
            "use_top_p": bool(self._widget_core.get_port_value("use_top_p")),
            "top_p": float(self._widget_core.get_port_value("top_p") or 0.95),
            "use_min_p": bool(self._widget_core.get_port_value("use_min_p")),
            "min_p": float(self._widget_core.get_port_value("min_p") or 0.0),
            "use_repeat_penalty": bool(self._widget_core.get_port_value("use_repeat_penalty")),
            "repeat_penalty": float(self._widget_core.get_port_value("repeat_penalty") or 1.0),
            "use_presence_penalty": bool(self._widget_core.get_port_value("use_presence_penalty")),
            "presence_penalty": float(self._widget_core.get_port_value("presence_penalty") or 0.0),
            "use_frequency_penalty": bool(self._widget_core.get_port_value("use_frequency_penalty")),
            "frequency_penalty": float(self._widget_core.get_port_value("frequency_penalty") or 0.0),
        }

    def _apply_preset(self, preset: InferenceSettingsPreset) -> None:
        self._widget_core.apply_port_value("temperature", preset.temperature)
        self._widget_core.apply_port_value("use_max_tokens", preset.use_max_tokens)
        self._widget_core.apply_port_value("max_tokens", preset.max_tokens)
        self._widget_core.apply_port_value("stop_strings", preset.stop_strings)
        self._widget_core.apply_port_value("top_k", preset.top_k)
        self._widget_core.apply_port_value("use_top_p", preset.use_top_p)
        self._widget_core.apply_port_value("top_p", preset.top_p)
        self._widget_core.apply_port_value("use_min_p", preset.use_min_p)
        self._widget_core.apply_port_value("min_p", preset.min_p)
        self._widget_core.apply_port_value("use_repeat_penalty", preset.use_repeat_penalty)
        self._widget_core.apply_port_value("repeat_penalty", preset.repeat_penalty)
        self._widget_core.apply_port_value("use_presence_penalty", preset.use_presence_penalty)
        self._widget_core.apply_port_value("presence_penalty", preset.presence_penalty)
        self._widget_core.apply_port_value("use_frequency_penalty", preset.use_frequency_penalty)
        self._widget_core.apply_port_value("frequency_penalty", preset.frequency_penalty)

    # ── Cleanup ──────────────────────────────────────────────────────

    def cleanup(self) -> None:
        self._preset_bar.cleanup()
        super().cleanup()

    # ── Compute ──────────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Build a gen_params dict — only keys in _VALID_GEN_KEYS."""
        params: Dict[str, Any] = {}

        # Always included
        params["temperature"] = inputs.get("temperature", 0.7)
        params["top_k"] = inputs.get("top_k", 40)

        # Checkbox-gated: only include if the toggle is active
        if inputs.get("use_max_tokens", False):
            params["max_tokens"] = inputs.get("max_tokens", 1024)

        if inputs.get("use_top_p", True):
            params["top_p"] = inputs.get("top_p", 0.95)

        if inputs.get("use_min_p", False):
            params["min_p"] = inputs.get("min_p", 0.0)

        if inputs.get("use_repeat_penalty", False):
            params["repeat_penalty"] = inputs.get("repeat_penalty", 1.0)

        if inputs.get("use_presence_penalty", False):
            params["presence_penalty"] = inputs.get("presence_penalty", 0.0)

        if inputs.get("use_frequency_penalty", False):
            params["frequency_penalty"] = inputs.get("frequency_penalty", 0.0)

        # Stop strings: comma-separated input → list
        raw_stop = inputs.get("stop_strings", "").strip()
        if raw_stop:
            params["stop"] = [s.strip() for s in raw_stop.split(",") if s.strip()]

        return {"gen_params": params}
