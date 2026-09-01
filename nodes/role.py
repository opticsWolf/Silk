# -*- coding: utf-8 -*-
"""Silk Role Node.

Defines a declarative :class:`Role` — persona instructions plus a hard
tool-selection rule — and outputs it on a ``silk_role`` port. The node
takes a ``silk_toolset`` input purely as a *catalog*: its tools populate
a checkbox tree for further downselection (a role can only narrow, never
widen, what the toolset grants — enforcement happens at dispatch inside
the ToolBox, where a denied call is rejected with ``role_denied``).

Role configurations can be stored as named presets (JSON on disk,
pydantic-validated on load) and recalled from the dropdown.
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLineEdit, QSpinBox

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from weave.widgets.markdown_widget import MarkdownWidget

from .silk_ports import SILK_ROLE_TYPE, SILK_TOOLSET_TYPE  # noqa: F401
from ..functions.hook_catalog import build_hooks
from ..functions.presets import PresetStore, RolePreset
from ..functions.role import Role, ToolSelector
from ..functions.toolset_build import tool_catalog
from ..widgets.hook_select import HookSelectWidget
from ..widgets.preset_bar import PresetBarWidget
from ..widgets.tool_tree import ToolTreeWidget

log = get_logger("SilkRole")


@register_node
class SilkRoleNode(ActiveNode):
    """Declarative agent role: persona + hard tool downselection of a toolset."""

    node_class: ClassVar[str] = "AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk Role"
    node_description: ClassVar[Optional[str]] = (
        "Persona instructions and a hard-enforced tool selection, "
        "downselected from the connected toolset."
    )
    node_tags: ClassVar[Optional[List[str]]] = ["silk", "agent", "role", "persona", "llm"]
    node_icon: ClassVar[Optional[str]] = "text-recognition"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1   # owns a hand-written state dict

    def __init__(self, title: str = "Silk Role", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        self.add_input("toolset", datatype="silk_toolset")
        self.add_input("instructions", datatype="string")
        self.add_output("role", datatype="silk_role")

        # ── Layout & WidgetCore ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        # ── Presets ──
        self._preset_store: PresetStore[RolePreset] = PresetStore(
            "silk_roles", RolePreset
        )
        self._preset_bar = PresetBarWidget(
            self._preset_store,
            collect=self._collect_preset,
            apply=self._apply_preset,
        )
        form.addRow("Preset:", self._preset_bar)

        # ── Widgets ──
        self._edit_id = QLineEdit()
        self._edit_id.setPlaceholderText("e.g. researcher")
        form.addRow("Role ID:", self._edit_id)
        self._widget_core.register_widget(
            "role_id", self._edit_id, role=PortRole.INTERNAL,
            datatype="string", default="role", policy=debounced(300),
            add_to_layout=False,
        )

        self._edit_instructions = MarkdownWidget(mode="editor")
        self._edit_instructions._text_edit.setPlaceholderText(
            "Persona / instructions injected as a [ROLE: …] system prompt block…"
        )
        self._edit_instructions._text_edit.setMaximumHeight(100)
        form.addRow("Instructions:", self._edit_instructions)
        self._widget_core.register_widget(
            "instructions", self._edit_instructions, role=PortRole.BIDIRECTIONAL,
            datatype="string", default="", add_to_layout=False,
        )

        self.chk_allow_all = QCheckBox()
        self.chk_allow_all.setToolTip(
            "Grant every tool of the connected toolset (skip downselection)."
        )
        form.addRow("Allow All Tools:", self.chk_allow_all)
        self._widget_core.register_widget(
            "allow_all", self.chk_allow_all, role=PortRole.INTERNAL,
            datatype="bool", default=False, add_to_layout=False,
        )

        # Tool downselection tree, populated from the connected toolset.
        self._tool_tree = ToolTreeWidget(checkable=True)
        form.addRow(self._tool_tree)
        self._widget_core.register_widget(
            "checked_tools", self._tool_tree, role=PortRole.INTERNAL,
            datatype="list", default=[], add_to_layout=False,
        )

        self._combo_risk = QComboBox()
        self._combo_risk.addItem("(no ceiling)", userData="")
        self._combo_risk.addItem("low", userData="low")
        self._combo_risk.addItem("medium", userData="medium")
        self._combo_risk.addItem("high", userData="high")
        form.addRow("Max Risk:", self._combo_risk)
        self._widget_core.register_widget(
            "max_risk", self._combo_risk, role=PortRole.INTERNAL,
            datatype="str", default="", add_to_layout=False,
        )

        self.spin_max_rounds = QSpinBox()
        self.spin_max_rounds.setRange(1, 64)
        self.spin_max_rounds.setValue(16)
        form.addRow("Max Rounds:", self.spin_max_rounds)
        self._widget_core.register_widget(
            "max_rounds", self.spin_max_rounds, role=PortRole.INTERNAL,
            datatype="int", default=16, add_to_layout=False,
        )

        # Behavioral hooks: installed by the RoleBinding on activation,
        # removed on deactivation — scoped to this role's runs only.
        # Value shape: {"names": [...], "configs": {name: {...}}}.
        self._hook_select = HookSelectWidget()
        form.addRow("Hooks:", self._hook_select)
        self._widget_core.register_widget(
            "hooks_config", self._hook_select, role=PortRole.INTERNAL,
            datatype="dict", default={}, add_to_layout=False,
        )

        # ── Visibility: the tree and risk ceiling are moot with Allow All
        #    on; show them only while it is off. ──
        self._widget_core.bind_visibility(
            trigger_port="allow_all",
            mapping={False: ["checked_tools", "max_risk"]},
        )

        # ── Mount ──
        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── Presets (main thread) ─────────────────────────────────────────

    def _collect_preset(self) -> dict:
        return {
            "role_id": str(self._widget_core.get_port_value("role_id") or "role"),
            "instructions": str(self._widget_core.get_port_value("instructions") or ""),
            "allow_all": bool(self._widget_core.get_port_value("allow_all")),
            "checked_tools": list(self._widget_core.get_port_value("checked_tools") or []),
            "max_risk": str(self._widget_core.get_port_value("max_risk") or ""),
            "max_rounds": int(self._widget_core.get_port_value("max_rounds") or 16),
            "hooks": list(
                (self._widget_core.get_port_value("hooks_config") or {}).get("names") or []
            ),
            "hook_configs": dict(
                (self._widget_core.get_port_value("hooks_config") or {}).get("configs") or {}
            ),
        }

    def _apply_preset(self, preset: RolePreset) -> None:
        self._widget_core.apply_port_value("role_id", preset.role_id)
        self._widget_core.apply_port_value("instructions", preset.instructions)
        self._widget_core.apply_port_value("allow_all", preset.allow_all)
        self._widget_core.apply_port_value("checked_tools", list(preset.checked_tools))
        self._widget_core.apply_port_value("max_risk", preset.max_risk)
        self._widget_core.apply_port_value("max_rounds", preset.max_rounds)
        self._widget_core.apply_port_value(
            "hooks_config",
            {"names": list(preset.hooks), "configs": dict(preset.hook_configs)},
        )

    # ── Worker thread ─────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        toolset = inputs.get("toolset")
        self._sync_catalog = tool_catalog(toolset) if toolset is not None else []
        # Infrastructure hooks already active on the connected toolset —
        # shown in the hook list so double-selection is a visible choice.
        self._sync_inherited_hooks = tuple(
            getattr(toolset, "catalog_hook_names", ())
        )

        role_id = str(inputs.get("role_id") or "role").strip() or "role"

        # Filter checked_tools by the currently available catalog. This
        # prevents the Role from allowing tools that were removed from
        # the upstream toolset, while the ToolTreeWidget still remembers
        # them in self._checked for preset transparency.
        available = {e["name"] for e in self._sync_catalog}
        checked = [
            str(n) for n in (inputs.get("checked_tools") or [])
            if str(n) in available
        ]

        selector = ToolSelector(
            allow_names=frozenset(checked),
            max_risk=(inputs.get("max_risk") or None),
            allow_all=bool(inputs.get("allow_all", False)),
        )
        role = Role(
            id=role_id,
            name=role_id,
            instructions=str(inputs.get("instructions") or ""),
            selector=selector,
            max_rounds=int(inputs.get("max_rounds") or 16),
            # Names/configs → callables happens here, at build time;
            # presets and node state only ever carry the data.
            hooks=build_hooks(
                (inputs.get("hooks_config") or {}).get("names") or [],
                (inputs.get("hooks_config") or {}).get("configs") or {},
            ),
        )
        return {"role": role}

    # ── State ───────────────────────────────────────────────────────

    def restore_state(self, state: Dict[str, Any]) -> None:
        # 1. Restore values silently (prevents eval storms & false undo history)
        with self._widget_core.suppress_signals():
            super().restore_state(state)

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        if hasattr(self, "_sync_catalog"):
            self._tool_tree.set_catalog(self._sync_catalog)
        if hasattr(self, "_sync_inherited_hooks"):
            self._hook_select.set_inherited(self._sync_inherited_hooks)

    def cleanup(self) -> None:
        self._preset_bar.cleanup()
        super().cleanup()
