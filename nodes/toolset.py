# -*- coding: utf-8 -*-
"""Silk ToolSet Node.

Derives a restricted **ToolSet** from an upstream Silk ToolBox: the tree
lists every tool of the toolbox grouped by category (category rows are
tri-state quick-selectors), and only the checked tools make it into the
emitted ``silk_toolset``. Agents accept *only* toolsets — never the raw
toolbox — so the full registry can't reach a model by accident.

The toolset is rebuilt from the toolbox's recipe as an independent
ToolBox instance, optionally re-rooted onto its own sandbox from a
``file_permissions`` input (per-path read / read-write grants from the
Select Files → Permission nodes). Selections can be stored as named
presets (JSON, pydantic-validated on load).
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel

from weave.widgetcore import WidgetCore, PortRole
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import FILE_PERMISSIONS_TYPE, SILK_TOOLSET_TYPE  # noqa: F401
from ..functions.file_grants import FileGrant, FileGrants
from ..functions.presets import PresetStore, ToolSetPreset
from ..functions.tools.okf_tools import OKF_LEVELS, OKF_TOOLS, narrow_okf_level
from ..functions.toolset_build import build_toolset, split_by_ceiling, tool_catalog
from ..functions.tool_groups import (
    categories_of, configurable_categories, group_config_title,
)
from ..widgets.group_config import ConfigDialogs
from ..widgets.preset_bar import PresetBarWidget
from ..widgets.split_form import SplitForm, SplitFormState
from ..widgets.tool_tree import (
    ALL_CATEGORIES, CategoryCombo, ToolDetailWidget, ToolTreeWidget,
)

log = get_logger("SilkToolSet")


@register_node
class SilkToolSetNode(SplitFormState, ActiveNode):
    """Checkbox-tree downselection of a ToolBox into an agent-facing ToolSet."""
    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk ToolSet"
    node_description: ClassVar[Optional[str]] = (
        "Selects a subset of ToolBox tools for an agent, with optional "
        "per-toolset sandbox permissions and named presets."
    )
    node_tags: ClassVar[Optional[List[str]]] = ["silk", "agent", "tools", "toolset", "llm"]
    node_icon: ClassVar[Optional[str]] = "slice"
    # Keeps the height the user drags it to: the groups share it.
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.GROW_ONLY
    node_state_api = 4   # owns a hand-written state dict
    # 1 = the shape this spec settled on. 2 registers the preset bar
    # (`preset` binding) so the mirror panel builds a working twin.
    # Additive: older states have no entry and restore unselected.
    # 3 adds the knowledge-graph level (`knowledge_level`), also additive:
    # absent, the toolset keeps the ToolBox's level.
    # 4 adds Allow All (`allow_all`) and the category filter
    # (`category_filter`), additive: absent, off and all categories.
    node_version = 4     # bump on any state-shape change (G20); 4: allow all, category

    #: The level combo's first entry: no narrowing.
    _LEVEL_AS_TOOLBOX = "as ToolBox"

    def __init__(self, title: str = "Silk ToolSet", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        self.add_input("toolbox", datatype="silk_toolbox")
        self.add_input("permissions", datatype="file_permissions")
        self.add_output("toolset", datatype="silk_toolset")
        # The grant this toolset actually ended up with, after the ToolBox
        # ceiling dropped whatever lay outside it (D16). Emitting it is what
        # makes the effective permission set visible in the graph instead of
        # buried inside a sandbox handle -- and it is what the Role and Agent
        # narrow further down the chain.
        self.add_output("permissions", datatype="file_permissions")

        # ── Layout & WidgetCore ──
        # Two groups, resizable against each other: the selection (preset
        # and the tool tree), and the focused tool's detail.
        self._split = SplitForm(2, stretch=(2, 1))
        self._widget_core = WidgetCore(layout=self._split.layout)
        self._widget_core.set_node(self)
        form = self._split.forms[0]

        # ── Presets ──
        self._preset_store: PresetStore[ToolSetPreset] = PresetStore(
            "silk_toolsets", ToolSetPreset
        )
        self._preset_bar = PresetBarWidget(
            self._preset_store,
            collect=self._collect_preset,
            apply=self._apply_preset,
        )
        form.addRow("Preset:", self._preset_bar)
        self._widget_core.register_widget(
            "preset", self._preset_bar, role=PortRole.INTERNAL,
            datatype="str", default="", add_to_layout=False,
        )

        # ── Tool selection: category filter, Allow All, tree ──
        self._tool_tree = ToolTreeWidget(checkable=True)

        # What this toolset's agents may do to the knowledge graph. Only
        # ever lower than the ToolBox's level (the ToolBox is the grant);
        # picking a higher one keeps the ToolBox's. Behind the gear on the
        # knowledge row, as on the ToolBox: it is the grant those tools
        # run under, so it sits with them. Registered, so it saves.
        self._knowledge_level = QComboBox()
        self._knowledge_level.addItems([self._LEVEL_AS_TOOLBOX, *OKF_LEVELS])
        self._knowledge_level.setToolTip(
            "read: search, read, traverse. write: also okf_ingest. curate: "
            "also update, supersede and delete.")
        self._knowledge_level.intro = (  # type: ignore[attr-defined]
            "Lower the knowledge graph's access for this toolset's agents. "
            "Never above the ToolBox's level: picking a higher one keeps "
            "the ToolBox's.")
        self._widget_core.register_widget(
            "knowledge_level", self._knowledge_level, role=PortRole.INTERNAL,
            datatype="string", default=self._LEVEL_AS_TOOLBOX,
            add_to_layout=False,
        )
        self._config_dialogs = ConfigDialogs()
        self._tool_tree.set_configurable(configurable_categories(("knowledge",)))
        self._tool_tree.config_requested.connect(self._open_group_config)
        self._combo_category = CategoryCombo(self._tool_tree)
        form.addRow("Category:", self._combo_category)
        self._widget_core.register_widget(
            "category_filter", self._combo_category, role=PortRole.INTERNAL,
            datatype="string", default=ALL_CATEGORIES, add_to_layout=False,
        )

        # Every tool the ToolBox offers, now and as it changes. The tree
        # shows them all ticked and locks; the user's own ticks stay
        # underneath for when it is lifted.
        self.chk_allow_all = QCheckBox()
        self.chk_allow_all.setToolTip(
            "Take every tool the connected ToolBox offers (skip "
            "downselection).")
        form.addRow("Allow All Tools:", self.chk_allow_all)
        self._widget_core.register_widget(
            "allow_all", self.chk_allow_all, role=PortRole.INTERNAL,
            datatype="bool", default=False, add_to_layout=False,
        )
        # Programmatic writes (presets, restore) block the box's signals
        # and are announced on port_value_written instead.
        self.chk_allow_all.toggled.connect(self._tool_tree.set_all_granted)
        self._widget_core.port_value_written.connect(self._on_port_written)

        self._split.fill(self._tool_tree)
        form.addRow(self._tool_tree)
        self._widget_core.register_widget(
            "checked_tools", self._tool_tree, role=PortRole.INTERNAL,
            datatype="list", default=[], add_to_layout=False,
        )

        form = self._split.forms[1]
        self._detail = ToolDetailWidget()
        self._split.fill(self._detail)
        self._split.add_above(form, "Details:", self._detail)
        # DISPLAY binding (no pushes): canvas-resolvable, menu suppressed
        # via the registered builder returning None.
        self._widget_core.register_widget(
            "tool_detail", self._detail, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )
        # UI-local wiring: focused tree row drives the detail preview.
        self._tool_tree.tool_focused.connect(self._detail.show_tool)

        self._label_status = QLabel("No toolbox connected.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        # ── Mount ──
        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── UI helpers (main thread only) ─────────────────────────────────

    def _on_port_written(self, port_name: str, _value: Any) -> None:
        if port_name == "allow_all":
            self._tool_tree.set_all_granted(self.chk_allow_all.isChecked())

    def _open_group_config(self, category: str) -> None:
        """The knowledge row's gear: this toolset's access level."""
        if category in categories_of("knowledge"):
            self._config_dialogs.open(
                category, self._knowledge_level, group_config_title(category))

    # ── Presets (main thread) ─────────────────────────────────────────

    def _collect_preset(self) -> dict:
        return {
            "checked_tools": list(self._widget_core.get_port_value("checked_tools") or []),
            "allow_all": bool(self._widget_core.get_port_value("allow_all")),
        }

    def _apply_preset(self, preset: ToolSetPreset) -> None:
        self._widget_core.apply_port_value("checked_tools", list(preset.checked_tools))
        self._widget_core.apply_port_value("allow_all", preset.allow_all)

    # ── Worker thread ─────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        toolbox = inputs.get("toolbox")
        checked = [str(n) for n in (inputs.get("checked_tools") or [])]
        permissions = inputs.get("permissions") or None

        if toolbox is None:
            self._sync_catalog: List[Dict[str, Any]] = []
            self._sync_status = "No toolbox connected."
            return {"toolset": None, "permissions": None}

        self._sync_catalog = tool_catalog(toolbox)
        if inputs.get("allow_all"):
            checked = [str(e["name"]) for e in self._sync_catalog]

        try:
            toolset = build_toolset(toolbox, checked, permissions)
        except ValueError as exc:
            self._sync_status = str(exc)
            return {"toolset": None, "permissions": None}

        wanted_level = str(inputs.get("knowledge_level") or "")
        level_note = ""
        if wanted_level in OKF_LEVELS:
            effective_level = narrow_okf_level(toolset, wanted_level)
            if effective_level:
                level_note = f" · knowledge: {effective_level}"
                if effective_level != wanted_level:
                    level_note += " (the ToolBox's ceiling)"

        available = {e["name"] for e in self._sync_catalog}
        # What the toolset really carries: a narrowed level drops okf
        # tools that are ticked and offered by the ToolBox.
        dropped = {n for n in OKF_TOOLS if n not in toolset.tools}
        selected = sorted((set(checked) & available) - dropped)
        # Preset fallback transparency: names checked (e.g. from a preset)
        # but absent from this toolbox stay remembered and re-activate if
        # the tool returns — report them instead of silently ignoring.
        unavailable = sorted(set(checked) - available)
        sandbox_note = ""
        if unavailable:
            sandbox_note += (
                f" · {len(unavailable)} unavailable (kept): "
                f"{', '.join(unavailable)}"
            )
        if permissions:
            _inside, outside = split_by_ceiling(
                permissions, getattr(toolbox, "base_sandbox", None)
            )
            sandbox_note = " · own sandbox"
            if outside:
                # The ToolBox sandbox roots are the hard ceiling: grants
                # outside them are never honoured, only reported.
                sandbox_note += (
                    f" · {len(outside)} path(s) outside the sandbox "
                    f"ceiling ignored"
                )
        self._sync_status = (
            f"{len(selected)}/{len(self._sync_catalog)} tools selected"
            f"{sandbox_note}{level_note}: "
            f"{', '.join(selected) if selected else 'none'}"
        )
        effective = FileGrants.coerce(permissions)
        if effective is not None:
            inside, _outside = split_by_ceiling(
                effective, getattr(toolbox, "base_sandbox", None)
            )
            effective = FileGrants(
                root=effective.root, roots=effective.effective_roots(),
                entries=[FileGrant(**e) for e in inside],
            )
        return {"toolset": toolset, "permissions": effective}

    # ── State ───────────────────────────────────────────────────────

    def migrate_state(cls, state: Dict[str, Any],
                      from_version: int) -> Dict[str, Any]:
        """api 1 → 2: the preset binding is new; older states restore
        with the placeholder selected. 2 → 3: the knowledge level is new;
        older states restore "as ToolBox", which is what they did.
        3 → 4: Allow All and the category filter are new; older states
        restore with Allow All off and every category shown."""
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        # 1. Restore values silently (prevents eval storms & false undo history)
        with self._widget_core.suppress_signals():
            super().restore_state(state)
        self._tool_tree.set_all_granted(self.chk_allow_all.isChecked())

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        if hasattr(self, "_sync_catalog"):
            self._tool_tree.set_catalog(self._sync_catalog)
        if hasattr(self, "_sync_status"):
            self._widget_core.push_display("status", self._sync_status)

    def cleanup(self) -> None:
        try:
            self._tool_tree.tool_focused.disconnect(self._detail.show_tool)
        except (RuntimeError, TypeError):
            pass
        self._preset_bar.cleanup()
        super().cleanup()
