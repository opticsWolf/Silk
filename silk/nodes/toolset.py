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

from PySide6.QtWidgets import QFormLayout, QLabel

from weave.widgetcore import WidgetCore, PortRole
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import FILE_PERMISSIONS_TYPE, SILK_TOOLSET_TYPE  # noqa: F401
from ..functions.presets import PresetStore, ToolSetPreset
from ..functions.toolset_build import build_toolset, split_by_ceiling, tool_catalog
from ..widgets.preset_bar import PresetBarWidget
from ..widgets.tool_tree import ToolDetailWidget, ToolTreeWidget

log = get_logger("SilkToolSet")


@register_node
class SilkToolSetNode(ActiveNode):
    """Checkbox-tree downselection of a ToolBox into an agent-facing ToolSet."""

    node_class: ClassVar[str] = "AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk ToolSet"
    node_description: ClassVar[Optional[str]] = (
        "Selects a subset of ToolBox tools for an agent, with optional "
        "per-toolset sandbox permissions and named presets."
    )
    node_tags: ClassVar[Optional[List[str]]] = ["silk", "agent", "tools", "toolset", "llm"]
    node_icon: ClassVar[Optional[str]] = "slice"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT

    def __init__(self, title: str = "Silk ToolSet", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        self.add_input("toolbox", datatype="silk_toolbox")
        self.add_input("permissions", datatype="file_permissions")
        self.add_output("toolset", datatype="silk_toolset")

        # ── Layout & WidgetCore ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

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

        # ── Tool selection tree ──
        self._tool_tree = ToolTreeWidget(checkable=True)
        form.addRow(self._tool_tree)
        self._widget_core.register_widget(
            "checked_tools", self._tool_tree, role=PortRole.INTERNAL,
            datatype="list", default=[], add_to_layout=False,
        )

        self._detail = ToolDetailWidget()
        form.addRow("Details:", self._detail)
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

    # ── Presets (main thread) ─────────────────────────────────────────

    def _collect_preset(self) -> dict:
        return {
            "checked_tools": list(self._widget_core.get_port_value("checked_tools") or []),
        }

    def _apply_preset(self, preset: ToolSetPreset) -> None:
        self._widget_core.apply_port_value("checked_tools", list(preset.checked_tools))

    # ── Worker thread ─────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        toolbox = inputs.get("toolbox")
        checked = [str(n) for n in (inputs.get("checked_tools") or [])]
        permissions = inputs.get("permissions") or None

        if toolbox is None:
            self._sync_catalog: List[Dict[str, Any]] = []
            self._sync_status = "No toolbox connected."
            return {"toolset": None}

        self._sync_catalog = tool_catalog(toolbox)

        try:
            toolset = build_toolset(toolbox, checked, permissions)
        except ValueError as exc:
            self._sync_status = str(exc)
            return {"toolset": None}

        available = {e["name"] for e in self._sync_catalog}
        selected = sorted(set(checked) & available)
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
            f"{sandbox_note}: {', '.join(selected) if selected else 'none'}"
        )
        return {"toolset": toolset}

    # ── State ───────────────────────────────────────────────────────

    def restore_state(self, state: Dict[str, Any]) -> None:
        # 1. Restore values silently (prevents eval storms & false undo history)
        with self._widget_core.suppress_signals():
            super().restore_state(state)

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
