# -*- coding: utf-8 -*-
"""Silk Toolchain Node.

Configures a **set** of external toolchains — Python interpreters/venvs,
ruff, mypy, radon, maturin, cargo — and emits them on a ``toolchains``
port. Pick a kind + executable (the dropdown remembers recent picks) and
press **Add**: the toolchain lands in the checkable list below, where
the checkbox enables/disables it, double-clicking the executable cell
edits the path, and the right-click menu removes entries.

The same kind may appear multiple times (two venvs!): tool names are
disambiguated at attach time with numbered suffixes (``run_python``,
``run_python_2``, …), each carrying its environment label in the
model-visible description.

Every enabled entry is version-probed on evaluation, so a broken path
fails here, visibly, instead of inside an agent run. Nodes still chain:
wire ``toolchains`` → ``toolchains`` to accumulate across nodes.
"""

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel

from weave.widgetcore import WidgetCore, PortRole
from weave.node.threaded import ThreadedNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from weave.widgets.path_history_picker import PathHistoryPicker
from weave.library.sync_tool_button.widgets.sync_tool_button import SyncToolButton

from .silk_ports import TOOLCHAINS_TYPE  # noqa: F401
from ..functions.tools.toolchains import (
    KNOWN_TOOLCHAINS,
    SPEC_PACKS,
    ToolchainError,
    probe_toolchain,
)
from ..widgets.toolchain_list import ToolchainListWidget

log = get_logger("SilkToolchain")


@register_node
class ToolchainNode(ThreadedNode):
    """Configures a checkable set of probed toolchains; chainable."""

    node_class: ClassVar[str] = "AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Toolchain"
    node_description: ClassVar[Optional[str]] = (
        "Configures external toolchains (python venvs, ruff, mypy, radon, "
        "maturin, cargo) for agent tools; checkable list, chainable."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "toolchain", "python", "venv", "cargo", "build",
    ]
    node_icon: ClassVar[Optional[str]] = "braces"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.GROW_ONLY

    def __init__(self, title: str = "Toolchain", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        self.add_input("toolchains", datatype="toolchains")  # chain input
        self.add_output("toolchains", datatype="toolchains")

        # ── Layout & WidgetCore ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        # ── Picker row: kind + executable + Add ──
        self._combo_kind = QComboBox()
        for kind in sorted(KNOWN_TOOLCHAINS):
            tools = ", ".join(s.tool_name for s in SPEC_PACKS.get(kind, []))
            self._combo_kind.addItem(kind)
            self._combo_kind.setItemData(
                self._combo_kind.count() - 1, f"Tools: {tools}",
                Qt.ItemDataRole.ToolTipRole,
            )
        form.addRow("Toolchain:", self._combo_kind)
        self._widget_core.register_widget(
            "kind", self._combo_kind, role=PortRole.INTERNAL,
            datatype="str", default="python", add_to_layout=False,
        )

        picker_row = QHBoxLayout()
        picker_row.setSpacing(4)
        self._exe_picker = PathHistoryPicker(mode="file")
        self._exe_picker.setToolTip(
            "Executable for the selected toolchain (e.g. a venv's "
            "python.exe). Leave empty to auto-locate on PATH."
        )
        picker_row.addWidget(self._exe_picker, stretch=1)

        self._btn_add = SyncToolButton(
            initial_text="Add", show_label=False, dimensions=24
        )
        self._btn_add.set_tooltip("Add this toolchain to the list below.")
        self._btn_add.clicked.connect(self._on_add)
        picker_row.addWidget(self._btn_add)
        form.addRow("Executable:", picker_row)
        self._widget_core.register_widget(
            "executable", self._exe_picker, role=PortRole.INTERNAL,
            datatype="str", default="", add_to_layout=False,
        )

        # ── Configured toolchains ──
        self._chain_list = ToolchainListWidget()
        form.addRow(self._chain_list)
        self._widget_core.register_widget(
            "toolchain_entries", self._chain_list, role=PortRole.INTERNAL,
            datatype="list", default=[], add_to_layout=False,
        )

        self._label_status = QLabel("No toolchains configured.")
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

    # ── UI (main thread) ──────────────────────────────────────────────

    def _on_add(self) -> None:
        kind = self._combo_kind.currentText()
        executable = str(self._widget_core.get_port_value("executable") or "")
        if not self._chain_list.add_entry(kind, executable):
            self._widget_core.push_display(
                "status", f"'{kind}' with this executable is already listed."
            )

    # ── State: keep the executable history across saves ───────────────

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["exe_history"] = self._exe_picker.history()
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        # 1. Restore values silently (prevents eval storms & false undo history)
        with self._widget_core.suppress_signals():
            super().restore_state(state)
        # 2. Restore non-widget internal state
        self._exe_picker.set_history(state.get("exe_history", []))

    # ── Worker thread ─────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        upstream = list(inputs.get("toolchains") or [])
        entries = [
            e for e in (inputs.get("toolchain_entries") or [])
            if isinstance(e, dict) and e.get("kind")
        ]

        if self.is_compute_cancelled():
            return {"toolchains": upstream}

        probed, lines = [], []
        for entry in entries:
            if not entry.get("enabled", True):
                continue
            kind = str(entry["kind"])
            executable = str(entry.get("executable") or "").strip()
            try:
                env = probe_toolchain(kind, executable or None)
            except ToolchainError as exc:
                lines.append(f"⚠ {kind}: {exc}")
                continue
            env = _labelled(env)
            probed.append(env)
            lines.append(f"{env.label}: {env.version}")

        if not lines:
            lines.append("No toolchains configured — Add one above.")
        self._sync_status = "\n".join(lines)
        return {"toolchains": upstream + probed}

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        if hasattr(self, "_sync_status"):
            self._widget_core.push_display("status", self._sync_status)

    def cleanup(self) -> None:
        self.cancel_compute()
        self._btn_add.cleanup()
        super().cleanup()


def _labelled(env):
    """Derive a human label from the executable location (venv name)."""
    from dataclasses import replace

    parts = Path(env.executable).parts
    # ...\<venv>\Scripts\python.exe → "<venv>"; fall back to the kind.
    container = parts[-3] if len(parts) >= 3 else ""
    label = f"{env.id} ({container})" if container else env.id
    return replace(env, label=label)
