# -*- coding: utf-8 -*-
"""Pool Monitor Node.

Takes a ``model_obj`` handle and emits a live snapshot of the GGUF pool
state. Refreshed on demand via an ``exec`` input pulse — wire any agent's
``done`` port here so the status updates after every run.

Only one monitor needed per pool regardless of how many agents share it.
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget

from weave.widgetcore import WidgetCore, PortRole
from weave.node.threaded import ThreadedManualNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.widgets.sync_button import SyncButton


def queue_note(info: dict) -> str:
    """What to add to the flags line about requests waiting (§22 q1c).

    One server serves one request at a time (D43), so a fan-out queues.
    That is correct and it looks like a hang, which is the failure D53
    named. Nothing is shown until it actually happens -- a line that is
    always there is one nobody reads.
    """
    queue = info.get("serialization") or {}
    if not queue.get("serialising"):
        return ""
    return (
        "  \u00b7  queued: {in_flight} in flight, peak {peak_in_flight}, "
        "{queued_requests} waited".format(
            in_flight=queue.get("in_flight", 0),
            peak_in_flight=queue.get("peak_in_flight", 0),
            queued_requests=queue.get("queued_requests", 0),
        )
    )


@register_node
class PoolMonitorNode(ThreadedManualNode):
    """Reads pool state from a model handle and emits a status dict."""

    # Worker → main-thread bridge for display updates.
    _display_update = Signal(str, str)

    node_class: ClassVar[str] = "AI"
    node_subclass: ClassVar[str] = "Monitor"
    node_name: ClassVar[Optional[str]] = "Pool Monitor"
    node_description: ClassVar[Optional[str]] = (
        "Displays GGUF pool usage: active/idle instances, capacity. "
        "Wire any agent's 'done' port to 'refresh' for live updates."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "pool", "monitor", "status", "observability",
    ]
    node_icon: ClassVar[Optional[str]] = "device-desktop-analytics"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT

    def __init__(self, title: str = "Pool Monitor", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        self.add_input("model_obj", datatype="gguf_model")
        self.add_input("refresh", datatype="exec")  # pulse from any agent's done port
        self.add_output("pool_status", datatype="dict")

        # ── Layout ──
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        form = QFormLayout()
        form.setSpacing(4)

        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        # Read-only display fields
        self._label_model = QLabel("—")
        self._label_model.setWordWrap(True)
        form.addRow("Model:", self._label_model)
        self._widget_core.register_widget(
            "display_model", self._label_model, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self._label_usage = QLabel("—")
        self._label_usage.setWordWrap(True)
        form.addRow("Usage:", self._label_usage)
        self._widget_core.register_widget(
            "display_usage", self._label_usage, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self._label_flags = QLabel("—")
        self._label_flags.setWordWrap(True)
        form.addRow("Flags:", self._label_flags)
        self._widget_core.register_widget(
            "display_flags", self._label_flags, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        # Manual refresh button
        self.btn_refresh = SyncButton(initial_text="Refresh")
        self.btn_refresh.clicked.connect(self.execute)
        form.addRow("", self.btn_refresh)
        self._widget_core.register_widget(
            "btn_refresh", self.btn_refresh, role=PortRole.INTERNAL,
            add_to_layout=False,
        )

        layout.addLayout(form)
        container = QWidget()
        container.setLayout(layout)

        # ── Signal wiring (worker → main thread) ──
        self._display_update.connect(self._on_display_update)

        # ── Mount ──
        self.set_content_widget(container)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── UI slots (main thread) ──────────────────────────────────────

    @Slot(str, str)
    def _on_display_update(self, field: str, value: str) -> None:
        mapping = {
            "model": "display_model",
            "usage": "display_usage",
            "flags": "display_flags",
        }
        target = mapping.get(field)
        if target:
            self._widget_core.push_display(target, value)

    # ── Compute (worker thread) ──────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_handle = inputs.get("model_obj")
        if not isinstance(model_handle, dict):
            return {"pool_status": None}

        pool = model_handle.get("pool")
        if pool is None:
            return {"pool_status": None}

        try:
            info = pool.snapshot()
        except Exception:
            return {"pool_status": None}

        # Emit display updates via queued signal (worker → main thread).
        self._display_update.emit("model", info.get("model_path", "—"))

        bound = info.get("bound_sessions", 0)
        idle = info.get("idle", 0)
        capacity = info.get("capacity", 0)
        total = info.get("total_instances", 0)
        self._display_update.emit(
            "usage",
            f"{bound} bound  \u00b7  {idle} idle  \u00b7  {total} total  \u00b7  capacity {capacity}",
        )

        clear_flag = "clear-on-return: on" if info.get("clear_on_return") else "clear-on-return: off"
        self._display_update.emit("flags", clear_flag + queue_note(info))

        return {"pool_status": info}

    # ── Cleanup ──────────────────────────────────────────────────────

    def cleanup(self) -> None:
        self.cancel_compute()
        try:
            self._display_update.disconnect()
        except (RuntimeError, TypeError):
            pass
        super().cleanup()
