# -*- coding: utf-8 -*-
"""Silk Hook Monitor Node.

Graph-native observability sink for the Agent node's ``tool_events``
stream: a rolling log of run / model / tool events with per-kind and
per-tool counters. Everything it shows is fed by the hook system — this
node is the "events out as ports" half of the hook design (behavior in
is configured on the ToolBox/Role nodes; it never flows through ports).

Wire ``Silk Agent.tool_events`` → ``event``. The ``counts`` output
re-emits the counter state as a dict for further graph reactions.
"""

from collections import deque
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QFormLayout, QLabel

from weave.node.threaded import ThreadedNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.widgetcore import WidgetCore, PortRole
from weave.logger import get_logger

from weave.widgets.markdown_widget import MarkdownWidget
from weave.widgets.sync_button import SyncButton

from ..functions.event_format import EventCounter, event_key, format_event

log = get_logger("SilkHookMonitor")

#: Rolling log length (lines kept / rendered).
LOG_CAPACITY = 200


@register_node
class SilkHookMonitorNode(ThreadedNode):
    """Rolling log + counters over an agent's hook-fed event stream."""

    node_class: ClassVar[str] = "Display"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Hook Monitor"
    node_description: ClassVar[Optional[str]] = (
        "Displays an agent's tool_events stream: rolling event log with "
        "per-kind and per-tool counters."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "hooks", "events", "monitor", "display",
    ]
    node_icon: ClassVar[Optional[str]] = "device-desktop-analytics"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.GROW_ONLY

    def __init__(self, title: str = "Hook Monitor", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # Rolling state (worker-thread mutated, main-thread rendered).
        self._lines: deque[str] = deque(maxlen=LOG_CAPACITY)
        self._counter = EventCounter()
        self._seen_key: Optional[tuple] = None
        self._pending_log: Optional[str] = None
        self._pending_summary: Optional[str] = None

        # ── Ports ──
        self.add_input("event", datatype="dict")
        self.add_output("counts", datatype="dict")

        # ── Layout & WidgetCore ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self._log_display = MarkdownWidget(mode="display", safe_mode=True)
        self._log_display._text_edit.setPlaceholderText("Waiting for agent events…")
        self._log_display._text_edit.setMinimumHeight(160)
        form.addRow(self._log_display)
        self._widget_core.register_widget(
            "log", self._log_display, role=PortRole.DISPLAY,
            datatype="string", default="", add_to_layout=False,
        )

        self._label_summary = QLabel("No events yet.")
        self._label_summary.setWordWrap(True)
        form.addRow("Totals:", self._label_summary)
        self._widget_core.register_widget(
            "summary", self._label_summary, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self.btn_clear = SyncButton(initial_text="Clear")
        self.btn_clear.clicked.connect(self._clear)
        form.addRow("", self.btn_clear)
        self._widget_core.register_widget(
            "btn_clear", self.btn_clear, role=PortRole.INTERNAL,
            add_to_layout=False,
        )

        # ── Mount ──
        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── UI (main thread) ──────────────────────────────────────────────

    def _clear(self) -> None:
        self._lines.clear()
        self._counter.clear()
        self._seen_key = None
        self._widget_core.push_display("log", "<i>Monitor cleared.</i>")
        self._widget_core.push_display("summary", "No events yet.")

    # ── Event ingestion (shared by stream + compute) ──────────────────

    def _record_event(self, event: Any) -> bool:
        """Record one event into the rolling log + counters.

        Returns True if the event was new (not a dedup / not an event).
        Deliberately pure state mutation — no widget writes — so it is
        safe from either the stream hook (main thread) or ``compute``
        (worker thread), and unit-testable without the display.
        """
        if not (isinstance(event, dict) and event.get("event")):
            return False
        key = event_key(event)
        # Dedup: spurious re-evaluations / re-delivered previews.
        if key is not None and key == self._seen_key:
            return False
        self._seen_key = key
        self._lines.append(format_event(event))
        self._counter.record(event)
        return True

    def _log_html(self) -> str:
        if self._lines:
            return f"<code>{'<br>'.join(self._lines)}</code>"
        return "<i>Waiting for agent events…</i>"

    def _summary_text(self) -> str:
        return self._counter.summary() if self._counter.kinds else "No events yet."

    # ── Streaming path (main thread) ──────────────────────────────────

    def on_upstream_stream(self, port_name: str, value: Any) -> None:
        """Consume the Agent's ``tool_events`` stream.

        ``tool_events`` is pushed via ``emit_stream`` — preview pulses that
        bypass the dataflow cache and never trigger ``compute()``. Reading
        ``inputs.get('event')`` in compute therefore never sees them (the
        symptom: the monitor stayed empty while a generic display, which
        also overrides this hook, showed the events). Record + render here.
        """
        if port_name == "event":
            if self._record_event(value):
                self._widget_core.push_display("log", self._log_html())
                self._widget_core.push_display("summary", self._summary_text())
                sb = self._log_display._text_edit.verticalScrollBar()
                sb.setValue(sb.maximum())
                # Keep the `counts` output live too — streams never run
                # compute(), so push the updated tally downstream as a preview.
                self.emit_stream("counts", self._counter.as_dict())
            return
        super().on_upstream_stream(port_name, value)

    # ── Worker thread ─────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_compute_cancelled():
            return {"counts": self._counter.as_dict()}

        # Fallback for a non-streaming dict source wired into `event`;
        # tool_events itself arrives via on_upstream_stream above.
        self._record_event(inputs.get("event"))
        self._pending_log = self._log_html()
        self._pending_summary = self._summary_text()
        return {"counts": self._counter.as_dict()}

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        if self._pending_log is not None:
            self._widget_core.push_display("log", self._pending_log)
            sb = self._log_display._text_edit.verticalScrollBar()
            sb.setValue(sb.maximum())
        if self._pending_summary is not None:
            self._widget_core.push_display("summary", self._pending_summary)

    def cleanup(self) -> None:
        self.cancel_compute()
        super().cleanup()
