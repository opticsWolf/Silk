# -*- coding: utf-8 -*-
"""Silk Task Hub Node.

The multi-agent progress surface (spec D58). N independent top-level
agents share no event port and never will (rule 3), so there is no wire
that shows them all — except the one place they all write to. Every
agent's plan is a ``plan-*.db`` under a sandbox root, and the ToolBox node
already aggregates the graph's roots, so the whole graph's plans arrive
here on **one** wire.

The board shows one section per plan, tasks grouped by lane, and
``claimed_by`` as the per-task agent badge — the field the store has
recorded since the schema was written and no view has ever shown, which
is why "who is doing what" was unanswerable in a graph running four
agents.

**What this node does not do.** D58 also gave the hub Approve/Reject
buttons that write a sign-off to the store. There is nothing left to
approve that way: D31–D33 deleted parked sign-off entirely, so a task
change is decided *during* the turn on the run's decision seam, not held
in a row until someone clicks. And D59 is explicit that only the asking
node — or its dock mirror — may answer a live request; the hub may
**count** them, which is what ``pending`` is. So the sign-off half of D58
is absorbed by the inline gate rather than reimplemented here, and the
``signed`` output it specified would have had nothing to pulse.

The hub never talks to a model and holds no run state: it is a database
viewer wearing a node costume, which is exactly what a node is.
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QFormLayout, QLabel

from weave.node.threaded import ThreadedNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.widgetcore import WidgetCore, PortRole
from weave.logger import get_logger

from weave.widgets.markdown_widget import MarkdownWidget
from weave.widgets.sync_button import SyncButton

from ..functions.task_board import PendingDecisions, board, render_board, scan_roots

log = get_logger("SilkTaskHub")


@register_node
class SilkTaskHubNode(ThreadedNode):
    """Kanban projection over every plan under the graph's sandbox roots."""
    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Display"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Task Hub"
    node_description: ClassVar[Optional[str]] = (
        "Shows every plan under the graph's sandbox roots as lanes, with "
        "the agent that claimed each task, and counts agents waiting on a "
        "decision."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "task", "plan", "kanban", "monitor", "display",
    ]
    node_icon: ClassVar[Optional[str]] = "table"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.GROW_ONLY

    def __init__(self, title: str = "Task Hub", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        self._pending = PendingDecisions()
        self._roots: List[str] = []
        self._pending_md: Optional[str] = None
        self._pending_summary: Optional[str] = None
        self._last_board: Dict[str, Any] = {}

        # ── Ports ──
        # `roots` is ToolBox.root_paths: the graph's sandbox ceiling is
        # already the aggregation point, so no per-agent wiring is needed.
        self.add_input("roots", datatype="dirpath_list")
        self.add_input("event", datatype="dict")
        self.add_input("refresh", datatype="exec")
        self.add_output("plans_json", datatype="dict")
        self.add_output("pending", datatype="int")

        # ── Layout & WidgetCore ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self._board = MarkdownWidget(mode="display", safe_mode=True)
        self._board._text_edit.setPlaceholderText("Wire ToolBox.root_paths here…")
        self._board._text_edit.setMinimumHeight(200)
        form.addRow(self._board)
        self._widget_core.register_widget(
            "board", self._board, role=PortRole.DISPLAY,
            datatype="string", default="", add_to_layout=False,
        )

        self._label_summary = QLabel("No plans scanned yet.")
        self._label_summary.setWordWrap(True)
        form.addRow("Status:", self._label_summary)
        self._widget_core.register_widget(
            "summary", self._label_summary, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self.btn_rescan = SyncButton(initial_text="Rescan")
        self.btn_rescan.clicked.connect(self._rescan)
        form.addRow("", self.btn_rescan)
        self._widget_core.register_widget(
            "btn_rescan", self.btn_rescan, role=PortRole.INTERNAL,
            add_to_layout=False,
        )

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── UI (main thread) ──────────────────────────────────────────────

    def _rescan(self) -> None:
        """Re-evaluate without needing an upstream change.

        Plans move because *agents* write to them, not because this node's
        inputs changed, so a hub with no way to re-read would go stale the
        moment the graph settled.

        `set_dirty` rather than `execute()`: `execute` is the *manual*
        node's slot, and this hub is a plain `ThreadedNode`, so the button
        raised `AttributeError` instead of rescanning.
        """
        self.set_dirty("rescan")

    # ── Ingestion ─────────────────────────────────────────────────────

    def _summary_text(self, data: dict) -> str:
        actors = data.get("actors") or []
        waiting = (
            f" · {self._pending.count} waiting on you" if self._pending.count
            else ""
        )
        return (
            f"{data.get('plan_count', 0)} plan(s) · "
            f"{data.get('open_tasks', 0)} open · "
            f"{len(actors)} agent(s){waiting}"
        )

    # ── Streaming path (main thread) ──────────────────────────────────

    def on_upstream_stream(self, port_name: str, value: Any) -> None:
        """Count decision requests as they pass (D58: count, never answer).

        Events arrive by ``emit_stream`` and never run ``compute()``, so the
        count is folded in here and pushed downstream as a preview.
        """
        if port_name == "event":
            if self._pending.record(value):
                self.emit_stream("pending", self._pending.count)
                self._widget_core.push_display(
                    "summary", self._summary_text(self._last_board))
            return
        super().on_upstream_stream(port_name, value)

    # ── Worker thread ─────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_compute_cancelled():
            return {"plans_json": None, "pending": self._pending.count}

        roots = inputs.get("roots") or []
        if roots:
            self._roots = [str(r) for r in roots]
        # A dict wired straight into `event` (rather than streamed) still
        # counts; the stream path above is the usual one.
        self._pending.record(inputs.get("event"))

        data = board(scan_roots(self._roots))
        if self.is_compute_cancelled():
            return {"plans_json": None, "pending": self._pending.count}

        self._last_board = data
        self._pending_md = render_board(data)
        self._pending_summary = self._summary_text(data)
        return {"plans_json": data, "pending": self._pending.count}

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        if self._pending_md is not None:
            self._widget_core.push_display("board", self._pending_md)
            self._pending_md = None
        if self._pending_summary is not None:
            self._widget_core.push_display("summary", self._pending_summary)
            self._pending_summary = None

    def cleanup(self) -> None:
        self.cancel_compute()
        super().cleanup()
