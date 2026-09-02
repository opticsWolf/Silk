# -*- coding: utf-8 -*-
"""Silk Sign-Off Node.

The human end of the task sign-off gate. Lists tasks the agent parked as
``awaiting_signoff`` (with the agent's summary), and lets the user **Approve** or
**Reject** them — the only action that can advance a gated task to ``done``.

Wire ``Silk Agent.events`` → ``event`` to auto-refresh the pending list as
the agent parks tasks, and set ``root`` to the sandbox working dir (or leave it —
it is learned from the plan events). On a decision the node pulses ``signed`` (to
re-trigger the Agent's ``run`` and continue) and re-emits the updated ``plan_json``.

Approval is recorded in the plan's revision log with the human as the actor, so
the audit trail shows who signed off.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout,
)
from PySide6.QtCore import Qt

from weave.node.threaded import ThreadedNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.widgetcore import WidgetCore, PortRole
from weave.logger import get_logger

from weave.widgets.sync_button import SyncButton

from ..functions.task_store import Conflict, SqliteTaskStore, plan_to_json
from ..functions.stream_events import EventType

log = get_logger("SilkSignOff")


@register_node
class SilkSignOffNode(ThreadedNode):
    """User Approve/Reject gate for tasks the agent parked for sign-off."""

    node_class: ClassVar[str] = "AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Sign-Off"
    node_description: ClassVar[Optional[str]] = (
        "Approve or reject tasks the agent submitted for sign-off; "
        "pulses 'signed' to resume the agent."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "plan", "signoff", "approval", "human",
    ]
    node_icon: ClassVar[Optional[str]] = "hand-click"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.GROW_ONLY

    def __init__(self, title: str = "Sign-Off", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        self._root: Optional[str] = None
        self._pending: List[dict] = []

        # ── Ports ──
        self.add_input("root", datatype="string")
        self.add_input("event", datatype="dict")
        self.add_output("plan_json", datatype="dict")
        self.add_output("signed", datatype="exec")

        # ── Layout & WidgetCore ──
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        self._widget_core = WidgetCore(layout=layout)
        self._widget_core.set_node(self)

        layout.addWidget(QLabel("Awaiting sign-off:"))
        self._list = QListWidget()
        self._list.setMinimumHeight(120)
        self._list.currentRowChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list)
        # INTERNAL, not DISPLAY: the list is interactive (the user selects a row),
        # and it is populated with custom items/roles, not via a port push.
        self._widget_core.register_widget(
            "pending_list", self._list, role=PortRole.INTERNAL,
            datatype="list", add_to_layout=False,
        )

        self._summary = QLabel("Select a task to review.")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        self._widget_core.register_widget(
            "summary", self._summary, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        row = QHBoxLayout()
        row.addWidget(QLabel("By:"))
        self._by = QLineEdit("user")
        self._by.setMaximumWidth(120)
        row.addWidget(self._by)
        row.addWidget(QLabel("Note:"))
        self._note = QLineEdit()
        self._note.setPlaceholderText("optional feedback (shown on reject)")
        row.addWidget(self._note)
        layout.addLayout(row)
        self._widget_core.register_widget(
            "by", self._by, role=PortRole.INTERNAL, datatype="string",
            default="user", add_to_layout=False,
        )
        self._widget_core.register_widget(
            "note", self._note, role=PortRole.INTERNAL, datatype="string",
            default="", add_to_layout=False,
        )

        actions = QHBoxLayout()
        self.btn_approve = SyncButton(initial_text="Approve")
        self.btn_approve.clicked.connect(lambda: self._decide(True))
        self.btn_reject = SyncButton(initial_text="Reject")
        self.btn_reject.clicked.connect(lambda: self._decide(False))
        actions.addWidget(self.btn_approve)
        actions.addWidget(self.btn_reject)
        layout.addLayout(actions)
        self._widget_core.register_widget(
            "btn_approve", self.btn_approve, role=PortRole.INTERNAL, add_to_layout=False,
        )
        self._widget_core.register_widget(
            "btn_reject", self.btn_reject, role=PortRole.INTERNAL, add_to_layout=False,
        )

        # ── Mount ──
        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── Pending list (main thread) ────────────────────────────────────

    def _set_summary(self, text: str) -> None:
        self._widget_core.push_display("summary", text)

    def _refresh_pending(self, pending: List[dict]) -> None:
        self._pending = list(pending)
        self._list.blockSignals(True)
        self._list.clear()
        for item in self._pending:
            li = QListWidgetItem(f"{item['id']}: {item.get('title', '')}")
            li.setData(Qt.ItemDataRole.UserRole, item["id"])
            self._list.addItem(li)
        self._list.blockSignals(False)
        if self._pending:
            self._list.setCurrentRow(0)
        else:
            self._set_summary("Nothing awaiting sign-off.")

    def _on_selection_changed(self, row: int) -> None:
        if 0 <= row < len(self._pending):
            it = self._pending[row]
            self._set_summary(f"{it['id']}: {it.get('summary') or '(no summary)'}")

    def _selected_id(self) -> Optional[str]:
        it = self._list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

    # ── Decision (main thread) ────────────────────────────────────────

    def _decide(self, approved: bool) -> None:
        task_id = self._selected_id()
        if not self._root or not task_id:
            self._set_summary("No task selected (or no plan root set).")
            return
        by = (self._by.text().strip() or "user")
        note = self._note.text().strip()
        try:
            store = SqliteTaskStore(root=self._root)
            result = store.sign_off(task_id=task_id, approved=approved, by=by, note=note)
        except Exception as exc:  # noqa: BLE001 - surface, never crash the UI
            log.error("sign_off failed: %s", exc)
            self._set_summary(f"Sign-off failed: {exc}")
            return

        if isinstance(result, Conflict):
            # e.g. another reviewer already decided it, or it's no longer parked.
            self._set_summary(f"Cannot sign off '{task_id}': {result.reason}")
            self._refresh_pending(store.pending_signoffs())
            return

        plan = store.load()
        if plan is not None:
            self.emit_stream("plan_json", plan_to_json(plan))
            self._refresh_pending(store.pending_signoffs())
        self._note.clear()
        verb = "approved" if approved else "rejected"
        self._set_summary(f"Task '{task_id}' {verb} by {by}.")
        # Resume the agent (wire signed -> agent.run to auto-continue).
        self.pulse("signed", payload=True)

    # ── Streaming path (main thread) ──────────────────────────────────

    def on_upstream_stream(self, port_name: str, value: Any) -> None:
        if port_name == "event":
            # The agent's one `events` port carries every type (D2/D3); a
            # pending list only ever changes when the plan does.
            if isinstance(value, dict) and value.get("type") not in (
                None, EventType.PLAN.value,
            ):
                return
            if self._root:
                self._refresh_pending(SqliteTaskStore(root=self._root).pending_signoffs())
            elif isinstance(value, dict) and isinstance(value.get("plan"), dict):
                plan = value["plan"]
                pending = [
                    {"id": t["id"], "title": t.get("title", ""),
                     "summary": t.get("signoff_summary", "")}
                    for t in plan.get("tasks", [])
                    if t.get("status") == "awaiting_signoff"
                ]
                if plan.get("pending_goal"):
                    pending.append({
                        "id": "goal", "title": "Goal revision",
                        "summary": plan.get("pending_goal_summary", ""),
                    })
                self._refresh_pending(pending)
            return
        super().on_upstream_stream(port_name, value)

    # ── Worker thread ─────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_compute_cancelled():
            return {"plan_json": None}
        root = inputs.get("root")
        if root:
            self._root = str(root)
        plan_json = None
        if self._root:
            try:
                store = SqliteTaskStore(root=self._root)
                self._pending_snapshot = store.pending_signoffs()
                plan = store.load()
                plan_json = plan_to_json(plan) if plan is not None else None
            except Exception as exc:  # noqa: BLE001
                log.debug("Sign-off load failed: %s", exc)
                self._pending_snapshot = []
        else:
            self._pending_snapshot = []
        return {"plan_json": plan_json}

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._refresh_pending(getattr(self, "_pending_snapshot", []))

    def cleanup(self) -> None:
        self.cancel_compute()
        super().cleanup()
