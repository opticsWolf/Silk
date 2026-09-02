# -*- coding: utf-8 -*-
"""Silk Plan Viewer Node.

The display **and** graph-composition surface for the agent task tracker. It
shows the current plan (rendered with the user's ``mordant`` markdown parser when
available, plain-markdown fallback otherwise) and exposes the plan as first-class
graph values on output ports:

* ``plan_json`` — the plan snapshot as a dict.
* ``plan_text`` — the rendered markdown view.
* ``plan_html`` — the mordant-rendered HTML (``None`` if mordant is absent).

Wire any of those into a **normal Write node** to persist on the graph's terms —
or leave them unconnected and rely on the store's own direct-write. The DB is the
sole source of truth; everything here is a projection.

Data reaches the node three ways (first that yields a plan wins):
1. an explicit ``plan`` dict on the input port (e.g. routed from a plan snapshot);
2. the ``event`` stream (the agent's ``events`` port: ``plan.summary``
   events carry the snapshot; every other type is ignored);
3. a ``root`` working-directory path — the node loads the newest plan DB there
   (and the Refresh button re-reads it out-of-band).
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QVBoxLayout

from weave.node.threaded import ThreadedNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.widgetcore import WidgetCore, PortRole
from weave.logger import get_logger

from weave.widgets.markdown_widget import MarkdownWidget
from weave.widgets.sync_button import SyncButton

from ..functions.task_store import (
    PlanRef, SqliteTaskStore, Plan, plan_from_json, plan_to_json,
    render_markdown,
)
from ..functions.plan_render import markdown_to_html
from ..functions.stream_events import EventType

log = get_logger("SilkPlanViewer")

_PLACEHOLDER = "_No plan yet. Wire a `root` path or a `plan` snapshot._"


@register_node
class SilkPlanViewerNode(ThreadedNode):
    """Live view of the agent's task plan, with json/text/html output ports."""

    node_class: ClassVar[str] = "Display"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Plan Viewer"
    node_description: ClassVar[Optional[str]] = (
        "Displays the agent's task plan (goal, task tree, course corrections) "
        "and emits it as json / markdown / html for a Write node."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "plan", "tasks", "viewer", "display",
    ]
    node_icon: ClassVar[Optional[str]] = "list-numbers"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.GROW_ONLY

    def __init__(self, title: str = "Plan Viewer", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        self._root: Optional[str] = None
        self._pending_md: Optional[str] = None
        self._pending_html: Optional[str] = None

        # ── Ports ──
        self.add_input("root", datatype="string")
        # The Task node's identity (D23). Given one, this viewer stops
        # guessing which plan a root means.
        self.add_input("plan_ref", datatype="silk_plan")
        self.add_input("plan", datatype="dict")
        self.add_input("event", datatype="dict")
        self.add_output("plan_json", datatype="dict")
        self.add_output("plan_text", datatype="string")
        self.add_output("plan_html", datatype="string")

        # ── Layout & WidgetCore ──
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        self._widget_core = WidgetCore(layout=layout)
        self._widget_core.set_node(self)

        self._view = MarkdownWidget(mode="display", safe_mode=True)
        self._view._text_edit.setMinimumHeight(200)
        layout.addWidget(self._view)
        self._widget_core.register_widget(
            "view", self._view, role=PortRole.DISPLAY,
            datatype="string", default="", add_to_layout=False,
        )

        self.btn_refresh = SyncButton(initial_text="Refresh")
        self.btn_refresh.clicked.connect(self.on_ui_change)  # set_dirty -> re-eval
        layout.addWidget(self.btn_refresh)
        self._widget_core.register_widget(
            "btn_refresh", self.btn_refresh, role=PortRole.INTERNAL,
            add_to_layout=False,
        )

        # ── Mount ──
        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()
        self._apply_display(_PLACEHOLDER, None)

    # ── Rendering (pure; worker- or main-thread safe) ─────────────────

    @staticmethod
    def _render(plan: Plan) -> tuple[dict, str, Optional[str]]:
        """Plan -> (json snapshot, markdown, html). One renderer; mordant only
        touches the html."""
        md = render_markdown(plan)
        return plan_to_json(plan), md, markdown_to_html(md)

    def _resolve_plan(self, *, plan_dict: Any = None, root: Any = None,
                      ref: Any = None) -> Optional[Plan]:
        """First source that yields a plan wins.

        An explicit snapshot, then the Task node's reference (D23), then a
        bare root. The reference outranks the root because that is the
        point of it: a root only says *where* to look, and looking picks
        the newest plan there, which is the wrong plan as soon as a second
        one exists.
        """
        if isinstance(plan_dict, dict) and (
            plan_dict.get("goal") or plan_dict.get("tasks") or plan_dict.get("plan_id")
        ):
            return plan_from_json(plan_dict)
        reference = PlanRef.coerce(ref)
        if reference is not None and (reference.is_explicit or reference.root):
            try:
                return reference.store().load()
            except Exception as exc:  # noqa: BLE001 - a bad ref must not crash eval
                log.debug("Plan load from %s failed: %s", reference, exc)
        if root:
            try:
                return SqliteTaskStore(root=str(root)).load()
            except Exception as exc:  # noqa: BLE001 - a bad path must not crash eval
                log.debug("Plan load from %s failed: %s", root, exc)
        return None

    # ── Display (main thread) ─────────────────────────────────────────

    def _apply_display(self, md: str, html: Optional[str]) -> None:
        if html is not None:
            self._view.set_html(html)          # mordant HTML (checkboxes, emoji)
        else:
            self._view.set_markdown(md)        # fallback: built-in md converter

    # ── Streaming path (main thread) ──────────────────────────────────

    def on_upstream_stream(self, port_name: str, value: Any) -> None:
        """Live-refresh from a plan ``event`` (audit hook) without a recompute.

        The event may carry a ``plan`` snapshot directly, or just signal a change
        (we then re-read the cached ``root``)."""
        if port_name == "event":
            # One port carries every event now (spec D2/D3), so the viewer
            # picks its own out rather than assuming everything arriving is
            # a plan. Anything else is not a change of plan and must not
            # trigger a re-render.
            if not isinstance(value, dict):
                return
            if value.get("type") not in (None, EventType.PLAN.value):
                return
            plan_dict = value.get("plan")
            plan = self._resolve_plan(plan_dict=plan_dict, root=self._root,
                                      ref=getattr(self, "_plan_ref", None))
            if plan is not None:
                pj, md, html = self._render(plan)
                self._apply_display(md, html)
                self.emit_stream("plan_json", pj)
                self.emit_stream("plan_text", md)
                self.emit_stream("plan_html", html)
            return
        super().on_upstream_stream(port_name, value)

    # ── Worker thread ─────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_compute_cancelled():
            return {"plan_json": None, "plan_text": None, "plan_html": None}

        root = inputs.get("root")
        if root:
            self._root = str(root)
        ref = inputs.get("plan_ref")
        if ref is not None:
            self._plan_ref = ref
        plan = self._resolve_plan(plan_dict=inputs.get("plan"), root=self._root,
                                  ref=getattr(self, "_plan_ref", None))

        if plan is None:
            self._pending_md, self._pending_html = _PLACEHOLDER, None
            return {"plan_json": None, "plan_text": None, "plan_html": None}

        pj, md, html = self._render(plan)
        self._pending_md, self._pending_html = md, html
        return {"plan_json": pj, "plan_text": md, "plan_html": html}

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        if self._pending_md is not None:
            self._apply_display(self._pending_md, self._pending_html)
            self._pending_md = self._pending_html = None

    def cleanup(self) -> None:
        self.cancel_compute()
        super().cleanup()
