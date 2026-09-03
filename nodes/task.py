# -*- coding: utf-8 -*-
"""Silk Task Node.

Names the plan an agent works on, instead of letting it be inferred
(spec D23). Without this node the task store picks the newest
``plan-*.db`` under the sandbox root, which is fine while one plan lives
there and quietly wrong the moment two do: two agents in one directory
cross-discover each other's plans, and which one they land on depends on
file timestamps. That is the whole of T4.

The node emits a ``PlanRef`` -- root, file, plan id, label -- which the
ToolBox node hands to the task tools and the Plan Viewer reads instead of
guessing. Existing plans under the root are listed with their goal and
open-task count, so choosing one is choosing something legible rather
than a filename.

Discovery-by-newest is kept as the unnamed case, because it is also the
mechanism by which several agents deliberately share one plan.
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QLineEdit

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import SILK_PLAN_TYPE  # noqa: F401
from ..functions.plan_discovery import scan_all
from ..functions.task_store import PlanRef, SqliteTaskStore

log = get_logger("SilkTask")

#: What the plan dropdown offers when no existing plan is chosen.
NEW_PLAN = "(new plan)"
NEWEST = "(newest under root)"


@register_node
class SilkTaskNode(ActiveNode):
    """Explicit plan identity: which plan, and where it lives."""
    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk Task"
    node_description: ClassVar[Optional[str]] = (
        "Names the plan agents work on — the plan file and its id — so the "
        "store never has to guess which plan a root means."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "task", "plan", "llm",
    ]
    node_icon: ClassVar[Optional[str]] = "list-numbers"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1
    node_version = 1     # bump on any state-shape change (G20)

    def __init__(self, title: str = "Silk Task", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        self.add_input("root", datatype="string")
        self.add_input("root_paths", datatype="dirpath_list")
        self.add_output("plan", datatype="silk_plan")

        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self._plan_choice = QComboBox()
        self._plan_choice.addItems([NEW_PLAN, NEWEST])
        form.addRow("Plan:", self._plan_choice)
        self._widget_core.register_widget(
            "plan_choice", self._plan_choice, role=PortRole.INPUT,
            datatype="string", default=NEW_PLAN, add_to_layout=False,
        )
        self.add_input("plan_choice", datatype="string")

        self._plan_name = QLineEdit("")
        self._plan_name.setPlaceholderText("refactor-parser  (names a new plan file)")
        form.addRow("Name:", self._plan_name)
        self._widget_core.register_widget(
            "plan_name", self._plan_name, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(500),
            add_to_layout=False,
        )
        self.add_input("plan_name", datatype="string")

        self._label_status = QLabel("Connect a root to name a plan.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── Worker thread ─────────────────────────────────────────────────

    @staticmethod
    def resolve_root(inputs: Dict[str, Any]) -> str:
        """The directory the plan lives under.

        The ToolBox node's ``root_paths`` output is the natural source, so
        the plan lands inside the sandbox ceiling rather than beside it.
        """
        roots = [str(p).strip() for p in (inputs.get("root_paths") or [])
                 if str(p).strip()]
        if roots:
            return roots[0]
        return str(inputs.get("root") or "").strip()

    @staticmethod
    def plan_ref(root: str, choice: Optional[str], name: Optional[str],
                 rows: List[dict]) -> PlanRef:
        """The reference these inputs name (D23).

        Three cases, and the third is the one that matters: an existing
        plan chosen by label is pinned to *its* file, so it stays the same
        plan when a newer one appears in the same directory.
        """
        choice = (choice or NEW_PLAN).strip()
        name = (name or "").strip()

        if choice == NEWEST:
            # No file named: the store falls back to newest-under-root,
            # which is how several agents share one plan on purpose.
            return PlanRef(root=root, label=NEWEST)

        for row in rows:
            if choice in (row.get("label"), row.get("db_path")):
                return PlanRef(root=root, db_path=row["db_path"],
                               plan_id=row.get("plan_id", ""),
                               label=row.get("label", choice))

        # A new plan, at a path this node names. Naming it is what makes it
        # findable again after a restart -- a generated stem would not be.
        stem = name or "plan-graph"
        if not stem.startswith("plan-"):
            stem = f"plan-{stem}"
        return PlanRef(root=root, db_path=str(_plan_path(root, stem)),
                       label=stem)

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        root = self.resolve_root(inputs)
        if not root:
            self._sync_rows: List[dict] = []
            self._sync_status = "Connect a root (or the ToolBox root_paths)."
            return {"plan": None}

        try:
            rows = scan_all(root)
        except OSError as exc:
            rows = []
            log.debug(f"Plan scan of {root} failed: {exc}")

        ref = self.plan_ref(root, inputs.get("plan_choice"),
                            inputs.get("plan_name"), rows)
        self._sync_rows = rows
        chosen = next((r for r in rows if r["db_path"] == ref.db_path), None)
        if chosen is not None:
            self._sync_status = (
                f"{chosen['label']} · {chosen.get('goal') or 'no goal yet'} · "
                f"{chosen.get('open_tasks', 0)}/{chosen.get('tasks', 0)} open"
            )
        elif ref.is_explicit:
            self._sync_status = f"{ref.label} · new plan, not created yet"
        else:
            self._sync_status = (
                f"newest of {len(rows)} plan(s) under the root — shared "
                f"discovery, not a named plan"
            )
        return {"plan": ref}

    # ── Main thread ───────────────────────────────────────────────────

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        if hasattr(self, "_sync_rows"):
            self._refresh_choices(self._sync_rows)
        if hasattr(self, "_sync_status"):
            self._widget_core.push_display("status", self._sync_status)

    def _refresh_choices(self, rows: List[dict]) -> None:
        """Rebuild the dropdown, keeping the current selection if it lives."""
        wanted = self._plan_choice.currentText()
        labels = [NEW_PLAN, NEWEST] + [r["label"] for r in rows]
        if labels == [self._plan_choice.itemText(i)
                      for i in range(self._plan_choice.count())]:
            return
        blocked = self._plan_choice.blockSignals(True)
        try:
            self._plan_choice.clear()
            self._plan_choice.addItems(labels)
            index = self._plan_choice.findText(wanted)
            self._plan_choice.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self._plan_choice.blockSignals(blocked)


def _plan_path(root: str, stem: str):
    """Where a named plan file goes: beside the root, else under .silk/plan.

    The suffix follows the backend this process asked for, because since
    T4's discovery went backend-blind the suffix is *how* a plan says
    which store opens it. Naming a plan on a process running the ledger
    and then reading it back through SQLite is the one mistake this
    removes.
    """
    from pathlib import Path

    from ..functions.ledger import BACKEND_LEDGER, requested_backend

    suffix = ".macrame" if requested_backend() == BACKEND_LEDGER else ".db"
    base = Path(root).resolve()
    if SqliteTaskStore._writable_dir(base):
        return base / f"{stem}{suffix}"
    return base / ".silk" / "plan" / f"{stem}{suffix}"
