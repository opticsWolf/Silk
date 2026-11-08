# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Silk MCP Aggregator Node.

Several MCP servers arrive on one wire; this node decides which of their
tools an agent may see (spec D20). The tree is the same
``widgets/tool_tree.py`` the ToolSet node uses, so a server is a category
row -- untick the row to switch the whole server off, untick a leaf to
drop one tool.

Unchecking never closes a session. The sessions belong to the MCP nodes
(D19), and a tool the user is toggling on and off should not cost a
handshake each time; what travels on is the same bundle with exclusions
recorded on it.

The downselection here is a *convenience*, not a boundary: it decides
what is attached to a ToolBox. What one agent may then call is the Role's
business, and the role gate is what actually refuses a call.
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QLabel

from weave.widgetcore import WidgetCore, PortRole
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import MCP_SERVERS_TYPE  # noqa: F401
from ..functions.mcp_session import MCPBundle
from ..widgets.split_form import SplitForm, SplitFormState
from ..widgets.tool_tree import ToolDetailWidget, ToolTreeWidget

log = get_logger("SilkMCPAggregator")


@register_node
class SilkMCPAggregatorNode(SplitFormState, ActiveNode):
    """Checkbox tree over the MCP servers on one wire."""
    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk MCP Aggregator"
    node_description: ClassVar[Optional[str]] = (
        "Combines several MCP servers onto one wire and enables or "
        "disables individual servers and tools."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "tools", "mcp", "llm",
    ]
    node_icon: ClassVar[Optional[str]] = "arrows-join"
    # Keeps the height the user drags it to: the groups share it.
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.GROW_ONLY
    node_state_api = 1
    node_version = 2     # bump on any state-shape change (G20); 2: splitter sizes

    def __init__(self, title: str = "Silk MCP Aggregator", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        self.add_input("mcp_in", datatype="mcp_servers")
        self.add_output("mcp", datatype="mcp_servers")

        # Two groups, resizable against each other: the tree, and the
        # focused tool's detail with the status.
        self._split = SplitForm(2, stretch=(2, 1))
        self._widget_core = WidgetCore(layout=self._split.layout)
        self._widget_core.set_node(self)
        form = self._split.forms[0]

        self._tool_tree = ToolTreeWidget(checkable=True)
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
        self._widget_core.register_widget(
            "tool_detail", self._detail, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )
        self._tool_tree.tool_focused.connect(self._detail.show_tool)

        self._label_status = QLabel("No MCP servers connected.")
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
    def select(bundle: MCPBundle, checked: Any) -> MCPBundle:
        """*bundle* with everything unchecked recorded as disabled.

        An empty selection means "nothing has been unticked yet", not
        "everything off": a freshly placed node has no tree state, and
        silently disabling every server would look like the servers
        failed to connect.
        """
        names = {str(name) for name in (checked or [])}
        if not names:
            return bundle
        available = {row["name"] for row in bundle.catalog()}
        disabled_tools = available - names
        disabled_servers = {
            session.spec.id for session in bundle.sessions
            if session.tools and all(
                entry["name"] in disabled_tools for entry in session.tools
            )
        }
        return MCPBundle(
            sessions=list(bundle.sessions),
            disabled_servers=set(bundle.disabled_servers) | disabled_servers,
            disabled_tools=set(bundle.disabled_tools) | disabled_tools,
        )

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        bundle = MCPBundle.coerce(inputs.get("mcp_in"))
        selected = self.select(bundle, inputs.get("checked_tools"))

        self._sync_catalog = bundle.catalog()
        enabled = [row["name"] for row in self._sync_catalog
                   if selected.permits(
                       row["category"].split(":", 1)[-1], row["name"])]
        self._sync_status = (
            f"{len(enabled)}/{len(self._sync_catalog)} MCP tool(s) enabled · "
            f"{selected.status}"
        )
        return {"mcp": selected}

    # ── Main thread ───────────────────────────────────────────────────

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
        super().cleanup()
