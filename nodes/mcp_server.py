# -*- coding: utf-8 -*-
"""Silk MCP Node.

Owns **one live session** to one MCP server and hands it downstream as a
handle (spec D19). The node holds the connection the way the GGUF Loader
holds a model: derived ToolBoxes attach to the same open session, so one
server means one handshake no matter how many agents use it, and closing
the node closes it.

The alternative -- attaching an MCP ToolSet through the ToolBox recipe --
is what this exists to avoid. Recipes are replayed per derived ToolSet,
per agent, per graph evaluation, and external ToolSets are entered and
exited around each dispatched batch; a stdio server would be respawned
constantly and a remote one re-authenticated for every tool call.

Credentials are named, never typed in and never saved (D22): the field
holds the *name* of an environment variable or an entry in
``~/.weave/silk/secrets.json``, resolved at connect time. A graph file or
preset built from this node stays shareable by construction.

Servers chain like toolchains: whatever arrives on ``mcp_in`` is passed
along with this server added, so several MCP nodes reach one ToolBox on
one wire.
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QLineEdit

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import MCP_SERVERS_TYPE  # noqa: F401
from ..functions.mcp_session import (
    HTTP, SSE, STDIO, MCPBundle, MCPServerSpec, MCPSession,
)

log = get_logger("SilkMCPNode")


@register_node
class SilkMCPServerNode(ActiveNode):
    """One MCP server, connected once and shared by handle."""

    node_class: ClassVar[str] = "AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk MCP Server"
    node_description: ClassVar[Optional[str]] = (
        "Connects to one MCP server and shares the live session with every "
        "downstream agent — one handshake per server, not per agent."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "tools", "mcp", "llm",
    ]
    node_icon: ClassVar[Optional[str]] = "node"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1
    node_version = 1     # bump on any state-shape change (G20)

    def __init__(self, title: str = "Silk MCP Server", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        self.add_input("mcp_in", datatype="mcp_servers")   # chain input
        self.add_output("mcp", datatype="mcp_servers")

        # ── Layout & WidgetCore ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self._server_id = QLineEdit("mcp")
        form.addRow("Server id:", self._server_id)
        self._widget_core.register_widget(
            "server_id", self._server_id, role=PortRole.INPUT,
            datatype="string", default="mcp", policy=debounced(500),
            add_to_layout=False,
        )
        self.add_input("server_id", datatype="string")

        self._transport = QComboBox()
        self._transport.addItems([STDIO, HTTP, SSE])
        form.addRow("Transport:", self._transport)
        self._widget_core.register_widget(
            "transport", self._transport, role=PortRole.INPUT,
            datatype="string", default=STDIO, add_to_layout=False,
        )
        self.add_input("transport", datatype="string")

        self._command = QLineEdit("")
        self._command.setPlaceholderText("python  (stdio only)")
        form.addRow("Command:", self._command)
        self._widget_core.register_widget(
            "command", self._command, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(500),
            add_to_layout=False,
        )
        self.add_input("command", datatype="string")

        self._args = QLineEdit("")
        self._args.setPlaceholderText("-m my_server --flag")
        form.addRow("Arguments:", self._args)
        self._widget_core.register_widget(
            "args", self._args, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(500),
            add_to_layout=False,
        )
        self.add_input("args", datatype="string")

        self._url = QLineEdit("")
        self._url.setPlaceholderText("http://localhost:8000/mcp")
        form.addRow("URL:", self._url)
        self._widget_core.register_widget(
            "url", self._url, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(500),
            add_to_layout=False,
        )
        self.add_input("url", datatype="string")

        self._credential = QLineEdit("")
        self._credential.setPlaceholderText("MY_SERVER_TOKEN (a name, never a value)")
        form.addRow("Credential:", self._credential)
        self._widget_core.register_widget(
            "credential", self._credential, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(500),
            add_to_layout=False,
        )
        self.add_input("credential", datatype="string")

        self._label_status = QLabel("Not connected.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        # The one live session this node owns (D19).
        self._session: Optional[MCPSession] = None

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── Worker thread ─────────────────────────────────────────────────

    @staticmethod
    def spec_from(inputs: Dict[str, Any]) -> MCPServerSpec:
        """The server description these inputs describe (D22-safe)."""
        raw_args = str(inputs.get("args") or "").strip()
        return MCPServerSpec(
            id=str(inputs.get("server_id") or "mcp").strip() or "mcp",
            transport=str(inputs.get("transport") or STDIO).strip() or STDIO,
            command=str(inputs.get("command") or "").strip(),
            args=raw_args.split() if raw_args else [],
            url=str(inputs.get("url") or "").strip(),
            credential=str(inputs.get("credential") or "").strip(),
        )

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        upstream = MCPBundle.coerce(inputs.get("mcp_in"))
        spec = self.spec_from(inputs)

        # Reconnect only when the description actually changed. A graph
        # re-evaluates for every unrelated edit, and dropping a live
        # session on each of those would undo the whole point of D19.
        if self._session is not None:
            if self._session.spec.to_dict() == spec.to_dict() \
                    and self._session.connected:
                self._sync_status = self._session.status
                return {"mcp": upstream.with_session(self._session)}
            self._session.close()
            self._session = None

        if not spec.is_valid():
            self._sync_status = (
                "Incomplete: a stdio server needs a command, a remote one "
                "needs a URL."
            )
            return {"mcp": upstream}

        session = MCPSession(spec)
        if not session.connect():
            self._sync_status = session.status
            # The failure is reported, never raised: one unreachable
            # server must not take the graph down with it.
            return {"mcp": upstream}

        self._session = session
        self._sync_status = session.status
        return {"mcp": upstream.with_session(session)}

    # ── Main thread ───────────────────────────────────────────────────

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        if hasattr(self, "_sync_status"):
            self._widget_core.push_display("status", self._sync_status)

    def cleanup(self) -> None:
        # The session is a live subprocess or socket. Node removal is the
        # only thing that closes it, so this cannot be skipped.
        if self._session is not None:
            self._session.close()
            self._session = None
        super().cleanup()
