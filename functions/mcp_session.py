# -*- coding: utf-8 -*-
"""One live MCP session per server, owned by a node (spec D19-D22).

`mcp_toolset.py` already speaks the protocol. What was missing is *where
the connection lives*, and the answer matters more than it looks.

ToolSets are derived per agent by replaying the ToolBox's `build_recipe`
attachers, and external ToolSets are entered and exited around each
dispatched batch. An MCP server attached that way would re-handshake per
agent per graph evaluation -- start a stdio subprocess, initialize, list
tools, tear it down, for every batch of tool calls. That is the sharpest
problem in this area and D19 is the answer to it: the **node** owns one
session per server, and everything downstream attaches to it by handle.

So a session here is not a ToolSet. It is a connection on its own thread
with its own event loop, and the tools it exposes are registered on the
ToolBox as ordinary tools whose executables call into that thread. The
handshake happens once, when the node connects. Replaying a recipe costs
a dict copy. And because they are ordinary registered tools, MCP tools
get the rest of Silk for free: the role gate, the approval hook, spill,
and discovery (§6) all work on them without knowing what MCP is.

Two smaller rules ride along:

**Namespacing (D21).** Every tool is prefixed with its server id, so two
servers offering `search` cannot collide and the model can see where a
tool comes from.

**Credentials are never persisted (D22).** A spec stores a credential
*name*. The value is resolved at connect time from the environment or
from `~/.weave/silk/secrets.json`, which is outside the graph -- so saved
graphs and presets stay shareable by construction.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from weave.logger import get_logger


log = get_logger("SilkMCP")

#: Where a credential value may live when it is not in the environment.
#: Outside the graph on purpose (D22).
SECRETS_FILE = Path.home() / ".weave" / "silk" / "secrets.json"

#: How long a tool call may block the dispatcher before it is abandoned.
#: A hung server must not hang the agent thread.
DEFAULT_CALL_TIMEOUT = 120.0

#: How long connecting may take before the node reports failure.
DEFAULT_CONNECT_TIMEOUT = 30.0

STDIO = "stdio"
HTTP = "http"
SSE = "sse"
TRANSPORTS = (STDIO, HTTP, SSE)


class MCPUnavailable(RuntimeError):
    """The MCP client library is not installed, or a server would not talk."""


def resolve_credential(name: str) -> Optional[str]:
    """The value behind a credential *name*, or ``None`` (D22).

    Environment first, then the secrets file. Nothing here writes, and the
    value is never handed back to the graph -- only used to build the
    headers of a connection.
    """
    if not name:
        return None
    value = os.environ.get(name)
    if value:
        return value
    try:
        if SECRETS_FILE.is_file():
            data = json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                found = data.get(name)
                if isinstance(found, str) and found:
                    return found
    except (OSError, ValueError) as exc:
        log.warning(f"Could not read {SECRETS_FILE}: {exc}")
    return None


@dataclass
class MCPServerSpec:
    """How to reach one MCP server. Data only -- no secret ever lands here.

    ``credential`` is the *name* of a credential, not its value: that is
    the whole of D22. It is resolved at connect time and put in an
    ``Authorization`` header (or ``credential_header``, for servers that
    want their own).
    """

    id: str = "mcp"
    transport: str = STDIO
    #: stdio
    command: str = ""
    args: list[str] = field(default_factory=list)
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    #: http / sse
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    #: D22: a name, resolved at connect time, never stored.
    credential: str = ""
    credential_header: str = "Authorization"
    credential_prefix: str = "Bearer "
    #: Optional allow-list applied at the server, before the role gate.
    allowed_tools: Optional[list[str]] = None

    @property
    def prefix(self) -> str:
        """The namespace every tool from this server carries (D21)."""
        return f"{self.id}_" if self.id else ""

    def is_valid(self) -> bool:
        if self.transport == STDIO:
            return bool(self.command)
        return bool(self.url)

    def to_dict(self) -> dict:
        """Plain data, safe to persist: the credential is only a name."""
        return {
            "id": self.id, "transport": self.transport,
            "command": self.command, "args": list(self.args),
            "cwd": self.cwd, "env": dict(self.env),
            "url": self.url, "headers": dict(self.headers),
            "credential": self.credential,
            "credential_header": self.credential_header,
            "credential_prefix": self.credential_prefix,
            "allowed_tools": list(self.allowed_tools)
                             if self.allowed_tools is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MCPServerSpec":
        if isinstance(data, MCPServerSpec):
            return data
        data = data or {}
        allowed = data.get("allowed_tools")
        return cls(
            id=str(data.get("id") or "mcp"),
            transport=str(data.get("transport") or STDIO),
            command=str(data.get("command") or ""),
            args=[str(a) for a in (data.get("args") or [])],
            cwd=str(data.get("cwd") or ""),
            env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
            url=str(data.get("url") or ""),
            headers={str(k): str(v)
                     for k, v in (data.get("headers") or {}).items()},
            credential=str(data.get("credential") or ""),
            credential_header=str(data.get("credential_header")
                                  or "Authorization"),
            credential_prefix=str(data.get("credential_prefix")
                                  if data.get("credential_prefix") is not None
                                  else "Bearer "),
            allowed_tools=[str(a) for a in allowed] if allowed else None,
        )

    def resolved_headers(self) -> dict:
        """Connection headers, with the credential resolved (D22).

        Separate from :meth:`build_toolset` because it is the half that is
        pure configuration: it can be checked, and its failure explained,
        without an MCP client library being installed at all.
        """
        headers = dict(self.headers)
        secret = resolve_credential(self.credential)
        if self.credential and not secret:
            raise MCPUnavailable(
                f"Credential '{self.credential}' is not set. Put it in the "
                f"environment, or in {SECRETS_FILE} -- never in the graph."
            )
        if secret:
            headers[self.credential_header] = f"{self.credential_prefix}{secret}"
        return headers

    def build_toolset(self) -> Any:
        """The `MCPToolset` this spec describes, connected to nothing yet."""
        from .mcp_toolset import MCP_AVAILABLE, MCPToolset, StdioTransport

        if not MCP_AVAILABLE:
            raise MCPUnavailable(
                "The 'mcp' package is not installed, so MCP servers cannot "
                "be reached. Install it, or remove the MCP node."
            )
        headers = self.resolved_headers()

        if self.transport == STDIO:
            client = StdioTransport(
                command=self.command,
                args=list(self.args),
                env=dict(self.env) or None,
                cwd=self.cwd or None,
            )
        else:
            client = self.url
        return MCPToolset(
            client, id=self.id, headers=headers,
            allowed_tools=list(self.allowed_tools) if self.allowed_tools else None,
        )


def tool_entries(tools: Any, prefix: str = "") -> list[dict]:
    """MCP tool objects as plain data, namespaced (D21).

    Plain data because this crosses to the UI thread, and because the
    ToolBox wants a schema dict, not a protocol object.
    """
    entries: list[dict] = []
    for tool in tools or []:
        name = getattr(tool, "name", None) or (
            tool.get("name") if isinstance(tool, dict) else None)
        if not name:
            continue
        description = getattr(tool, "description", None) or (
            tool.get("description") if isinstance(tool, dict) else "") or ""
        schema = getattr(tool, "inputSchema", None) or (
            tool.get("inputSchema") if isinstance(tool, dict) else None)
        entries.append({
            "name": f"{prefix}{name}",
            "server_name": name,
            "description": description,
            "parameters": schema or {"type": "object", "properties": {}},
        })
    return entries


class MCPSession:
    """One connection to one server, alive for as long as the node is.

    The session runs its own event loop on its own thread. That is not
    decoration: the agent dispatches tool batches inside `asyncio.run`,
    a fresh loop each time, and an MCP session belongs to the loop that
    opened it. Keeping the loop alive on a thread of its own is what lets
    one handshake serve every batch, every agent, and every evaluation
    (D19).
    """

    def __init__(self, spec: MCPServerSpec) -> None:
        self.spec = spec
        self.tools: list[dict] = []
        self.error: str = ""
        self._toolset: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._closing = False
        self._participant = None

    # -- lifecycle -------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._toolset is not None and not self._closing

    @property
    def status(self) -> str:
        if self.error:
            return f"{self.spec.id}: {self.error}"
        if self.connected:
            return f"{self.spec.id}: connected, {len(self.tools)} tool(s)"
        return f"{self.spec.id}: not connected"

    def connect(self, timeout: float = DEFAULT_CONNECT_TIMEOUT) -> bool:
        """Open the session and list its tools once. Idempotent."""
        if self.connected:
            return True
        if not self.spec.is_valid():
            self.error = ("incomplete server configuration: a stdio server "
                          "needs a command, a remote one needs a URL")
            return False

        self.error = ""
        self._closing = False
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._serve, name=f"silk-mcp-{self.spec.id}", daemon=True,
        )
        self._thread.start()
        self._register_for_shutdown()
        if not self._ready.wait(timeout):
            self.error = f"timed out after {timeout:g}s while connecting"
            self.close()
            return False
        return self.connected

    def _serve(self) -> None:
        """The session's whole life, on its own loop."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            toolset = self.spec.build_toolset()
            loop.run_until_complete(toolset.__aenter__())
            self.tools = tool_entries(
                loop.run_until_complete(toolset.list_tools()),
                self.spec.prefix,
            )
            self._toolset = toolset
            log.info(f"MCP '{self.spec.id}' connected: "
                     f"{len(self.tools)} tool(s)")
        except Exception as exc:      # noqa: BLE001 - reported, never raised
            self.error = str(exc) or exc.__class__.__name__
            log.warning(f"MCP '{self.spec.id}' failed to connect: {exc}")
            self._ready.set()
            loop.close()
            self._loop = None
            return

        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                if self._toolset is not None:
                    loop.run_until_complete(self._toolset.__aexit__(None, None, None))
            except Exception as exc:      # noqa: BLE001 - shutting down anyway
                log.debug(f"MCP '{self.spec.id}' close: {exc}")
            self._toolset = None
            loop.close()
            self._loop = None

    def _register_for_shutdown(self) -> None:
        """Stop this server before the process hands off (spec D80).

        A stdio server is a subprocess of *this* process. It re-resolves
        its credentials at connect (D22), so a relaunched Weave
        reconnects cleanly -- provided the outgoing one actually stopped
        its servers rather than leaving them parented to nothing.
        """
        if self._participant is not None:
            return
        try:
            from weave.engine.shutdown import (
                get_shutdown_registry, install_shutdown_handlers,
            )
        except Exception:  # noqa: BLE001 - a plugin must not need the host
            return
        install_shutdown_handlers()
        self._participant = get_shutdown_registry().register(
            f"MCP server '{self.spec.id}'",
            lambda timeout_s=0.0: (self.close(), True)[1],
        )

    def close(self) -> None:
        """Close the session. Safe to call twice, and from any thread."""
        self._closing = True
        loop, thread = self._loop, self._thread
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive() \
                and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._thread = None
        self.tools = []
        if self._participant is not None:
            try:
                from weave.engine.shutdown import get_shutdown_registry

                get_shutdown_registry().unregister(self._participant)
            except Exception:  # noqa: BLE001 - closing anyway
                pass
            self._participant = None

    # -- calling ---------------------------------------------------------

    def call(self, name: str, args: dict,
             timeout: float = DEFAULT_CALL_TIMEOUT) -> Any:
        """Run one tool call on the session's loop and wait for it.

        Called from the agent's worker thread, which is why this is a
        blocking bridge rather than a coroutine: the dispatcher already
        runs it off the UI thread, and a hung server is bounded by
        *timeout* rather than by the agent's patience.
        """
        if not self.connected or self._loop is None:
            raise MCPUnavailable(
                f"MCP server '{self.spec.id}' is not connected"
                + (f": {self.error}" if self.error else "")
            )
        server_name = name
        if self.spec.prefix and name.startswith(self.spec.prefix):
            server_name = name[len(self.spec.prefix):]
        future = asyncio.run_coroutine_threadsafe(
            self._toolset.direct_call_tool(server_name, dict(args or {})),
            self._loop,
        )
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise MCPUnavailable(
                f"'{name}' did not answer within {timeout:g}s. The server "
                f"may be busy or wedged; try again, or work without it."
            ) from None


def attach_mcp_tools(
    toolbox: Any,
    sessions: Any,
    *,
    selection: Optional[Callable[[str, str], bool]] = None,
    risk: str = "medium",
) -> list[str]:
    """Register every enabled MCP tool on *toolbox*. Returns their names.

    Registration is flat and by hand rather than through `add_toolset`,
    for the reason at the top of this module: an external ToolSet is
    entered and exited around each dispatched batch, and this must not
    re-handshake. What lands on the ToolBox is an ordinary tool whose
    executable talks to a session that is already open.

    *selection* answers ``(server_id, tool_name) -> bool`` and is how the
    Aggregator's checkbox tree turns individual tools off (D20).
    """
    attached: list[str] = []
    for session in sessions or []:
        if not session.connected:
            continue
        server = session.spec.id
        for entry in session.tools:
            name = entry["name"]
            if selection is not None and not selection(server, name):
                continue
            definition = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": entry.get("description", ""),
                    "parameters": entry.get("parameters")
                                  or {"type": "object", "properties": {}},
                },
            }
            toolbox.tools[name] = {
                "definition": definition,
                "args_model": None,
                "executable": _executable(session, name),
                "is_async": False,
                "procedure": None,
                "source": f"mcp:{server}",
                "timeout": None,
                # An MCP server is someone else's code reached over a wire.
                # It gets a risk band that the approval gate can see, and
                # the role gate treats it like any other tool.
                "requires_approval": False,
                "sequential": False,
                "tags": frozenset({"mcp", server}),
                "category": f"mcp:{server}",
                "risk": risk,
            }
            toolbox.tool_search.register_tool(name, definition)
            attached.append(name)
    if attached:
        log.info(f"Attached {len(attached)} MCP tool(s): "
                 f"{', '.join(attached[:8])}"
                 + (" ..." if len(attached) > 8 else ""))
    return attached


def _executable(session: MCPSession, name: str) -> Callable[..., Any]:
    def _call(**kwargs: Any) -> Any:
        return session.call(name, kwargs)
    return _call


@dataclass
class MCPBundle:
    """The live servers travelling one wire, plus what is switched off.

    One shape for the whole chain (D20): an MCP node appends its session
    to whatever arrived on its input, and the Aggregator adds nothing but
    exclusions. Carrying the sessions and the selection together is what
    lets a ToolBox attach exactly what the checkbox tree says without
    knowing which node made the decision.

    Exclusions are stored rather than a filtered list, because a server
    that is temporarily unchecked must keep its session -- unchecking a
    tool is not a reason to tear down a connection and re-handshake it
    when the box is ticked again.
    """

    sessions: list = field(default_factory=list)
    disabled_servers: set = field(default_factory=set)
    disabled_tools: set = field(default_factory=set)

    @classmethod
    def coerce(cls, value: Any) -> "MCPBundle":
        """Whatever arrived on an ``mcp_servers`` port, as a bundle."""
        if isinstance(value, MCPBundle):
            return value
        if value is None:
            return cls()
        if isinstance(value, MCPSession):
            return cls(sessions=[value])
        if isinstance(value, (list, tuple)):
            return cls(sessions=[s for s in value if isinstance(s, MCPSession)])
        return cls()

    def with_session(self, session: "MCPSession") -> "MCPBundle":
        """This bundle plus *session*, replacing any server of the same id.

        Replacing rather than appending is what makes re-evaluation safe:
        a node that reconnects emits a new session for the same id, and
        two sessions for one server would double every tool it offers.
        """
        kept = [s for s in self.sessions if s.spec.id != session.spec.id]
        return MCPBundle(
            sessions=[*kept, session],
            disabled_servers=set(self.disabled_servers),
            disabled_tools=set(self.disabled_tools),
        )

    def permits(self, server: str, tool: str) -> bool:
        """The selection predicate `attach_mcp_tools` asks (D20)."""
        return server not in self.disabled_servers \
            and tool not in self.disabled_tools

    def enabled_sessions(self) -> list:
        return [s for s in self.sessions
                if s.spec.id not in self.disabled_servers]

    def catalog(self) -> list[dict]:
        """Every tool on the wire as plain-data catalog rows.

        The shape `widgets/tool_tree.py` already renders, grouped by
        server so the tree's category rows become per-server switches.
        """
        rows: list[dict] = []
        for session in self.sessions:
            for entry in session.tools:
                rows.append({
                    "name": entry["name"],
                    "description": entry.get("description", ""),
                    "parameters": entry.get("parameters") or {},
                    "category": f"mcp:{session.spec.id}",
                    "tags": ["mcp", session.spec.id],
                    "risk": "medium",
                })
        return rows

    @property
    def status(self) -> str:
        if not self.sessions:
            return "No MCP servers connected."
        parts = [s.status for s in self.sessions]
        off = len(self.disabled_tools) + len(self.disabled_servers)
        return " · ".join(parts) + (f" · {off} disabled" if off else "")


def attach_bundle(bundle: Any) -> Callable:
    """A `build_recipe` attacher for the servers on one wire.

    The recipe is replayed per derived ToolSet, per agent, per graph
    evaluation, which is exactly why this closes over *live sessions*
    instead of connection settings: replaying it registers dict entries
    and touches no server (D19).
    """
    resolved = MCPBundle.coerce(bundle)

    def _attach(toolbox: Any, _sandbox: Any = None) -> None:
        attach_mcp_tools(
            toolbox, resolved.enabled_sessions(), selection=resolved.permits,
        )

    return _attach
