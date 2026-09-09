# -*- coding: utf-8 -*-
"""Node-owned MCP sessions: one handshake per server (spec D19-D22).

The claims worth pinning have nothing to do with the protocol, which
`mcp_toolset.py` already speaks. They are about *ownership*: that a
session is opened once and survives every dispatch, that its tools become
ordinary ToolBox tools (so the role gate and discovery apply to them
without knowing what MCP is), that names are namespaced by server, and
that no credential value is ever written into the graph.

The server here is a fake `MCPToolset` -- the real one needs a live
process, and none of these claims are about the wire.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from silk.functions.mcp_session import (
    HTTP, STDIO, MCPServerSpec, MCPSession, MCPUnavailable, attach_mcp_tools,
    resolve_credential, tool_entries,
)
from silk.functions.tool_box import ToolBox


# ── a fake server ────────────────────────────────────────────────────────


class FakeToolset:
    """Stands in for `MCPToolset`: counts handshakes, answers calls."""

    def __init__(self, tools=None, *, fail: str = "") -> None:
        self.entered = 0
        self.exited = 0
        self.calls: list[tuple[str, dict]] = []
        self.headers: dict[str, str] = {}
        self._tools = tools or [
            SimpleNamespace(name="search", description="Search the corpus.",
                            inputSchema={"type": "object",
                                         "properties": {"q": {"type": "string"}}}),
            SimpleNamespace(name="fetch", description="Fetch a document.",
                            inputSchema=None),
        ]
        self._fail = fail

    async def __aenter__(self):
        if self._fail:
            raise RuntimeError(self._fail)
        self.entered += 1
        return self

    async def __aexit__(self, *exc):
        self.exited += 1
        return False

    async def list_tools(self):
        return self._tools

    async def direct_call_tool(self, name, args, **kwargs):
        self.calls.append((name, dict(args)))
        return f"{name}:{json.dumps(args, sort_keys=True)}"


def _session(spec: MCPServerSpec, toolset: FakeToolset) -> MCPSession:
    session = MCPSession(spec)
    session.spec.build_toolset = lambda: toolset      # type: ignore[method-assign]
    return session


@pytest.fixture
def spec() -> MCPServerSpec:
    return MCPServerSpec(id="docs", transport=STDIO, command="python",
                         args=["-m", "server"])


@pytest.fixture
def session(spec):
    fake = FakeToolset()
    live = _session(spec, fake)
    assert live.connect(timeout=5), live.error
    live.fake = fake                                   # type: ignore[attr-defined]
    yield live
    live.close()


# ── D19: one session, owned by the node ──────────────────────────────────


def test_connecting_handshakes_once(session):
    assert session.connected
    assert session.fake.entered == 1


def test_the_session_survives_many_calls(session):
    for _ in range(5):
        session.call("docs_search", {"q": "x"})
    assert session.fake.entered == 1, (
        "re-entering per call is the failure D19 exists to prevent: a "
        "stdio server would be respawned for every batch of tool calls"
    )
    assert len(session.fake.calls) == 5


def test_connect_is_idempotent(session):
    assert session.connect() is True
    assert session.fake.entered == 1


def test_close_shuts_the_session_down(session):
    session.close()
    assert not session.connected
    assert session.tools == []


def test_close_twice_is_harmless(session):
    session.close()
    session.close()


def test_calling_a_closed_session_says_so(session):
    session.close()
    with pytest.raises(MCPUnavailable) as caught:
        session.call("docs_search", {})
    assert "not connected" in str(caught.value)


def test_a_server_that_will_not_start_is_reported_not_raised(spec):
    live = _session(spec, FakeToolset(fail="no such command"))
    assert live.connect(timeout=5) is False
    assert "no such command" in live.error
    assert "no such command" in live.status
    live.close()


def test_an_incomplete_spec_never_starts_a_thread():
    live = MCPSession(MCPServerSpec(id="docs", transport=STDIO, command=""))
    assert live.connect() is False
    assert "needs a command" in live.error


def test_a_remote_spec_needs_a_url():
    assert not MCPServerSpec(id="x", transport=HTTP).is_valid()
    assert MCPServerSpec(id="x", transport=HTTP, url="http://h/mcp").is_valid()


# ── D21: namespacing ─────────────────────────────────────────────────────


def test_tools_are_prefixed_with_the_server_id(session):
    assert {tool["name"] for tool in session.tools} == {"docs_search", "docs_fetch"}


def test_the_prefix_is_stripped_before_the_call_goes_out(session):
    session.call("docs_search", {"q": "x"})
    assert session.fake.calls[0][0] == "search", (
        "the namespace is Silk's, not the server's"
    )


def test_two_servers_offering_the_same_tool_do_not_collide():
    first = tool_entries([SimpleNamespace(name="search", description="", inputSchema=None)], "a_")
    second = tool_entries([SimpleNamespace(name="search", description="", inputSchema=None)], "b_")
    assert first[0]["name"] != second[0]["name"]


def test_a_tool_without_a_schema_still_gets_one():
    entry = tool_entries([SimpleNamespace(name="t", description="", inputSchema=None)])[0]
    assert entry["parameters"] == {"type": "object", "properties": {}}


# ── D22: credentials are names, not values ───────────────────────────────


def test_a_spec_persists_the_name_and_not_the_value(monkeypatch):
    monkeypatch.setenv("DOCS_TOKEN", "s3cret")
    spec = MCPServerSpec(id="docs", transport=HTTP, url="http://h/mcp",
                         credential="DOCS_TOKEN")
    data = json.dumps(spec.to_dict())
    assert "DOCS_TOKEN" in data
    assert "s3cret" not in data, (
        "a saved graph or preset must stay shareable by construction"
    )


def test_a_resolved_credential_becomes_an_authorization_header(monkeypatch):
    monkeypatch.setenv("DOCS_TOKEN", "s3cret")
    spec = MCPServerSpec(id="docs", transport=HTTP, url="http://h/mcp",
                         credential="DOCS_TOKEN")
    assert spec.resolved_headers()["Authorization"] == "Bearer s3cret"


def test_a_server_may_name_its_own_credential_header(monkeypatch):
    monkeypatch.setenv("DOCS_TOKEN", "s3cret")
    spec = MCPServerSpec(id="docs", transport=HTTP, url="http://h/mcp",
                         credential="DOCS_TOKEN", credential_header="X-Api-Key",
                         credential_prefix="")
    assert spec.resolved_headers() == {"X-Api-Key": "s3cret"}


def test_a_credential_resolves_from_the_environment(monkeypatch):
    monkeypatch.setenv("DOCS_TOKEN", "s3cret")
    assert resolve_credential("DOCS_TOKEN") == "s3cret"


def test_a_credential_resolves_from_the_secrets_file(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets.json"
    secrets.write_text(json.dumps({"DOCS_TOKEN": "from-file"}), encoding="utf-8")
    monkeypatch.delenv("DOCS_TOKEN", raising=False)
    monkeypatch.setattr(
        "silk.functions.credentials.SECRETS_FILE", secrets)
    assert resolve_credential("DOCS_TOKEN") == "from-file"


def test_an_unreadable_secrets_file_is_not_fatal(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets.json"
    secrets.write_text("{not json", encoding="utf-8")
    monkeypatch.delenv("DOCS_TOKEN", raising=False)
    monkeypatch.setattr(
        "silk.functions.credentials.SECRETS_FILE", secrets)
    assert resolve_credential("DOCS_TOKEN") is None


def test_an_unset_credential_refuses_to_connect_and_says_where_to_put_it(monkeypatch):
    monkeypatch.delenv("DOCS_TOKEN", raising=False)
    monkeypatch.setattr(
        "silk.functions.credentials.SECRETS_FILE",
        __import__("pathlib").Path("/nonexistent/secrets.json"))
    spec = MCPServerSpec(id="docs", transport=HTTP, url="http://h/mcp",
                         credential="DOCS_TOKEN")
    with pytest.raises(MCPUnavailable) as caught:
        spec.resolved_headers()
    assert "never in the graph" in str(caught.value)


def test_a_spec_round_trips_through_plain_data():
    spec = MCPServerSpec(id="docs", transport=STDIO, command="python",
                         args=["-m", "s"], credential="TOKEN",
                         allowed_tools=["search"])
    assert MCPServerSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


def test_from_dict_tolerates_an_empty_mapping():
    assert MCPServerSpec.from_dict(None).id == "mcp"


# ── attaching: MCP tools are ordinary tools ──────────────────────────────


def test_attached_tools_are_registered_and_callable(session):
    box = ToolBox()
    attached = attach_mcp_tools(box, [session])
    assert set(attached) == {"docs_search", "docs_fetch"}
    result = box.tools["docs_search"]["executable"](q="x")
    assert result.startswith("search:")


def test_attached_tools_are_advertised_with_the_server_schema(session):
    box = ToolBox()
    attach_mcp_tools(box, [session])
    schema = box.tools["docs_search"]["definition"]["function"]["parameters"]
    assert schema["properties"]["q"]["type"] == "string"


def test_attached_tools_are_discoverable(session):
    from silk.functions.tool_discovery import discover

    box = ToolBox()
    attach_mcp_tools(box, [session])
    assert "docs_search" in {hit["name"] for hit in discover(box, "search corpus")}, (
        "MCP tools participate in discovery like any other tool (§10)"
    )


def test_attached_tools_carry_their_server_as_category_and_tag(session):
    box = ToolBox()
    attach_mcp_tools(box, [session])
    meta = box.tools["docs_search"]
    assert meta["category"] == "mcp:docs"
    assert "mcp" in meta["tags"] and "docs" in meta["tags"]


def test_attached_tools_obey_the_role_gate(session):
    from silk.functions.role import Role, RoleBinding, ToolSelector

    box = ToolBox()
    attach_mcp_tools(box, [session])
    role = Role(id="reader", selector=ToolSelector(allow_tags=frozenset({"read"})))
    binding = RoleBinding.activate(role, box)
    try:
        assert not box.role_permits("docs_search"), (
            "an MCP tool is someone else's code; it is gated like any other"
        )
        results = asyncio.run(box.execute_tool_calls_async([SimpleNamespace(
            id="c1", function=SimpleNamespace(name="docs_search", arguments="{}"),
        )]))
    finally:
        binding.deactivate()
    assert "role_denied" in results[0]["content"]


def test_the_selection_callback_turns_individual_tools_off(session):
    box = ToolBox()
    attached = attach_mcp_tools(
        box, [session], selection=lambda server, name: name.endswith("search"))
    assert attached == ["docs_search"]
    assert "docs_fetch" not in box.tools


def test_a_disconnected_session_attaches_nothing(spec):
    box = ToolBox()
    assert attach_mcp_tools(box, [MCPSession(spec)]) == []


def test_attached_tools_dispatch_through_the_toolbox(session):
    box = ToolBox()
    attach_mcp_tools(box, [session])
    results = asyncio.run(box.execute_tool_calls_async([SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name="docs_search", arguments='{"q": "x"}'),
    )]))
    assert results[0]["name"] == "docs_search"
    assert "search:" in results[0]["content"]
