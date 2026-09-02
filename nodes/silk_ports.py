# -*- coding: utf-8 -*-
"""Custom port types shared by the Silk node suite.

Registered once at import (R2.5.6). Every silk node module imports its
types from here; each registration is guarded so re-imports and test
re-runs never trip the PortRegistry duplicate check.

Types:
    gguf_model        dict handle {"backend": "gguf", "model": Llama | "pool": pool}
    silk_toolbox      a live ToolBox registry instance (the full catalog)
    silk_toolset      a ToolBox restricted to a selection — the only tool
                      surface an Agent node accepts
    silk_role         a Role (declarative agent configuration)
    file_permissions  a FileGrants model (or the equivalent dict) — roots
                      plus per-path read / read_write / blocked grants,
                      validated at the port boundary (D17)
    dirpath_list      ordered list of directory paths (sandbox roots)
    toolchains        list of ToolchainEnv handles (configured executables)
"""
from __future__ import annotations

from weave.node.port_registry import PortRegistry

from ..functions.file_grants import FileGrants

def _permissions_label(value) -> str:
    """Port label for a file grant, however it is carried."""
    grants = FileGrants.coerce(value) if FileGrants.is_valid(value) else None
    if grants is None:
        return "<no permissions>"
    return f"<Permissions: {len(grants.entries)} paths>"


if "gguf_model" not in PortRegistry._by_name:
    PortRegistry.register(
        name="gguf_model",
        python_type=dict,
        color_index=232,
        type_id=None,
        default=lambda: {},
        validator=lambda v: bool(v) and isinstance(v, dict)
                            and v.get("backend") == "gguf"
                            and ("model" in v or "pool" in v),
        formatter=lambda v: "<GGUF Pool/Model>" if isinstance(v, dict) else str(v),
        casts_to={},
    )

if "silk_toolbox" not in PortRegistry._by_name:
    PortRegistry.register(
        name="silk_toolbox",
        python_type=object,
        color_index=201,
        type_id=None,
        default=lambda: None,
        validator=lambda v: v is None or hasattr(v, "execute_tool_calls_async"),
        formatter=lambda v: (
            f"<Silk ToolBox ({len(getattr(v, 'tools', {}))} tools)>"
            if v is not None else "<no toolbox>"
        ),
        casts_to={},
    )

if "silk_toolset" not in PortRegistry._by_name:
    PortRegistry.register(
        name="silk_toolset",
        python_type=object,
        color_index=205,
        type_id=None,
        default=lambda: None,
        validator=lambda v: v is None or hasattr(v, "execute_tool_calls_async"),
        formatter=lambda v: (
            f"<Silk ToolSet ({len(getattr(v, 'tools', {}))} tools)>"
            if v is not None else "<no toolset>"
        ),
        casts_to={},
    )

if "file_permissions" not in PortRegistry._by_name:
    PortRegistry.register(
        name="file_permissions",
        python_type=dict,
        color_index=214,
        type_id=None,
        default=lambda: None,
        # D17: the structure is a Pydantic model, and the port is where
        # it is validated. A malformed grant used to be discovered by the
        # sandbox behaving oddly two nodes later.
        validator=FileGrants.is_valid,
        formatter=lambda v: _permissions_label(v),
        casts_to={},
    )

if "dirpath_list" not in PortRegistry._by_name:
    PortRegistry.register(
        name="dirpath_list",
        python_type=list,
        color_index=150,
        type_id=None,
        default=lambda: [],
        validator=lambda v: v is None or (
            isinstance(v, (list, tuple)) and all(isinstance(p, str) for p in v)
        ),
        formatter=lambda v: (
            f"<{len(v)} folder(s)>" if isinstance(v, (list, tuple)) else "<no folders>"
        ),
        casts_to={},
    )

# A single dirpath connects to any dirpath_list input (wrapped in a
# one-element list) — mirrors the identical cast in weave.library.file_tree.
_dirpath = PortRegistry._by_name.get("dirpath")
_dirpath_list = PortRegistry._by_name.get("dirpath_list")
if _dirpath is not None and _dirpath_list is not None:
    _cast_key = (_dirpath.type_id, _dirpath_list.type_id)
    if _cast_key not in PortRegistry._cast_registry:
        PortRegistry._cast_registry[_cast_key] = (
            lambda p: [str(p)] if p else []
        )

if "toolchains" not in PortRegistry._by_name:
    PortRegistry.register(
        name="toolchains",
        python_type=list,
        color_index=226,
        type_id=None,
        default=lambda: [],
        validator=lambda v: v is None or (
            isinstance(v, (list, tuple))
            and all(hasattr(t, "executable") for t in v)
        ),
        formatter=lambda v: (
            f"<{len(v)} toolchain(s)>" if isinstance(v, (list, tuple))
            else "<no toolchains>"
        ),
        casts_to={},
    )

if "silk_role" not in PortRegistry._by_name:
    PortRegistry.register(
        name="silk_role",
        python_type=object,
        color_index=237,
        type_id=None,
        default=lambda: None,
        validator=lambda v: v is None or (hasattr(v, "selector") and hasattr(v, "id")),
        formatter=lambda v: (
            f"<Role: {getattr(v, 'id', '?')}>" if v is not None else "<no role>"
        ),
        casts_to={},
    )

# A single self-describing agent-to-agent message (AgentMessage.to_dict shape).
if "agent_message" not in PortRegistry._by_name:
    PortRegistry.register(
        name="agent_message",
        python_type=dict,
        color_index=141,
        type_id=None,
        default=lambda: None,
        validator=lambda v: v is None or (
            isinstance(v, dict) and "content" in v
        ),
        formatter=lambda v: (
            f"<Msg {v.get('kind', '?')} from {v.get('sender') or '?'}>"
            if isinstance(v, dict) else "<no message>"
        ),
        casts_to={},
    )

# A chainable list of AgentSpec worker bundles (model + toolset + role), fed to
# the Orchestrator node. Built up node-by-node like the toolchains chain.
if "silk_agents" not in PortRegistry._by_name:
    PortRegistry.register(
        name="silk_agents",
        python_type=list,
        color_index=99,
        type_id=None,
        default=lambda: [],
        validator=lambda v: v is None or (
            isinstance(v, (list, tuple))
            and all(hasattr(s, "model_handle") for s in v)
        ),
        formatter=lambda v: (
            f"<{len(v)} agent(s)>" if isinstance(v, (list, tuple)) else "<no agents>"
        ),
        casts_to={},
    )

GGUF_MODEL_TYPE = PortRegistry._by_name["gguf_model"]
SILK_TOOLBOX_TYPE = PortRegistry._by_name["silk_toolbox"]
SILK_TOOLSET_TYPE = PortRegistry._by_name["silk_toolset"]
SILK_ROLE_TYPE = PortRegistry._by_name["silk_role"]
FILE_PERMISSIONS_TYPE = PortRegistry._by_name["file_permissions"]
DIRPATH_LIST_TYPE = PortRegistry._by_name["dirpath_list"]
TOOLCHAINS_TYPE = PortRegistry._by_name["toolchains"]
AGENT_MESSAGE_TYPE = PortRegistry._by_name["agent_message"]
SILK_AGENTS_TYPE = PortRegistry._by_name["silk_agents"]
