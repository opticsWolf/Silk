# -*- coding: utf-8 -*-
"""Custom port types shared by the Silk node suite.

Registered once at import (R2.5.6). Every silk node module imports its
types from here; each registration is guarded so re-imports and test
re-runs never trip the PortRegistry duplicate check.

Types:
    model_handle      dict handle {"backend": <name>, "model": client |
                      "pool": pool}. One port for every backend: a local
                      llama.cpp pool ("gguf") and a remote
                      OpenAI-compatible endpoint ("openai") are the same
                      wire, because the Agent needs a client, not a
                      provenance. Renamed from `gguf_model` 2026-09-18,
                      which named one case of it.
    silk_toolbox      a live ToolBox registry instance (the full catalog)
    silk_toolset      a ToolBox restricted to a selection — the only tool
                      surface an Agent node accepts
    silk_role         a Role (declarative agent configuration)
    file_permissions  a FileGrants model (or the equivalent dict) — roots
                      plus per-path read / read_write / blocked grants,
                      validated at the port boundary (D17)
    dirpath_list      ordered list of directory paths (sandbox roots)
    toolchains        list of ToolchainEnv handles (configured executables)
    mcp_servers       an MCPBundle: live MCP sessions plus the tools the
                      Aggregator switched off (D19-D21)
    silk_plan         a PlanRef: which plan file, named outright rather
                      than discovered by mtime (D23)
    silk_classifier   a ClassifierHandle: the Classifier node's backend chain,
                      spec and meter; each run takes ``for_run`` (D-S8)
    classifier_model  a decision backend as a handle: a Classifier Loader's
                      pool (``backend: systemone`` + ``pool``) or a Jev
                      Loader's hosted endpoint (``backend: systemone`` +
                      ``url``). Answers typed questions; never chats.
                      The Classifier and Ensemble nodes take these, never
                      a chat ``model_handle``.
    okf_graph         an OkfGraph: a served OKFgraph knowledge graph, owned
                      by the Knowledge Graph node
    onnx_embedder     an ONNX embedding model *described* (embroider id,
                      device, precision, cache dir) for okf serve to load
                      in its own process; never loaded in Weave's
    reranker_model    a reranker described: an embroider cross-encoder,
                      or an external /v1/rerank host by URL and family
    endpoint          provider info (provider key, base URL, credential
                      *name*) from an Endpoint node -- what a Model node
                      connects through. A handle to a configuration, not
                      to a connection: nothing is probed until the Model
                      node asks.
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


def _model_handle_label(value) -> str:
    """Port label for a model handle: which backend, and which model.

    The backend is named because the difference matters to whoever reads
    the canvas -- a local pool and a hosted endpoint have different costs,
    different failure modes and, for a paid endpoint, a different bill.
    """
    if not isinstance(value, dict):
        return str(value)
    backend = str(value.get("backend") or "?")
    alias = value.get("model_alias") or value.get("model_path") or ""
    name = str(alias).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return f"<{backend}: {name}>" if name else f"<{backend} model>"


if "model_handle" not in PortRegistry._by_name:
    PortRegistry.register(
        name="model_handle",
        python_type=dict,
        color_index=232,
        type_id=None,
        default=lambda: {},
        # A backend *name* rather than a fixed one: the port's job is to
        # refuse a handle nothing can generate from, not to keep a list of
        # who may generate. A client lives under "model" (one client, the
        # remote case) or "pool" (checked out per session, the local one).
        # Absence passes: an empty dict is what ``default()`` returns and
        # ``None`` is what an unwired port delivers, so refusing them
        # would make the type call every disconnected instance of itself
        # invalid. Only a *populated* handle is held to the rest, which
        # is the same shape ``silk_toolbox`` below uses for ``None``.
        # ``functions/model_fallback.py`` asks a stricter question -- is
        # this runnable -- and correctly rejects the empty case.
        validator=lambda v: v is None or (isinstance(v, dict)
                            and (not v or (bool(v.get("backend"))
                                 and ("model" in v or "pool" in v)))),
        formatter=_model_handle_label,
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

# A chainable list of WorkerSpec bundles (model + toolset + role), fed to the
# Orchestrator node. Built up node-by-node like the toolchains chain.
#
# Named for what the Orchestrator calls them -- its input port is `workers`
# and its tool is `list_workers` -- so the wire reads the same at both ends.
# It used to be `silk_agents`, which made a Worker node look like something
# an Agent node would accept. It is not: nothing but an Orchestrator takes
# this type.
if "silk_workers" not in PortRegistry._by_name:
    PortRegistry.register(
        name="silk_workers",
        python_type=list,
        color_index=99,
        type_id=None,
        default=lambda: [],
        validator=lambda v: v is None or (
            isinstance(v, (list, tuple))
            and all(hasattr(s, "model_handle") for s in v)
        ),
        formatter=lambda v: (
            f"<{len(v)} worker(s)>" if isinstance(v, (list, tuple))
            else "<no workers>"
        ),
        casts_to={},
    )

# The live MCP servers on one wire, plus whichever of their tools the
# Aggregator switched off (D19-D21). The value is an MCPBundle carrying
# open sessions -- handles, not settings: what travels the graph is a
# connection that already exists, which is the whole of D19.
if "mcp_servers" not in PortRegistry._by_name:
    PortRegistry.register(
        name="mcp_servers",
        python_type=object,
        color_index=45,
        type_id=None,
        default=lambda: None,
        validator=lambda v: v is None or hasattr(v, "enabled_sessions"),
        formatter=lambda v: (
            f"<MCP: {len(v.enabled_sessions())} server(s)>"
            if v is not None and hasattr(v, "enabled_sessions")
            else "<no MCP servers>"
        ),
        casts_to={},
    )

# Which plan, and where it lives (D23). A PlanRef -- naming the plan file
# rather than letting the store pick the newest one under a root, which is
# how two unrelated plans in one directory used to cross-discover.
if "silk_plan" not in PortRegistry._by_name:
    PortRegistry.register(
        name="silk_plan",
        python_type=object,
        color_index=178,
        type_id=None,
        default=lambda: None,
        validator=lambda v: v is None or hasattr(v, "is_explicit")
                            or isinstance(v, dict),
        formatter=lambda v: f"<Plan {v}>" if v is not None else "<no plan>",
        casts_to={},
    )

# The classifier resource (D-S8): a ClassifierHandle from a Classifier node,
# consumed by the Agent node next to model_obj. Unwired means humans decide.
if "silk_classifier" not in PortRegistry._by_name:
    PortRegistry.register(
        name="silk_classifier",
        python_type=object,
        color_index=118,
        type_id=None,
        default=lambda: None,
        validator=lambda v: v is None or (hasattr(v, "for_run") and hasattr(v, "spec")),
        formatter=lambda v: (
            f"<Classifier: {getattr(getattr(v, 'spec', None), 'mode', '?')}>"
            if v is not None else "<no classifier>"
        ),
        casts_to={},
    )

# A served knowledge graph: an OkfGraph from the Knowledge Graph node, which
# owns the `okf serve` process (or attached to one already running). A
# handle like mcp_servers -- the ToolBox registers tools over it and never
# starts a server of its own.
if "okf_graph" not in PortRegistry._by_name:
    PortRegistry.register(
        name="okf_graph",
        python_type=object,
        color_index=104,
        type_id=None,
        default=lambda: None,
        validator=lambda v: v is None or (hasattr(v, "call") and hasattr(v, "spec")),
        formatter=lambda v: (
            f"<Knowledge graph: {getattr(getattr(v, 'spec', None), 'db_path', '?')}>"
            if v is not None else "<no knowledge graph>"
        ),
        casts_to={},
    )

# An ONNX embedding model as a *description*: which embroider model, on
# which device and precision, cached where. Not a model_handle on purpose:
# a model_handle is a model live in Weave's process, and an ONNX model
# must never load there (a second ONNX Runtime beside one already running
# takes the interpreter down). The consumer -- the Knowledge Graph node --
# hands it to `okf serve`, which loads it with embroider in its own process.
if "onnx_embedder" not in PortRegistry._by_name:
    PortRegistry.register(
        name="onnx_embedder",
        python_type=dict,
        color_index=122,
        type_id=None,
        default=lambda: None,
        validator=lambda v: v is None or (isinstance(v, dict)
                            and (not v or (v.get("runtime") == "embroider"
                                 and v.get("role") == "embedding"))),
        formatter=lambda v: (
            f"<ONNX embedder: {v.get('model_id') or 'default'} "
            f"({v.get('device', 'auto')})>"
            if isinstance(v, dict) and v else "<no ONNX embedder>"
        ),
        casts_to={},
    )

# A reranker (the second retrieval stage) as a description, whoever runs
# it: ``runtime: embroider`` is a cross-encoder okf loads itself (the ONNX
# Reranker node); ``runtime: external`` is a ``POST /v1/rerank`` host
# (llama-server --rerank, TEI) by URL and declared family. Typed by role,
# not by runtime, so one port takes either.
if "reranker_model" not in PortRegistry._by_name:
    PortRegistry.register(
        name="reranker_model",
        python_type=dict,
        color_index=126,
        type_id=None,
        default=lambda: None,
        validator=lambda v: v is None or (isinstance(v, dict)
                            and (not v or (v.get("role") == "rerank" and (
                                v.get("runtime") == "embroider"
                                or (v.get("runtime") == "external"
                                    and bool(v.get("endpoint_url"))))))),
        formatter=lambda v: (
            f"<Reranker: {v.get('model_id') or 'default'} "
            f"({v.get('runtime', '?')})>"
            if isinstance(v, dict) and v else "<no reranker>"
        ),
        casts_to={},
    )

ONNX_EMBEDDER_TYPE = PortRegistry._by_name["onnx_embedder"]
RERANKER_MODEL_TYPE = PortRegistry._by_name["reranker_model"]

MODEL_HANDLE_TYPE = PortRegistry._by_name["model_handle"]
# Provider info on a wire: what an Endpoint node selected, what a Model
# node connects through. The credential travels by name (D22); the Model
# node resolves it at connect time. Absence passes for the same reason it
# does on model_handle: an unwired port must not invalidate the node.
if "endpoint" not in PortRegistry._by_name:
    PortRegistry.register(
        name="endpoint",
        python_type=dict,
        color_index=81,
        type_id=None,
        default=lambda: None,
        validator=lambda v: v is None or (isinstance(v, dict)
                            and (not v or (bool(v.get("provider"))
                                 and bool(v.get("base_url"))))),
        formatter=lambda v: (
            f"<Endpoint: {v.get('provider', '?')} @ "
            f"{str(v.get('base_url') or '?').split('://', 1)[-1]}>"
            if isinstance(v, dict) and v else "<no endpoint>"
        ),
        casts_to={},
    )

# A decision backend on a wire: what a Classifier Loader or a Jev Loader
# emits, what the Classifier and Ensemble nodes take. Stricter than
# model_handle on purpose: a chat client or pool can never land where a
# typed-questions backend is expected. Absence passes for the same reason
# it does on model_handle: an unwired port must not invalidate the node.
if "classifier_model" not in PortRegistry._by_name:
    PortRegistry.register(
        name="classifier_model",
        python_type=dict,
        color_index=172,
        type_id=None,
        default=lambda: None,
        validator=lambda v: v is None or (isinstance(v, dict)
                            and (not v or (v.get("backend") == "systemone"
                                 and v.get("role") == "decision"
                                 and ("pool" in v or "url" in v)))),
        formatter=lambda v: (
            f"<Classifier model: {v.get('model') or 'local pool'}>"
            if isinstance(v, dict) and v else "<no classifier model>"
        ),
        casts_to={},
    )

SILK_TOOLBOX_TYPE = PortRegistry._by_name["silk_toolbox"]
SILK_TOOLSET_TYPE = PortRegistry._by_name["silk_toolset"]
SILK_ROLE_TYPE = PortRegistry._by_name["silk_role"]
FILE_PERMISSIONS_TYPE = PortRegistry._by_name["file_permissions"]
DIRPATH_LIST_TYPE = PortRegistry._by_name["dirpath_list"]
TOOLCHAINS_TYPE = PortRegistry._by_name["toolchains"]
AGENT_MESSAGE_TYPE = PortRegistry._by_name["agent_message"]
SILK_WORKERS_TYPE = PortRegistry._by_name["silk_workers"]
MCP_SERVERS_TYPE = PortRegistry._by_name["mcp_servers"]
SILK_PLAN_TYPE = PortRegistry._by_name["silk_plan"]
SILK_CLASSIFIER_TYPE = PortRegistry._by_name["silk_classifier"]
OKF_GRAPH_TYPE = PortRegistry._by_name["okf_graph"]
ENDPOINT_HANDLE_TYPE = PortRegistry._by_name["endpoint"]
CLASSIFIER_MODEL_TYPE = PortRegistry._by_name["classifier_model"]
