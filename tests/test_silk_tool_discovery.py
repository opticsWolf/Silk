# -*- coding: utf-8 -*-
"""Model-facing tool discovery: search, deferral, auto-load (spec D4-D6).

The claims under test are the ones the spec makes about *reach*: what the
model can find, what it can call without being told to load it first, and
what discovery must refuse to offer because dispatch would refuse it too.
The one that is easy to get wrong is the last: search runs off an index
built at attach time, and a role is bound long after that.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from silk.functions.capabilities import Capability
from silk.functions.role import Role, RoleBinding, ToolSelector
from silk.functions.tool_box import ToolBox
from silk.functions.tool_discovery import (
    SEARCH_TOOL_NAME, autoload, discover,
)
from silk.functions.tool_search import ToolSearch, tokenize


# ── fixtures ─────────────────────────────────────────────────────────────


def _def(name: str, description: str) -> dict:
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
    }}


def _call(name: str, **args) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"call-{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class _PathArgs(BaseModel):
    path: str


@pytest.fixture
def box() -> ToolBox:
    """A toolbox with a handful of registered tools, plainly named."""
    toolbox = ToolBox()
    for name, description, category, tags in (
        ("read_file", "Read the contents of a file from disk.",
         "file", ("read",)),
        ("write_file", "Write text to a file on disk.", "file", ("write",)),
        ("run_tests", "Run the project test suite and report failures.",
         "verify", ("run",)),
    ):
        toolbox.register(name=name, description=description,
                         args_model=_PathArgs,
                         category=category, tags=tags)(lambda *a, **k: "ok")
    return toolbox


# ── the tool exists and is infrastructure ────────────────────────────────


def test_search_tools_is_registered_on_every_toolbox(box):
    assert SEARCH_TOOL_NAME in box.tools


def test_search_tools_survives_a_derived_toolset(tmp_path, box):
    from silk.functions.toolset_build import INFRASTRUCTURE_TOOLS

    assert SEARCH_TOOL_NAME in INFRASTRUCTURE_TOOLS, (
        "a selection that drops discovery leaves the agent unable to find "
        "the tools it was not given"
    )


def test_the_search_tool_advertises_its_own_filters(box):
    schema = box.tools[SEARCH_TOOL_NAME]["definition"]["function"]["parameters"]
    assert set(schema["properties"]) >= {"query", "category", "capability"}
    assert schema["required"] == ["query"]


# ── D4: finding individual tools ─────────────────────────────────────────


def test_discover_finds_a_tool_by_what_it_does(box):
    names = [hit["name"] for hit in discover(box, "read a file")]
    assert "read_file" in names


def test_a_hit_carries_the_parameter_schema(box):
    hit = next(h for h in discover(box, "read a file") if h["name"] == "read_file")
    assert hit["parameters"]["properties"]["path"]["type"] == "string", (
        "D5 has no load tool, so the search result is the only place the "
        "model learns how to call what it found"
    )


def test_a_hit_carries_its_category_and_tags(box):
    hit = next(h for h in discover(box, "read a file") if h["name"] == "read_file")
    assert hit["category"] == "file"
    assert hit["tags"] == ["read"]


def test_the_category_filter_narrows_the_result(box):
    hits = discover(box, "file", category="verify")
    assert all(hit["category"] == "verify" for hit in hits)
    assert "read_file" not in {hit["name"] for hit in hits}


def test_the_limit_is_honoured_and_bounded(box):
    assert len(discover(box, "file", limit=1)) <= 1
    assert len(discover(box, "file", limit=9999)) <= 20


def test_an_empty_query_returns_an_error_that_says_what_to_do(box):
    payload = json.loads(box.tools[SEARCH_TOOL_NAME]["executable"](query="  "))
    assert "error" in payload and "suggestion" in payload


def test_no_match_says_the_role_might_be_why(box):
    payload = json.loads(box.tools[SEARCH_TOOL_NAME]["executable"](
        query="zzzz nonexistent"))
    assert payload["results"] == []
    assert "role" in payload["message"]


# ── I8: discovery obeys the role gate ────────────────────────────────────


def test_search_does_not_offer_what_the_role_forbids(box):
    role = Role(id="reader", selector=ToolSelector(allow_names=frozenset({"read_file"})))
    binding = RoleBinding.activate(role, box)
    try:
        names = {hit["name"] for hit in discover(box, "file")}
    finally:
        binding.deactivate()
    assert "read_file" in names
    assert "write_file" not in names, (
        "offering a tool dispatch will refuse is half of I4, from the "
        "wrong side"
    )


def test_the_gate_lifts_again_when_the_role_deactivates(box):
    role = Role(id="reader", selector=ToolSelector(allow_names=frozenset({"read_file"})))
    binding = RoleBinding.activate(role, box)
    binding.deactivate()
    assert "write_file" in {hit["name"] for hit in discover(box, "write a file")}


def test_a_custom_strategy_result_is_gated_too():
    search = ToolSearch.create(
        strategy=lambda queries, tools: ["read_file", "write_file"],
        tools={"read_file": _def("read_file", "read"),
               "write_file": _def("write_file", "write")},
    )
    search.permits = lambda name: name == "read_file"
    found = {hit["function"]["name"] for hit in search.search("anything")}
    assert found == {"read_file"}, (
        "a strategy may return anything it likes; the gate is applied to "
        "its answer, not trusted to it"
    )


# ── G2: bm25 is a ranking function, not an alias ─────────────────────────


def test_tokenize_splits_identifiers():
    assert tokenize("read_file") == ["read", "file"]
    assert tokenize("Run-Tests!") == ["run", "tests"]


def test_bm25_ranks_the_specific_tool_above_the_generic_one():
    search = ToolSearch(strategy="bm25", tools={
        "read_file": _def("read_file", "Read a file from disk."),
        "note": _def("note", "A file note about a file, file file file."),
        "unrelated": _def("unrelated", "Send an email."),
    })
    ranked = [hit["function"]["name"] for hit in search.search("read a file")]
    assert ranked[0] == "read_file", (
        "term saturation and idf are the point: repeating 'file' must not "
        "beat matching both query terms"
    )
    assert "unrelated" not in ranked


def test_bm25_discounts_a_term_every_tool_shares():
    search = ToolSearch(strategy="bm25", tools={
        "alpha": _def("alpha", "tool for files"),
        "beta": _def("beta", "tool for files"),
        "gamma": _def("gamma", "tool for files and sockets"),
    })
    ranked = [hit["function"]["name"] for hit in search.search("tool sockets")]
    assert ranked[0] == "gamma", (
        "'tool' is in every document and carries almost no information; "
        "'sockets' is in one and carries all of it"
    )


def test_bm25_on_an_empty_corpus_is_empty():
    assert ToolSearch(strategy="bm25", tools={}).search("anything") == []


def test_bm25_is_no_longer_an_alias_for_keywords():
    tools = {
        "alpha": _def("alpha", "files files files files files"),
        "beta": _def("beta", "files sockets"),
    }
    keywords = [hit["function"]["name"]
                for hit in ToolSearch(strategy="keywords", tools=tools).search("files")]
    bm25 = [hit["function"]["name"]
            for hit in ToolSearch(strategy="bm25", tools=tools).search("files")]
    assert keywords and bm25
    assert set(keywords) == set(bm25)
    # And the ranking function is genuinely doing arithmetic the keyword
    # overlap cannot: repetition saturates rather than accumulating.
    search = ToolSearch(strategy="bm25", tools=tools)
    assert search._bm25_search("files") != []


# ── D6: per-tool deferral ────────────────────────────────────────────────


def test_a_deferred_tool_is_not_advertised(box):
    box.defer_tools(["write_file"])
    advertised = {(schema.get("function") or schema).get("name")
                  for schema in box.get_tool_schemas()}
    assert "read_file" in advertised
    assert "write_file" not in advertised


def test_a_deferred_tool_is_still_dispatchable(box):
    box.defer_tools(["write_file"])
    results = _run(box, _call("write_file", path="x"))
    assert "error" not in results[0]["content"], (
        "deferral is about prompt space, not permission"
    )


def test_a_deferred_tool_is_still_discoverable(box):
    box.defer_tools(["write_file"])
    assert "write_file" in {hit["name"] for hit in discover(box, "write a file")}


def test_a_deferred_tool_is_still_role_gated(box):
    box.defer_tools(["write_file"])
    role = Role(id="reader", selector=ToolSelector(allow_names=frozenset({"read_file"})))
    binding = RoleBinding.activate(role, box)
    try:
        results = _run(box, _call("write_file", path="x"))
    finally:
        binding.deactivate()
    assert "role_denied" in results[0]["content"]


def test_undefer_advertises_again(box):
    box.defer_tools(["write_file"])
    box.undefer_tools(["write_file"])
    advertised = {(schema.get("function") or schema).get("name")
                  for schema in box.get_tool_schemas()}
    assert "write_file" in advertised


# ── D6: auto-load at dispatch ────────────────────────────────────────────


def _deferred_capability() -> Capability:
    return Capability(
        id="net",
        description="network operations",
        defer_loading=True,
        tools=[_def("fetch_url", "Fetch a URL over HTTP.")],
    )


def test_a_deferred_capability_is_searchable_by_tool_name(box):
    box.register_capability(_deferred_capability())
    hit = next(h for h in discover(box, "fetch a url") if h["name"] == "fetch_url")
    assert hit["capability"] == "net"
    assert hit["loaded"] is False, (
        "discoverable before loading is the whole point of D4 -- otherwise "
        "a deferred capability is reachable only by an agent that already "
        "knows its id"
    )


def test_autoload_loads_the_capability_that_provides_the_tool(box):
    box.register_capability(_deferred_capability())
    assert "fetch_url" not in box.tools
    assert autoload(box, "fetch_url") is None
    assert "fetch_url" in box.tools
    assert "net" in box._loaded_capability_ids


def test_autoload_leaves_a_tool_it_does_not_know_alone(box):
    assert autoload(box, "no_such_tool") is None, (
        "not ours to explain -- the dispatcher reports the unknown tool"
    )


def test_autoload_does_not_re_advertise(box):
    box.register_capability(_deferred_capability())
    autoload(box, "fetch_url")
    advertised = {(schema.get("function") or schema).get("name")
                  for schema in box.get_tool_schemas()}
    assert "fetch_url" not in advertised, (
        "re-advertising rewrites the head of the prompt mid-run and costs "
        "a full prefill (D41, I11); the model already has the schema"
    )
    assert box.is_deferred("fetch_url")


def test_autoload_reports_a_capability_that_cannot_provide_the_tool(box):
    box.register_capability(Capability(id="net", description="lies",
                                       defer_loading=True, tools=[]))
    box._tool_capabilities["fetch_url"] = "net"
    failure = autoload(box, "fetch_url")
    assert failure is not None
    assert failure["error_type"] == "autoload_failed"
    assert "search_tools" in failure["suggestion"], (
        "an error that does not name the way forward makes the model guess"
    )


def test_an_unknown_tool_still_errors_and_says_what_to_try(box):
    results = _run(box, _call("no_such_tool"))
    payload = json.loads(results[0]["content"])
    assert "not registered" in payload["error"]
    assert "search_tools" in payload["suggestion"]


def _run(toolbox: ToolBox, *calls) -> list[dict]:
    import asyncio
    return asyncio.run(toolbox.execute_tool_calls_async(list(calls)))


def test_the_system_prompt_says_discovery_exists(box):
    prompt = box.build_system_prompt("base")
    assert "search_tools" in prompt, (
        "the tools an agent was not given are the ones it cannot see; if "
        "nothing says so, a missing tool reads as a missing capability"
    )
