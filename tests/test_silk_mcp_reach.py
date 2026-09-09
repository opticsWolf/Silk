# -*- coding: utf-8 -*-
"""What an MCP server can reach, said out loud (G21 residue 2, D84).

Silk sandboxes every file tool it registers. An MCP server is another
process: it writes where its own process can, and the file-permissions
port on the canvas describes none of it. Silk cannot fix that -- the
authority was granted when the user launched the server -- so the fix is
to stop the surface implying otherwise.

These tests pin the two honesty rules: the notice is a heuristic that
says so, and a generic tool vocabulary must not trip it, because a notice
that fires on `create_issue` is a notice people learn to ignore.
"""

from __future__ import annotations



from silk.functions.mcp_reach import (  # noqa: E402
    file_tools,
    reach_notice,
)


def _tool(name, description=""):
    return {"name": name, "description": description}


def test_a_filesystem_server_is_reported():
    tools = [_tool("write_file", "Write content to a file at PATH"),
             _tool("read_file", "Read the file at PATH"),
             _tool("list_directory", "List the entries of a directory")]
    found = file_tools(tools)
    assert found["write"] == ["write_file"]
    assert found["read"] == ["list_directory", "read_file"]


def test_the_notice_says_what_it_means():
    notice = reach_notice("fs", [_tool("write_file", "writes a file")])
    assert "fs" in notice and "write_file" in notice
    assert "its own process" in notice, (
        "the point is not that the tool writes -- it is that Silk's "
        "sandbox does not cover the process doing it"
    )


def test_a_server_with_no_filesystem_tools_is_not_reported():
    """A notice that fires on everything is one nobody reads."""
    tools = [_tool("create_issue", "Create an issue in the tracker"),
             _tool("list_repos", "List the repositories for a user"),
             _tool("search_code", "Search code across the organisation")]
    assert reach_notice("github", tools) == ""


def test_a_verb_without_a_filesystem_subject_is_not_a_finding():
    assert reach_notice("db", [_tool("delete_row", "Delete a row")]) == ""
    assert reach_notice("q", [_tool("rewrite_query", "Rewrite a query")]) == ""


def test_a_description_can_be_what_gives_it_away():
    """Servers name tools however they like; the description is evidence."""
    notice = reach_notice("odd", [_tool("persist", "saves the given file to disk")])
    assert "persist" in notice


def test_reads_alone_are_still_reported_but_separately():
    notice = reach_notice("ro", [_tool("read_file", "read a file")])
    assert "reads (read_file)" in notice and "writes" not in notice


def test_empty_and_malformed_tool_lists_are_survivable():
    assert reach_notice("none", []) == ""
    assert reach_notice("none", None) == ""
    assert file_tools([{}, None]) == {"write": [], "read": []}
