# -*- coding: utf-8 -*-
"""Chat Log tool-role rendering (pure function; imports PySide6 but builds
no widgets, so it is safe to run in-process)."""

from __future__ import annotations

import sys


from silk.nodes.chat_display import _format_turn, _format_tool_result


def test_plain_turn_renders_user_and_ai_only():
    md = _format_turn({"user": "hi", "ai": "hello", "timestamp": 0})
    assert "👤 User" in md and "hi" in md
    assert "🤖 AI" in md and "hello" in md
    assert "🔧 Tool" not in md


def test_tool_turns_render_between_user_and_ai_in_order():
    turn = {
        "user": "read the file",
        "ai": "here is the summary",
        "timestamp": 0,
        "turns": [
            {"role": "tool_call", "tool": "read_file", "args": {"path": "a.txt"}},
            {"role": "tool_result", "tool": "read_file",
             "result": '{"content": "5 facts", "error": null}', "error": False},
        ],
    }
    md = _format_turn(turn)
    assert "🔧 Tool call" in md and "read_file(path='a.txt')" in md
    assert "🔧 Tool result" in md and "✓" in md
    # The result envelope is unwrapped so the file text shows, not raw JSON.
    assert "5 facts" in md
    assert (md.index("👤 User") < md.index("Tool call")
            < md.index("Tool result") < md.index("🤖 AI"))


def test_error_tool_result_marked():
    turn = {"user": "x", "ai": "y", "timestamp": 0, "turns": [
        {"role": "tool_result", "tool": "write_file",
         "result": '{"error": "denied"}', "error": True},
    ]}
    assert "⚠️" in _format_turn(turn)


def test_long_tool_result_is_truncated():
    out = _format_tool_result("A" * 2000)
    assert len(out) < 1200 and "2000 chars total" in out


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
