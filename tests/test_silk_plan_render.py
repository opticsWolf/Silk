# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Markdown -> HTML for the plan surfaces, as Qt will actually draw it.

Qt's rich text is an HTML subset with no form controls: a GFM task item
rendered as ``<input type="checkbox">`` disappears without a trace, which
is how the Plan Viewer came to show no checkboxes at all. And mordant
leaves GFM off unless asked, so ``- [x]`` was literal text before that.
"""

from __future__ import annotations

import pytest

from silk.functions import plan_render
from silk.functions.plan_render import (
    BOX_CHECKED,
    BOX_OPEN,
    checkboxes_as_glyphs,
    markdown_to_html,
)


def test_a_checkbox_becomes_a_glyph_in_any_attribute_order():
    html = ('<li><input checked="" disabled="" type="checkbox"> a</li>'
            '<li><input type="checkbox" disabled> b</li>'
            '<li><INPUT TYPE="checkbox" CHECKED> c</li>')
    assert checkboxes_as_glyphs(html) == (
        f"<li>{BOX_CHECKED} a</li><li>{BOX_OPEN} b</li><li>{BOX_CHECKED} c</li>")


def test_other_inputs_are_left_alone():
    assert checkboxes_as_glyphs('<input type="text">') == '<input type="text">'


@pytest.mark.skipif(not plan_render.MORDANT_AVAILABLE, reason="mordant not installed")
def test_task_lists_and_emoji_render_as_qt_can_draw_them():
    html = markdown_to_html("## :white_check_mark: Done\n\n- [x] shipped\n- [ ] open\n")
    assert f"{BOX_CHECKED} shipped" in html and f"{BOX_OPEN} open" in html
    assert "[x]" not in html and "[ ]" not in html, "GFM is on"
    assert "<input" not in html
    assert "✅" in html and ":white_check_mark:" not in html


@pytest.mark.skipif(not plan_render.MORDANT_AVAILABLE, reason="mordant not installed")
def test_raw_html_from_an_agent_is_not_rendered():
    html = markdown_to_html("- [ ] <b>x</b> <img src=a onerror=b()>\n")
    assert "<b>" not in html and "<img" not in html
