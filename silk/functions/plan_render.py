# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0

Markdown -> HTML for the Plan Viewer, via the user's ``mordant`` parser.

Qt-free and dependency-optional: ``mordant`` is a viewer-only dependency
(``pip install mordant``). When it is absent, :func:`markdown_to_html` returns
``None`` and the caller falls back to plain text — a missing optional dep never
breaks the graph.

``highlighting_mode="Attribute"`` is deliberate: mordant then inlines ``style=``
rather than emitting CSS classes, which Qt's ``QTextEdit``/``QTextBrowser`` (a
limited HTML subset) will not resolve.
"""
from __future__ import annotations

from typing import Optional

try:  # viewer-only optional dependency
    import mordant  # type: ignore
    MORDANT_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure => graceful fallback
    mordant = None  # type: ignore[assignment]
    MORDANT_AVAILABLE = False


def markdown_to_html(
    markdown_text: str, *, theme: str = "InspiredGitHub",
) -> Optional[str]:
    """Render *markdown_text* to HTML via mordant, or ``None`` if unavailable.

    GFM task lists become checkboxes, ``:emoji:`` shortcodes expand, and
    ```` ```mermaid ```` blocks render — see the plan's markdown renderer.
    """
    if mordant is None:
        return None
    try:
        return mordant.markdown_to_html(
            markdown_text,
            highlighting_mode="Attribute",
            highlighting_theme=theme,
        )
    except Exception:  # noqa: BLE001 - never let a render error break the node
        return None
