# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Markdown -> HTML for the Plan Viewer, via the user's ``mordant`` parser.

Qt-free and dependency-optional: ``mordant`` is a viewer-only dependency
(``pip install mordant``). When it is absent, :func:`markdown_to_html` returns
``None`` and the caller falls back to plain text — a missing optional dep never
breaks the graph.

**The floor is readable markdown, always (T6).** That is the rendering
guarantee: every plan is legible with nothing installed, and ``mordant``
is an upgrade (task-list checkboxes, ``:emoji:`` shortcodes, mermaid,
syntax highlighting), never a requirement. Degradation is
announced in the **log**, once per process, and nowhere else: a missing
optional dependency is an install fact, not a property of the user's
graph, and a permanent line on a node would put it in front of someone
who cannot act on it every time they open the canvas. A render that
*fails* is a different thing from one that is absent -- an installed
renderer raising is a bug somewhere -- so it is logged at a different
level rather than folded into the same silence.

``highlighting_mode="Attribute"`` is deliberate: mordant then inlines ``style=``
rather than emitting CSS classes, which Qt's ``QTextEdit``/``QTextBrowser`` (a
limited HTML subset) will not resolve.
"""
from __future__ import annotations

from typing import Optional

from weave.logger import get_logger

try:  # viewer-only optional dependency
    import mordant
    MORDANT_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure => graceful fallback
    mordant = None
    MORDANT_AVAILABLE = False

log = get_logger("SilkPlanRender")


#: Said once per process, not once per render: a plan viewer re-renders on
#: every plan event, and a notice repeated at that rate is noise.
_NOTIFIED: dict[str, bool] = {"missing": False, "failed": False}


def renderer_notice() -> str:
    """One log line about the renderer in use, or ``""`` when styled.

    Empty means the styled renderer is in use and there is nothing to
    say. The text is deliberately about the *upgrade*, not about a
    failure: plain markdown is the guaranteed floor, not a broken state.
    """
    if MORDANT_AVAILABLE:
        return ""
    return ("Plain markdown — install `mordant` (pip install mordant) for "
            "task-list checkboxes, emoji shortcodes and mermaid diagrams.")


def markdown_to_html(
    markdown_text: str, *, theme: str = "InspiredGitHub",
) -> Optional[str]:
    """Render *markdown_text* to HTML via mordant, or ``None`` if unavailable.

    GFM task lists become checkboxes, ``:emoji:`` shortcodes expand, and
    ```` ```mermaid ```` blocks render — see the plan's markdown renderer.
    """
    if mordant is None:
        if not _NOTIFIED["missing"]:
            _NOTIFIED["missing"] = True
            # Warning, not info: a fallback exists, so nothing is broken,
            # but the surface is quietly worse than it should be and the
            # user is the only one who can change that.
            log.warning(renderer_notice())
        return None
    try:
        return mordant.markdown_to_html(
            markdown_text,
            highlighting_mode="Attribute",
            highlighting_theme=theme,
        )
    except Exception as exc:  # noqa: BLE001 - never let a render error break the node
        # An *installed* renderer that raises is a bug somewhere, not a
        # missing extra: the fallback is the same but the silence is not
        # earned. Once per process, because the viewer re-renders on every
        # plan event and a loud loop helps nobody.
        if not _NOTIFIED["failed"]:
            _NOTIFIED["failed"] = True
            log.warning(f"mordant is installed but failed to render: {exc}. "
                        "Falling back to plain markdown for this session.")
        return None
