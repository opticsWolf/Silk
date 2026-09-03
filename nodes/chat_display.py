# -*- coding: utf-8 -*-
"""
Chat Display Node

A sink node that continuously appends chat turns to a running log,
rendering the entire thread beautifully as HTML using MarkdownConverter.

Wire ``Silk Agent.events`` → ``event``. The agent emits one typed stream
(spec D2/D3); this node keeps the ``chat.turn`` events and ignores the
rest, so the same wire that feeds a monitor feeds the log.
"""

import time
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QFormLayout

from weave.logger import get_logger
log = get_logger("ChatDisplayNode")

from weave.widgets.markdown_widget import MarkdownWidget

from weave.node.threaded import ThreadedNode
from weave.registry import register_node
from weave.widgetcore import WidgetCore, PortRole
from weave.node import VerticalSizePolicy

from ..functions.stream_events import EventType


#: Tool-result bodies (file contents, etc.) can be large; cap what the log
#: renders so one read doesn't flood the thread.
_TOOL_RESULT_MAX = 800


def _format_tool_result(body: str) -> str:
    """Compact, readable rendering of a tool-result body for the chat log.

    Unwraps the ``{"content": …}`` envelope when present so a file read shows
    its text rather than escaped JSON, and truncates very long output.
    """
    text = str(body or "")
    try:
        import json
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            text = data["content"]
    except (ValueError, TypeError):
        pass
    if len(text) > _TOOL_RESULT_MAX:
        text = text[:_TOOL_RESULT_MAX] + f"\n… ({len(text)} chars total)"
    return text


def _format_turn(turn: dict) -> str:
    """Render one ``chat.turn`` event to markdown, including tool turns.

    Backward compatible: a turn without a ``turns`` list renders exactly the
    old user/AI pair. Tool calls and results are interleaved between the user
    prompt and the AI answer, in the order the agent produced them.
    """
    time_str = time.strftime("%I:%M %p", time.localtime(turn.get("timestamp", time.time())))
    parts = [f"**👤 User** ({time_str}):\n\n{turn.get('user', '')}\n\n"]

    for t in turn.get("turns") or ():
        if not isinstance(t, dict):
            continue
        role = t.get("role")
        tool = t.get("tool", "?")
        if role == "tool_call":
            args = t.get("args")
            if isinstance(args, dict):
                arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            else:
                arg_str = str(args or "")
            parts.append(f"**🔧 Tool call** — `{tool}({arg_str})`\n\n")
        elif role == "tool_result":
            mark = "⚠️" if t.get("error") else "✓"
            body = _format_tool_result(t.get("result", ""))
            parts.append(
                f"**🔧 Tool result** {mark} `{tool}`:\n\n```\n{body}\n```\n\n"
            )

    ai_text = turn.get("ai", "")
    if ai_text:
        parts.append(f"**🤖 AI** ({time_str}):\n\n{ai_text}\n\n")
    parts.append("---\n\n")
    return "".join(parts)


@register_node
class ChatDisplayNode(ThreadedNode):

    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore
    display_updated = Signal(str)

    node_class: ClassVar[str] = "Display"
    node_subclass: ClassVar[str] = "Chat"
    node_name: ClassVar[Optional[str]] = "Chat Log Display"
    node_description: ClassVar[Optional[str]] = "Accumulates and renders a running markdown chat log."
    node_tags: ClassVar[List[str]] = ["markdown", "chat", "display", "log"]
    node_icon: ClassVar[Optional[str]] = "markdown"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.GROW_ONLY

    def __init__(self, title: str = "Chat Display", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # Initialize converter for off-thread processing (R9.3 compliance)
        from weave.widgets.markdown_converter import MarkDownConverter
        self._converter = MarkDownConverter(safe_mode=True, hard_breaks=True)

        # Internal state for the continuous log
        self._chat_log_md: str = ""
        self._last_turn_id: Optional[str] = None
        self._pending_html: Optional[str] = None

        # Ports
        self.add_input("event", datatype="dict")

        # Layout
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self._display_widget = MarkdownWidget(mode="display", safe_mode=True, hard_breaks=True)

        # Manually place widget in layout (R4.2)
        form.addRow("", self._display_widget)

        # Register with WidgetCore as a DISPLAY role (no graph output port,
        # R6.2). The widget's own write path renders HTML in display mode,
        # so all updates go through push_display — never direct setHtml.
        self._widget_core.register_widget(
            "display", self._display_widget,
            role=PortRole.DISPLAY, datatype="string", default="",
            add_to_layout=False,
        )

        # Clear button
        self.btn_clear = QPushButton("Clear Chat Log")
        self.btn_clear.clicked.connect(self._clear_log)
        form.addRow("", self.btn_clear)

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, 'patch_proxy'):
            self._widget_core.patch_proxy()

    def _clear_log(self) -> None:
        """Reset the chat history visually and internally."""
        self._chat_log_md = ""
        self._last_turn_id = None
        self._widget_core.push_display("display", "<i>Chat log cleared.</i>")

    # ── Event handling ────────────────────────────────────────────────

    def _append_turn(self, event: Any) -> bool:
        """Append one ``chat.turn`` event to the log. True if it was new.

        Pure state mutation, no widget writes, so it is safe from either
        the stream hook (main thread) or ``compute`` (worker thread).
        """
        if not isinstance(event, dict):
            return False
        kind = event.get("type")
        # `None` keeps a hand-wired plain dict working; anything else on the
        # one shared events port is somebody else's event.
        if kind not in (None, EventType.CHAT_TURN.value):
            return False
        turn_id = event.get("turn_id")
        # Dedup: spurious re-evaluations and re-delivered previews.
        if turn_id and turn_id == self._last_turn_id:
            return False
        self._last_turn_id = turn_id
        self._chat_log_md += _format_turn(event)
        return True

    def _render_log(self) -> str:
        if not self._chat_log_md:
            return "<i>Waiting for chat data...</i>"
        if self._converter is None:
            return self._chat_log_md
        try:
            return self._converter.convert(self._chat_log_md)
        except Exception as exc:  # noqa: BLE001 - a render must not kill the node
            log.error("Markdown conversion failed: %s", exc_info=exc)
            return "<p style='color:#d32f2f; font-weight:bold;'>Conversion Error</p>"

    def on_upstream_stream(self, port_name: str, value: Any) -> None:
        """Consume the agent's ``events`` stream.

        Stream previews bypass the dataflow cache and never call
        ``compute()``, so a turn read there would never arrive. Conversion
        happens on the main thread here, which is the cost of live
        rendering; a turn is one exchange, not a token, so it is paid once
        per turn rather than once per delta.
        """
        if port_name == "event":
            if self._append_turn(value):
                self._widget_core.push_display("display", self._render_log())
                sb = self._display_widget._text_edit.verticalScrollBar()
                sb.setValue(sb.maximum())
            return
        super().on_upstream_stream(port_name, value)

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Processes incoming chat turns without blocking the UI."""
        if self.is_compute_cancelled():
            return {}

        # Fallback for a plain dict wired into `event`; the agent's own
        # turns arrive as stream previews, which never run compute() (see
        # on_upstream_stream below).
        self._append_turn(inputs.get("event"))

        # Convert the accumulated markdown string to HTML off-thread (R9.3),
        # then store it for main-thread application (R9.5: a sink returns {}).
        self._pending_html = self._render_log()
        return {}

    def on_evaluate_finished(self) -> None:
        """Applies the converted HTML to the Qt display on the main thread."""
        super().on_evaluate_finished()

        if self._pending_html is not None:
            self._widget_core.push_display("display", self._pending_html)
            self.display_updated.emit(self._pending_html)

            # Auto-scroll to bottom (view state, not a widget value write).
            sb = self._display_widget._text_edit.verticalScrollBar()
            sb.setValue(sb.maximum())

    def cleanup(self) -> None:
        if hasattr(self, 'cancel_compute'):
            self.cancel_compute()
        super().cleanup()
