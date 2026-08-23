# -*- coding: utf-8 -*-
"""
Chat Display Node

A sink node that continuously appends chat turns to a running log,
rendering the entire thread beautifully as HTML using MarkdownConverter.
Designed to receive `chat_turn` dicts from the Silk Agent node.
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
    """Render one ``chat_turn`` dict to markdown, including tool turns.

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
        self.add_input("chat_turn", datatype="dict")

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

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Processes incoming chat turns without blocking the UI."""
        if self.is_compute_cancelled():
            return {}

        turn = inputs.get("chat_turn")

        if isinstance(turn, dict):
            t_id = turn.get("turn_id")

            # Check if this is a newly generated turn to avoid duplicate
            # appends on spurious re-evaluations
            if t_id and t_id != self._last_turn_id:
                self._last_turn_id = t_id
                self._chat_log_md += _format_turn(turn)

        # Convert the accumulated markdown string to HTML off-thread (R9.3)
        html_output = ""
        if self._chat_log_md and self._converter:
            try:
                html_output = self._converter.convert(self._chat_log_md)
            except Exception as e:
                log.error("Markdown conversion failed in worker thread: %s", exc_info=e)
                html_output = "<p style='color:#d32f2f; font-weight:bold;'>Conversion Error</p>"
        elif not self._chat_log_md:
             html_output = "<i>Waiting for chat data...</i>"

        # Store result for main-thread application (R9.5 sink node returns {})
        self._pending_html = html_output
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
