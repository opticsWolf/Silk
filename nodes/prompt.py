# -*- coding: utf-8 -*-
"""Silk Prompt Node.

The chat box for an agent: type, press Enter, and the agent runs on what
was typed. It is a node rather than a box on the agent so that one input
can drive whatever it is wired to -- an Agent, an Orchestrator, the next
step of a network -- and so the agent stays a small row of controls.

Wiring
------
``send`` → the agent's ``run``. The message rides the pulse as its
payload, so no data wire is needed, and a message sent while the agent
is busy is queued for the next run instead of stopping it (a changed
data input would stop a running node). The Send button's arrow sends it
as a *steer* instead -- it joins the running turn at its next step --
or as an *interrupt*, which cuts the current action and carries on with
the message. ``@worker message`` goes to that running worker of an
Orchestrator. ``stop`` → ``stop`` and ``new_chat`` → ``clear`` are the
other two buttons; Stop is press-twice (after the current action, then
now). ``prompt`` carries the last sent
message for anything else that wants it (a log, a file); wiring it into
the agent's ``user_prompt`` as well works when idle, but a send during a
run then stops that run.

Enter sends, Ctrl+Enter steers, Ctrl+Shift+Enter interrupts,
Shift+Enter starts a new line, Ctrl+Up / Ctrl+Down step through what was
sent before. The box empties on send; the history is
kept with the graph.
"""

from typing import Any, ClassVar, Dict, List, Optional, cast

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QHBoxLayout, QMenu, QPlainTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from weave.library.sync_tool_button.widgets.sync_tool_button import SyncToolButton
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.widgetcore import PortRole, WidgetCore

from ..functions.run_control import MODE_INTERRUPT, MODE_QUEUE, MODE_STEER
from ..widgets.run_queue import StopEscalation

#: Sent messages kept for Ctrl+Up / Ctrl+Down, oldest dropped first.
HISTORY_MAX = 50

#: How long after a Stop the button offers Force stop. The prompt cannot
#: see the agent's run; the agent escalates a second stop by itself too.
FORCE_WINDOW_MS = 8000


@register_node
class SilkPromptNode(ActiveNode):
    """Chat input: sends a prompt to an agent on an exec pulse."""
    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk Prompt"
    node_description: ClassVar[Optional[str]] = (
        "Chat input for an agent: Enter sends the message on an exec "
        "pulse; Stop and New chat drive the agent's stop and clear."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "prompt", "chat", "input", "llm",
    ]
    node_icon: ClassVar[Optional[str]] = "message"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.GROW_ONLY
    node_state_api = 1   # owns a hand-written state dict
    node_version = 1     # bump on any state-shape change (G20)

    def __init__(self, title: str = "Silk Prompt", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        #: The last message sent (the `prompt` output) and what came before.
        self._sent: str = ""
        self._history: List[str] = []
        #: Where Ctrl+Up / Ctrl+Down stands: an index into the history, or
        #: None while the box holds a fresh draft.
        self._browse: Optional[int] = None
        self._draft: str = ""

        self.add_output("prompt", datatype="string")
        self.add_output("send", datatype="exec")
        self.add_output("stop", datatype="exec")
        self.add_output("new_chat", datatype="exec")

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        self._widget_core = WidgetCore()
        self._widget_core.set_node(self)

        # Not a WidgetCore binding: a keystroke must not dirty the graph,
        # since anything downstream that is running would stop.
        self._editor = QPlainTextEdit()
        self._editor.setPlaceholderText(
            "Message the agent… Enter sends, Ctrl+Enter steers, "
            "Ctrl+Shift+Enter interrupts, Shift+Enter for a new line, "
            "Ctrl+Up / Ctrl+Down for earlier messages."
        )
        self._editor.setMinimumHeight(60)
        self._editor.installEventFilter(self)
        layout.addWidget(self._editor, stretch=1)

        buttons = QHBoxLayout()
        self._btn_send = SyncToolButton(
            initial_text="Send", show_label=False, dimensions=24,
        )
        self._btn_send.set_tooltip(
            "Send the message (Enter). Wire `send` to the agent's `run`: "
            "the message travels with the pulse. Sent while the agent is "
            "busy, it is queued for the next run. The arrow steers the "
            "running turn instead (Ctrl+Enter) or interrupts it "
            "(Ctrl+Shift+Enter).")
        self._btn_send.clicked.connect(lambda _c=False: self.send())
        send_menu = QMenu(self._btn_send)
        for label, mode, keys in (
            ("Send (queue)", MODE_QUEUE, "Enter"),
            ("Steer", MODE_STEER, "Ctrl+Enter"),
            ("Interrupt", MODE_INTERRUPT, "Ctrl+Shift+Enter"),
        ):
            action = send_menu.addAction(f"{label}\t{keys}")
            action.triggered.connect(
                lambda _c=False, m=mode: self.send(m))
        self._btn_send.setMenu(send_menu)
        self._btn_send.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        buttons.addWidget(self._btn_send)

        self._btn_stop = SyncToolButton(
            initial_text="Stop", show_label=False, dimensions=24,
        )
        # Press-twice: after the current action, then (pulsing red) now.
        self._stop = StopEscalation(
            self._btn_stop, on_stop=self._pulse_stop,
            stop_tip="Stop the agent after its current action; press "
                     "again to cut the action now. Queued messages stay "
                     "on the agent. Wire `stop` to the agent's `stop`.",
            window_ms=FORCE_WINDOW_MS,
            quiet=self._widget_core.suppress_signals,
        )
        self._stop.set_running(True)
        stop_menu = QMenu(self._btn_stop)
        soft = stop_menu.addAction("Stop after this action")
        soft.triggered.connect(self.stop)
        force = stop_menu.addAction("Force stop now")
        force.triggered.connect(self.force_stop)
        self._btn_stop.setMenu(stop_menu)
        self._btn_stop.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        buttons.addWidget(self._btn_stop)

        self._btn_new = SyncToolButton(
            initial_text="New", show_label=False, dimensions=24,
        )
        self._btn_new.set_tooltip(
            "New chat: the agent forgets the conversation and a wired "
            "Chat Log starts over. Wire `new_chat` to the agent's "
            "`clear`. The plan, the knowledge graph and always-allow "
            "grants are kept.")
        self._btn_new.clicked.connect(self.new_chat)
        buttons.addWidget(self._btn_new)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        for port, button in (("btn_send", self._btn_send),
                             ("btn_stop", self._btn_stop),
                             ("btn_new", self._btn_new)):
            self._widget_core.register_widget(
                port, button, role=PortRole.INTERNAL, add_to_layout=False,
            )

        container = QWidget()
        container.setLayout(layout)
        self.set_content_widget(container)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── Keys ─────────────────────────────────────────────────────────

    def eventFilter(self, obj: Any, event: QEvent) -> bool:
        if obj is self._editor and event.type() == QEvent.Type.KeyPress:
            key_event = cast(QKeyEvent, event)
            key = key_event.key()
            mods = key_event.modifiers()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
                shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
                if shift and not ctrl:
                    return False
                self.send(MODE_INTERRUPT if ctrl and shift
                          else MODE_STEER if ctrl else MODE_QUEUE)
                return True
            if mods & Qt.KeyboardModifier.ControlModifier:
                if key == Qt.Key.Key_Up:
                    self._step_history(-1)
                    return True
                if key == Qt.Key.Key_Down:
                    self._step_history(+1)
                    return True
        return super().eventFilter(obj, event)

    def _step_history(self, step: int) -> None:
        """Ctrl+Up / Ctrl+Down: older / newer sent messages, then the draft."""
        if not self._history:
            return
        if self._browse is None:
            if step > 0:
                return
            self._draft = self._editor.toPlainText()
            index = len(self._history) - 1
        else:
            index = self._browse + step
        if index < 0:
            index = 0
        if index >= len(self._history):
            self._browse = None
            self._set_text(self._draft)
            return
        self._browse = index
        self._set_text(self._history[index])

    def _set_text(self, text: str) -> None:
        self._editor.setPlainText(text)
        self._editor.moveCursor(self._editor.textCursor().MoveOperation.End)

    # ── Actions ──────────────────────────────────────────────────────

    def send(self, mode: str = MODE_QUEUE) -> None:
        """Send what is in the box. An empty box sends nothing.

        Queued, the payload is the text itself; a steer or an interrupt
        sends ``{"text", "mode"}``.
        """
        text = self._editor.toPlainText().strip()
        if not text:
            return
        self._sent = text
        if not self._history or self._history[-1] != text:
            self._history.append(text)
            del self._history[:-HISTORY_MAX]
        self._browse = None
        self._draft = ""
        self._editor.clear()
        # `prompt` follows for whatever else reads it; the agent takes
        # the message from the pulse (see the module docstring).
        self.set_dirty("prompt_sent")
        self.pulse("send", payload=text if mode == MODE_QUEUE
                   else {"text": text, "mode": mode})

    def stop(self) -> None:
        """Stop after the current action (a second press: now)."""
        self._stop.press()

    def force_stop(self) -> None:
        self._stop.press_force()

    def _pulse_stop(self, force: bool) -> None:
        self.pulse("stop", payload="force" if force else True)

    def new_chat(self) -> None:
        self.pulse("new_chat", payload=True)

    # ── Dataflow ─────────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"prompt": self._sent}

    # ── State ────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["silk_prompt"] = {
            "sent": self._sent,
            "history": list(self._history),
            "draft": self._editor.toPlainText(),
        }
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        with self._widget_core.suppress_signals():
            super().restore_state(state)
        saved = state.get("silk_prompt")
        if not isinstance(saved, dict):
            return
        self._sent = str(saved.get("sent") or "")
        self._history = [str(h) for h in (saved.get("history") or [])
                         if str(h).strip()][-HISTORY_MAX:]
        self._browse = None
        self._editor.setPlainText(str(saved.get("draft") or ""))
