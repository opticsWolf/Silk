# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

What waits on a running agent, in the agent node itself.

:class:`RunQueueView` lists a :class:`RunControl`'s messages -- queued for
the next run, steering this one, or interrupting it -- one row each, with
the mode as a button (click to re-file it) and a remove button. Under an
orchestrator it also lists the workers running right now, each with its
own Stop and its own waiting messages. Hidden while there is nothing to
show, so an idle agent stays one icon row.

:class:`StopEscalation` is the press-twice stop: the first press asks the
run to stop after the current action; while that is pending the button
pulses red, and a second press cuts the action now.
"""

from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QVBoxLayout, QWidget

from weave.library.sync_tool_button.widgets.sync_tool_button import SyncToolButton

from ..functions.run_control import (
    MODE_INTERRUPT,
    MODE_QUEUE,
    MODE_STEER,
    STOP_FORCE,
    STOP_NONE,
    RunControl,
)

#: Button label (a semantic icon) and what the mode means, per mode.
MODE_LOOK = {
    MODE_QUEUE: ("Queue", "Queued: sent as the next run's prompt."),
    MODE_STEER: ("Steer", "Steer: joins this run at its next step."),
    MODE_INTERRUPT: ("Interrupt",
                     "Interrupt: cuts the current action and carries on "
                     "with this message."),
}

#: How long a row's text may be before it is cut, with the whole in the
#: tooltip. The node is narrow; a long prompt is still one row.
TEXT_LIMIT = 60

#: The pulse of a pending stop, in ms per phase.
PULSE_MS = 450


def _elide(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= TEXT_LIMIT else flat[:TEXT_LIMIT - 1] + "…"


class RunQueueView(QWidget):
    """The pending messages and running workers of one control tree.

    Main thread only; the control is thread-safe, and :meth:`refresh` is
    what its change listener should reach (through a queued signal).
    """

    def __init__(self, control: RunControl,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._control = control
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(1)
        #: Per worker control, the press-twice state of its Stop.
        self._stops: dict[int, StopEscalation] = {}
        self.setVisible(False)

    def refresh(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        kept: dict[int, StopEscalation] = {}
        rows = 0
        for item in self._control.pending():
            self._layout.addWidget(self._message_row(item, indent=0))
            rows += 1
        for kid in self._control.children():
            rows += self._worker_rows(kid, depth=1, kept=kept)
        for key, stop in self._stops.items():
            if key not in kept:
                stop.reset()
        self._stops = kept
        self.setVisible(rows > 0)

    # -- rows --------------------------------------------------------------

    def _message_row(self, item: dict[str, Any], indent: int) -> QWidget:
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(12 * indent, 0, 0, 0)
        line.setSpacing(2)
        label_text, tip = MODE_LOOK.get(item["mode"], MODE_LOOK[MODE_QUEUE])
        mode = SyncToolButton(initial_text=label_text, show_label=False,
                              dimensions=18)
        mode.set_tooltip(f"{tip}\nClick to file it differently.")
        mode.clicked.connect(
            lambda _c=False, i=item["id"], b=mode, q=indent == 0:
                self._refile_menu(i, b, allow_queue=q))
        line.addWidget(mode)
        text = QLabel(_elide(item["text"]))
        text.setToolTip(item["text"])
        line.addWidget(text, stretch=1)
        drop = SyncToolButton(initial_text="Remove", show_label=False,
                              dimensions=18)
        drop.set_tooltip("Remove this message; it is never sent.")
        drop.clicked.connect(
            lambda _c=False, i=item["id"]: self._control.remove(i))
        line.addWidget(drop)
        return row

    def _worker_rows(self, kid: RunControl, depth: int,
                     kept: "dict[int, StopEscalation]") -> int:
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(12 * (depth - 1), 0, 0, 0)
        line.setSpacing(2)
        name = QLabel(f"⇢ {kid.label or 'worker'}")
        name.setToolTip(
            f"Worker '{kid.label}' is running. Send '@{kid.label} …' "
            "from the prompt to steer it alone.")
        line.addWidget(name, stretch=1)
        button = SyncToolButton(initial_text="Stop", show_label=False,
                                dimensions=18)
        stop = self._stops.get(id(kid))
        if stop is None:
            stop = StopEscalation(button, on_stop=kid.request_stop,
                                  idle_label="Stop")
        else:
            stop.rebind(button)
        stop.set_running(True)
        if kid.stop_level() >= STOP_FORCE:
            button.setEnabled(False)
        elif kid.stop_level() > STOP_NONE:
            stop.mark_stopping()
        kept[id(kid)] = stop
        line.addWidget(button)
        self._layout.addWidget(row)
        rows = 1
        for item in kid.pending():
            self._layout.addWidget(self._message_row(item, indent=depth))
            rows += 1
        for grandkid in kid.children():
            rows += self._worker_rows(grandkid, depth + 1, kept)
        return rows

    def _refile_menu(self, item_id: int, anchor: QWidget,
                     allow_queue: bool) -> None:
        menu = QMenu(anchor)
        # A worker has no next run to queue for: steer or interrupt only.
        modes = (MODE_QUEUE, MODE_STEER, MODE_INTERRUPT) if allow_queue \
            else (MODE_STEER, MODE_INTERRUPT)
        for mode in modes:
            label, tip = MODE_LOOK[mode]
            action = menu.addAction(label)
            action.setToolTip(tip)
            action.triggered.connect(
                lambda _c=False, m=mode: self._control.set_mode(item_id, m))
        menu.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))

    def cleanup(self) -> None:
        for stop in self._stops.values():
            stop.reset()
        self._stops = {}


class StopEscalation:
    """Press-twice stop on a :class:`SyncToolButton`.

    ``on_stop(force)`` is called with ``False`` for the first press (stop
    after the current action) and ``True`` for the second (cut it now).
    While the soft stop is pending the button pulses between a red Force
    Stop and a plain one, so the second press is visibly on offer.
    """

    def __init__(self, button: SyncToolButton,
                 on_stop: Callable[[bool], None],
                 idle_label: str = "Stop",
                 idle_tip: str = "",
                 stop_tip: str = "Stop after the current action. Press "
                                 "again to cut the action now.",
                 force_tip: str = "Stopping after the current action — "
                                  "press to cut it now (force stop).",
                 window_ms: int = 0,
                 connect: bool = True,
                 quiet: Optional[Callable[[], ContextManager[Any]]] = None) -> None:
        #: Wraps every label change. A button registered with WidgetCore
        #: reports a new label as an edit, and an edit dirties the node --
        #: which, for a running agent, cancels the run it is labelling.
        self._quiet = quiet or nullcontext
        self._on_stop = on_stop
        self._idle_label = idle_label
        self._idle_tip = idle_tip
        self._connect = connect
        self._stop_tip = stop_tip
        self._force_tip = force_tip
        #: Without a run state to follow (the Prompt node cannot see the
        #: agent's), the offer of a force stop lapses after this long.
        self._window_ms = window_ms
        self._running = False
        self._stopping = False
        self._phase = False
        self._timer = QTimer()
        self._timer.setInterval(PULSE_MS)
        self._timer.timeout.connect(self._pulse)
        self._lapse = QTimer()
        self._lapse.setSingleShot(True)
        self._lapse.timeout.connect(self.reset)
        self._button: SyncToolButton = button
        self.rebind(button)

    def rebind(self, button: SyncToolButton) -> None:
        self._button = button
        if self._connect:
            button.clicked.connect(self.press)
        self._show()

    @property
    def stopping(self) -> bool:
        return self._stopping

    def set_running(self, running: bool) -> None:
        self._running = running
        if running:
            self._show()
        else:
            self.reset()

    def press(self) -> None:
        if self._stopping:
            self._stopping = False
            self._timer.stop()
            self._lapse.stop()
            self._on_stop(True)
            self._show()
            return
        self._on_stop(False)
        self.mark_stopping()

    def press_force(self) -> None:
        """Skip the first step: cut the action now."""
        self._stopping = False
        self._timer.stop()
        self._lapse.stop()
        self._on_stop(True)
        self._show()

    def mark_stopping(self) -> None:
        """A soft stop is pending (pressed here, or arrived another way)."""
        if self._stopping:
            return
        self._stopping = True
        self._phase = True
        self._timer.start()
        if self._window_ms:
            self._lapse.start(self._window_ms)
        self._show()

    def reset(self) -> None:
        self._stopping = False
        self._timer.stop()
        self._lapse.stop()
        self._show()

    def _pulse(self) -> None:
        self._phase = not self._phase
        self._show()

    def _show(self) -> None:
        button = self._button
        try:
            with self._quiet():
                self._paint(button)
        except RuntimeError:        # the row was rebuilt under us
            self._timer.stop()
            self._lapse.stop()

    def _paint(self, button: SyncToolButton) -> None:
        if self._stopping:
            # Red Force Stop, then the theme's plain stop: the pulse.
            button.set_label("Force Stop" if self._phase else "Stopping")
            button.set_icon_name("player-stop")
            button.set_tooltip(self._force_tip)
        elif self._running:
            button.set_label("Stop")
            button.set_tooltip(self._stop_tip)
        else:
            button.set_label(self._idle_label)
            button.set_tooltip(self._idle_tip or self._stop_tip)
