# -*- coding: utf-8 -*-
"""RunControl -- what a person can do to an agent while it runs.

Qt-free and thread-safe: the node posts from the GUI thread, the loop
reads from its worker thread, and an orchestrator's workers each run on a
thread of their own.

Three kinds of message wait here, each with an id the UI can remove:

``queue``
    A new prompt for the *next* run. The loop never reads these; the node
    takes them when the run ends.
``steer``
    Joins the *current* run at its next step: after the running tool batch,
    or instead of ending when the model has just answered. Nothing is cut.
``interrupt``
    Cuts the current action now -- the model stream stops at the next
    token, tools not yet started are skipped -- and the run carries on
    with the message. A tool already running cannot be killed; its result
    is kept, and the message follows it.

And two ways to stop:

soft
    Finish the action in progress, take no next one.
force
    Cut the action in progress, as an interrupt does, and end the run.

Controls form a tree: an orchestrator's delegation makes a :meth:`child`
per worker, a stop on the parent reaches every worker below it, and the
node lists the children so each worker can be steered or stopped alone.
"""
from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from typing import Any, Optional

MODE_QUEUE = "queue"
MODE_STEER = "steer"
MODE_INTERRUPT = "interrupt"
MODES = (MODE_QUEUE, MODE_STEER, MODE_INTERRUPT)

STOP_NONE = 0
STOP_SOFT = 1
STOP_FORCE = 2

#: Ids are unique across the whole tree, so a UI holding a mixed list of a
#: parent's and its workers' messages can remove any of them by id alone.
_ids = itertools.count(1)


class RunControl:
    """Pending messages and the stop level for one agent run (and below)."""

    def __init__(self, label: str = "",
                 parent: Optional["RunControl"] = None) -> None:
        self.label = label
        self._parent = parent
        self._lock = threading.RLock()
        self._items: list[dict[str, Any]] = []
        self._stop = STOP_NONE
        self._abort: Optional[Callable[[], None]] = None
        self._children: list[RunControl] = []
        self._listeners: list[Callable[[], None]] = []

    # -- wiring --------------------------------------------------------------

    def bind_abort(self, abort: Optional[Callable[[], None]]) -> None:
        """What "cut the current action" means for this run.

        The loop binds its engine's ``request_stop`` for the length of a
        run: the stream breaks at the next token, and the toolbox's stop
        predicate skips calls that have not started.
        """
        with self._lock:
            self._abort = abort
            pending = abort is not None and (
                self._stop == STOP_FORCE or self._has(MODE_INTERRUPT))
        if pending:
            # Asked before the run could hear it: cut as soon as it can.
            abort()

    def subscribe(self, listener: Callable[[], None]) -> None:
        """Call *listener* (on any thread) whenever the tree changes."""
        with self._lock:
            self._listeners.append(listener)

    def _changed(self) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener()
        if self._parent is not None:
            self._parent._changed()

    def child(self, label: str) -> "RunControl":
        """A control for one delegated worker, stopped with this one."""
        kid = RunControl(label, parent=self)
        with self._lock:
            self._children.append(kid)
        self._changed()
        return kid

    def close(self) -> None:
        """Detach from the parent: the worker's run is over."""
        parent = self._parent
        if parent is None:
            return
        with parent._lock:
            if self in parent._children:
                parent._children.remove(self)
        parent._changed()

    def children(self) -> list["RunControl"]:
        with self._lock:
            return list(self._children)

    # -- the person's side ---------------------------------------------------

    def post(self, text: str, mode: str = MODE_QUEUE) -> Optional[int]:
        """Add a message; its id, or ``None`` for blank text."""
        text = str(text or "").strip()
        if not text:
            return None
        if mode not in MODES:
            mode = MODE_QUEUE
        item = {"id": next(_ids), "text": text, "mode": mode}
        with self._lock:
            self._items.append(item)
            abort = self._abort if mode == MODE_INTERRUPT else None
        if abort is not None:
            abort()
        self._changed()
        return item["id"]

    def remove(self, item_id: int) -> bool:
        """Drop a pending message here or anywhere below; True if found."""
        with self._lock:
            for i, item in enumerate(self._items):
                if item["id"] == item_id:
                    del self._items[i]
                    break
            else:
                return any(kid.remove(item_id) for kid in self.children())
        self._changed()
        return True

    def set_mode(self, item_id: int, mode: str) -> bool:
        """Re-file a pending message (queue it, steer with it, interrupt)."""
        if mode not in MODES:
            return False
        with self._lock:
            item = next((it for it in self._items if it["id"] == item_id), None)
            if item is None:
                return any(kid.set_mode(item_id, mode) for kid in self.children())
            item["mode"] = mode
            abort = self._abort if mode == MODE_INTERRUPT else None
        if abort is not None:
            abort()
        self._changed()
        return True

    def pending(self) -> list[dict[str, Any]]:
        """This run's waiting messages, oldest first (copies)."""
        with self._lock:
            return [dict(item) for item in self._items]

    def request_stop(self, force: bool = False) -> None:
        """Soft: no next action. Force: cut the current one too.

        Reaches every worker below; a soft stop never downgrades a force.
        """
        level = STOP_FORCE if force else STOP_SOFT
        with self._lock:
            raised = level > self._stop
            self._stop = max(self._stop, level)
            abort = self._abort if force else None
            kids = list(self._children)
        if abort is not None:
            abort()
        for kid in kids:
            kid.request_stop(force)
        if raised:
            self._changed()

    def stop_level(self) -> int:
        with self._lock:
            own = self._stop
        parent = self._parent.stop_level() if self._parent is not None else STOP_NONE
        return max(own, parent)

    def reset(self) -> None:
        """A new run starts: forget the last one's stop, keep the messages.

        Steers that missed the last run are the next run's prompt now, so
        they are re-filed as queued.
        """
        with self._lock:
            self._stop = STOP_NONE
            for item in self._items:
                item["mode"] = MODE_QUEUE
        self._changed()

    # -- the loop's side -----------------------------------------------------

    def _has(self, mode: str) -> bool:
        return any(item["mode"] == mode for item in self._items)

    def interrupt_pending(self) -> bool:
        with self._lock:
            return self._has(MODE_INTERRUPT)

    def has_steering(self) -> bool:
        """Is there a steer or an interrupt for the running loop to take?"""
        with self._lock:
            return self._has(MODE_STEER) or self._has(MODE_INTERRUPT)

    def take_steering(self) -> list[dict[str, Any]]:
        """Remove and return the steers and interrupts, oldest first."""
        with self._lock:
            taken = [it for it in self._items if it["mode"] != MODE_QUEUE]
            self._items = [it for it in self._items if it["mode"] == MODE_QUEUE]
        if taken:
            self._changed()
        return taken

    def take_queued(self) -> list[dict[str, Any]]:
        """Remove and return everything waiting (the node, between runs)."""
        with self._lock:
            taken, self._items = self._items, []
        if taken:
            self._changed()
        return taken

    def should_halt_tools(self) -> bool:
        """The toolbox's predicate: start no further call in this batch."""
        return self.stop_level() > STOP_NONE or self.interrupt_pending()


def merge_texts(items: list[dict[str, Any]]) -> str:
    """Several waiting messages as one prompt, in the order they came."""
    return "\n\n".join(str(item.get("text", "")) for item in items
                       if str(item.get("text", "")).strip())
