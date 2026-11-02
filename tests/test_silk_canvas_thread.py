# -*- coding: utf-8 -*-
"""The canvas seam is served on the main thread wherever it is built (D70).

The Agent node builds its ``CanvasAuthor`` inside ``compute``, on a pool
worker. A QObject's queued slots run on the thread it lives on, so an
author left on the worker queued every request back to the thread that
was blocked waiting for it: ``list_placeable_nodes`` and ``describe_graph``
came back "the canvas did not respond in time" after the full timeout.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6.QtWidgets", reason="Qt required")

from PySide6.QtCore import QCoreApplication, QRunnable, QThreadPool  # noqa: E402

from silk.functions.blocking_seam import CAUSE_ANSWERED  # noqa: E402
from silk.functions.main_thread_call import MainThreadCall  # noqa: E402
from silk.nodes.graph_canvas import CanvasAuthor  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class _NoScene:
    """A node with no canvas: the op refuses, but the main thread answers."""

    def scene(self):
        return None


def test_an_author_built_on_a_worker_is_served_by_the_main_thread(qapp):
    out: dict = {}

    class Run(QRunnable):
        def run(self):
            seam = MainThreadCall(timeout_s=3.0)
            author = CanvasAuthor(_NoScene(), seam)
            seam.attach(author.deliver)
            out["lives_on_main"] = author.thread() is qapp.thread()
            out["result"] = seam.call("describe_graph")
            out["author"] = author          # keep it alive past the call

    QThreadPool.globalInstance().start(Run())
    deadline = time.monotonic() + 5.0
    while "result" not in out and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)

    assert out.get("lives_on_main") is True
    result = out["result"]
    assert result.cause == CAUSE_ANSWERED, (
        f"the main thread never answered: {result.failure_text()}")
    assert not result.ok, "no canvas: refused by the main thread, not lost"
