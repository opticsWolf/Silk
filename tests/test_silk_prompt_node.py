# -*- coding: utf-8
"""The chat loop: Prompt node → Agent (run / stop / clear) → Chat Log.

Qt node tests with in-test imports, so the file still collects where
PySide6 is missing. Runs never start: the agent's execute and cancel are
replaced, since what is under test is which one a pulse reaches.
"""

from __future__ import annotations

import pytest

from silk.functions.stream_events import (
    EventContextCleared, EventType, to_wire,
)


@pytest.fixture(autouse=True)
def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])


@pytest.fixture()
def prompt():
    from silk.nodes.prompt import SilkPromptNode
    node = SilkPromptNode()
    node.pulses = []
    node.pulse = lambda port, payload=None: node.pulses.append((port, payload))
    yield node
    node.cleanup()


@pytest.fixture()
def agent():
    from silk.nodes.agent import SilkAgentNode
    node = SilkAgentNode()
    node.calls = []
    node.execute = lambda: node.calls.append("execute")
    node.cancel_compute = lambda: node.calls.append("cancel")
    node.wire = []
    node._fan_out_stream = lambda port, value: node.wire.append((port, value))
    yield node
    node._is_computing = False
    node._run_queued = False
    node._clear_pending = False
    node._stop.reset()
    node._control.take_queued()
    # A node left to the garbage collector takes its graphics items down
    # at a random later moment, mid-way through building the next one.
    node.cleanup()


def test_context_cleared_is_part_of_the_vocabulary():
    wire = to_wire(EventContextCleared(turns_dropped=3), run_id="", seq=0)
    assert wire["type"] == EventType.CONTEXT_CLEARED.value == "context.cleared"
    assert wire["turns_dropped"] == 3


# ── Prompt node ──────────────────────────────────────────────────────


def test_send_pulses_the_message_and_empties_the_box(prompt):
    prompt._editor.setPlainText("  summarise the plan  ")
    prompt.send()
    assert prompt.pulses == [("send", "summarise the plan")]
    assert prompt._editor.toPlainText() == ""
    assert prompt.compute({}) == {"prompt": "summarise the plan"}


def test_empty_box_sends_nothing(prompt):
    prompt._editor.setPlainText("   \n ")
    prompt.send()
    assert prompt.pulses == []


def test_stop_and_new_chat_pulse_their_ports(prompt):
    prompt.stop()
    prompt.new_chat()
    assert [p for p, _ in prompt.pulses] == ["stop", "new_chat"]


def test_prompt_stop_is_press_twice(prompt):
    prompt.stop()
    prompt.stop()
    assert prompt.pulses == [("stop", True), ("stop", "force")]
    prompt.force_stop()
    assert prompt.pulses[-1] == ("stop", "force")


def test_steer_and_interrupt_send_a_mode(prompt):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    for mods, mode in (
        (Qt.KeyboardModifier.ControlModifier, "steer"),
        (Qt.KeyboardModifier.ControlModifier
         | Qt.KeyboardModifier.ShiftModifier, "interrupt"),
    ):
        prompt._editor.setPlainText("now this")
        key = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, mods)
        assert prompt.eventFilter(prompt._editor, key)
        assert prompt.pulses[-1] == ("send", {"text": "now this", "mode": mode})


def test_history_steps_back_and_returns_to_the_draft(prompt):
    for text in ("one", "two"):
        prompt._editor.setPlainText(text)
        prompt.send()
    prompt._editor.setPlainText("draft")
    prompt._step_history(-1)
    assert prompt._editor.toPlainText() == "two"
    prompt._step_history(-1)
    prompt._step_history(-1)          # clamps at the oldest
    assert prompt._editor.toPlainText() == "one"
    prompt._step_history(+1)
    prompt._step_history(+1)
    assert prompt._editor.toPlainText() == "draft"


def test_state_keeps_history_sent_and_draft(prompt):
    from silk.nodes.prompt import SilkPromptNode
    prompt._editor.setPlainText("first")
    prompt.send()
    prompt._editor.setPlainText("half-typed")
    saved = prompt.get_state()
    other = SilkPromptNode()
    try:
        other.restore_state(saved)
        assert other._history == ["first"]
        assert other.compute({}) == {"prompt": "first"}
        assert other._editor.toPlainText() == "half-typed"
    finally:
        other.cleanup()


# ── Agent exec inputs ────────────────────────────────────────────────


def test_run_pulse_carries_the_prompt(agent):
    agent.on_pulse("run", None, payload="  hello  ")
    assert agent.calls == ["execute"]
    assert agent._next_prompt == "hello"


def test_run_pulse_without_text_falls_back_to_the_wire(agent):
    agent.on_pulse("run", None, payload=True)     # an upstream `done`
    assert agent.calls == ["execute"]
    assert agent._next_prompt is None


def test_run_during_a_run_queues_each_message(agent):
    agent._is_computing = True
    agent.on_pulse("run", None, payload="first")
    agent.on_pulse("run", None, payload="second")
    assert agent.calls == []                       # never stops the run
    assert [(i["text"], i["mode"]) for i in agent._control.pending()] == [
        ("first", "queue"), ("second", "queue")]


def test_steer_and_interrupt_payloads_join_the_run(agent):
    cut = []
    agent._control.bind_abort(lambda: cut.append(True))
    agent._is_computing = True
    agent.on_pulse("run", None, payload={"text": "use X", "mode": "steer"})
    assert not cut                                 # a steer cuts nothing
    agent.on_pulse("run", None, payload={"text": "stop that", "mode": "interrupt"})
    assert cut == [True]                           # an interrupt cuts now
    assert [i["mode"] for i in agent._control.pending()] == ["steer", "interrupt"]
    assert agent.calls == []


def test_steer_while_idle_just_runs(agent):
    agent.on_pulse("run", None, payload={"text": "go", "mode": "steer"})
    assert agent.calls == ["execute"] and agent._next_prompt == "go"


def test_stop_is_press_twice(agent):
    agent._is_computing = True
    agent.on_pulse("run", None, payload="later")
    agent.on_pulse("stop", None, payload=True)
    # First: after the current action -- no cancel, the queue is kept.
    assert agent.calls == []
    assert agent._control.stop_level() == 1 and agent._stop.stopping
    assert agent.btn_run.label() in ("Force Stop", "Stopping")
    agent.on_pulse("stop", None, payload=True)
    assert agent.calls == ["cancel"]               # second: now
    assert [i["text"] for i in agent._control.pending()] == ["later"]


def test_force_payload_skips_the_first_step(agent):
    agent._is_computing = True
    agent.on_pulse("stop", None, payload="force")
    assert agent.calls == ["cancel"]


def test_queued_messages_run_after_the_run_ends(agent, monkeypatch):
    from PySide6.QtCore import QTimer
    scheduled = []
    monkeypatch.setattr(QTimer, "singleShot",
                        lambda _ms, fn: scheduled.append(fn))
    agent._is_computing = True
    agent.on_pulse("run", None, payload="next one")
    agent._cleanup_after_worker()
    assert len(scheduled) == 1
    assert [i["text"] for i in agent._control.pending()] == ["next one"]


def test_a_stopped_run_holds_the_queue(agent, monkeypatch):
    from PySide6.QtCore import QTimer
    scheduled = []
    monkeypatch.setattr(QTimer, "singleShot",
                        lambda _ms, fn: scheduled.append(fn))
    agent._is_computing = True
    agent.on_pulse("run", None, payload={"text": "late steer", "mode": "steer"})
    agent.on_pulse("stop", None, payload=True)
    agent._cleanup_after_worker()
    assert scheduled == []                         # no auto-run after a stop
    # The steer missed its run: it waits as the next run's prompt.
    assert [(i["text"], i["mode"]) for i in agent._control.pending()] == [
        ("late steer", "queue")]
    assert agent._control.stop_level() == 0


def test_start_sends_the_waiting_messages_first():
    from silk.nodes.agent import SilkAgentNode
    from weave.node.threaded import ThreadedManualNode
    node = SilkAgentNode()
    started = []
    original = ThreadedManualNode.execute
    ThreadedManualNode.execute = lambda self: started.append(self._next_prompt)
    try:
        node._control.post("queued one")
        node._next_prompt = "fresh"
        node.execute()
        assert started == ["queued one\n\nfresh"]
        assert node._control.pending() == []
    finally:
        ThreadedManualNode.execute = original
        node._stop.reset()


def test_at_worker_goes_to_that_running_worker(agent):
    agent._is_computing = True
    kid = agent._control.child("coder")
    agent.on_pulse("run", None, payload="@coder use the cache")
    assert [(i["text"], i["mode"]) for i in kid.pending()] == [
        ("use the cache", "steer")]               # a worker has no next run
    assert agent._control.pending() == []
    agent.on_pulse("run", None, payload="@nobody hello")
    assert [i["text"] for i in agent._control.pending()] == ["@nobody hello"]
    kid.close()


def test_queue_view_lists_messages_and_workers(agent):
    view = agent._queue_view
    view.refresh()
    assert view.isHidden()
    item = agent._control.post("one")
    kid = agent._control.child("coder")
    kid.post("two", "steer")
    view.refresh()
    assert not view.isHidden()
    assert agent._control.remove(item) and kid.remove(kid.pending()[0]["id"])
    kid.close()
    view.refresh()
    assert view.isHidden()


def test_clear_pulse_forgets_and_tells_the_wire(agent):
    agent._history[:] = [{"role": "user", "content": "x"},
                         {"role": "assistant", "content": "y"}]
    agent.on_pulse("clear", None, payload=True)
    assert agent._history == []
    [(port, wire)] = agent.wire
    assert port == "events"
    assert wire["type"] == "context.cleared" and wire["turns_dropped"] == 2


def test_clear_during_a_run_waits_for_it(agent):
    agent._history[:] = [{"role": "user", "content": "x"}]
    agent._is_computing = True
    agent.clear_context()
    assert agent._history and agent._clear_pending and not agent.wire
    agent._cleanup_after_worker()
    assert agent._history == [] and not agent._clear_pending
    assert agent.wire[0][1]["type"] == "context.cleared"


def test_run_button_toggles_between_start_and_stop():
    from silk.nodes.agent import SilkAgentNode
    node = SilkAgentNode()
    assert node.btn_run.label() == "Start"
    assert node.btn_clear.label() == "Clear"
    assert {"stop", "clear", "run"} <= {p.name for p in node.inputs}


def test_old_task_box_and_preview_are_dropped_on_load():
    from silk.nodes.agent import SilkAgentNode
    state = {"widget_data": {"user_prompt": "old task",
                             "preview_display": "<i>…</i>",
                             "budget": "requests=3"}}
    out = SilkAgentNode.migrate_state(state, 4)
    assert out["widget_data"] == {"budget": "requests=3"}


# ── Chat Log ─────────────────────────────────────────────────────────


def test_chat_log_starts_over_on_context_cleared():
    from silk.nodes.chat_display import ChatDisplayNode
    node = ChatDisplayNode()
    try:
        assert node._append_turn({"type": "chat.turn", "turn_id": "t1",
                                  "user": "hi", "ai": "hello"})
        assert "hello" in node._chat_log_md
        assert node._append_turn(to_wire(EventContextCleared(),
                                         run_id="", seq=0))
        assert node._chat_log_md == "" and node._last_turn_id is None
    finally:
        node.cleanup()
