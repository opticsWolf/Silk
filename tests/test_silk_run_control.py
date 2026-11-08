# -*- coding: utf-8 -*-
"""Steer, interrupt, queue and the two stops (Qt-free).

RunControl on its own, then through AgentLoop with a fake engine whose
stream honours the stop flag the way GraphEngine's does, and a fake
toolbox whose calls can post to the control mid-batch -- which is when a
person presses something.
"""

from __future__ import annotations

import json

from silk.functions.agent_loop import AgentLoop
from silk.functions.reflection import ReflectionConfig
from silk.functions.run_control import (
    MODE_INTERRUPT, MODE_QUEUE, MODE_STEER, STOP_FORCE, STOP_NONE, STOP_SOFT,
    RunControl, merge_texts,
)
from silk.functions.stream_events import (
    OUTCOME_COMPLETED, OUTCOME_STOPPED, EventRunResult, EventSteered,
    EventToolCall,
)
from silk.functions.usage_limits import UsageLimits

FENCE = '```tool_call\n{"name": "%s", "arguments": {}}\n```'


class Engine:
    """Scripted engine; a response may be a callable run before streaming."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.usage_limits = UsageLimits()
        self.reflection_config = ReflectionConfig(max_retries=2)
        self.history: list[dict] = []
        self.last_stats: dict = {}
        self._stop = False
        self.requests = 0
        #: Called with the word index while streaming (to press mid-answer).
        self.on_word = None

    def stream_response(self, gen_params):
        self.requests += 1
        text = self._responses.pop(0) if self._responses else "(exhausted)"
        finish = "stop"
        for i, word in enumerate(text.split(" ")):
            if self.on_word is not None:
                self.on_word(i)
            if self._stop:
                finish = "stopped"
                break
            yield word + " "
        self.last_stats = {"tokens": 1, "input_tokens": 1, "tps": 1.0,
                           "finish_reason": finish}

    def append_message(self, role, content, **stats):
        self.history.append({"role": role, "content": content, **stats})

    def count_prompt_tokens(self):
        return 1

    def request_stop(self):
        self._stop = True

    def clear_stop(self):
        self._stop = False

    def stop_requested(self):
        return self._stop


class Toolbox:
    def __init__(self, actions=None):
        self.tools = {"a": {}, "b": {}}
        self.actions = actions or {}
        self.executed: list[str] = []
        self._should_stop = None

    def bind_stop(self, predicate):
        self._should_stop = predicate

    async def execute_tool_calls_async(self, calls):
        out = []
        for tc in calls:
            name = tc.function.name
            if self._should_stop is not None and self._should_stop():
                body = "Not run: stopped."
            else:
                self.executed.append(name)
                action = self.actions.get(name)
                if action is not None:
                    action()
                body = json.dumps({"ok": name})
            out.append({"tool_call_id": tc.id, "name": name, "content": body})
        return out


def two_calls() -> str:
    return "calling\n" + FENCE % "a" + "\n" + FENCE % "b"


def run(engine, toolbox, control):
    return list(AgentLoop(engine, toolbox, control=control).run("go", {}))


def result(events) -> EventRunResult:
    return [e for e in events if isinstance(e, EventRunResult)][-1]


def user_turns(engine) -> list[str]:
    return [m["content"] for m in engine.history if m["role"] == "user"]


# ── RunControl on its own ─────────────────────────────────────────────


def test_post_remove_and_refile():
    control = RunControl()
    a = control.post("  one ")
    b = control.post("two", MODE_STEER)
    assert control.post("   ") is None
    assert [i["text"] for i in control.pending()] == ["one", "two"]
    assert control.set_mode(a, MODE_STEER)
    assert control.remove(b) and not control.remove(b)
    assert control.pending() == [{"id": a, "text": "one", "mode": MODE_STEER}]


def test_interrupt_cuts_through_the_bound_abort():
    cut = []
    control = RunControl()
    control.post("x", MODE_INTERRUPT)          # before a run can hear it
    control.bind_abort(lambda: cut.append(1))
    assert cut == [1]                          # cut as soon as it can
    control.post("y", MODE_STEER)
    assert cut == [1]
    control.set_mode(control.pending()[1]["id"], MODE_INTERRUPT)
    assert cut == [1, 1]


def test_stops_reach_workers_and_never_downgrade():
    parent = RunControl("orch")
    kid = parent.child("coder")
    cut = []
    kid.bind_abort(lambda: cut.append("kid"))
    parent.request_stop(force=False)
    assert kid.stop_level() == STOP_SOFT and cut == []
    parent.request_stop(force=True)
    assert kid.stop_level() == STOP_FORCE and cut == ["kid"]
    parent.request_stop(force=False)
    assert parent.stop_level() == STOP_FORCE
    kid.close()
    assert parent.children() == []


def test_remove_reaches_a_worker_message_and_changes_are_heard():
    heard = []
    parent = RunControl()
    parent.subscribe(lambda: heard.append(1))
    kid = parent.child("w")
    item = kid.post("hint", MODE_STEER)
    assert parent.remove(item) and kid.pending() == []
    assert len(heard) >= 3                     # child, post, remove


def test_reset_requeues_late_steers():
    control = RunControl()
    control.post("late", MODE_STEER)
    control.request_stop()
    control.reset()
    assert control.stop_level() == STOP_NONE
    assert control.pending()[0]["mode"] == MODE_QUEUE
    assert merge_texts(control.take_queued() + [{"text": "new"}]) == "late\n\nnew"


# ── Through the loop ──────────────────────────────────────────────────


def test_steer_joins_after_the_tool_batch():
    control = RunControl()
    engine = Engine([two_calls(), "done"])
    box = Toolbox({"a": lambda: control.post("prefer b", MODE_STEER)})
    events = run(engine, box, control)
    assert box.executed == ["a", "b"]          # a steer cuts nothing
    steered = [e for e in events if isinstance(e, EventSteered)]
    assert [(s.text, s.interrupted) for s in steered] == [("prefer b", False)]
    assert user_turns(engine)[-1] == "prefer b"
    assert result(events).outcome == OUTCOME_COMPLETED


def test_steer_keeps_a_finished_answer_from_being_the_last_word():
    control = RunControl()
    engine = Engine(["first answer", "second answer"])
    engine.on_word = lambda i: (control.post("also check y", MODE_STEER)
                                if engine.requests == 1 and i == 0 else None)
    events = run(engine, None, control)
    assert engine.requests == 2
    assert result(events).text.strip() == "second answer"


def test_interrupt_cuts_the_stream_and_carries_on():
    control = RunControl()
    engine = Engine(["a long answer that goes " + FENCE % "a", "fine"])
    engine.on_word = lambda i: (control.post("stop, do z", MODE_INTERRUPT)
                                if engine.requests == 1 and i == 2 else None)
    box = Toolbox()
    events = run(engine, box, control)
    assert box.executed == []                  # the half-built call is dropped
    assert not [e for e in events if isinstance(e, EventToolCall)]
    cut = [m for m in engine.history if m.get("finish_reason") == "interrupted"]
    assert cut and cut[0]["content"].strip() == "a long"
    steered = [e for e in events if isinstance(e, EventSteered)]
    assert steered and steered[0].interrupted
    assert result(events).outcome == OUTCOME_COMPLETED
    assert not engine.stop_requested()


def test_interrupt_during_tools_skips_the_rest_of_the_batch():
    control = RunControl()
    engine = Engine([two_calls(), "ok"])
    box = Toolbox({"a": lambda: control.post("enough", MODE_INTERRUPT)})
    events = run(engine, box, control)
    assert box.executed == ["a"]               # b never started
    assert user_turns(engine)[-1] == "enough"
    assert result(events).outcome == OUTCOME_COMPLETED


def test_soft_stop_finishes_the_action_then_ends():
    control = RunControl()
    engine = Engine([two_calls(), "never asked"])
    box = Toolbox({"a": lambda: control.request_stop()})
    events = run(engine, box, control)
    assert box.executed == ["a"]               # no next action
    assert engine.requests == 1
    assert result(events).outcome == OUTCOME_STOPPED


def test_soft_stop_lets_the_answer_finish():
    control = RunControl()
    engine = Engine(["the whole answer here"])
    engine.on_word = lambda i: control.request_stop() if i == 1 else None
    events = run(engine, None, control)
    assert result(events).text.strip() == "the whole answer here"


def test_force_stop_cuts_the_stream():
    control = RunControl()
    engine = Engine(["one two three four " + FENCE % "a"])
    engine.on_word = lambda i: control.request_stop(force=True) if i == 1 else None
    box = Toolbox()
    events = run(engine, box, control)
    assert box.executed == []
    assert result(events).text.strip() == "one"


def test_force_stop_during_tools_is_stopped_not_completed():
    engine = Engine([two_calls(), "never"])
    box = Toolbox({"a": engine.request_stop})  # e.g. the node's cancel
    events = run(engine, box, None)
    assert box.executed == ["a"]
    assert engine.requests == 1
    assert result(events).outcome == OUTCOME_STOPPED
