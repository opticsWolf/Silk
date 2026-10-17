# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The classifier, bound per run (C6, D-S8).

The ``silk_classifier`` port carries a :class:`~.classifier.ClassifierHandle`
into the Agent node. Like the decision seam (D38), the handle is graph-scoped
and the *use* of it is run-scoped: each run takes ``handle.for_run(run_id)``,
which snapshots the node's spec, and binds that view onto the toolset for the
run's duration. The approval gate, the classifier hooks and the ``classify``
tool all read it from there, late, so the gate attached at ToolBox build time
sees the classifier wired *now*.

Both run edges go through this module, so the Agent node, the Orchestrator
and a headless embedder bind and unbind the same way:

    start = begin_classifier_run(toolset, handle, run_id=..., agent_model=...)
    if start.error: refuse the run
    ... run ...
    notices = end_classifier_run(toolset)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Where a run parks its classifier view (the gate, hooks and tool read it).
RUN_CLASSIFIER_ATTR = "_run_classifier"
#: Calls the gate let through on the classifier's word this run.
TRIAGED_ATTR = "_triaged_calls"
#: Whether this module registered ``classify`` for the current run.
_CLASSIFY_OWNED_ATTR = "_classify_registered_by_port"
#: Set by the approval gate when its policy uses the ``triage`` level.
GATE_USES_TRIAGE_ATTR = "_gate_uses_triage"


def bind_run_classifier(toolbox: Any, run_classifier: Any) -> None:
    """Point the toolset at this run's classifier view (or at nothing)."""
    try:
        setattr(toolbox, RUN_CLASSIFIER_ATTR, run_classifier)
        setattr(toolbox, TRIAGED_ATTR, [])
    except AttributeError:      # a toolbox that forbids attributes
        pass


def run_classifier(toolbox: Any) -> Any:
    """This run's classifier view, or ``None``."""
    return getattr(toolbox, RUN_CLASSIFIER_ATTR, None)


def note_triaged(toolbox: Any, entry: dict) -> None:
    """The gate records each call it allowed without asking."""
    calls = getattr(toolbox, TRIAGED_ATTR, None)
    if calls is None:
        calls = []
        try:
            setattr(toolbox, TRIAGED_ATTR, calls)
        except AttributeError:
            return
    calls.append(entry)


def triaged_calls(toolbox: Any) -> list:
    return list(getattr(toolbox, TRIAGED_ATTR, None) or [])


@dataclass
class ClassifierRunStart:
    classifier: Any = None
    notices: list = field(default_factory=list)
    error: str = ""


def begin_classifier_run(toolbox: Any, handle: Any, *, run_id: str,
                         agent_model: Any = None) -> ClassifierRunStart:
    """Bind the classifier for one run; register ``classify`` while wired.

    Refuses (``error``) when the judge model is the model the agent runs on:
    a run that would grade itself does not start (C3). Everything else is a
    notice for the status line, said once per run.
    """
    from .classifier import ClassifierHandle
    from .classifier_local import refuse_self_grading
    from .classify_tool import CLASSIFY_TOOL, register_classify_tool

    start = ClassifierRunStart()
    if handle is not None and not isinstance(handle, ClassifierHandle):
        start.notices.append("The classifier port carries something that is not a "
                             "Classifier; ignored for this run.")
        handle = None
    if handle is not None:
        try:
            refuse_self_grading(handle.judge_model, agent_model)
        except ValueError as exc:
            start.error = f"Classifier: {exc}. Wire a different model into its judge port."
            bind_run_classifier(toolbox, None)
            return start
        start.classifier = handle.for_run(run_id)
    bind_run_classifier(toolbox, start.classifier)

    tools = getattr(toolbox, "tools", None)
    if start.classifier is not None and tools is not None and CLASSIFY_TOOL not in tools:
        register_classify_tool(toolbox)
        setattr(toolbox, _CLASSIFY_OWNED_ATTR, True)

    mode = getattr(start.classifier, "mode", "off") if start.classifier else "off"
    if getattr(toolbox, GATE_USES_TRIAGE_ATTR, False) and mode == "off":
        start.notices.append(
            "The approval policy uses 'triage', but no Classifier is connected (or it is "
            "off): those calls will ask you.")
    elif mode == "shadow" and getattr(toolbox, GATE_USES_TRIAGE_ATTR, False):
        start.notices.append("Classifier in shadow mode: it scores triage-level calls "
                             "and records the result, and you are still asked.")
    return start


def end_classifier_run(toolbox: Any) -> list:
    """Unbind; remove a port-registered ``classify``; summarise what triage did."""
    from .classify_tool import CLASSIFY_TOOL

    notices = []
    calls = triaged_calls(toolbox)
    if calls:
        models = sorted({str(c.get("model", "")) for c in calls if c.get("model")})
        notices.append(f"{len(calls)} call(s) allowed by triage without asking"
                       + (f" ({', '.join(models)})." if models else "."))
    bind_run_classifier(toolbox, None)
    if getattr(toolbox, _CLASSIFY_OWNED_ATTR, False):
        unregister = getattr(toolbox, "unregister", None)
        if callable(unregister):
            unregister(CLASSIFY_TOOL)
        setattr(toolbox, _CLASSIFY_OWNED_ATTR, False)
    return notices


def inherit_run_classifier(parent: Any, child: Any) -> tuple:
    """For delegation: give *child* the parent's run classifier.

    Returns what :func:`restore_run_classifier` needs to put the child back,
    the same push/restore the orchestrator does for depth and chain (D52.2).
    """
    had = hasattr(child, RUN_CLASSIFIER_ATTR)
    prev = getattr(child, RUN_CLASSIFIER_ATTR, None)
    prev_triaged = getattr(child, TRIAGED_ATTR, None)
    rc = run_classifier(parent)
    if rc is not None:
        setattr(child, RUN_CLASSIFIER_ATTR, rc)
        setattr(child, TRIAGED_ATTR, getattr(parent, TRIAGED_ATTR, None) or [])
    return had, prev, prev_triaged, rc is not None


def restore_run_classifier(child: Any, saved: tuple) -> None:
    had, prev, prev_triaged, pushed = saved
    if not pushed:
        return
    if had:
        setattr(child, RUN_CLASSIFIER_ATTR, prev)
        setattr(child, TRIAGED_ATTR, prev_triaged)
    else:
        for attr in (RUN_CLASSIFIER_ATTR, TRIAGED_ATTR):
            try:
                delattr(child, attr)
            except AttributeError:
                pass
