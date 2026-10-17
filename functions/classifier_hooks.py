# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Catalog hooks that use the classifier (C4). Optional, Role-selectable.

None of these decides whether a call runs - that is the gate's job, and the
gate holds its classifier step itself (C2). These hooks advise, bounce once,
or file things for later:

``task_drift``      wrap_tool_execute on ``task_add``: a task the classifier
                    thinks is not needed for the goal, or duplicates one, is
                    bounced **once** with a request to say how it serves the
                    goal. The retry goes through.
``prose_question``  after_run: a run that ended by asking the user something
                    in plain text files that question as a deferred question
                    on the task (S3), so it shows up in the Inbox.
``findings_gate``   after_run: user decisions with a reason are classified
                    (category / durable / grounded / relation); good ones go to
                    the OKFgraph producer, borderline ones to an Inbox card.

Each hook is inert - passes through, writes nothing - when no classifier is
bound, when it is ``off``, or when no backend answered. In ``shadow`` mode a
hook records what it would have done and does nothing else.

Hooks get their run-scoped dependencies from a :class:`HookContext` of
resolvers over the toolbox they are registered on (:func:`context_for`): the
classifier, plan and ledger are bound per run, after the catalog or a Role
built the hook map -- the same late binding the gate uses for its seam. The
toolbox reaches them through :class:`~.hooks.ToolboxBound`.

``prose_question`` needs the work ledger (S1/S3) and ``findings_gate`` the
decision log (S2) and a knowledge sink (S6); until a toolbox carries those,
the hooks are selectable and do nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from weave.logger import get_logger

from .hooks import HOOK_AFTER_RUN, HOOK_WRAP_TOOL_EXECUTE, HookEntry, ToolboxBound

log = get_logger("SilkClassifierHooks")


@dataclass
class HookContext:
    classifier: Callable[[], Any] = lambda: None          # RunClassifier | None
    goal: Callable[[], str] = lambda: ""
    task_titles: Callable[[], list] = lambda: []
    current_task: Callable[[], Optional[str]] = lambda: None
    ledger: Callable[[], Any] = lambda: None              # WorkLedger | None
    run_id: Callable[[], str] = lambda: ""
    recent_decisions: Callable[[str], list] = lambda run_id: []
    nearest_rule: Callable[[str], Optional[dict]] = lambda text: None
    write_finding: Callable[[dict], Any] = lambda note: None
    offer_finding: Callable[[dict], Any] = lambda note: None
    shadow_log: list = field(default_factory=list)       # what shadow mode would have done


def _live(ctx: HookContext):
    c = ctx.classifier()
    if c is None or getattr(c, "mode", "off") == "off":
        return None
    return c


# ── task_drift ───────────────────────────────────────────────────────────────

class TaskDriftConfig(BaseModel):
    bounce_below: float = Field(0.3, ge=0, le=1, description="Bounce when P(needed) is below.")
    duplicate_above: float = Field(0.8, ge=0, le=1, description="Bounce when P(duplicate) is above.")


DRIFT_QUESTIONS = {
    "needed": {"type": "noul", "instructions":
               "Is the new task needed to reach the goal given in the state?"},
    "duplicate": {"type": "noul", "instructions":
                  "Does the new task duplicate one of the existing tasks in the state?"},
}


def make_task_drift(config: Optional[BaseModel], ctx: HookContext):
    cfg = config if isinstance(config, TaskDriftConfig) else TaskDriftConfig()
    bounced: set[str] = set()

    async def drift(handler: Optional[Callable] = None, tool_name: str = "",
                    tool_args: Optional[dict] = None, **_kw: Any) -> Any:
        if handler is None:
            raise TypeError("task_drift is middleware: it wraps a handler")
        c = _live(ctx)
        args = dict(tool_args or {})
        title = str(args.get("title", "")).strip()
        if c is None or not title or title.lower() in bounced:
            return await handler()
        v = c.ask({"goal": ctx.goal(), "existing_tasks": ctx.task_titles()[:40],
                   "new_task": title, "rationale": str(args.get("rationale", ""))},
                  DRIFT_QUESTIONS, purpose="hook")
        if not v.ok:
            return await handler()
        needed, dup = v.noul("needed"), v.noul("duplicate")
        if needed >= cfg.bounce_below and dup <= cfg.duplicate_above:
            return await handler()
        if c.mode == "shadow":
            ctx.shadow_log.append({"hook": "task_drift", "title": title, "needed": needed,
                                   "duplicate": dup, "would": "bounce"})
            return await handler()
        bounced.add(title.lower())                 # once per title: the retry goes through
        why = ("it looks like a duplicate of an existing task" if dup > cfg.duplicate_above
               else "it does not look needed for the goal")
        return json.dumps({
            "error": f"Not added: {why}.",
            "suggestion": "If it is needed, call task_add again with a rationale that says "
                          "how it serves the goal; otherwise continue with the existing tasks.",
            "advisory": {"needed": round(needed, 3), "duplicate": round(dup, 3),
                         "model": v.model},
        }, ensure_ascii=False)

    return {HOOK_WRAP_TOOL_EXECUTE: [HookEntry(callback=drift, tools=frozenset({"task_add"}))]}


# ── prose_question ───────────────────────────────────────────────────────────

class ProseQuestionConfig(BaseModel):
    threshold: float = Field(0.8, ge=0, le=1)
    min_confidence: float = Field(0.6, ge=0, le=1)
    max_chars: int = Field(600, ge=80)


PROSE_QUESTIONS = {
    "ending": {"type": "choice", "instructions": "How does this final reply end?",
               "criteria": {
                   "asks_user": "it asks the user to decide, choose, confirm or clarify something",
                   "reports": "it reports results or progress and asks nothing",
                   "other": None}},
}


def _tail(text: str, n: int) -> str:
    paras = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return (paras[-1] if paras else text.strip())[-n:]


def make_prose_question(config: Optional[BaseModel], ctx: HookContext):
    cfg = config if isinstance(config, ProseQuestionConfig) else ProseQuestionConfig()

    def after(final_text: str = "", **_kw: Any) -> None:
        c, task, led = _live(ctx), ctx.current_task(), ctx.ledger()
        if c is None or not task or led is None or not (final_text or "").strip():
            return
        v = c.ask({"reply_end": _tail(final_text, cfg.max_chars)}, PROSE_QUESTIONS,
                  purpose="hook")
        if not v.ok:
            return
        a = v.answers["ending"]
        p = float(a["probabilities"].get("asks_user", 0.0))
        if a["choice"] != "asks_user" or p < cfg.threshold or a["confidence"] < cfg.min_confidence:
            return
        prompt = _tail(final_text, cfg.max_chars)
        if c.mode == "shadow":
            ctx.shadow_log.append({"hook": "prose_question", "task": task, "would": "defer",
                                   "p": p})
            return
        led.defer_question(task, prompt=prompt, kind="question", tool="ask_user",
                           run_id=ctx.run_id(), triage={"hook": "prose_question",
                                                        "p": round(p, 3), "model": v.model})

    return {HOOK_AFTER_RUN: [HookEntry(callback=after)]}


# ── findings_gate ────────────────────────────────────────────────────────────

class FindingsGateConfig(BaseModel):
    category: float = Field(0.80, ge=0, le=1)
    durable: float = Field(0.70, ge=0, le=1)
    grounded: float = Field(0.70, ge=0, le=1)
    ask_floor: float = Field(0.50, ge=0, le=1)


FINDINGS_QUESTIONS = {
    "category": {"type": "choice", "instructions": "What kind of knowledge does this decision "
                 "express, if it were written down for future work?",
                 "criteria": {"dogma": "a value or principle the user holds",
                              "rule": "a standing rule for how work should be done",
                              "evidence": "an observed fact about this project",
                              "decision": "a choice made for a specific situation that will recur",
                              "none": "nothing worth keeping"}},
    "durable": {"type": "noul", "instructions":
                "Will this still be true and useful in a month?"},
    "grounded": {"type": "noul", "instructions":
                 "Is it stated by the user, or directly supported by the state, rather than guessed?"},
    "relation": {"type": "choice", "instructions":
                 "How does it relate to the nearest existing rule in the state?",
                 "criteria": {"supersedes": None, "supports": None, "contradicts": None,
                              "unrelated": None}},
}


def route_finding(answers: dict, cfg: FindingsGateConfig) -> str:
    cat = answers["category"]
    kind, p_kind = cat["choice"], float(cat["probabilities"].get(cat["choice"], 0.0))
    dur, gnd = float(answers["durable"]["noul"]), float(answers["grounded"]["noul"])
    rel = answers["relation"]["choice"]
    if kind not in ("rule", "decision", "dogma"):
        return "drop"
    if rel == "contradicts":
        return "ask"                          # never overwrite a rule silently
    if p_kind >= cfg.category and dur >= cfg.durable and gnd >= cfg.grounded:
        return "auto"
    if min(p_kind, dur, gnd) >= cfg.ask_floor:
        return "ask"
    return "drop"


def make_findings_gate(config: Optional[BaseModel], ctx: HookContext):
    cfg = config if isinstance(config, FindingsGateConfig) else FindingsGateConfig()

    def after(**_kw: Any) -> None:
        c = _live(ctx)
        if c is None:
            return
        for d in ctx.recent_decisions(ctx.run_id()):
            if d.get("actor") != "user" or not str(d.get("reason", "")).strip():
                continue
            nearest = ctx.nearest_rule(d.get("prompt", ""))
            v = c.ask({"decision": d.get("prompt", ""), "outcome": d.get("outcome"),
                       "reason": d.get("reason"), "nearest_rule": nearest},
                      FINDINGS_QUESTIONS, purpose="hook")
            if not v.ok:
                continue
            route = route_finding(v.answers, cfg)
            note = {"title": d.get("reason", "")[:80], "body": d.get("reason", ""),
                    "category": v.answers["category"]["choice"],
                    "source_decision": d.get("id"), "model": v.model,
                    "relations": ({v.answers["relation"]["choice"]: nearest["id"]}
                                  if nearest and v.answers["relation"]["choice"]
                                  in ("supersedes", "supports", "contradicts") else {})}
            if c.mode == "shadow":
                if route != "drop":
                    ctx.shadow_log.append({"hook": "findings_gate", "would": route, "note": note})
                continue
            if route == "auto":
                ctx.write_finding(note)
            elif route == "ask":
                ctx.offer_finding(note)

    return {HOOK_AFTER_RUN: [HookEntry(callback=after)]}


# ── the toolbox the hooks run on ─────────────────────────────────────────────

def context_for(toolbox: Any) -> HookContext:
    """Resolvers over *toolbox*'s run bindings. Read at call time, never cached:
    the classifier, plan and identity are bound per run."""
    from .classifier_binding import run_classifier
    from .remember import run_identity

    def plan():
        store = getattr(toolbox, "_task_store", None)
        try:
            return store.load() if store is not None else None
        except Exception:                                   # a broken store is no plan
            return None

    def current_task() -> Optional[str]:
        p = plan()
        if p is None:
            return None
        active = [t for t in p.tasks if t.status == "in_progress"]
        if len(active) == 1:
            return active[0].id
        me = run_identity(toolbox).agent
        mine = [t for t in active if getattr(t, "claimed_by", None) == me]
        return mine[0].id if len(mine) == 1 else None

    knowledge = lambda: getattr(toolbox, "_knowledge", None)           # noqa: E731  (S6)
    decisions = lambda: getattr(toolbox, "_decision_log", None)        # noqa: E731  (S2)

    return HookContext(
        classifier=lambda: run_classifier(toolbox),
        goal=lambda: (plan().goal.text if plan() is not None else ""),
        task_titles=lambda: ([t.title for t in plan().tasks] if plan() is not None else []),
        current_task=current_task,
        ledger=lambda: getattr(toolbox, "_work_ledger", None),        # S1/S3
        run_id=lambda: run_identity(toolbox).run_id,
        recent_decisions=lambda run: (decisions().for_run(run) if decisions() else []),
        nearest_rule=lambda text: (knowledge().nearest_rule(text) if knowledge() else None),
        write_finding=lambda note: (knowledge().write_finding(note) if knowledge() else None),
        offer_finding=lambda note: (knowledge().offer_finding(note) if knowledge() else None),
    )


def _toolbox_bound(make_map: Callable[[Optional[BaseModel], HookContext], dict],
                   config: Optional[BaseModel], name: str) -> dict:
    """Turn a ``make_*(config, ctx)`` map into one whose callbacks bind to the
    toolbox they are registered on (:class:`~.hooks.ToolboxBound`)."""
    proto = make_map(config, HookContext())         # shape only: events + bindings
    out: dict = {}
    for event, entries in proto.items():
        for i, entry in enumerate(entries):
            def make(toolbox, _event=event, _i=i):
                return make_map(config, context_for(toolbox))[_event][_i].callback
            out.setdefault(event, []).append(HookEntry(
                callback=ToolboxBound(make, name), tools=entry.tools,
                categories=entry.categories, essential=entry.essential))
    return out


def task_drift_hooks(config: Optional[BaseModel] = None) -> dict:
    return _toolbox_bound(make_task_drift, config, "task_drift")


def prose_question_hooks(config: Optional[BaseModel] = None) -> dict:
    return _toolbox_bound(make_prose_question, config, "prose_question")


def findings_gate_hooks(config: Optional[BaseModel] = None) -> dict:
    return _toolbox_bound(make_findings_gate, config, "findings_gate")
