# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The LLM-callable ``classify`` tool (C5).

For the user's own work: sort issues, triage emails, label rows, check a list
of claims against a source - many items, typed answers, cheap.

Kept apart from approvals by construction:

* it asks with ``purpose="tool"``, so its answers sit in their own cache
  namespace and can never be read back as a gate answer;
* the gate's triage ``state`` is built from the ledger, OKFgraph and the
  call's own arguments - never from the transcript, where this tool's results
  live (``triage.build_state``);
* every result says ``"advisory": true`` and names the model.

It is registered only when a classifier is connected (the port decides the
catalog, D6-style), sits in category ``classifier``, has a per-run call budget
and a per-call item cap, and hosted backends are used only when the project
allowed egress.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .classifier import QuestionError, validate_questions

MAX_QUESTIONS = 8
CLASSIFY_TOOL = "classify"


def run_classifier(toolbox: Any) -> Any:
    from .classifier_binding import run_classifier as _rc
    return _rc(toolbox)


#: What the Role's catalog shows while no classifier is wired, so a Role can
#: allow the tool before the port exists (``toolset_build.with_port_tools``).
CATALOG_ENTRY = {
    "name": CLASSIFY_TOOL,
    "description": ("Classify items with typed questions. Present only while a Classifier "
                    "node is wired to the agent; advisory."),
    "parameters": {},
    "category": "classifier",
    "tags": ["classifier", "port"],
    "risk": "low",
}


class ClassifyArgs(BaseModel):
    items: list[str] = Field(..., min_length=1,
                             description="The texts to classify, one answer set per item.")
    questions: dict[str, dict] = Field(..., description=(
        "Named questions. Each: {type: noul|choice|score, instructions: str, criteria?}. "
        "noul = yes/no probability; choice = criteria {option: description}; "
        "score = criteria [lowest, ..., highest] (2-10 levels)."))
    context: str = Field("", description="Shared context given with every item.")

    @field_validator("questions")
    @classmethod
    def _valid(cls, q: dict) -> dict:
        if len(q) > MAX_QUESTIONS:
            raise ValueError(f"at most {MAX_QUESTIONS} questions per call")
        try:
            return validate_questions(q)
        except QuestionError as exc:
            raise ValueError(str(exc)) from exc


def register_classify_tool(toolbox: Any) -> None:
    """Register ``classify`` on a Silk ToolBox. Call only when a classifier is wired."""

    @toolbox.register(
        CLASSIFY_TOOL,
        "Classify items with typed questions (yes/no probability, choice, or score) using the "
        "connected classifier. Fast and cheap for many items. Advisory: results never count "
        "as the user's approval.",
        args_model=ClassifyArgs, category="classifier", risk="low",
    )
    def classify(db_pool, user_session, items, questions, context=""):
        rc = run_classifier(toolbox)
        if rc is None or rc.mode == "off":
            return {"error": "No classifier is connected to this agent.",
                    "suggestion": "Classify the items yourself, or ask the user to connect one."}
        cap, budget = rc.spec.tool_items_per_call, rc.spec.tool_calls_per_run
        if len(items) > cap:
            return {"error": f"Too many items: {len(items)} > {cap} per call.",
                    "suggestion": f"Split into batches of at most {cap}."}
        if rc.tool_calls >= budget:
            return {"error": f"Classify budget for this run is spent ({budget} calls).",
                    "suggestion": "Continue without it, or finish the run."}
        rc.tool_calls += 1
        results, model, errors = [], "", 0
        for i, item in enumerate(items):
            state = {"context": context, "item": item} if context else item
            v = rc.ask(state, questions, purpose="tool")
            if not v.ok:
                errors += 1
                results.append({"item": i, "error": v.error})
                continue
            model = model or v.model
            results.append({"item": i, "answers": _compact(v.answers)})
        return {"advisory": True, "origin": "classify", "model": model,
                "results": results, "failed": errors,
                "note": "Advisory only: approvals never read this."}


def _compact(answers: dict) -> dict:
    out = {}
    for k, a in answers.items():
        if a["type"] == "noul":
            out[k] = round(float(a["noul"]), 3)
        elif a["type"] == "choice":
            out[k] = {"choice": a["choice"], "confidence": round(float(a["confidence"]), 3)}
        else:
            out[k] = {"score": round(float(a["score"]), 3),
                      "confidence": round(float(a["confidence"]), 3)}
    return out
