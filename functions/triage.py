# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Approval triage and the completion check, on any :class:`Classifier` (S4).

Rules enforced here, whatever the classifier says:

* Only a call whose effective level is ``triage`` can skip the human, only in
  ``live`` mode, and only inside the bands calibrated **for the model that
  answered** (``spec.bands[model]``). An uncalibrated model never skips.
* ``shadow`` mode computes the would-be route and records it; it always asks.
* The classifier never denies. A low score raises priority and may suggest
  "deny" to the human.
* No classifier, mode ``off``, or no backend answered: ask the human.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .classifier import BandsConfig, Verdict
from .signoff import SIGNOFF_LEVELS as LEVELS      # agent < triage < human

INTENTS = {
    "routine": "an ordinary step of the task: edit, build, test, read",
    "exploratory": "inspects or measures without changing anything that matters",
    "destructive": "deletes, overwrites or resets work or data",
    "exfiltration": "sends project data or secrets outside the project",
    "escalation": "changes permissions, grants, policies, credentials or its own tools",
}
RISK_LEVELS = ["none", "low", "moderate", "high", "severe"]

APPROVAL_QUESTIONS: dict[str, dict] = {
    "reversible": {"type": "noul", "instructions":
                   "Could the user fully undo the effect of this action without losing anything "
                   "(for example through version control, a trash folder, or a re-run)?"},
    "in_scope": {"type": "noul", "instructions":
                 "Is this action clearly part of the current task and goal given in the state?"},
    "consistent": {"type": "noul", "instructions":
                   "Is this action consistent with the user's prior decisions and rules given in "
                   "the state, and not something the user has previously refused?"},
    "risk": {"type": "score", "instructions":
             "How much harm could this action cause if it turns out to be wrong?",
             "criteria": RISK_LEVELS},
    "intent": {"type": "choice", "instructions": "What kind of action is this?",
               "criteria": INTENTS},
}


def stricter(*levels: str) -> str:
    """agent < triage < human; the stricter answer wins (as today, one rung more)."""
    return max(levels, key=LEVELS.index)


def effective_level(level: str, classifier: Any) -> str:
    """``triage`` with nothing to triage with is ``human`` (C3)."""
    if level == "triage" and (classifier is None or getattr(classifier, "mode", "off") == "off"):
        return "human"
    return level


@dataclass
class Triage:
    route: str                       # "allow" | "ask"
    priority: float                  # 0..1, inbox ordering
    why: str
    suggestion: Optional[str] = None # "approve" | "deny" | None (shown to the human)
    shadow_route: Optional[str] = None
    verdict: dict = field(default_factory=dict)

    def record(self) -> dict:
        return {"route": self.route, "shadow_route": self.shadow_route,
                "priority": round(self.priority, 3), "why": self.why,
                "suggestion": self.suggestion, **self.verdict}


def build_state(*, goal: str, task: str, acceptance: list[str], tool: str, args: dict,
                prior: list[dict], rules: list[str], max_arg_chars: int = 2000) -> dict:
    """The triage ``state``: built from the ledger, OKFgraph and the call's own
    arguments only. Never from the transcript, and never from a ``classify``
    tool result (C5)."""
    raw = json.dumps(args, ensure_ascii=False, default=str)
    return {
        "goal": goal, "task": task, "acceptance": acceptance,
        "action": {"tool": tool, "arguments": raw[:max_arg_chars]},
        "prior_decisions": [{"prompt": p.get("prompt", "")[:200], "outcome": p.get("outcome"),
                             "reason": p.get("reason", "")} for p in prior[:12]],
        "rules": rules[:12],
    }


def route_approval(answers: dict, *, level: str, bands: Optional[BandsConfig]) -> Triage:
    """Pure mapping from answers to a route; unit-tested without any backend."""
    n = lambda k: float(answers[k]["noul"])          # noqa: E731
    rev, scope, cons = n("reversible"), n("in_scope"), n("consistent")
    risk, intent = answers["risk"], answers["intent"]
    r_score, r_conf = float(risk["score"]), float(risk.get("confidence", 0.0))
    i_choice, i_conf = str(intent["choice"]), float(intent.get("confidence", 0.0))

    priority = min(1.0, 0.35 * (r_score / 4) + 0.25 * (1 - rev) + 0.2 * (1 - cons)
                   + 0.2 * (1 - scope))
    suggestion = None
    if cons < 0.2:
        suggestion = "deny"
    elif rev > 0.9 and scope > 0.9 and r_score < 1.0:
        suggestion = "approve"

    if level != "triage":
        return Triage("ask", priority, f"policy level '{level}' always asks", suggestion)
    if bands is None:
        return Triage("ask", priority, "no calibrated bands for this model", suggestion)
    checks = [
        (rev >= bands.reversible, f"reversible {rev:.2f} < {bands.reversible}"),
        (scope >= bands.in_scope, f"in_scope {scope:.2f} < {bands.in_scope}"),
        (cons >= bands.consistent, f"consistent {cons:.2f} < {bands.consistent}"),
        (r_score <= bands.max_risk and r_conf >= bands.min_confidence,
         f"risk {r_score:.2f} (conf {r_conf:.2f})"),
        (i_choice in bands.allow_intents and i_conf >= bands.min_confidence,
         f"intent {i_choice} (conf {i_conf:.2f})"),
    ]
    failed = [why for ok, why in checks if not ok]
    if failed:
        return Triage("ask", priority, "; ".join(failed), suggestion)
    return Triage("allow", priority, "all bands met", None)


def triage_approval(classifier: Any, state: dict, *, level: str) -> Optional[Triage]:
    """Ask the classifier and route. ``None`` when there is nothing to ask with.

    Runs for ``triage`` *and* ``human`` levels (ranking and suggestion), in
    ``shadow`` and ``live`` mode. Never raises.
    """
    if classifier is None or classifier.mode == "off":
        return None
    try:
        v: Verdict = classifier.ask(state, APPROVAL_QUESTIONS, purpose="gate")
    except Exception as exc:                                     # pragma: no cover
        return Triage("ask", 0.5, "triage failed", verdict={"error": str(exc)})
    if not v.ok:
        return Triage("ask", 0.5, "triage unavailable", verdict=v.record())
    t = route_approval(v.answers, level=level, bands=classifier.bands_for(v.model))
    t.verdict = v.record()
    if classifier.mode == "shadow":
        t.shadow_route, t.route = t.route, "ask"
        if t.shadow_route == "allow":
            t.why = "shadow: would have allowed"
    return t


# ── completion check ─────────────────────────────────────────────────────────

def completion_questions(acceptance: list[str]) -> dict:
    return {f"c{i}": {"type": "noul", "instructions":
                      f"Does the evidence in the state show that this criterion is met: {c}"}
            for i, c in enumerate(acceptance)}


def check_completion(classifier: Any, *, acceptance: list[str], evidence: str,
                     met: float = 0.8, unmet: float = 0.4) -> dict:
    """``{"verdict": accept|reject|ask, "unmet": [...], "scores": {...}}``.

    ``reject`` names the criteria back to the agent; ``ask`` shows the scores
    to the human. In ``shadow`` mode a would-be reject becomes ``ask``.
    """
    if classifier is None or classifier.mode == "off" or not acceptance:
        return {"verdict": "ask", "unmet": [], "scores": {}, "error": "no classifier"}
    v = classifier.ask({"evidence": evidence[:8000]}, completion_questions(acceptance),
                       purpose="gate")
    if not v.ok:
        return {"verdict": "ask", "unmet": [], "scores": {}, "error": v.error}
    ps = {k: v.noul(k) for k in v.answers}
    low = [acceptance[int(k[1:])] for k, p in sorted(ps.items()) if p < unmet]
    if low:
        verdict = "reject"
    elif all(p >= met for p in ps.values()):
        verdict = "accept"
    else:
        verdict = "ask"
    if classifier.mode == "shadow" and verdict != "ask":
        return {"verdict": "ask", "shadow_verdict": verdict, "unmet": low, "scores": ps}
    return {"verdict": verdict, "unmet": low, "scores": ps}
