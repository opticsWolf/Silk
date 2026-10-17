# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The classifier resource behind the ``silk_classifier`` port (S4, C1-C7).

One object, three consumers:

* the approval gate (triage, completion check) - purpose ``"gate"``;
* catalog hooks (task drift, prose questions, findings gate) - ``"hook"``;
* the LLM-callable ``classify`` tool - purpose ``"tool"``.

The wire format everywhere is Jev's ``/v1/systemone`` shape, so a question set
written once runs on every backend::

    questions = {"name": {"type": "noul" | "choice" | "score",
                          "instructions": str,
                          "criteria": {option: description|None} | [level, ...]}}
    answers   = {"name": {"type": "noul", "noul": p}
                         | {"type": "choice", "choice": k, "probabilities": {...}, "confidence": c}
                         | {"type": "score", "score": s, "legend": {...}, "probabilities": {...},
                            "confidence": c}}

Backends form a chain (hosted Jev, then a local judge model, then nothing).
A backend that fails is skipped and, after repeated failures, rested by a
circuit breaker. When every backend fails the verdict carries the error and
no answers: the caller's fallback is the human, never an LLM's opinion.

The acting agent's own model is never a backend. A local judge is a separate
model used only to score fixed options by log-probability.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Protocol, Sequence

from pydantic import BaseModel, Field, field_validator

MODES = ("off", "shadow", "live")
PURPOSES = ("gate", "hook", "tool")
JEV_PRICE_PER_MTOK = 0.042          # USD per million input tokens; output is free
MAX_CHOICE_OPTIONS = 255
SCORE_LEVELS = (2, 10)
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class ClassifierError(RuntimeError):
    pass


class QuestionError(ValueError):
    pass


# ── questions ────────────────────────────────────────────────────────────────

def validate_questions(questions: dict) -> dict:
    """Check a question set against Jev's rules; return it unchanged."""
    if not isinstance(questions, dict) or not questions:
        raise QuestionError("questions must be a non-empty mapping")
    for name, q in questions.items():
        if not isinstance(name, str) or not _NAME.match(name):
            raise QuestionError(f"bad question name {name!r}")
        if not isinstance(q, dict) or not str(q.get("instructions", "")).strip():
            raise QuestionError(f"{name}: instructions are required")
        kind = q.get("type")
        crit = q.get("criteria")
        if kind == "noul":
            if crit is not None:
                raise QuestionError(f"{name}: noul takes no criteria")
        elif kind == "choice":
            if not isinstance(crit, dict) or not (2 <= len(crit) <= MAX_CHOICE_OPTIONS):
                raise QuestionError(f"{name}: choice needs 2..{MAX_CHOICE_OPTIONS} options")
        elif kind == "score":
            lo, hi = SCORE_LEVELS
            if not isinstance(crit, list) or not (lo <= len(crit) <= hi):
                raise QuestionError(f"{name}: score needs {lo}..{hi} ordered levels")
        else:
            raise QuestionError(f"{name}: unknown type {kind!r}")
    return questions


def check_answers(questions: dict, answers: dict) -> dict:
    """Refuse an answer set that does not match the questions (D36 spirit)."""
    for name, q in questions.items():
        a = answers.get(name)
        if not isinstance(a, dict) or a.get("type") != q["type"]:
            raise ClassifierError(f"{name}: missing or mistyped answer")
        if q["type"] == "noul":
            p = float(a["noul"])
            if not 0.0 <= p <= 1.0:
                raise ClassifierError(f"{name}: probability out of range")
        elif q["type"] == "choice":
            if a.get("choice") not in q["criteria"]:
                raise ClassifierError(f"{name}: choice not among the options")
        else:
            s = float(a["score"])
            if not 0.0 <= s <= len(q["criteria"]) - 1:
                raise ClassifierError(f"{name}: score off the scale")
    return answers


# ── verdicts ─────────────────────────────────────────────────────────────────

@dataclass
class Verdict:
    answers: dict
    backend: str = ""               # which link of the chain answered
    model: str = ""                 # versioned model, the key for calibrated bands
    purpose: str = ""
    cached: bool = False
    input_tokens: int = 0
    cost_usd: float = 0.0
    error: str = ""                 # set when no backend answered
    tried: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.error

    def noul(self, name: str) -> float:
        return float(self.answers[name]["noul"])

    def record(self) -> dict:
        return {"backend": self.backend, "model": self.model, "purpose": self.purpose,
                "cached": self.cached, "input_tokens": self.input_tokens,
                "cost_usd": round(self.cost_usd, 8), "error": self.error, "tried": self.tried,
                "answers": self.answers}


class Classifier(Protocol):
    """What the port carries. The gate, hooks and tool see only this."""
    mode: str

    def ask(self, state: Any, questions: dict, *, purpose: str) -> Verdict: ...


# ── backends ─────────────────────────────────────────────────────────────────

class Backend(Protocol):
    name: str
    model: str
    egress: bool                     # True when state leaves the machine

    def answer(self, state: Any, questions: dict) -> tuple[dict, int]: ...


def _estimate_tokens(state: Any, questions: dict) -> int:
    return max(1, len(json.dumps([state, questions], ensure_ascii=False, default=str)) // 4)


class JevBackend:
    """Hosted Jev. ``http(url, headers, body) -> dict`` is injectable for tests."""

    egress = True

    def __init__(self, *, key: Optional[str], model: str = "jev/jev-1.13.0",
                 url: str = "https://api.typesafe.ai/v1/systemone", timeout: float = 4.0,
                 http: Optional[Callable[[str, dict, dict, float], dict]] = None) -> None:
        if model.endswith("jev-latest"):
            raise ValueError("pin a Jev version: bands are calibrated per model")
        self.name, self.model, self.url, self.timeout = "jev", model, url, timeout
        self._key = key
        self._http = http or _urllib_post

    def answer(self, state: Any, questions: dict) -> tuple[dict, int]:
        if not self._key:
            raise ClassifierError("no Jev key configured")
        body = {"model": self.model, "state": state, "questions": questions}
        resp = self._http(self.url, {"Authorization": f"Bearer {self._key}",
                                     "Content-Type": "application/json"}, body, self.timeout)
        used = int((resp.get("usage") or {}).get("input_tokens") or _estimate_tokens(state, questions))
        if resp.get("model"):
            self.model = "jev/" + str(resp["model"]).removeprefix("jev/")
        return resp["answers"], used


def _urllib_post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers,
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310
        return json.loads(r.read())


Scorer = Callable[[str, Sequence[str]], Sequence[float]]
"""``scorer(prompt, options) -> log-probabilities``, one per option.

In Silk this is a GGUF model in the pool, scoring each option's tokens after
the prompt (no sampling). It must not be the model the agent is running on.
"""


def _softmax(logps: Sequence[float]) -> list[float]:
    m = max(logps)
    ex = [math.exp(x - m) for x in logps]
    z = sum(ex)
    return [e / z for e in ex]


def _certainty(ps: Sequence[float]) -> float:
    """1 - normalised entropy: 1 when one option has it all, 0 when uniform."""
    if len(ps) < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in ps if p > 0)
    return max(0.0, 1.0 - h / math.log(len(ps)))


class LocalJudgeBackend:
    """A local model as a judge: fixed options, scored by log-probability.

    Same questions, same answer shapes as Jev, so the gate does not know which
    one answered - only ``model`` differs, and with it the calibrated bands.
    One prompt per question (Jev answers all in one pass).
    """

    egress = False

    def __init__(self, scorer: Scorer, *, model: str = "local/judge") -> None:
        self.name, self.model, self._score = "local", model, scorer

    @staticmethod
    def prompt(state: Any, name: str, q: dict) -> str:
        head = ("You are a strict classifier. Read the state and answer the question "
                "with exactly one of the allowed answers.\n")
        body = json.dumps(state, ensure_ascii=False, default=str)
        if q["type"] == "noul":
            allowed = "yes, no"
        elif q["type"] == "choice":
            allowed = "; ".join(f"{k}: {v}" if v else k for k, v in q["criteria"].items())
        else:
            allowed = ", ".join(q["criteria"])
        return f"{head}State: {body}\nQuestion: {q['instructions']}\nAllowed: {allowed}\nAnswer:"

    def answer(self, state: Any, questions: dict) -> tuple[dict, int]:
        out, used = {}, 0
        for name, q in questions.items():
            p = self.prompt(state, name, q)
            used += len(p) // 4
            if q["type"] == "noul":
                yes, _no = _softmax(self._score(p, [" yes", " no"]))
                out[name] = {"type": "noul", "noul": yes}
            elif q["type"] == "choice":
                keys = list(q["criteria"])
                ps = _softmax(self._score(p, [f" {k}" for k in keys]))
                best = max(range(len(keys)), key=lambda i: ps[i])
                out[name] = {"type": "choice", "choice": keys[best],
                             "probabilities": dict(zip(keys, ps)), "confidence": _certainty(ps)}
            else:
                levels = list(q["criteria"])
                ps = _softmax(self._score(p, [f" {lv}" for lv in levels]))
                out[name] = {"type": "score", "score": sum(i * pi for i, pi in enumerate(ps)),
                             "legend": {str(i): lv for i, lv in enumerate(levels)},
                             "probabilities": {str(i): pi for i, pi in enumerate(ps)},
                             "confidence": _certainty(ps)}
        return out, used


class FakeBackend:
    """Scripted backend for tests and demos: ``fn(state, questions) -> answers``."""

    def __init__(self, fn: Callable[[Any, dict], dict], *, model: str = "fake/1",
                 egress: bool = False, name: str = "fake") -> None:
        self.name, self.model, self.egress, self._fn = name, model, egress, fn
        self.calls: list = []

    def answer(self, state: Any, questions: dict) -> tuple[dict, int]:
        self.calls.append((state, questions))
        return self._fn(state, questions), _estimate_tokens(state, questions)


# ── redaction, breaker, cache ───────────────────────────────────────────────

DEFAULT_REDACT = (
    r"(?i)(api[_-]?key|token|secret|password|passwd|authorization)\s*[:=]\s*\S+",
    r"sk-[A-Za-z0-9]{16,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
)


class Redactor:
    def __init__(self, patterns: Sequence[str] = DEFAULT_REDACT) -> None:
        self._res = [re.compile(p) for p in patterns]

    def text(self, s: str) -> str:
        for r in self._res:
            s = r.sub("[REDACTED]", s)
        return s

    def __call__(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {k: self(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self(v) for v in value]
        return value


class Breaker:
    """Skip a backend for ``cooldown_s`` after ``threshold`` failures in a row."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.threshold, self.cooldown_s, self.clock = threshold, cooldown_s, clock
        self._fails: dict[str, int] = {}
        self._until: dict[str, float] = {}

    def open(self, name: str) -> bool:
        return self._until.get(name, 0.0) > self.clock()

    def failed(self, name: str) -> None:
        n = self._fails.get(name, 0) + 1
        self._fails[name] = n
        if n >= self.threshold:
            self._until[name] = self.clock() + self.cooldown_s
            self._fails[name] = 0

    def ok(self, name: str) -> None:
        self._fails[name] = 0
        self._until.pop(name, None)


class _Cache:
    """LRU with TTL. Keys include the purpose: a ``classify`` tool call can
    never warm, or be read as, an answer the gate relies on (C5)."""

    def __init__(self, size: int = 512, ttl_s: float = 600.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.size, self.ttl_s, self.clock = size, ttl_s, clock
        self._d: OrderedDict[str, tuple[float, Verdict]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(purpose: str, state: Any, questions: dict) -> str:
        blob = json.dumps([purpose, state, questions], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, k: str) -> Optional[Verdict]:
        with self._lock:
            hit = self._d.get(k)
            if hit is None or hit[0] < self.clock():
                self._d.pop(k, None)
                return None
            self._d.move_to_end(k)
            return hit[1]

    def put(self, k: str, v: Verdict) -> None:
        with self._lock:
            self._d[k] = (self.clock() + self.ttl_s, v)
            self._d.move_to_end(k)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


# ── the chain ────────────────────────────────────────────────────────────────

class Meter:
    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.cost_usd = 0.0
        self.failures = 0
        self.by_purpose: dict[str, int] = {}

    def add(self, v: Verdict) -> None:
        self.calls += 1
        self.by_purpose[v.purpose] = self.by_purpose.get(v.purpose, 0) + 1
        if v.ok and not v.cached:
            self.input_tokens += v.input_tokens
            self.cost_usd += v.cost_usd
        if not v.ok:
            self.failures += 1


class ChainClassifier:
    """Backends in order; the first that answers well wins."""

    def __init__(self, backends: Sequence[Backend], *, mode: str = "shadow",
                 egress_allowed: bool = False, redactor: Optional[Redactor] = None,
                 breaker: Optional[Breaker] = None, cache: Optional[_Cache] = None,
                 meter: Optional[Meter] = None) -> None:
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.backends = list(backends)
        self.egress_allowed = egress_allowed
        self.redact = redactor or Redactor()
        self.breaker = breaker or Breaker()
        self.cache = cache or _Cache()
        self.meter = meter or Meter()

    def ask(self, state: Any, questions: dict, *, purpose: str) -> Verdict:
        if purpose not in PURPOSES:
            raise ValueError(purpose)
        validate_questions(questions)
        if self.mode == "off":
            v = Verdict({}, purpose=purpose, error="classifier is off")
            self.meter.add(v)
            return v
        key = self.cache.key(purpose, state, questions)
        hit = self.cache.get(key)
        if hit is not None:
            v = Verdict(hit.answers, hit.backend, hit.model, purpose, True, hit.input_tokens,
                        0.0, "", list(hit.tried))
            self.meter.add(v)
            return v
        tried: list = []
        for b in self.backends:
            if b.egress and not self.egress_allowed:
                tried.append(f"{b.name}: egress not allowed for this project")
                continue
            if self.breaker.open(b.name):
                tried.append(f"{b.name}: resting after repeated failures")
                continue
            sent = self.redact(state) if b.egress else state
            try:
                answers, used = b.answer(sent, questions)
                check_answers(questions, answers)
            except Exception as exc:      # every failure moves down the chain
                self.breaker.failed(b.name)
                tried.append(f"{b.name}: {type(exc).__name__}: {exc}")
                continue
            self.breaker.ok(b.name)
            cost = used * JEV_PRICE_PER_MTOK / 1e6 if b.name == "jev" else 0.0
            v = Verdict(answers, b.name, b.model, purpose, False, used, cost, "", tried)
            self.cache.put(key, v)
            self.meter.add(v)
            return v
        v = Verdict({}, purpose=purpose, error="no classifier backend answered", tried=tried)
        self.meter.add(v)
        return v


# ── node settings (the Classifier node's form, stored in presets) ────────────

class BandsConfig(BaseModel):
    reversible: float = Field(0.90, ge=0, le=1)
    in_scope: float = Field(0.85, ge=0, le=1)
    consistent: float = Field(0.80, ge=0, le=1)
    max_risk: float = Field(1.0, ge=0, le=4)
    min_confidence: float = Field(0.70, ge=0, le=1)
    allow_intents: list[str] = Field(default_factory=lambda: ["routine", "exploratory"])


class BackendConfig(BaseModel):
    kind: Literal["jev", "local", "fake"]
    model: str = ""
    url: str = "https://api.typesafe.ai/v1/systemone"
    timeout_s: float = Field(4.0, gt=0, le=30)

    @field_validator("model")
    @classmethod
    def _pinned(cls, v: str) -> str:
        if v.endswith("latest"):
            raise ValueError("pin a model version: bands are calibrated per model")
        return v


class ClassifierSpec(BaseModel):
    """Everything the Classifier node stores. Secrets are not in here: the
    Jev key comes from Silk's credential store by name."""

    mode: Literal["off", "shadow", "live"] = "shadow"
    backends: list[BackendConfig] = Field(default_factory=lambda: [
        BackendConfig(kind="jev", model="jev/jev-1.13.0")])
    egress_allowed: bool = False          # project opt-in for hosted backends
    redact: list[str] = Field(default_factory=lambda: list(DEFAULT_REDACT))
    bands: dict[str, BandsConfig] = Field(default_factory=dict)   # per model id
    default_bands: Optional[BandsConfig] = None   # None: an uncalibrated model never skips
    tool_calls_per_run: int = Field(10, ge=0)
    tool_items_per_call: int = Field(50, ge=1, le=500)
    cache_ttl_s: float = Field(600.0, ge=0)
    breaker_threshold: int = Field(3, ge=1)
    breaker_cooldown_s: float = Field(60.0, ge=0)
    key_name: str = "jev"

    def bands_for(self, model: str) -> Optional[BandsConfig]:
        return self.bands.get(model, self.default_bands)


def build_classifier(spec: ClassifierSpec, *, secrets: Callable[[str], Optional[str]] = lambda n: None,
                     local_scorer: Optional[Scorer] = None,
                     fake: Optional[Callable[[Any, dict], dict]] = None,
                     http: Optional[Callable] = None,
                     clock: Callable[[], float] = time.monotonic) -> ChainClassifier:
    backends: list = []
    for b in spec.backends:
        if b.kind == "jev":
            backends.append(JevBackend(key=secrets(spec.key_name), model=b.model or "jev/jev-1.13.0",
                                       url=b.url, timeout=b.timeout_s, http=http))
        elif b.kind == "local":
            if local_scorer is None:
                continue                  # no judge model loaded: link skipped, not an error
            backends.append(LocalJudgeBackend(local_scorer, model=b.model or "local/judge"))
        elif fake is not None:
            backends.append(FakeBackend(fake, model=b.model or "fake/1"))
    return ChainClassifier(
        backends, mode=spec.mode, egress_allowed=spec.egress_allowed,
        redactor=Redactor(spec.redact),
        breaker=Breaker(spec.breaker_threshold, spec.breaker_cooldown_s, clock),
        cache=_Cache(ttl_s=spec.cache_ttl_s, clock=clock))


@dataclass
class ClassifierHandle:
    """The value on the ``silk_classifier`` port.

    Captured once per run (D38): editing the node mid-run affects the next run.
    ``for_run`` gives the per-run view: the tool budget lives there.
    """
    spec: ClassifierSpec
    classifier: ChainClassifier
    judge_model: Any = None          # the local judge's pool, for the self-grading check

    def for_run(self, run_id: str) -> "RunClassifier":
        return RunClassifier(self, run_id)


class RunClassifier:
    def __init__(self, handle: ClassifierHandle, run_id: str) -> None:
        self.handle, self.run_id = handle, run_id
        self.spec = handle.spec.model_copy(deep=True)       # policy snapshot
        self.mode = self.spec.mode
        self.tool_calls = 0

    def ask(self, state: Any, questions: dict, *, purpose: str) -> Verdict:
        return self.handle.classifier.ask(state, questions, purpose=purpose)

    def bands_for(self, model: str) -> Optional[BandsConfig]:
        return self.spec.bands_for(model)

    @property
    def meter(self) -> Meter:
        return self.handle.classifier.meter
