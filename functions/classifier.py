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
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, Sequence

from pydantic import BaseModel, Field, field_validator

from .systemone import SystemOneClient

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


class SystemOneBackend:
    """One ``/v1/systemone`` endpoint, hosted or local.

    Jev and Laya are interchangeable here: same questions, same answers,
    same usage shape. Only the URL, the credential, the model id (the
    bands key) and the egress flag differ.

    ``http(url, headers, body) -> dict`` is injectable for tests, and
    ``client`` (anything with ``decide(state, questions, timeout)``)
    adapts a pool-backed decision model to the same shape.
    """

    def __init__(self, *, key: Optional[str] = None,
                 model: str = "systemone/1",
                 url: str = "https://api.typesafe.ai/v1/systemone",
                 timeout: float = 4.0, egress: bool = True,
                 name: str = "systemone",
                 http: Optional[Callable[[str, dict, dict, float], dict]] = None,
                 client: Any = None) -> None:
        if model.endswith("latest"):
            raise ValueError("pin a model version: bands are calibrated per model")
        self.name, self.model, self.url, self.timeout = name, model, url, timeout
        self.egress = egress
        self._key = key
        if client is not None:
            self._http = _pool_http(client)
            self._needs_key = False
        else:
            self._http = http or _urllib_post
            self._needs_key = http is None

    def answer(self, state: Any, questions: dict) -> tuple[dict, int]:
        if self._needs_key and not self._key:
            raise ClassifierError(f"no key configured for {self.name}")
        body = {"model": self.model, "state": state, "questions": questions}
        resp = self._http(self.url, {"Authorization": f"Bearer {self._key}",
                                     "Content-Type": "application/json"}, body, self.timeout)
        used = int((resp.get("usage") or {}).get("input_tokens")
                   or _estimate_tokens(state, questions))
        if self.name == "jev" and resp.get("model"):
            # Hosted Jev echoes its canonical model id; the bands key
            # follows it. Pool adapters never send "model".
            self.model = "jev/" + str(resp["model"]).removeprefix("jev/")
        return resp["answers"], used


class JevBackend(SystemOneBackend):
    """Hosted Jev: one ``/v1/systemone`` endpoint among others.

    Kept with its original signature, wire format and egress so existing
    specs, presets and tests keep working; new code reaches for
    ``SystemOneBackend`` directly.
    """

    egress = True

    def __init__(self, *, key: Optional[str], model: str = "jev/jev-1.13.0",
                 url: str = "https://api.typesafe.ai/v1/systemone",
                 timeout: float = 4.0,
                 http: Optional[Callable[[str, dict, dict, float], dict]] = None) -> None:
        super().__init__(key=key, model=model, url=url, timeout=timeout,
                         egress=True, name="jev", http=http)


def _pool_http(client: Any) -> Callable[[str, dict, dict, float], dict]:
    """Adapt a decision pool (``decide(state, questions, timeout)``) to the
    ``http(url, headers, body)`` shape backends use."""
    def post(url: str, headers: dict, body: dict, timeout: float) -> dict:
        return {"answers": client.decide(body["state"], body["questions"],
                                          timeout=timeout),
                "usage": {}}
    return post


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
                             "probabilities": dict(zip(keys, ps, strict=True)), "confidence": _certainty(ps)}
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
            cost = used * JEV_PRICE_PER_MTOK / 1e6 if b.egress else 0.0
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
    # ``jev`` and ``systemone`` are the same backend class (one
    # /v1/systemone endpoint each): ``jev`` is the hosted product with
    # its egress, ``systemone`` a decision-model endpoint that is local
    # by default and hosted only when ``egress`` says so. ``key_name`` is
    # the per-backend credential, falling back to the spec default.
    kind: Literal["jev", "systemone", "local", "fake"]
    model: str = ""
    url: str = "https://api.typesafe.ai/v1/systemone"
    timeout_s: float = Field(4.0, gt=0, le=30)
    egress: Optional[bool] = None  # None: jev True, everything else False
    key_name: Optional[str] = None  # None: use the spec default

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
    tool_calls_per_run: int = Field(100, ge=0)
    tool_items_per_call: int = Field(50, ge=1, le=500)
    cache_ttl_s: float = Field(600.0, ge=0)
    breaker_threshold: int = Field(3, ge=1)
    breaker_cooldown_s: float = Field(60.0, ge=0)
    key_name: str = "JEV_API_KEY"

    def bands_for(self, model: str) -> Optional[BandsConfig]:
        return self.bands.get(model, self.default_bands)


def decision_member(handle: Any, *,
                    timeout: float = 4.0) -> Optional[tuple[str, "BackendConfig", dict]]:
    """A wired classifier-model handle as (bands id, spec entry, build kwarg).

    Pool-backed (Classifier Loader): the id is ``local/<file stem>``, the
    entry is plain, and the kwarg carries a client for it -- answering
    stays local, keyless, without egress. Hosted (Jev Loader): the id,
    URL and credential name come from the handle and egress is on; a
    key that does not resolve fails at answer time, recorded in the
    verdict's ``tried`` list, never at build time.

    ``None`` when the handle is not a classifier model (unwired, or a
    chat handle on the wrong port -- the port type refuses those on the
    canvas; this only meets ``None`` and hand-built dicts in tests).
    """
    if not isinstance(handle, dict) or not handle \
            or handle.get("backend") != "systemone":
        return None
    pool = handle.get("pool")
    if pool is not None:
        path = getattr(pool, "model_path", "") or ""
        model_id = f"local/{Path(path).stem}" if path else "local/decision"
        try:
            client = SystemOneClient(pool.base_url, pool.model_alias)
        except AttributeError:
            return None
        return model_id, BackendConfig(kind="systemone", model=model_id,
                                       egress=False), {"systemone": client}
    url = str(handle.get("url") or "").strip().rstrip("/")
    if not url:
        return None
    model_id = str(handle.get("model") or "").strip() or "systemone/1"
    credential = str(handle.get("credential") or "").strip()
    entry = BackendConfig(kind="systemone", model=model_id, url=url,
                          timeout=timeout,
                          key_name=credential or None, egress=True)
    return model_id, entry, {}


def build_classifier(spec: ClassifierSpec, *, secrets: Callable[[str], Optional[str]] = lambda n: None,
                     local_scorer: Optional[Scorer] = None,
                     systemone: Any = None,
                     fake: Optional[Callable[[Any, dict], dict]] = None,
                     http: Optional[Callable] = None,
                     clock: Callable[[], float] = time.monotonic) -> ChainClassifier:
    backends: list = []
    for b in spec.backends:
        secret = secrets(b.key_name or spec.key_name)
        if b.kind == "jev":
            backends.append(JevBackend(key=secret, model=b.model or "jev/jev-1.13.0",
                                       url=b.url, timeout=b.timeout_s, http=http))
        elif b.kind == "systemone":
            if systemone is not None:
                # A wired decision pool answers locally: no key, no egress.
                backends.append(SystemOneBackend(
                    client=systemone, model=b.model or "local/decision",
                    url=b.url, timeout=b.timeout_s, egress=False))
            else:
                # Hosted endpoint (wired Jev loader or hand-written spec).
                # A missing key is an answer-time failure, recorded in the
                # verdict's tried list -- never a build error.
                backends.append(SystemOneBackend(
                    key=secret, model=b.model or "systemone/1",
                    url=b.url, timeout=b.timeout_s,
                    egress=bool(b.egress), http=http))
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


# ── ensembles: several classifiers, one verdict ────────────────────────

def combine_answers(questions: dict,
                    verdicts: list[Verdict]) -> tuple[dict, str]:
    """Several ok verdicts on the same questions as one answer set.

    Equal weights: every member counts once. Per type: ``choice`` is a
    probability-weighted vote (the winner's share, times the members'
    mean certainty, is the combined confidence); ``score`` and ``noul``
    are means whose confidence is discounted by the spread, so
    disagreeing members tank below ``min_confidence`` and the human is
    asked. The combined model id joins the member ids, sorted -- it
    needs its own bands row, exactly like any other model.
    """
    oks = [v for v in verdicts if v.ok and v.answers]
    if not oks:
        raise ClassifierError("nothing to combine: no member answered")
    if len(oks) == 1:
        return dict(oks[0].answers), oks[0].model
    models = sorted({v.model for v in oks if v.model}) or ["ensemble"]
    combined: dict[str, Any] = {}
    for qid, member_answers in _answers_by_question(oks).items():
        question = questions.get(qid) if isinstance(questions, dict) else None
        qtype = (question.get("type") if isinstance(question, dict) else None)
        qtype = qtype or member_answers[0][1].get("type")
        if qtype == "choice":
            combined[qid] = _combine_choice(member_answers)
        elif qtype == "score":
            scale = _score_scale(question)
            combined[qid] = _combine_scored(member_answers, scale, "score")
        else:
            combined[qid] = _combine_scored(member_answers, 1.0, "noul")
    return combined, "+".join(models)


def _answers_by_question(
        verdicts: list[Verdict]) -> dict[str, list[tuple[Verdict, dict]]]:
    """Member answers grouped by question, in member order."""
    grouped: dict[str, list[tuple[Verdict, dict]]] = {}
    for v in verdicts:
        for qid, answer in v.answers.items():
            if isinstance(answer, dict):
                grouped.setdefault(qid, []).append((v, answer))
    return grouped


def _mean_confidence(member_answers: list[tuple[Verdict, dict]]) -> float:
    """Mean stated certainty. Missing is 0: a member that states no
    confidence earns no early stop and drags the combination."""
    confs = [float(a.get("confidence", 0.0) or 0.0)
             for _, a in member_answers]
    return sum(confs) / len(confs) if confs else 0.0


def _combine_choice(member_answers: list[tuple[Verdict, dict]]) -> dict:
    options: dict[str, float] = {}
    for _, a in member_answers:
        probs = a.get("probabilities")
        if isinstance(probs, dict):
            for option, prob in probs.items():
                try:
                    options[str(option)] = options.get(str(option), 0.0) + float(prob)
                except (TypeError, ValueError):
                    continue
    n = len(member_answers)
    averaged = {option: total / n for option, total in options.items()}
    winner = max(sorted(averaged), key=averaged.__getitem__) if averaged else ""
    if not winner:
        fallback = str(member_answers[0][1].get("choice") or "")
        return {"type": "choice", "choice": fallback,
                "probabilities": {}, "confidence": 0.0}
    return {"type": "choice", "choice": winner,
            "probabilities": averaged,
            "confidence": averaged[winner] * _mean_confidence(member_answers)}


def _score_scale(question: Any) -> float:
    criteria = question.get("criteria") if isinstance(question, dict) else None
    if isinstance(criteria, (list, tuple)) and len(criteria) > 1:
        return float(len(criteria) - 1)
    return 0.0


def _combine_scored(member_answers: list[tuple[Verdict, dict]],
                    scale: float, qtype: str) -> dict:
    key = "score" if qtype == "score" else "noul"
    values = []
    for _, a in member_answers:
        try:
            values.append(float(a.get(key, 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)
    mean = sum(values) / len(values)
    spread = (max(values) - min(values)) / scale if scale > 0 else 0.0
    spread = min(max(spread, 0.0), 1.0)
    value = min(max(mean, 0.0), scale) if qtype == "score" and scale > 0 \
        else min(max(mean, 0.0), 1.0)
    return {"type": qtype, key: value,
            "confidence": _mean_confidence(member_answers) * (1.0 - spread)}


class EnsembleClassifier:
    """Several member chains asked as one, by policy.

    Members are ``(label, chain)`` pairs in input order; ``a``/``b``/``c``
    on the canvas. A member whose mode is ``off`` is skipped. Policies:
    ``first`` returns the first ok verdict untouched; ``consensus`` asks
    everybody (in threads) and combines; ``cascade`` asks in order and
    stops early when every answer's confidence clears ``agreement``.
    Early stops and single answers keep the member's verdict whole --
    member model id, member bands -- and only a real combination mints
    the joined id, which needs its own bands row.
    """

    name = "ensemble"

    def __init__(self, members: Sequence[tuple[str, Any]], *, mode: str = "shadow",
                 policy: str = "cascade", agreement: float = 0.7,
                 fallback: Any = None) -> None:
        if policy not in ("first", "consensus", "cascade"):
            raise ClassifierError(f"unknown ensemble policy {policy!r}")
        self.members = list(members)
        self.mode = mode
        self.policy = policy
        self.agreement = agreement
        #: A judge chain asked only when no decision member answered --
        #: the fallback, never a voter (its entropy-derived confidence
        #: must not dilute two agreeing decision models).
        self.fallback = fallback
        self.meter = Meter()

    @property
    def backends(self) -> list:
        """Member links flattened, for stats that name what answered."""
        found = []
        for _, chain in self.members:
            for backend in getattr(chain, "backends", []):
                found.append(backend)
        return found

    def _active(self) -> list[tuple[str, Any]]:
        return [(label, chain) for label, chain in self.members
                if getattr(chain, "mode", "shadow") != "off"]

    def ask(self, state: Any, questions: dict, *, purpose: str) -> Verdict:
        if purpose not in PURPOSES:
            raise ValueError(purpose)
        validate_questions(questions)
        if self.mode == "off":
            v = Verdict({}, purpose=purpose, error="ensemble is off")
            self.meter.add(v)
            return v
        if self.policy == "consensus":
            return self._ask_consensus(state, questions, purpose=purpose)
        return self._ask_sequential(state, questions, purpose=purpose)

    # ── policies ──

    def _ask_sequential(self, state: Any, questions: dict, *, purpose: str) -> Verdict:
        tried: list[str] = []
        oks: list[Verdict] = []
        for label, chain in self._active():
            try:
                v = chain.ask(state, questions, purpose=purpose)
            except Exception as exc:      # a broken member, not a broken ensemble
                tried.append(f"member {label}: {type(exc).__name__}: {exc}")
                continue
            tried.extend(f"member {label}: {entry}" for entry in v.tried)
            if not v.ok:
                continue
            oks.append(v)
            if self.policy == "first":
                return self._record(_with_tried(v, tried), purpose)
            if self._clears(v, self.agreement):
                return self._record(_with_tried(v, tried), purpose)
        return self._finish(oks, tried, purpose, questions, state)

    def _ask_consensus(self, state: Any, questions: dict, *, purpose: str) -> Verdict:
        active = self._active()
        results: list[Any] = [None] * len(active)

        def _ask_one(index: int, chain: Any) -> None:
            try:
                results[index] = chain.ask(state, questions, purpose=purpose)
            except Exception as exc:  # noqa: BLE001 - one member's failure
                results[index] = exc

        threads = [threading.Thread(target=_ask_one, args=(i, chain), daemon=True,
                                    name=f"silk-ensemble-{i}")
                   for i, (_, chain) in enumerate(active)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        tried: list[str] = []
        oks: list[Verdict] = []
        for (label, _), result in zip(active, results, strict=True):
            if isinstance(result, Exception):
                tried.append(f"member {label}: {type(result).__name__}: {result}")
            elif result is not None:
                tried.extend(f"member {label}: {entry}" for entry in result.tried)
                if result.ok:
                    oks.append(result)
        return self._finish(oks, tried, purpose, questions, state)

    def _finish(self, oks: list[Verdict], tried: list[str],
              purpose: str, questions: dict, state: Any) -> Verdict:
        if not oks:
            if self.fallback is not None:
                try:
                    v = self.fallback.ask(state, questions, purpose=purpose)
                except Exception as exc:  # noqa: BLE001 - a broken fallback
                    tried.append(f"fallback: {type(exc).__name__}: {exc}")
                    v = None
                if v is not None and v.ok:
                    return self._record(_with_tried(v, tried), purpose)
                if v is not None:
                    tried.extend(f"fallback: {entry}" for entry in v.tried)
            v = Verdict({}, purpose=purpose,
                        error="no ensemble member answered", tried=tried)
            return self._record(v, purpose)
        if len(oks) == 1:
            return self._record(_with_tried(oks[0], tried), purpose)
        answers, model = combine_answers(questions, oks)
        v = Verdict(answers, self.name, model, purpose, False,
                    sum(v.input_tokens for v in oks),
                    round(sum(v.cost_usd for v in oks), 8), "", tried)
        return self._record(v, purpose)

    @staticmethod
    def _clears(verdict: Verdict, agreement: float) -> bool:
        """Every answer states confidence at or above the threshold."""
        if not verdict.answers:
            return False
        for answer in verdict.answers.values():
            try:
                if float(answer.get("confidence", 0.0) or 0.0) < agreement:
                    return False
            except (TypeError, ValueError, AttributeError):
                return False
        return True

    def _record(self, v: Verdict, purpose: str) -> Verdict:
        v.purpose = purpose
        self.meter.add(v)
        return v


def _with_tried(v: Verdict, tried: list[str]) -> Verdict:
    """The verdict with the ensemble's full narrative attached."""
    return Verdict(v.answers, v.backend, v.model, v.purpose, v.cached,
                   v.input_tokens, v.cost_usd, v.error, list(tried))


class EnsembleSpec(BaseModel):
    """What an Ensemble node stores. Member chains arrive by wire."""
    policy: Literal["first", "consensus", "cascade"] = "cascade"
    agreement: float = Field(0.7, ge=0, le=1)
    mode: Literal["off", "shadow", "live"] = "shadow"
    bands: dict[str, BandsConfig] = Field(default_factory=dict)
    tool_calls_per_run: int = Field(100, ge=0)
    tool_items_per_call: int = Field(50, ge=1, le=500)

    def bands_for(self, model: str) -> Optional[BandsConfig]:
        return self.bands.get(model)


def build_ensemble(spec: EnsembleSpec,
                   members: Sequence[tuple[str, Any]],
                   fallback: Any = None) -> EnsembleClassifier:
    """An ensemble from a spec, labelled member chains, and an optional
    judge chain asked only when no decision member answered."""
    return EnsembleClassifier(
        list(members), mode=spec.mode,
        policy=spec.policy, agreement=spec.agreement, fallback=fallback)


@dataclass
class EnsembleHandle(ClassifierHandle):
    """The Ensemble node's ``classifier`` value: a ClassifierHandle whose
    chain is an :class:`EnsembleClassifier`.

    A subclass on purpose: the gate, hooks and tool only ever check
    ``isinstance(handle, ClassifierHandle)`` and then use ``for_run`` +
    ``spec``. ``members`` names the wired slots; ``judge_model`` (from
    the base) carries the fallback judge for the self-grading refusal.
    """
    members: tuple = ()
