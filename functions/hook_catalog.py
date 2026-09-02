# -*- coding: utf-8 -*-
"""Named hook catalog — vetted hook implementations selectable by name.

Behavior stays code, selection becomes data: nodes offer these hooks as
checkboxes and presets store only the **names + config values**. A
preset can therefore never smuggle executable behavior — it can only
pick from this catalog and parameterize it through each entry's pydantic
``config_model`` (invalid configs fall back to defaults, logged).

Two consumption paths, matching the two hook layers:

``attach_catalog_hooks(toolbox, sandbox, names, configs)``
    Attach-function shaped, so the ToolBox node can put it into the
    build recipe → the hooks are *infrastructure*: replayed into every
    derived ToolSet, always on.

``build_hooks(names, configs)``
    Returns the ``{event: [callables]}`` dict a :class:`~.role.Role`
    accepts — the *behavioral* layer, installed by RoleBinding on
    activation and removed on deactivation.

Hook maps may include middleware events (``wrap_*``) — registration
sites route them through :func:`~.hooks.register_hook_map`, and the
ToolBox emits ``wrap_tool_execute`` around every execution, so catalog
hooks can deny calls (budget) or rewrite results (redaction).

Every factory builds fresh callables per call, so two toolboxes/roles
never share closure state (timing dicts, budgets, counters).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Mapping, Optional

from pydantic import BaseModel, Field, ValidationError

from weave.logger import get_logger

from .signoff import CHANGE_TYPES, preset_policy

from .hooks import (
    HookEntry,
    HOOK_AFTER_RUN,
    HOOK_AFTER_TOOL_EXECUTE,
    HOOK_BEFORE_RUN,
    HOOK_BEFORE_TOOL_EXECUTE,
    HOOK_TOOL_DENIED,
    HOOK_WRAP_TOOL_EXECUTE,
    register_hook_map,
)

if TYPE_CHECKING:
    from .tool_box import ToolBox
    from .tools.file_sandbox import FileToolSandbox

log = get_logger("SilkHooks")

#: {event_name: [callables]} — the shape Role.hooks and the registry consume.
HookMap = dict[str, list[Callable]]


@dataclass(frozen=True)
class HookSpec:
    """One catalog entry: a named, described, optionally configurable
    hook factory. ``factory(config)`` receives a validated instance of
    ``config_model`` (or ``None`` for config-less hooks)."""

    name: str
    description: str
    factory: Callable[[Optional[BaseModel]], HookMap]
    config_model: Optional[type[BaseModel]] = None


# ── starter hook implementations (config-less) ───────────────────────────


def _make_log_tool_calls(_config: Optional[BaseModel] = None) -> HookMap:
    """Log every tool dispatch, result and denial via the weave logger."""

    def before(tool_name: str = "", tool_args: Optional[dict] = None, **_kw: Any) -> None:
        keys = ", ".join(sorted((tool_args or {}).keys())) or "-"
        log.info(f"[hook:log] → {tool_name}({keys})")

    def after(tool_name: str = "", tool_result: str = "", **_kw: Any) -> None:
        preview = (tool_result or "")[:120].replace("\n", " ")
        log.info(f"[hook:log] ← {tool_name}: {preview}")

    def denied(tool_name: str = "", **_kw: Any) -> None:
        log.warning(f"[hook:log] ✗ {tool_name} denied by active role")

    return {
        HOOK_BEFORE_TOOL_EXECUTE: [before],
        HOOK_AFTER_TOOL_EXECUTE: [after],
        HOOK_TOOL_DENIED: [denied],
    }


def _make_timing(_config: Optional[BaseModel] = None) -> HookMap:
    """Log wall-clock duration of every tool execution."""
    started: dict[str, float] = {}

    def before(tool_name: str = "", **_kw: Any) -> None:
        started[tool_name] = time.perf_counter()

    def after(tool_name: str = "", **_kw: Any) -> None:
        t0 = started.pop(tool_name, None)
        if t0 is not None:
            log.info(f"[hook:timing] {tool_name} took {time.perf_counter() - t0:.3f}s")

    return {
        HOOK_BEFORE_TOOL_EXECUTE: [before],
        HOOK_AFTER_TOOL_EXECUTE: [after],
    }


def _make_usage_meter(_config: Optional[BaseModel] = None) -> HookMap:
    """Count tool calls / denials per run and log a summary at run end."""
    counts: dict[str, int] = {}
    denials: dict[str, int] = {}

    def before(tool_name: str = "", **_kw: Any) -> None:
        counts[tool_name] = counts.get(tool_name, 0) + 1

    def denied(tool_name: str = "", **_kw: Any) -> None:
        denials[tool_name] = denials.get(tool_name, 0) + 1

    def run_started(**_kw: Any) -> None:
        counts.clear()
        denials.clear()

    def run_finished(**_kw: Any) -> None:
        used = ", ".join(f"{n}×{c}" for n, c in sorted(counts.items())) or "none"
        summary = f"[hook:usage] tools used: {used}"
        if denials:
            summary += " · denied: " + ", ".join(
                f"{n}×{c}" for n, c in sorted(denials.items())
            )
        log.info(summary)

    return {
        HOOK_BEFORE_RUN: [run_started],
        HOOK_BEFORE_TOOL_EXECUTE: [before],
        HOOK_TOOL_DENIED: [denied],
        HOOK_AFTER_RUN: [run_finished],
    }


# ── configurable middleware hooks ────────────────────────────────────────


class RedactSecretsConfig(BaseModel):
    """Config for the redact_secrets hook (regex patterns over results)."""

    patterns: list[str] = Field(
        default_factory=lambda: [
            r"AKIA[0-9A-Z]{16}",                      # AWS access key ids
            r"(?i)bearer\s+[a-z0-9._\-]{16,}",        # bearer tokens
            r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*\S+",
        ],
        description="Regex patterns; every match in a tool result is replaced.",
    )
    replacement: str = Field(
        "[REDACTED]", description="Replacement text for matched secrets."
    )


def _make_redact_secrets(config: Optional[BaseModel]) -> HookMap:
    """Middleware: rewrite tool results, masking secret-looking matches."""
    cfg = config if isinstance(config, RedactSecretsConfig) else RedactSecretsConfig()
    compiled: list[re.Pattern] = []
    for pattern in cfg.patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            log.warning(f"[hook:redact] invalid pattern {pattern!r} skipped: {exc}")

    async def redact(handler: Callable = None, tool_name: str = "", **_kw: Any) -> Any:
        result = await handler()
        if not isinstance(result, str) or not compiled:
            return result
        redacted = result
        for pattern in compiled:
            redacted = pattern.sub(cfg.replacement, redacted)
        if redacted != result:
            log.info(f"[hook:redact] masked secret(s) in '{tool_name}' result")
        return redacted

    return {HOOK_WRAP_TOOL_EXECUTE: [redact]}


class ToolBudgetConfig(BaseModel):
    """Config for the tool_budget hook (per-run call ceilings)."""

    max_calls: int = Field(
        25, ge=1, description="Maximum tool calls per run (all tools combined)."
    )
    max_calls_per_tool: int = Field(
        0, ge=0, description="Per-tool ceiling; 0 = no per-tool limit."
    )


def _make_tool_budget(config: Optional[BaseModel]) -> HookMap:
    """Middleware: hard-deny tool calls beyond a per-run budget.

    Denials are structured ``budget_exceeded`` errors — non-retryable
    for reflection (the budget cannot recover within the run), and the
    call never reaches the executable.
    """
    cfg = config if isinstance(config, ToolBudgetConfig) else ToolBudgetConfig()
    total = {"count": 0}
    per_tool: dict[str, int] = {}

    def run_started(**_kw: Any) -> None:
        total["count"] = 0
        per_tool.clear()

    async def enforce(handler: Callable = None, tool_name: str = "", **_kw: Any) -> Any:
        tool_count = per_tool.get(tool_name, 0)
        if total["count"] >= cfg.max_calls or (
            cfg.max_calls_per_tool and tool_count >= cfg.max_calls_per_tool
        ):
            log.warning(f"[hook:budget] '{tool_name}' denied — budget exhausted")
            return json.dumps({
                "error": (
                    f"Tool budget exhausted "
                    f"({total['count']} calls used, limit {cfg.max_calls})."
                ),
                "error_type": "budget_exceeded",
                "suggestion": "Finish the task with the information already gathered.",
            }, ensure_ascii=False)
        total["count"] += 1
        per_tool[tool_name] = tool_count + 1
        return await handler()

    return {
        HOOK_BEFORE_RUN: [run_started],
        HOOK_WRAP_TOOL_EXECUTE: [enforce],
    }


class TaskAuditConfig(BaseModel):
    """Config for the task_audit hook (rationale quality on plan changes)."""

    strict: bool = Field(
        True, description="Reject trivial rationales on the consequential plan "
                          "tools (add / complete / rescope / revise-goal).",
    )
    min_words: int = Field(2, ge=1, description="Minimum words in a rationale.")
    min_chars: int = Field(8, ge=1, description="Minimum characters in a rationale.")
    trivial: list[str] = Field(
        default_factory=lambda: [
            "n/a", "na", "because", "idk", "ok", "okay", "done", "-", "none", "tbd",
        ],
        description="Rationales rejected outright (case-insensitive exact match).",
    )


#: Plan tools whose rationale the audit hook holds to a quality bar, and the
#: wider set it timestamps. The tool schemas already reject a *blank* rationale;
#: this hook additionally bounces a trivial one ("n/a", "ok").
_AUDIT_GUARDED = frozenset(
    {"task_add", "task_complete", "task_rescope", "goal_revise"}
)
_AUDIT_PLAN_TOOLS = _AUDIT_GUARDED | frozenset(
    {"plan_start", "task_update", "task_claim"}
)


def _make_task_audit(config: Optional[BaseModel]) -> HookMap:
    """Guard the plan's rationale quality + keep a timestamped audit trail.

    Strict mode is a ``wrap_tool_execute`` middleware: if a consequential plan
    tool arrives with a *trivial* rationale it is short-circuited with a
    corrective result (the store is never touched), nudging the model to state a
    real reason — the reflection-style bounce from the plan's §7. It never blocks
    a genuine rationale, and never touches non-plan tools.
    """
    cfg = config if isinstance(config, TaskAuditConfig) else TaskAuditConfig()
    trivial = {t.strip().lower() for t in cfg.trivial}

    def _too_weak(rationale: Any) -> bool:
        text = str(rationale or "").strip()
        if len(text) < cfg.min_chars or len(text.split()) < cfg.min_words:
            return True
        return text.lower() in trivial

    async def enforce(
        handler: Callable = None, tool_name: str = "",
        tool_args: Optional[dict] = None, **_kw: Any,
    ) -> Any:
        # No `tool_name in _AUDIT_GUARDED` test here: the binding below says
        # which tools this applies to, and the registry does the filtering
        # (D13). The set is the same set; the difference is that it is now
        # visible from outside the closure.
        if cfg.strict:
            rationale = (tool_args or {}).get("rationale")
            if _too_weak(rationale):
                log.warning(
                    f"[hook:task_audit] '{tool_name}' rationale too weak: "
                    f"{str(rationale)!r}"
                )
                return json.dumps({
                    "error": "Rationale is too vague — explain *why* in a full "
                             "sentence.",
                    "suggestion": "State what prompted this change and what it "
                                  "achieves, then call the tool again.",
                }, ensure_ascii=False)
        return await handler()

    def after(tool_name: str = "", **_kw: Any) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        log.info(f"[hook:task_audit] {stamp} plan op: {tool_name}")

    return {
        HOOK_WRAP_TOOL_EXECUTE: [
            HookEntry(callback=enforce, tools=_AUDIT_GUARDED)
        ],
        HOOK_AFTER_TOOL_EXECUTE: [
            HookEntry(callback=after, tools=_AUDIT_PLAN_TOOLS)
        ],
    }


_Level = Literal["agent", "human"]


class SignoffConfig(BaseModel):
    """Per-change-type approval policy for the task sign-off gate.

    ``preset`` picks a ready-made policy; set it to ``custom`` to use the per-type
    levels below. ``agent`` = the agent self-signs (applies now); ``human`` = the
    change needs the user's approval, and until the inline decision seam exists
    (D30) it is refused rather than held (D31–D33 deleted the parked path).
    """

    preset: Literal["custom", "auto", "completions", "final", "strict"] = Field(
        "auto", description="Ready-made policy; 'custom' uses the per-type levels.",
    )
    add: _Level = Field("agent", description="Adding a task (task_add).")
    complete: _Level = Field("agent", description="Completing a task (task_complete).")
    complete_final: _Level = Field(
        "agent", description="The completion that closes the plan (last open task).",
    )
    rescope: _Level = Field(
        "agent", description="Dropping / re-scoping a task (task_rescope).",
    )
    goal: _Level = Field(
        "agent", description="Revising the goal / acceptance (goal_revise).",
    )


def signoff_policy_from_config(cfg: SignoffConfig) -> dict:
    """Resolve a :class:`SignoffConfig` to a ``{change_type: level}`` policy."""
    if cfg.preset != "custom":
        return preset_policy(cfg.preset)
    return {t: getattr(cfg, t) for t in CHANGE_TYPES}


def _make_signoff(_config: Optional[BaseModel] = None) -> HookMap:
    """Inert lifecycle-hook map: the sign-off *gate* is a store-aware middleware
    wired by :func:`attach_catalog_hooks` (which has the toolbox), not by the
    config-less factory path. This entry exists so the hook is selectable and
    configurable in the standard hook UI."""
    return {}


# ── the catalog ──────────────────────────────────────────────────────────

HOOK_CATALOG: dict[str, HookSpec] = {
    spec.name: spec
    for spec in (
        HookSpec(
            name="log_tool_calls",
            description="Log every tool call, result and role denial.",
            factory=_make_log_tool_calls,
        ),
        HookSpec(
            name="timing",
            description="Log the wall-clock duration of each tool execution.",
            factory=_make_timing,
        ),
        HookSpec(
            name="usage_meter",
            description="Count tool calls / denials per run; summary at run end.",
            factory=_make_usage_meter,
        ),
        HookSpec(
            name="redact_secrets",
            description=(
                "Mask secret-looking matches (API keys, tokens) in tool "
                "results before the model sees them."
            ),
            factory=_make_redact_secrets,
            config_model=RedactSecretsConfig,
        ),
        HookSpec(
            name="tool_budget",
            description=(
                "Hard-deny tool calls beyond a per-run budget "
                "(total and optional per-tool ceiling)."
            ),
            factory=_make_tool_budget,
            config_model=ToolBudgetConfig,
        ),
        HookSpec(
            name="task_audit",
            description=(
                "Hold the task plan's rationale to a quality bar (bounce trivial "
                "'n/a'-style reasons on add/complete/rescope/revise-goal) and log "
                "a timestamped trail of plan changes."
            ),
            factory=_make_task_audit,
            config_model=TaskAuditConfig,
        ),
        HookSpec(
            name="signoff",
            description=(
                "Require user sign-off before task changes take effect, per change "
                "type (agent self-signs vs human approval refuses the change). "
                "Needs Task Planning; configured here, enforced on the ToolBox."
            ),
            factory=_make_signoff,
            config_model=SignoffConfig,
        ),
    )
}


def catalog_names() -> list[str]:
    """Sorted names for node UIs and preset validation."""
    return sorted(HOOK_CATALOG)


def resolve_config(
    spec: HookSpec, raw: Optional[Mapping[str, Any]]
) -> Optional[BaseModel]:
    """Validate a raw config dict for *spec*; invalid → defaults, logged."""
    if spec.config_model is None:
        return None
    try:
        return spec.config_model.model_validate(dict(raw or {}))
    except ValidationError as exc:
        log.warning(
            f"Invalid config for hook '{spec.name}' — using defaults: {exc}"
        )
        return spec.config_model()


def build_hooks(
    names: Iterable[str],
    configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> HookMap:
    """Merge the named catalog entries into one ``{event: [callables]}``.

    *configs* maps hook name → raw config dict (validated through the
    entry's ``config_model``). Unknown names (e.g. from a preset written
    by a newer version) are skipped with a warning rather than failing
    the whole configuration.
    """
    configs = configs or {}
    merged: HookMap = {}
    for name in names:
        spec = HOOK_CATALOG.get(str(name))
        if spec is None:
            log.warning(f"Unknown hook '{name}' — not in the catalog, skipped.")
            continue
        hook_map = spec.factory(resolve_config(spec, configs.get(str(name))))
        for event, callbacks in hook_map.items():
            merged.setdefault(event, []).extend(callbacks)
    return merged


def attach_catalog_hooks(
    toolbox: "ToolBox",
    sandbox: "Optional[FileToolSandbox]" = None,
    names: Iterable[str] = (),
    configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    """Recipe-compatible attacher: register catalog hooks on *toolbox*.

    Signature matches the ToolBox build recipe (``attacher(toolbox,
    sandbox)``); *sandbox* is unused but keeps the replay uniform. Being
    part of the recipe makes these hooks **infrastructure**: every
    ToolSet derived from the toolbox re-creates them automatically.

    The attached names are also recorded on the toolbox
    (``catalog_hook_names``) so downstream UIs (e.g. the Role node's hook
    list) can show which hooks are already active at the infrastructure
    layer — ticking the same hook again on a role is legal but doubles
    it, and that should be visible, not surprising.

    ``signoff`` is special: it is selectable/configurable like any catalog
    hook, but its gate is a **store-aware** ``wrap_tool_execute`` middleware,
    so it is wired here (where the toolbox — and its task store — is in
    scope) rather than through the config-less ``build_hooks`` factory.
    """
    names = tuple(str(n) for n in names)
    register_hook_map(toolbox.hooks, build_hooks(names, configs))

    if "signoff" in names:
        from .signoff import attach_signoff_gate  # local: avoid import cycle
        cfg = resolve_config(HOOK_CATALOG["signoff"], (configs or {}).get("signoff"))
        attach_signoff_gate(
            toolbox, sandbox, policy=signoff_policy_from_config(cfg),  # type: ignore[arg-type]
        )

    existing = tuple(getattr(toolbox, "catalog_hook_names", ()))
    toolbox.catalog_hook_names = tuple(  # type: ignore[attr-defined]
        dict.fromkeys((*existing, *(str(n) for n in names)))
    )
