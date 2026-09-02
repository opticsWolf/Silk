# -*- coding: utf-8 -*-
"""The Role model: declarative agent configurations over a ToolBox.

A **Role** is serialisable data describing *how* an agent runs: which subset
of the toolbox it may use, its persona instructions, model-setting overrides,
behavioural hooks, and which capabilities to activate. Activating a role
against a ToolBox produces a **RoleBinding** — the live object that installs
the enforcement predicate, registers hooks/capabilities, and knows how to
cleanly deactivate.

Design contract (see ROLE_AGENTLOOP_PLAN.md in the silk-playground repo):

* The role's toolset is a **hard boundary** enforced at dispatch time inside
  ``ToolBox.execute_tool_calls_async`` — not just prompt-side filtering.
* Hooks are layered: ToolBox-level hooks are invariant infrastructure and
  fire for every execution; role hooks are behavioural, registered on
  activation and removed on deactivation. Because ``HookRegistry`` runs
  ``before_*`` FIFO and ``after_*`` LIFO, the earlier-registered toolbox
  hooks always run *outside* the role's hooks.
* Roles are the production activator of capabilities: no fourth extension
  axis — a role composes the existing primitives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .capabilities import BaseCapability
    from .tool_box import ToolBox


# Ordered risk scale for ToolSelector.max_risk. Tools declare their risk at
# registration (ToolBox.register(risk=...)); unknown/absent risk counts as
# "low" so untagged tools stay selectable by permissive selectors.
RISK_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class ToolSelector:
    """Declarative tool-subset rule, evaluated live at dispatch time.

    Semantics: ``deny_names`` always wins; otherwise a tool is permitted if
    ``allow_all`` is set, its name is in ``allow_names``, or any of its
    registered tags/category match — subject to the ``max_risk`` ceiling.
    An all-empty selector permits **nothing** (deny-by-default).
    """

    allow_names: frozenset[str] = frozenset()
    allow_tags: frozenset[str] = frozenset()
    allow_categories: frozenset[str] = frozenset()
    max_risk: Optional[str] = None
    deny_names: frozenset[str] = frozenset()
    allow_all: bool = False

    def permits(self, name: str, meta: Optional[dict[str, Any]]) -> bool:
        """Whether tool *name* (with ToolBox meta dict *meta*) is permitted."""
        if name in self.deny_names:
            return False

        risk = (meta or {}).get("risk", "low")
        if self.max_risk is not None:
            if RISK_ORDER.get(risk, 0) > RISK_ORDER.get(self.max_risk, 0):
                return False

        if self.allow_all or name in self.allow_names:
            return True

        tags = set((meta or {}).get("tags") or ())
        if tags & self.allow_tags:
            return True

        category = (meta or {}).get("category")
        return bool(category and category in self.allow_categories)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_names": sorted(self.allow_names),
            "allow_tags": sorted(self.allow_tags),
            "allow_categories": sorted(self.allow_categories),
            "max_risk": self.max_risk,
            "deny_names": sorted(self.deny_names),
            "allow_all": self.allow_all,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolSelector":
        return cls(
            allow_names=frozenset(d.get("allow_names") or ()),
            allow_tags=frozenset(d.get("allow_tags") or ()),
            allow_categories=frozenset(d.get("allow_categories") or ()),
            max_risk=d.get("max_risk"),
            deny_names=frozenset(d.get("deny_names") or ()),
            allow_all=bool(d.get("allow_all", False)),
        )


#: Selector for the implicit default role: everything allowed, nothing denied.
ALLOW_ALL = ToolSelector(allow_all=True)


@dataclass
class Role:
    """A named, declarative agent configuration.

    Pure data — holds no live tool references. ``hooks`` maps hook event
    names (see ``hooks.py`` HOOK_* constants) to callables; these are the
    *behavioural* layer registered on activation. Capabilities are given as
    instances (graph nodes construct them) and are registered on the
    toolbox when the role activates.
    """

    id: str
    name: str = ""
    description: str = ""
    instructions: str = ""
    selector: ToolSelector = field(default_factory=lambda: ALLOW_ALL)
    model_settings: dict[str, Any] = field(default_factory=dict)
    capabilities: list["BaseCapability"] = field(default_factory=list)
    hooks: dict[str, list[Callable]] = field(default_factory=dict)
    max_rounds: Optional[int] = None
    #: File access this role passes on, as data (spec D16/D17). The role
    #: never holds a live sandbox -- it holds the *grant*, so the port is
    #: visible in the graph and each layer can only narrow what it received.
    #: ``None`` means "this role says nothing about files", which leaves the
    #: toolset's own sandbox exactly as it was.
    file_grants: Any = None

    def system_prompt_block(self) -> str:
        """The ``[ROLE: …]`` block contributed to the system prompt."""
        if not self.instructions:
            return ""
        return f"[ROLE: {self.name or self.id}]\n{self.instructions.strip()}"

    # -- serialisation (hooks/capabilities are code, not data) -------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "selector": self.selector.to_dict(),
            "model_settings": dict(self.model_settings),
            "max_rounds": self.max_rounds,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Role":
        return cls(
            id=d["id"],
            name=d.get("name", d["id"]),
            description=d.get("description", ""),
            instructions=d.get("instructions", ""),
            selector=ToolSelector.from_dict(d.get("selector") or {"allow_all": True}),
            model_settings=dict(d.get("model_settings") or {}),
            max_rounds=d.get("max_rounds"),
        )


DEFAULT_ROLE = Role(
    id="default",
    name="Default",
    description="Unrestricted role — full toolbox, no persona.",
    selector=ALLOW_ALL,
)


class RoleBinding:
    """A Role activated against a ToolBox. Owns everything reversible.

    Use as a context manager or call :meth:`deactivate` explicitly.  Only
    one binding may be active on a toolbox at a time; activating a second
    one raises rather than silently stacking filters.
    """

    def __init__(self, role: Role, toolbox: "ToolBox") -> None:
        self.role = role
        self.toolbox = toolbox
        self._active = False
        self._registered_hooks: list[tuple[str, Callable]] = []
        self._added_capability_ids: list[str] = []
        self._added_tool_names: set[str] = set()

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def activate(cls, role: Role, toolbox: "ToolBox") -> "RoleBinding":
        binding = cls(role, toolbox)
        binding._activate()
        return binding

    def _activate(self) -> None:
        if getattr(self.toolbox, "_role_binding", None) is not None:
            raise RuntimeError(
                f"ToolBox already has an active role "
                f"'{self.toolbox._role_binding.role.id}'; deactivate it first."
            )

        # 1. Register the role's capabilities, honouring declared ordering.
        for capability in _order_capabilities(self.role.capabilities):
            before = set(self.toolbox.tools)
            self.toolbox.register_capability(capability)
            self._added_capability_ids.append(capability.id)
            self._added_tool_names |= set(self.toolbox.tools) - before

        # 2. Register behavioural hooks (wrap_* events go to the
        #    middleware layer). Toolbox infrastructure hooks were
        #    registered earlier (at build), so FIFO/LIFO emission keeps
        #    them outside the role layer automatically.
        from .hooks import register_hook_map
        self._registered_hooks = register_hook_map(
            self.toolbox.hooks, self.role.hooks
        )

        # 3. Install the hard enforcement predicate (prompt + dispatch side).
        self.toolbox.set_role_filter(self.role.selector.permits)

        self.toolbox._role_binding = self  # type: ignore[attr-defined]
        self._active = True

    def deactivate(self) -> None:
        """Reverse activation completely; the toolbox ends up pristine."""
        if not self._active:
            return

        self.toolbox.set_role_filter(None)

        from .hooks import unregister_hook_map
        unregister_hook_map(self.toolbox.hooks, self._registered_hooks)
        self._registered_hooks.clear()

        for cap_id in self._added_capability_ids:
            self.toolbox._capabilities.pop(cap_id, None)
            self.toolbox._loaded_capability_ids.discard(cap_id)
            # Capability-carried hooks were registered on load; remove
            # exactly those (tracked per capability id by the ToolBox).
            unregister_hook_map(
                self.toolbox.hooks,
                self.toolbox._capability_hooks.pop(cap_id, []),
            )
        for tool_name in self._added_tool_names:
            self.toolbox.tools.pop(tool_name, None)
        self._added_capability_ids.clear()
        self._added_tool_names.clear()
        # Refresh the load_capability tool description (deferred list shrank).
        self.toolbox.register_load_capability_tool()

        self.toolbox._role_binding = None  # type: ignore[attr-defined]
        self._active = False

    def __enter__(self) -> "RoleBinding":
        if not self._active:
            self._activate()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.deactivate()

    # -- run integration -----------------------------------------------------

    def system_prompt_block(self) -> str:
        return self.role.system_prompt_block()

    def effective_gen_params(self, base: dict[str, Any]) -> dict[str, Any]:
        """Overlay precedence: explicit *base* (GUI/node) > role > defaults.

        Keys present in *base* win; the role only fills gaps it defines.
        """
        merged = dict(self.role.model_settings)
        merged.update(base)
        return merged


def _order_capabilities(capabilities: list["BaseCapability"]) -> list["BaseCapability"]:
    """Sort capabilities by their declared ordering; validate requirements.

    ``position == 'outermost'`` sorts first, ``'innermost'`` last, everything
    else keeps the caller's order (stable sort). A missing ``requires``
    dependency raises ValueError — better to fail activation loudly than run
    a capability without its prerequisite.
    """
    present = {c.id for c in capabilities}
    for cap in capabilities:
        ordering = cap.get_ordering()
        if ordering is None:
            continue
        missing = [req for req in ordering.requires if req not in present]
        if missing:
            raise ValueError(
                f"Capability '{cap.id}' requires missing capabilities: {missing}"
            )

    def sort_key(cap: "BaseCapability") -> int:
        ordering = cap.get_ordering()
        if ordering is None or ordering.position is None:
            return 1
        if ordering.position == "outermost":
            return 0
        if ordering.position == "innermost":
            return 2
        return 1

    return sorted(capabilities, key=sort_key)
