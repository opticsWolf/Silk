"""Middleware lifecycle hooks for observability and error recovery.

Extends the basic hook system with middleware semantics:
- wrap_* hooks that can short-circuit or modify behavior
- Error counterpart hooks (on_*_error)
- Hook ordering via wraps/wrapped_by declarations
- Event stream wrapping for streaming capabilities

Inspired by Pydantic AI's capability middleware hooks.
"""
from __future__ import annotations

import asyncio

from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from collections.abc import Awaitable
    from .protocols import AgentEngine as ChatEngine
    from .run_context import RunContext
    from typing import Protocol

    # Import BaseCapability for type checking only (to avoid circular import)
    from .capabilities import BaseCapability as _BaseCapability  # noqa: F401

    # Wrap handler type aliases (matching Pydantic AI's naming)
    class WrapToolExecuteHandler(Protocol):
        """Handler type for wrap_tool_execute middleware."""
        def __call__(self, tool_name: str, tool_args: dict, **kwargs) -> Awaitable[str]: ...



# â”€â”€ Hook Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Existing events (backward compatible)
HOOK_BEFORE_MODEL_REQUEST = "before_model_request"
HOOK_AFTER_MODEL_RESPONSE = "after_model_response"
HOOK_BEFORE_TOOL_EXECUTE = "before_tool_execute"
HOOK_AFTER_TOOL_EXECUTE = "after_tool_execute"
HOOK_BEFORE_RUN = "before_run"
HOOK_AFTER_RUN = "after_run"

HOOK_AFTER_COMPACTION = "after_compaction"
"""Fired when history was actually compacted (§12, D24/D25).

The one deliberate invalidation Silk allows (I11), and therefore the one
thing a memory of a run must be able to record: in the ledger a compaction
is an assertion *about* earlier turns, not the destruction of them."""

HOOK_TOOL_DENIED = "tool_denied"
"""Fired when the active role's filter blocks a tool call at dispatch.
Audit/logging surface; the denied call never reaches the executable."""

# Alias for after_model_request (used in Pydantic AI)
HOOK_AFTER_MODEL_REQUEST = "after_model_request"

# New middleware events
HOOK_ON_MODEL_REQUEST_ERROR = "on_model_request_error"
"""Called when a model request fails with an exception."""

HOOK_WRAP_TOOL_VALIDATE = "wrap_tool_validate"
"""Wraps tool argument validation."""

HOOK_ON_TOOL_VALIDATE_ERROR = "on_tool_validate_error"
"""Called when tool argument validation fails."""

HOOK_WRAP_TOOL_EXECUTE = "wrap_tool_execute"
"""Wraps tool execution."""

HOOK_ON_TOOL_EXECUTE_ERROR = "on_tool_execute_error"
"""Called when tool execution fails with an exception."""

HOOK_ON_OUTPUT_VALIDATE_ERROR = "on_output_validate_error"
"""Called when output validation fails."""

HOOK_ON_OUTPUT_PROCESS_ERROR = "on_output_process_error"
"""Called when output processing fails."""


#: Every event name this module declares. Registration checks membership,
#: so a typo ("after_toool_execute") is refused instead of registering
#: cleanly against a name nothing will ever emit.
KNOWN_EVENTS = frozenset({
    HOOK_BEFORE_MODEL_REQUEST, HOOK_AFTER_MODEL_REQUEST,
    HOOK_AFTER_MODEL_RESPONSE, HOOK_ON_MODEL_REQUEST_ERROR,
    HOOK_AFTER_COMPACTION,
    HOOK_BEFORE_TOOL_EXECUTE, HOOK_AFTER_TOOL_EXECUTE, HOOK_TOOL_DENIED,
    HOOK_ON_TOOL_VALIDATE_ERROR, HOOK_ON_TOOL_EXECUTE_ERROR,
    HOOK_ON_OUTPUT_VALIDATE_ERROR, HOOK_ON_OUTPUT_PROCESS_ERROR,
    HOOK_BEFORE_RUN, HOOK_AFTER_RUN,
    HOOK_WRAP_TOOL_VALIDATE, HOOK_WRAP_TOOL_EXECUTE,
})

#: The events something actually emits. A hook registered on anything else
#: registers cleanly and then silently never fires -- which is worse than an
#: error, because the hook's *absence* is what you have to notice (G3/D15).
#: So registration on a declared-but-unwired event raises.
#:
#: This set is now every event this module declares: §22 q2 closed the five
#: open ``WRAP_*`` names by wiring one and deleting four. The guard stays --
#: it is what catches a typo -- but it currently has nothing but misspellings
#: to catch, which is the state it was built to reach.
WIRED_EVENTS = frozenset({
    HOOK_BEFORE_MODEL_REQUEST, HOOK_AFTER_MODEL_REQUEST,
    HOOK_AFTER_MODEL_RESPONSE, HOOK_ON_MODEL_REQUEST_ERROR,
    HOOK_AFTER_COMPACTION,
    HOOK_BEFORE_TOOL_EXECUTE, HOOK_AFTER_TOOL_EXECUTE, HOOK_TOOL_DENIED,
    HOOK_ON_TOOL_VALIDATE_ERROR, HOOK_ON_TOOL_EXECUTE_ERROR,
    HOOK_ON_OUTPUT_VALIDATE_ERROR, HOOK_ON_OUTPUT_PROCESS_ERROR,
    HOOK_BEFORE_RUN, HOOK_AFTER_RUN,
    HOOK_WRAP_TOOL_VALIDATE, HOOK_WRAP_TOOL_EXECUTE,
})

#: Names that are unwired *and* known, precomputed for the message.
UNWIRED_EVENTS = frozenset(KNOWN_EVENTS - WIRED_EVENTS)


class UnwiredHookEvent(ValueError):
    """Raised when a hook registers on an event nothing emits.

    A ``ValueError`` so a caller that already guards registration keeps
    working, and its own class so a caller that wants to tell "this event
    does not exist yet" from "these arguments are wrong" can.
    """


def _check_event(event: str, *, middleware: bool) -> None:
    """Refuse a registration that could never fire.

    Unknown names are refused too: the whole failure mode here is a hook
    that looks installed and is not, and a misspelled event name produces
    exactly that.
    """
    if event in WIRED_EVENTS:
        return
    kind = "middleware" if middleware else "callback"
    if event in UNWIRED_EVENTS:
        raise UnwiredHookEvent(
            f"Cannot register a {kind} on '{event}': the event is declared "
            "but nothing emits it yet, so the hook would never fire. Wire "
            "the event first (see DESIGN_SPEC_DRAFT D15), or register on "
            "one of: " + ", ".join(sorted(WIRED_EVENTS))
        )
    raise UnwiredHookEvent(
        f"Cannot register a {kind} on '{event}': no such hook event. "
        "Known events are: " + ", ".join(sorted(KNOWN_EVENTS))
    )

class EssentialHookError(ValueError):
    """Raised when something tries to remove a hook declared essential.

    A ``ValueError`` for the same reason :class:`UnwiredHookEvent` is: a
    caller that already guards hook wiring keeps working. Its own class so
    "you may not drop this" reads differently from "that event does not
    exist" (spec D14, invariant I7).
    """


@dataclass(frozen=True)
class HookEntry:
    """One registration: the callback plus what it declares about itself.

    Two declarations, both from spec section 8:

    *Binding* (D13) -- ``tools`` and ``categories`` say which tool calls the
    hook applies to. Empty means every tool, which is what every hook meant
    before this existed. The point is that a hook scoped to ``write_file``
    now says so in the registry, where config and a UI can see it, instead
    of opening with ``if tool_name != "write_file": return`` in a body
    nothing outside the function can read.

    *Tier* (D14) -- ``essential`` marks a hook that must survive derivation:
    it rides the recipe into every ToolSet built from a ToolBox, and
    :meth:`HookRegistry.unregister` refuses to take it off. The approval
    gate is the first hook that genuinely must not be droppable, but the
    "infrastructure hooks: part of the recipe" comment in ``nodes/toolbox``
    has been asserting this informally for a while.
    """

    callback: Callable
    #: Tool names this hook applies to; empty = every tool.
    tools: frozenset[str] = frozenset()
    #: Tool categories this hook applies to; empty = every category.
    categories: frozenset[str] = frozenset()
    #: Survives derivation and cannot be unregistered (D14, I7).
    essential: bool = False

    @property
    def bound(self) -> bool:
        """Whether this entry declares a binding at all."""
        return bool(self.tools or self.categories)

    def applies_to(self, tool_name: Any, category: Any) -> bool:
        """Whether the hook should fire for this tool.

        An unbound hook fires for everything, including events that carry no
        tool at all (``before_run``). A *bound* hook fires only for a tool it
        names, which means it stays silent on the tool-less events --
        deliberately: a hook that declared "I am about ``write_file``" has
        nothing to say when no tool is involved.
        """
        if not self.bound:
            return True
        if tool_name and str(tool_name) in self.tools:
            return True
        return bool(category and str(category) in self.categories)


def bind_tools(*tools: str, categories: Iterable[str] = ()) -> Callable:
    """Declare which tools a hook applies to (D13), as a decorator.

    :meth:`HookRegistry.register` reads the declaration off the function, so
    a hook can carry its own binding through a :data:`HookMap` -- which is a
    plain ``{event: [callables]}`` dict with nowhere else to put it.
    """

    def decorate(func: Callable) -> Callable:
        func._hook_tools = frozenset(tools)  # type: ignore[attr-defined]
        func._hook_categories = frozenset(categories)  # type: ignore[attr-defined]
        return func

    return decorate


def essential(func: Callable) -> Callable:
    """Declare a hook essential (D14), as a decorator."""
    func._hook_essential = True  # type: ignore[attr-defined]
    return func


# â”€â”€ Hook Registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class HookRegistry:
    """Registry for lifecycle hooks with middleware support.

    Supports both simple emit-based hooks (before/after) and middleware
    hooks (wrap_*) that can short-circuit or modify behavior.

    Attributes:
        _hooks: Dict mapping event names to lists of callbacks.
        _middleware: Dict mapping middleware event names to lists of handlers.
        _ordering: Dict mapping capability IDs to ordering constraints.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookEntry]] = {}
        self._middleware: dict[str, list[HookEntry]] = {}
        self._ordering: dict[str, dict[str, Any]] = {}
        # How to find a tool's category, for category-bound hooks (D13).
        # The registry does the filtering, so it needs the answer -- but it
        # has no tool registry of its own, so the ToolBox lends it one.
        # Unbound until then: an unknown category matches nothing, which is
        # the safe direction (a category-bound hook stays quiet rather than
        # firing for every tool).
        self._category_of: Callable[[str], Any] | None = None

    def bind_categories(self, resolver: Callable[[str], Any] | None) -> None:
        """Teach the registry how to map a tool name to its category."""
        self._category_of = resolver

    def _category(self, tool_name: Any) -> Any:
        if not tool_name or self._category_of is None:
            return None
        try:
            return self._category_of(str(tool_name))
        except Exception:
            return None

    @staticmethod
    def _entry(
        callback: Callable,
        tools: Iterable[str] | None,
        categories: Iterable[str] | None,
        essential_hook: bool | None,
    ) -> HookEntry:
        """Build an entry, falling back to what the callback declares.

        An explicit argument always wins; the decorators exist for the hook
        maps, which are plain dicts with nowhere to put a keyword.
        """
        if isinstance(callback, HookEntry):
            return callback
        return HookEntry(
            callback=callback,
            tools=frozenset(
                tools if tools is not None
                else getattr(callback, "_hook_tools", ())
            ),
            categories=frozenset(
                categories if categories is not None
                else getattr(callback, "_hook_categories", ())
            ),
            essential=bool(
                essential_hook if essential_hook is not None
                else getattr(callback, "_hook_essential", False)
            ),
        )

    def _applicable(self, event: str, kwargs: dict[str, Any]) -> list[HookEntry]:
        """The entries of *event* that apply to this call (D13)."""
        entries = self._hooks.get(event, [])
        if not any(entry.bound for entry in entries):
            return list(entries)
        tool_name = kwargs.get("tool_name")
        category = self._category(tool_name)
        return [e for e in entries if e.applies_to(tool_name, category)]

    def entries(self, event: str) -> list[HookEntry]:
        """The registrations on *event*, declarations and all."""
        return list(self._hooks.get(event, ()))

    def callbacks(self, event: str) -> list[Callable]:
        """Just the callables on *event*, in registration order."""
        return [entry.callback for entry in self._hooks.get(event, ())]

    def make_outermost(self, event: str, entry: HookEntry) -> None:
        """Move *entry* to the front of the middleware chain for *event*.

        The chain runs first-registered outermost, and a middleware may
        return without calling ``handler()`` -- so anything ahead of a
        guard can answer a call the guard never sees. A guard that must be
        monotonic (the approval gate, spec D37/I10) says so here instead of
        relying on the order a config file happened to produce.
        """
        entries = self._middleware.get(event)
        if not entries or entries[0] is entry:
            return
        self._middleware[event] = [
            entry, *(e for e in entries if e is not entry)
        ]

    def middleware_entries(self, event: str) -> list[HookEntry]:
        """The middleware registrations on *event*, declarations and all."""
        return list(self._middleware.get(event, ()))

    def essential_entries(self) -> list[tuple[str, HookEntry]]:
        """Every essential registration, as ``(event, entry)`` pairs (D14).

        What derivation copies forward: a ToolSet rebuilt from a recipe gets
        the recipe's hooks back automatically, but an essential hook
        registered *outside* the recipe would otherwise be lost, and I7 says
        it must not be.
        """
        pairs: list[tuple[str, HookEntry]] = []
        for event, entries in self._hooks.items():
            pairs.extend((event, e) for e in entries if e.essential)
        for event, entries in self._middleware.items():
            pairs.extend((event, e) for e in entries if e.essential)
        return pairs

    def has_middleware(self, event: str, tool_name: Any = None) -> bool:
        """Whether any middleware on *event* applies to this tool."""
        entries = self._middleware.get(event, [])
        if not entries:
            return False
        if tool_name is None:
            return True
        category = self._category(tool_name)
        return any(e.applies_to(tool_name, category) for e in entries)

    def register(
        self,
        event: str,
        callback: Callable,
        *,
        tools: Iterable[str] | None = None,
        categories: Iterable[str] | None = None,
        essential: bool | None = None,
    ) -> HookEntry:
        """Register a callback for an event.

        Args:
            event: The event name (e.g. HOOK_BEFORE_MODEL_REQUEST).
            callback: The callback function, or a ready
                :class:`HookEntry`.
            tools: Tool names this hook applies to; ``None`` defers to what
                the callback declares, and an empty set means every tool
                (D13).
            categories: Tool categories, resolved through the registry's
                category binding.
            essential: Whether the hook survives derivation and refuses
                removal (D14); ``None`` defers to the callback.

        Returns:
            The :class:`HookEntry` that was stored -- the handle
            :meth:`unregister` wants when the same callable is registered
            twice with different bindings.

        Raises:
            UnwiredHookEvent: if nothing emits *event* -- an unknown name,
                or one declared but not yet wired (D15).
        """
        _check_event(event, middleware=False)
        entry = self._entry(callback, tools, categories, essential)
        self._hooks.setdefault(event, []).append(entry)
        return entry

    def register_middleware(
        self,
        event: str,
        handler: Callable,
        *,
        tools: Iterable[str] | None = None,
        categories: Iterable[str] | None = None,
        essential: bool | None = None,
    ) -> HookEntry:
        """Register a middleware handler for an event.

        Middleware handlers receive a 'handler' callable that they can
        invoke to execute the next layer in the middleware chain.

        Args:
            event: The event name (e.g. HOOK_WRAP_TOOL_EXECUTE).
            handler: The middleware handler callable.

        Raises:
            UnwiredHookEvent: if nothing emits *event*. Note that of the six
                declared ``wrap_*`` events only ``wrap_tool_execute`` is
                wired today; the rest are open (T2).
        """
        _check_event(event, middleware=True)
        entry = self._entry(handler, tools, categories, essential)
        self._middleware.setdefault(event, []).append(entry)
        return entry

    def register_ordered(
        self,
        capability_id: str,
        position: str | None = None,
        wraps: list[str] | None = None,
        wrapped_by: list[str] | None = None,
        requires: list[str] | None = None,
    ) -> None:
        """Register ordering constraints for a capability.

        Args:
            capability_id: The capability ID.
            position: 'outermost' or 'innermost', or None for user-provided order.
            wraps: Capability IDs that this capability wraps around.
            wrapped_by: Capability IDs that wrap around this capability.
            requires: Capability IDs that must be present.
        """
        self._ordering[capability_id] = {
            "position": position,
            "wraps": wraps or [],
            "wrapped_by": wrapped_by or [],
            "requires": requires or [],
        }

    def emit(self, event: str, **kwargs) -> None:
        """Emit an event with kwargs passed to callbacks.

        For `before_*` events, callbacks fire in registration order (FIFO).
        For `after_*` events, callbacks fire in reverse registration order (LIFO),
        matching Pydantic AI's hook ordering semantics.

        Args:
            event: The event name.
            **kwargs: Keyword arguments passed to callbacks.
        """
        # Per-tool binding is applied here, once, for every hook -- which is
        # the whole point of D13. A hook that only cares about `write_file`
        # used to have to say so in its own body, so the config could not
        # show it and two hooks could not agree on what "applies" means.
        entries = self._applicable(event, kwargs)
        if event.startswith("after_"):
            # After hooks fire in reverse order (LIFO)
            entries = list(reversed(entries))
        for entry in entries:
            try:
                entry.callback(**kwargs)
            except Exception:
                # Don't let hook exceptions break the run
                pass

    async def emit_middleware(
        self,
        event: str,
        innermost: Callable | None = None,
        **kwargs: Any,
    ) -> Any:
        """Emit a middleware event with kwargs passed to handlers.

        Middleware handlers receive a 'handler' callable that they can
        invoke to execute the next layer in the chain — or skip it to
        short-circuit (e.g. deny a tool call), or post-process its
        result (e.g. redact secrets).

        Args:
            event: The event name.
            innermost: The actual operation being wrapped (e.g. the tool
                execution). When None, the innermost layer returns
                ``kwargs.get("default_result")``.
            **kwargs: Keyword arguments passed to handlers.

        Returns:
            The result from the (possibly short-circuited) chain.
        """
        tool_name = kwargs.get("tool_name")
        category = self._category(tool_name)
        handlers = [
            e.callback for e in self._middleware.get(event, [])
            if e.applies_to(tool_name, category)
        ]

        async def _default_innermost(**kw):
            return kwargs.get("default_result", None)

        innermost_fn = innermost or _default_innermost

        async def _chain(index: int, **kw: Any) -> Any:
            # Index-based recursion over a stable list. A middleware may
            # invoke its ``handler`` more than once (retry / error-recovery
            # is an advertised use case); each invocation must re-run the
            # *remaining* chain. The old version popped from a shared list,
            # so a second ``handler()`` call found it emptied and silently
            # skipped every middleware between it and the innermost op.
            if index >= len(handlers):
                return await innermost_fn(**kw)
            handler = handlers[index]

            def _next(**overrides: Any) -> Any:
                # The context travels with the call. `await handler()` --
                # what every pass-through middleware writes -- used to hand
                # the *rest* of the chain no `tool_name` at all, so any
                # middleware behind a pass-through saw an anonymous call
                # and let it by. That is how the policy gate deferring to
                # the load floor (D77) and to the requires_approval floor
                # (D81) silently disabled both. Overrides still win, which
                # is what lets a middleware repair arguments.
                return _chain(index + 1, **{**kw, **overrides})

            return await handler(handler=_next, **kw)

        return await _chain(0, **kwargs)

    def unregister_middleware(self, event: str, handler: Callable) -> None:
        """Unregister a middleware handler for an event.

        Args:
            event: The event name.
            handler: The middleware handler to remove.
        """
        self._remove(self._middleware, event, handler)

    def unregister(self, event: str, callback: Callable) -> None:
        """Unregister a callback for an event.

        Args:
            event: The event name.
            callback: The callback function.
        """
        self._remove(self._hooks, event, callback)

    @staticmethod
    def _remove(
        store: dict[str, list[HookEntry]], event: str, target: Any
    ) -> None:
        """Drop *target* from *store*, refusing an essential hook (D14).

        *target* is either the callable or the :class:`HookEntry` handed
        back by ``register``; the entry form is what disambiguates one
        callable registered twice with different bindings.
        """
        entries = store.get(event)
        if not entries:
            return
        if isinstance(target, HookEntry):
            # The exact registration -- which is what tells two
            # registrations of one callable with different bindings apart.
            doomed = [e for e in entries if e is target]
        else:
            doomed = [e for e in entries if e.callback is target]
        for entry in doomed:
            if entry.essential:
                raise EssentialHookError(
                    f"Cannot unregister '{event}' hook "
                    f"{getattr(entry.callback, '__name__', entry.callback)!r}: "
                    "it is declared essential, so it rides the recipe into "
                    "every derived toolset and cannot be dropped downstream "
                    "(spec D14, invariant I7)."
                )
        store[event] = [e for e in entries if not any(e is d for d in doomed)]

    def clear(self, *, keep_essential: bool = True) -> None:
        """Clear registered hooks, keeping the essential tier (D14).

        Clearing is how a registry gets reused, and an essential hook that a
        reuse silently dropped would be exactly the failure I7 names. Pass
        ``keep_essential=False`` to tear the registry down completely --
        appropriate when the registry itself is being discarded, not when it
        is being re-populated.
        """
        if keep_essential:
            kept = {
                event: [e for e in entries if e.essential]
                for event, entries in self._hooks.items()
            }
            kept_mw = {
                event: [e for e in entries if e.essential]
                for event, entries in self._middleware.items()
            }
        else:
            kept, kept_mw = {}, {}
        self._hooks = {k: v for k, v in kept.items() if v}
        self._middleware = {k: v for k, v in kept_mw.items() if v}
        self._ordering.clear()

    def get_ordered_capabilities(self) -> list[str]:
        """Get capability IDs sorted by ordering constraints.

        Returns:
            List of capability IDs in execution order.
        """
        # Simple sort: outermost first, then innermost
        ordered = []
        remaining = list(self._ordering.keys())

        while remaining:
            # Find outermost or first remaining
            for cap_id in remaining:
                order = self._ordering[cap_id]
                if order["position"] == "outermost":
                    ordered.append(cap_id)
                    remaining.remove(cap_id)
                    break
            if remaining:
                # Add first remaining
                ordered.append(remaining.pop(0))

        return ordered


# â”€â”€ Middleware Hook Decorators â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def on_model_request_error(func: Callable) -> Callable:
    """Decorator to register an on_model_request_error handler."""
    func._hook_event = HOOK_ON_MODEL_REQUEST_ERROR  # type: ignore[attr-defined]
    return func


def wrap_tool_validate(func: Callable) -> Callable:
    """Decorator to register a wrap_tool_validate middleware."""
    func._hook_event = HOOK_WRAP_TOOL_VALIDATE  # type: ignore[attr-defined]
    func._is_middleware = True  # type: ignore[attr-defined]
    return func


def on_tool_validate_error(func: Callable) -> Callable:
    """Decorator to register an on_tool_validate_error handler."""
    func._hook_event = HOOK_ON_TOOL_VALIDATE_ERROR  # type: ignore[attr-defined]
    return func


def wrap_tool_execute(func: Callable) -> Callable:
    """Decorator to register a wrap_tool_execute middleware."""
    func._hook_event = HOOK_WRAP_TOOL_EXECUTE  # type: ignore[attr-defined]
    func._is_middleware = True  # type: ignore[attr-defined]
    return func


def on_tool_execute_error(func: Callable) -> Callable:
    """Decorator to register an on_tool_execute_error handler."""
    func._hook_event = HOOK_ON_TOOL_EXECUTE_ERROR  # type: ignore[attr-defined]
    return func


def on_output_validate_error(func: Callable) -> Callable:
    """Decorator to register an on_output_validate_error handler."""
    func._hook_event = HOOK_ON_OUTPUT_VALIDATE_ERROR  # type: ignore[attr-defined]
    return func


def on_output_process_error(func: Callable) -> Callable:
    """Decorator to register an on_output_process_error handler."""
    func._hook_event = HOOK_ON_OUTPUT_PROCESS_ERROR  # type: ignore[attr-defined]
    return func


# â”€â”€ Hook Context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class HookContext:
    """Context passed to hook callbacks.

    Attributes:
        engine: The ChatEngine instance.
        run_context: The RunContext for the current run.
        request: The model request (for before_model_request).
        response: The model response (for after_model_response).
        tool_name: The tool name (for before/after_tool_execute).
        tool_args: The tool arguments (for before/after_tool_execute).
        tool_result: The tool result (for after_tool_execute).
        error: The error (for on_*_error hooks).
    """

    def __init__(
        self,
        engine: ChatEngine | None = None,
        run_context: RunContext | None = None,
        request: dict | None = None,
        response: dict | None = None,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        tool_result: str | None = None,
        error: Exception | None = None,
        event_stream: AsyncIterable | None = None,
    ) -> None:
        self.engine = engine
        self.run_context = run_context
        self.request = request
        self.response = response
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result
        self.error = error
        self.event_stream = event_stream


# â”€â”€ Hooks Capability â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class Hooks:
    """Decorator-based lifecycle hook registration capability.

    The recommended way to add lifecycle hooks for application-level
    concerns like logging, metrics, and lightweight validation. No
    subclassing needed.

    Matches Pydantic AI's `Hooks` capability pattern.

    Attributes:
        id: Unique identifier for the capability.
        description: Human-readable description.
        defer_loading: If True, hooks only fire after the capability is loaded.
        _hooks: Internal dict mapping event names to lists of callbacks.
    """

    def __init__(
        self,
        id: str = "hooks",
        description: str = "Lifecycle hooks for observability and error recovery.",
        defer_loading: bool = False,
    ) -> None:
        self.id = id
        self.description = description
        self.defer_loading = defer_loading
        self._registry: HookRegistry = HookRegistry()
        self._on = _HooksOn(self._registry)

    @property
    def on(self) -> _HooksOn:
        """Decorator namespace for registering hooks.

        Usage:
            @hooks.on.before_model_request
            async def log_request(ctx, request):
                print(f'Sending {len(request["messages"])} messages')
                return request
        """
        return self._on

    def get_tools(self) -> list[dict]:
        """Hooks don't provide tools.

        Returns:
            Empty list.
        """
        return []

    def get_instructions(self) -> str:
        """Hooks don't provide instructions.

        Returns:
            Empty string.
        """
        return ""

    async def on_model_request_error(
        self,
        ctx: RunContext,
        error: Exception,
    ) -> Any:
        """Called when a model request fails (raise-to-propagate, return-to-recover)."""
        # `_hooks` holds `HookEntry` records, not bare callables:
        # calling the entry raised TypeError inside the handler
        # that exists to recover from an error (D15).
        for entry in self._registry._hooks.get(HOOK_ON_MODEL_REQUEST_ERROR, []):
            try:
                result = entry.callback(ctx=ctx, error=error)
                if result is not None:
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
            except Exception as e:
                raise e  # Stop chain on first raise
        raise error

    async def wrap_tool_execute(
        self,
        ctx: RunContext,
        tool_name: str,
        tool_args: dict,
        handler: Callable[[str, dict], Awaitable[str]],
    ) -> str:
        """Wraps tool execution via registered middleware."""
        return await self._registry.emit_middleware(
            HOOK_WRAP_TOOL_EXECUTE,
            default_result=await handler(tool_name, tool_args),
        )

    async def on_tool_execute_error(
        self,
        ctx: RunContext,
        tool_name: str,
        error: Exception,
    ) -> str:
        """Called when tool execution fails (raise-to-propagate, return-to-recover)."""
        # `_hooks` holds `HookEntry` records, not bare callables:
        # calling the entry raised TypeError inside the handler
        # that exists to recover from an error (D15).
        for entry in self._registry._hooks.get(HOOK_ON_TOOL_EXECUTE_ERROR, []):
            try:
                result = entry.callback(ctx=ctx, tool_name=tool_name, error=error)
                if result is not None:
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
            except Exception as e:
                raise e
        raise error

    async def on_output_validate_error(
        self,
        ctx: RunContext,
        output: Any,
        error: Exception,
    ) -> Any:
        """Called when output validation fails (raise-to-propagate, return-to-recover)."""
        # `_hooks` holds `HookEntry` records, not bare callables:
        # calling the entry raised TypeError inside the handler
        # that exists to recover from an error (D15).
        for entry in self._registry._hooks.get(HOOK_ON_OUTPUT_VALIDATE_ERROR, []):
            try:
                result = entry.callback(ctx=ctx, output=output, error=error)
                if result is not None:
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
            except Exception as e:
                raise e
        raise error

    async def on_output_process_error(
        self,
        ctx: RunContext,
        output: Any,
        error: Exception,
    ) -> Any:
        """Called when output processing fails (raise-to-propagate, return-to-recover)."""
        # `_hooks` holds `HookEntry` records, not bare callables:
        # calling the entry raised TypeError inside the handler
        # that exists to recover from an error (D15).
        for entry in self._registry._hooks.get(HOOK_ON_OUTPUT_PROCESS_ERROR, []):
            try:
                result = entry.callback(ctx=ctx, output=output, error=error)
                if result is not None:
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
            except Exception as e:
                raise e
        raise error

    def emit(self, event: str, **kwargs) -> None:
        """Emit an event to registered callbacks."""
        self._registry.emit(event, **kwargs)

    async def emit_middleware(self, event: str, **kwargs) -> Any:
        """Emit a middleware event to registered handlers."""
        return await self._registry.emit_middleware(event, **kwargs)


class _HooksOn:
    """Decorator namespace for registering hooks.

    Provides `@hooks.on.before_model_request`, `@hooks.on.model_request`, etc.
    matching Pydantic AI's `hooks.on.*` pattern.
    """

    def __init__(self, registry: HookRegistry) -> None:
        self._registry = registry

    def before_model_request(self, func: Any) -> Any:
        """Register a before_model_request hook."""
        func._hook_event = HOOK_BEFORE_MODEL_REQUEST
        self._registry.register(HOOK_BEFORE_MODEL_REQUEST, func)
        return func

    def after_model_request(self, func: Any) -> Any:
        """Register an after_model_request hook."""
        func._hook_event = HOOK_AFTER_MODEL_REQUEST
        self._registry.register(HOOK_AFTER_MODEL_REQUEST, func)
        return func

    def after_model_response(self, func: Any) -> Any:
        """Register an after_model_response hook."""
        func._hook_event = HOOK_AFTER_MODEL_RESPONSE
        self._registry.register(HOOK_AFTER_MODEL_RESPONSE, func)
        return func

    def on_model_request_error(self, func: Any) -> Any:
        """Register an on_model_request_error handler."""
        func._hook_event = HOOK_ON_MODEL_REQUEST_ERROR
        self._registry.register(HOOK_ON_MODEL_REQUEST_ERROR, func)
        return func

    def before_tool_execute(self, func: Any) -> Any:
        """Register a before_tool_execute hook."""
        func._hook_event = HOOK_BEFORE_TOOL_EXECUTE
        self._registry.register(HOOK_BEFORE_TOOL_EXECUTE, func)
        return func

    def after_tool_execute(self, func: Any) -> Any:
        """Register an after_tool_execute hook."""
        func._hook_event = HOOK_AFTER_TOOL_EXECUTE
        self._registry.register(HOOK_AFTER_TOOL_EXECUTE, func)
        return func

    def wrap_tool_execute(self, func: Any) -> Any:
        """Register a wrap_tool_execute (middleware) hook."""
        func._hook_event = HOOK_WRAP_TOOL_EXECUTE
        func._is_middleware = True
        self._registry.register_middleware(HOOK_WRAP_TOOL_EXECUTE, func)
        return func

    def on_tool_execute_error(self, func: Any) -> Any:
        """Register an on_tool_execute_error handler."""
        func._hook_event = HOOK_ON_TOOL_EXECUTE_ERROR
        self._registry.register(HOOK_ON_TOOL_EXECUTE_ERROR, func)
        return func

    def before_run(self, func: Any) -> Any:
        """Register a before_run hook."""
        func._hook_event = HOOK_BEFORE_RUN
        self._registry.register(HOOK_BEFORE_RUN, func)
        return func

    def after_run(self, func: Any) -> Any:
        """Register an after_run hook."""
        func._hook_event = HOOK_AFTER_RUN
        self._registry.register(HOOK_AFTER_RUN, func)
        return func

    def on_output_validate_error(self, func: Any) -> Any:
        """Register an on_output_validate_error handler."""
        func._hook_event = HOOK_ON_OUTPUT_VALIDATE_ERROR
        self._registry.register(HOOK_ON_OUTPUT_VALIDATE_ERROR, func)
        return func

    def on_output_process_error(self, func: Any) -> Any:
        """Register an on_output_process_error handler."""
        func._hook_event = HOOK_ON_OUTPUT_PROCESS_ERROR
        self._registry.register(HOOK_ON_OUTPUT_PROCESS_ERROR, func)
        return func



# ── Hook-map registration helpers ─────────────────────────────────────────
#
# A "hook map" is the {event: callable | [callables]} shape carried by
# Role.hooks, capability.get_hooks() and the hook catalog. These helpers
# are the single place that knows wrap_* events go to the middleware
# registry — every registration site (RoleBinding, capability loading,
# catalog attachment) routes through them so removal is always symmetric.


def is_middleware_event(event: str) -> bool:
    """Whether *event* belongs to the middleware (wrap_*) layer."""
    return event.startswith("wrap_")


def register_hook_map(
    registry: HookRegistry,
    mapping: dict[str, Any] | None,
) -> list[tuple[str, Any]]:
    """Register a hook map; returns (event, entry) pairs for removal.

    A map's values may be bare callables or :class:`HookEntry` instances,
    so a hook map -- a plain dict with no room for keywords -- can still
    carry a per-tool binding or the essential flag (D13, D14). A callable
    decorated with :func:`bind_tools` or :func:`essential` carries them too.

    All or nothing. Registration refuses an event nothing emits
    (:class:`UnwiredHookEvent`), and a map that names one is a
    *configuration* error -- a Role or capability asking to be called by
    something that will never call it. Half-installing the rest would leave
    the run in a state nobody described, so anything already registered is
    rolled back and the error propagates to whoever is building the run.
    """
    registered: list[tuple[str, Any]] = []
    try:
        for event, callbacks in (mapping or {}).items():
            if not isinstance(callbacks, (list, tuple)):
                callbacks = [callbacks]
            for callback in callbacks:
                if is_middleware_event(event):
                    entry = registry.register_middleware(event, callback)
                else:
                    entry = registry.register(event, callback)
                registered.append((event, entry))
    except UnwiredHookEvent:
        unregister_hook_map(registry, registered)
        raise
    return registered


def unregister_hook_map(
    registry: HookRegistry,
    registered: list[tuple[str, Any]],
) -> None:
    """Remove exactly the pairs returned by :func:`register_hook_map`.

    Raises :class:`EssentialHookError` if the map contained an essential
    hook: a layer that installed one does not get to take it away again
    (D14). Deactivating a Role therefore cannot strip the infrastructure
    tier, which is the direction I7 cares about.
    """
    for event, entry in registered:
        if is_middleware_event(event):
            registry.unregister_middleware(event, entry)
        else:
            registry.unregister(event, entry)
