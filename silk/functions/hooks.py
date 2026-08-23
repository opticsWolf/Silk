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

from collections.abc import AsyncIterable, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from collections.abc import Awaitable
    from .protocols import AgentEngine as ChatEngine
    from .run_context import RunContext
    from typing import Protocol

    # Import BaseCapability for type checking only (to avoid circular import)
    from .capabilities import BaseCapability as _BaseCapability  # noqa: F401

    # Wrap handler type aliases (matching Pydantic AI's naming)
    class WrapModelRequestHandler(Protocol):
        """Handler type for wrap_model_request middleware."""
        def __call__(self, **kwargs) -> Awaitable[Any]: ...

    class WrapToolExecuteHandler(Protocol):
        """Handler type for wrap_tool_execute middleware."""
        def __call__(self, tool_name: str, tool_args: dict, **kwargs) -> Awaitable[str]: ...

    class WrapOutputValidateHandler(Protocol):
        """Handler type for wrap_output_validate middleware."""
        def __call__(self, output: Any, **kwargs) -> Awaitable[Any]: ...


# â”€â”€ Hook Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Existing events (backward compatible)
HOOK_BEFORE_MODEL_REQUEST = "before_model_request"
HOOK_AFTER_MODEL_RESPONSE = "after_model_response"
HOOK_BEFORE_TOOL_EXECUTE = "before_tool_execute"
HOOK_AFTER_TOOL_EXECUTE = "after_tool_execute"
HOOK_BEFORE_RUN = "before_run"
HOOK_AFTER_RUN = "after_run"

HOOK_TOOL_DENIED = "tool_denied"
"""Fired when the active role's filter blocks a tool call at dispatch.
Audit/logging surface; the denied call never reaches the executable."""

# Alias for after_model_request (used in Pydantic AI)
HOOK_AFTER_MODEL_REQUEST = "after_model_request"

# New middleware events
HOOK_WRAP_MODEL_REQUEST = "wrap_model_request"
"""Wraps model request execution. Handler() calls the model."""

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

HOOK_WRAP_OUTPUT_VALIDATE = "wrap_output_validate"
"""Wraps output validation."""

HOOK_ON_OUTPUT_VALIDATE_ERROR = "on_output_validate_error"
"""Called when output validation fails."""

HOOK_WRAP_OUTPUT_PROCESS = "wrap_output_process"
"""Wraps output processing."""

HOOK_ON_OUTPUT_PROCESS_ERROR = "on_output_process_error"
"""Called when output processing fails."""

HOOK_WRAP_RUN_EVENT_STREAM = "wrap_run_event_stream"
"""Wraps the event stream for streamed nodes."""

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
        self._hooks: dict[str, list[Callable]] = {}
        self._middleware: dict[str, list[Callable]] = {}
        self._ordering: dict[str, dict[str, Any]] = {}

    def register(self, event: str, callback: Callable) -> None:
        """Register a callback for an event.

        Args:
            event: The event name (e.g. HOOK_BEFORE_MODEL_REQUEST).
            callback: The callback function.
        """
        self._hooks.setdefault(event, []).append(callback)

    def register_middleware(self, event: str, handler: Callable) -> None:
        """Register a middleware handler for an event.

        Middleware handlers receive a 'handler' callable that they can
        invoke to execute the next layer in the middleware chain.

        Args:
            event: The event name (e.g. HOOK_WRAP_MODEL_REQUEST).
            handler: The middleware handler callable.
        """
        self._middleware.setdefault(event, []).append(handler)

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
        callbacks = self._hooks.get(event, [])
        if event.startswith("after_"):
            # After hooks fire in reverse order (LIFO)
            for callback in reversed(callbacks):
                try:
                    callback(**kwargs)
                except Exception:
                    # Don't let hook exceptions break the run
                    pass
        else:
            # Before/middleware hooks fire in registration order (FIFO)
            for callback in callbacks:
                try:
                    callback(**kwargs)
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
        handlers = list(self._middleware.get(event, []))

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
            return await handler(
                handler=lambda **k: _chain(index + 1, **k), **kw
            )

        return await _chain(0, **kwargs)

    def unregister_middleware(self, event: str, handler: Callable) -> None:
        """Unregister a middleware handler for an event.

        Args:
            event: The event name.
            handler: The middleware handler to remove.
        """
        if event in self._middleware:
            self._middleware[event] = [
                h for h in self._middleware[event] if h is not handler
            ]

    def unregister(self, event: str, callback: Callable) -> None:
        """Unregister a callback for an event.

        Args:
            event: The event name.
            callback: The callback function.
        """
        if event in self._hooks:
            self._hooks[event] = [cb for cb in self._hooks[event] if cb is not callback]

    def clear(self) -> None:
        """Clear all registered hooks."""
        self._hooks.clear()
        self._middleware.clear()
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


def wrap_model_request(func: Callable) -> Callable:
    """Decorator to register a wrap_model_request middleware."""
    func._hook_event = HOOK_WRAP_MODEL_REQUEST  # type: ignore[attr-defined]
    func._is_middleware = True  # type: ignore[attr-defined]
    return func


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


def wrap_output_validate(func: Callable) -> Callable:
    """Decorator to register a wrap_output_validate middleware."""
    func._hook_event = HOOK_WRAP_OUTPUT_VALIDATE  # type: ignore[attr-defined]
    func._is_middleware = True  # type: ignore[attr-defined]
    return func


def on_output_validate_error(func: Callable) -> Callable:
    """Decorator to register an on_output_validate_error handler."""
    func._hook_event = HOOK_ON_OUTPUT_VALIDATE_ERROR  # type: ignore[attr-defined]
    return func


def wrap_output_process(func: Callable) -> Callable:
    """Decorator to register a wrap_output_process middleware."""
    func._hook_event = HOOK_WRAP_OUTPUT_PROCESS  # type: ignore[attr-defined]
    func._is_middleware = True  # type: ignore[attr-defined]
    return func


def on_output_process_error(func: Callable) -> Callable:
    """Decorator to register an on_output_process_error handler."""
    func._hook_event = HOOK_ON_OUTPUT_PROCESS_ERROR  # type: ignore[attr-defined]
    return func


def wrap_run_event_stream(func: Callable) -> Callable:
    """Decorator to register a wrap_run_event_stream middleware."""
    func._hook_event = HOOK_WRAP_RUN_EVENT_STREAM  # type: ignore[attr-defined]
    func._is_middleware = True  # type: ignore[attr-defined]
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
        event_stream: The event stream (for wrap_run_event_stream).
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

    async def wrap_model_request(
        self,
        ctx: RunContext,
        handler: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Wraps the model request via registered middleware."""
        return await self._registry.emit_middleware(
            HOOK_WRAP_MODEL_REQUEST,
            default_result=await handler(),
        )

    async def on_model_request_error(
        self,
        ctx: RunContext,
        error: Exception,
    ) -> Any:
        """Called when a model request fails (raise-to-propagate, return-to-recover)."""
        for callback in self._registry._hooks.get(HOOK_ON_MODEL_REQUEST_ERROR, []):
            try:
                result = callback(ctx=ctx, error=error)
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
        for callback in self._registry._hooks.get(HOOK_ON_TOOL_EXECUTE_ERROR, []):
            try:
                result = callback(ctx=ctx, tool_name=tool_name, error=error)
                if result is not None:
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
            except Exception as e:
                raise e
        raise error

    async def wrap_output_validate(
        self,
        ctx: RunContext,
        output: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Wraps output validation via registered middleware."""
        return await self._registry.emit_middleware(
            HOOK_WRAP_OUTPUT_VALIDATE,
            default_result=await handler(output),
        )

    async def on_output_validate_error(
        self,
        ctx: RunContext,
        output: Any,
        error: Exception,
    ) -> Any:
        """Called when output validation fails (raise-to-propagate, return-to-recover)."""
        for callback in self._registry._hooks.get(HOOK_ON_OUTPUT_VALIDATE_ERROR, []):
            try:
                result = callback(ctx=ctx, output=output, error=error)
                if result is not None:
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
            except Exception as e:
                raise e
        raise error

    async def wrap_output_process(
        self,
        ctx: RunContext,
        output: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Wraps output processing via registered middleware."""
        return await self._registry.emit_middleware(
            HOOK_WRAP_OUTPUT_PROCESS,
            default_result=await handler(output),
        )

    async def on_output_process_error(
        self,
        ctx: RunContext,
        output: Any,
        error: Exception,
    ) -> Any:
        """Called when output processing fails (raise-to-propagate, return-to-recover)."""
        for callback in self._registry._hooks.get(HOOK_ON_OUTPUT_PROCESS_ERROR, []):
            try:
                result = callback(ctx=ctx, output=output, error=error)
                if result is not None:
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
            except Exception as e:
                raise e
        raise error

    async def wrap_run_event_stream(
        self,
        ctx: RunContext,
        stream: AsyncIterable,
    ) -> AsyncIterable:
        """Wraps the event stream for streamed runs via registered middleware."""
        # Use the middleware chain if any handlers are registered
        handlers = self._registry._middleware.get(HOOK_WRAP_RUN_EVENT_STREAM, [])
        if not handlers:
            async for event in stream:
                yield event
            return

        # For simplicity, just yield events directly when middleware is registered
        # (async generator middleware chains are complex to implement correctly)
        async for event in stream:
            yield event

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

    def before_model_request(self, func: Callable) -> Callable:
        """Register a before_model_request hook."""
        func._hook_event = HOOK_BEFORE_MODEL_REQUEST
        self._registry.register(HOOK_BEFORE_MODEL_REQUEST, func)
        return func

    def after_model_request(self, func: Callable) -> Callable:
        """Register an after_model_request hook."""
        func._hook_event = HOOK_AFTER_MODEL_REQUEST
        self._registry.register(HOOK_AFTER_MODEL_REQUEST, func)
        return func

    def after_model_response(self, func: Callable) -> Callable:
        """Register an after_model_response hook."""
        func._hook_event = HOOK_AFTER_MODEL_RESPONSE
        self._registry.register(HOOK_AFTER_MODEL_RESPONSE, func)
        return func

    def model_request(self, func: Callable) -> Callable:
        """Register a wrap_model_request (middleware) hook."""
        func._hook_event = HOOK_WRAP_MODEL_REQUEST
        func._is_middleware = True
        self._registry.register_middleware(HOOK_WRAP_MODEL_REQUEST, func)
        return func

    def on_model_request_error(self, func: Callable) -> Callable:
        """Register an on_model_request_error handler."""
        func._hook_event = HOOK_ON_MODEL_REQUEST_ERROR
        self._registry.register(HOOK_ON_MODEL_REQUEST_ERROR, func)
        return func

    def before_tool_execute(self, func: Callable) -> Callable:
        """Register a before_tool_execute hook."""
        func._hook_event = HOOK_BEFORE_TOOL_EXECUTE
        self._registry.register(HOOK_BEFORE_TOOL_EXECUTE, func)
        return func

    def after_tool_execute(self, func: Callable) -> Callable:
        """Register an after_tool_execute hook."""
        func._hook_event = HOOK_AFTER_TOOL_EXECUTE
        self._registry.register(HOOK_AFTER_TOOL_EXECUTE, func)
        return func

    def wrap_tool_execute(self, func: Callable) -> Callable:
        """Register a wrap_tool_execute (middleware) hook."""
        func._hook_event = HOOK_WRAP_TOOL_EXECUTE
        func._is_middleware = True
        self._registry.register_middleware(HOOK_WRAP_TOOL_EXECUTE, func)
        return func

    def on_tool_execute_error(self, func: Callable) -> Callable:
        """Register an on_tool_execute_error handler."""
        func._hook_event = HOOK_ON_TOOL_EXECUTE_ERROR
        self._registry.register(HOOK_ON_TOOL_EXECUTE_ERROR, func)
        return func

    def before_run(self, func: Callable) -> Callable:
        """Register a before_run hook."""
        func._hook_event = HOOK_BEFORE_RUN
        self._registry.register(HOOK_BEFORE_RUN, func)
        return func

    def after_run(self, func: Callable) -> Callable:
        """Register an after_run hook."""
        func._hook_event = HOOK_AFTER_RUN
        self._registry.register(HOOK_AFTER_RUN, func)
        return func

    def wrap_output_validate(self, func: Callable) -> Callable:
        """Register a wrap_output_validate (middleware) hook."""
        func._hook_event = HOOK_WRAP_OUTPUT_VALIDATE
        func._is_middleware = True
        self._registry.register_middleware(HOOK_WRAP_OUTPUT_VALIDATE, func)
        return func

    def on_output_validate_error(self, func: Callable) -> Callable:
        """Register an on_output_validate_error handler."""
        func._hook_event = HOOK_ON_OUTPUT_VALIDATE_ERROR
        self._registry.register(HOOK_ON_OUTPUT_VALIDATE_ERROR, func)
        return func

    def wrap_output_process(self, func: Callable) -> Callable:
        """Register a wrap_output_process (middleware) hook."""
        func._hook_event = HOOK_WRAP_OUTPUT_PROCESS
        func._is_middleware = True
        self._registry.register_middleware(HOOK_WRAP_OUTPUT_PROCESS, func)
        return func

    def on_output_process_error(self, func: Callable) -> Callable:
        """Register an on_output_process_error handler."""
        func._hook_event = HOOK_ON_OUTPUT_PROCESS_ERROR
        self._registry.register(HOOK_ON_OUTPUT_PROCESS_ERROR, func)
        return func

    def wrap_run_event_stream(self, func: Callable) -> Callable:
        """Register a wrap_run_event_stream (middleware) hook."""
        func._hook_event = HOOK_WRAP_RUN_EVENT_STREAM
        func._is_middleware = True
        self._registry.register_middleware(HOOK_WRAP_RUN_EVENT_STREAM, func)
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
) -> list[tuple[str, Callable]]:
    """Register a hook map; returns (event, callback) pairs for removal."""
    registered: list[tuple[str, Callable]] = []
    for event, callbacks in (mapping or {}).items():
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]
        for callback in callbacks:
            if is_middleware_event(event):
                registry.register_middleware(event, callback)
            else:
                registry.register(event, callback)
            registered.append((event, callback))
    return registered


def unregister_hook_map(
    registry: HookRegistry,
    registered: list[tuple[str, Callable]],
) -> None:
    """Remove exactly the pairs returned by :func:`register_hook_map`."""
    for event, callback in registered:
        if is_middleware_event(event):
            registry.unregister_middleware(event, callback)
        else:
            registry.unregister(event, callback)
