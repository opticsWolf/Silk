"""Composable capabilities for bundling tools, instructions, and settings.

Capabilities are reusable units of agent behavior that bundle tools,
instructions, model settings, and lifecycle hooks into composable units.

Based on Pydantic AI's capability architecture with adaptations for silk.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from .run_context import RunContext
    from .toolset import PrefixedToolSet


class CapabilityOrdering:
    """Ordering constraints for a capability within the middleware chain.

    Attributes:
        position: 'outermost', 'innermost', or None for user-provided order.
        wraps: Capability IDs that this capability wraps around.
        wrapped_by: Capability IDs that wrap around this capability.
        requires: Capability IDs that must be present.
    """

    def __init__(
        self,
        position: str | None = None,
        wraps: list[str] | None = None,
        wrapped_by: list[str] | None = None,
        requires: list[str] | None = None,
    ) -> None:
        self.position = position
        self.wraps = wraps or []
        self.wrapped_by = wrapped_by or []
        self.requires = requires or []


class BaseCapability(ABC):
    """Base class for capabilities with middleware hook support.

    Attributes:
        id: Unique identifier for the capability.
        description: Human-readable description of the capability.
        defer_loading: If True, the capability is loaded on-demand.
    """

    def __init__(
        self,
        id: str,
        description: str,
        defer_loading: bool = False,
    ) -> None:
        self.id = id
        self.description = description
        self.defer_loading = defer_loading

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """Return tools provided by this capability.

        Returns:
            A list of tool definitions (as dicts).
        """
        ...

    @abstractmethod
    def get_instructions(self) -> str:
        """Return instructions provided by this capability.

        Returns:
            A string of instructions for the model.
        """
        ...

    def get_model_settings(self) -> dict[str, Any]:
        """Return model settings provided by this capability.

        Returns:
            A dict of model settings (e.g. temperature, top_p).
        """
        return {}

    def get_hooks(self) -> dict[str, Any]:
        """Return lifecycle hooks provided by this capability.

        Registered by ``ToolBox._load_capability`` when the capability
        loads (and removed again when a RoleBinding deactivates).

        Returns:
            A dict mapping HOOK_* event names to a callable or a list of
            callables. Default: no hooks.
        """
        return {}

    def get_ordering(self) -> CapabilityOrdering | None:
        """Return ordering constraints for this capability.

        Override to declare a fixed position ('outermost' / 'innermost'),
        relative ordering (wraps / wrapped_by other capability IDs),
        or dependency requirements (requires).

        Returns:
            A CapabilityOrdering instance, or None for default behavior.
        """
        return None

    def get_description(self) -> str | Callable[[RunContext], str] | None:
        """Return a human-readable description of this capability.

        Override to return a callable that receives RunContext for dynamic
        descriptions (e.g. shown in the load_capability catalog).

        Returns:
            A static string, a callable receiving RunContext, or None.
        """
        return self.description

    async def for_run(self, ctx: RunContext) -> BaseCapability:
        """Return the capability instance to use for this agent run.

        Called once per run, before get_*() re-extraction and before any hooks fire.
        Override to return a fresh instance for per-run state isolation.
        Default: return self (shared across runs).

        Args:
            ctx: The run context.

        Returns:
            The capability instance for this run.
        """
        return self

    # â”€â”€ Run lifecycle hooks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def before_run(self, ctx: RunContext) -> None:
        """Called before the agent run starts. Observe-only.

        Use wrap_run for modification.

        Args:
            ctx: The run context.
        """
        pass

    async def after_run(self, ctx: RunContext, result: Any) -> Any:
        """Called after the agent run completes. Can modify the result.

        Args:
            ctx: The run context.
            result: The run result.

        Returns:
            The (possibly modified) run result.
        """
        return result

    async def wrap_run(
        self,
        ctx: RunContext,
        handler: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Wraps the entire agent run. handler() executes the run.

        If handler() raises and this method catches the exception and
        returns a result instead, the error is suppressed and the recovery
        result is used.

        If this method does not call handler() (short-circuit), the run
        is skipped and the returned result is used directly.

        Args:
            ctx: The run context.
            handler: A callable that executes the agent run.

        Returns:
            The run result or a short-circuit result.
        """
        return await handler()

    async def on_run_error(self, ctx: RunContext, error: Exception) -> Any:
        """Called when the agent run fails with an exception.

        This is the error counterpart to after_run: while after_run is
        called on success, on_run_error is called on failure (after
        wrap_run has had its chance to recover).

        Raise the original error (or a different exception) to propagate it.
        Return a result to suppress the error and recover the run.

        Args:
            ctx: The run context.
            error: The exception that was raised.

        Returns:
            A recovery result, or raise to propagate.
        """
        raise error

    # â”€â”€ Model request lifecycle hooks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def before_model_request(
        self,
        ctx: RunContext,
        request_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Called before each model request. Can modify messages, settings, and parameters.

        Args:
            ctx: The run context.
            request_context: The model request context (messages, settings, etc.).

        Returns:
            The (possibly modified) request context.
        """
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext,
        request_context: dict[str, Any],
        response: Any,
    ) -> Any:
        """Called after each model response. Can modify the response before further processing.

        Raise ModelRetry to reject the response and ask the model to try again.
        The original response is still appended to message history so the model
        can see what it said.

        Args:
            ctx: The run context.
            request_context: The model request context.
            response: The model response.

        Returns:
            The (possibly modified) model response.
        """
        return response

    # â”€â”€ Tool validate lifecycle hooks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def before_tool_validate(
        self,
        ctx: RunContext,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Modify raw args before validation.

        Raise ModelRetry to skip validation and ask the model to redo the tool call.

        Args:
            ctx: The run context.
            tool_name: The name of the tool.
            tool_args: The raw tool arguments.

        Returns:
            The (possibly modified) raw tool arguments.
        """
        return tool_args

    async def after_tool_validate(
        self,
        ctx: RunContext,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Modify validated args. Called only on successful validation.

        Raise ModelRetry to reject the validated args and ask the model to redo the tool call.

        Args:
            ctx: The run context.
            tool_name: The name of the tool.
            tool_args: The validated tool arguments.

        Returns:
            The (possibly modified) validated tool arguments.
        """
        return tool_args

    # â”€â”€ Tool execute lifecycle hooks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def before_tool_execute(
        self,
        ctx: RunContext,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Modify validated args before execution.

        Raise ModelRetry to skip execution and ask the model to redo the tool call.

        Args:
            ctx: The run context.
            tool_name: The name of the tool.
            tool_args: The validated tool arguments.

        Returns:
            The (possibly modified) validated tool arguments.
        """
        return tool_args

    async def after_tool_execute(
        self,
        ctx: RunContext,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Modify result after execution.

        Raise ModelRetry to reject the tool result and ask the model to redo the tool call.

        Args:
            ctx: The run context.
            tool_name: The name of the tool.
            tool_args: The validated tool arguments.
            result: The tool result.

        Returns:
            The (possibly modified) tool result.
        """
        return result

    # â”€â”€ Output validate lifecycle hooks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def before_output_validate(
        self,
        ctx: RunContext,
        output: Any,
    ) -> Any:
        """Modify raw model output before validation/parsing.

        The primary hook for pre-parse repair and normalization of model output.
        Fires only for structured output that requires parsing.

        Raise ModelRetry to skip validation and ask the model to try again.

        Args:
            ctx: The run context.
            output: The raw output.

        Returns:
            The (possibly modified) raw output.
        """
        return output

    async def after_output_validate(
        self,
        ctx: RunContext,
        output: Any,
    ) -> Any:
        """Modify validated output after successful parsing. Called only on success.

        Raise ModelRetry to reject the validated output and ask the model to try again.

        Args:
            ctx: The run context.
            output: The validated output.

        Returns:
            The (possibly modified) validated output.
        """
        return output

    # â”€â”€ Output process lifecycle hooks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def before_output_process(
        self,
        ctx: RunContext,
        output: Any,
    ) -> Any:
        """Modify validated output before processing (extraction, output function call).

        Raise ModelRetry to skip processing and ask the model to try again.

        Args:
            ctx: The run context.
            output: The validated output.

        Returns:
            The (possibly modified) output.
        """
        return output

    async def after_output_process(
        self,
        ctx: RunContext,
        output: Any,
    ) -> Any:
        """Modify result after output processing.

        Raise ModelRetry to reject the result and ask the model to try again.

        Args:
            ctx: The run context.
            output: The processed output.

        Returns:
            The (possibly modified) output.
        """
        return output

    # â”€â”€ Tool preparation hooks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def prepare_tools(
        self,
        ctx: RunContext,
        tool_defs: list[dict],
    ) -> list[dict]:
        """Filter or modify function tool definitions for this step.

        Return a filtered or modified list. The result flows into both
        the model's request parameters and ToolBox.tools, so filtering
        also blocks tool execution.

        Args:
            ctx: The run context.
            tool_defs: The list of tool definitions.

        Returns:
            The (possibly filtered/modified) list of tool definitions.
        """
        return tool_defs

    async def prepare_output_tools(
        self,
        ctx: RunContext,
        tool_defs: list[dict],
    ) -> list[dict]:
        """Filter or modify output tool definitions for this step.

        Return a filtered or modified list. The result flows into both
        the model's request parameters and ToolBox.tools.

        Args:
            ctx: The run context.
            tool_defs: The list of output tool definitions.

        Returns:
            The (possibly filtered/modified) list of output tool definitions.
        """
        return tool_defs

    # â”€â”€ Deferred tool call hooks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext,
        requests: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Handle deferred tool calls (approval-required or externally-executed).

        Called when a tool raises ApprovalRequired or CallDeferred during
        execution, or the model calls a tool registered with requires_approval=True.

        Return a dict of resolved tool call results to resolve some or all calls.
        Return None to leave all calls unresolved.

        Args:
            ctx: The run context.
            requests: The deferred tool call requests.

        Returns:
            Resolved tool call results, or None.
        """
        return None

    # â”€â”€ Wrapper toolset hook â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def get_wrapper_toolset(
        self,
        toolset: ToolSet,
    ) -> ToolSet | None:
        """Wrap the agent's assembled toolset, or return None to leave it unchanged.

        Called per-run with the combined non-output toolset (after the
        prepare_tools hook has already wrapped it). Output tools are added
        separately and are not included.

        When multiple capabilities provide wrappers, they follow middleware
        semantics: the first capability in the list wraps outermost.

        Use this to apply cross-cutting toolset wrappers like
        FilteredToolset, PrefixedToolset, or custom ToolSet subclasses.

        Args:
            toolset: The assembled toolset.

        Returns:
            A wrapped toolset, or None to leave unchanged.
        """
        return None

    # â”€â”€ Convenience methods â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def prefix_tools(self, prefix: str) -> "PrefixedToolSet":
        """Returns a new capability that wraps this one and prefixes its tool names.

        Only this capability's tools are prefixed; other agent tools are unaffected.

        Args:
            prefix: The prefix to add to tool names.

        Returns:
            A new PrefixedToolSet wrapping this capability's tools.
        """
        from .toolset import PrefixedToolSet

        return PrefixedToolSet(self, prefix)

    # â”€â”€ Middleware hooks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def wrap_model_request(
        self,
        ctx: RunContext,
        handler: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Wraps the model request. handler() calls the model.

        Override to inspect or modify the request before execution,
        inspect or modify the response after execution, or short-circuit
        by returning a result without calling handler().

        Args:
            ctx: The run context.
            handler: A callable that executes the model request.

        Returns:
            The model response or a short-circuit result.
        """
        return await handler()

    async def on_model_request_error(
        self,
        ctx: RunContext,
        error: Exception,
    ) -> Any:
        """Called when a model request fails with an exception.

        Override to recover from errors by returning a result,
        or raise the error to propagate it.

        Args:
            ctx: The run context.
            error: The exception that was raised.

        Returns:
            A recovery result, or raise to propagate.
        """
        raise error

    async def wrap_tool_validate(
        self,
        ctx: RunContext,
        tool_name: str,
        tool_args: dict,
        handler: Callable[[str, dict], Awaitable[dict]],
    ) -> dict:
        """Wraps tool argument validation. handler() runs the validation.

        Args:
            ctx: The run context.
            tool_name: The name of the tool.
            tool_args: The raw tool arguments.
            handler: A callable that validates and returns validated args.

        Returns:
            The validated tool arguments.
        """
        return await handler(tool_name, tool_args)

    async def on_tool_validate_error(
        self,
        ctx: RunContext,
        tool_name: str,
        error: Exception,
    ) -> dict:
        """Called when tool argument validation fails.

        Override to recover from validation errors by returning
        default args, or raise the error to propagate it.

        Args:
            ctx: The run context.
            tool_name: The name of the tool.
            error: The validation exception.

        Returns:
            Default validated args, or raise to propagate.
        """
        raise error

    async def wrap_tool_execute(
        self,
        ctx: RunContext,
        tool_name: str,
        tool_args: dict,
        handler: Callable[[str, dict], Awaitable[str]],
    ) -> str:
        """Wraps tool execution. handler() runs the tool.

        Args:
            ctx: The run context.
            tool_name: The name of the tool.
            tool_args: The validated tool arguments.
            handler: A callable that executes the tool.

        Returns:
            The tool result.
        """
        return await handler(tool_name, tool_args)

    async def on_tool_execute_error(
        self,
        ctx: RunContext,
        tool_name: str,
        error: Exception,
    ) -> str:
        """Called when tool execution fails with an exception.

        Override to recover from execution errors by returning
        a result, or raise the error to propagate it.

        Args:
            ctx: The run context.
            tool_name: The name of the tool.
            error: The execution exception.

        Returns:
            A recovery result, or raise to propagate.
        """
        raise error

    async def wrap_output_validate(
        self,
        ctx: RunContext,
        output: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Wraps output validation. handler() performs the validation.

        Args:
            ctx: The run context.
            output: The raw output from the model.
            handler: A callable that validates the output.

        Returns:
            The validated output.
        """
        return await handler(output)

    async def on_output_validate_error(
        self,
        ctx: RunContext,
        output: Any,
        error: Exception,
    ) -> Any:
        """Called when output validation fails.

        Override to recover from validation errors by returning
        a default output, or raise the error to propagate it.

        Args:
            ctx: The run context.
            output: The raw output.
            error: The validation exception.

        Returns:
            A default output, or raise to propagate.
        """
        raise error

    async def wrap_output_process(
        self,
        ctx: RunContext,
        output: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Wraps output processing. handler() processes the output.

        Args:
            ctx: The run context.
            output: The validated output.
            handler: A callable that processes the output.

        Returns:
            The processed output.
        """
        return await handler(output)

    async def on_output_process_error(
        self,
        ctx: RunContext,
        output: Any,
        error: Exception,
    ) -> Any:
        """Called when output processing fails.

        Override to recover from processing errors by returning
        a default output, or raise the error to propagate it.

        Args:
            ctx: The run context.
            output: The validated output.
            error: The processing exception.

        Returns:
            A default output, or raise to propagate.
        """
        raise error

    async def wrap_run_event_stream(
        self,
        ctx: RunContext,
        stream: AsyncIterable,
    ) -> AsyncIterable:
        """Wraps the event stream for streamed runs.

        Args:
            ctx: The run context.
            stream: The event stream to wrap.

        Yields:
            The wrapped event stream events.
        """
        async for event in stream:
            yield event

    # NOTE: a previous "legacy hooks" block re-declared before/after_tool_execute
    # here with no-op bodies, silently overriding the richer modify-args/
    # modify-result variants above. Removed during the Weave port.


class Capability(BaseCapability):
    """Convenience capability for bundling tools + instructions.

    This is a concrete implementation of BaseCapability that allows
    easy creation of capabilities with tools and instructions.

    Attributes:
        id: Unique identifier for the capability.
        description: Human-readable description.
        tools: List of tool definitions.
        instructions: Instructions for the model.
        defer_loading: If True, the capability is loaded on-demand.
    """

    def __init__(
        self,
        id: str,
        description: str,
        tools: list[dict] | None = None,
        instructions: str = "",
        defer_loading: bool = False,
    ) -> None:
        super().__init__(id, description, defer_loading)
        self._tools = tools or []
        # Store instructions as a list for consistent handling by decorator
        if instructions:
            self._instructions: list[str | Callable[..., str]] = [instructions]
        else:
            self._instructions: list[str | Callable[..., str]] = []
        self._tool_plain_registry: list[Callable] = []
        self._tool_registry: list[Callable] = []

    def get_tools(self) -> list[dict]:
        """Return the tools provided by this capability.

        Returns:
            A list of tool definitions.
        """
        return self._tools

    def get_instructions(self) -> str:
        """Return the instructions for this capability.

        Returns:
            A string of instructions.
        """
        if not self._instructions:
            return ""
        parts = []
        for item in self._instructions:
            if callable(item):
                # Call the function to get the string
                try:
                    result = item()
                    if result is not None:
                        parts.append(str(result))
                except TypeError:
                    # Function might need arguments (e.g. RunContext),
                    # try calling with None
                    try:
                        result = item(None)
                        if result is not None:
                            parts.append(str(result))
                    except (TypeError, AttributeError):
                        # Function needs real context, skip in simple contexts
                        pass
            else:
                parts.append(item)
        return "\n".join(parts)

    # â”€â”€ Decorator for plain (no-ctx) function tools â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @overload
    def tool_plain(
        self,
        func: Callable[..., Any],
    ) -> Callable[..., Any]: ...

    @overload
    def tool_plain(
        self,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def tool_plain(
        self,
        func: Callable[..., Any] | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a plain (no-RunContext) function tool on this capability.

        Args:
            func: The function to register as a tool.
            name: Override name for the tool.
            description: Override description for the tool.
            retries: Max number of retries if the tool call fails.
            sequential: Whether this tool must be executed sequentially.
            requires_approval: Whether this tool requires human approval.
            metadata: Additional metadata for the tool.
            timeout: Max execution time in seconds.
            defer_loading: Whether this tool is deferred (hidden until loaded).

        Returns:
            The decorated function, or a decorator if func is None.
        """
        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or f.__name__
            tool_desc = description or (f.__doc__ or "").strip()
            tool_def: dict[str, Any] = {
                "name": tool_name,
                "description": tool_desc,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
            if retries is not None:
                tool_def["retries"] = retries
            if sequential:
                tool_def["sequential"] = True
            if requires_approval:
                tool_def["requires_approval"] = True
            if metadata:
                tool_def["metadata"] = metadata
            if timeout is not None:
                tool_def["timeout"] = timeout
            if defer_loading:
                tool_def["defer_loading"] = True
            self._tools.append(tool_def)
            self._tool_plain_registry.append(f)
            return f

        if func is None:
            return decorator
        return decorator(func)

    # â”€â”€ Decorator for context-aware function tools â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @overload
    def tool(
        self,
        func: Callable[..., Any],
    ) -> Callable[..., Any]: ...

    @overload
    def tool(
        self,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a context-aware function tool on this capability.

        Mirrors Capability.tool_plain but the function should accept RunContext
        as its first argument.

        Args:
            func: The function to register as a tool.
            name: Override name for the tool.
            description: Override description for the tool.
            retries: Max number of retries if the tool call fails.
            sequential: Whether this tool must be executed sequentially.
            requires_approval: Whether this tool requires human approval.
            metadata: Additional metadata for the tool.
            timeout: Max execution time in seconds.
            defer_loading: Whether this tool is deferred (hidden until loaded).

        Returns:
            The decorated function, or a decorator if func is None.
        """
        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or f.__name__
            tool_desc = description or (f.__doc__ or "").strip()
            tool_def: dict[str, Any] = {
                "name": tool_name,
                "description": tool_desc,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
            if retries is not None:
                tool_def["retries"] = retries
            if sequential:
                tool_def["sequential"] = True
            if requires_approval:
                tool_def["requires_approval"] = True
            if metadata:
                tool_def["metadata"] = metadata
            if timeout is not None:
                tool_def["timeout"] = timeout
            if defer_loading:
                tool_def["defer_loading"] = True
            self._tools.append(tool_def)
            self._tool_registry.append(f)
            return f

        if func is None:
            return decorator
        return decorator(func)

    # â”€â”€ Decorator for instructions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @overload
    def instructions(
        self,
        func: Callable[[RunContext], str],
    ) -> Callable[[RunContext], str]: ...

    @overload
    def instructions(
        self,
        func: Callable[[], str],
    ) -> Callable[[], str]: ...

    @overload
    def instructions(
        self,
        /,
    ) -> Callable[[Callable[..., str]], Callable[..., str]]: ...

    def instructions(
        self,
        func: Callable[..., str] | None = None,
        /,
    ) -> Callable[..., str] | Callable[[Callable[..., str]], Callable[..., str]]:
        """Decorator to register an instructions function on this capability.

        The function may take RunContext (or no arguments), may be sync or async,
        and is appended to any instructions provided via the instructions= field.

        Args:
            func: The instructions function to register.

        Returns:
            The decorated function, or a decorator if func is None.
        """
        def decorator(f: Callable[..., str]) -> Callable[..., str]:
            self._instructions.append(f)
            return f

        if func is None:
            return decorator
        self._instructions.append(func)
        return func


class ToolSet(BaseCapability):
    """Capability that wraps a tool set.

    This capability wraps a collection of tools (a "tool set") and
    provides them to the agent.

    Attributes:
        id: Unique identifier for the capability.
        description: Human-readable description.
        tool_set: The tool set to wrap.
        defer_loading: If True, the capability is loaded on-demand.
    """

    def __init__(
        self,
        id: str,
        description: str,
        tool_set: list[dict],
        defer_loading: bool = False,
    ) -> None:
        super().__init__(id, description, defer_loading)
        self._tool_set = tool_set

    def get_tools(self) -> list[dict]:
        """Return the tools from the tool set.

        Returns:
            A list of tool definitions.
        """
        return self._tool_set

    def get_instructions(self) -> str:
        """Return empty instructions (tool sets don't provide instructions).

        Returns:
            An empty string.
        """
        return ""


class HooksCapability(BaseCapability):
    """Capability that provides lifecycle hooks.

    This capability allows registering hooks for before/after model
    requests and tool execution.

    Attributes:
        id: Unique identifier for the capability.
        description: Human-readable description.
        hooks: Dict mapping event names to callback functions.
        defer_loading: If True, the capability is loaded on-demand.
    """

    def __init__(
        self,
        id: str,
        description: str,
        hooks: dict[str, Any] | None = None,
        defer_loading: bool = False,
    ) -> None:
        super().__init__(id, description, defer_loading)
        self._hooks = hooks or {}

    def get_tools(self) -> list[dict]:
        """Return empty tools (hooks don't provide tools).

        Returns:
            An empty list.
        """
        return []

    def get_instructions(self) -> str:
        """Return empty instructions (hooks don't provide instructions).

        Returns:
            An empty string.
        """
        return ""

    def get_hooks(self) -> dict[str, Any]:
        """Return the hooks registered with this capability.

        Returns:
            A dict mapping event names to callbacks.
        """
        return self._hooks


class DeferredCapability(BaseCapability):
    """On-demand capability that is loaded when requested.

    This capability is not loaded until the model explicitly requests
    it via a load_capability tool call. This is useful for reducing
    prompt size by only loading capabilities when needed.

    Attributes:
        id: Unique identifier for the capability.
        description: Human-readable description.
        capability: The underlying capability to load.
    """

    def __init__(
        self,
        id: str,
        description: str,
        capability: BaseCapability,
    ) -> None:
        super().__init__(id, description, defer_loading=True)
        self._capability = capability
        self._loaded = False

    def get_tools(self) -> list[dict]:
        """Return tools from the underlying capability if loaded.

        Returns:
            A list of tool definitions if loaded, empty list otherwise.
        """
        if self._loaded:
            return self._capability.get_tools()
        return []

    def get_instructions(self) -> str:
        """Return instructions from the underlying capability if loaded.

        Returns:
            Instructions string if loaded, empty string otherwise.
        """
        if self._loaded:
            return self._capability.get_instructions()
        return ""

    def get_model_settings(self) -> dict[str, Any]:
        """Return model settings from the underlying capability if loaded.

        Returns:
            Model settings dict if loaded, empty dict otherwise.
        """
        if self._loaded:
            return self._capability.get_model_settings()
        return {}

    def load(self) -> None:
        """Load the underlying capability."""
        self._loaded = True

    def is_loaded(self) -> bool:
        """Check if the capability is loaded.

        Returns:
            True if the capability is loaded.
        """
        return self._loaded
