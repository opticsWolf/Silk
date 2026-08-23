"""MCPToolset for local MCP (Model Context Protocol) server connections.

Supports:
- SSE (Server-Sent Events) transport
- Streamable HTTP transport
- StdioTransport (local MCP servers)
- Authorization tokens and custom headers
- Tool discovery and execution via MCP protocol
- Resource management (list, read)
- Prompt management (list, get)
- Server capabilities tracking
- Cache invalidation on list_changed notifications
- Sampling support (MCP sampling protocol)
- Logging support
- Tool error behavior configuration (retry vs error)
- Process tool call hooks
- Structured content handling (JSON parsing)
- BinaryContent handling (images, audio)
- ResourceLink/EmbeddedResource support
- MCP capability wrapper class
- load_mcp_toolsets function for multi-server configs
- Environment variable expansion in configs
- Task-augmented execution (MCP SEP-1686)

Inspired by Pydantic AI's MCP capability architecture.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from abc import ABC
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, overload

try:
    from mcp.client import session as mcp_session
    from mcp.client import sse as mcp_sse
    from mcp.client import streamable_http as mcp_http
    from mcp.types import Tool as MCPTool
    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    mcp_session = mcp_sse = mcp_http = None  # type: ignore[assignment]
    MCPTool = Any  # type: ignore[assignment,misc]
    MCP_AVAILABLE = False

from .toolset import ToolSet


if TYPE_CHECKING:
    from mcp.types import (
        ContentBlock,
        EmbeddedResource,
        ResourceLink,
    )


# ---------------------------------------------------------------------------
# MCPError exception
# ---------------------------------------------------------------------------


class MCPError(RuntimeError):
    """Raised when an MCP server returns an error response.

    This exception wraps error responses from MCP servers, following the ErrorData schema
    from the MCP specification.

    Attributes:
        message: The error message.
        code: The error code returned by the server.
        data: Additional information about the error, if provided by the server.
    """

    def __init__(
        self,
        message: str,
        code: int,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.data = data
        super().__init__(message)

    @classmethod
    def from_mcp_sdk(cls, error: Any) -> MCPError:
        """Create an MCPError from an MCP SDK McpError.

        Args:
            error: An McpError from the MCP SDK.

        Returns:
            An MCPError instance.
        """
        # Extract error data from the McpError.error attribute
        error_data = getattr(error, 'error', None)
        if error_data is None:
            return cls(message=str(error), code=-32600)
        return cls(
            message=getattr(error_data, 'message', str(error)),
            code=getattr(error_data, 'code', -32600),
            data=getattr(error_data, 'data', None),
        )

    def __str__(self) -> str:
        if self.data:
            return f'{self.message} (code: {self.code}, data: {self.data})'
        return f'{self.message} (code: {self.code})'


# ---------------------------------------------------------------------------
# Resource types
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ResourceAnnotations:
    """Additional properties describing MCP entities.

    Attributes:
        audience: Intended audience for this entity.
        priority: Priority level for this entity, ranging from 0.0 to 1.0.
        last_modified: ISO 801 timestamp of the last modification.
    """

    audience: list[str] | None = None
    priority: float | None = None
    last_modified: str | None = None


@dataclass(kw_only=True)
class Icon:
    """An icon for display in user interfaces.

    Attributes:
        src: URL or data URI for the icon.
        mime_type: Optional MIME type for the icon.
        sizes: Optional list of strings specifying icon dimensions.
    """

    src: str
    mime_type: str | None = None
    sizes: list[str] | None = None


@dataclass(kw_only=True)
class BaseResource(ABC):
    """Base class for MCP resources.

    Attributes:
        name: The programmatic name of the resource.
        title: Human-readable title for UI contexts.
        description: A description of what this resource represents.
        mime_type: The MIME type of the resource, if known.
        annotations: Optional annotations for the resource.
        icons: Optional icons for the resource.
        metadata: Optional metadata for the resource.
    """

    name: str
    title: str | None = None
    description: str | None = None
    mime_type: str | None = None
    annotations: ResourceAnnotations | None = None
    icons: list[Icon] | None = None
    metadata: dict[str, Any] | None = None


@dataclass(kw_only=True)
class Resource(BaseResource):
    """A resource that can be read from an MCP server.

    Attributes:
        uri: The URI of the resource.
        size: The size of the raw resource content in bytes, if known.
    """

    uri: str
    size: int | None = None


@dataclass(kw_only=True)
class ResourceTemplate(BaseResource):
    """A template for parameterized resources on an MCP server.

    Attributes:
        uri_template: URI template (RFC 6570) for constructing resource URIs.
    """

    uri_template: str


@dataclass(kw_only=True)
class ResourceLink:
    """A resource link referenced in a prompt or tool call result.

    Unlike EmbeddedResource, this does not include the resource content directly.

    Attributes:
        uri: The URI of the linked resource.
        name: The programmatic name of the linked resource.
        title: Human-readable title for UI contexts.
        description: A description of what this linked resource represents.
        mime_type: The MIME type of the linked resource, if known.
        size: The size of the raw resource content in bytes, if known.
        annotations: Optional annotations for the linked resource.
        icons: Optional icons for the linked resource.
        metadata: Optional metadata for the linked resource.
        type: Discriminator for resource link content.
    """

    uri: str
    name: str
    title: str | None = None
    description: str | None = None
    mime_type: str | None = None
    size: int | None = None
    annotations: ResourceAnnotations | None = None
    icons: list[Icon] | None = None
    metadata: dict[str, Any] | None = None
    type: Literal['resource_link'] = 'resource_link'


@dataclass(kw_only=True)
class EmbeddedResource:
    """A resource embedded into a prompt or tool call result.

    Contains the actual resource content alongside its metadata.

    Attributes:
        uri: The URI of the embedded resource.
        content: The content of the resource.
        type: Discriminator for embedded resource content.
        mime_type: The MIME type of the resource, if known.
        annotations: Optional annotations for the resource.
        metadata: Optional metadata for the resource.
        resource_metadata: _meta carried on the nested resource contents.
    """

    uri: str
    content: str | bytes
    type: Literal['resource'] = 'resource'
    mime_type: str | None = None
    annotations: ResourceAnnotations | None = None
    metadata: dict[str, Any] | None = None
    resource_metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Prompt types
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class PromptArgument:
    """An argument for a prompt template.

    Attributes:
        name: The name of the argument.
        title: Human-readable title for the argument.
        description: A human-readable description of the argument.
        required: Whether the argument is required or optional.
    """

    name: str
    title: str | None = None
    description: str | None = None
    required: bool | None = None


@dataclass(kw_only=True)
class Prompt:
    """A prompt or prompt template that the server offers.

    Attributes:
        name: The programmatic name of the prompt.
        title: Human-readable title for prompt.
        description: An optional description of what this prompt provides.
        arguments: A list of arguments to use for templating the prompt.
        icons: An optional list of icons for this prompt.
        metadata: Optional metadata for the prompt.
    """

    name: str
    title: str | None = None
    description: str | None = None
    arguments: list[PromptArgument] | None = None
    icons: list[Icon] | None = None
    metadata: dict[str, Any] | None = None


PromptRole = Literal['user', 'assistant']


@dataclass(kw_only=True)
class PromptMessage:
    """A message returned as part of a prompt result.

    Attributes:
        role: The role of the message sender.
        content: The content of the message.
    """

    role: PromptRole
    content: ContentBlock


@dataclass(kw_only=True)
class PromptResult:
    """The result of a get_prompt request.

    Attributes:
        messages: The prompt messages.
        description: An optional description for the prompt.
        metadata: Optional metadata for the prompt.
    """

    messages: list[PromptMessage]
    description: str | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Server capabilities
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ServerCapabilities:
    """Capabilities that an MCP server supports.

    Attributes:
        experimental: Experimental, non-standard capabilities that the server supports.
        logging: Whether the server supports sending log messages to the client.
        prompts: Whether the server offers any prompt templates.
        prompts_list_changed: Whether the server will emit notifications when the list of prompts changes.
        resources: Whether the server offers any resources to read.
        resources_list_changed: Whether the server will emit notifications when the list of resources changes.
        tools: Whether the server offers any tools to call.
        tools_list_changed: Whether the server will emit notifications when the list of tools changes.
        completions: Whether the server offers autocompletion suggestions for prompts and resources.
    """

    experimental: list[str] | None = None
    logging: bool = False
    prompts: bool = False
    prompts_list_changed: bool = False
    resources: bool = False
    resources_list_changed: bool = False
    tools: bool = False
    tools_list_changed: bool = False
    completions: bool = False


# ---------------------------------------------------------------------------
# Content block types
# ---------------------------------------------------------------------------

ContentBlock = str | bytes | dict[str, Any] | list[Any]
"""A content block that can be used in prompts and tool results."""


# ---------------------------------------------------------------------------
# Tool result types
# ---------------------------------------------------------------------------

ToolResult = (
    str
    | bytes
    | dict[str, Any]
    | list[Any]
    | Sequence[str | bytes | dict[str, Any] | list[Any]]
)
"""The result type of an MCP tool call."""


# ---------------------------------------------------------------------------
# Process tool callback
# ---------------------------------------------------------------------------

class CallToolFunc(Protocol):
    """A callable that invokes an MCP tool.

    Attributes:
        __call__: Invoke the tool with name, arguments, and optional metadata.
    """

    async def __call__(
        self,
        name: str,
        args: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult: ...


ProcessToolCallback = Callable[
    [
        Any,  # RunContext
        CallToolFunc,
        str,
        dict[str, Any],
    ],
    Awaitable[ToolResult],
]
"""A process tool callback.

It accepts a run context, the original tool call function, a tool name, and arguments.

Allows wrapping an MCP server tool call to customize it, including adding extra request
metadata.
"""


# ---------------------------------------------------------------------------
# Environment variable expansion
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(:-([^}]*))?\}')


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in a JSON structure.

    Environment variables can be referenced using:
    - ${VAR_NAME} syntax â€” expands to the value of VAR_NAME, raises if not defined
    - ${VAR_NAME:-default} syntax â€” expands to VAR_NAME if set, otherwise the default

    Args:
        value: The value to expand (can be str, dict, list, or other JSON types).

    Returns:
        The value with all environment variables expanded.

    Raises:
        ValueError: If an environment variable is not defined and no default value is provided.
    """
    if isinstance(value, str):
        def replace_match(match: re.Match[str]) -> str:
            var_name = match.group(1)
            has_default = match.group(2) is not None
            default_value = match.group(3) if has_default else None

            if var_name in os.environ:
                return os.environ[var_name]
            elif has_default:
                return default_value or ''
            else:
                raise ValueError(f'Environment variable ${{{var_name}}} is not defined')

        return _ENV_VAR_PATTERN.sub(replace_match, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    else:
        return value


# ---------------------------------------------------------------------------
# StdioTransport for local MCP servers
# ---------------------------------------------------------------------------

class StdioTransport:
    """Transport for local MCP servers using stdio (standard I/O).

    This transport connects to local MCP servers by spawning a subprocess
    and communicating via standard input/output streams.

    Attributes:
        command: The command to execute.
        args: Arguments to pass to the command.
        env: Environment variables for the subprocess.
        cwd: Working directory for the subprocess.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd

    async def __aenter__(self) -> StdioTransport:
        """Enter the transport context."""
        import asyncio
        self._process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
            cwd=self.cwd,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the transport context."""
        if hasattr(self, '_process') and self._process:
            self._process.terminate()
            try:
                await self._process.wait(timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()

    def get_streams(self) -> tuple[Any, Any]:
        """Get the read and write streams for the transport.

        Returns:
            A tuple of (read_stream, write_stream).
        """
        if not hasattr(self, '_process') or not self._process:
            raise RuntimeError("Transport not entered. Call __aenter__ first.")
        return self._process.stdin, self._process.stdout


# ---------------------------------------------------------------------------
# MCPToolset
# ---------------------------------------------------------------------------

MCPToolsetClient: TypeAlias = str | Path | StdioTransport | Any
"""Anything MCPToolset accepts as its client argument â€” a URL string, a script Path, StdioTransport, etc."""


@dataclass(kw_only=True)
class MCPToolset(ToolSet):
    """A toolset for connecting to an MCP server.

    MCPToolset is the recommended way to use Model Context Protocol servers in Silk.
    It supports a wide range of transports (HTTP, SSE, stdio, in-process MCP servers)
    and provides comprehensive tool, resource, and prompt management.

    Pass any input that can build a transport from â€” a URL, a script path, a transport
    instance â€” or use the convenience methods for common configurations.

    Attributes:
        client: How to connect to the MCP server (URL, Path, StdioTransport).
        id: Optional unique identifier for this toolset.
        max_retries: Maximum number of times a tool call may be retried after a ModelRetry.
        tool_error_behavior: 'retry' (default) raises ModelRetry on tool errors so the model
            can self-correct; 'error' propagates the underlying exception.
        process_tool_call: Hook to wrap tool calls for custom retry policies or telemetry.
        cache_tools: Whether to cache the list of tools across get_tools() calls.
        cache_resources: Whether to cache the list of resources across list_resources() calls.
        cache_prompts: Whether to cache the list of prompts across list_prompts() calls.
        include_instructions: Whether to include the server's initialize instructions string
            in the agent's instruction set.
        include_return_schema: Whether to include each tool's outputSchema in the schema
            sent to the model.
        sampling_model: A model that the server may sample from via the MCP sampling flow.
        log_level: Log level requested from the server via logging/setLevel after initialization.
        headers: HTTP headers for the connection.
        allowed_tools: Optional list of tool names to include.
        description: Optional description of the MCP server.

    Example â€” connect to a streamable-HTTP MCP server:
        ```python
        from .mcp_toolset import MCPToolset

        toolset = MCPToolset('http://localhost:8000/mcp')
        ```

    Example â€” connect to a local stdio MCP server:
        ```python
        from .mcp_toolset import MCPToolset, StdioTransport

        toolset = MCPToolset(StdioTransport('python', ['my_mcp_server.py']))
        ```
    """

    client: MCPToolsetClient
    transport: str = 'sse'
    _id: str | None = None
    _server_info: Any = None
    _server_capabilities: ServerCapabilities | None = None
    _instructions: str | None = None
    _cached_tools: list[MCPTool] | None = None
    _cached_resources: list[Resource] | None = None
    _cached_prompts: list[Prompt] | None = None
    _running_count: int = 0
    _exit_stack: AsyncExitStack | None = None
    _user_message_handler: Any = None
    _session: Any = None
    _tools: dict[str, dict] | None = None
    max_retries: int | None = None
    tool_error_behavior: Literal['retry', 'error'] = 'retry'
    process_tool_call: ProcessToolCallback | None = None
    cache_tools: bool = True
    cache_resources: bool = True
    cache_prompts: bool = True
    include_instructions: bool = False
    include_return_schema: bool | None = None
    sampling_model: Any = None
    log_level: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    allowed_tools: list[str] | None = None
    description: str | None = None
    _initialized: bool = False

    def __init__(
        self,
        client: MCPToolsetClient,
        *,
        id: str | None = None,
        max_retries: int | None = None,
        tool_error_behavior: Literal['retry', 'error'] = 'retry',
        process_tool_call: ProcessToolCallback | None = None,
        cache_tools: bool = True,
        cache_resources: bool = True,
        cache_prompts: bool = True,
        include_instructions: bool = False,
        include_return_schema: bool | None = None,
        sampling_model: Any = None,
        log_level: Any = None,
        # Backward compatibility with old API
        url: str | None = None,
        transport: str = 'sse',
        headers: dict[str, str] | None = None,
        authorization_token: str | None = None,
        allowed_tools: list[str] | None = None,
        description: str | None = None,
    ) -> None:
        # Support old API: url + transport -> client
        if url is not None:
            self.client = url
            self.transport = transport
        else:
            self.client = client
            self.transport = 'sse'  # default

        self._id = id
        self.max_retries = max_retries
        self.tool_error_behavior = tool_error_behavior
        self.process_tool_call = process_tool_call
        self.cache_tools = cache_tools
        self.cache_resources = cache_resources
        self.cache_prompts = cache_prompts
        self.include_instructions = include_instructions
        self.include_return_schema = include_return_schema
        self.sampling_model = sampling_model
        self.log_level = log_level

        # Backward compatibility
        self.headers = dict(headers) if headers else {}
        self.allowed_tools = allowed_tools
        self.description = description or f"MCP server: {url}" if url else None

        # Merge authorization_token into headers
        if authorization_token:
            self.headers['Authorization'] = authorization_token

        # Initialize other attributes
        self._server_info = None
        self._server_capabilities = None
        self._instructions = None
        self._cached_tools = None
        self._cached_resources = None
        self._cached_prompts = None
        self._tools = None
        self._running_count = 0
        self._exit_stack = None
        self._user_message_handler = None
        self._session = None
        self._initialized = False

    @property
    def id(self) -> str | None:
        return self._id

    @property
    def label(self) -> str:
        if self._id:
            # Include client URL in label when id is provided
            if isinstance(self.client, str):
                return f'{self.__class__.__name__}({self._id!r}, {self.client!r})'
            return f'{self.__class__.__name__}({self._id!r})'
        # Include client URL in label for better debugging
        if isinstance(self.client, str):
            return f'{self.__class__.__name__}({self.client!r})'
        return repr(self)

    @property
    def server_info(self) -> Any:
        """The server-implementation info sent during initialization.

        Raises AttributeError when accessed before the toolset has been entered.
        """
        if self._server_info is None:
            raise AttributeError(f'{self.__class__.__name__}.server_info is only available after initialization.')
        return self._server_info

    @property
    def capabilities(self) -> ServerCapabilities:
        """The capabilities advertised by the server during initialization.

        Raises AttributeError when accessed before the toolset has been entered.
        """
        if self._server_capabilities is None:
            raise AttributeError(f'{self.__class__.__name__}.capabilities is only available after initialization.')
        return self._server_capabilities

    @property
    def instructions(self) -> str | None:
        """The instructions sent by the server during initialization.

        Raises AttributeError when accessed before the toolset has been entered.
        """
        if not hasattr(self, '_initialized') or not self._initialized:
            raise AttributeError(f'{self.__class__.__name__}.instructions is only available after initialization.')
        return self._instructions

    @property
    def is_running(self) -> bool:
        """Whether the toolset is currently entered (the MCP session is open)."""
        return self._running_count > 0

    def _invalidate_tools_cache(self) -> None:
        self._cached_tools = None

    def _invalidate_resources_cache(self) -> None:
        self._cached_resources = None

    def _invalidate_prompts_cache(self) -> None:
        self._cached_prompts = None

    async def __aenter__(self) -> MCPToolset:
        """Enter the toolset context. Connect to the MCP server and initialize."""
        if self._running_count == 0:
            async with AsyncExitStack() as exit_stack:
                await exit_stack.enter_async_context(self)
                # Initialize the session
                if self._session:
                    await self._session.initialize()
                    self._server_info = getattr(self._session, 'server_info', None)
                    self._server_capabilities = self._parse_capabilities()
                    self._instructions = getattr(self._session, 'instructions', None)
                self._exit_stack = exit_stack.pop_all()
                self._initialized = True
        self._running_count += 1
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the toolset context. Disconnect from the MCP server."""
        if self._running_count == 0:
            raise ValueError(f'{self.__class__.__name__}.__aexit__ called more times than __aenter__')
        self._running_count -= 1
        if self._running_count == 0 and self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._server_info = None
            self._server_capabilities = None
            self._instructions = None
            self._cached_tools = None
            self._cached_resources = None
            self._cached_prompts = None
        return None

    def _parse_capabilities(self) -> ServerCapabilities:
        """Parse server capabilities from the initialization result."""
        if not hasattr(self, '_session') or not self._session:
            return ServerCapabilities()

        # Extract capabilities from the session
        init_result = getattr(self._session, 'initialize_result', None)
        if init_result is None:
            return ServerCapabilities()

        caps = getattr(init_result, 'capabilities', None)
        if caps is None:
            return ServerCapabilities()

        return ServerCapabilities(
            experimental=getattr(caps, 'experimental', None),
            logging=getattr(caps, 'logging', False),
            prompts=getattr(caps, 'prompts', False),
            prompts_list_changed=getattr(caps, 'prompts_list_changed', False),
            resources=getattr(caps, 'resources', False),
            resources_list_changed=getattr(caps, 'resources_list_changed', False),
            tools=getattr(caps, 'tools', False),
            tools_list_changed=getattr(caps, 'tools_list_changed', False),
            completions=getattr(caps, 'completions', False),
        )

    async def list_tools(self) -> list[MCPTool]:
        """Retrieve the tools currently exposed by the server.

        When cache_tools is enabled (default), results are cached and invalidated by
        notifications/tools/list_changed or the toolset's last __aexit__.

        Returns:
            A list of MCPTool objects.
        """
        if self.cache_tools and self._cached_tools is not None:
            return self._cached_tools
        async with self:
            if not self._session:
                raise RuntimeError("MCP session not initialized")
            result = await self._session.list_tools()
            if self.cache_tools:
                self._cached_tools = result.tools
            return result.tools

    async def get_tools(self) -> dict[str, dict]:
        """Return the tools discovered from the MCP server.

        Returns:
            A dict mapping tool names to tool definitions.

        Raises:
            RuntimeError: If the toolset is not initialized.
        """
        if not self._initialized or not hasattr(self, '_tools') or self._tools is None:
            raise RuntimeError("MCPToolset not initialized. Call __aenter__ first.")
        return self._tools

    async def list_resources(self) -> list[Resource]:
        """Retrieve the resources currently exposed by the server.

        When cache_resources is enabled (default), results are cached and invalidated by
        notifications/resources/list_changed or the toolset's last __aexit__.

        Returns:
            A list of Resource objects.

        Raises:
            MCPError: If the server returns an error.
        """
        if self.cache_resources and self._cached_resources is not None:
            return self._cached_resources
        async with self:
            if not self._session:
                raise RuntimeError("MCP session not initialized")
            try:
                result = await self._session.list_resources()
                resources = [Resource(uri=r.uri, name=r.name) for r in result.resources]
                if self.cache_resources:
                    self._cached_resources = resources
                return resources
            except Exception as e:
                raise MCPError.from_mcp_sdk(e) from e

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        """Retrieve the resource templates currently exposed by the server.

        Returns:
            A list of ResourceTemplate objects.

        Raises:
            MCPError: If the server returns an error.
        """
        async with self:
            if not self._session:
                raise RuntimeError("MCP session not initialized")
            try:
                result = await self._session.list_resource_templates()
                return [ResourceTemplate(uri_template=t.uriTemplate, name=t.name) for t in result.templates]
            except Exception as e:
                raise MCPError.from_mcp_sdk(e) from e

    @overload
    async def read_resource(self, uri: str) -> str | bytes: ...

    @overload
    async def read_resource(self, uri: Resource) -> str | bytes: ...

    async def read_resource(
        self,
        uri: str | Resource,
    ) -> str | bytes:
        """Read the contents of a specific resource by URI.

        Args:
            uri: The URI of the resource to read, or a Resource object.

        Returns:
            The resource contents â€” a single value if the resource has one content item.
            Text content is returned as str, binary content as bytes.

        Raises:
            MCPError: If the server returns an error.
        """
        resource_uri = uri if isinstance(uri, str) else uri.uri
        async with self:
            if not self._session:
                raise RuntimeError("MCP session not initialized")
            try:
                result = await self._session.read_resource(resource_uri)
                if result and len(result) > 0:
                    content = result[0]
                    if hasattr(content, 'text'):
                        return content.text
                    elif hasattr(content, 'blob'):
                        return base64.b64decode(content.blob)
                    return str(content)
                return ""
            except Exception as e:
                raise MCPError.from_mcp_sdk(e) from e

    async def list_prompts(self) -> list[Prompt]:
        """Retrieve the prompts currently exposed by the server.

        When cache_prompts is enabled (default), results are cached and invalidated by
        notifications/prompts/list_changed or the toolset's last __aexit__.

        Returns:
            A list of Prompt objects.

        Raises:
            MCPError: If the server returns an error.
        """
        if self.cache_prompts and self._cached_prompts is not None:
            return self._cached_prompts
        async with self:
            if not self._session:
                raise RuntimeError("MCP session not initialized")
            try:
                result = await self._session.list_prompts()
                prompts = [Prompt(name=p.name, description=p.description) for p in result.prompts]
                if self.cache_prompts:
                    self._cached_prompts = prompts
                return prompts
            except Exception as e:
                raise MCPError.from_mcp_sdk(e) from e

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> PromptResult:
        """Retrieve a specific prompt from the server, optionally parameterized.

        Args:
            name: The name of the prompt to retrieve.
            arguments: Arguments to parameterize the prompt, if applicable.

        Returns:
            A PromptResult object containing the prompt messages.

        Raises:
            MCPError: If the server doesn't advertise the prompts capability or returns an error.
        """
        async with self:
            if not self._session:
                raise RuntimeError("MCP session not initialized")
            try:
                result = await self._session.get_prompt(name, arguments or {})
                messages = [
                    PromptMessage(
                        role=msg.role,
                        content=getattr(msg, 'content', ''),
                    )
                    for msg in result.messages
                ]
                return PromptResult(
                    messages=messages,
                    description=result.description,
                    metadata=getattr(result, 'metadata', None),
                )
            except Exception as e:
                raise MCPError.from_mcp_sdk(e) from e

    async def direct_call_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        use_task: bool = False,
    ) -> Any:
        """Call a tool on the server directly.

        Args:
            name: The name of the tool to call.
            args: The arguments to pass to the tool.
            metadata: Optional request-level _meta payload sent alongside the call.
            use_task: When True, send the call with task=True per MCP SEP-1686 so
                the server wraps execution in a durable, cancelable, pollable task.

        Returns:
            The tool result.

        Raises:
            ModelRetry: If the tool errors and tool_error_behavior='retry' (the default).
            Exception: If the tool errors and tool_error_behavior='error'.
        """
        async with self:
            if not self._session:
                raise RuntimeError("MCP session not initialized")
            try:
                if use_task:
                    tool_task = await self._session.call_tool(name, args, task=True, meta=metadata)
                    result = await tool_task.result()
                else:
                    result = await self._session.call_tool(name, args, meta=metadata)
            except Exception as e:
                if self.tool_error_behavior == 'retry':
                    # Raise ModelRetry so the model can self-correct
                    from .reflection import ModelRetry
                    raise ModelRetry(message=str(e)) from e
                raise

        # Prefer structured content if all parts are text
        if result and hasattr(result, 'structured_content') and result.structured_content:
            structured = result.structured_content
            if isinstance(structured, dict) and len(structured) == 1 and 'result' in structured:
                return structured['result']
            return structured

        return _map_mcp_tool_results(result.content if hasattr(result, 'content') else [])

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: Any,
        tool: Any,
    ) -> Any:
        """Call a tool on the server.

        Args:
            name: The name of the tool to call.
            tool_args: The arguments to pass to the tool.
            ctx: The run context.
            tool: The tool definition.

        Returns:
            The tool result.
        """
        use_task = bool((getattr(tool, 'tool_def', {}).get('metadata') or {}).get('task'))
        if self.process_tool_call is not None:
            return await self.process_tool_call(
                ctx,
                lambda n, a, **kw: self.direct_call_tool(n, a, **kw),
                name,
                tool_args,
            )
        return await self.direct_call_tool(name, tool_args, use_task=use_task)

    async def get_instructions(self) -> str | None:
        """Return the server's instructions if include_instructions is enabled."""
        if not self.include_instructions:
            return None
        if not self._initialized or self._instructions is None:
            return None
        return self._instructions

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._server_info = None
        self._server_capabilities = None
        self._instructions = None
        self._cached_tools = None
        self._cached_resources = None
        self._cached_prompts = None
        self._tools = None
        self._session = None
        self._initialized = False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _map_mcp_tool_results(parts: Sequence[Any]) -> ToolResult:
    """Map MCP tool result parts to Silk tool results.

    Args:
        parts: The content parts from the MCP tool result.

    Returns:
        The mapped tool result.
    """
    mapped = [_map_mcp_tool_result(part) for part in parts]
    return mapped[0] if len(mapped) == 1 else mapped


def _map_mcp_tool_result(part: Any) -> str | bytes | dict[str, Any] | list[Any]:
    """Map a single MCP tool result part.

    Args:
        part: The content part to map.

    Returns:
        The mapped content.
    """
    if hasattr(part, 'type') and part.type == 'text':
        text = part.text
        if text.startswith(('[', '{')):
            try:
                return json.loads(text)
            except ValueError:
                pass
        return text
    elif hasattr(part, 'type') and part.type == 'image':
        if hasattr(part, 'data'):
            return base64.b64decode(part.data)
        return f"[Image: {getattr(part, 'mime_type', 'unknown')}]"
    elif hasattr(part, 'type') and part.type == 'audio':
        if hasattr(part, 'data'):
            return base64.b64decode(part.data)
        return f"[Audio: {getattr(part, 'mime_type', 'unknown')}]"
    elif hasattr(part, 'type') and part.type == 'resource_link':
        return str(getattr(part, 'uri', ''))
    elif hasattr(part, 'type') and part.type == 'embedded_resource':
        return EmbeddedResource(
            uri=getattr(part, 'uri', ''),
            content=getattr(part, 'content', ''),
            mime_type=getattr(part, 'mime_type', None),
        )
    return str(part)


# ---------------------------------------------------------------------------
# load_mcp_toolsets
# ---------------------------------------------------------------------------


def load_mcp_toolsets(config_path: str | Path) -> list[ToolSet]:
    """Load MCPToolsets from a configuration file.

    The configuration file uses the same mcpServers JSON shape as Claude Desktop,
    Cursor, and the MCP specification. Each server entry produces one MCPToolset,
    wrapped in a PrefixedToolset using the server's name as prefix to disambiguate
    tools across multiple servers.

    Environment variables can be referenced in the configuration file using:
    - ${VAR_NAME} syntax â€” expands to the value of VAR_NAME, raises if not defined
    - ${VAR_NAME:-default} syntax â€” expands to VAR_NAME if set, otherwise the default

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A list of toolsets, one per server in the config file, each prefixed with the server name.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file does not match the schema.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file {config_path} not found')

    config_data = json.loads(config_path.read_text())
    expanded_config_data = _expand_env_vars(config_data)
    if not isinstance(expanded_config_data, dict):
        raise ValueError(f'Expected JSON object at root of {config_path}, got {type(expanded_config_data).__name__}')
    servers = expanded_config_data.get('mcpServers')
    if not isinstance(servers, dict):
        raise ValueError(f'Expected mcpServers object in {config_path}')

    from .toolset import PrefixedToolSet

    toolsets: list[ToolSet] = []
    for name, server in servers.items():
        if 'command' in server:
            transport = StdioTransport(
                command=server['command'],
                args=list(server.get('args') or []),
                env=server.get('env'),
                cwd=str(server['cwd']) if server.get('cwd') is not None else None,
            )
            toolset = MCPToolset(transport, id=name)
        elif 'url' in server:
            toolset = MCPToolset(server['url'], id=name)
        else:
            raise ValueError(f'MCP server config {name!r} must have either command or url')
        toolsets.append(PrefixedToolSet(toolset, f'{name}_'))

    return toolsets
