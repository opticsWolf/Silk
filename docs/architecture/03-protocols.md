## The two protocol contracts

`functions/protocols.py` pins what were previously duck-typed seams into two
`@runtime_checkable` `Protocol`s. The `AgentLoop` binds to these, not to
concrete classes — so tests substitute fakes and alternative engines (remote
APIs, mock models) drop in without touching the loop.

```python
@runtime_checkable
class AgentEngine(Protocol):
    """One model request per stream_response() call. Owns the history.
    Never executes tools, never loops — multi-turn belongs to the AgentLoop."""
    usage_limits: Any
    reflection_config: Any
    history: list[dict[str, Any]]
    last_stats: dict[str, Any]
    def stream_response(self, gen_params: dict[str, Any]) -> Iterator[str]: ...
    def append_message(self, role: str, content: str, **stats: Any) -> None: ...
    def count_prompt_tokens(self) -> int: ...
    def request_stop(self) -> None: ...
    def stop_requested(self) -> bool: ...

@runtime_checkable
class ToolRegistry(Protocol):
    """What the loop needs from a tool registry (ToolBox satisfies it)."""
    tools: dict[str, dict[str, Any]]
    async def execute_tool_calls_async(self, tool_calls: list[Any]) -> list[dict]: ...
```

`GraphEngine` (see [Model layer](06-model-layer.md#model-layer)) is the production
`AgentEngine`; `ToolBox` is the production `ToolRegistry`.

