"""Run context for carrying dependencies through a run.

Provides a typed context object that carries dependencies, usage stats,
and other run-time information through the engine and tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocols import AgentEngine as ChatEngine
    from .usage import UsageStats
    from .usage_limits import UsageLimits


@dataclass
class RunContext:
    """Context carried through a run.

    Attributes:
        engine: The ChatEngine instance.
        deps: Dependency dict (e.g. db_pool, user_session).
        usage: Usage statistics for the run.
        usage_limits: Usage limits for the run.
        model_settings: Model settings for the run.
        run_step: Current step number in the run.
        loaded_capability_ids: IDs of loaded capabilities.
        available_capability_ids: IDs of available capabilities.
        discovered_tool_names: Names of discovered tools.
        available_tool_names: Names of available tools.
        capability_loaded: Whether the current capability is loaded.
    """

    engine: ChatEngine | None = None
    """The ChatEngine instance."""

    deps: dict = field(default_factory=dict)
    """Dependency dict (e.g. db_pool, user_session)."""

    usage: UsageStats | None = None
    """Usage statistics for the run."""

    usage_limits: UsageLimits | None = None
    """Usage limits for the run."""

    model_settings: dict = field(default_factory=dict)
    """Model settings for the run."""

    run_step: int = 0
    """Current step number in the run."""

    loaded_capability_ids: set[str] = field(default_factory=set)
    """IDs of loaded capabilities."""

    available_capability_ids: set[str] = field(default_factory=set)
    """IDs of available capabilities."""

    discovered_tool_names: set[str] = field(default_factory=set)
    """Names of discovered tools."""

    available_tool_names: set[str] = field(default_factory=set)
    """Names of available tools."""

    capability_loaded: bool = False
    """Whether the current capability is loaded."""

    def with_deps(self, **deps) -> RunContext:
        """Create a new RunContext with additional dependencies.

        Args:
            **deps: Additional dependencies.

        Returns:
            A new RunContext with the additional dependencies.
        """
        new_deps = dict(self.deps)
        new_deps.update(deps)
        return RunContext(
            engine=self.engine,
            deps=new_deps,
            usage=self.usage,
            usage_limits=self.usage_limits,
            model_settings=self.model_settings,
            run_step=self.run_step,
            loaded_capability_ids=self.loaded_capability_ids,
            available_capability_ids=self.available_capability_ids,
            discovered_tool_names=self.discovered_tool_names,
            available_tool_names=self.available_tool_names,
            capability_loaded=self.capability_loaded,
        )
