"""
Directory-based tool discovery.

A *plugin* is any ``*.py`` file in the tools directory (names starting with
``__`` are skipped) that exposes one or more module-level ``attach_*`` callables.
Each ``attach_*`` takes the ToolBox as its first argument and may declare further
parameters (``sandbox``, an http client, â€¦); the loader injects those **by name**
from a context dict, so there is no positional-count heuristic and no per-tool
import wired into the harness.

Two entry points:

  * :meth:`ToolLoader.sync`     â€” incremental load/refresh into a *live* ToolBox.
    New files are attached, changed files are pruned-and-reattached, deleted
    files are pruned. This is what powers add/refresh while the harness runs.
  * :meth:`ToolLoader.discover` â€” stateless enumeration into throwaway ToolBoxes,
    for a UI that wants to list available tools without committing them.

Modules that expose no ``attach_*`` (library modules like ``file_sandbox`` or
``command_advice``) and modules that fail to import (e.g. a GUI module pulled in
by mistake) are skipped and reported, never fatal.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..tool_box import ToolBox


# â”€â”€ Dependency injection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def inject(attacher: Callable, toolbox: "ToolBox", context: dict[str, Any]) -> None:
    """
    Call *attacher* with *toolbox* as the first positional argument, supplying
    any further parameters by name from *context*. A required parameter that is
    absent from the context raises ``TypeError`` (surfaced as a per-plugin error
    by the loader, not a crash).
    """
    params = list(inspect.signature(attacher).parameters.values())
    kwargs: dict[str, Any] = {}
    for p in params[1:]:  # params[0] is the toolbox
        if p.name in context:
            kwargs[p.name] = context[p.name]
        elif p.default is inspect.Parameter.empty and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(
                f"attacher '{attacher.__name__}' requires '{p.name}', "
                f"which was not provided in the plugin context "
                f"({', '.join(context) or 'empty'})."
            )
    attacher(toolbox, **kwargs)


# â”€â”€ Reports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class LoadReport:
    """Summary of a :meth:`ToolLoader.sync` pass."""
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # source -> message

    def __bool__(self) -> bool:
        return bool(self.added or self.updated or self.removed or self.errors)


@dataclass
class DiscoveredModule:
    """One plugin module enumerated by :meth:`ToolLoader.discover`."""
    source: str
    path: Path
    attachers: list[Callable]
    tools: dict[str, dict]            # tool_name -> registered metadata (dry-run)
    error: str | None = None


# â”€â”€ Loader â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ToolLoader:
    """Scans a directory for tool plugins and (re)attaches them to a ToolBox."""

    def __init__(self, directory: "str | Path"):
        self.directory = Path(directory).resolve()
        # mtime (ns) of every file we have examined, so unchanged files are
        # skipped on the next sync. Includes no-attacher / broken files.
        self._mtimes: dict[str, int] = {}
        # sources that successfully registered at least one tool.
        self._attached: set[str] = set()

    # -- internals ----------------------------------------------------------

    def _plugin_files(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(
            p for p in self.directory.glob("*.py")
            if not p.name.startswith("__")
        )

    @staticmethod
    def _import_module(source: str, path: Path):
        """(Re)exec *path* as a fresh module object and return it."""
        mod_name = f"dynamic_tools.{source}"
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module          # so dataclasses/typing resolve
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _attachers(module) -> list[Callable]:
        return [
            obj for name, obj in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("attach_")
        ]

    # -- live load / refresh ------------------------------------------------

    def sync(self, toolbox: "ToolBox", context: dict[str, Any]) -> LoadReport:
        """
        Reconcile *toolbox* with the current directory contents. Idempotent:
        unchanged files are untouched, changed files are pruned and reattached,
        deleted files are pruned.
        """
        report = LoadReport()
        present: set[str] = set()

        for path in self._plugin_files():
            source = path.stem
            present.add(source)
            mtime = path.stat().st_mtime_ns
            if self._mtimes.get(source) == mtime:
                continue  # unchanged since last scan

            is_update = source in self._attached
            if is_update:
                toolbox.prune_source(source)
                self._attached.discard(source)

            try:
                module = self._import_module(source, path)
                attachers = self._attachers(module)
                if not attachers:
                    # Library module sharing the directory â€” not a plugin.
                    self._mtimes[source] = mtime
                    continue
                for fn in attachers:
                    toolbox.attach_plugin(source, fn, **context)
            except Exception as e:  # noqa: BLE001 - one bad plugin must not break the rest
                toolbox.prune_source(source)
                report.errors[source] = f"{type(e).__name__}: {e}"
                self._mtimes[source] = mtime
                continue

            self._mtimes[source] = mtime
            self._attached.add(source)
            (report.updated if is_update else report.added).append(source)

        # Files that disappeared since the last scan.
        for source in sorted(set(self._mtimes) - present):
            if source in self._attached:
                toolbox.prune_source(source)
                self._attached.discard(source)
                report.removed.append(source)
            self._mtimes.pop(source, None)

        return report

    # -- stateless enumeration ----------------------------------------------

    def discover(self, context: dict[str, Any]) -> list[DiscoveredModule]:
        """
        Import every plugin and dry-run its attachers against throwaway
        ToolBoxes to learn which tools they register â€” without mutating any live
        box. Useful for a UI that lists/toggles tools before a session starts.
        """
        from ..tool_box import ToolBox  # local import avoids an import cycle

        modules: list[DiscoveredModule] = []
        for path in self._plugin_files():
            source = path.stem
            try:
                module = self._import_module(source, path)
            except Exception as e:  # noqa: BLE001
                modules.append(DiscoveredModule(source, path, [], {}, error=f"{type(e).__name__}: {e}"))
                continue

            attachers = self._attachers(module)
            if not attachers:
                continue  # library module, not a plugin

            scratch = ToolBox(None, None)
            error = None
            try:
                for fn in attachers:
                    scratch.attach_plugin(source, fn, **context)
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
            modules.append(
                DiscoveredModule(source, path, attachers, dict(scratch.tools), error=error)
            )
        return modules
