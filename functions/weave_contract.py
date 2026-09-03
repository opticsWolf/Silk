# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

What Silk needs from Weave, written down (G20, D83).

Silk reaches into Weave internals that are stable by convention rather
than by declaration -- `PortRegistry._by_name` for port registration, the
node registry and its metadata for graph authoring, the undo command map,
the panel mirror contract, `emit_stream` / `pulse`. None of it is
versioned. The failure that gap describes is not that a refactor breaks
Silk; it is that a refactor breaks Silk **silently**, as an
`AttributeError` from inside a run, three layers from the thing that
moved.

This module is the declared floor. It does not stop a refactor and it
does not pretend to be a version number: it is the *list*, checked at
load, so a moved internal is reported by name, next to the reason Silk
wanted it, before anything tries to use it.

Two rules the entries follow:

- **Only what Silk actually touches.** A list that drifts from the code
  is worse than none, because it reports the wrong things confidently.
  Every entry here has a caller.
- **A finding, never an exception.** A missing internal is reported and
  the plugin still loads: the nodes that do not use it still work, and a
  user who can see "Weave moved `PortRegistry._by_name`" can act, where a
  plugin that refuses to import tells them only that Silk is broken.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from weave.logger import get_logger

log = get_logger("SilkWeaveContract")

__all__ = [
    "REQUIREMENTS",
    "Requirement",
    "check_contract",
    "contract_report",
    "log_contract",
]

#: Shapes an entry may require. Deliberately coarse -- this checks that
#: the thing is still *there and still that kind of thing*, not that its
#: signature is unchanged, which is a promise no attribute check can keep.
KIND_ANY = "any"
KIND_CALLABLE = "callable"
KIND_MAPPING = "mapping"
KIND_CLASS = "class"


@dataclass(frozen=True)
class Requirement:
    """One thing Silk needs from Weave, and why.

    *attr* may be dotted (``PortRegistry._by_name``), which is how the
    private reaches -- the ones with no public promise at all -- are
    named.
    """

    module: str
    attr: str
    why: str
    kind: str = KIND_ANY

    @property
    def path(self) -> str:
        return f"{self.module}.{self.attr}"


#: Everything Silk reaches for, grouped by the seam it belongs to. Keep
#: this in step with the imports: an entry with no caller should go.
REQUIREMENTS: tuple[Requirement, ...] = (
    # -- ports: the private reach G20 names first ------------------------
    Requirement("weave.node.port_registry", "PortRegistry",
                "the port type registry", KIND_CLASS),
    Requirement("weave.node.port_registry", "PortRegistry._by_name",
                "nodes/silk_ports.py registers Silk's port types by "
                "checking this dict first -- there is no public "
                "'is this port type registered' call", KIND_MAPPING),
    Requirement("weave.node.port_registry", "PortRegistry._cast_registry",
                "the same file registers the dirpath -> dirpath_list cast",
                KIND_MAPPING),
    Requirement("weave.node.port_utils", "ConnectionFactory",
                "graph authoring builds connections through it"),
    Requirement("weave.node.port_utils", "PortUtils",
                "port lookup for place/connect (D69)"),

    # -- the node registry: graph authoring (§18) ------------------------
    Requirement("weave.registry", "register_node",
                "every Silk node is registered through it", KIND_CALLABLE),
    Requirement("weave.registry", "NODE_REGISTRY",
                "list_placeable_nodes reads the registry to tell the model "
                "what exists (D69)"),
    Requirement("weave.registry", "NODE_REGISTRY.get_node_class",
                "the graph canvas resolves a class name to place it",
                KIND_CALLABLE),
    Requirement("weave.registry", "NODE_REGISTRY.get_all_nodes",
                "the whitelist editor lists what an agent could be allowed "
                "to place (D71)", KIND_CALLABLE),
    Requirement("weave.registry", "NODE_REGISTRY.add_listener",
                "the whitelist editor follows a suite being loaded mid-session",
                KIND_CALLABLE),
    Requirement("weave.registry.metadata", "get_display_name",
                "the descriptions the model reads are the node UI's own",
                KIND_CALLABLE),
    Requirement("weave.registry.metadata", "get_description", "as above",
                KIND_CALLABLE),
    Requirement("weave.registry.metadata", "get_tags", "as above",
                KIND_CALLABLE),
    Requirement("weave.canvas.undo_manager", "default_registry_map",
                "D72: an agent's graph edits go through the undo stack, so "
                "the user can undo them like their own"),

    # -- node base classes and the stream port ---------------------------
    Requirement("weave.node.base", "ActiveNode", "base of the Silk nodes",
                KIND_CLASS),
    Requirement("weave.node.base", "ActiveNode.emit_stream",
                "the one typed event port (D1) is an emit_stream port",
                KIND_CALLABLE),
    Requirement("weave.node.base", "ActiveNode.pulse",
                "downstream evaluation after a run", KIND_CALLABLE),
    Requirement("weave.node.threaded", "ThreadedNode",
                "runs off the GUI thread", KIND_CLASS),
    Requirement("weave.node.threaded", "ThreadedManualNode",
                "the Agent node's base: manual execute + worker thread",
                KIND_CLASS),
    Requirement("weave.node", "VerticalSizePolicy", "node layout"),

    # -- widgets and mirrors ---------------------------------------------
    Requirement("weave.widgetcore", "WidgetCore",
                "every Silk node's widget surface", KIND_CLASS),
    Requirement("weave.widgetcore", "PortRole", "port declaration"),
    Requirement("weave.widgetcore.binding_policy", "debounced",
                "editor bindings", KIND_CALLABLE),
    Requirement("weave.panel.mirror_contracts", "MirrorContract",
                "D59: the Decision Inbox and the pool monitor are panel "
                "mirrors of node state", KIND_CLASS),

    # -- engine services --------------------------------------------------
    Requirement("weave.engine.shutdown", "get_shutdown_registry",
                "the model pool and MCP sessions are closed on shutdown",
                KIND_CALLABLE),
    Requirement("weave.engine.suite_loader", "user_plugin_dir",
                "§19: where a machine-authored suite may be written",
                KIND_CALLABLE),
    Requirement("weave.engine.validation", "validate_suite",
                "D78: the linter's verdict lands before the human is asked "
                "to approve a load", KIND_CALLABLE),
    Requirement("weave.engine.migration", "resolve_node",
                "G20's second half: Weave resolves a saved node against the "
                "running class, which is what Silk's node_version and "
                "node_state_api feed", KIND_CALLABLE),
)


def _resolve(req: Requirement) -> tuple[bool, Optional[Any], str]:
    """Look one requirement up. Returns (found, value, note)."""
    try:
        obj: Any = importlib.import_module(req.module)
    except Exception as exc:      # noqa: BLE001 -- any import failure is a finding
        return False, None, f"module import failed: {exc}"
    for part in req.attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return False, None, "attribute is gone"
        except Exception as exc:  # noqa: BLE001 -- a property that raises
            return False, None, f"attribute raised: {exc}"
    return True, obj, ""


def _shape_ok(value: Any, kind: str) -> bool:
    if kind == KIND_CALLABLE:
        return callable(value)
    if kind == KIND_MAPPING:
        return hasattr(value, "get") and hasattr(value, "__contains__")
    if kind == KIND_CLASS:
        return isinstance(value, type)
    return True


def check_contract(
    requirements: Iterable[Requirement] = REQUIREMENTS,
    *,
    resolve: Optional[Callable[[Requirement], tuple[bool, Any, str]]] = None,
) -> list[str]:
    """Every requirement this Weave does not satisfy, as sentences.

    Total: an entry whose module will not import is a finding like any
    other, so one broken seam cannot hide the rest.
    """
    finder = resolve or _resolve
    findings: list[str] = []
    for req in requirements:
        found, value, note = finder(req)
        if not found:
            findings.append(f"{req.path} -- {note}. Silk needs it: {req.why}.")
            continue
        if not _shape_ok(value, req.kind):
            findings.append(
                f"{req.path} is no longer a {req.kind} "
                f"(it is {type(value).__name__}). Silk needs it: {req.why}."
            )
    return findings


def contract_report(requirements: Iterable[Requirement] = REQUIREMENTS) -> str:
    """One line for the load log; empty when everything is there."""
    findings = check_contract(requirements)
    if not findings:
        return ""
    head = (f"Silk depends on {len(findings)} Weave internal(s) that this "
            f"Weave does not provide:")
    return "\n".join([head, *(f"  - {f}" for f in findings)])


def log_contract(requirements: Iterable[Requirement] = REQUIREMENTS) -> str:
    """Check at load and log what is missing. Returns the report."""
    report = contract_report(requirements)
    if report:
        log.warning(report)
    return report
