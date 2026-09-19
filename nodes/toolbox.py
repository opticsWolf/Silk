# -*- coding: utf-8 -*-
"""Silk ToolBox Node.

Assembles a live :class:`ToolBox` — the single registry of ALL tools an
agent network may use — and outputs it on a ``silk_toolbox`` port. Tool
groups are toggled per-node; every file/search tool runs inside a
:class:`FileToolSandbox` spanning the configured sandbox roots.

Sandbox roots are the **hard ceiling** of the whole graph: ToolSets may
narrow the reachable paths (via ``file_permissions``) but can never
escape these roots. They come from the ``sandbox_roots`` wire and from
nowhere else — a Folder node for one root, a Folder List for several
(the ``dirpath`` → ``dirpath_list`` cast wraps the single case). A
ceiling that can also be set on the node is a ceiling you cannot read
off the canvas, which is why there is no picker here. The effective
roots are re-emitted on ``root_paths`` for downstream nodes (e.g. the
Checkable Folder Tree).

Toolchains (python venv, ruff, mypy, radon, maturin, cargo — from
Toolchain nodes) contribute their structured tool packs to the recipe,
and appear in the tree under their own categories (``code``, ``lint``,
``build``) like every other tool.

The node body shows a category-grouped tree of every registered tool,
with a checkbox per tool and a structured detail preview. The group
checkboxes decide what is *attached*; the ticks decide what survives —
a group is a capability, a tick is an instrument. The narrowing is the
last entry of the build recipe, so every derived ToolSet replays it.
"""

from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel, QSpinBox

from weave.widgetcore import WidgetCore, PortRole
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import SILK_TOOLBOX_TYPE  # noqa: F401
from ..functions.mcp_session import MCPBundle, attach_bundle
from ..functions.tool_box import ToolBox
from ..functions.toolset_build import prune_to_selection, tool_catalog
from ..functions.tools.file_sandbox import FileToolSandbox
from ..functions.tools.file_read import attach_file_read_tools
from ..functions.tools.file_write import attach_file_write_tools
from ..functions.tools.file_manipulate import attach_file_manipulate_tools
from ..functions.tools.graph_authoring import attach_graph_tools
from ..functions.tools.suite_tools import attach_suite_tools
from ..functions.import_reach import import_reach_warning
from ..functions.self_modify import user_plugin_root
from ..functions.tools.recall_tool import attach_recall_tool
from ..functions.embeddings import embedder_for
from ..functions.tools.ripgrep_tool import attach_ripgrep_tools
from ..functions.tools.toolchains import attach_toolchain_tools
from ..functions.tools.task_tracker import attach_task_tools
from ..functions.hook_catalog import attach_catalog_hooks, default_hook_config
from ..widgets.hook_select import HookSelectWidget
from ..widgets.node_whitelist import NodeWhitelistWidget
from ..widgets.tool_tree import ToolDetailWidget, ToolTreeWidget

log = get_logger("SilkToolBox")

_ALL_CATEGORIES = "All categories"


@register_node
class SilkToolBoxNode(ActiveNode):
    """Builds a ToolBox with sandboxed tool groups for downstream agents."""
    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk ToolBox"
    node_description: ClassVar[Optional[str]] = (
        "Registry of all agent tools; sandbox roots (wired) as hard "
        "ceiling, toolchain packs, per-tool selection and detail preview."
    )
    node_tags: ClassVar[Optional[List[str]]] = ["silk", "agent", "tools", "sandbox", "llm"]
    node_icon: ClassVar[Optional[str]] = "grid-dots"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 2   # owns a hand-written state dict
    node_version = 2     # bump on any state-shape change (G20)

    def __init__(self, title: str = "Silk ToolBox", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        # Single input for both shapes: a plain dirpath casts into the
        # list (wrapped) via the registered dirpath→dirpath_list cast.
        self.add_input("sandbox_roots", datatype="dirpath_list")
        self.add_input("toolchains", datatype="toolchains")
        # Live MCP sessions, owned by MCP nodes upstream (D19). What
        # arrives is a connection that is already open, so the recipe
        # entry below registers tools without touching a server.
        self.add_input("mcp", datatype="mcp_servers")
        # Which plan the task tools work on (D23). Unwired, the store
        # falls back to the newest plan under the sandbox root -- shared
        # discovery, which is also how two unrelated plans in one root
        # used to find each other.
        self.add_input("plan", datatype="silk_plan")
        # The vector half of memory (§17): an *embedding* model, not the
        # agent's chat model. Unwired, recall is keyword search, which is
        # what it has always been -- so this port adds a capability and
        # never changes one.
        self.add_input("embedding_model", datatype="model_handle")
        self.add_output("toolbox", datatype="silk_toolbox")
        self.add_output("root_paths", datatype="dirpath_list")

        # ── Layout & WidgetCore ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        # ── Widgets ──
        # No root picker: the sandbox roots are the hard ceiling of the
        # whole graph, and a ceiling that can be set in two places is a
        # ceiling nobody can read off the canvas. The wire is the only
        # way in — a Folder node for one root, a Folder List for several.
        self.chk_read = QCheckBox()
        self.chk_read.setChecked(True)
        form.addRow("File Read Tools:", self.chk_read)
        self._widget_core.register_widget(
            "enable_read", self.chk_read, role=PortRole.INTERNAL,
            datatype="bool", default=True, add_to_layout=False,
        )

        self.chk_write = QCheckBox()
        form.addRow("File Write Tools:", self.chk_write)
        self._widget_core.register_widget(
            "enable_write", self.chk_write, role=PortRole.INTERNAL,
            datatype="bool", default=False, add_to_layout=False,
        )

        self.chk_manipulate = QCheckBox()
        form.addRow("File Manage Tools:", self.chk_manipulate)
        self._widget_core.register_widget(
            "enable_manipulate", self.chk_manipulate, role=PortRole.INTERNAL,
            datatype="bool", default=False, add_to_layout=False,
        )

        self.chk_ripgrep = QCheckBox()
        self.chk_ripgrep.setChecked(True)
        form.addRow("Ripgrep Search:", self.chk_ripgrep)
        self._widget_core.register_widget(
            "enable_ripgrep", self.chk_ripgrep, role=PortRole.INTERNAL,
            datatype="bool", default=True, add_to_layout=False,
        )

        self.chk_planning = QCheckBox()
        self.chk_planning.setToolTip(
            "Task planning & tracking tools (plan_start, task_add, task_complete, "
            "task_rescope, …). The plan is stored in the sandbox root. "
            "Add the 'signoff' hook below to require user approval of changes."
        )
        form.addRow("Task Planning:", self.chk_planning)
        self._widget_core.register_widget(
            "enable_planning", self.chk_planning, role=PortRole.INTERNAL,
            datatype="bool", default=False, add_to_layout=False,
        )

        self.chk_recall = QCheckBox()
        self.chk_recall.setToolTip(
            "Memory search (recall): keyword search over turns and runs "
            "remembered in this sandbox root's history ledger — including "
            "ones from earlier sessions and ones compaction dropped. "
            "Needs the 'ledger' extra (macrame-db); without it the tool "
            "says so rather than returning nothing. Wire an embedding "
            "model to the embedding_model port to add vector search."
        )
        form.addRow("Recall (memory):", self.chk_recall)
        self._widget_core.register_widget(
            "enable_recall", self.chk_recall, role=PortRole.INTERNAL,
            datatype="bool", default=False, add_to_layout=False,
        )

        self.spin_read_kib = QSpinBox()
        self.spin_read_kib.setRange(1, 16384)
        self.spin_read_kib.setValue(512)
        form.addRow("Max Read (KiB):", self.spin_read_kib)
        self._widget_core.register_widget(
            "max_read_kib", self.spin_read_kib, role=PortRole.INTERNAL,
            datatype="int", default=512, add_to_layout=False,
        )

        # Graph authoring (§18, D71). The whitelist is the safety
        # property, not the checkbox: an empty list means the tools are
        # mounted and every placement is refused, and there is no
        # "allow all".
        self._node_whitelist = NodeWhitelistWidget()
        form.addRow("Placeable Nodes:", self._node_whitelist)
        self._widget_core.register_widget(
            "placeable_nodes", self._node_whitelist, role=PortRole.INTERNAL,
            datatype="list", default=[], add_to_layout=False,
        )

        # Plugin authoring (§19, D75-D77). The load verb is the only
        # missing step of write-verify-observe-load, and it is the one
        # that crosses an execution boundary the sandbox cannot: every
        # load asks the user, every time, and shows them the diff.
        self.chk_self_modify = QCheckBox()
        self.chk_self_modify.setToolTip(
            "Plugin authoring: lets the agent write node suites into "
            "~/.weave/plugins and load them into this session. Loading "
            "always asks you, shows the diff, and cannot be pre-approved "
            "by a grant or a Role — importing runs the code with the full "
            "authority of this process. Core and Silk stay read-only."
        )
        form.addRow("Plugin authoring:", self.chk_self_modify)
        self._widget_core.register_widget(
            "enable_self_modify", self.chk_self_modify, role=PortRole.INTERNAL,
            datatype="bool", default=False, add_to_layout=False,
        )

        # Infrastructure hooks: part of the recipe, so every derived
        # ToolSet re-creates them — always on, outside any role layer.
        # Value shape: {"names": [...], "configs": {name: {...}}}.
        # Everything in the catalog except the four that can refuse a
        # call or stop to ask (GATING_HOOKS) starts ticked: observation
        # is what people wish they had turned on after a run, and a hook
        # that was never ticked leaves nothing to go back to.
        self._hook_select = HookSelectWidget()
        form.addRow("Hooks:", self._hook_select)
        self._widget_core.register_widget(
            "hooks_config", self._hook_select, role=PortRole.INTERNAL,
            datatype="dict", default=default_hook_config(),
            add_to_layout=False,
        )
        self._hook_select.set_value(default_hook_config())

        # ── Overview: category quick-select + tool tree + detail ──
        self._combo_category = QComboBox()
        self._combo_category.addItem(_ALL_CATEGORIES)
        self._combo_category.setToolTip("Filter the overview to one tool category.")
        form.addRow("Category:", self._combo_category)
        # Register as INTERNAL so the selected category filter survives saves
        self._widget_core.register_widget(
            "category_filter", self._combo_category, role=PortRole.INTERNAL,
            datatype="string", default=_ALL_CATEGORIES, add_to_layout=False,
        )

        # Checkable: the group boxes above decide what gets *attached*,
        # these decide what survives. The two are not redundant -- a
        # group is a capability ("this box can write files"), a tick is
        # an instrument ("but not move_file"). Only tools the tree has
        # actually offered are eligible to be dropped (see _seen_tools),
        # so switching a group on adds its tools live rather than adding
        # them already-excluded.
        self._tool_tree = ToolTreeWidget(checkable=True)
        form.addRow(self._tool_tree)
        self._widget_core.register_widget(
            "enabled_tools", self._tool_tree, role=PortRole.INTERNAL,
            datatype="list", default=[], add_to_layout=False,
        )
        # Every tool the tree has ever shown. The node has to tell "never
        # offered" from "offered and unticked": without it, a tool that
        # appears when a group is switched on would arrive unchecked and
        # be pruned on the same evaluation that created it.
        self._seen_tools: set[str] = set()

        self._detail = ToolDetailWidget()
        form.addRow("Details:", self._detail)
        self._widget_core.register_widget(
            "tool_detail", self._detail, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        # UI-local wiring (no port involvement): filter + detail preview.
        self._combo_category.currentTextChanged.connect(self._on_category_changed)
        self._tool_tree.tool_focused.connect(self._detail.show_tool)

        self._label_status = QLabel("No toolbox built yet.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        # ── Mount ──
        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── State: which tools the tree has already offered ────────────────

    @staticmethod
    def migrate_state(state: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """api 1 → 2: the root picker's history is gone, seen_tools is new.

        ``root_history`` fed a folder dropdown that no longer exists, and
        the root itself is now the ``sandbox_roots`` wire's to supply, so
        a graph saved before this comes back needing that wire. Dropping
        the key is the whole migration: an absent ``seen_tools`` means the
        tree has offered nothing yet, which is exactly true of a graph
        restored into this class, and the first build re-offers every
        tool ticked.
        """
        state.pop("root_history", None)
        return state

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        # Saved because it is the difference between "unticked" and
        # "never shown". Reload without it and every tool looks new, so
        # a deliberately unticked tool would come back ticked.
        state["seen_tools"] = sorted(self._seen_tools)
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        # 1. Restore values silently (prevents eval storms & false undo history)
        with self._widget_core.suppress_signals():
            super().restore_state(state)
        # 2. Restore non-widget internal state
        self._seen_tools = {str(n) for n in (state.get("seen_tools") or ())}

    # ── UI helpers (main thread only) ─────────────────────────────────

    def _on_category_changed(self, text: str) -> None:
        self._tool_tree.set_category_filter("" if text == _ALL_CATEGORIES else text)

    def _adopt_new_tools(self, catalog: List[Dict[str, Any]]) -> None:
        """Tick every tool the tree has not offered before.

        A tool arrives because the user switched a group on, so the
        honest default is *on*: the alternative is a group that appears
        to do nothing until its tools are ticked one by one. Tools
        already seen keep whatever state they were left in, ticked or
        not, which is what makes an untick stick.
        """
        names = {str(entry["name"]) for entry in catalog}
        fresh = names - self._seen_tools
        self._seen_tools |= names
        if not fresh:
            return
        with self._widget_core.suppress_signals():
            self._tool_tree.set_value(
                sorted(set(self._tool_tree.get_value() or ()) | fresh)
            )

    def _refresh_overview(self, catalog: List[Dict[str, Any]]) -> None:
        self._adopt_new_tools(catalog)
        self._tool_tree.set_catalog(catalog)
        current = self._combo_category.currentText()
        self._combo_category.blockSignals(True)
        self._combo_category.clear()
        self._combo_category.addItem(_ALL_CATEGORIES)
        for category in self._tool_tree.categories():
            self._combo_category.addItem(category)
        index = self._combo_category.findText(current)
        self._combo_category.setCurrentIndex(index if index >= 0 else 0)
        self._combo_category.blockSignals(False)
        self._on_category_changed(self._combo_category.currentText())

    # ── Worker thread ─────────────────────────────────────────────────

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # One source, the wire. A dirpath output casts into the list
        # (wrapped) on connection, so a single Folder node and a Folder
        # List node both land here without the node caring which.
        roots = [
            str(p).strip() for p in (inputs.get("sandbox_roots") or [])
            if str(p).strip()
        ]

        if not roots:
            return {"toolbox": None, "root_paths": []}

        # All roots are allowed (the hard ceiling); the first is the
        # working root (cwd for toolchain processes, relative-path base).
        writes = bool(inputs.get("enable_write") or
                      inputs.get("enable_manipulate"))
        # `str | Path` because that is what the sandbox accepts; a
        # `list[str]` is not a `list[str | Path]` to a typechecker, and
        # the widening is free here.
        writable: List[Union[str, Path]] = []
        if inputs.get("enable_self_modify", False):
            # D76: the agent authors plugins in its own root, and that
            # root is the only place the load verb will look. Adding it
            # here rather than asking the user to wire it keeps the two
            # halves of the grant -- write here, load from here -- from
            # drifting apart.
            plugin_root = user_plugin_root()
            try:
                plugin_root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                log.warning(f"could not create the plugin root: {exc}")
            if str(plugin_root) not in roots:
                roots = roots + [str(plugin_root)]
            if not writes:
                # Write access to the plugin root only: plugin authoring
                # is not a reason to make the user's project writable.
                writes = True
                writable = [str(plugin_root)]

        # Every file tool is sandboxed; `import` is not. A writable root
        # that Python will import from is a deferred grant of the whole
        # process's authority, redeemable by anything that imports it --
        # D77 covers the `load_suite` path, and this is the residue it
        # does not: a root chosen inside the venv, inside Weave, or
        # anywhere on sys.path, granted without the file-permissions UI
        # ever suggesting that much (G21). Reported, never refused: the
        # legitimate case is real (D76 plugin authoring writes into an
        # importable tree on purpose).
        self._import_reach = (
            import_reach_warning(writable or roots) if writes else ""
        )
        if self._import_reach:
            log.warning(self._import_reach)

        sandbox = FileToolSandbox(
            root_dir=roots[0],
            allowed_paths=list(roots),
            max_read_bytes=int(inputs.get("max_read_kib", 512)) * 1024,
            write_enabled=writes,
            writable_paths=writable or None,
        )

        # Recipe: which attach groups built this box. ToolSet nodes replay
        # it against their own (ceiling-capped) sandbox to derive subsets.
        recipe: List[Any] = []
        if inputs.get("enable_read", True):
            recipe.append(("file_read", attach_file_read_tools))
        if inputs.get("enable_write", False):
            recipe.append(("file_write", attach_file_write_tools))
        if inputs.get("enable_manipulate", False):
            recipe.append(("file_manipulate", attach_file_manipulate_tools))
        if inputs.get("enable_ripgrep", True):
            recipe.append(("ripgrep", attach_ripgrep_tools))

        # Task planning tools. The optional user sign-off *gate* is the 'signoff'
        # catalog hook (configured in the Hooks list below); attach_catalog_hooks
        # wires it with the store attach_task_tools put on the box, so task_tracker
        # must precede the hooks recipe entry.
        if inputs.get("enable_planning", False):
            plan_ref = inputs.get("plan")
            recipe.append((
                "task_tracker",
                partial(attach_task_tools, plan=plan_ref)
                if plan_ref is not None else attach_task_tools,
            ))

        # Graph authoring. Mounted whenever the user has ticked a class:
        # the whitelist *is* the grant, so an empty one means the pack has
        # nothing to offer and stays out of the prompt entirely (D71).
        placeable = [str(n) for n in (inputs.get("placeable_nodes") or ())
                     if str(n).strip()]
        if placeable:
            recipe.append((
                "graph_authoring",
                partial(attach_graph_tools, whitelist=tuple(placeable)),
            ))

        # Plugin authoring. The user plugin root joins the sandbox as a
        # writable path when this is on: an agent that may load code it
        # wrote needs somewhere to write it, and D76 says that somewhere
        # is ~/.weave/plugins and nowhere else. Weave core, Silk and the
        # virtualenv stay outside every root, so the existing static
        # narrowing is what keeps them read-only -- no new mechanism.
        if inputs.get("enable_self_modify", False):
            recipe.append(("suite_tools", attach_suite_tools))

        # Memory. Attached after the task tools so a box that has both
        # reads as plan-then-memory in the tool list, which is the order
        # an agent uses them in.
        if inputs.get("enable_recall", False):
            # An embedding model turns recall into hybrid search; without
            # one this is the same keyword attach as before. The embedder
            # is built per compute so a re-run picks up a model that was
            # wired since, and it is shared by every root the box reads:
            # two roots indexed by two models would be two incomparable
            # rankings merged into one list.
            embedder = embedder_for(inputs.get("embedding_model"))
            recipe.append((
                "recall", partial(attach_recall_tool, embedder=embedder),
            ))

        toolchains = tuple(inputs.get("toolchains") or ())
        if toolchains:
            recipe.append((
                "toolchains",
                partial(attach_toolchain_tools, toolchains=toolchains),
            ))

        mcp = inputs.get("mcp")
        if mcp is not None and MCPBundle.coerce(mcp).enabled_sessions():
            # Deliberately not `add_toolset`: an external ToolSet is
            # entered and exited around every dispatched batch, which for
            # MCP means a handshake per batch per agent. The recipe entry
            # closes over the open sessions instead (D19).
            recipe.append(("mcp", attach_bundle(mcp)))

        hooks_config = inputs.get("hooks_config") or {}
        hook_names = tuple(str(n) for n in (hooks_config.get("names") or ()))
        if hook_names:
            recipe.append((
                "hooks",
                partial(
                    attach_catalog_hooks,
                    names=hook_names,
                    configs=dict(hooks_config.get("configs") or {}),
                ),
            ))

        # Last, after everything that registers: the per-tool narrowing.
        # In the recipe rather than applied here, so a ToolSet replaying
        # this recipe narrows the same way -- otherwise a derived set
        # would come back holding tools this box was told to drop.
        offered = frozenset(str(n) for n in (self._seen_tools or ()))
        if offered:
            keep = frozenset(
                str(n) for n in (inputs.get("enabled_tools") or ())
            )
            recipe.append((
                "tool_selection",
                partial(prune_to_selection, keep=keep, offered=offered),
            ))

        toolbox = ToolBox()
        for source_name, attacher in recipe:
            with toolbox._attributing_to(source_name):
                attacher(toolbox, sandbox)
        toolbox.build_recipe = tuple(recipe)  # type: ignore[attr-defined]
        toolbox.base_sandbox = sandbox  # type: ignore[attr-defined]

        return {"toolbox": toolbox, "root_paths": list(roots)}

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        toolbox = self._get_cached_value("toolbox")
        if toolbox is None:
            self._refresh_overview([])
            self._widget_core.push_display(
                "status",
                "Connect a folder (or a folder list) to sandbox_roots to build "
                "the toolbox.",
            )
        else:
            catalog = tool_catalog(toolbox)
            self._refresh_overview(catalog)
            roots = self._get_cached_value("root_paths") or []
            categories = ", ".join(self._tool_tree.categories())
            reach = getattr(self, "_import_reach", "")
            self._widget_core.push_display(
                "status",
                f"{len(catalog)} tools in {categories or 'no categories'} · "
                f"{len(roots)} sandbox root(s)."
                # The warning goes where the root count already is: this
                # is a property of the roots the user just chose, and a
                # log line alone is a warning nobody reads (G21).
                + ("\n! " + reach if reach else ""),
            )
