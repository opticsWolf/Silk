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

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QSpinBox,
    QVBoxLayout,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import SILK_TOOLBOX_TYPE  # noqa: F401
from ..functions.mcp_session import MCPBundle, attach_bundle
from ..functions.tool_box import ToolBox
from ..functions.toolset_build import prune_to_selection, tool_catalog
from ..functions.tool_groups import (
    GROUP_ATTACHERS,
    GROUP_CONFIG_LABELS,
    WRITING_GROUPS,
    categories_of,
    coerce_paths,
    configurable_categories,
    default_selection,
    groups_for,
    static_catalog,
)
from ..functions.tools.file_sandbox import FileToolSandbox
from ..functions.tools.graph_authoring import attach_graph_tools
from ..functions.import_reach import import_reach_warning
from ..functions.self_modify import user_plugin_root
from ..functions.tools.recall_tool import attach_recall_tool
from ..functions.embeddings import embedder_for
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
    node_state_api = 3   # owns a hand-written state dict
    node_version = 3     # bump on any state-shape change (G20)

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
        self.spin_read_kib = QSpinBox()
        self.spin_read_kib.setRange(1, 16384)
        self.spin_read_kib.setValue(512)
        form.addRow("Max Read (KiB):", self.spin_read_kib)
        self._widget_core.register_widget(
            "max_read_kib", self.spin_read_kib, role=PortRole.INTERNAL,
            datatype="int", default=512, add_to_layout=False,
        )

        # Graph authoring's whitelist (§18, D71). It has no row of its
        # own any more: it is not a preference sitting beside the tools,
        # it is the grant those tools run under, so it lives behind the
        # gear on the ``graph`` category row in the tree below. The
        # widget is still registered — it is where the value is kept and
        # how it saves — it is simply shown in a dialog instead.
        self._node_whitelist = NodeWhitelistWidget()
        self._widget_core.register_widget(
            "placeable_nodes", self._node_whitelist, role=PortRole.INTERNAL,
            datatype="list", default=[], add_to_layout=False,
        )
        self._config_dialogs: Dict[str, QDialog] = {}

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

        # The only tool selector there is. There used to be a row of
        # group checkboxes above it, and they asked the same question
        # twice: a group was on and a tool was ticked, and the two could
        # disagree. Now a group is attached because something in it is
        # ticked (see ``groups_for``), and nothing else decides.
        #
        # Seeded from the static catalog rather than from a built box,
        # so the tree has something to tick before a root is wired --
        # the root is usually the thing being wired *because* tools are
        # wanted, and a tree that fills in afterwards is a tree that
        # cannot be set up first.
        self._tool_tree = ToolTreeWidget(checkable=True)
        catalog = static_catalog()
        self._tool_tree.set_catalog(catalog)
        self._tool_tree.set_configurable(configurable_categories())
        self._tool_tree.config_requested.connect(self._open_group_config)
        form.addRow(self._tool_tree)
        self._widget_core.register_widget(
            "enabled_tools", self._tool_tree, role=PortRole.INTERNAL,
            datatype="list", default=default_selection(), add_to_layout=False,
        )
        self._tool_tree.set_value(default_selection())
        # Every tool the tree has ever shown. The node has to tell "never
        # offered" from "offered and unticked": without it, a tool that
        # appears when a toolchain is wired would arrive unchecked and be
        # pruned on the same evaluation that created it. The static tools
        # have just been offered, so they start here.
        self._seen_tools: set[str] = {str(e["name"]) for e in catalog}

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

    #: api 2 → 3: the group checkboxes whose job the tool ticks took over.
    _RETIRED_GROUP_KEYS = (
        "enable_read", "enable_write", "enable_manipulate", "enable_ripgrep",
        "enable_planning", "enable_recall", "enable_self_modify",
    )

    @staticmethod
    def migrate_state(state: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """api 1 → 3.

        1 → 2: the root picker's history is gone, seen_tools is new.
        2 → 3: the group checkboxes are gone. Their values are dropped
        rather than translated into ticks, because a group that was on
        did not mean every tool in it was wanted -- and the tools that
        *were* ticked are already in ``enabled_tools``, which survives
        untouched. A graph saved with a group on and nothing ticked in
        it comes back with that group off, which is the honest reading
        of a selection that names no tool.

        ``root_history`` fed a folder dropdown that no longer exists, and
        the root itself is now the ``sandbox_roots`` wire's to supply, so
        a graph saved before this comes back needing that wire. Dropping
        the key is the whole migration: an absent ``seen_tools`` means the
        tree has offered nothing yet, which is exactly true of a graph
        restored into this class, and the first build re-offers every
        tool ticked.
        """
        state.pop("root_history", None)
        values = state.get("widget_data")
        if isinstance(values, dict):
            for key in SilkToolBoxNode._RETIRED_GROUP_KEYS:
                values.pop(key, None)
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

    def _open_group_config(self, category: str) -> None:
        """Open the settings behind a category row's gear.

        Only graph authoring has any, for now. Its whitelist is modal on
        purpose: it is a grant, and a grant edited in a panel that can
        drift out of view is a grant nobody re-reads.
        """
        widget = self._config_widget(category)
        if widget is None:
            return
        dialog = self._config_dialogs.get(category)
        if dialog is None:
            dialog = QDialog()
            dialog.setWindowTitle(self._config_title(category))
            layout = QVBoxLayout(dialog)
            layout.addWidget(widget)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(dialog.reject)
            buttons.accepted.connect(dialog.accept)
            layout.addWidget(buttons)
            dialog.resize(420, 480)
            self._config_dialogs[category] = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _config_widget(self, category: str) -> Optional[Any]:
        """The widget that configures *category*, if there is one."""
        if category in categories_of("graph_authoring"):
            return self._node_whitelist
        return None

    @staticmethod
    def _config_title(category: str) -> str:
        for group, label in GROUP_CONFIG_LABELS.items():
            if category in categories_of(group):
                return label.rstrip("…").strip()
        return category

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

    def _available_catalog(self) -> List[Dict[str, Any]]:
        """Every tool the user could tick, whether it is attached or not.

        The static half is always there, so a group with nothing ticked
        keeps its row and can be switched back on -- untick the last
        tool of a group and the group itself would otherwise disappear,
        taking the only way to get it back. The dynamic half (toolchain
        packs, MCP servers) is whatever the last build actually offered,
        because nothing can predict it: those tools exist only while
        their toolchain or server is wired.
        """
        catalog = static_catalog()
        known = {str(e["name"]) for e in catalog}
        for entry in getattr(self, "_offered_catalog", ()) or ():
            if str(entry["name"]) not in known:
                catalog.append(entry)
        return catalog

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
        # List node both land here without the node caring which -- but
        # the cast is consulted when the wire is drawn and not when the
        # value travels, so what actually arrives from a Folder node is
        # a bare Path. coerce_paths takes either.
        roots = coerce_paths(inputs.get("sandbox_roots"))

        if not roots:
            return {"toolbox": None, "root_paths": []}

        # The selection is the only switch. A group is attached because
        # the user ticked something in it; nothing else votes.
        selection = frozenset(
            str(n) for n in (inputs.get("enabled_tools") or ()) if str(n).strip()
        )
        groups = groups_for(selection)

        # All roots are allowed (the hard ceiling); the first is the
        # working root (cwd for toolchain processes, relative-path base).
        # Write access is derived, not declared: ticking write_file *is*
        # the request for it, and there is no second place to ask.
        writes = bool(WRITING_GROUPS & set(groups))
        authoring = "suite_tools" in groups
        # `str | Path` because that is what the sandbox accepts; a
        # `list[str]` is not a `list[str | Path]` to a typechecker, and
        # the widening is free here.
        writable: List[Union[str, Path]] = []
        if authoring:
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

        # Graph authoring's whitelist *is* the grant (D71): an empty one
        # means the pack has nothing to offer, so it stays out of the
        # prompt entirely rather than mounting eight tools that refuse
        # everything. Ticking the tools with no classes allowed is a
        # half-finished setup, and the status line says so.
        placeable = [str(n) for n in (inputs.get("placeable_nodes") or ())
                     if str(n).strip()]
        self._graph_hint = (
            "graph tools are ticked but no node class is allowed - open "
            "the group's settings in the tree to choose some"
            if "graph_authoring" in groups and not placeable else ""
        )

        # Recipe: which attach groups built this box. ToolSet nodes replay
        # it against their own (ceiling-capped) sandbox to derive subsets.
        # Order is GROUP_ATTACHERS' order: read before write, plan before
        # memory, which is the order an agent reads its tool list in.
        recipe: List[Any] = []
        for group in groups:
            attacher: Any
            if group == "task_tracker":
                # The optional user sign-off *gate* is the 'signoff'
                # catalog hook; attach_catalog_hooks wires it with the
                # store attach_task_tools puts on the box, so this must
                # precede the hooks entry -- it does, hooks come last.
                plan_ref = inputs.get("plan")
                attacher = (partial(attach_task_tools, plan=plan_ref)
                            if plan_ref is not None else attach_task_tools)
            elif group == "graph_authoring":
                if not placeable:
                    continue
                attacher = partial(
                    attach_graph_tools, whitelist=tuple(placeable),
                )
            elif group == "recall":
                # An embedding model turns recall into hybrid search;
                # without one this is keyword search, as it has always
                # been. Built per compute so a re-run picks up a model
                # wired since, and shared by every root the box reads:
                # two roots indexed by two models would be two
                # incomparable rankings merged into one list.
                attacher = partial(
                    attach_recall_tool,
                    embedder=embedder_for(inputs.get("embedding_model")),
                )
            else:
                attacher = GROUP_ATTACHERS[group]
            recipe.append((group, attacher))

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
            if source_name != "tool_selection":
                # What the tree should *list*, taken one step before the
                # narrowing is applied. Reading it off the finished box
                # instead is what made an unticked tool disappear from
                # the tree altogether: the box no longer had it, so the
                # catalog no longer mentioned it, so there was nothing
                # left to tick back on.
                self._offered_catalog = tool_catalog(toolbox)
        toolbox.build_recipe = tuple(recipe)  # type: ignore[attr-defined]
        toolbox.base_sandbox = sandbox  # type: ignore[attr-defined]

        return {"toolbox": toolbox, "root_paths": list(roots)}

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        toolbox = self._get_cached_value("toolbox")
        # The tree lists what is *offered*, not what survived: an
        # unticked tool has to stay in the tree, or unticking it would
        # be irreversible. With no toolbox at all that is the static
        # catalog, which is also what the node shows before anything is
        # wired -- the tools are pickable first and built second.
        self._refresh_overview(self._available_catalog())

        if toolbox is None:
            self._widget_core.push_display(
                "status",
                "Connect a folder (or a folder list) to sandbox_roots to build "
                "the toolbox.",
            )
            return

        catalog = tool_catalog(toolbox)
        roots = self._get_cached_value("root_paths") or []
        categories = ", ".join(self._tool_tree.categories())
        notes = [n for n in (getattr(self, "_import_reach", ""),
                             getattr(self, "_graph_hint", "")) if n]
        self._widget_core.push_display(
            "status",
            f"{len(catalog)} of {len(self._tool_tree.tool_names())} tools "
            f"in {categories or 'no categories'} \u00b7 "
            f"{len(roots)} sandbox root(s)."
            # The warnings go where the root count already is: they are
            # properties of the setup the user just made, and a log line
            # alone is a warning nobody reads (G21).
            + "".join("\n! " + n for n in notes),
        )
