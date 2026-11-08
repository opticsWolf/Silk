# -*- coding: utf-8
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The tool nodes' resizable groups: one splitter, sizes kept in the state.

Qt-only; the imports sit in the tests so the file still collects where
PySide6 is missing.
"""

from __future__ import annotations

import pytest

NODES = [
    ("silk.nodes.toolbox", "SilkToolBoxNode", 3),
    ("silk.nodes.role", "SilkRoleNode", 3),
    ("silk.nodes.toolset", "SilkToolSetNode", 2),
    ("silk.nodes.mcp_aggregator", "SilkMCPAggregatorNode", 2),
]


@pytest.fixture(autouse=True)
def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])


def test_restore_ignores_malformed_sizes():
    from silk.widgets.split_form import SplitForm
    split = SplitForm(2)
    split.splitter.resize(100, 200)
    before = split.sizes()
    for bad in (None, [1], [1, "a"], [-1, 5], [0, 0], "120,80"):
        split.restore(bad)
        assert split.sizes() == before


@pytest.mark.parametrize("module, name, panes", NODES)
def test_node_keeps_its_pane_sizes(module, name, panes):
    import importlib
    from weave.node import VerticalSizePolicy
    from silk.widgets.split_form import STATE_KEY
    cls = getattr(importlib.import_module(module), name)
    assert cls.vertical_size_policy is VerticalSizePolicy.GROW_ONLY
    node = cls()
    try:
        assert len(node._split.forms) == panes
        node._split.splitter.resize(200, 100 * panes)
        sizes = [40 + 20 * i for i in range(panes)]
        node._split.splitter.setSizes(sizes)
        saved = node.get_state()
        assert saved[STATE_KEY] == node._split.sizes()

        other = cls()
        try:
            other._split.splitter.resize(200, 100 * panes)
            other.restore_state(saved)
            assert other._split.sizes() == saved[STATE_KEY]
        finally:
            other.cleanup()
    finally:
        node.cleanup()


@pytest.mark.parametrize("module, name, panes", NODES)
def test_the_tall_widgets_take_their_groups_spare_height(module, name, panes):
    import importlib
    from PySide6.QtWidgets import QSizePolicy
    cls = getattr(importlib.import_module(module), name)
    node = cls()
    try:
        tree = node._tool_tree
        assert (tree.sizePolicy().verticalPolicy()
                is QSizePolicy.Policy.Expanding)
        assert node._split.splitter.restack() is True
    finally:
        node.cleanup()


@pytest.mark.parametrize("module, name", [
    ("silk.nodes.role", "SilkRoleNode"),
    ("silk.nodes.toolset", "SilkToolSetNode"),
])
def test_allow_all_ticks_every_tool_and_locks_the_tree(module, name):
    import importlib
    from PySide6.QtCore import Qt
    node = getattr(importlib.import_module(module), name)()
    try:
        tree = node._tool_tree
        tree.set_catalog([{"name": n, "category": "file", "risk": "low"}
                          for n in ("a", "b", "c")])
        node._widget_core.apply_port_value("checked_tools", ["b"])
        hidden_before = tree.isHidden()

        node.chk_allow_all.setChecked(True)
        assert tree.isHidden() == hidden_before     # nothing hides
        # Locked row by row: the tree stays usable (details, gears).
        assert tree._tree.isEnabled()
        kids = tree._tree.topLevelItem(0)
        assert all(kids.child(i).checkState(0) == Qt.CheckState.Checked
                   for i in range(kids.childCount()))
        assert not any(_tickable(kids.child(i))
                       for i in range(kids.childCount()))
        assert not _tickable(kids)
        # Display only: the user's own ticks are the value, kept.
        assert node._widget_core.get_port_value("checked_tools") == ["b"]

        node.chk_allow_all.setChecked(False)
        assert all(_tickable(kids.child(i))
                   for i in range(kids.childCount()))
        ticked = [kids.child(i).text(0) for i in range(kids.childCount())
                  if kids.child(i).checkState(0) == Qt.CheckState.Checked]
        assert ticked == ["b"]
    finally:
        node.cleanup()


def _write(node, port, value) -> None:
    """A programmatic write, with the tick its announcement waits for."""
    from PySide6.QtWidgets import QApplication
    node._widget_core.apply_port_value(port, value)
    QApplication.processEvents()


def _tickable(item) -> bool:
    from PySide6.QtCore import Qt
    return bool(item.flags() & Qt.ItemFlag.ItemIsUserCheckable)


def _shown(tree) -> dict:
    """``{tool: (shown ticked, tickable)}`` over the whole tree."""
    from PySide6.QtCore import Qt
    out = {}
    for i in range(tree._tree.topLevelItemCount()):
        row = tree._tree.topLevelItem(i)
        for j in range(row.childCount()):
            kid = row.child(j)
            out[kid.text(0)] = (kid.checkState(0) == Qt.CheckState.Checked,
                                _tickable(kid))
    return out


RISKY = [
    {"name": "peek", "category": "file", "risk": "low"},
    {"name": "edit", "category": "file", "risk": "medium"},
    {"name": "wipe", "category": "file", "risk": "high"},
]


def test_max_risk_unticks_and_locks_the_tools_above_it():
    from silk.nodes.role import SilkRoleNode
    node = SilkRoleNode()
    try:
        tree = node._tool_tree
        tree.set_catalog(RISKY)
        node._widget_core.apply_port_value("checked_tools", ["edit", "wipe"])

        _write(node, "max_risk", "medium")
        assert _shown(tree) == {"peek": (False, True), "edit": (True, True),
                                "wipe": (False, False)}
        wipe = tree._tree.topLevelItem(0).child(2)
        assert wipe.toolTip(0).startswith("Above the Max Risk ceiling (medium)")
        # The tick underneath is kept: it is the user's, not the lock's.
        assert node._widget_core.get_port_value("checked_tools") == ["edit", "wipe"]

        # The ceiling wins over Allow All, as ToolSelector.permits does.
        node.chk_allow_all.setChecked(True)
        assert _shown(tree)["wipe"] == (False, False)
        assert _shown(tree)["peek"] == (True, False)
        node.chk_allow_all.setChecked(False)

        _write(node, "max_risk", "")
        assert _shown(tree)["wipe"] == (True, True)
    finally:
        node.cleanup()


def test_a_category_click_ticks_and_unticks_only_the_free_tools():
    from PySide6.QtCore import Qt
    from silk.nodes.role import SilkRoleNode
    node = SilkRoleNode()
    try:
        tree = node._tool_tree
        tree.set_catalog(RISKY)
        node._widget_core.apply_port_value("checked_tools", ["wipe"])
        _write(node, "max_risk", "low")
        row = tree._tree.topLevelItem(0)

        row.setCheckState(0, Qt.CheckState.Checked)        # a click
        assert node._widget_core.get_port_value("checked_tools") == ["peek", "wipe"]
        # Part-ticked (wipe shows locked and unticked); a second click
        # unticks the free tools rather than sticking at ticked.
        row.setCheckState(0, Qt.CheckState.Checked)
        assert node._widget_core.get_port_value("checked_tools") == ["wipe"]
    finally:
        node.cleanup()


def test_a_restored_role_shows_its_locks():
    from silk.nodes.role import SilkRoleNode
    node = SilkRoleNode()
    try:
        _write(node, "max_risk", "low")
        node._widget_core.apply_port_value("allow_all", True)
        state = node.get_state()
    finally:
        node.cleanup()
    other = SilkRoleNode()
    try:
        other.restore_state(state)
        assert other._tool_tree.risk_ceiling() == "low"
        assert other._tool_tree.all_granted()
    finally:
        other.cleanup()


def test_the_toolset_keeps_its_knowledge_level_behind_the_gear():
    from PySide6.QtWidgets import QFormLayout
    from silk.functions.tool_groups import categories_of
    from silk.nodes.toolset import SilkToolSetNode
    node = SilkToolSetNode()
    try:
        labels = []
        for form in node._split.forms:
            for r in range(form.rowCount()):
                item = form.itemAt(r, QFormLayout.ItemRole.LabelRole)
                if item is not None and item.widget() is not None:
                    labels.append(item.widget().text())
        assert "Knowledge:" not in labels

        category = sorted(categories_of("knowledge"))[0]
        assert category in node._tool_tree._configurable
        node._tool_tree.config_requested.emit(category)
        dialog = node._config_dialogs.get(category)
        assert dialog is not None
        assert node._knowledge_level.window() is dialog
        dialog.close()
    finally:
        node.cleanup()


@pytest.mark.parametrize("module, name", [
    ("silk.nodes.toolbox", "SilkToolBoxNode"),
    ("silk.nodes.toolset", "SilkToolSetNode"),
    ("silk.nodes.role", "SilkRoleNode"),
])
def test_the_category_dropdown_follows_the_catalog_and_filters(module, name):
    import importlib
    from silk.widgets.tool_tree import ALL_CATEGORIES
    node = getattr(importlib.import_module(module), name)()
    try:
        tree, combo = node._tool_tree, node._combo_category
        tree.set_catalog([
            {"name": "a", "category": "file", "risk": "low"},
            {"name": "b", "category": "web", "risk": "low"},
        ])
        entries = [combo.itemText(i) for i in range(combo.count())]
        assert entries == [ALL_CATEGORIES, "file", "web"]

        combo.setCurrentText("web")
        shown = [tree._tree.topLevelItem(i).text(0)
                 for i in range(tree._tree.topLevelItemCount())
                 if not tree._tree.topLevelItem(i).isHidden()]
        assert shown == ["web  (1)"]

        # A new catalog keeps the pick while it is still offered.
        tree.set_catalog([
            {"name": "b", "category": "web", "risk": "low"},
            {"name": "c", "category": "code", "risk": "low"},
        ])
        assert combo.currentText() == "web"
        shown = [tree._tree.topLevelItem(i).text(0)
                 for i in range(tree._tree.topLevelItemCount())
                 if not tree._tree.topLevelItem(i).isHidden()]
        assert shown == ["web  (1)"]
    finally:
        node.cleanup()


def test_a_toolset_with_allow_all_carries_every_tool_of_the_box(tmp_path):
    from silk.functions.tool_box import ToolBox
    from silk.functions.tools.file_read import attach_file_read_tools
    from silk.functions.tools.file_sandbox import FileToolSandbox
    from silk.nodes.toolset import SilkToolSetNode

    sandbox = FileToolSandbox(root_dir=tmp_path)
    recipe = (("file_read", attach_file_read_tools),)
    box = ToolBox()
    attach_file_read_tools(box, sandbox)
    box.build_recipe = recipe
    box.base_sandbox = sandbox
    offered = set(box.tools)
    assert len(offered) > 1

    node = SilkToolSetNode()
    try:
        out = node.compute({"toolbox": box, "checked_tools": [],
                            "allow_all": True})
        assert offered <= set(out["toolset"].tools)
        out = node.compute({"toolbox": box, "checked_tools": [],
                            "allow_all": False})
        assert "read_file" not in out["toolset"].tools   # infrastructure stays
    finally:
        node.cleanup()
