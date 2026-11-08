# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

The Knowledge Graph node's paths are history pickers.

Qt suite (needs the Weave env; errors here) -- same convention as the
other node suites.
"""
import pytest


@pytest.fixture()
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_db_path_is_a_save_mode_filepath_history_picker(qt_app):
    from weave.widgets.path_history_picker import PathHistoryPicker
    from silk.nodes.knowledge_graph import SilkKnowledgeGraphNode
    node = SilkKnowledgeGraphNode()
    assert isinstance(node._db_path, PathHistoryPicker)
    assert node._db_path._mode == "save_file"
    assert node._widget_core._bindings["db_path"].datatype == "filepath"


def test_paths_default_into_the_first_root_with_a_graph(tmp_path):
    from pathlib import Path
    from silk.nodes.knowledge_graph import SilkKnowledgeGraphNode
    resolve = SilkKnowledgeGraphNode.resolve_paths
    empty, first, second = (tmp_path / n for n in ("empty", "first", "second"))
    for root in (empty, first, second):
        (root / ".okfgraph").mkdir(parents=True)
    (first / ".okfgraph" / "okfgraph.db").write_bytes(b"")
    (second / "okfgraph.db").write_bytes(b"")
    (second / "project.db").write_bytes(b"")     # not a graph: not offered
    roots = [str(empty), str(first), str(second)]

    db, bundle, found = resolve({"roots": roots})
    assert found == [str(first / ".okfgraph" / "okfgraph.db"), str(second / "okfgraph.db")]
    assert db == found[0], "the first root holding a graph wins"
    assert Path(bundle) == first / ".okfgraph" / "bundle"

    db, bundle, _ = resolve({"roots": roots[2:]})
    assert Path(bundle) == second / ".okfgraph" / "bundle", "beside a root-level db too"

    db, bundle, found = resolve({"roots": [str(empty)]})
    assert found == [] and Path(db) == empty / ".okfgraph" / "okfgraph.db"

    typed = str(tmp_path / "mine.db")
    db, bundle, _ = resolve({"roots": roots, "db_path": typed, "bundle_root": "b"})
    assert (db, bundle) == (typed, "b"), "what is typed or wired wins"
    assert resolve({}) == ("", "", [])


def test_bundle_root_is_a_folder_history_picker(qt_app):
    from weave.widgets.path_history_picker import PathHistoryPicker
    from silk.nodes.knowledge_graph import SilkKnowledgeGraphNode
    node = SilkKnowledgeGraphNode()
    assert isinstance(node._bundle_root, PathHistoryPicker)
    assert node._bundle_root._mode == "folder"
    assert node._widget_core._bindings["bundle_root"].datatype == "dirpath"
