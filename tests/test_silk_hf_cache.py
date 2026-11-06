# -*- coding: utf-8
"""The HF cache browser: layout scan, autodetect, dropdowns, filepath out.

Function tests build a fake hub tree (no network, no huggingface_hub).
Qt node tests live at the bottom with in-fixture imports, so this file
still collects where PySide6 is missing.
"""

from __future__ import annotations

import os

import pytest

from silk.functions.hf_cache import (
    default_cache_dir, describe_size, parse_repo_id,
    resolve_snapshot, scan_repos, snapshot_files,
)


def _write(path: str, size: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(bytes((i % 251 for i in range(size))))


@pytest.fixture()
def hub(tmp_path):
    root = tmp_path / "hub"
    gguf = root / "models--org--gguf"
    _write(str(gguf / "snapshots" / "abc123" / "model.gguf"), 1024)
    _write(str(gguf / "snapshots" / "abc123" / "config.json"), 100)
    _write(str(gguf / "snapshots" / "old111" / "pytorch_model.bin"), 512)
    (gguf / "refs").mkdir(parents=True, exist_ok=True)
    (gguf / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    shards = root / "models--org--shards"
    _write(str(shards / "snapshots" / "def456" / "model-00001-of-00002.safetensors"), 300)
    _write(str(shards / "snapshots" / "def456" / "model-00002-of-00002.safetensors"), 300)
    _write(str(shards / "snapshots" / "def456" / "model.safetensors.index.json"), 50)
    empty = root / "models--org--empty"
    _write(str(empty / "snapshots" / "zzz" / "readme.md"), 10)
    _write(str(root / "datasets--org--ds" / "snapshots" / "q" / "data.parquet"), 10)
    (root / "random-file.txt").write_text("x", encoding="utf-8")
    return str(root)


# ── Layout scan ──────────────────────────────────────────────────────
def test_parse_repo_id_only_accepts_models():
    assert parse_repo_id("models--org--name") == "org/name"
    assert parse_repo_id("models--qwen--qwen3") == "qwen/qwen3"
    assert parse_repo_id("datasets--org--ds") is None
    assert parse_repo_id("random-file.txt") is None


def test_scan_finds_weight_repos_sorted_and_skips_the_rest(hub):
    repos = scan_repos(hub)
    assert [r.repo_id for r in repos] == ["org/gguf", "org/shards"]
    gguf = repos[0]
    assert gguf.revision == "abc123" and gguf.via_ref is True
    assert [f.name for f in gguf.files] == ["model.gguf"]
    assert not any(r.loose for r in repos)


def test_scan_repos_includes_loose_subfolders_by_default(tmp_path):
    from silk.functions.hf_cache import scan_loose_weights
    root = tmp_path / "shelf"
    (root / "qwen").mkdir(parents=True)
    (root / "qwen" / "model.gguf").write_bytes(b"0" * 64)
    (root / "vendor" / "deep").mkdir(parents=True)
    (root / "vendor" / "deep" / "m.safetensors").write_bytes(b"0" * 128)
    (root / "notes.txt").write_text("x", encoding="utf-8")
    hidden = root / ".hidden"
    hidden.mkdir(exist_ok=True)
    (hidden / "h.gguf").write_bytes(b"0")
    (root / "top.gguf").write_bytes(b"0" * 32)
    hub = root / "models--org--hub"
    (hub / "snapshots" / "c" ).mkdir(parents=True)
    (hub / "snapshots" / "c" / "h.gguf").write_bytes(b"0")
    loose = scan_loose_weights(str(root))
    assert [r.repo_id for r in loose] == ["(root)", "qwen",
                                          "vendor/deep"]
    assert all(r.loose and r.revision is None for r in loose)
    assert [f.name for f in loose[1].files] == ["model.gguf"]
    both = scan_repos(str(root))
    assert [r.repo_id for r in both] == ["(root)", "org/hub", "qwen",
                                         "vendor/deep"]


def test_loose_scan_respects_its_bounds(tmp_path):
    from silk.functions.hf_cache import scan_loose_weights
    root = tmp_path / "deep"
    far = root / "a" / "b" / "c" / "d" / "e"
    far.mkdir(parents=True)
    (far / "far.gguf").write_bytes(b"0")
    near = root / "a" / "shallow.gguf"
    near.parent.mkdir(parents=True, exist_ok=True)
    near.write_bytes(b"0")
    assert [r.repo_id for r in scan_loose_weights(str(root))] == ["a"]
    assert [r.repo_id for r in scan_loose_weights(str(root), max_depth=9)] == [
        "a", "a/b/c/d/e"]
    assert scan_loose_weights(str(root), max_files=0) == []


def test_snapshot_resolution_prefers_refs_main_then_newest(hub):
    commit, via_ref = resolve_snapshot(os.path.join(hub, "models--org--gguf"))
    assert (commit, via_ref) == ("abc123", True)
    commit, via_ref = resolve_snapshot(os.path.join(hub, "models--org--shards"))
    assert (commit, via_ref) == ("def456", False)


def test_weight_files_skip_configs_and_sort_shards_last(hub):
    files = snapshot_files(os.path.join(hub, "models--org--shards"), "def456")
    assert [f.name for f in files] == ["model-00001-of-00002.safetensors",
                                       "model-00002-of-00002.safetensors"]
    assert all(f.is_shard for f in files)
    assert snapshot_files(os.path.join(hub, "models--org--empty"), "zzz") == []
    assert scan_repos(os.path.join("does", "not", "exist")) == []


def test_describe_size():
    assert describe_size(512) == "512 B"
    assert describe_size(2048) == "2.0 KB"
    assert describe_size(3 * 1024 ** 3) == "3.0 GB"


def test_find_cache_dir_follows_hf_precedence(monkeypatch, tmp_path):
    from silk.functions.hf_cache import find_cache_dir as find
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    custom = tmp_path / "hub"
    custom.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(custom))
    path, source = find("  ")
    assert (path, source) == (str(custom), "env:HF_HUB_CACHE")
    monkeypatch.delenv("HF_HUB_CACHE")
    path, source = default_cache_dir()
    assert source == "default (missing)" and path.endswith("hub")
    path, source = find(str(custom))
    assert (path, source) == (str(custom), "explicit")


# ── Qt wiring (Weave env; imports local so collection works anywhere) ──
@pytest.fixture()
def node():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from silk.nodes.hf_cache_loader import SilkHFCacheNode
    return SilkHFCacheNode()


def gather(node, wired=None):
    core = node._widget_core
    inputs = {name: core.get_port_value(name) for name in core.bindings()}
    inputs.update({"cache_path": None, "search": None,
                   "model": None, "file": None})
    inputs.update(wired or {})
    return inputs


def test_candidate_cache_dirs_lists_every_existing_root(tmp_path, monkeypatch):
    from silk.functions import hf_cache
    first = tmp_path / "first"
    first.mkdir()
    second_hub = tmp_path / "second" / "hub"
    second_hub.mkdir(parents=True)
    home_hub = tmp_path / "home" / ".cache" / "huggingface" / "hub"
    home_hub.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(first))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "missing"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "second"))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(tmp_path / "home")))
    found = hf_cache.candidate_cache_dirs()
    assert [p for p, _s in found] == [str(first), str(second_hub),
                                      str(home_hub)]
    assert [s for _p, s in found] == ["env:HF_HUB_CACHE", "env:HF_HOME",
                                      "default"]


def test_candidate_cache_dirs_dedupes_aliased_roots(tmp_path, monkeypatch):
    from silk.functions import hf_cache
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(tmp_path / "nope")))
    assert hf_cache.candidate_cache_dirs() == [(str(hub), "env:HF_HUB_CACHE")]


def test_node_ports_and_scan(node, hub):
    ports = {p.name: p.datatype for p in node.inputs}
    assert ports == {"cache_path": "dirpath", "search": "string",
                     "model": "string", "file": "string"}
    assert {p.name: p.datatype for p in node.outputs} == {
        "model_path": "filepath", "file_info": "dict"}
    node._widget_core.apply_port_value("cache_path", hub, user_edit=False)
    node.refresh_cache()
    assert node._combo_model.count() == 2
    node._search.setText("gguf")
    node.refresh_cache()
    assert node._combo_model.count() == 1
    assert node._combo_model.itemText(0) == "org/gguf"


def test_cache_path_is_a_history_picker_with_folder_dialog(node, tmp_path):
    from weave.widgets.path_history_picker import PathHistoryPicker
    assert isinstance(node._path, PathHistoryPicker)
    node._path.set_history([str(tmp_path / "hub-a")])
    saved = node.get_state()
    assert saved["cache_history"] == [str(tmp_path / "hub-a")]
    node._path.set_history([])
    node.restore_state(saved)
    assert node._path.history() == [str(tmp_path / "hub-a")]
    assert node._btn_rescan.text() == "Rescan"
    assert "btn_rescan" in node._widget_core._bindings


def test_lists_share_their_model_with_mirrors(node):
    assert type(node._combo_model).__mirror__.shared_model is True
    assert type(node._combo_file).__mirror__.shared_model is True


def test_identical_rescan_is_quiet(node, hub):
    from PySide6.QtTest import QSignalSpy
    node._widget_core.apply_port_value("cache_path", hub, user_edit=False)
    node.refresh_cache()
    assert node._combo_model.count() == 2
    spy = QSignalSpy(node._combo_model.model().modelReset)
    file_spy = QSignalSpy(node._combo_file.model().modelReset)
    node.refresh_cache()
    assert len(spy) == 0 and len(file_spy) == 0


def test_refill_publishes_the_restored_pick(node, hub):
    node._widget_core.apply_port_value("cache_path", hub, user_edit=False)
    node.refresh_cache()
    assert node._last_admit.get("model") == \
        node._combo_model.currentText()
    assert node._last_admit.get("file") == \
        node._combo_file.currentText()


def test_no_dangling_self_calls_cache_node():
    import ast as _ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "nodes" / "hf_cache_loader.py")
    tree = _ast.parse(src.read_text())
    cls = next(n for n in _ast.walk(tree)
               if isinstance(n, _ast.ClassDef)
               and n.name == "SilkHFCacheNode")
    defined = {n.name for n in cls.body
               if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
    called = {n.func.attr for n in _ast.walk(cls)
              if isinstance(n, _ast.Call)
              and isinstance(n.func, _ast.Attribute)
              and isinstance(n.func.value, _ast.Name)
              and n.func.value.id == "self"}
    allow = {"add_input", "add_output", "set_content_widget"}
    assert not called - defined - allow


def test_refresh_lists_discovered_caches_first(node, tmp_path, monkeypatch):
    import silk.nodes.hf_cache_loader as loader_mod
    monkeypatch.setattr(loader_mod, "candidate_cache_dirs",
                        lambda: [(str(tmp_path / "hub-b"), "env:X"),
                                 (str(tmp_path / "hub-a"), "env:Y")])
    node._path.set_history([str(tmp_path / "mine"),
                            str(tmp_path / "hub-a")])
    node.refresh_cache()
    assert node._path.history() == [str(tmp_path / "hub-b"),
                                     str(tmp_path / "hub-a"),
                                     str(tmp_path / "mine")]


def test_compute_resolves_the_pick_to_a_snapshot_path(node, hub):
    out = node.compute(gather(node, {"cache_path": hub, "model": "org/gguf",
                                     "file": "model.gguf"}))
    assert out["model_path"] is not None and out["model_path"].endswith(
        os.path.join("snapshots", "abc123", "model.gguf"))
    assert out["file_info"]["repo"] == "org/gguf"
    assert out["file_info"]["revision"] == "abc123"


def test_compute_without_a_match_emits_nothing(node, hub):
    out = node.compute(gather(node, {"cache_path": hub, "model": "org/nope"}))
    assert out["model_path"] is None
    out = node.compute(gather(node, {"cache_path": str(hub) + "-missing"}))
    assert out["model_path"] is None
