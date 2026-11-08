# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

HF Cache -- a HuggingFace hub-cache browser as a node.

Lists the models already sitting in the local hub cache
(``.../hub/models--org--name``) -- plus plain subfolders holding weight
files, so a downloads shelf scans the same way as the cache. Filters them
through a search box,
and hands the chosen weight file's snapshot path downstream as a
``filepath`` -- which plugs straight into any loader's ``model_path``.
Pure local reads, no downloads, no Hub account: if the box has the
weights, this node finds them.

Ports
    in   cache_path : dirpath       hub dir; blank means autodetect
    in   search     : string        substring filter over the repo list
    in   model      : string        repo id (``org/name``)
    in   file       : string        weight file name inside the repo
    out  model_path : filepath      snapshot path of the chosen weights
    out  file_info  : dict          cache dir, repo, revision, file, size
"""
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.panel.mirror_contracts import MirrorContract
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger
from weave.widgets.path_history_picker import PathHistoryPicker
from weave.widgets.sync_button import SyncButton

from ..functions.hf_cache import (
    HFCachedRepo, candidate_cache_dirs, describe_size, find_cache_dir,
    scan_repos,
)

log = get_logger("HFCacheNode")


class _SharedCombo(QComboBox):
    """QComboBox whose item model is shared with its mirror.

    Stock combos clone their items, so a refilled list (rescan, new
    downloads) never reaches the mirror panel. Sharing the model keeps
    every surface's list live; the picked VALUE still travels through
    bindings as usual. Same contract as the HF Download node's lists.
    """

    __mirror__ = MirrorContract(
        clone=lambda _src, _binding: _SharedCombo(),
        value_signal="currentTextChanged",
        default_datatype="str",
        shared_model=True,
    )


@register_node
class SilkHFCacheNode(ActiveNode):
    """Browses the local HuggingFace hub cache for model weight files."""

    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    node_name: ClassVar[Optional[str]] = "HF Cache"
    node_description: ClassVar[Optional[str]] = (
        "Lists models already in the local HuggingFace cache, filters them "
        "by search, and emits the chosen weight file's path -- straight into "
        "a loader's model path. Local reads only, never downloads.")
    node_tags: ClassVar[Optional[List[str]]] = ["silk", "huggingface", "cache", "models"]
    node_icon: ClassVar[Optional[str]] = "database"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 2

    @classmethod
    def migrate_state(cls, state: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """api 1 → 2: the weights output is ``model_path`` (same snapshot
        path, matching the loaders' input name); re-draw wires from it.
        ``cache_path`` is an INPUT ``dirpath`` behind a history picker
        now (was a form-only string) -- a saved path carries over. The
        folder history travels in ``cache_history``.
        """
        return state
    # 2: `cache_dir` (the resolved cache folder) is the second output,
    # ahead of `file_info`.
    node_version = 2

    def __init__(self, title: str = "HF Cache", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        # cache_path/model/file are wireable (one add_input each with
        # their widget). Pure outputs are declared here.
        self.add_output("model_path", datatype="filepath")
        # The cache folder actually in use (typed, or the one found), for
        # nodes that take a cache folder: the ONNX loaders, HF Download.
        self.add_output("cache_dir", datatype="dirpath")
        self.add_output("file_info", datatype="dict")

        # ── Form ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self._path = PathHistoryPicker(mode="folder")
        self._path.setToolTip(
            "Hub cache dir (the folder holding models--*). Blank autodetects: "
            "HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE, HF_HOME/hub, "
            "XDG_CACHE_HOME/huggingface/hub, then ~/.cache/huggingface/hub. "
            "Recent roots are kept in the dropdown.")
        form.addRow("Cache path:", self._path)
        self._widget_core.register_widget(
            "cache_path", self._path, role=PortRole.INPUT,
            datatype="dirpath", default="", add_to_layout=False)
        self.add_input("cache_path", datatype="dirpath")

        path_row = QHBoxLayout()
        self._btn_rescan = SyncButton(initial_text="Rescan",
                                      initial_icon="refresh")
        self._btn_rescan.setToolTip("Re-read the cache dir (new downloads appear).")
        self._btn_rescan.clicked.connect(lambda: self.refresh_cache())
        path_row.addWidget(self._btn_rescan)
        path_row.addStretch(1)
        form.addRow("", path_row)
        self._widget_core.register_widget(
            "btn_rescan", self._btn_rescan, role=PortRole.INTERNAL,
            add_to_layout=False)

        self._search = QLineEdit()
        self._search.setPlaceholderText("filter models…")
        self._search.setToolTip("Substring filter over repo ids (org/name).")
        form.addRow("Search:", self._search)
        self._widget_core.register_widget(
            "search", self._search, role=PortRole.INTERNAL, datatype="string",
            default="", add_to_layout=False)

        self._combo_model = _SharedCombo()
        self._combo_model.setToolTip("Cached model repos (org/name).")
        form.addRow("Model:", self._combo_model)
        self._widget_core.register_widget(
            "model", self._combo_model, role=PortRole.INPUT,
            datatype="string", default="", add_to_layout=False,
        )
        self.add_input("model", datatype="string")

        self._combo_file = _SharedCombo()
        self._combo_file.setToolTip("Weight files in the selected snapshot.")
        form.addRow("File:", self._combo_file)
        self._widget_core.register_widget(
            "file", self._combo_file, role=PortRole.INPUT,
            datatype="string", default="", add_to_layout=False,
        )
        self.add_input("file", datatype="string")

        self._label_status = QLabel("Cache not scanned yet.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self.set_content_widget(self._widget_core)

        # Cascade wiring (selection restores after every refill).
        # NOTE: no editingFinished off the folder picker (composite
        # widgets expose none) -- path edits from any surface rescan via
        # on_ui_change below; the Rescan button stays the explicit kick.
        self._search.textChanged.connect(lambda: self.refresh_cache(keep=True))
        self._combo_model.currentIndexChanged.connect(self._on_model_changed)

        self._repos: List[HFCachedRepo] = []
        self._sync_status: str = "Cache not scanned yet."
        self._last_status: str = ""
        self._last_admit: Dict[str, str] = {}
        self.refresh_cache()

    # ── State: keep the folder history across saves ──────────────────
    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["cache_history"] = self._path.history()
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        with self._widget_core.suppress_signals():
            super().restore_state(state)
        self._path.set_history(state.get("cache_history", []))

    def on_ui_change(self) -> None:
        # Sanctioned all-surfaces hook: mirror/undo/restore edits arrive
        # via bindings with widget signals held. Rescans are idempotent
        # (blocked refills, no admits), so this terminates everywhere.
        super().on_ui_change()
        self.refresh_cache(keep=True)

    # ── Scan (main thread: touches widgets) ──────────────────────────
    def _admit(self, port_name: str, value: str) -> None:
        """Publish a value only when it differs from the last publish.

        Programmatic refills restore selections with signals held, so the
        value bus never hears about them; the admit replays the current
        pick so mirrors and downstream stay in sync. Idempotent: every
        admit echoes through ``port_value_written`` into ``on_ui_change``
        on every surface, and unconditional admits keep such rings
        spinning. User edits bypass this record; the next call re-sends
        at most once and converges. Same guard as HF Download.
        """
        if value != self._last_admit.get(port_name):
            self._last_admit[port_name] = value
            self._widget_core.apply_port_value(
                port_name, value, user_edit=False)

    def _set_status(self, text: str) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        try:
            self._widget_core.push_display("status", text)
        except RuntimeError:
            pass

    def refresh_cache(self, keep: bool = False) -> None:
        """Re-read the cache dir and refill both combos, keeping selection."""
        path, source = find_cache_dir(self._widget_core.get_port_value("cache_path"))
        self._offer_discovered_roots()
        repos = scan_repos(path) if __import__("os").path.isdir(path) else []
        self._repos = repos
        filt = (self._widget_core.get_port_value("search") or "").strip().lower()
        want_model = self._widget_core.get_port_value("model") or ""
        want_file = self._widget_core.get_port_value("file") or ""

        items = [repo.repo_id for repo in repos
                 if not filt or filt in repo.repo_id.lower()]
        current = [self._combo_model.itemText(i)
                   for i in range(self._combo_model.count())]
        if current != items:
            self._combo_model.blockSignals(True)
            self._combo_model.clear()
            for repo_id in items:
                self._combo_model.addItem(repo_id)
            self._combo_model.blockSignals(False)
        if want_model:
            idx = self._combo_model.findText(want_model)
            if idx >= 0 and idx != self._combo_model.currentIndex():
                self._combo_model.blockSignals(True)
                self._combo_model.setCurrentIndex(idx)
                self._combo_model.blockSignals(False)

        self._refill_files(want_file)
        self._admit("model", self._combo_model.currentText())
        n_files = sum(len(r.files) for r in repos)
        if not __import__("os").path.isdir(path):
            self._sync_status = (f"Cache not found: {path} ({source}) -- set the path.")
        elif not repos:
            self._sync_status = f"No cached models with weight files in {path}."
        else:
            picked = self._current_pick()
            n_loose = sum(1 for r in repos if r.loose)
            loose_bit = f" ({n_loose} loose)" if n_loose else ""
            self._sync_status = (
                f"{len(repos)} repos{loose_bit}, {n_files} weight files"
                f" ({source})."
                + (f" Picked {picked[0]} > {picked[1]}." if picked else ""))
        self._set_status(self._sync_status)

    def _offer_discovered_roots(self) -> None:
        """Seed the folder dropdown with every existing cache root.

        Several roots can coexist while autodetect uses only the winner;
        offering the rest keeps a second cache one pick away. Discovered
        roots lead (fresh facts), the user's own picks follow, capped at
        the picker's history size. The blank autodetect value is untouched.
        """
        found = [p for p, _s in candidate_cache_dirs()]
        if not found:
            return
        merged = list(found)
        for entry in self._path.history():
            if entry not in merged:
                merged.append(entry)
        self._path.set_history(merged[:PathHistoryPicker.MAX_HISTORY])

    def _refill_files(self, want_file: str = "") -> None:
        repo = self._repo_by_id(self._combo_model.currentText())
        items = []
        if repo is not None:
            for f in repo.files:
                items.append((f"{f.name} ({describe_size(f.size)})",
                              f.name))
        current = [(self._combo_file.itemText(i),
                    self._combo_file.itemData(i))
                   for i in range(self._combo_file.count())]
        if current != items:
            self._combo_file.blockSignals(True)
            self._combo_file.clear()
            for text, data in items:
                self._combo_file.addItem(text, userData=data)
            self._combo_file.blockSignals(False)
        if want_file:
            idx = self._combo_file.findData(want_file)
            if idx >= 0 and idx != self._combo_file.currentIndex():
                self._combo_file.blockSignals(True)
                self._combo_file.setCurrentIndex(idx)
                self._combo_file.blockSignals(False)
        self._admit("file", self._combo_file.currentText())

    def _on_model_changed(self, _index: int) -> None:
        self._refill_files()

    def _repo_by_id(self, repo_id: str) -> Optional[HFCachedRepo]:
        for repo in self._repos:
            if repo.repo_id == repo_id:
                return repo
        return None

    def _current_pick(self) -> Optional[tuple[str, str]]:
        repo = self._repo_by_id(self._combo_model.currentText())
        if repo is None:
            return None
        name = self._combo_file.currentData()
        if not name:
            return None
        return repo.repo_id, str(name)

    # ── Main thread ───────────────────────────────────────────────────
    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        try:
            self._widget_core.push_display("status", self._sync_status)
        except RuntimeError:
            pass

    # ── Worker thread ─────────────────────────────────────────────────
    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import os
        path, source = find_cache_dir(inputs.get("cache_path"))
        info: Dict[str, Any] = {"cache_dir": path, "source": source,
                                "repos": 0, "repo": None, "revision": None,
                                "file": None, "size": 0}
        if not os.path.isdir(path):
            self._sync_status = (f"Cache not found: {path} ({source}) -- set the path.")
            return {"model_path": None, "cache_dir": None, "file_info": info}
        repos = scan_repos(path)
        info["repos"] = len(repos)
        repo_id = str(inputs.get("model") or "")
        repo = next((r for r in repos if r.repo_id == repo_id), None)
        if repo is None:
            self._sync_status = (f"{len(repos)} repos ({source}). "
                                 + ("Pick a model." if not repo_id
                                    else f"{repo_id} is not in the cache."))
            return {"model_path": None, "cache_dir": path, "file_info": info}
        info.update(repo=repo.repo_id, revision=repo.revision,
                    via_ref=repo.via_ref)
        want = str(inputs.get("file") or "")
        picked = next((f for f in repo.files if f.name == want), None)
        if picked is None:
            picked = repo.files[0] if repo.files else None
        if picked is None or not os.path.isfile(picked.path):
            self._sync_status = f"{repo.repo_id}: no weight files in this snapshot."
            return {"model_path": None, "cache_dir": path, "file_info": info}
        info.update(file=picked.name, size=picked.size)
        picked_id = f"{repo.repo_id} > {picked.name}"
        self._sync_status = (f"{len(repos)} repos ({source}). Picked {picked_id}.")
        return {"model_path": picked.path, "cache_dir": path, "file_info": info}

    def cleanup(self) -> None:
        super().cleanup()
