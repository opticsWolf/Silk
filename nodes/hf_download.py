# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

HF Download -- download HuggingFace-hosted models to this machine.

Searches the Hub, lists a repo's files, and downloads with resume:
interrupted downloads keep a ``.partial`` + ``.resume.json`` sidecar,
so Cancel, a dropped connection or a restart continues where it left
off instead of starting over. Nothing here needs ``huggingface_hub``:
search, listing and ``Range`` downloads are plain HTTPS (stdlib), the
token is sent only to ``huggingface.co`` (stripped on CDN redirect),
and Hub-cache targets are written in real hub layout so the HF Cache
browser node picks them up immediately.

The worker runs on the node's own thread: the canvas stays responsive,
other nodes keep evaluating, and Cancel cooperates between chunks.
"""
import os
import re
import threading
import time
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtCore import QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QWidget,
)

from weave.node.threaded import ThreadedManualNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger
from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import BindingPolicy, STREAM_ONLY
from weave.panel.mirror_contracts import MirrorContract
from weave.widgets.path_history_picker import PathHistoryPicker
from weave.widgets.sync_button import SyncButton


class _SharedCombo(QComboBox):
    """QComboBox whose item model is shared with its mirror.

    Stock combos clone their items, so a repopulated list (search hits,
    revision files) never reaches the mirror panel. Sharing the model
    keeps every surface's list live; the picked VALUE still travels
    through bindings as usual.
    """

    __mirror__ = MirrorContract(
        clone=lambda _src, _binding: _SharedCombo(),
        value_signal="currentTextChanged",
        default_datatype="str",
        shared_model=True,
    )


def _looks_like_sha(rev: str) -> bool:
    """Long hex strings pass through to the Hub as commit pins."""
    return len(rev) >= 7 and all(
        c in "0123456789abcdefABCDEF" for c in rev)


class _SharedEditCombo(_SharedCombo):
    """Editable shared-model combo (revision: refs listed, SHAs typable)."""

    __mirror__ = MirrorContract(
        clone=lambda _src, _binding: _SharedEditCombo(),
        value_signal="currentTextChanged",
        default_datatype="str",
        shared_model=True,
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

from ..functions.credentials import resolve_credential
from ..functions.hf_cache import (
    WEIGHT_SUFFIXES, describe_size, find_cache_dir, hub_repo_dir,
    link_blob, write_ref_main,
)
from ..functions.hf_hub import (
    CONFIG_FILES, HubError, HubFile, download_file, filter_files,
    repo_files, repo_refs, resolve_url, resume_state, search_models,
    with_companions,
)

log = get_logger("HFDownloadNode")

TARGETS = (("Hub cache", "hub"), ("Custom folder", "custom"))
SORTS = (("Most downloaded", "downloads"), ("Most liked", "likes"),
         ("Recently updated", "lastModified"))
KINDS = (("Any file", "any"), ("safetensors", "safetensors"), ("GGUF", "gguf"))


def _count(n: Any) -> str:
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        return "0"
    for unit in ("", "k", "M"):
        if n < 1000 or unit == "M":
            return f"{n}{unit}" if unit == "" else f"{n / 1000:.1f}{unit}"
        n //= 1
    return str(n)


def _role_of(filename: str) -> str:
    """``weight`` or ``companion`` (index/config/tokenizer) for the report."""
    if filename.endswith(".index.json") or filename in CONFIG_FILES:
        return "companion"
    return "weight"


@register_node
class SilkHFDownloadNode(ThreadedManualNode):
    """Downloads HuggingFace-hosted models with resume and progress."""

    search_done = Signal(object)     # List[dict] | HubError
    files_done = Signal(object)      # (files, pinned_rev) | HubError
    progress_detail = Signal(int, int)  # downloaded, total (main-thread bar+label)

    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    node_name: ClassVar[Optional[str]] = "HF Download"
    node_description: ClassVar[Optional[str]] = (
        "Searches HuggingFace, downloads repos or single files with resume, "
        "progress and cancel. Targets the hub cache (visible to the HF Cache "
        "node) or any folder. Stdlib only; the token never leaves huggingface.co.")
    node_tags: ClassVar[Optional[List[str]]] = ["silk", "huggingface", "download", "models"]
    node_icon: ClassVar[Optional[str]] = "cloud-download"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 3
    node_version = 1

    @classmethod
    def migrate_state(cls, state: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """api 1 → 2: the Folder port is a ``dirpath`` behind a history
        picker (editable recent-folders dropdown + native dialog).
        The value stays a plain string; re-draw any wires from string
        ports. The folder history travels in ``target_history``.
        api 2 → 3: ``target_path`` is ``target_folder`` (same folder
        type, clearer name). A saved folder carries over; re-draw wires.
        """
        if from_version < 3:
            values = state.get("widget_data")
            if isinstance(values, dict) and "target_folder" not in values:
                if values.get("target_path"):
                    values["target_folder"] = values["target_path"]
        return state

    def __init__(self, title: str = "HF Download", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        # Widget-backed inputs register below (one add_input each);
        # pure outputs are declared here.
        self.add_output("file_path", datatype="filepath")
        self.add_output("download_info", datatype="dict")

        # ── Form ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        search_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("search the Hub…")
        self._search.setToolTip("Matches repo ids and tags (Hub search API).")
        search_row.addWidget(self._search, stretch=1)
        self._btn_search = SyncButton(initial_text="Search", initial_icon="search")
        self._btn_search.setToolTip("Search huggingface.co (public index).")
        self._btn_search.clicked.connect(self._on_search)
        search_row.addWidget(self._btn_search)
        form.addRow("Search:", search_row)
        self._widget_core.register_widget(
            "search", self._search, role=PortRole.INTERNAL, datatype="string",
            default="", add_to_layout=False)

        self._combo_sort = QComboBox()
        for label, value in SORTS:
            self._combo_sort.addItem(label, userData=value)
        form.addRow("Sort:", self._combo_sort)
        self._widget_core.register_widget(
            "sort", self._combo_sort, role=PortRole.INTERNAL,
            datatype="string", default="downloads", add_to_layout=False)

        self._combo_kind = QComboBox()
        for label, value in KINDS:
            self._combo_kind.addItem(label, userData=value)
        self._combo_kind.setToolTip("Filter the file list by weight kind.")
        form.addRow("Kind:", self._combo_kind)
        self._widget_core.register_widget(
            "kind", self._combo_kind, role=PortRole.INTERNAL,
            datatype="string", default="any", add_to_layout=False)

        self._combo_results = _SharedCombo()
        self._combo_results.setToolTip("Search hits -- picking one fills Repo.")
        form.addRow("Results:", self._combo_results)
        self._widget_core.register_widget(
            "results", self._combo_results, role=PortRole.INTERNAL,
            datatype="string", default="",
            policy=BindingPolicy(persist=False, restore=False),
            add_to_layout=False)

        repo_row = QHBoxLayout()
        self._repo = QLineEdit()
        self._repo.setPlaceholderText("org/name")
        self._repo.setToolTip("Repo id to download from (typed or picked above).")
        repo_row.addWidget(self._repo, stretch=1)
        self._btn_open = SyncButton(initial_text="Open repo",
                                    initial_icon="link")
        self._btn_open.setToolTip(
            "Open this repo on huggingface.co in the default browser.")
        self._btn_open.clicked.connect(self._on_open_repo)
        repo_row.addWidget(self._btn_open)
        form.addRow("Repo:", repo_row)
        self._widget_core.register_widget(
            "repo", self._repo, role=PortRole.INPUT,
            datatype="string", default="", add_to_layout=False)
        self.add_input("repo", datatype="string")

        self._revision = _SharedEditCombo()
        self._revision.setEditText("main")
        self._revision.setToolTip(
            "Branch, tag or commit sha. Branches and tags are listed "
            "once the repo is known; anything else can still be typed.")
        form.addRow("Revision:", self._revision)
        self._widget_core.register_widget(
            "revision", self._revision, role=PortRole.INPUT,
            datatype="string", default="main", add_to_layout=False)
        self.add_input("revision", datatype="string")

        self._file_filter = QLineEdit()
        self._file_filter.setPlaceholderText("filter files…")
        self._file_filter.setToolTip(
            "Substring filter over the file list (local, instant). "
            "The current pick is always kept, even when hidden.")
        form.addRow("Filter files:", self._file_filter)
        self._widget_core.register_widget(
            "file_filter", self._file_filter, role=PortRole.INTERNAL,
            datatype="string", default="", add_to_layout=False)

        self._combo_file = _SharedCombo()
        self._combo_file.setToolTip("Files in the selected revision (weights first).")
        form.addRow("File:", self._combo_file)
        self._widget_core.register_widget(
            "file", self._combo_file, role=PortRole.INPUT,
            datatype="string", default="", add_to_layout=False)
        self.add_input("file", datatype="string")

        self._credential = QLineEdit()
        self._credential.setText("HF_TOKEN")
        self._credential.setToolTip(
            "Credential NAME holding a Hub token (env, then secrets file) -- "
            "for private/gated repos. The value is never stored; it is sent "
            "only to huggingface.co, never to the CDN.")
        form.addRow("Credential:", self._credential)
        self._widget_core.register_widget(
            "credential", self._credential, role=PortRole.INTERNAL,
            datatype="string", default="HF_TOKEN", add_to_layout=False)

        self._combo_target = QComboBox()
        for label, value in TARGETS:
            self._combo_target.addItem(label, userData=value)
        self._combo_target.setToolTip(
            "Hub cache: real hub layout (the HF Cache node sees it). "
            "Custom folder: plain <dir>/<org>--<name>/<file>.")
        form.addRow("Target:", self._combo_target)
        self._widget_core.register_widget(
            "target", self._combo_target, role=PortRole.INPUT,
            datatype="string", default="hub", add_to_layout=False)
        self.add_input("target", datatype="string")

        self._target_folder = PathHistoryPicker(mode="folder")
        self._target_folder.setToolTip(
            "Hub cache root (blank autodetects). Recent roots are kept "
            "in the dropdown.")
        form.addRow("Folder:", self._target_folder)
        self._widget_core.register_widget(
            "target_folder", self._target_folder, role=PortRole.INPUT,
            datatype="dirpath", default="", add_to_layout=False)
        self.add_input("target_folder", datatype="dirpath")

        self._chk_all = QCheckBox("All weight files")
        self._chk_all.setToolTip("Download every weight file, not just File.")
        form.addRow("", self._chk_all)
        self._widget_core.register_widget(
            "download_all", self._chk_all, role=PortRole.INTERNAL,
            datatype="bool", default=False, add_to_layout=False)

        self._chk_config = QCheckBox("Include config + tokenizer")
        self._chk_config.setChecked(True)
        self._chk_config.setToolTip(
            "Also download config.json, tokenizer files and friends. "
            "Harmless kilobytes for GGUF; load-bearing for safetensors. "
            "Shard indexes (*.index.json) always join weight downloads.")
        form.addRow("", self._chk_config)
        self._widget_core.register_widget(
            "include_config", self._chk_config, role=PortRole.INTERNAL,
            datatype="bool", default=True, add_to_layout=False)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        form.addRow(self._progress)
        self._widget_core.register_widget(
            "progress", self._progress,
            role=PortRole.DISPLAY, datatype="float",
            add_to_layout=False, policy=STREAM_ONLY,
        )

        btn_row = QHBoxLayout()
        self._btn_download = SyncButton(initial_text="Download",
                                        initial_icon="download")
        self._btn_download.setToolTip(
            "Download the selection (resumes a partial automatically).")
        self._btn_download.clicked.connect(self.execute)
        btn_row.addWidget(self._btn_download)
        self._btn_resume = SyncButton(initial_text="Resume",
                                      initial_icon="player-play")
        self._btn_resume.setToolTip("Continue the interrupted download.")
        self._btn_resume.clicked.connect(self.execute)
        btn_row.addWidget(self._btn_resume)
        self._btn_cancel = SyncButton(initial_text="Cancel", initial_icon="cancel")
        self._btn_cancel.setToolTip("Stop the running download (keeps the partial).")
        self._btn_cancel.clicked.connect(self.cancel_compute)
        btn_row.addWidget(self._btn_cancel)
        btn_row.addStretch(1)
        form.addRow("", btn_row)
        for name, btn in (("btn_search", self._btn_search),
                          ("btn_open", self._btn_open),
                          ("btn_download", self._btn_download),
                          ("btn_resume", self._btn_resume),
                          ("btn_cancel", self._btn_cancel)):
            self._widget_core.register_widget(
                name, btn, role=PortRole.INTERNAL, add_to_layout=False)

        self._label_status = QLabel("Pick a repo to download.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self.set_content_widget(self._widget_core)

        # Worker signals (main thread slots).
        self.search_done.connect(self._on_search_done)
        self.files_done.connect(self._on_files_done)
        self.progress_detail.connect(self._on_progress_detail)
        self.compute_progress.connect(self._on_progress_percent)
        self.compute_started.connect(lambda: self._on_run_state("Downloading…"))
        self.compute_cancelled.connect(
            lambda: self._on_run_state("Cancelled -- Resume to continue."))
        self.compute_error.connect(self._on_compute_error)

        # UI cascade. Enablement + the file view follow the ADMITTED
        # values via on_ui_change (all surfaces: canvas edits, mirror
        # writes, undo, restore). Widget-signal cascades alone miss
        # mirror edits, which arrive via bindings with signals held.
        self._combo_results.currentIndexChanged.connect(self._on_result_picked)
        self._repo.editingFinished.connect(self._fetch_files)
        self._revision.lineEdit().editingFinished.connect(self._fetch_files)

        self._files: List[HubFile] = []
        self._pinned_rev: str = "main"
        self._seq = 0
        self._last_results_text = ""
        self._last_admit: Dict[str, str] = {}
        self._last_fetch_key: Optional[tuple] = None
        self._fetch_timer = QTimer(self)
        self._fetch_timer.setSingleShot(True)
        self._fetch_timer.timeout.connect(self._on_fetch_debounce)
        self._last_tick: Optional[tuple] = None
        self._last_push: float = 0.0
        self._sync_status: str = "Pick a repo to download."
        self._refresh_buttons()

    # ── Search + file listing (transient threads, signals back) ─────
    def _token(self) -> Optional[str]:
        name = (self._widget_core.get_port_value("credential") or "").strip()
        if not name:
            return None
        try:
            return resolve_credential(name) or None
        except Exception:  # noqa: BLE001 -- anonymous fallback, reported below
            return None

    def _on_search(self) -> None:
        query = (self._widget_core.get_port_value("search") or "").strip()
        if not query:
            return
        self._seq += 1
        seq = self._seq
        sort = self._widget_core.get_port_value("sort") or "downloads"
        token = self._token()
        self._set_status(f"Searching for '{query}'…")
        thread = threading.Thread(
            target=self._search_worker, args=(seq, query, sort, token),
            daemon=True)
        thread.start()

    def _search_worker(self, seq: int, query: str, sort: str,
                       token: Optional[str]) -> None:
        try:
            self.search_done.emit((seq, search_models(query, sort=sort, token=token)))
        except HubError as exc:
            self.search_done.emit((seq, exc))
        except Exception as exc:  # noqa: BLE001 -- never die silent
            self.search_done.emit(
                (seq, HubError("network", f"Search worker failed: {exc!r}")))

    def _on_search_done(self, payload: object) -> None:
        seq, result = payload  # type: ignore[misc]
        if seq != self._seq:
            return                              # stale thread: ignore
        if isinstance(result, HubError):
            self._set_status(f"Search failed: {result}")
            self._sync_status = self._label_status.text()
            return
        self._combo_results.blockSignals(True)
        self._combo_results.clear()
        for entry in result:
            label = (f"{entry['id']} (↓{_count(entry.get('downloads'))} "
                     f"· ♥{_count(entry.get('likes'))})")
            self._combo_results.addItem(label, userData=entry["id"])
        self._combo_results.blockSignals(False)
        self._sync_status = f"{len(result)} repos found -- pick one."
        self._set_status(self._sync_status)
        if result:
            self._combo_results.blockSignals(True)
            self._combo_results.setCurrentIndex(0)
            self._combo_results.blockSignals(False)
            self._last_results_text = self._combo_results.currentText()
            self._admit("results", self._last_results_text)
            self._on_result_picked(0)

    def _on_result_picked(self, index: int) -> None:
        repo_id = self._combo_results.itemData(index)
        if not repo_id:
            return
        self._repo.blockSignals(True)
        self._repo.setText(str(repo_id))
        self._repo.blockSignals(False)
        self._admit("repo", str(repo_id))
        self._fetch_files()

    def _fetch_files(self) -> None:
        repo_id = (self._widget_core.get_port_value("repo") or "").strip()
        revision = ((self._widget_core.get_port_value("revision") or "").strip()
                    or "main")
        self._last_fetch_key = (repo_id, revision)
        # Fresh listing context: drop the previous repo's text filter, or
        # the new files filter down to nothing and the list looks broken.
        if (self._widget_core.get_port_value("file_filter") or "").strip():
            self._file_filter.blockSignals(True)
            self._file_filter.setText("")
            self._file_filter.blockSignals(False)
            self._admit("file_filter", "")
        if not repo_id:
            self._files = []
            self._populate_file_combo()
            return
        self._seq += 1
        seq = self._seq
        token = self._token()
        thread = threading.Thread(
            target=self._files_worker, args=(seq, repo_id, revision, token),
            daemon=True)
        thread.start()

    def _files_worker(self, seq: int, repo_id: str, revision: str,
                      token: Optional[str]) -> None:
        try:
            refs = repo_refs(repo_id, token=token)
            names = refs["branches"] + refs["tags"]
            resolved = revision
            if (names and revision not in names
                    and not _looks_like_sha(revision)):
                # Renamed default branch, typo, stale pin: list the repo's
                # own default instead of 404ing. SHAs pass through.
                resolved = (refs["branches"] or names)[0]
            files, pinned = repo_files(repo_id, revision=resolved, token=token)
            self.files_done.emit((seq, (repo_id, files, pinned, refs,
                                        resolved)))
        except HubError as exc:
            self.files_done.emit((seq, exc))
        except Exception as exc:  # noqa: BLE001 -- never die silent
            self.files_done.emit(
                (seq, HubError("network", f"Listing worker failed: {exc!r}")))

    def _on_files_done(self, payload: object) -> None:
        seq, result = payload  # type: ignore[misc]
        if seq != self._seq:
            return
        if isinstance(result, HubError):
            self._sync_status = f"Listing failed: {result}"
            self._set_status(self._sync_status)
            return
        repo_id, files, pinned, refs, resolved = result
        if repo_id != (self._widget_core.get_port_value("repo") or "").strip():
            return  # stale: the box moved on while this listed
        self._populate_revision_combo(refs, resolved)
        self._last_fetch_key = (repo_id, self._revision_text())
        self._files = list(files)
        self._pinned_rev = pinned or "main"
        self._populate_file_combo()
        self._admit("file", self._combo_file.currentText())
        self._refresh_buttons()

    def _revision_text(self) -> str:
        return (self._widget_core.get_port_value("revision") or "").strip()

    def _populate_revision_combo(self, refs: Dict[str, List[str]],
                                 resolved: str) -> None:
        """Refill the revision list: branches, then tags, pick preserved.

        The typed value always survives (SHAs are typable, not listed),
        so a custom pin is never clobbered by a fresh refs listing.
        """
        names = list(refs.get("branches", [])) + list(refs.get("tags", []))
        current = self._revision_text()
        # Show what was actually listed: keep the typed value when it is
        # listed or is the resolved pin (SHA); otherwise correct to it.
        want = (current if (current and (current in names or current == resolved))
                else resolved)
        shown = list(names)
        if want and want not in shown:
            shown.append(want)
        if (shown == [self._revision.itemText(i)
                      for i in range(self._revision.count())]
                and self._revision.currentText() == (want or "")):
            return  # identical: keep model echoes quiet (mirror loops)
        self._revision.blockSignals(True)
        self._revision.clear()
        for name in shown:
            self._revision.addItem(name)
        self._revision.blockSignals(False)
        if want:
            self._revision.setEditText(want)
        self._admit("revision", self._revision.currentText())

    def _populate_file_combo(self) -> None:
        """Refill the file list: kind filter, text filter, weights first.

        The current pick is always kept (prepended when filtered out), so
        narrowing the view never silently reselects -- compute downloads
        exactly what the File port says.
        """
        kind = self._widget_core.get_port_value("kind") or "any"
        filt = (self._widget_core.get_port_value("file_filter") or "").strip()
        filt = filt.lower()
        shown = filter_files(self._files, kind)
        if filt:
            shown = [f for f in shown if filt in f.name.lower()]
        weights = [f for f in shown
                   if f.name.lower().endswith(WEIGHT_SUFFIXES)]
        weights.sort(key=lambda f: -(f.size or 0))
        rest = sorted([f for f in shown if f not in weights],
                      key=lambda f: f.name.lower())
        ordered = weights + rest
        current = (self._widget_core.get_port_value("file") or "").strip()
        if current and current not in [f.name for f in ordered]:
            keep = next((f for f in self._files if f.name == current), None)
            ordered = ([keep] if keep is not None else []) + ordered
        display = [f"{f.name} ({describe_size(f.size)})" if f.size else f.name
                   for f in ordered]
        if display == [self._combo_file.itemText(i)
                       for i in range(self._combo_file.count())]:
            return  # identical: keep model echoes quiet (mirror loops)
        self._combo_file.blockSignals(True)
        self._combo_file.clear()
        for f in ordered:
            size = f" ({describe_size(f.size)})" if f.size else ""
            self._combo_file.addItem(f"{f.name}{size}", userData=f.name)
        if current:
            idx = self._combo_file.findData(current)
            if idx >= 0:
                self._combo_file.setCurrentIndex(idx)
        self._combo_file.blockSignals(False)
        scope = f" (showing {len(ordered)} of {len(self._files)})" \
            if filt or kind != "any" else ""
        self._set_status(f"{len(self._files)} files @ {self._pinned_rev} "
                         f"-- pick one to download.{scope}")

    # ── Run state (main thread) ──────────────────────────────────────
    def _set_status(self, text: str) -> None:
        if text == self._sync_status and self._label_status.text() == text:
            return  # identical push: keep display echoes quiet
        self._sync_status = text
        try:
            self._widget_core.push_display("status", text)
        except RuntimeError:
            pass

    def _on_run_state(self, text: str) -> None:
        self._sync_status = text
        self._set_status(text)
        self._refresh_buttons()

    def _on_compute_error(self, message: object) -> None:
        self._sync_status = f"Failed: {message}"
        self._set_status(self._sync_status)
        self._refresh_buttons()

    def _on_progress_percent(self, percent: int) -> None:
        self._progress.setValue(max(0, min(100, percent)) * 10)

    def _on_progress_detail(self, detail: object) -> None:
        done, total = detail  # type: ignore[misc]
        now = time.monotonic()
        rate = ""
        if self._last_tick is not None:
            prev_t, prev_done = self._last_tick
            dt = now - prev_t
            if dt > 0.2 and done >= prev_done:
                rate = f" · {describe_size(int((done - prev_done) / dt))}/s"
        self._last_tick = (now, done)
        if total and now - self._last_push >= 1.0:
            self._last_push = now
            pct = int(done * 100 / total)
            self._set_status(
                f"Downloading {describe_size(done)} / {describe_size(total)} "
                f"({pct}%){rate}")
        elif not total:
            self._set_status(f"Downloading {describe_size(done)}{rate}")

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        try:
            self._widget_core.push_display("status", self._sync_status)
        except RuntimeError:
            pass
        self._refresh_buttons()

    # ── Buttons ──────────────────────────────────────────────────────
    def _dest_for(self, repo_id: str, revision: str, filename: str,
                  mode: str, path_input: str) -> Optional[str]:
        if mode == "custom":
            root = path_input.strip()
            if not root:
                return None
            safe = repo_id.replace("/", "--")
            return os.path.join(os.path.expanduser(root), safe, filename)
        root = path_input.strip() or find_cache_dir()[0]
        return os.path.join(root, "models--" + repo_id.replace("/", "--"),
                            "snapshots", revision, filename)

    def _selection(self) -> Optional[tuple]:
        """Current (url, dest, size, sha) or None when nothing to resume."""
        get = self._widget_core.get_port_value
        repo_id = (get("repo") or "").strip()
        if not repo_id:
            return None
        revision = (get("revision") or "").strip() or "main"
        pinned = self._pinned_rev if self._pinned_rev else revision
        want = (get("file") or "").strip()
        known = {f.name: f for f in self._files}
        if want and want in known:
            target = known[want]
        else:
            weights = [f for f in self._files
                       if f.name.lower().endswith(WEIGHT_SUFFIXES)]
            pool = sorted(weights, key=lambda f: -(f.size or 0)) or self._files
            target = pool[0] if pool else None
        if target is None:
            return None
        mode = get("target") or "hub"
        dest = self._dest_for(repo_id, pinned, target.name, mode,
                              get("target_folder") or "")
        if dest is None:
            return None
        return (resolve_url(repo_id, pinned, target.name), dest,
                target.size, target.sha256)

    def on_ui_change(self) -> None:
        # Sanctioned all-surfaces hook: canvas edits, mirror writes,
        # undo, restore and deserialization all land here (DEFAULT
        # policy has triggers_eval + user_triggers_eval). Recompute
        # enablement and the file view from the admitted values.
        super().on_ui_change()
        self._populate_file_combo()
        self._refresh_buttons()
        self._follow_result_pick()
        self._schedule_fetch()

    def _admit(self, port_name: str, value: str) -> None:
        """Publish a value only when it differs from the last publish.

        Every admit echoes through ``port_value_written`` into
        ``on_ui_change`` on every surface; with shared models each echo
        rebuilds views, which re-signal. Unconditional admits keep such
        rings spinning -- idempotent admits keep every ring one pass long.
        User edits bypass this record (the framework admits them); the
        next helper call then re-sends at most once and converges.
        """
        if value != self._last_admit.get(port_name):
            self._last_admit[port_name] = value
            self._widget_core.apply_port_value(
                port_name, value, user_edit=False)

    def _follow_result_pick(self) -> None:
        # Mirror picks arrive via bindings (node widget signals held),
        # so the results -> repo fill cannot ride currentIndexChanged.
        # A changed hit text means a pick (repopulates update the
        # baseline themselves); a repo that already matches ends the
        # chain, so this terminates on every path.
        res_text = self._combo_results.currentText()
        if res_text == self._last_results_text:
            return
        self._last_results_text = res_text
        picked = self._combo_results.itemData(
            self._combo_results.currentIndex())
        if not picked:
            return
        repo_text = (self._widget_core.get_port_value("repo") or "").strip()
        if str(picked) != repo_text:
            self._repo.setText(str(picked))
            self._admit("repo", str(picked))
            self._fetch_files()

    def _schedule_fetch(self) -> None:
        # Repo/revision typed anywhere (mirror included -- its line
        # edits never emit editingFinished node-side). Debounced so
        # keystrokes coalesce into one listing; the timer re-arms
        # while a download runs instead of dropping the fetch.
        repo = (self._widget_core.get_port_value("repo") or "").strip()
        rev = (self._widget_core.get_port_value("revision") or "").strip()
        rev = rev or "main"
        if (repo, rev) != self._last_fetch_key and not self._fetch_timer.isActive():
            if repo:
                self._fetch_timer.start(600)

    def _on_fetch_debounce(self) -> None:
        if self._is_computing:
            self._fetch_timer.start(600)
            return
        repo = (self._widget_core.get_port_value("repo") or "").strip()
        rev = (self._widget_core.get_port_value("revision") or "").strip()
        rev = rev or "main"
        if (repo, rev) == self._last_fetch_key:
            return
        self._fetch_files()

    def _refresh_buttons(self) -> None:
        running = bool(self._is_computing)
        has_repo = bool((self._widget_core.get_port_value("repo") or "").strip())
        resumable = False
        if not running and has_repo:
            sel = self._selection()
            if sel is not None:
                url, dest, _size, _sha = sel
                resumable = bool(resume_state(dest, url)["resumable"])
        mode = self._widget_core.get_port_value("target") or "hub"
        is_custom = mode == "custom"
        self._target_folder.setToolTip(
            "Hub cache root is autodetected -- switch to Custom folder "
            "to choose."
            if not is_custom else
            "Target folder (required). Recent folders are kept in the dropdown.")
        # set_port_enabled (not widget.setEnabled): WidgetCore's event
        # filter turns it into port_enabled_changed, which the canvas
        # bridge fans out to every surface (node form + mirror panels).
        # A bare widget.setEnabled only ever touches one instance.
        try:
            self._widget_core.set_port_enabled(
                "btn_download", not running and has_repo)
            self._widget_core.set_port_enabled(
                "btn_resume", not running and resumable)
            self._widget_core.set_port_enabled("btn_cancel", running)
            self._widget_core.set_port_enabled("btn_open", has_repo)
            self._widget_core.set_port_enabled(
                "target_folder", not running and is_custom)
        except RuntimeError:
            pass

    def _on_open_repo(self) -> None:
        repo_id = (self._widget_core.get_port_value("repo") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo_id):
            self._set_status("Open repo: enter an org/name id first.")
            return
        if not QDesktopServices.openUrl(QUrl(f"https://huggingface.co/{repo_id}")):
            self._set_status("Open repo: no browser answered.")

    # ── State: keep the folder history across saves ──────────────────
    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["target_history"] = self._target_folder.history()
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        with self._widget_core.suppress_signals():
            super().restore_state(state)
        self._target_folder.set_history(state.get("target_history", []))

    # ── Worker ───────────────────────────────────────────────────────
    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        repo_id = str(inputs.get("repo") or "").strip()
        info: Dict[str, Any] = {"repo": repo_id or None, "revision": None,
                                "files": [], "target": None, "bytes": 0}
        if not repo_id:
            self._sync_status = "Enter a repo id (org/name) or pick a search hit."
            return {"file_path": None, "download_info": info}
        revision = str(inputs.get("revision") or "").strip() or "main"
        cred_name = str(inputs.get("credential") or "").strip()
        token = resolve_credential(cred_name) if cred_name else None
        if cred_name and token is None:
            log.warning(f"Credential '{cred_name}' not found -- trying anonymous.")
        try:
            files, pinned = repo_files(repo_id, revision=revision, token=token)
        except HubError as exc:
            raise RuntimeError(str(exc)) from exc
        info["revision"] = pinned
        want_all = bool(inputs.get("download_all", False))
        include_config = bool(inputs.get("include_config", True))
        if want_all:
            queue = [f for f in files
                     if f.name.lower().endswith(WEIGHT_SUFFIXES)] or list(files)
        else:
            want = str(inputs.get("file") or "").strip()
            known = {f.name: f for f in files}
            if want and want in known:
                queue = [known[want]]
            else:
                weights = sorted(
                    [f for f in files if f.name.lower().endswith(WEIGHT_SUFFIXES)],
                    key=lambda f: -(f.size or 0))
                queue = [(weights or list(files))[0]] if files else []
        if not queue:
            self._sync_status = f"{repo_id}: no files to download."
            return {"file_path": None, "download_info": info}
        queue = with_companions(files, queue, include_config=include_config)
        mode = str(inputs.get("target") or "hub")
        path_input = str(inputs.get("target_folder") or "")
        info["target"] = "hub" if mode != "custom" else path_input
        if mode == "custom" and not path_input.strip():
            raise RuntimeError("Custom folder target needs a folder path.")
        if mode != "custom":
            root = path_input.strip() or find_cache_dir()[0]
            hub_repo_dir(root, repo_id)          # ensure layout dirs exist
        else:
            root = ""

        def progress(done: int, total: int) -> None:
            if total:
                self.report_progress(int(done * 100 / total))
                self.emit_stream("progress", done / total)
            self.progress_detail.emit(done, total)

        last: Optional[str] = None
        for i, target in enumerate(queue):
            url = resolve_url(repo_id, pinned, target.name)
            if mode != "custom":
                dest = os.path.join(
                    root, "models--" + repo_id.replace("/", "--"),
                    "snapshots", pinned, target.name)
            else:
                dest = os.path.join(os.path.expanduser(path_input.strip()),
                                    repo_id.replace("/", "--"), target.name)
            if os.path.isfile(dest) and target.size and \
                    os.path.getsize(dest) == target.size:
                info["files"].append({"name": target.name, "path": dest,
                                      "bytes": target.size, "skipped": True,
                                      "role": _role_of(target.name)})
                info["bytes"] += target.size or 0
                last = dest
                continue
            try:
                result = download_file(
                    url, dest, expected_size=target.size, sha256=target.sha256,
                    token=token, progress=progress,
                    is_cancelled=self.is_compute_cancelled)
            except HubError as exc:
                if exc.kind == "cancelled":
                    self._sync_status = (f"Cancelled at file {i + 1}/{len(queue)} "
                                         f"-- Resume to continue.")
                    return {"file_path": last, "download_info": info}
                raise RuntimeError(str(exc)) from exc
            if mode != "custom":
                repo_dir = os.path.dirname(os.path.dirname(
                    os.path.dirname(result["path"])))
                write_ref_main(repo_dir, pinned)
                if target.sha256:
                    link_blob(repo_dir, target.sha256, result["path"])
            info["files"].append({"name": target.name, "path": result["path"],
                                  "bytes": result["bytes"],
                                  "resumed": result["resumed"],
                                  "sha256_verified": result["sha256_verified"],
                                  "role": _role_of(target.name)})
            info["bytes"] += result["bytes"]
            last = result["path"]
        done = len([f for f in info["files"] if not f.get("skipped")])
        self._sync_status = (f"Downloaded {done} file(s), "
                             f"{describe_size(info['bytes'])} from {repo_id}.")
        return {"file_path": last, "download_info": info}

    def cleanup(self) -> None:
        self.cancel_compute()
        self._seq += 1                      # orphan in-flight search/listing threads
        try:
            self.search_done.disconnect()
            self.files_done.disconnect()
            self.progress_detail.disconnect()
        except (RuntimeError, TypeError):
            pass
        super().cleanup()
