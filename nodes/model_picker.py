# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

The model picker: a dropdown, a max spinner and a reduce slider.

Two nodes need exactly this surface -- the full Model Endpoint (which
owns its provider, URL and credential) and the pure Model node (which
takes all three through its ``endpoint`` input). One mixin so the two
never disagree about what "the model's window" means.

The provider rows (dropdown, base URL, credential name, per-provider
memory) are a second mixin below: the Model Endpoint and the Endpoint
node both own them, while the Model node takes them by wire.

Borrowed from the OpenCode nodes, minus the catalogue:

* **The model is a dropdown**, filled from the provider's remembered
  listing (``functions/endpoint_settings.py``). The Refresh button
  re-asks the provider's own ``/models`` off the GUI thread; switching
  providers re-lists automatically, so the dropdown is current without
  being asked. Editable, so a name the listing does not (yet) know is
  still usable, and a saved graph's model survives a listing that no
  longer names it.
* **The max spinner is the window**, auto-detected from the listing's
  ``context_length`` when the gateway says one, editable when it does
  not. A manual value is stored per model, so it survives a re-pick.
* **The slider can hold it lower.** Its top is the spinner; a value
  below it is stored per model and is the default the next time anyone
  picks that model. Fully right means "the maximum".

The mixin owns no provider: the node answers `_picker_provider_key`
(which settings file) and `_picker_source` (which URL and credential
name to list and connect through). It also owns `_handle` and
`_last_spec`, which a refresh invalidates, and an "Info:" display,
which the node pushes its connect status to.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Tuple

from PySide6.QtCore import QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QSlider, QSpinBox,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.library.sync_tool_button.widgets.sync_tool_button import SyncToolButton
from weave.logger import get_logger

from ..functions.credentials import resolve_credential
from ..functions.endpoint_settings import EndpointSettings
from ..functions.model_endpoint import PROVIDERS, provider_for
from ..functions.port_text import text_input

log = get_logger("SilkModelPicker")

#: The slider's range when the window is unknown. 0 stays "unknown"; the
#: top is generous rather than a claim about the model.
UNKNOWN_CONTEXT_MAX = 262_144
#: Below this a coding agent cannot hold its own system prompt.
MIN_CONTEXT = 4_096
CONTEXT_STEP = 1_024
#: The spinner's ceiling: 16M tokens, above every window shipped today.
SPINNER_CEILING = 16_000_000


def _tokens(value: int) -> str:
    if value <= 0:
        return "unknown"
    if value >= 1_048_576 and value % 1_048_576 == 0:
        return f"{value // 1_048_576}M"
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}k"
    return f"{value:,}"


class ModelPickerMixin:
    """The model dropdown + max spinner + reduce slider, shared.

    Contract with the node: `_picker_provider_key() -> str`,
    `_picker_source() -> (base_url, credential_name)`, attributes
    `_handle` / `_last_spec`, and an ``info`` display pushed through
    WidgetCore. `init_picker(form, widget_core)` builds the rows;
    `sync_picker()` fits the slider after a restore or a re-run;
    `cleanup_picker()` stops the timers.
    """

    #: (ok, provider-key, message), from the refresh thread.
    _refresh_ready = Signal(bool, str, str)

    _widget_core: WidgetCore

    # ── construction ─────────────────────────────────────────────────

    def init_picker(self, form: QFormLayout, wc: WidgetCore) -> None:
        # ── model + refresh ──
        model_row = QHBoxLayout()
        self._model = QComboBox()
        # Editable: a model added this morning is usable before anyone
        # refreshes, and a saved graph's model survives a listing that
        # no longer names it.
        self._model.setEditable(True)
        self._model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._model.setToolTip(
            "Models this provider listed last time anyone asked. Type a "
            "name to use one the listing does not name yet.")
        model_row.addWidget(self._model, stretch=1)
        self._btn_refresh = SyncToolButton(
            initial_text="Refresh", initial_icon="refresh",
            show_label=False, dimensions=24,
        )
        self._btn_refresh.set_tooltip(
            "Re-ask this provider's /models and remember the answer.")
        self._btn_refresh.clicked.connect(self._on_refresh)
        model_row.addWidget(self._btn_refresh)
        form.addRow("Model:", model_row)
        wc.register_widget(
            "model", self._model, role=PortRole.INPUT, datatype="string",
            default="", policy=debounced(400), add_to_layout=False,
        )
        self.add_input("model", datatype="string")

        # ── context max + reduce slider ──
        self._max = QSpinBox()
        self._max.setRange(0, SPINNER_CEILING)
        self._max.setSingleStep(CONTEXT_STEP)
        self._max.setSpecialValueText("auto")
        self._max.setToolTip(
            "The model's context window. Filled in from the listing when "
            "the provider advertises one; type it when it does not. A "
            "manual value is remembered for this model. The slider below "
            "can only reduce.")
        form.addRow("Max context:", self._max)
        wc.register_widget(
            "context_max", self._max, role=PortRole.INPUT,
            datatype="int", default=0, policy=debounced(400),
            add_to_layout=False,
        )
        self.add_input("context_max", datatype="int")

        ctx_row = QHBoxLayout()
        self._context = QSlider(Qt.Orientation.Horizontal)
        self._context.setSingleStep(CONTEXT_STEP)
        self._context.setPageStep(CONTEXT_STEP * 8)
        self._context.setToolTip(
            "Context window to use. The top is the max above; a lower "
            "value is remembered for this model. Fully right means "
            "\"the maximum\".")
        ctx_row.addWidget(self._context, stretch=1)
        self._context_label = QLabel("unknown")
        self._context_label.setMinimumWidth(64)
        self._context_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ctx_row.addWidget(self._context_label)
        form.addRow("Context:", ctx_row)
        # The widest range any listing knew, so a restored value is not
        # clamped before the restored model has set the real range.
        self._context.setRange(0, UNKNOWN_CONTEXT_MAX)
        wc.register_widget(
            "context_length", self._context, role=PortRole.INPUT,
            datatype="int", default=0, policy=debounced(300),
            add_to_layout=False,
        )
        self.add_input("context_length", datatype="int")
        wc.register_widget(
            "context_label", self._context_label, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self._native_tools = QCheckBox("Native tool calling")
        self._native_tools.setToolTip(
            "Send tool schemas in the request. Set from the listing when "
            "you pick a model; untick it to use fenced tool calls instead, "
            "which work with any model.")
        form.addRow("", self._native_tools)
        wc.register_widget(
            "supports_tools", self._native_tools, role=PortRole.INPUT,
            datatype="bool", default=False, add_to_layout=False,
        )
        self.add_input("supports_tools", datatype="bool")

        self._label_models = QLabel("")
        self._label_models.setWordWrap(True)
        form.addRow("Models:", self._label_models)
        wc.register_widget(
            "models_status", self._label_models, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        # Persisting a dragged slider (or a typed max) on every tick
        # would be a file write per pixel; once it settles is enough.
        self._persist_timer = QTimer(wc)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(600)
        self._persist_timer.timeout.connect(self._persist_choices)

        self._picker: Dict[str, Any] = {"key": "", "settings": None}
        self._refreshing = False

        self._model.currentTextChanged.connect(self._on_model_changed)
        self._max.valueChanged.connect(self._on_max_changed)
        self._context.valueChanged.connect(self._on_context_changed)
        self._refresh_ready.connect(self._apply_refresh)

        self.refresh_picker(auto=True)

    def cleanup_picker(self) -> None:
        self._persist_timer.stop()
        try:
            self._refresh_ready.disconnect(self._apply_refresh)
        except (RuntimeError, TypeError):
            pass

    # ── the node answers these ───────────────────────────────────────

    def _picker_provider_key(self) -> str:
        raise NotImplementedError

    def _picker_source(self) -> Tuple[str, str]:
        """(base_url, credential name) to list and connect through."""
        raise NotImplementedError

    # ── settings, one object per provider key ────────────────────────

    def _picker_settings(self) -> EndpointSettings:
        key = self._picker_provider_key()
        current = self._picker.get("settings")
        if current is None or self._picker.get("key") != key:
            current = EndpointSettings(key)
            self._picker = {"key": key, "settings": current}
        return current

    # ── main thread: the dropdown, the spinner, the slider ───────────

    def _current_model(self) -> str:
        return self._model.currentText().strip()

    def _fill_models(self, prefer: str = "") -> None:
        """Rebuild the dropdown, keeping what is shown if it can."""
        settings = self._picker_settings()
        wanted = self._current_model() or prefer or settings.selected
        ids = settings.model_ids()
        blocked = self._model.blockSignals(True)
        try:
            self._model.clear()
            for model_id in ids:
                self._model.addItem(model_id)
                entry = settings.model(model_id)
                if entry is not None and entry.context_max:
                    self._model.setItemData(
                        self._model.count() - 1,
                        f"context {_tokens(entry.context_max)}",
                        Qt.ItemDataRole.ToolTipRole)
            if wanted:
                index = self._model.findText(wanted)
                if index >= 0:
                    self._model.setCurrentIndex(index)
                else:
                    self._model.setEditText(wanted)
            elif ids:
                self._model.setCurrentIndex(0)
        finally:
            self._model.blockSignals(blocked)

    def _effective_max(self) -> int:
        """The ceiling the slider obeys: the spinner, else the listing."""
        if self._max.value() > 0:
            return self._max.value()
        return self._picker_settings().max_for(self._current_model())

    def _sync_context_range(self, reset_value: bool) -> None:
        """Fit the slider to the model shown.

        ``reset_value`` is for a model the person just picked: the slider
        jumps to that model's stored choice, or its maximum. Otherwise
        the value is only clamped into the new range -- a restored graph
        keeps what it saved.
        """
        maximum = self._effective_max()
        if maximum:
            low, high = min(MIN_CONTEXT, maximum), maximum
        else:
            low, high = 0, UNKNOWN_CONTEXT_MAX
        value = self._context.value()
        if reset_value:
            value = self._picker_settings().context_for(self._current_model())
            if maximum and not value:
                value = maximum
        elif maximum and value <= 0:
            # Was unknown, is now known: "unknown" meant "whatever it is".
            value = maximum
        value = max(low, min(high, value))
        if (self._context.minimum(), self._context.maximum()) != (low, high) \
                or value != self._context.value():
            # Not blocked: the new value must reach WidgetCore and the
            # port, or the handle downstream keeps the old window.
            self._context.setRange(low, high)
            self._context.setValue(value)
        self._push_context_label()

    def _sync_max_from_listing(self) -> None:
        """The spinner shows the stored max, or the detected one, or auto."""
        stored = self._picker_settings().max_for(self._current_model())
        if self._max.value() != stored:
            blocked = self._max.blockSignals(True)
            try:
                self._max.setValue(stored)
            finally:
                self._max.blockSignals(blocked)

    def _push_context_label(self) -> None:
        value = self._context.value()
        maximum = self._effective_max()
        if not maximum:
            text = _tokens(value) if value else "unknown"
        elif value >= maximum:
            text = f"{_tokens(value)} (max)"
        else:
            text = f"{_tokens(value)} / {_tokens(maximum)}"
        self._widget_core.push_display("context_label", text)

    def _push_models_status(self, extra: str = "") -> None:
        settings = self._picker_settings()
        if settings.models:
            text = (f"{len(settings.models)} models · {settings.describe_age()} · "
                    f"{settings.source or self._picker_provider_key()}")
        else:
            text = "No listing yet: press Refresh."
        if extra:
            text = f"{extra}\n{text}"
        self._widget_core.push_display("models_status", text)

    def _sync_tools_from_listing(self) -> None:
        """Tick the box the way the listing answers for the model shown.

        Only on a pick: afterwards the box is the person's, and an untick
        means fences. A model the listing says nothing about leaves the
        box as it is. On a restore the saved value lands after the
        model's, so it wins.
        """
        answer = self._picker_settings().tool_call(self._current_model())
        if answer is None:
            return
        if self._native_tools.isChecked() != answer:
            # Programmatic, never user_edit=True from node code (WV403).
            self._widget_core.apply_port_value("supports_tools", answer)

    def sync_picker(self) -> None:
        """Fit the slider after a restore or a re-run (main thread)."""
        self._sync_max_from_listing()
        self._sync_context_range(reset_value=False)

    @Slot(str)
    def _on_model_changed(self, _text: str) -> None:
        self._sync_max_from_listing()
        self._sync_context_range(reset_value=True)
        self._sync_tools_from_listing()
        model_id = self._current_model()
        if model_id in self._picker_settings().models:
            self._picker_settings().set_selected(model_id)

    @Slot(int)
    def _on_max_changed(self, _value: int) -> None:
        self._sync_context_range(reset_value=False)
        self._persist_timer.start()

    @Slot(int)
    def _on_context_changed(self, _value: int) -> None:
        self._push_context_label()
        self._persist_timer.start()

    @Slot()
    def _persist_choices(self) -> None:
        model_id = self._current_model()
        settings = self._picker_settings()
        if model_id in settings.models:
            settings.set_max(model_id, self._max.value())
            settings.set_context(model_id, self._context.value())

    # ── refresh: off the GUI thread ──────────────────────────────────

    def refresh_picker(self, auto: bool = False) -> None:
        """Refill from the remembered listing, then re-list in the back.

        ``auto`` is the provider-switch path: the cached list shows
        instantly, and the network updates it when it answers. The button
        calls this too, explicitly.
        """
        self._ensure_picker_key()
        self._fill_models()
        self._sync_max_from_listing()
        self._sync_context_range(reset_value=False)
        self._push_models_status()
        base_url, _credential = self._picker_source()
        if base_url:
            self._on_refresh()
        elif auto:
            self._push_models_status("No URL yet.")

    def _ensure_picker_key(self) -> None:
        """Drop a stale settings object after the provider key moved."""
        if self._picker.get("key") != self._picker_provider_key():
            self._picker = {"key": "", "settings": None}
            self._picker_settings()

    @Slot()
    def _on_refresh(self) -> None:
        if self._refreshing:
            return
        key = self._picker_provider_key()
        base_url, credential = self._picker_source()
        if not base_url:
            self._push_models_status("No URL yet: set one before refreshing.")
            return
        self._refreshing = True
        self._btn_refresh.setEnabled(False)
        self._push_models_status("Refreshing…")
        headers: Dict[str, str] = {}
        secret = resolve_credential(credential) if credential else None
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        threading.Thread(
            target=self._refresh_worker, args=(key, base_url, headers),
            daemon=True, name=f"silk-endpoint-{key}",
        ).start()

    def _refresh_worker(self, key: str, base_url: str,
                        headers: Dict[str, str]) -> None:
        """Daemon thread: the listing request and the file write."""
        # A fresh instance, swapped in on the GUI thread: the worker that
        # runs `compute` may be reading the current one right now.
        settings = EndpointSettings(key)
        try:
            message = settings.refresh(base_url, headers=headers)
        except Exception as exc:  # noqa: BLE001 - reported on the node
            self._refresh_ready.emit(False, key, str(exc))
            return
        self._refresh_ready.emit(True, key, message)

    @Slot(bool, str, str)
    def _apply_refresh(self, ok: bool, key: str, message: str) -> None:
        self._refreshing = False
        self._btn_refresh.setEnabled(True)
        if not ok:
            log.warning(f"Endpoint listing refresh failed — {message}")
            self._push_models_status(f"Refresh failed: {message}")
            return
        if key != self._picker_provider_key():
            # Switched away mid-flight: the file is updated anyway, and
            # the current provider refreshes on its own turn.
            return
        self._picker = {"key": "", "settings": None}
        before = self._current_model()
        self._fill_models(self._picker_settings().selected)
        self._push_models_status(message)
        log.info(f"Endpoint listing — {message}")
        after = self._current_model()
        if after != before:
            # The dropdown went from empty to its first entry with
            # signals blocked; a programmatic write tells WidgetCore, and
            # its port_value_written schedules the re-run (never
            # user_edit=True from node code, WV403).
            self._widget_core.apply_port_value("model", after)
            self._sync_max_from_listing()
            self._sync_context_range(reset_value=True)
            self._sync_tools_from_listing()
        else:
            self._sync_max_from_listing()
            self._sync_context_range(reset_value=False)
        # A new listing may change the window of the same model name.
        self._handle = None
        self._last_spec = ()

    # ── worker thread: what compute needs ────────────────────────────

    @staticmethod
    def picker_spec_from(inputs: Dict[str, Any]) -> tuple:
        """(model, max, length, tools) from the picker's ports."""
        return (
            text_input(inputs, "model"),
            int(inputs.get("context_max") or 0),
            int(inputs.get("context_length") or 0),
            bool(inputs.get("supports_tools")),
        )

    def clamp_length(self, model: str, length: int, limit: int = 0) -> int:
        """The length to connect with: a wired value clamped to the max.

        ``limit`` is the spinner when set, else the listing's max; 0 is
        unknown and clamps nothing -- ``connect`` then asks the gateway.
        """
        ceiling = limit or self._picker_settings().max_for(model)
        if ceiling and length > ceiling:
            return ceiling
        return length


class ProviderConfigMixin:
    """The provider rows: dropdown, base URL, credential name, memory.

    Two nodes own these -- the Model Endpoint and the Endpoint node --
    so neither reimplements the snapshot-before-switch that keeps a
    typed URL when providers are flipped through and back. The node
    connects its own `_on_provider_changed` slot and calls
    `_provider_changed_fill()` from it, then whatever follows (the
    Model Endpoint re-lists; the Endpoint node has nothing to re-list).
    """

    _widget_core: WidgetCore

    def init_provider_config(self, form: QFormLayout, wc: WidgetCore) -> None:
        self._provider = QComboBox()
        for preset in PROVIDERS:
            self._provider.addItem(preset.label, userData=preset.key)
        self._provider.setToolTip(
            "A preset fills the URL and the usual credential name. Both "
            "stay editable per provider -- the useful endpoint is often "
            "your own proxy."
        )
        form.addRow("Provider:", self._provider)
        wc.register_widget(
            "provider", self._provider, role=PortRole.INPUT,
            datatype="string", default=PROVIDERS[0].key, add_to_layout=False,
        )
        self.add_input("provider", datatype="string")

        self._base_url = QLineEdit("")
        self._base_url.setPlaceholderText("http://localhost:1234/v1")
        self._base_url.setToolTip(
            "Remembered per provider: switching away and back restores "
            "what you typed.")
        form.addRow("Base URL:", self._base_url)
        wc.register_widget(
            "base_url", self._base_url, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(500),
            add_to_layout=False,
        )
        self.add_input("base_url", datatype="string")

        self._credential = QLineEdit("")
        self._credential.setPlaceholderText(
            "OPENROUTER_API_KEY (a name, never a value)")
        self._credential.setToolTip(
            "The NAME of an environment variable, or of an entry in "
            "~/.weave/silk/secrets.json. Remembered per provider. The "
            "value is read at connect time and never saved in the graph "
            "(D22)."
        )
        form.addRow("Credential:", self._credential)
        wc.register_widget(
            "credential", self._credential, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(500),
            add_to_layout=False,
        )
        self.add_input("credential", datatype="string")

        # An edit is the person's; a programmatic fill is the provider's.
        # Only edits are persisted, on a timer so typing is one write.
        self._edit_timer = QTimer(wc)
        self._edit_timer.setSingleShot(True)
        self._edit_timer.setInterval(600)
        self._edit_timer.timeout.connect(self._persist_provider_edits)
        self._base_url.textEdited.connect(
            lambda _text: self._edit_timer.start())
        self._credential.textEdited.connect(
            lambda _text: self._edit_timer.start())

        self._provider_key: str = ""

    @staticmethod
    def provider_spec_from(inputs: Dict[str, Any]) -> tuple:
        """(provider, base URL, credential name) from the provider rows."""
        return (
            text_input(inputs, "provider", "custom"),
            text_input(inputs, "base_url"),
            text_input(inputs, "credential"),
        )

    def _provider_changed_fill(self) -> str:
        """Snapshot the rows being left; fill the rows being shown.

        Returns the new provider key. Fill order is stored edit, else
        the preset default -- never the other provider's leftovers.
        """
        key = str(self._provider.currentData() or "custom")
        if self._provider_key and self._provider_key != key:
            old = EndpointSettings(self._provider_key)
            old.set_base_url(self._base_url.text())
            old.set_credential(self._credential.text())
        self._provider_key = key
        settings = EndpointSettings(key)
        preset = provider_for(key)
        url = settings.base_url or preset.base_url
        cred = settings.credential or preset.credential
        if self._base_url.text() != url:
            self._base_url.setText(url)
        if self._credential.text() != cred:
            self._credential.setText(cred)
        return key

    def _persist_provider_edits(self) -> None:
        settings = EndpointSettings(
            self._provider_key or str(self._provider.currentData() or "custom"))
        settings.set_base_url(self._base_url.text())
        settings.set_credential(self._credential.text())
