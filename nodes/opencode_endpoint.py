# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Silk OpenCode endpoints -- OpenCode Zen and OpenCode Go as model handles.

The generic Model Endpoint takes any URL and asks nothing of it until
connect time. These two know their provider, so they can do more:

* **The model is a dropdown**, filled from a catalogue file the node
  reads when it opens (``~/.weave/silk/model_catalogs/<plan>.json``). The
  Refresh button re-asks OpenCode's ``/v1/models`` and models.dev, off the
  GUI thread. The first node of a kind with no file refreshes once by
  itself, so there is something to pick.
* **The context length sets itself**, from the catalogue's ``limit.context``
  for the chosen model. OpenCode's own listing does not carry it; models.dev
  does (see `functions/model_catalog.py`).
* **The slider can hold it lower.** Its maximum is the model's window; a
  value below it is stored per model in the same file, so it survives a
  refresh and is the default the next time anyone picks that model.

Both emit the same ``model_handle`` as the GGUF Loader and the Model
Endpoint, so everything downstream takes them unchanged.
"""

import threading
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtCore import QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QSlider,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger
from weave.library.sync_tool_button.widgets.sync_tool_button import SyncToolButton

from .silk_ports import MODEL_HANDLE_TYPE  # noqa: F401
from ..functions.credentials import resolve_credential
from ..functions.model_catalog import ModelCatalog, catalog_for
from ..functions.model_endpoint import connect
from ..functions.port_text import text_input

log = get_logger("SilkOpenCodeNode")

#: The slider's range when the window is unknown. 0 stays "unknown"; the
#: top is generous rather than a claim about the model.
UNKNOWN_CONTEXT_MAX = 262_144
#: Below this a coding agent cannot hold its own system prompt.
MIN_CONTEXT = 4_096
CONTEXT_STEP = 1_024


def _tokens(value: int) -> str:
    if value <= 0:
        return "unknown"
    if value >= 1_048_576 and value % 1_048_576 == 0:
        return f"{value // 1_048_576}M"
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}k"
    return f"{value:,}"


class _OpenCodeEndpointBase(ActiveNode):
    """Shared body; the subclasses only name their catalogue."""

    _widget_core: WidgetCore

    #: Key into `functions.model_catalog.CATALOGS`.
    catalog_key: ClassVar[str] = ""

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    node_icon: ClassVar[Optional[str]] = "node"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1
    node_version = 1     # bump on any state-shape change (G20)

    #: (ok, catalogue-or-None, message), from the refresh thread.
    _catalog_ready = Signal(bool, object, str)

    #: One automatic first refresh per catalogue per process, not per node:
    #: ten nodes dropped on an empty machine are one request, not ten.
    _auto_refreshed: ClassVar[set] = set()

    def __init__(self, title: str = "OpenCode", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self._spec_def = catalog_for(self.catalog_key)
        # Local file read: cheap, and done before the state restore so a
        # saved model name finds its row in the dropdown.
        self._catalog = ModelCatalog(self._spec_def)

        self.add_output("model_obj", datatype="model_handle")

        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        # ── model + refresh ──
        model_row = QHBoxLayout()
        self._model = QComboBox()
        # Editable: a model OpenCode added this morning is usable before
        # anyone refreshes, and a saved graph's model survives a catalogue
        # that no longer lists it.
        self._model.setEditable(True)
        self._model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._model.setToolTip(
            "Models from the catalogue file. Type a name to use one the "
            "catalogue does not list yet.")
        model_row.addWidget(self._model, stretch=1)
        self._btn_refresh = SyncToolButton(
            initial_text="Refresh", initial_icon="refresh",
            show_label=False, dimensions=24,
        )
        self._btn_refresh.set_tooltip(
            f"Re-fetch the {self._spec_def.label} model list and context "
            f"lengths, and rewrite {self._catalog.path.name}.")
        self._btn_refresh.clicked.connect(self._on_refresh)
        model_row.addWidget(self._btn_refresh)
        form.addRow("Model:", model_row)
        self._widget_core.register_widget(
            "model", self._model, role=PortRole.INPUT, datatype="string",
            default="", policy=debounced(400), add_to_layout=False,
        )
        self.add_input("model", datatype="string")

        # ── context slider ──
        ctx_row = QHBoxLayout()
        self._context = QSlider(Qt.Orientation.Horizontal)
        self._context.setSingleStep(CONTEXT_STEP)
        self._context.setPageStep(CONTEXT_STEP * 8)
        self._context.setToolTip(
            "Context window to use. The maximum is the model's own window; "
            "a lower value is remembered for this model in the catalogue "
            "file. Fully right means \"the model's maximum\".")
        ctx_row.addWidget(self._context, stretch=1)
        self._context_label = QLabel("unknown")
        self._context_label.setMinimumWidth(64)
        self._context_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ctx_row.addWidget(self._context_label)
        form.addRow("Context:", ctx_row)
        # The widest range the catalogue knows, so a restored value is not
        # clamped before the restored model has set the real range.
        widest = max((m.context_max for m in self._catalog.models.values()),
                     default=0)
        self._context.setRange(0, max(widest, UNKNOWN_CONTEXT_MAX))
        self._widget_core.register_widget(
            "context_length", self._context, role=PortRole.INPUT,
            datatype="int", default=0, policy=debounced(300),
            add_to_layout=False,
        )
        self.add_input("context_length", datatype="int")
        self._widget_core.register_widget(
            "context_label", self._context_label, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        # ── credential ──
        self._credential = QLineEdit(self._spec_def.credential)
        self._credential.setPlaceholderText(
            f"{self._spec_def.credential} (a name, never a value)")
        self._credential.setToolTip(
            "The NAME of an environment variable, or of an entry in "
            "~/.weave/silk/secrets.json. The value is read at connect time "
            "and never saved in the graph (D22).")
        form.addRow("Credential:", self._credential)
        self._widget_core.register_widget(
            "credential", self._credential, role=PortRole.INPUT,
            datatype="string", default=self._spec_def.credential,
            policy=debounced(500), add_to_layout=False,
        )
        self.add_input("credential", datatype="string")

        self._native_tools = QCheckBox("Native tool calling")
        self._native_tools.setToolTip(
            "Send tool schemas in the request. Turned on automatically when "
            "the catalogue says the model calls tools; tick it to force it.")
        form.addRow("", self._native_tools)
        self._widget_core.register_widget(
            "supports_tools", self._native_tools, role=PortRole.INPUT,
            datatype="bool", default=False, add_to_layout=False,
        )
        self.add_input("supports_tools", datatype="bool")

        self._label_catalog = QLabel("")
        self._label_catalog.setWordWrap(True)
        form.addRow("Catalogue:", self._label_catalog)
        self._widget_core.register_widget(
            "catalog_status", self._label_catalog, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self._label_status = QLabel("Not connected.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self._handle: Optional[Dict[str, Any]] = None
        self._last_spec: tuple = ()
        self._sync_status: str = "Not connected."
        self._refreshing = False

        # Persisting a dragged slider on every tick would be a file write
        # per pixel; once it settles is enough.
        self._persist_timer = QTimer(self._widget_core)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(600)
        self._persist_timer.timeout.connect(self._persist_context)

        self._fill_models(self._catalog.selected)
        self._sync_context_range(reset_value=True)
        self._push_catalog_status()

        self._model.currentTextChanged.connect(self._on_model_changed)
        self._context.valueChanged.connect(self._on_context_changed)
        self._catalog_ready.connect(self._apply_refresh)
        # A graph restore, or a wire, writes the model with signals blocked;
        # the slider still has to fit the model it now names.
        self._widget_core.port_value_written.connect(self._on_port_written)

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

        if not self._catalog.models and self.catalog_key not in self._auto_refreshed:
            type(self)._auto_refreshed.add(self.catalog_key)
            QTimer.singleShot(0, self._on_refresh)

    # ── Main thread: the dropdown and the slider ─────────────────────

    def _current_model(self) -> str:
        return self._model.currentText().strip()

    def _fill_models(self, prefer: str = "") -> None:
        """Rebuild the dropdown, keeping what is shown if it can."""
        wanted = self._current_model() or prefer
        ids = self._catalog.model_ids()
        blocked = self._model.blockSignals(True)
        try:
            self._model.clear()
            for model_id in ids:
                entry = self._catalog.model(model_id)
                self._model.addItem(model_id)
                if entry is not None:
                    tip = entry.name or model_id
                    tip += f" · context {_tokens(entry.context_max)}"
                    if not entry.speaks_chat:
                        tip += f" · documented on /{entry.api}"
                    self._model.setItemData(
                        self._model.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
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

    def _sync_context_range(self, reset_value: bool) -> None:
        """Fit the slider to the model shown.

        ``reset_value`` is for a model the person just picked: the slider
        jumps to that model's stored choice, or its maximum. Otherwise the
        value is only clamped into the new range -- a restored graph keeps
        what it saved.
        """
        model_id = self._current_model()
        maximum = self._catalog.context_max(model_id)
        if maximum:
            low, high = min(MIN_CONTEXT, maximum), maximum
        else:
            low, high = 0, UNKNOWN_CONTEXT_MAX
        value = self._context.value()
        if reset_value:
            value = self._catalog.context_for(model_id)
        elif maximum and value <= 0:
            # Was unknown, is now known: "unknown" meant "whatever it is".
            value = self._catalog.context_for(model_id)
        value = max(low, min(high, value))
        if (self._context.minimum(), self._context.maximum()) != (low, high) \
                or value != self._context.value():
            # Not blocked: the new value must reach WidgetCore and the
            # port, or the handle downstream keeps the old window.
            self._context.setRange(low, high)
            self._context.setValue(value)
        self._push_context_label()

    def _push_context_label(self) -> None:
        value = self._context.value()
        maximum = self._catalog.context_max(self._current_model())
        if not maximum:
            text = _tokens(value) if value else "unknown"
        elif value >= maximum:
            text = f"{_tokens(value)} (max)"
        else:
            text = f"{_tokens(value)} / {_tokens(maximum)}"
        self._widget_core.push_display("context_label", text)

    def _push_catalog_status(self, extra: str = "") -> None:
        cat = self._catalog
        if cat.models:
            text = (f"{len(cat.models)} models · {cat.describe_age()} · "
                    f"{cat.source or 'cached'}")
        else:
            text = "No catalogue yet: press Refresh."
        if extra:
            text = f"{extra}\n{text}"
        self._widget_core.push_display("catalog_status", text)

    @Slot(str)
    def _on_model_changed(self, _text: str) -> None:
        self._sync_context_range(reset_value=True)
        model_id = self._current_model()
        if model_id in self._catalog.models:
            self._catalog.set_selected(model_id)

    @Slot(str, object)
    def _on_port_written(self, port: str, _value: Any) -> None:
        if port == "model":
            self._sync_context_range(reset_value=False)
        elif port == "context_length":
            self._push_context_label()

    @Slot(int)
    def _on_context_changed(self, _value: int) -> None:
        self._push_context_label()
        self._persist_timer.start()

    @Slot()
    def _persist_context(self) -> None:
        model_id = self._current_model()
        if model_id in self._catalog.models:
            self._catalog.set_context(model_id, self._context.value())

    # ── Refresh: off the GUI thread ──────────────────────────────────

    @Slot()
    def _on_refresh(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        self._btn_refresh.setEnabled(False)
        self._push_catalog_status("Refreshing…")
        credential = self._credential.text().strip()
        threading.Thread(
            target=self._refresh_worker, args=(credential,),
            daemon=True, name=f"silk-{self.catalog_key}-catalog",
        ).start()

    def _refresh_worker(self, credential: str) -> None:
        """Daemon thread: both network requests and the file write."""
        headers: Dict[str, str] = {}
        secret = resolve_credential(credential) if credential else None
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        # A fresh instance, swapped in on the GUI thread: the worker that
        # runs `compute` may be reading the current one right now.
        catalog = ModelCatalog(self._spec_def, self._catalog.path.parent)
        try:
            message = catalog.refresh(headers=headers)
        except Exception as exc:  # noqa: BLE001 - reported on the node
            self._catalog_ready.emit(False, None, str(exc))
            return
        if not secret:
            message += (f" · no key under '{credential or '(none)'}'"
                        if credential else "")
        self._catalog_ready.emit(True, catalog, message)

    @Slot(bool, object, str)
    def _apply_refresh(self, ok: bool, catalog: Any, message: str) -> None:
        self._refreshing = False
        self._btn_refresh.setEnabled(True)
        if not ok or catalog is None:
            log.warning(f"{self._spec_def.label} catalogue refresh failed — {message}")
            self._push_catalog_status(f"Refresh failed: {message}")
            return
        before = self._current_model()
        self._catalog = catalog
        self._fill_models(catalog.selected)
        self._push_catalog_status(message)
        log.info(f"{self._spec_def.label} catalogue — {message}")
        after = self._current_model()
        if after != before:
            # The dropdown went from empty to its first entry with signals
            # blocked; tell WidgetCore, as an edit, so the graph re-runs.
            self._widget_core.apply_port_value("model", after, user_edit=True)
            self._sync_context_range(reset_value=True)
        else:
            self._sync_context_range(reset_value=False)
        # A new catalogue may change the window of the same model name.
        self._last_spec = ()

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._widget_core.push_display("status", self._sync_status)

    # ── Worker thread ────────────────────────────────────────────────

    @staticmethod
    def spec_from(inputs: Dict[str, Any]) -> tuple:
        return (
            text_input(inputs, "model"),
            text_input(inputs, "credential"),
            int(inputs.get("context_length") or 0),
            bool(inputs.get("supports_tools")),
        )

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            model, credential, context_length, native = self.spec_from(inputs)
        except ValueError as exc:
            self._handle = None
            self._last_spec = ()
            self._sync_status = f"Invalid input: {exc}."
            return {"model_obj": None}

        catalog = self._catalog
        entry = catalog.model(model)
        maximum = entry.context_max if entry else 0
        if maximum and context_length > maximum:
            # A wired-in value can exceed the slider; the window cannot.
            context_length = maximum
        if entry is not None and entry.tool_call and not native:
            native = True

        spec = (model, credential, context_length, native)
        if self._handle is not None and spec == self._last_spec:
            return {"model_obj": self._handle}

        if not model:
            self._handle = None
            self._last_spec = spec
            self._sync_status = ("Pick a model." if catalog.models else
                                 "No models yet: press Refresh.")
            return {"model_obj": None}

        handle, status, _models = connect(
            self._spec_def.base_url, model, credential=credential,
            context_length=context_length, provider=self.catalog_key,
            supports_tools=native,
        )
        if handle is not None:
            handle["provider"] = self.catalog_key
            if (self._spec_def.metered and "pricing" not in handle
                    and entry is not None and entry.cost_input is not None
                    and entry.cost_output is not None):
                # models.dev quotes USD per million tokens.
                handle["pricing"] = {
                    "input": entry.cost_input / 1e6,
                    "output": entry.cost_output / 1e6,
                    "currency": "USD",
                    "source": "models.dev",
                }
            if entry is not None and not entry.speaks_chat:
                status += (f"  ·  ⚠ OpenCode documents this model on "
                           f"/{entry.api}; Silk sends /chat/completions, "
                           f"which may be refused")
            if entry is None and catalog.models:
                status += "  ·  not in the catalogue (typed by hand)"
        self._handle = handle
        self._last_spec = spec
        self._sync_status = status
        if handle is None:
            log.warning(f"{self._spec_def.label} not connected — {status}")
        else:
            log.info(f"{self._spec_def.label} ready — {status}")
        return {"model_obj": handle}

    def cleanup(self) -> None:
        self._handle = None
        super().cleanup()


@register_node
class SilkOpenCodeZenNode(_OpenCodeEndpointBase):
    """OpenCode Zen (pay as you go), as a model handle."""

    catalog_key: ClassVar[str] = "opencode"
    node_name: ClassVar[Optional[str]] = "OpenCode Zen"
    node_description: ClassVar[Optional[str]] = (
        "OpenCode Zen's curated models, picked from a cached list with "
        "their context windows filled in."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "llm", "opencode", "zen", "remote", "inference",
    ]

    def __init__(self, title: str = "OpenCode Zen", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)


@register_node
class SilkOpenCodeGoNode(_OpenCodeEndpointBase):
    """OpenCode Go (subscription), as a model handle."""

    catalog_key: ClassVar[str] = "opencode-go"
    node_name: ClassVar[Optional[str]] = "OpenCode Go"
    node_description: ClassVar[Optional[str]] = (
        "OpenCode Go's subscription models, picked from a cached list with "
        "their context windows filled in."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "llm", "opencode", "go", "remote", "inference",
    ]

    def __init__(self, title: str = "OpenCode Go", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
