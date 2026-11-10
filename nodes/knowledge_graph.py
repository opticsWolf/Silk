# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Silk Knowledge Graph Node.

Owns **one served OKFgraph knowledge graph** and hands it downstream as a
handle, the way the MCP node owns a session (D19): the node starts
``okf serve`` for the database (or attaches to the one already serving
it), every ToolBox wired to it registers its tools over that one process,
and removing the node stops what it started. ``functions/okf_graph.py``
has the reasons OKFgraph never runs inside Weave's process.

**Device and model are the graph's, not the call's.** OKFgraph pins a
graph's vectors to the model and precision they were made with (CPU means
FP32, a GPU FP16) and refuses to mix them. So the node reads the pin on
every evaluation and says when the settings would break it:

* the graph was already embedded some other way (the CLI, another harness,
  an older setting) -- a warning, and the **Re-embed** button;
* the wired model was changed, on a graph that matched the old one --
  the graph is re-embedded with the new model right away, since that
  is what changing it means. OKFgraph keeps the database file as a
  backup until the rebuild succeeds.

A server someone else started keeps its own settings: the node attaches,
says so, and never re-embeds automatically on its behalf.

**Threaded, and cancellable.** Starting ``okf serve`` takes seconds and a
re-embed can take hours, so compute runs on a worker. The re-embed rides
OKFgraph's job lane (0.12.1): progress is shown as it goes, and pressing
the button again while it runs stops the job at its next batch boundary.
A stopped re-embed leaves the old pin in place and is not restarted on
its own; pressing Re-embed continues it.

**Where the graph lives.** With ``roots`` wired (the same Folder / Folder
List the ToolBox takes), the node looks in every root for OKFgraph's
``okfgraph.db`` and for databases in ``<root>/.okfgraph/``. An empty
database field takes the first one found -- or, with none, a new
``<first root>/.okfgraph/okfgraph.db`` -- and the rest wait in the
dropdown. Bundles default to ``.okfgraph/bundle`` beside the database.

**Models on wires.** The node has no model fields: the models come from
loader nodes, and the Embedder row picks which wire embeds.

* ``embedder`` -- the Embedder Loader's ``model_obj`` (a GGUF model on
  Silk's llama.cpp server). okf embeds through that server on its
  external lane. Everything okf must be told about the model is read
  from the GGUF file, never typed: the family (``nomic-v1.5`` when it is
  one, which brings okf's prefixes; otherwise the model's own name), the
  pooling the server applies (``<arch>.pooling_type``) and the width
  (``<arch>.embedding_length``). The query/document prefixes come from
  the model's name (``functions/embed_prefixes.py``) unless the loader
  overrides them, and are declared to okf. The pin is
  ``gguf/<file stem>``, plus ``+p<6 hex>`` naming a declared table, so a
  graph is never searched with prefixes its vectors were made without;
  one embedded before Silk declared them is served without them until
  Re-embed. Tokens are counted by the server's ``/tokenize``.
* ``onnx_embedder`` -- Silk KG Embedder (ONNX)'s ``embedder_spec``, which
  okf loads with embroider in its own process (device, precision, cache
  folder). The ONNX Loader's embedder is refused here: a graph's
  vectors are okf's own.
* ``reranker`` -- Silk KG Reranker (ONNX)'s cross-encoder, or the Reranker
  Loader's GGUF reranker or the ONNX Loader's embroider one
  (Silk's own ``/v1/rerank`` servers, which okf reaches on its external
  rerank lane, counting pairs with the server's ``/tokenize``), or any
  external host described on that port. Rerank
  writes no vectors, so changing it restarts the server and never
  re-embeds.

With no embedder wired the graph keeps the model it was last embedded
with -- unplugging a loader is not a request to re-embed. A GGUF model
lives in Silk's server, so a graph embedded with one waits for it to be
wired again rather than being served without it.
"""

import os
import re
import time
from functools import lru_cache
from pathlib import Path

from typing import Any, ClassVar, Dict, List, Optional, Tuple
from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.node.threaded import ThreadedNode
from weave.node import VerticalSizePolicy
from weave.node.port_registry import coerce_path_list
from weave.registry import register_node
from weave.logger import get_logger
from weave.library.sync_tool_button.widgets.sync_tool_button import SyncToolButton
from weave.widgets.path_history_picker import PathHistoryPicker

from .silk_ports import (  # noqa: F401
    MODEL_HANDLE_TYPE, OKF_GRAPH_TYPE, ONNX_EMBEDDER_TYPE, RERANKER_MODEL_TYPE,
)
from ..functions.port_text import text_input
from ..functions import embed_prefixes
from ..functions.gguf_meta import EmbeddingMeta, read_embedding_meta
from ..functions.okf_graph import (
    DEFAULT_EMBED_MODEL, DEVICES, DIM_RANGE, EMBED_PRECISIONS, FAMILY_LADDERS,
    FAMILY_PREFIXES,
    IMAGE_PRECISIONS, LADDER_DIMS, RERANK_PRECISIONS, VISION_MAX_DIM, OkfError,
    OkfGraph, OkfSpec, default_bundle_root, default_db_path, default_executable,
    find_databases, mismatch_text, model_cache, vision_partner,
)

log = get_logger("SilkKnowledgeGraphNode")

#: The Embedder row: which wire supplies the embedding model.
EMBED_SOURCES = ("onnx", "gguf")

#: The settings that decide what the vectors are made with.
_EMBED_KEYS = ("device", "model_id", "embedding_dim", "precision", "backend",
               "embed_family", "endpoint_url")

#: An OkfSpec with every default, for settings saved before a key existed.
_DEFAULT_SPEC = OkfSpec(db_path="")


@register_node
class SilkKnowledgeGraphNode(ThreadedNode):
    """One OKFgraph knowledge graph, served once and shared by handle."""
    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Knowledge"
    node_name: ClassVar[Optional[str]] = "Silk Knowledge Graph"
    node_description: ClassVar[Optional[str]] = (
        "Serves one OKFgraph knowledge graph (okf serve) to downstream "
        "ToolBoxes: search, read, traverse and ingest, on CPU or GPU."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "tools", "knowledge", "okfgraph", "memory",
    ]
    node_icon: ClassVar[Optional[str]] = "binary-tree-2"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 6   # owns a hand-written state dict
    # 1 = string line edit. 2 makes `db_path` a filepath picker
    # (BIDIRECTIONAL, save mode): values persist as before, wires now
    # carry the filepath type HF/model ports plug straight into. 3: the
    # pickers keep a history (`db_history`, `bundle_history`), and
    # `bundle_root` is a folder (dirpath) rather than a string.
    # 4: the embedder / onnx_embedder / reranker inputs and `embed_pooling`.
    # 5: the model comes from the wires alone. The Device, Model, Backend,
    # endpoint and Lane rows (and their inputs) are gone; `embed_source`
    # (auto / onnx / gguf) picks the wire. 6: no auto; a wired reranker
    # reranks every search, none wired reranks none.
    node_version = 6     # 3: roots discovery, history pickers, threaded

    #: Live re-embed progress, from the worker to the status label.
    _progress_text = Signal(str)

    def __init__(self, title: str = "Silk Knowledge Graph", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        # Where to look for a graph: the same roots the ToolBox is
        # sandboxed to, so the project's graph is found without a path.
        self.add_input("roots", datatype="dirpath_list")
        # The models, from loader nodes (see the module docstring).
        self.add_input("embedder", datatype="model_handle")
        self.add_input("onnx_embedder", datatype="onnx_embedder")
        self.add_input("reranker", datatype="reranker_model")
        self.add_output("knowledge", datatype="okf_graph")

        # ── Layout & WidgetCore ──
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self._db_path = PathHistoryPicker(
            mode="save_file",
            filters=["Knowledge DB (*.db)", "All Files (*)"])
        self._db_path.setToolTip(
            "The graph's database file. Created on the first ingest; one "
            "server owns it, and the okf CLI reaches the same graph "
            "through that server while it runs. Empty with roots wired: "
            "the first graph found in them, else a new "
            "<root>/.okfgraph/okfgraph.db. Every graph found, and every "
            "one picked before, is in the dropdown.")
        form.addRow("Database:", self._db_path)
        self._widget_core.register_widget(
            "db_path", self._db_path, role=PortRole.BIDIRECTIONAL,
            datatype="filepath", default="", policy=debounced(600),
            add_to_layout=False,
        )
        self.add_input("db_path", datatype="filepath")

        self._bundle_root = PathHistoryPicker(mode="folder")
        self._bundle_root.setToolTip(
            "Where ingested documents are written as Markdown bundles. "
            "Empty: .okfgraph/bundle beside the database.")
        form.addRow("Bundles:", self._bundle_root)
        self._widget_core.register_widget(
            "bundle_root", self._bundle_root, role=PortRole.INPUT,
            datatype="dirpath", default="", policy=debounced(600),
            add_to_layout=False,
        )
        self.add_input("bundle_root", datatype="dirpath")

        embed_row = QHBoxLayout()
        self._embed_source = QComboBox()
        self._embed_source.addItems(list(EMBED_SOURCES))
        self._embed_source.setToolTip(
            "Which embedding model makes the graph's vectors.\n"
            "onnx: Silk KG Embedder (ONNX) on onnx_embedder, which okf runs itself "
            "(embroider). Unwired: the ONNX model the graph was last "
            "embedded with, else okf's default.\n"
            "gguf: the Embedder Loader's model_obj on embedder, served by "
            "Silk's llama.cpp server and started when the graph needs it. "
            "Its family, pooling and width are read from the GGUF file.\n"
            "Switching lanes re-embeds the graph.")
        embed_row.addWidget(self._embed_source, stretch=1)
        self._btn_reembed = SyncToolButton(
            initial_text="Re-embed", initial_icon="refresh",
            show_label=False, dimensions=24,
        )
        self._btn_reembed.set_tooltip(
            "Rebuild every vector with the chosen embedder; press again "
            "while it runs to stop it. A stopped re-embed keeps the old "
            "pin, and the next one continues where it stopped.")
        self._btn_reembed.clicked.connect(self._on_reembed)
        embed_row.addWidget(self._btn_reembed)
        form.addRow("Embedder:", embed_row)
        self._widget_core.register_widget(
            "embed_source", self._embed_source, role=PortRole.INPUT,
            datatype="string", default="onnx", add_to_layout=False,
        )
        self.add_input("embed_source", datatype="string")

        self._dim = QComboBox()
        self._dim.addItems([str(d) for d in LADDER_DIMS])
        self._dim.setCurrentText("512")
        self._dim.setToolTip(
            "Vector width (a Matryoshka step) for ONNX models and for GGUF "
            "families okf has a ladder for (nomic-v1.5, up to 768). Any "
            "other GGUF embedder is stored at its own width. Applies when "
            "the graph is created; a re-embed cannot move widths, so a new "
            "width needs a fresh database.")
        form.addRow("Vector dim:", self._dim)
        self._widget_core.register_widget(
            "embedding_dim", self._dim, role=PortRole.INPUT,
            datatype="string", default="512", policy=debounced(600),
            add_to_layout=False,
        )
        self.add_input("embedding_dim", datatype="string")

        self._executable = QLineEdit("")
        self._executable.setPlaceholderText(
            default_executable() or "okf (not found: install silk[knowledge-cpu])")
        self._executable.setToolTip(
            "The okf executable. Empty: the one installed with Silk's "
            "knowledge extra, else the first on PATH.")
        form.addRow("okf:", self._executable)
        self._widget_core.register_widget(
            "executable", self._executable, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(600),
            add_to_layout=False,
        )
        self.add_input("executable", datatype="string")

        self._label_status = QLabel("Not running.")
        self._label_status.setWordWrap(True)
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        # The one served graph this node owns or attached to.
        self._graph: Optional[OkfGraph] = None
        # The database, device and model last seen agreeing with the pin.
        # Saved with the node: a change from these is the user's, and is
        # what re-embeds without asking.
        self._applied: Dict[str, str] = {}
        self._reembed_requested = False
        # The embed settings whose re-embed the user stopped. Not
        # re-embedded on their own again -- a stop that the next
        # evaluation undoes is no stop -- until Re-embed is pressed.
        self._stopped_for: Optional[Tuple[str, ...]] = None
        self._user_stopped = False
        self._last_reembed = ""
        # The last model-cache answer: (key, report, when asked).
        self._model_cache: Tuple[Tuple[str, ...], Dict[str, Any], float] = ((), {}, 0.0)
        # Set by compute when no working graph comes out of it; the node
        # turns red until an evaluation serves one again.
        self._failed = False
        # Set by cleanup: a worker still finishing must not leave a
        # server behind for a node that is gone.
        self._closed = False
        # What the worker found and chose, for the pickers (main thread).
        self._sync_found: List[str] = []
        self._sync_paths: Tuple[str, str] = ("", "")

        self._progress_text.connect(
            self._show_progress, Qt.ConnectionType.QueuedConnection)
        self.compute_started.connect(self._on_compute_started)
        self.compute_finished.connect(self._on_compute_finished)
        self.compute_cancelled.connect(self._on_compute_cancelled)

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── UI (main thread) ──────────────────────────────────────────────

    def _on_reembed(self) -> None:
        if self._is_computing:
            # The same button stops it: the job lane cancels at the next
            # batch boundary, so this answers within a poll or two.
            self._user_stopped = True
            self._widget_core.push_display("status", "Stopping…")
            self.cancel_compute()
            return
        self._reembed_requested = True
        self._stopped_for = None
        self._widget_core.push_display("status", "Re-embedding…")
        self.set_dirty("reembed")

    def _show_progress(self, text: str) -> None:
        self._widget_core.push_display("status", text)

    def _on_compute_started(self) -> None:
        self._btn_reembed.set_icon_name("player-stop")
        self._btn_reembed.set_tooltip("Stop: the graph keeps its old pin.")

    def _on_compute_finished(self) -> None:
        self._btn_reembed.set_icon_name("refresh")
        self._btn_reembed.set_tooltip(
            "Rebuild every vector with this device and model; press again "
            "while it runs to stop it.")

    def _on_compute_cancelled(self) -> None:
        if not self._user_stopped:
            return    # a newer evaluation replaced this one; it reports
        self._user_stopped = False
        self._widget_core.push_display(
            "status", getattr(self, "_sync_status", "") or "Stopped.")
        # The cancelled run's result was discarded, but it may have cycled
        # the server: run once more so the wire carries the live handle.
        self.set_dirty("reembed_stopped")

    # ── State ─────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["applied"] = dict(self._applied)
        state["db_history"] = self._db_path.history()
        state["bundle_history"] = self._bundle_root.history()
        return state

    @classmethod
    def migrate_state(cls, state: Dict[str, Any],
                      from_version: int) -> Dict[str, Any]:
        """api 1 → 6: the pickers persist the same string values, the
        histories are new (absent = empty). 4 adds the model inputs.

        5 dropped the model fields and 6 the ``auto`` lane: a scene keeps
        the lane its graph was on -- ``gguf`` for the external lane (the
        old ``backend`` row, or what ``applied`` last served), ``onnx``
        otherwise. The other dropped values are ignored on restore; the
        graph's model carries over in ``applied``, which is what an
        unwired ONNX lane serves.
        """
        if from_version < 6:
            widgets = state.get("widget_data")
            applied = state.get("applied")
            if isinstance(widgets, dict) and widgets.get("embed_source") not in EMBED_SOURCES:
                external = (widgets.get("backend") == "external"
                            or (isinstance(applied, dict)
                                and applied.get("backend") == "external"))
                widgets["embed_source"] = "gguf" if external else "onnx"
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        with self._widget_core.suppress_signals():
            super().restore_state(state)
        applied = state.get("applied")
        self._applied = (
            {str(k): str(v) for k, v in applied.items()}
            if isinstance(applied, dict) else {}
        )
        self._db_path.set_history([str(p) for p in state.get("db_history") or ()])
        self._bundle_root.set_history(
            [str(p) for p in state.get("bundle_history") or ()])

    # ── Worker thread ─────────────────────────────────────────────────

    @staticmethod
    def resolve_paths(inputs: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        """``(db_path, bundle_root, found)`` with the roots' defaults applied.

        A database typed or wired in wins. Otherwise the first graph found
        in the roots, otherwise a new one in the first root's
        ``.okfgraph``. The bundle root, left empty, is ``.okfgraph/bundle``
        beside whichever database that is.
        """
        found = find_databases(coerce_path_list(inputs.get("roots")))
        db_path = text_input(inputs, "db_path").strip()
        if not db_path:
            roots = coerce_path_list(inputs.get("roots"))
            db_path = found[0] if found else (default_db_path(roots[0]) if roots else "")
        bundle = text_input(inputs, "bundle_root").strip()
        if not bundle and db_path:
            bundle = default_bundle_root(db_path)
        return db_path, bundle, found

    @staticmethod
    def embed_source(inputs: Dict[str, Any]) -> str:
        """``gguf`` or ``onnx``: the lane the Embedder row picks."""
        chosen = text_input(inputs, "embed_source", "onnx")
        return chosen if chosen in EMBED_SOURCES else "onnx"

    @staticmethod
    def spec_from(inputs: Dict[str, Any],
                  applied: Optional[Dict[str, str]] = None) -> OkfSpec:
        """The server spec: paths from *inputs*, the model from the wires.

        With no embedder wired on the ONNX lane, the model the graph was
        last embedded with (*applied*), else okf's default.
        """
        raw_dim = text_input(inputs, "embedding_dim", "512").strip()
        try:
            dim = int(raw_dim)
        except ValueError:
            dim = 512
        if dim not in LADDER_DIMS:
            dim = 512
        applied = applied or {}
        spec = OkfSpec(
            db_path=text_input(inputs, "db_path"),
            executable=text_input(inputs, "executable"),
            bundle_root=text_input(inputs, "bundle_root"),
            embedding_dim=dim,
        )
        if SilkKnowledgeGraphNode.embed_source(inputs) == "gguf":
            gguf = _gguf_embedder(inputs.get("embedder"))
            if gguf.get("endpoint_url"):
                # Silk's llama.cpp server, on okf's external lane. The
                # server answers any model name with the one it has loaded,
                # so the wire's name can be the pin: stable across ports
                # and restarts.
                spec = replace(
                    spec, backend="external", endpoint_url=gguf["endpoint_url"],
                    model_id=gguf["model_id"], embed_family=gguf["family"],
                    embed_pooling=gguf["pooling"],
                    embedding_dim=_gguf_dim(gguf, dim),
                    embed_counter=gguf["counter"], embed_counter_gguf=gguf["path"],
                    embed_prefixes=gguf["prefixes"],
                )
        else:
            onnx = _onnx_description(inputs.get("onnx_embedder"))
            if onnx is None and applied.get("backend", "embroider") == "embroider":
                onnx = applied
            if onnx:
                # An empty model id in a saved record is okf's own default:
                # what graphs made before Silk had one were embedded with.
                spec = replace(
                    spec, model_id=str(onnx.get("model_id") or ""),
                    device=_choice(onnx.get("device"), DEVICES),
                    precision=_choice(onnx.get("precision"), EMBED_PRECISIONS),
                    cache_dir=str(onnx.get("cache_dir") or ""),
                    image_model_id=str(onnx.get("image_model_id") or ""),
                    image_precision=_choice(onnx.get("image_precision"),
                                            IMAGE_PRECISIONS),
                )
            else:
                spec = replace(spec, model_id=DEFAULT_EMBED_MODEL)
        return _with_reranker(spec, inputs.get("reranker"))

    @staticmethod
    def wiring_problem(inputs: Dict[str, Any]) -> str:
        """Why the wired models cannot be used as they are, or ``""``."""
        failed = _gguf_reranker(inputs.get("reranker")).get("error")
        if failed:
            return f"The reranker's server did not start: {failed}"
        handle = inputs.get("embedder")
        if SilkKnowledgeGraphNode.embed_source(inputs) == "onnx" or not handle:
            return ""
        gguf = _gguf_embedder(handle)
        if gguf.get("prefix_error"):
            return f"The Embedder Loader's prefixes: {gguf['prefix_error']}"
        if gguf.get("error"):
            return f"The embedder's server did not start: {gguf['error']}"
        if not gguf and isinstance(handle, dict) and handle.get("backend") == "onnx":
            return ("The embedder input takes a GGUF embedding model. An ONNX "
                    "model for the graph is chosen with Silk KG Embedder "
                    "(ONNX) on onnx_embedder (Embedder row onnx): okf runs it "
                    "itself, so the graph's vectors and pin stay okf's.")
        if not gguf:
            return ("The embedder input takes the Embedder Loader's model_obj "
                    "(a GGUF embedding model on Silk's llama.cpp server); "
                    "this model handle is not one.")
        width = gguf["width"]
        if (gguf["family"] not in FAMILY_LADDERS and width
                and not DIM_RANGE[0] <= width <= DIM_RANGE[1]):
            return (f"{gguf['model_id']} makes {width}-dim vectors; okf stores "
                    f"{DIM_RANGE[0]} to {DIM_RANGE[1]}. Load a smaller "
                    "embedding model.")
        return ""

    @staticmethod
    def _embed_key(spec: OkfSpec) -> Tuple[str, ...]:
        return (spec.db_path,) + tuple(str(getattr(spec, k)) for k in _EMBED_KEYS)

    def _changed_here(self, spec: OkfSpec) -> bool:
        """Whether *spec* is a change from the settings that last matched."""
        applied = self._applied
        if applied.get("db_path") != spec.db_path:
            return False
        before = tuple(applied.get(k, str(getattr(_DEFAULT_SPEC, k)))
                       for k in _EMBED_KEYS)
        now = tuple(str(getattr(spec, k)) for k in _EMBED_KEYS)
        return before != now

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        db_path, bundle, found = self.resolve_paths(inputs)
        self._sync_found = found
        self._sync_paths = (db_path, bundle)
        # Red unless this evaluation serves a graph that can search.
        self._failed = True
        problem = self.wiring_problem(inputs)
        if problem:
            # Not served on a guess: a graph embedded with the wrong model
            # is pinned to it, and undoing that is a re-embed.
            self._discard_graph()
            self._sync_status = problem
            return {"knowledge": None}
        spec = self.spec_from(dict(inputs, db_path=db_path, bundle_root=bundle),
                              self._applied)
        reembed_requested, self._reembed_requested = self._reembed_requested, False
        held = None if reembed_requested else _held_unprefixed(spec, self._applied)
        if held is not None:
            spec = held
        source = self.embed_source(inputs)
        wired = _wired_embedder(inputs, source)
        if source == "gguf" and not wired:
            # A GGUF model lives in Silk's server: without it there is
            # nothing to embed queries with, and serving the graph on
            # another model would mean re-embedding all of it.
            self._discard_graph()
            if (self._applied.get("backend") == "external"
                    and self._applied.get("db_path") == spec.db_path):
                self._sync_status = (
                    f"Waiting for the embedder: this graph was embedded with "
                    f"{self._applied.get('model_id') or 'a GGUF model'}, which "
                    "is not wired (or its loader unloaded it). Wire the "
                    "Embedder Loader's model_obj again, or set the Embedder "
                    "row to onnx to re-embed with an ONNX model.")
            else:
                self._sync_status = (
                    "The Embedder row is gguf: wire the Embedder Loader's "
                    "model_obj into embedder.")
            return {"knowledge": None}

        # Restart only when the settings actually changed: the device and
        # the model are server flags, and a graph re-evaluates for every
        # unrelated edit upstream.
        graph = self._graph
        if graph is not None and (graph.spec != spec or not graph.running):
            graph.close()
            self._graph = graph = None

        if not spec.db_path:
            self._failed = False   # a new node, not a broken one
            self._sync_status = (
                "Choose a database file for the graph, or wire roots to "
                "find (or create) one in <root>/.okfgraph.")
            return {"knowledge": None}

        if graph is None:
            try:
                graph = OkfGraph.open(spec, should_stop=self.is_compute_cancelled)
            except OkfError as exc:
                # Reported, never raised: a graph that will not start must
                # not take the agents downstream with it.
                self._sync_status = _error_text("Not running", exc)
                return {"knowledge": None}
            self._graph = graph
        if self._closed or self.is_compute_cancelled():
            self._failed = False
            if self._closed:
                self._discard_graph()
                return {"knowledge": None}
            return {"knowledge": graph}

        try:
            pins = graph.pins()
        except OkfError as exc:
            self._sync_status = "\n".join([
                graph.describe(), *_model_lines(spec, wired),
                _error_text("! Could not read the embedding pin", exc)])
            return {"knowledge": graph}

        kept = ""
        adopted = (_adopt_pinned_model(spec, pins)
                   if not wired and graph.owned
                   and self._applied.get("db_path") != spec.db_path else None)
        unprefixed = (_adopt_unprefixed(spec, pins)
                      if adopted is None and graph.owned and not reembed_requested
                      else None)
        if unprefixed is not None:
            # Embedded before Silk declared this model's prefixes: served
            # as it was made, rather than re-embedded unasked.
            adopted = unprefixed
        if adopted is not None:
            # A graph this node has not served before, embedded with
            # another model: keep that model rather than re-embed it (or
            # leave it unsearchable) because Silk's default changed.
            graph.close()
            self._graph = graph = None
            try:
                graph = OkfGraph.open(adopted, should_stop=self.is_compute_cancelled)
                pins = graph.pins()
            except OkfError as exc:
                self._discard_graph()
                self._sync_status = _error_text("Not running", exc)
                return {"knowledge": None}
            self._graph, spec = graph, adopted
            if unprefixed is None:
                kept = (f"Kept {spec.model_id}: the graph was embedded with it. "
                        f"New graphs get {DEFAULT_EMBED_MODEL}.")

        lines = [graph.describe(), *_model_lines(spec, wired)]
        if kept:
            lines.append(kept)
        if unprefixed is not None or held is not None:
            lines.append(
                "Embedded without the query/document prefixes this model is "
                "trained with, and served that way. Press Re-embed to embed "
                "with them: search gets better.")
        if spec.backend == "embroider" and graph.owned:
            lines.extend(self._cache_lines(spec))
        if self._last_reembed:
            lines.append(self._last_reembed)
        if not graph.owned:
            lines.append(
                "Attached to a server started elsewhere: its device and model "
                "apply, not the ones set here.")

        mismatch = mismatch_text(pins)
        stopped = self._stopped_for == self._embed_key(spec)
        if mismatch and graph.owned and (
                reembed_requested or (self._changed_here(spec) and not stopped)):
            self._last_reembed = self._reembed(graph, spec)
            lines.append(self._last_reembed)
            graph = self._graph   # _reembed may have cycled the server
            if graph is None:
                self._sync_status = "\n".join(lines)
                return {"knowledge": None}
            try:
                pins = graph.pins()
            except OkfError as exc:
                lines.append(_error_text("! Could not read the embedding pin", exc))
                self._sync_status = "\n".join(lines)
                return {"knowledge": graph}
            mismatch = mismatch_text(pins)
        elif reembed_requested and not mismatch:
            lines.append("Nothing to re-embed: the graph already matches these settings.")

        lines.append(_pin_summary(pins))
        if mismatch:
            lines.append(
                f"! Embedded with other settings ({mismatch}). Search and "
                "ingest fail until it is re-embedded: press Re-embed, or set "
                "them back.")
        else:
            self._failed = False
            self._last_reembed = ""
            self._stopped_for = None
            self._applied = {
                "db_path": spec.db_path, "device": spec.device,
                "model_id": spec.model_id, "embedding_dim": spec.embedding_dim,
                "precision": spec.precision, "cache_dir": spec.cache_dir,
                "image_model_id": spec.image_model_id,
                "image_precision": spec.image_precision,
                "backend": spec.backend,
                "source": "wire" if wired else "fields", "embed_family": spec.embed_family,
                "endpoint_url": spec.endpoint_url,
            }
        self._sync_status = "\n".join(lines)
        return {"knowledge": graph}

    def _reembed(self, graph: OkfGraph, spec: OkfSpec) -> str:
        """Re-embed *graph* under *spec*; the line to show.

        okfgraph 0.12's reembed re-pins in place, so the graph never
        changes width nor endures a half-model state. The one relaunch
        the node ever does is the declared one: a pin-exempt boot
        (``OKFGRAPH_ALLOW_REPIN=1``) around the re-embed itself, and the
        server the node leaves behind reboots pin-clean."""
        log.info(f"re-embedding {spec.db_path}")
        try:
            if self._graph is not None and self._graph.running:
                # The escape lives for exactly one server lifetime.
                self._graph.close()
                self._graph = None
            self._graph, report = OkfGraph.repinned(
                spec, should_stop=self.is_compute_cancelled,
                on_progress=self._report_job,
            )
        except OkfError as exc:
            if self._graph is None and not self._closed:
                try:
                    self._graph = OkfGraph.open(spec)
                except OkfError:
                    pass
            if exc.code == "CANCELLED":
                self._stopped_for = self._embed_key(spec)
                return "Re-embed stopped before it began."
            return _error_text("! Re-embed failed", exc)
        if self._closed:
            self._discard_graph()
        counts = report.get("reembedded") or {}
        done = ", ".join(
            f"{v} {k.replace('_', ' ')}" for k, v in counts.items() if v
        ) or "nothing"
        seconds = float(report.get("seconds") or 0.0)
        if report.get("cancelled"):
            self._stopped_for = self._embed_key(spec)
            return (f"Re-embed stopped after {done} in {seconds:.0f}s: the "
                    "graph keeps its old pin. Press Re-embed to continue.")
        line = f"Re-embedded {done} in {seconds:.0f}s."
        pinned = (report.get("pins_after") or {}).get("pinned") or {}
        if pinned.get("embedding_dim"):
            line += f" width {pinned['embedding_dim']}."
        if pinned.get("model_id"):
            line += f" pinned to {pinned['model_id']}."
        return line

    def _report_job(self, record: Dict[str, Any]) -> None:
        """One job-lane poll: the progress bar and the status line."""
        done = int(record.get("done") or 0)
        total = int(record.get("total") or 0)
        phase = str(record.get("phase") or "starting")
        if total:
            self.report_progress(done * 100 // total)
        stopping = " (stopping…)" if record.get("status") == "cancelling" else ""
        count = f" {done}/{total}" if total else ""
        self._progress_text.emit(f"Re-embedding {phase}{count}{stopping}")

    def _discard_graph(self) -> None:
        graph, self._graph = self._graph, None
        if graph is not None:
            graph.close()

    # ── Main thread ───────────────────────────────────────────────────

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._offer_paths()
        if hasattr(self, "_sync_status"):
            self._widget_core.push_display("status", self._sync_status)
        if self._failed:
            # Visual only: the wire still carries None (or the mismatched
            # graph), and the next evaluation that serves one clears it.
            self._enter_error_state()

    def _cache_lines(self, spec: OkfSpec) -> List[str]:
        """Whether the embedding model is downloaded; asked once per
        model, and again a minute later while it is not."""
        key = (spec.resolved_executable(), spec.model_id, spec.device,
               spec.precision, spec.cache_dir)
        old_key, report, when = self._model_cache
        if key != old_key or (not report.get("cached")
                              and time.monotonic() - when > 60.0):
            report = model_cache(spec)
            self._model_cache = (key, report, time.monotonic())
        return _cache_text(report)

    def _offer_paths(self) -> None:
        """Fill an empty field with what compute chose; list the rest.

        The first root's graph goes into the field, so the choice can be
        read off the node; every other graph found, and every path picked
        before, stays one click away in the dropdown.
        """
        db_path, bundle = self._sync_paths
        db_items = [p for p in [db_path, *self._sync_found] if p]
        bundle_items = [default_bundle_root(p) for p in db_items]
        for picker, port, chosen, items in (
                (self._db_path, "db_path", db_path, db_items),
                (self._bundle_root, "bundle_root", bundle, bundle_items)):
            merged = list(dict.fromkeys(items + picker.history()))
            picker.set_history(merged[:PathHistoryPicker.MAX_HISTORY])
            if chosen and not str(picker.get_value() or "").strip():
                with self._widget_core.suppress_signals():
                    self._widget_core.apply_port_value(port, chosen)

    def cleanup(self) -> None:
        # An owned graph is a live server holding the database file. Node
        # removal is the only thing that stops it, so this cannot be skipped.
        # Cancel first: a worker mid re-embed checks _closed before it
        # leaves a server behind.
        self._closed = True
        self.cancel_compute()
        self._discard_graph()
        super().cleanup()


def _choice(value: Any, choices: Tuple[str, ...]) -> str:
    text = str(value or "").strip().lower()
    return text if text in choices else "auto"


def _onnx_description(value: Any) -> Optional[Dict[str, Any]]:
    """A Silk KG Embedder (ONNX) node's wire value, or ``None``."""
    if isinstance(value, dict) and value.get("runtime") == "embroider":
        return value
    return None


@lru_cache(maxsize=32)
def _embedding_meta_cached(path: str, _mtime: float) -> EmbeddingMeta:
    try:
        return read_embedding_meta(path)
    except (OSError, ValueError):
        return EmbeddingMeta()


def _embedding_meta(path: str) -> EmbeddingMeta:
    """The GGUF's embedding identity, read once per file version."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return EmbeddingMeta()
    return _embedding_meta_cached(path, mtime)


def _family_for(name: Optional[str], path: str) -> str:
    """okf's family label for a GGUF: ``nomic-v1.5`` when it is one (okf
    has its prefixes and ladder), else the model's own name, made a legal
    label (no ``/``)."""
    text = (name or Path(path).stem).lower().replace("_", "-")
    if "nomic-embed-text-v1.5" in text or "nomic-embed-text-v1-5" in text:
        return "nomic-v1.5"
    return re.sub(r"[^a-z0-9.-]+", "-", text).strip("-.")[:64] or "gguf"


def _gguf_dim(gguf: Dict[str, Any], dim: int) -> int:
    """The width to store: the row's step where the family has a ladder,
    else the model's own width (okf cannot shorten it)."""
    ladder = FAMILY_LADDERS.get(gguf["family"])
    if ladder:
        return dim if dim in ladder else ladder[-1]
    return int(gguf["width"] or dim)


def _gguf_embedder(handle: Any) -> Dict[str, Any]:
    """``{endpoint_url, model_id, path, family, pooling, width}`` for an
    Embedder Loader handle.

    Empty for anything else: a chat model's pool, a remote endpoint, or
    nothing wired. okf appends ``/v1/embeddings`` to the endpoint itself,
    so the pool's ``.../v1`` base loses its suffix here. The family,
    pooling and width come from the GGUF file (``width`` 0 when it does
    not say); llama.cpp pools by the file's own key, so that is what the
    vectors are.
    """
    if not (isinstance(handle, dict) and handle.get("backend") == "gguf"
            and handle.get("role") == "embedding"):
        return {}
    pool = handle.get("pool")
    if pool is None:
        return {}
    try:
        # Asking for the address starts a lazy pool's server: this is the
        # demand the Embedder Loader waits for.
        base = pool.base_url
        base = str(base() if callable(base) else base).rstrip("/")
        path = str(pool.model_path)
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return {"error": str(exc) or type(exc).__name__}
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    if not base or not path:
        return {}
    meta = _embedding_meta(path)
    family = _family_for(meta.name, path)
    name = f"{meta.name or ''} {Path(path).stem}"
    try:
        table = embed_prefixes.resolve(str(handle.get("prefixes") or ""), name)
    except ValueError as exc:
        return {"prefix_error": str(exc)}
    # okf brings its own table for the families it knows: declared as the
    # family, under the pin graphs made before Silk had tables carry.
    builtin = (family in FAMILY_PREFIXES
               and table == embed_prefixes.prefixes_for(name))
    tag = "" if builtin else embed_prefixes.pin_tag(table)
    return {"endpoint_url": base,
            "model_id": f"gguf/{Path(path).stem}" + (f"+p{tag}" if tag else ""),
            "plain_model_id": f"gguf/{Path(path).stem}",
            "prefixes": "" if builtin else embed_prefixes.declaration(table),
            "counter": ("server-tokenize" if getattr(pool, "serves_tokenize", False)
                        else "gguf-meta"),
            "path": path, "family": family,
            "pooling": meta.pooling or "mean", "width": meta.width or 0}


def _wired_embedder(inputs: Dict[str, Any], source: str) -> str:
    """The wire that supplies the model on *source*'s lane, or ``""``."""
    if source == "gguf":
        return "gguf" if _gguf_embedder(inputs.get("embedder")).get("endpoint_url") else ""
    return "onnx" if _onnx_description(inputs.get("onnx_embedder")) else ""


def _model_lines(spec: OkfSpec, wired: str) -> List[str]:
    """What embeds and what reranks."""
    lines: List[str] = []
    if wired == "gguf":
        lines.append(
            f"Embedding through the wired GGUF embedder {spec.model_id} "
            f"(from its file: family {spec.embed_family}, "
            f"{spec.embed_pooling} pooling, {spec.embedding_dim} dims); it "
            "must stay loaded for search and ingest to work.")
    elif wired == "onnx":
        lines.append(
            f"Embedding with the wired ONNX model {spec.model_id or 'default'} "
            f"on {spec.device} ({spec.precision}).")
    else:
        lines.append(
            f"No embedder wired: okf loads "
            f"{spec.model_id or 'its default model (jina v5 small)'} itself, on "
            f"{spec.device}. Wire Silk KG Embedder (ONNX) or an Embedder Loader to "
            "change it.")
    if wired != "gguf":
        lines.append(_image_line(spec))
    if spec.rerank_model_id:
        where = (f"at {spec.rerank_endpoint_url}" if spec.rerank_backend == "external"
                 else f"on {spec.rerank_device} ({spec.rerank_precision})")
        lines.append(f"Reranking every search with {spec.rerank_model_id} {where}.")
    else:
        lines.append("No reranker wired: searches are not reranked.")
    return lines


def _image_line(spec: OkfSpec) -> str:
    """How images are embedded for image-content search."""
    image = spec.image_model_id or vision_partner(spec.model_id)
    if not image:
        return (f"No image model pairs with {spec.model_id or 'the default model'}: "
                "images are searched by their captions.")
    if spec.embedding_dim > VISION_MAX_DIM:
        return (f"Image search needs a vector width of at most {VISION_MAX_DIM} "
                f"({spec.embedding_dim} here): images are searched by their captions.")
    return f"Embedding images with {image} ({spec.image_precision})."


def _gguf_reranker(handle: Any) -> Dict[str, Any]:
    """``{endpoint_url}`` for a Reranker Loader (or ONNX Loader as a reranker)
    handle, ``{error}`` when its server would not start, empty for
    anything else.

    Asking the lazy pool for its address starts the server: this is the
    demand the loader waits for.
    """
    if not (isinstance(handle, dict) and handle.get("runtime") in _POOL_RUNTIMES
            and handle.get("role") == "rerank"):
        return {}
    pool = handle.get("pool")
    if pool is None:
        return {}
    try:
        base = pool.base_url
        base = str(base() if callable(base) else base).rstrip("/")
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return {"error": str(exc) or type(exc).__name__}
    return {"endpoint_url": base} if base else {}


def _with_reranker(spec: OkfSpec, reranker: Any) -> OkfSpec:
    """*spec* with the wired reranker's rows; unchanged when none is."""
    if not (isinstance(reranker, dict) and reranker.get("role") == "rerank"):
        return spec
    model_id = str(reranker.get("model_id") or "")
    if reranker.get("runtime") in _POOL_RUNTIMES:
        gguf = _gguf_reranker(reranker)
        if not gguf.get("endpoint_url"):
            return spec
        # A Silk rerank server on okf's external lane: okf cuts pairs to
        # the server's context and counts them with its /tokenize.
        return replace(
            spec, rerank_backend="external", rerank_model_id=model_id,
            rerank_endpoint_url=gguf["endpoint_url"],
            rerank_family=str(reranker.get("family") or "gguf-reranker"),
            rerank_max_tokens=int(reranker.get("max_tokens") or 0),
            rerank_count="server-tokenize",
        )
    if reranker.get("runtime") == "external" and reranker.get("endpoint_url"):
        return replace(
            spec, rerank_backend="external", rerank_model_id=model_id,
            rerank_endpoint_url=str(reranker.get("endpoint_url") or ""),
            rerank_family=str(reranker.get("family") or ""),
            rerank_api_key_env=str(reranker.get("api_key_env") or ""),
        )
    if reranker.get("runtime") != "embroider":
        return spec
    return replace(
        spec, rerank_backend="embroider", rerank_model_id=model_id,
        rerank_device=_choice(reranker.get("device"), DEVICES),
        rerank_precision=_choice(reranker.get("precision"), RERANK_PRECISIONS),
        # One cache per server: the embedder's folder, if it named one.
        cache_dir=spec.cache_dir or str(reranker.get("cache_dir") or ""),
    )


#: Reranker runtimes that are a Silk server behind a lazy pool: the
#: Reranker Loader's GGUF one and the ONNX Loader's embroider one.
_POOL_RUNTIMES = ("gguf", "onnx")

#: The pin suffix naming a GGUF model's prefix table (``+p`` and six hex).
_PREFIX_PIN = re.compile(r"\+p[0-9a-f]{6}$")


def _unprefixed(spec: OkfSpec) -> Optional[OkfSpec]:
    """*spec* without its prefix table, or None when it declares none."""
    if spec.backend != "external" or not _PREFIX_PIN.search(spec.model_id):
        return None
    return replace(spec, model_id=_PREFIX_PIN.sub("", spec.model_id),
                   embed_prefixes="")


def _adopt_unprefixed(spec: OkfSpec, pins: Dict[str, Any]) -> Optional[OkfSpec]:
    """*spec* without prefixes when the graph was embedded with this model
    before Silk declared its table: the pin is the plain id, nothing else
    disagrees."""
    plain = _unprefixed(spec)
    if plain is None or pins.get("empty"):
        return None
    pinned = str((pins.get("pinned") or {}).get("model_id") or "")
    fields = [str(f) for f in (pins.get("mismatch") or ())]
    return plain if pinned == plain.model_id and fields == ["model_id"] else None


def _held_unprefixed(spec: OkfSpec, applied: Dict[str, Any]) -> Optional[OkfSpec]:
    """*spec* without prefixes while this node last served the same graph
    on the same model without them (until Re-embed)."""
    plain = _unprefixed(spec)
    if (plain is None or applied.get("db_path") != spec.db_path
            or applied.get("model_id") != plain.model_id):
        return None
    return plain


def _adopt_pinned_model(spec: OkfSpec, pins: Dict[str, Any]) -> Optional[OkfSpec]:
    """*spec* on the ONNX model the graph is pinned to, when only the
    model disagrees and that model is one embroider loads; else None."""
    pinned = pins.get("pinned") or {}
    model = str(pinned.get("model_id") or "")
    fields = [str(f) for f in (pins.get("mismatch") or ())]
    if (pins.get("empty") or not model or model == spec.model_id
            or model.startswith("gguf/") or fields != ["model_id"]):
        return None
    return replace(spec, model_id=model)


def _cache_text(report: Dict[str, Any]) -> List[str]:
    """The model-cache line, or nothing when okf could not be asked."""
    if not report:
        return []
    model = report.get("model_id") or "The embedding model"
    where = report.get("cache_dir") or "the Hugging Face cache"
    if report.get("cached"):
        size = (report.get("disk_usage_bytes") or 0) / 1024 ** 3
        return [f"Model files: {report.get('repo') or model} "
                f"({report.get('precision')}, {size:.2f} GB) in {where}."]
    return [f"Not downloaded yet: {report.get('repo') or model} "
            f"({report.get('precision')}). The first ingest or search fetches "
            f"it from Hugging Face into {where}; offline, both fail until it "
            "is there."]


def _error_text(what: str, exc: OkfError) -> str:
    line = f"{what}: {exc.message}"
    return f"{line}\n{exc.remedy}" if exc.remedy else line


def _pin_summary(pins: Dict[str, Any]) -> str:
    """What the graph holds and what its vectors were made with."""
    if pins.get("empty"):
        configured = pins.get("configured") or {}
        model = configured.get("model_id") or "the default model"
        dim = configured.get("embedding_dim") or "?"
        return f"Empty graph: the first ingest pins it to {model} at dim {dim}."
    pinned = pins.get("pinned") or {}
    # The model pin is recorded by the session's first embed, so a graph
    # this server just filled may not have one recorded yet.
    return (
        f"Embedded with {pinned.get('model_id') or 'a model not yet recorded'} "
        f"({pinned.get('precision') or 'engine-native'}) at dim "
        f"{pinned.get('embedding_dim') or '?'}: "
        f"{pins.get('concepts', 0)} concepts, {pins.get('chunks', 0)} chunks."
    )
