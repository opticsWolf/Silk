# -*- coding: utf-8 -*-
"""Silk Knowledge Graph Node.

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
* the settings were changed *here*, on a graph that matched the old ones --
  the graph is re-embedded under the new settings right away, since that
  is what changing them means. OKFgraph keeps the database file as a
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
"""

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

from .silk_ports import OKF_GRAPH_TYPE  # noqa: F401
from ..functions.port_text import text_input
from ..functions.okf_graph import (
    DEVICES, KNOWN_MODELS, LADDER_DIMS, OkfError, OkfGraph, OkfSpec,
    default_bundle_root, default_db_path, default_executable, find_databases,
    mismatch_text,
)

log = get_logger("SilkKnowledgeGraphNode")

#: The settings that decide what the vectors are made with.
_EMBED_KEYS = ("device", "model_id", "embedding_dim", "backend",
               "embed_family", "endpoint_url")


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
    node_subclass: ClassVar[str] = "Agents"
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
    node_state_api = 3   # owns a hand-written state dict
    # 1 = string line edit. 2 makes `db_path` a filepath picker
    # (BIDIRECTIONAL, save mode): values persist as before, wires now
    # carry the filepath type HF/model ports plug straight into. 3: the
    # pickers keep a history (`db_history`, `bundle_history`), and
    # `bundle_root` is a folder (dirpath) rather than a string.
    node_version = 3     # 3: roots discovery, history pickers, threaded

    #: Live re-embed progress, from the worker to the status label.
    _progress_text = Signal(str)

    def __init__(self, title: str = "Silk Knowledge Graph", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── Ports ──
        # Where to look for a graph: the same roots the ToolBox is
        # sandboxed to, so the project's graph is found without a path.
        self.add_input("roots", datatype="dirpath_list")
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

        self._device = QComboBox()
        self._device.addItems(list(DEVICES))
        self._device.setToolTip(
            "Where the embedding model runs. 'cuda' needs the knowledge-gpu "
            "extra; 'auto' takes the GPU when one answers. CPU vectors are "
            "FP32 and GPU vectors FP16, so switching re-embeds the graph.")
        form.addRow("Device:", self._device)
        self._widget_core.register_widget(
            "device", self._device, role=PortRole.INPUT,
            datatype="string", default="auto", add_to_layout=False,
        )
        self.add_input("device", datatype="string")

        model_row = QHBoxLayout()
        self._model = QComboBox()
        # Editable: a model OKFgraph's registry gained since this list was
        # written is usable by typing its id.
        self._model.setEditable(True)
        self._model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._model.addItems(list(KNOWN_MODELS))
        self._model.setToolTip(
            "The embedding model. A graph keeps the model it was embedded "
            "with; changing it here re-embeds the graph (0.12 does that in "
            "place, re-pinning; no fresh database needed unless the width "
            "below changes).")
        model_row.addWidget(self._model, stretch=1)
        self._btn_reembed = SyncToolButton(
            initial_text="Re-embed", initial_icon="refresh",
            show_label=False, dimensions=24,
        )
        self._btn_reembed.set_tooltip(
            "Rebuild every vector with this device and model; press again "
            "while it runs to stop it. A stopped re-embed keeps the old "
            "pin, and the next one continues where it stopped.")
        self._btn_reembed.clicked.connect(self._on_reembed)
        model_row.addWidget(self._btn_reembed)
        form.addRow("Model:", model_row)
        self._widget_core.register_widget(
            "model_id", self._model, role=PortRole.INPUT,
            datatype="string", default=KNOWN_MODELS[0], policy=debounced(600),
            add_to_layout=False,
        )
        self.add_input("model_id", datatype="string")

        self._dim = QComboBox()
        self._dim.addItems([str(d) for d in LADDER_DIMS])
        self._dim.setCurrentText("512")
        self._dim.setToolTip(
            "Vector width (a Matryoshka ladder step). Applies when the graph "
            "is created; once it has vectors the width is what they were made "
            "at -- okfgraph reports it pinned below, and changing it needs a "
            "fresh database by design (a re-embed cannot move widths).")
        form.addRow("Vector dim:", self._dim)
        self._widget_core.register_widget(
            "embedding_dim", self._dim, role=PortRole.INPUT,
            datatype="string", default="512", policy=debounced(600),
            add_to_layout=False,
        )
        self.add_input("embedding_dim", datatype="string")

        ext_row = QHBoxLayout()
        self._backend = QComboBox()
        self._backend.addItems(["embroider", "external"])
        self._backend.setToolTip(
            "The embedding lane. 'embroider' = OKFgraph's own ONNX crate "
            "(Jina v5; downloads on first use). 'external' = an OpenAI-shaped "
            "POST /v1/embeddings host you point at: llama.cpp with "
            "--embeddings, LM Studio, TEI -- declared family and endpoint, "
            "never probed into shape.")
        ext_row.addWidget(self._backend, stretch=1)
        self._endpoint = QLineEdit("")
        self._endpoint.setPlaceholderText("http://127.0.0.1:8931 (external lane)")
        self._endpoint.setToolTip(
            "Backend 'external' only: the embed server's base URL -- okfgraph "
            "appends /v1/embeddings. The model field above becomes the wire's "
            "model id and the family below flavours the prefixes/contract.")
        ext_row.addWidget(self._endpoint, stretch=1)
        form.addRow("Backend:", ext_row)
        self._widget_core.register_widget(
            "backend", self._backend, role=PortRole.INPUT,
            datatype="string", default="embroider", policy=debounced(600),
            add_to_layout=False,
        )
        self.add_input("backend", datatype="string")
        self._widget_core.register_widget(
            "endpoint_url", self._endpoint, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(600),
            add_to_layout=False,
        )
        self.add_input("endpoint_url", datatype="string")

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

        fam_row = QHBoxLayout()
        self._family = QLineEdit("")
        self._family.setPlaceholderText("nomic-v1.5 (declared family)")
        self._family.setToolTip(
            "Backend 'external' only: the declared embedding family -- drives "
            "pooling, per-side prefixes and the Matryoshka ladder inside "
            "okfgraph. Declared, never guessed; 'nomic-v1.5' is the built-in "
            "reference (native 768, search_query:/search_document:).")
        fam_row.addWidget(self._family, stretch=1)
        self._api_key_env = QLineEdit("")
        self._api_key_env.setPlaceholderText("API key env var (optional)")
        self._api_key_env.setToolTip(
            "Backend 'external' only: name of an environment variable that "
            "holds the endpoint's bearer key. The value is never stored here.")
        fam_row.addWidget(self._api_key_env, stretch=1)
        form.addRow("Lane:", fam_row)
        self._widget_core.register_widget(
            "embed_family", self._family, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(600),
            add_to_layout=False,
        )
        self.add_input("embed_family", datatype="string")
        self._widget_core.register_widget(
            "api_key_env", self._api_key_env, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(600),
            add_to_layout=False,
        )
        self.add_input("api_key_env", datatype="string")

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
        """api 1 → 3: the pickers persist the same string values, the
        histories are new (absent = empty). Older states restore untouched."""
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
    def spec_from(inputs: Dict[str, Any]) -> OkfSpec:
        device = text_input(inputs, "device", "auto")
        backend = text_input(inputs, "backend", "embroider")
        raw_dim = text_input(inputs, "embedding_dim", "512").strip()
        try:
            dim = int(raw_dim)
        except ValueError:
            dim = 512
        if dim not in LADDER_DIMS:
            dim = 512
        spec = OkfSpec(
            db_path=text_input(inputs, "db_path"),
            executable=text_input(inputs, "executable"),
            bundle_root=text_input(inputs, "bundle_root"),
            device=device if device in DEVICES else "auto",
            model_id=text_input(inputs, "model_id"),
            embedding_dim=dim,
            backend=backend if backend in ("embroider", "external") else "embroider",
            embed_family=text_input(inputs, "embed_family", ""),
            endpoint_url=text_input(inputs, "endpoint_url", ""),
            api_key_env=text_input(inputs, "api_key_env", ""),
        )
        # Honest external lane: drop the external-only rows when the
        # lane is the default crate; the family label rides otherwise.
        if spec.backend == "embroider":
            spec = replace(spec, embed_family="", endpoint_url="",
                           api_key_env="")
        elif not spec.embed_family.strip() or not spec.endpoint_url.strip():
            # An external lane without its declared contract would boot
            # refusals; keep the crate lane instead of a broken state.
            spec = replace(spec, backend="embroider", embed_family="",
                           endpoint_url="", api_key_env="")
        return spec

    @staticmethod
    def _embed_key(spec: OkfSpec) -> Tuple[str, ...]:
        return (spec.db_path,) + tuple(str(getattr(spec, k)) for k in _EMBED_KEYS)

    def _changed_here(self, spec: OkfSpec) -> bool:
        """Whether *spec* is a change from the settings that last matched."""
        applied = self._applied
        if applied.get("db_path") != spec.db_path:
            return False
        before = tuple(applied.get(k, "") for k in _EMBED_KEYS)
        now = tuple(str(getattr(spec, k)) for k in _EMBED_KEYS)
        return before != now

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        db_path, bundle, found = self.resolve_paths(inputs)
        self._sync_found = found
        self._sync_paths = (db_path, bundle)
        spec = self.spec_from(dict(inputs, db_path=db_path, bundle_root=bundle))
        reembed_requested, self._reembed_requested = self._reembed_requested, False

        # Restart only when the settings actually changed: the device and
        # the model are server flags, and a graph re-evaluates for every
        # unrelated edit upstream.
        graph = self._graph
        if graph is not None and (graph.spec != spec or not graph.running):
            graph.close()
            self._graph = graph = None

        if not spec.db_path:
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
        if self._closed:
            self._discard_graph()
            return {"knowledge": None}
        if self.is_compute_cancelled():
            return {"knowledge": graph}

        lines = [graph.describe()]
        if self._last_reembed:
            lines.append(self._last_reembed)
        if not graph.owned:
            lines.append(
                "Attached to a server started elsewhere: its device and model "
                "apply, not the ones set here.")
        try:
            pins = graph.pins()
        except OkfError as exc:
            lines.append(_error_text("! Could not read the embedding pin", exc))
            self._sync_status = "\n".join(lines)
            return {"knowledge": graph}

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
            self._last_reembed = ""
            self._stopped_for = None
            self._applied = {
                "db_path": spec.db_path, "device": spec.device,
                "model_id": spec.model_id, "embedding_dim": spec.embedding_dim,
                "backend": spec.backend, "embed_family": spec.embed_family,
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
