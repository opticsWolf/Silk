# -*- coding: utf-8 -*-
"""Silk Agent Node.

The autonomous tool-calling agent as a graph node. Wires a GGUF model
(pool), a Silk ToolSet, and a Silk Role into the Qt-free
:class:`AgentLoop`: stream one model response, execute ``tool_call``
fences through the toolset (hard role enforcement at dispatch), feed
results back, repeat until the model answers.

The agent deliberately accepts only a ``silk_toolset`` — the restricted,
per-agent surface derived by a ToolSet node — never the raw ToolBox, so
the full tool registry cannot reach a model by accident.

Execution & chaining
--------------------
Manual: click *Run Agent* (or Enter in the prompt editor). Remote: pulse
the ``run`` Exec input — Manual nodes auto-execute on an incoming pulse —
and on completion the node pulses its ``done`` Exec output. Chain agent
networks by wiring ``response`` → next agent's ``user_prompt`` and
``done`` → next agent's ``run``.

Threading: ``compute()`` drives the AgentLoop on the worker thread; deltas
reach the UI via a queued Qt signal and downstream nodes via
``emit_stream``. Cancellation maps onto the engine's stop flag.
"""

import copy
import itertools
import uuid
from contextlib import ExitStack
from typing import Any, ClassVar, Dict, List, Optional, cast

from PySide6.QtCore import Qt, QEvent, Signal, Slot
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.node.threaded import ThreadedManualNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from weave.widgets.markdown_widget import MarkdownWidget
from weave.widgets.sync_button import SyncButton

from .silk_ports import GGUF_MODEL_TYPE, SILK_ROLE_TYPE, SILK_TOOLSET_TYPE  # noqa: F401
from ..functions.agent_loop import AgentLoop, DEFAULT_MAX_ROUNDS
from ..functions.event_sink import RunSink
from ..functions.approval import bind_run_seam, headless_refusals
from ..functions.graph_author import CanvasBinding, RunScope, bind_canvas
from ..functions.self_modify import (
    ALWAYS_APPROVE, ChangeSet, attach_change_tracking, bind_changes,
)
from ..functions.main_thread_call import (
    DEFAULT_TIMEOUT_S as CANVAS_TIMEOUT_S, MainThreadCall,
)
from .graph_canvas import CanvasAuthor
from ..functions.compaction import Compactor
from ..functions.file_grants import resolve_grants
from ..functions.decision_registry import REGISTRY as DECISIONS
from ..functions.decision_seam import (
    DEFAULT_TIMEOUT_S,
    DecisionRequest,
    DecisionSeam,
)
from ..functions.grants import SCOPE_ALWAYS, SCOPE_ONCE, SCOPE_RUN
from ..functions.graph_engine import GraphEngine
from ..functions.remember import RunIdentity, bind_run_identity
from ..functions.usage_limits import describe_budget, parse_budget
from ..functions.hooks import (
    HOOK_AFTER_MODEL_RESPONSE,
    HOOK_AFTER_RUN,
    HOOK_AFTER_TOOL_EXECUTE,
    HOOK_BEFORE_MODEL_REQUEST,
    HOOK_BEFORE_RUN,
    HOOK_TOOL_DENIED,
    HOOK_WRAP_TOOL_EXECUTE,
)
from ..functions.messaging import AgentMessage
from ..functions.role import DEFAULT_ROLE, RoleBinding
from ..functions.task_store import plan_changed_event  # Qt-free
from ..functions.stream_events import (
    OUTCOME_COMPLETED,
    EventChatTurn,
    EventCompaction,
    EventDecisionRequest,
    EventDecisionResponse,
    EventDelta,
    EventError,
    EventModelRequest,
    EventModelResponse,
    EventPlan,
    EventReflection,
    EventRunFinished,
    EventRunResult,
    EventToolCall,
    EventToolDenied,
    EventToolResult,
    to_wire,
)
from ..functions.subagent import compose_system_prompt

log = get_logger("SilkAgent")


@register_node
class SilkAgentNode(ThreadedManualNode):
    """Autonomous tool-calling agent over model + toolbox + role."""
    node_state_api = 1   # owns a hand-written state dict
    # Bumped on any change to what this node persists or how its
    # ports are shaped; `node_state_api` is what says old state can
    # no longer be restored (G20). 1 = the shape this spec settled on.
    # 2: the `event_log` port (T7, D85). Same `node_state_api`: an older
    # state simply has no entry for it and restores with the box off,
    # which is also the default.
    node_version = 2

    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    #: This run's durable event log, or None when the box is off
    #: (T7, D85). Opened by the first event, closed with the run.
    _run_sink: Optional[RunSink] = None

    # Worker → main-thread bridges (V6 R11.1).
    chunk_streamed = Signal(str)
    status_changed = Signal(str)
    # A gated tool call is blocked on the worker thread and needs the user
    # (D48). The request travels as a plain dict so the queued connection
    # carries nothing Qt has to marshal specially.
    decision_requested = Signal(dict)
    decision_settled = Signal(str)

    node_class: ClassVar[str] = "AI"
    node_subclass: ClassVar[str] = "Agents"
    node_name: ClassVar[Optional[str]] = "Silk Agent"
    node_description: ClassVar[Optional[str]] = (
        "Autonomous tool-calling agent: model + toolbox + role, "
        "with Exec chaining for agent networks."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "agent", "llm", "tools", "inference", "autonomous",
    ]
    node_icon: ClassVar[Optional[str]] = "robot-face"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT

    def __init__(self, title: str = "Silk Agent", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # ── State (persisted via get_state/restore_state) ──
        self._history: List[Dict[str, Any]] = []
        self._last_run_ok: bool = False
        self._session_id: str = str(uuid.uuid4())  # persistent pool session key
        # The run's decision seam, and the request currently on screen.
        # Both are None between runs; `cancel_compute` reads the seam, which
        # is why it lives on the node rather than in `compute`'s frame.
        self._seam: Optional[DecisionSeam] = None
        #: The canvas seam and its main-thread resolver, per run (D70).
        self._canvas_seam: Optional[MainThreadCall] = None
        self._canvas_author: Optional[CanvasAuthor] = None
        self._pending_decision: Optional[dict] = None
        #: This run's caps, built once from the budget field and handed
        #: to whoever asks first -- the Orchestrator needs the object
        #: before the engine exists, and both must count into the same
        #: one. ``None`` between runs (D26).
        self._run_budget: Optional[Any] = None
        self._run_budget_built: bool = False
        self._emit_event: Optional[Any] = None

        # ── Ports ──
        self.add_input("model_obj", datatype="gguf_model")
        self.add_input("toolset", datatype="silk_toolset")
        self.add_input("role", datatype="silk_role")
        self.add_input("system_prompt", datatype="string")
        self.add_input("user_prompt", datatype="string")
        # Clean A2A: an inbound AgentMessage becomes the task when no direct
        # prompt is given, and its provenance/correlation flow to the outbox.
        self.add_input("inbox", datatype="agent_message")
        self.add_input("run", datatype="exec")  # pulse → auto-execute (§13)
        self.add_input(
            "inference_settings", datatype="dict",
        )  # optional gen_params dict from an Inference Settings node
        # The last link of the ToolSet → Role → Agent chain (D16). Whatever
        # arrives here narrows the toolset's sandbox for this run and only
        # this run; it can never widen it, and it cannot switch confinement
        # back on (D18).
        self.add_input("permissions", datatype="file_permissions")
        # A ceiling for the run, in text, because a run that may cost
        # anything is the default and someone has to be able to say
        # otherwise without writing Python (D26). Wired input wins.
        self.add_input("budget", datatype="string")
        # The run log switch, wirable like every other setting: a graph
        # that turns recording on for a batch of runs should not need
        # someone to tick a box first (T7, D85).
        self.add_input("event_log", datatype="bool")

        self.add_output("response", datatype="string")
        # Clean A2A: the reply wrapped as a self-describing AgentMessage
        # (sender = this agent, correlation echoing any inbound message).
        self.add_output("outbox", datatype="agent_message")
        # One typed vocabulary, one port (spec D2/D3). Everything the run
        # says -- lifecycle, model rounds, tool calls, plan advances, chat
        # turns, decisions -- arrives here as a dict carrying `type`, plus
        # the run identity every consumer needs to merge streams. The old
        # `tool_events`, `plan_events` and `chat_turn` ports are gone; a
        # saved graph wired to them must be re-wired (a hard break, D3).
        self.add_output("events", datatype="dict")
        self.add_output("done", datatype="exec")  # pulses when a run finishes

        # ── Layout & WidgetCore ──
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        form = QVBoxLayout()

        self._widget_core = WidgetCore()
        self._widget_core.set_node(self)

        # Prompt editor (Enter to send, Shift+Enter for newline).
        self.text_prompt = MarkdownWidget(mode="editor")
        self.text_prompt._text_edit.setPlaceholderText(
            "Task for the agent (Shift+Enter for newline)…"
        )
        self.text_prompt._text_edit.setMaximumHeight(80)
        self.text_prompt._text_edit.installEventFilter(self)
        form.addWidget(QLabel("Task:"))
        form.addWidget(self.text_prompt)
        self._widget_core.register_widget(
            "user_prompt", self.text_prompt, role=PortRole.BIDIRECTIONAL,
            datatype="string", default="", add_to_layout=False,
        )

        # Clear context button (always visible)
        controls = QHBoxLayout()
        self.btn_clear = SyncButton(initial_text="Clear Context")
        self.btn_clear.clicked.connect(self._clear_context)
        controls.addWidget(self.btn_clear)
        controls.addStretch()
        form.addLayout(controls)

        self._widget_core.register_widget(
            "btn_clear", self.btn_clear, role=PortRole.INTERNAL, add_to_layout=False,
        )

        # Live streaming preview.
        form.addWidget(QLabel("Preview:"))
        self.text_preview = MarkdownWidget(mode="display")
        self.text_preview._text_edit.setPlaceholderText("Agent response preview…")
        form.addWidget(self.text_preview)
        self._widget_core.register_widget(
            "preview_display", self.text_preview, role=PortRole.DISPLAY,
            datatype="string", default="", add_to_layout=False,
        )

        # Approval prompt. Hidden until a gated call blocks on it: the
        # decision surface is *in* the node because the run is inside
        # compute(), where no graph channel can reach it (D48/I12).
        self._decision_box = QWidget()
        decision_layout = QVBoxLayout(self._decision_box)
        decision_layout.setContentsMargins(0, 0, 0, 0)
        self._label_decision = QLabel("")
        self._label_decision.setWordWrap(True)
        self._label_decision.setStyleSheet("font-weight: bold;")
        decision_layout.addWidget(self._label_decision)
        buttons = QHBoxLayout()
        # Deny first, and it is what Escape-by-closing amounts to: the
        # safe answer should never be the one that takes an extra look.
        for label, approved, remember in (
            ("Deny", False, SCOPE_ONCE),
            ("Allow once", True, SCOPE_ONCE),
            ("Allow this run", True, SCOPE_RUN),
            ("Always allow", True, SCOPE_ALWAYS),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, a=approved, r=remember:
                    self._answer_decision(a, r)
            )
            buttons.addWidget(button)
        decision_layout.addLayout(buttons)
        self._decision_box.setVisible(False)
        form.addWidget(self._decision_box)
        self._widget_core.register_widget(
            "decision", self._decision_box, role=PortRole.INTERNAL,
            add_to_layout=False,
        )

        # A durable account of the run (T7, D85). Off by default: a run
        # log is a file this node writes on the user's disk, and the
        # honest default for that is "only when asked". What it records
        # is shape and size, never text -- see `functions/event_sink.py`.
        self.check_event_log = QCheckBox("Record run log (metadata only)")
        self.check_event_log.setToolTip(
            "Write one JSONL file per run to ~/.weave/silk/runs/: which "
            "events happened, in what order, how big. No prompts, no "
            "completions, no tool arguments -- only their names and "
            "lengths. Compaction drops turns permanently, so this is the "
            "only account of a long run's middle."
        )
        form.addWidget(self.check_event_log)
        self._widget_core.register_widget(
            "event_log", self.check_event_log, role=PortRole.INPUT,
            datatype="bool", default=False, add_to_layout=False,
        )

        # The run's ceiling. Empty means uncapped, which is what every
        # run did before this field existed; anything unreadable stops
        # the run rather than quietly running without the cap.
        self.edit_budget = QLineEdit()
        self.edit_budget.setPlaceholderText(
            "Budget (empty = no cap), e.g. requests=20, tools=50, output=8k"
        )
        self.edit_budget.setToolTip(
            "Caps for this run, comma separated: requests, tool_calls, "
            "output_tokens, input_tokens. '8k' means 8000. An "
            "unreadable budget refuses the run."
        )
        form.addWidget(QLabel("Budget:"))
        form.addWidget(self.edit_budget)
        self._widget_core.register_widget(
            "budget", self.edit_budget, role=PortRole.INPUT,
            datatype="string", default="", policy=debounced(300),
            add_to_layout=False,
        )

        # Status readout (role, tool activity, errors).
        self._label_status = QLabel("Idle.")
        self._label_status.setWordWrap(True)
        form.addWidget(QLabel("Status:"))
        form.addWidget(self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        # Run/Cancel action button.
        self.btn_run = SyncButton(initial_text="Run Agent")
        self.btn_run.setFixedHeight(30)
        self.btn_run.setStyleSheet("font-weight: bold;")
        self.btn_run.clicked.connect(self.execute)

        layout.addLayout(form)
        layout.addWidget(self.btn_run)
        self._widget_core.register_widget(
            "btn_run", self.btn_run, role=PortRole.INTERNAL, add_to_layout=False,
        )

        # Subclasses add their own rows here (the Orchestrator's delegation
        # depth, D55). Kept as an attribute rather than rebuilt per subclass
        # so an added row lands inside the same form as everything else.
        self._form_layout = form

        container = QWidget()
        container.setLayout(layout)

        # ── Signal wiring (worker → main thread; disconnected in cleanup) ──
        self.chunk_streamed.connect(self._on_chunk_streamed)
        self.status_changed.connect(self._on_status_changed)
        self.decision_requested.connect(self._on_decision_requested)
        self.decision_settled.connect(self._on_decision_settled)

        # ── Mount ──
        self.set_content_widget(container)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── Enter-to-send ────────────────────────────────────────────────

    def eventFilter(self, obj: Any, event: QEvent) -> bool:
        if obj == self.text_prompt._text_edit and event.type() == QEvent.Type.KeyPress:
            key_event = cast(QKeyEvent, event)
            if key_event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if key_event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    return False
                self.execute()
                return True
        return super().eventFilter(obj, event)

    # ── State persistence ────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["silk_agent"] = {
            "history": self._history,
        }
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        with self._widget_core.suppress_signals():
            super().restore_state(state)
            saved = state.get("silk_agent", {})
            self._history = copy.deepcopy(saved.get("history", []))

    # ── UI slots ─────────────────────────────────────────────────────

    @Slot(str)
    def _on_chunk_streamed(self, text: str) -> None:
        self._widget_core.push_display("preview_display", text)
        sb = self.text_preview._text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    @Slot(str)
    def _on_status_changed(self, msg: str) -> None:
        self._widget_core.push_display("status", msg)

    @Slot(dict)
    def _on_decision_requested(self, request: dict) -> None:
        """Show the question. **Main thread**, while the run is blocked."""
        self._pending_decision = request
        self._label_decision.setText(str(request.get("prompt") or "Approve?"))
        self._decision_box.setVisible(True)
        # The canvas is the "who needs me" dashboard (D59): a blocked node
        # changes how it pulses, so a graph of ten agents shows which one
        # is waiting without opening anything.
        self._enter_waiting_state(True)
        # ... and the registry is the directory the Decision Inbox reads.
        # It holds a weak reference to this node and nothing that could
        # answer for it -- the dock answers by mirroring the widget above.
        DECISIONS.register(request, node=self,
                           agent=str(getattr(self, "title", "") or "agent"))

    @Slot(str)
    def _on_decision_settled(self, decision_id: str) -> None:
        settled = decision_id or str(
            (self._pending_decision or {}).get("decision_id") or ""
        )
        self._pending_decision = None
        self._label_decision.setText("")
        self._decision_box.setVisible(False)
        self._enter_waiting_state(False)
        if settled:
            DECISIONS.unregister(settled)

    def _enter_waiting_state(self, waiting: bool) -> None:
        """Blocked-on-a-human, on the canvas itself (D59).

        Reuses the shipped pulse animation rather than adding a visual
        vocabulary: the waveform changes to a heartbeat while a node is
        waiting and goes back to whatever the theme asked for after.
        """
        try:
            if waiting:
                self._pulse_waveform_before = self.get_pulse_waveform()
                # Order matters: starting the pulse re-reads the theme's
                # waveform, so asking for the heartbeat first would be
                # overwritten by the start.
                self._start_computing_pulse()
                self.set_pulse_waveform("heartbeat")
            else:
                previous = getattr(self, "_pulse_waveform_before", None)
                if previous:
                    self.set_pulse_waveform(previous)
                    self._pulse_waveform_before = ""
        except AttributeError:      # pragma: no cover - host without the mixin
            log.debug("Node has no pulse animation; waiting state is silent")

    def _answer_decision(self, approved: bool, remember: str) -> None:
        """Deliver the user's answer to the waiting run. **Main thread.**

        The seam decides whether the answer still applies -- the run may
        have been stopped or the request timed out while the panel was up --
        so nothing here reads or repairs its state (D42).
        """
        request, seam = self._pending_decision, self._seam
        self._on_decision_settled("")
        if not request or seam is None:
            return
        decision_id = str(request.get("decision_id") or "")
        kind = str(request.get("kind") or "approval")
        if approved:
            landed = seam.approve(decision_id, actor="user", kind=kind,
                                  remember=remember)
        else:
            landed = seam.deny(decision_id, actor="user", kind=kind,
                               reason="the user declined this call")
        if not landed:
            # Late: stopped, timed out, or already answered. The run has
            # long since been told no; saying so beats a silent no-op.
            self.status_changed.emit("That request had already expired.")
            return
        emit = self._emit_event
        if emit is not None:
            emit(EventDecisionResponse(
                decision_id=decision_id, kind=kind, approved=approved,
                actor="user",
                reason="" if approved else "the user declined this call",
            ))

    def _clear_context(self) -> None:
        # Release the agent's dedicated pool instance so its KV cache is
        # wiped and returned to the general idle queue.
        cached_val = self._get_cached_value("model_obj")
        if isinstance(cached_val, dict) and "pool" in cached_val:
            pool = cached_val["pool"]
            release = getattr(pool, "release_session", None)
            try:
                if release is not None:
                    release(self._session_id)
            except Exception as exc:
                log.debug(f"Clear context: pool release failed: {exc}")

        self._history.clear()
        self._widget_core.push_display("preview_display", "<i>Agent context cleared.</i>")
        self._widget_core.push_display("status", "Idle.")

    # ── Execution ───────────────────────────────────────────────────

    def cancel_compute(self) -> None:
        """Stop, including a run that is blocked on a decision (D38).

        The loop's generator is inside a single ``next()`` while the gate
        waits and is polling nothing, so the stop flag alone would not be
        read until an answer arrived. Stop therefore cancels the seam
        *directly*; every waiter denies and the run unwinds normally.
        """
        seam = self._seam
        if seam is not None:
            seam.cancel("the user stopped the run")
        self.decision_settled.emit("")
        super().cancel_compute()

    def execute(self) -> None:
        if self._is_computing:
            self.cancel_compute()
            return
        self.btn_run.set_label("Cancel Run")
        self._widget_core.push_display("preview_display", "<i>Running…</i>")
        super().execute()

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self.btn_run.set_label("Run Agent")
        if self._last_run_ok:
            self._widget_core.apply_port_value("user_prompt", "")
            # Edge-trigger downstream agents / sinks in the network.
            self.pulse("done", payload=True)

    def _cleanup_after_worker(self, skip_results: bool = False) -> None:
        # The keyword is not decoration: Weave calls this with
        # `skip_results=True` when a result arrives for a cancelled or
        # disabled node, and an override without it raises TypeError on
        # exactly the path that is already going wrong.
        self.btn_run.set_label("Run Agent")
        super()._cleanup_after_worker(skip_results=skip_results)

    def cleanup(self) -> None:
        self.cancel_compute()
        # A node deleted mid-question leaves a row nobody can answer. The
        # registry's weak reference would prune it eventually; saying so
        # now means the dock never offers the dead button at all.
        pending = self._pending_decision or {}
        if pending.get("decision_id"):
            DECISIONS.unregister(str(pending["decision_id"]))
        try:
            self.chunk_streamed.disconnect()
        except (RuntimeError, TypeError):
            pass  # already disconnected / node partially destroyed
        try:
            self.status_changed.disconnect()
        except (RuntimeError, TypeError):
            pass
        super().cleanup()

    # ── Worker thread ────────────────────────────────────────────────

    def take_run_budget(self, inputs: Dict[str, Any]) -> Optional[Any]:
        """This run's :class:`UsageLimits`, built once (spec D26).

        The Orchestrator needs the object before ``compute`` builds an
        engine, because the same instance is what every worker spends
        from -- a fan-out whose orchestrator and workers counted into
        two objects would have a cap that means nothing. So whoever
        asks first builds it and the other gets the same one; it is
        released at the end of the run, since counters that survived
        into the next run would make a second run start exhausted.

        Raises:
            ValueError: if the field cannot be read. Callers turn this
                into a refused run.
        """
        if not self._run_budget_built:
            self._run_budget = parse_budget(inputs.get("budget"))
            self._run_budget_built = True
        return self._run_budget

    def release_run_budget(self) -> None:
        """Forget this run's counters; the next run starts with its own."""
        self._run_budget = None
        self._run_budget_built = False

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._last_run_ok = False
        model_handle = inputs.get("model_obj")
        if (
            not isinstance(model_handle, dict)
            or model_handle.get("backend") != "gguf"
            or not ("model" in model_handle or "pool" in model_handle)
        ):
            self.compute_error.emit("No valid GGUF model connected.")
            return {"response": "Error: no valid GGUF model connected."}

        prompt = str(inputs.get("user_prompt") or "").strip()
        # Clean A2A: fall back to an inbound message when no direct prompt is
        # wired; its content becomes the task, prefaced with the sender's
        # provenance. The message threads through to the outbox reply.
        inbox_raw = inputs.get("inbox")
        inbox_msg: Optional[AgentMessage] = None
        if isinstance(inbox_raw, dict) and inbox_raw.get("content"):
            inbox_msg = AgentMessage.from_dict(inbox_raw)
            if not prompt:
                prompt = f"{inbox_msg.context_header()}\n{inbox_msg.content}".strip()

        if not prompt:
            self.compute_error.emit("Empty task prompt.")
            return {"response": "Error: empty task prompt."}

        # A budget nobody can read is not a budget. Refusing here --
        # before the role is bound or a token is spent -- is the whole
        # point: running anyway would run *without* the cap that was
        # asked for, and the run would look like it obeyed one (D26).
        try:
            budget = self.take_run_budget(inputs)
        except ValueError as exc:
            self.release_run_budget()
            self.compute_error.emit(f"Budget: {exc}")
            return {"response": f"Error: budget not readable - {exc}"}

        toolset = inputs.get("toolset")
        role = inputs.get("role") or DEFAULT_ROLE

        # File access, narrowed once more and applied in place: the file
        # tools closed over this sandbox when they were registered, so a
        # replacement object would be built and ignored, and rebuilding the
        # ToolBox would drop what was attached to it live -- the
        # orchestrator's delegation tools, this run's approval gate (D16).
        file_scope = ExitStack()
        grants = self.file_grants(inputs, role)
        sandbox = getattr(toolset, "base_sandbox", None)
        if grants is not None and sandbox is not None:
            file_scope.enter_context(
                sandbox.restrict({e.path: e.mode for e in grants.entries})
            )
            log.debug(f"Run file access narrowed to: {grants.summary()}")

        binding: Optional[RoleBinding] = None
        event_hooks: list = []
        # Set once the per-run event emitter exists (only when a toolset is
        # wired — no toolset, no hook stream). Subclasses observe through it.
        emit_run_event: Optional[Any] = None
        # Named before the try so the exit path can release this run's
        # decision rows even when the run never got as far as an id.
        run_id = ""
        # How the run ended, taken from the result event rather than
        # guessed from what came back (G13).
        outcome = OUTCOME_COMPLETED
        # The write-watching middleware, when this run has the load verbs.
        change_entry = None
        try:
            if toolset is not None:
                try:
                    binding = RoleBinding.activate(role, toolset)
                except RuntimeError as exc:
                    # Another agent node holds this toolset right now.
                    self.compute_error.emit(str(exc))
                    return {"response": f"Error: {exc}"}

                # Observability: feed the tool_events output from the hook
                # system (registered per run, removed in finally). run_id +
                # seq give downstream monitors a dedup key across graph
                # re-evaluations.
                run_id = str(uuid.uuid4())
                seq = itertools.count()
                plan_rev = {"n": None}  # last plan revision streamed (dedup)
                # Who this is. A run_id alone identifies a *run*; once two
                # agents' streams are merged anywhere -- a Hook Monitor, the
                # Task Hub, an orchestrator re-emitting its workers -- there
                # is no way back to which node produced a line (spec D60.1).
                # Title for a human, uuid because titles are not unique.
                identity = {
                    "agent": str(getattr(self, "title", "") or "agent"),
                    "agent_id": str(getattr(self, "unique_id", "") or ""),
                }

                # One file per run, opened by its first event and closed
                # in the finally below (T7, D85).
                sink = RunSink() if bool(inputs.get("event_log")) else None
                self._run_sink = sink

                def _emit_event(event: Any, **extra: Any) -> None:
                    """Put one typed event on the `events` port.

                    The envelope is applied here and nowhere else: run_id +
                    seq are the dedup key across graph re-evaluations, and
                    the identity pair says which node produced the line once
                    two streams are merged (spec D60.1).
                    """
                    wire = to_wire(event, run_id=run_id, seq=next(seq),
                                   **identity, **extra)
                    self.emit_stream("events", wire, throttle_ms=0)
                    if sink is not None:
                        # Redacted inside the sink, not here: the widget
                        # on the other end of the port still wants the
                        # whole event.
                        sink.write(wire)

                # Handed to subclasses (the Orchestrator node) so a worker's
                # events can be re-emitted on this node's own stream.
                emit_run_event = _emit_event
                # ... and to the main thread, so an answered decision can
                # put its response on the same port the request went out on.
                self._emit_event = _emit_event

                def _ask(request: DecisionRequest) -> None:
                    """Put a decision on screen. **Worker thread**, blocked.

                    Outbound on the emission path, inbound through the
                    seam's threading primitive: the loop's generator is
                    mid-next() and cannot yield one (D48).
                    """
                    _emit_event(EventDecisionRequest(
                        decision_id=request.decision_id, kind=request.kind,
                        prompt=request.prompt, tool_name=request.tool_name,
                        tool_args=dict(request.tool_args),
                    ))
                    self.decision_requested.emit({
                        "decision_id": request.decision_id,
                        "kind": request.kind,
                        "prompt": request.prompt,
                        "tool_name": request.tool_name,
                        # Carried so the exit path can release what this
                        # run asked and never answered (D59).
                        "run_id": run_id,
                    })
                    self.status_changed.emit("Waiting for your approval\u2026")

                self._seam = DecisionSeam(_ask, timeout_s=DEFAULT_TIMEOUT_S)
                bind_run_seam(toolset, self._seam)

                # The same identity, for the ledger's keys (§17, D60): a
                # remembered turn and a streamed event must be joinable, so
                # the run id the memory hook files under is the run id on
                # the events port -- not one it invented for itself.
                bind_run_identity(toolset, RunIdentity(
                    run_id=run_id, agent=identity["agent"],
                    agent_id=identity["agent_id"], session=self._session_id,
                ))

                # The canvas seam (D70): the same waiter, resolved by the
                # main thread instead of a person. Bound per run and per
                # agent, so the run scope that says "you may remove only
                # what you placed" (D73) cannot outlive its run.
                self._canvas_seam = MainThreadCall(timeout_s=CANVAS_TIMEOUT_S)
                self._canvas_author = CanvasAuthor(self, self._canvas_seam)
                self._canvas_seam.attach(self._canvas_author.deliver)
                bind_canvas(toolset, CanvasBinding(
                    seam=self._canvas_seam, scope=RunScope(),
                    agent_uid=str(getattr(self, "unique_id", "")),
                ))

                # What this run writes, so a load approval can show the
                # diff rather than a suite name (D77). Costs one dict
                # entry per file touched, and only when the load verbs
                # are mounted -- an agent that cannot load code has
                # nothing to show anybody.
                if any(name in toolset.tools for name in ALWAYS_APPROVE):
                    change_entry = attach_change_tracking(toolset, ChangeSet())

                def _emit_plan_if_changed() -> None:
                    # Live plan updates for the Plan Viewer. No-op when the
                    # toolset has no task store; dedup-by-revision means reads
                    # and unchanged state never re-stream.
                    store = getattr(toolset, "_task_store", None)
                    event = (
                        plan_changed_event(store, plan_rev["n"])
                        if store is not None else None
                    )
                    if event is not None:
                        plan_rev["n"] = event["revision"]
                        _emit_event(EventPlan(revision=event["revision"],
                                              plan=event["plan"]))
                        # Nothing else is read out of the plan. The node used
                        # to infer "the user must approve something" from the
                        # plan's *shape* and end the turn; nothing is parked
                        # any more, so the inference is gone with it (D32).

                # The hook path. Each of these types has exactly one
                # producer (spec D2): the loop yields run.start, tool.call,
                # tool.result and the rest from its generator, so nothing
                # here re-announces them. What is left is what the loop
                # cannot say -- the per-round model events, a denial the
                # loop never sees, and run.finished, which invariant I2
                # guarantees on every exit path including the ones where no
                # EventRunResult is ever yielded.
                def _on_run_started(**_kw: Any) -> None:
                    _emit_plan_if_changed()  # show an existing/resumed plan

                def _on_run_finished(final_text: str = "", rounds: int = 0,
                                     elapsed_s: float = 0.0, **_kw: Any) -> None:
                    _emit_event(EventRunFinished(
                        rounds=rounds, elapsed_s=elapsed_s,
                        chars=len(final_text or ""),
                    ))

                def _on_model_request(round_index: int = 0, **_kw: Any) -> None:
                    _emit_event(EventModelRequest(round=round_index + 1))

                def _on_model_response(text: str = "", round_index: int = 0,
                                       finish_reason: Any = None,
                                       **_kw: Any) -> None:
                    _emit_event(EventModelResponse(
                        round=round_index + 1, chars=len(text or ""),
                        finish_reason=finish_reason,
                    ))

                def _on_after(tool_name: str = "", tool_result: str = "",
                              **_kw: Any) -> None:
                    _emit_plan_if_changed()  # a plan mutation bumps the revision

                def _on_denied(tool_name: str = "", **_kw: Any) -> None:
                    _emit_event(EventToolDenied(tool_name=tool_name))

                for event_name, callback in (
                    (HOOK_BEFORE_RUN, _on_run_started),
                    (HOOK_AFTER_RUN, _on_run_finished),
                    (HOOK_BEFORE_MODEL_REQUEST, _on_model_request),
                    (HOOK_AFTER_MODEL_RESPONSE, _on_model_response),
                    (HOOK_AFTER_TOOL_EXECUTE, _on_after),
                    (HOOK_TOOL_DENIED, _on_denied),
                ):
                    toolset.hooks.register(event_name, callback)
                    event_hooks.append((event_name, callback))

            system_prompt = self._compose_system_prompt(
                str(inputs.get("system_prompt") or ""), role, toolset
            )

            engine = GraphEngine(
                model_handle,
                system_prompt=system_prompt,
                history=self._history,
                session_id=self._session_id,
                usage_limits=budget,
            )
            if budget is not None:
                log.info(f"Run budget: {describe_budget(budget)}.")
            engine.clear_stop()

            loop = AgentLoop(
                engine,
                toolset,
                max_rounds=role.max_rounds or DEFAULT_MAX_ROUNDS,
                # A full context stops being the end of the run (D24). The
                # compactor is inert until there is pressure to answer --
                # and stays inert when the backend does not report a
                # context window, because there is then no pressure to
                # measure and compacting on a hunch costs two prefills.
                compactor=Compactor(),
            )

            # Build gen_params from internal defaults (no limit on tokens).
            # When an Inference Settings node is wired, its dict overrides
            # these defaults entirely.
            gen_params: Dict[str, Any] = {
                "temperature": 0.7,
            }

            # Merge external inference settings (from an Inference Settings
            # node) — they override the agent's internal fallback defaults.
            external_settings = inputs.get("inference_settings")
            if isinstance(external_settings, dict):
                gen_params.update(external_settings)

            if binding is not None:
                gen_params = binding.effective_gen_params(gen_params)

            self.status_changed.emit(f"Role '{role.id}' active — thinking…")

            # Subclass hook: everything the run needs is now built, and the
            # loop has not started. The Orchestrator node uses this to make
            # its workers observable and interruptible (spec D54).
            self._attach_run_observers(toolset, emit_run_event)

            final_text = ""
            run_error: Optional[str] = None
            # Ordered tool turns between the user prompt and the AI answer,
            # so the Chat Log can render tools as first-class turns.
            tool_turns: list[dict[str, Any]] = []
            for event in loop.run(prompt, gen_params):
                if self.is_compute_cancelled():
                    # Cancellation ends the run at the next round boundary
                    # (the current round drains normally).
                    engine.request_stop()

                # Every loop event is part of the one vocabulary, so it
                # goes out the one port -- content-light by the wire format,
                # which is where that rule is enforced rather than at each
                # emit site. Deltas are the exception to *emitting*: one
                # event per token would flood the graph, and `response`
                # already carries the text.
                if emit_run_event is not None and not isinstance(event, EventDelta):
                    emit_run_event(event)

                if isinstance(event, EventDelta):
                    self.chunk_streamed.emit(event.cumulative_text)
                    self.emit_stream("response", event.cumulative_text, throttle_ms=50)
                elif isinstance(event, EventToolCall):
                    self.status_changed.emit(f"Tool: {event.tool_name}…")
                    tool_turns.append({
                        "role": "tool_call",
                        "tool": event.tool_name,
                        "args": event.tool_args,
                        "call_id": event.call_id,
                    })
                elif isinstance(event, EventToolResult):
                    marker = "error" if event.error else "ok"
                    self.status_changed.emit(f"Tool {event.tool_name}: {marker}")
                    tool_turns.append({
                        "role": "tool_result",
                        "tool": event.tool_name,
                        "result": event.result,
                        "error": bool(event.error),
                    })
                elif isinstance(event, EventCompaction):
                    self.status_changed.emit(
                        f"Compacted {event.turns_dropped} turns to free context…"
                    )
                elif isinstance(event, EventReflection):
                    self.status_changed.emit(
                        f"Reflection retry {event.retry_count + 1}/{event.max_retries}…"
                    )
                elif isinstance(event, EventError):
                    run_error = event.error
                elif isinstance(event, EventRunResult):
                    final_text = event.text
                    outcome = event.outcome or OUTCOME_COMPLETED

            if run_error and not final_text:
                self.compute_error.emit(run_error)
                self.status_changed.emit(f"Failed: {run_error}")
                return {"response": f"Error: {run_error}"}

            # How the run ended is a field, not an inference (G13). A run
            # that exhausted its rounds or its budget has a final text and
            # is not a success; reading "there is text" is what made those
            # endings show as "Done." for as long as the field existed
            # without a consumer.
            self._last_run_ok = (
                outcome == OUTCOME_COMPLETED and not self.is_compute_cancelled()
            )
            if outcome != OUTCOME_COMPLETED:
                self.status_changed.emit(
                    f"Ended: {outcome}"
                    + (f" - {run_error}" if run_error else "")
                )
            refused = (headless_refusals(toolset)
                       if toolset is not None else 0)
            if refused and self._last_run_ok:
                # A headless run refusing every gated call is behaving
                # correctly and looks broken. Once, at the end, with the
                # number -- not once per call (§22 q1d, D53).
                self.status_changed.emit(
                    f"Done — {refused} call(s) refused: nothing in this "
                    f"graph could ask you.")
            else:
                self.status_changed.emit(
                    "Done." if self._last_run_ok else "Cancelled.")

            agent_name = str(getattr(self, "title", "") or "agent")
            out_kind = "result" if self._last_run_ok else "error"
            outbox = (
                inbox_msg.reply(final_text, sender=agent_name, kind=out_kind)
                if inbox_msg is not None
                else AgentMessage(content=final_text, sender=agent_name,
                                  kind=out_kind)
            )
            # The chat turn is an event now, not a port (D3). It is one of
            # the two deliberately content-carrying members of the
            # vocabulary: a chat log that cannot show what was said is not a
            # chat log.
            if emit_run_event is not None:
                emit_run_event(EventChatTurn(
                    turn_id=str(uuid.uuid4()),
                    user=prompt or "",
                    ai=final_text,
                    turns=tool_turns,
                ))

            return {
                "response": final_text,
                "outbox": outbox.to_dict(),
            }
        except Exception as exc:  # never let the worker die silently
            log.error(f"Agent run failed: {exc}", exc_info=True)
            self.compute_error.emit(str(exc))
            return {"response": f"Error: {exc}"}
        finally:
            # The seam dies with the run it belongs to: close it before
            # anything else, so a call still blocked on the way out is
            # denied rather than left waiting on a node that has moved on.
            if self._seam is not None:
                self._seam.close()
                self._seam = None
            if self._canvas_seam is not None:
                self._canvas_seam.close()
                self._canvas_seam = None
            self._canvas_author = None
            self._emit_event = None
            if self._run_sink is not None:
                self._run_sink.close()
                self._run_sink = None
            # An identity left bound would file the next run's turns under
            # this run's id, and the ledger is append-only: a wrong turn
            # can only be superseded afterwards, never corrected.
            if toolset is not None:
                bind_run_identity(toolset, None)
            # The budget is the run's, not the node's: a second run
            # inheriting the first one's counters would start already
            # spent (D26).
            self.release_run_budget()
            self.decision_settled.emit("")
            # A stopped or timed-out run never answers what it asked, and a
            # row for an agent that is no longer running is a button that
            # does nothing (D59).
            if run_id:
                DECISIONS.clear_run(run_id)
            if toolset is not None:
                bind_run_seam(toolset, None)
                bind_canvas(toolset, None)
                # The change record is this run's, like the run scope: the
                # next run's approval must not show diffs this one wrote.
                bind_changes(toolset, None)
                if change_entry is not None:
                    toolset.hooks.unregister_middleware(
                        HOOK_WRAP_TOOL_EXECUTE, change_entry.callback)
                    change_entry = None
            self._detach_run_observers(toolset)
            if toolset is not None:
                for event_name, callback in event_hooks:
                    toolset.hooks.unregister(event_name, callback)
            if binding is not None:
                binding.deactivate()
            # The sandbox is a live graph object shared with the next run,
            # so the narrowing is undone here even when the run failed.
            file_scope.close()

    # -- file access (spec D16/D17) ------------------------------------------

    @staticmethod
    def file_grants(inputs: Dict[str, Any], role: Any) -> Any:
        """The grant this run may use, or ``None`` for "nothing was said".

        Two ways in -- the one that travelled through the Role, and one
        wired straight to this node -- composed by narrowing in
        :func:`resolve_grants`, so adding a wire can only reduce access.
        """
        return resolve_grants(inputs.get("permissions"),
                              getattr(role, "file_grants", None))

    # -- run-scoped observers (subclass seam) --------------------------------

    def _attach_run_observers(self, toolset: Any, emit_event: Any) -> None:
        """Install anything that must watch this run. No-op for a plain agent.

        Called after the engine, loop and role binding exist and before the
        first event is pulled, so an override sees a fully built run.

        Args:
            toolset: the ToolBox this run executes on (may be ``None``).
            emit_event: ``emit_event(event)`` — the same emitter the node's
                own hook callbacks use, taking a typed event from the
                vocabulary, so anything an override forwards lands on
                ``events`` already stamped with the run id, sequence number
                and agent identity. ``None`` when no toolset is wired.
        """

    def _detach_run_observers(self, toolset: Any) -> None:
        """Remove what :meth:`_attach_run_observers` installed. No-op here."""

    @staticmethod
    def _compose_system_prompt(base: str, role: Any, toolset: Any) -> str:
        """base prompt + [ROLE] block + capability/procedure blocks + tool protocol.

        Delegates to the shared, Qt-free composer so the Agent node and the
        sub-agent runner build identical system prompts.
        """
        return compose_system_prompt(base, role, toolset)
