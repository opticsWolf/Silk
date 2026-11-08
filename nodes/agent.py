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
The prompt comes from outside: a Silk Prompt node is the chat box
(``send`` → ``run``, carrying the message as the pulse payload;
``stop`` → ``stop``; ``new_chat`` → ``clear``), or any text on
``user_prompt``. The reply is read downstream (``events`` → Chat Log
Display). The node itself keeps one icon row: Run/Stop and Clear
context. A message sent during a run is queued for the next one, or --
sent as a steer or an interrupt -- joins the run itself; what waits is
listed under the icon row, where each message can be re-filed or
removed. Stop is press-twice: after the current action, then now. On
completion the node pulses its ``done`` Exec output. Chain agent networks by wiring
``response`` → next agent's ``user_prompt`` and ``done`` → next agent's
``run``.

Threading: ``compute()`` drives the AgentLoop on the worker thread; deltas
reach the UI via a queued Qt signal and downstream nodes via
``emit_stream``. Cancellation maps onto the engine's stop flag.
"""

import copy
import itertools
import time
import uuid
from contextlib import ExitStack
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtCore import QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.widgetcore.binding_policy import debounced
from weave.node.threaded import ThreadedManualNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.canvas.rust_control import served_control
from weave.logger import get_logger

from weave.library.sync_tool_button.widgets.sync_tool_button import SyncToolButton

from .silk_ports import (  # noqa: F401
    MODEL_HANDLE_TYPE, SILK_CLASSIFIER_TYPE, SILK_ROLE_TYPE, SILK_TOOLSET_TYPE,
)
from ..functions.agent_loop import AgentLoop, DEFAULT_MAX_ROUNDS
from ..functions.run_control import (
    MODE_QUEUE, MODE_STEER, MODES, STOP_NONE, RunControl, merge_texts,
)
from ..widgets.run_queue import RunQueueView, StopEscalation
from ..functions.event_sink import RunSink, sink_for
from ..functions.approval import bind_run_seam, headless_refusals
from ..functions.classifier_binding import begin_classifier_run, end_classifier_run
from ..functions.graph_author import CanvasBinding, RunScope, bind_canvas
from ..functions.tools.node_dev_tools import bind_weave
from ..functions.self_modify import (
    ALWAYS_APPROVE, ChangeSet, attach_change_tracking, bind_changes,
)
from ..functions.main_thread_call import (
    DEFAULT_TIMEOUT_S as CANVAS_TIMEOUT_S, MainThreadCall,
)
from .graph_canvas import CanvasAuthor
from ..functions.compaction import compactor_for
from ..functions.file_grants import resolve_grants
from ..functions.decision_registry import REGISTRY as DECISIONS
from ..functions.decision_seam import (
    DEFAULT_TIMEOUT_S,
    DecisionRequest,
    DecisionSeam,
)
from ..functions.grants import SCOPE_ALWAYS, SCOPE_ONCE, SCOPE_RUN
from ..functions.graph_engine import GraphEngine, handle_gen_params
from ..functions.remember import RunIdentity, bind_run_identity
from ..functions.cost_ledger import record_run
from ..functions.port_text import text_input
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
    OUTCOME_ERROR,
    EventChatTurn,
    EventCompaction,
    EventContextCleared,
    EventDecisionRequest,
    EventDecisionResponse,
    EventDelta,
    EventError,
    EventModelRequest,
    EventModelResponse,
    EventPlan,
    EventModelSwitch,
    EventReflection,
    EventRunFinished,
    EventRunResult,
    EventSteered,
    EventToolCall,
    EventToolDenied,
    EventToolResult,
    to_wire,
)
from ..functions.subagent import compose_system_prompt, handle_supports_tools

log = get_logger("SilkAgent")


def _message(payload: Any) -> tuple[str, str]:
    """A `run` pulse's payload as (text, mode).

    A string is a queued message; a dict says its mode. Anything else
    is a bare pulse.
    """
    if isinstance(payload, dict):
        text = payload.get("text")
        mode = payload.get("mode", MODE_QUEUE)
    else:
        text, mode = payload, MODE_QUEUE
    text = text.strip() if isinstance(text, str) else ""
    return text, mode if mode in MODES else MODE_QUEUE


@register_node
class SilkAgentNode(ThreadedManualNode):
    """Autonomous tool-calling agent over model + toolbox + role."""
    node_state_api = 2   # owns a hand-written state dict
    # Bumped on any change to what this node persists or how its
    # ports are shaped; `node_state_api` is what says old state can
    # no longer be restored (G20).
    # 1 = the shape this spec settled on (the `event_log` port and box
    # came and went under it -- an older state simply has no entry and
    # restores with defaults, which the removal honors too).
    # 2: the `inference_settings` port is gone -- sampler settings ride
    # the model wire now (the loader stamps handle["gen_params"]). A wire
    # into it is dropped on load; re-draw from the loader instead.
    # 5: the Task box and Preview are gone (`user_prompt` is a plain
    # port), and `stop` / `clear` exec inputs are new.
    node_version = 5

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
    status_changed = Signal(str)
    # A gated tool call is blocked on the worker thread and needs the user
    # (D48). The request travels as a plain dict so the queued connection
    # carries nothing Qt has to marshal specially.
    decision_requested = Signal(dict)
    decision_settled = Signal(str)
    # The run control changed (a message posted, taken or removed, a
    # worker started or ended) -- on whatever thread did it.
    control_changed = Signal()

    node_class: ClassVar[str] = "Silk AI"
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

    _RUN_TIP: ClassVar[str] = (
        "Run the agent on the prompt wired into user_prompt (or on the "
        "messages waiting below). Press again to stop after the current "
        "action, a third time to cut it now; the arrow has both. A pulse "
        "on `run` starts it too (a Prompt node's send brings its message "
        "along); `stop` stops it the same way."
    )
    _CLEAR_TIP: ClassVar[str] = (
        "Clear context: forget this agent's conversation, so the next "
        "run starts a new chat (a pulse on `clear` does the same). The "
        "plan, the knowledge graph and always-allow grants are kept. "
        "Workers start fresh on every delegation, so there is nothing "
        "of theirs to clear."
    )

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
        #: Asked for while a run was on; done when it ends (clear first,
        #: then the next run, see `_cleanup_after_worker`). A bare `run`
        #: pulse queues a re-run; a message is queued on the control.
        self._run_queued: bool = False
        self._clear_pending: bool = False
        #: A prompt that came on a `run` pulse: the next run's.
        self._next_prompt: Optional[str] = None
        #: Queued, steering and interrupting messages, and the stop level;
        #: shared with the loop (and, under an Orchestrator, its workers).
        self._control = RunControl(label=str(title))
        #: The running engine, so a force stop reaches the stream even
        #: while the loop is inside a tool batch and yields nothing.
        self._engine: Optional[GraphEngine] = None

        # ── Ports ──
        self.add_input("model_obj", datatype="model_handle")
        self.add_input("toolset", datatype="silk_toolset")
        self.add_input("role", datatype="silk_role")
        self.add_input("system_prompt", datatype="string")
        # The prompt: the message this run answers. A Silk Prompt node,
        # or any text (an upstream agent's `response`, a file).
        self.add_input("user_prompt", datatype="string",
                       desc="Prompt: the message the next run answers.")
        # Clean A2A: an inbound AgentMessage becomes the task when no direct
        # prompt is given, and its provenance/correlation flow to the outbox.
        self.add_input("inbox", datatype="agent_message")
        self.add_input("run", datatype="exec")  # pulse → auto-execute (§13)
        # The Prompt node's Stop and New chat; any exec source works.
        self.add_input("stop", datatype="exec")
        self.add_input("clear", datatype="exec")
        # The last link of the ToolSet → Role → Agent chain (D16). Whatever
        # arrives here narrows the toolset's sandbox for this run and only
        # this run; it can never widen it, and it cannot switch confinement
        # back on (D18).
        self.add_input("permissions", datatype="file_permissions")
        # A ceiling for the run, in text, because a run that may cost
        # anything is the default and someone has to be able to say
        # otherwise without writing Python (D26). Wired input wins.
        self.add_input("budget", datatype="string")
        # The classifier resource (D-S8), next to the model: approval triage,
        # classifier hooks and the classify tool read it for one run at a
        # time. Unwired, `triage` levels ask the human (C3).
        self.add_input("classifier", datatype="silk_classifier")

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

        # One row: Run/Stop, Clear context, and what the run is doing.
        # The prompt comes in on `user_prompt` -- a Silk Prompt node, or
        # any text -- and the reply goes out on `response` and `events`,
        # so the conversation is read where it is written, not here.
        controls = QHBoxLayout()
        self.btn_run = SyncToolButton(
            initial_text="Start", show_label=False, dimensions=24,
        )
        self.btn_run.set_tooltip(self._RUN_TIP)
        self.btn_run.clicked.connect(self.execute)
        # Start, then Stop (after the current action), then -- pulsing red
        # while that stop is pending -- Force stop. The arrow offers both
        # stops directly.
        self._stop = StopEscalation(
            self.btn_run, on_stop=self._request_stop,
            idle_label="Start", idle_tip=self._RUN_TIP, connect=False,
            quiet=self._widget_core.suppress_signals,
        )
        stop_menu = QMenu(self.btn_run)
        self._act_stop = stop_menu.addAction("Stop after this action")
        self._act_stop.triggered.connect(lambda: self._request_stop(False))
        self._act_force = stop_menu.addAction("Force stop now")
        self._act_force.triggered.connect(lambda: self._request_stop(True))
        stop_menu.aboutToShow.connect(self._sync_stop_menu)
        self.btn_run.setMenu(stop_menu)
        self.btn_run.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        controls.addWidget(self.btn_run)
        self._widget_core.register_widget(
            "btn_run", self.btn_run, role=PortRole.INTERNAL, add_to_layout=False,
        )

        self.btn_clear = SyncToolButton(
            initial_text="Clear", show_label=False, dimensions=24,
        )
        self.btn_clear.set_tooltip(self._CLEAR_TIP)
        self.btn_clear.clicked.connect(self.clear_context)
        controls.addWidget(self.btn_clear)
        self._widget_core.register_widget(
            "btn_clear", self.btn_clear, role=PortRole.INTERNAL, add_to_layout=False,
        )

        # Status readout (role, tool activity, errors).
        self._label_status = QLabel("Idle.")
        self._label_status.setWordWrap(True)
        controls.addWidget(self._label_status, stretch=1)
        form.addLayout(controls)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        # What waits on the run, and (Orchestrator) the workers running
        # now. Hidden while empty.
        self._queue_view = RunQueueView(self._control)
        form.addWidget(self._queue_view)

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

        layout.addLayout(form)

        # Subclasses add their own rows here (the Orchestrator's delegation
        # depth, D55). Kept as an attribute rather than rebuilt per subclass
        # so an added row lands inside the same form as everything else.
        self._form_layout = form

        container = QWidget()
        container.setLayout(layout)

        # ── Signal wiring (worker → main thread; disconnected in cleanup) ──
        self.status_changed.connect(self._on_status_changed)
        self.decision_requested.connect(self._on_decision_requested)
        self.decision_settled.connect(self._on_decision_settled)
        self.control_changed.connect(self._queue_view.refresh)
        self._control.subscribe(self.control_changed.emit)

        # ── Mount ──
        self.set_content_widget(container)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── State persistence ────────────────────────────────────────────

    @classmethod
    def migrate_state(cls, state: Dict[str, Any],
                      from_version: int) -> Dict[str, Any]:
        """api 1 → 2: drop any residue of the removed ``inference_settings``
        port; the dict now rides the model wire from the loader, so an old
        wire into the agent is dropped on load and re-drawn there.

        Any version: the Task box and the Preview are gone (the prompt is
        a wire now, the reply is read downstream). A typed-in task is not
        restored into anything; the wire, if there was one, still is.
        """
        values = state.get("widget_data")
        if isinstance(values, dict):
            if from_version < 2:
                values.pop("inference_settings", None)
            values.pop("user_prompt", None)
            values.pop("preview_display", None)
        return state

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
        # A state saved mid-run carries the button as Stop.
        self._show_run_button()
        values = state.get("widget_data")
        if isinstance(values, dict) and values.get("event_log"):
            log.warning(
                f"'{getattr(self, 'title', 'Silk Agent')}' had its run log "
                "switched on. The run log is now the ToolBox's 'run_log' "
                "hook: tick it there to keep recording."
            )

    # ── UI slots ─────────────────────────────────────────────────────

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

    def clear_context(self) -> None:
        """Forget the conversation: a new chat starts with the next run.

        A run in flight keeps the history it started with, so the clear
        waits for it to end. Says so on ``events`` (``context.cleared``),
        which is what tells a Chat Log on the same wire to start over.
        """
        if self._is_computing:
            self._clear_pending = True
            self._widget_core.push_display(
                "status", "Context clears when this run ends.")
            return
        self._clear_pending = False
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

        dropped = len(self._history)
        self._history.clear()
        self._widget_core.push_display("status", "Context cleared.")
        # Main thread and between runs, where the threaded stream gate
        # would drop it: straight to the wire.
        self._fan_out_stream("events", to_wire(
            EventContextCleared(turns_dropped=dropped), run_id="", seq=0,
            agent=str(getattr(self, "title", "") or "agent"),
            agent_id=str(getattr(self, "unique_id", "") or ""),
        ))

    # ── Exec inputs ─────────────────────────────────────────────────

    def on_pulse(self, port_name: str, sender: Any, payload: Any = None) -> None:
        """``run`` sends a message, ``stop`` stops, ``clear`` clears context.

        A ``run`` pulse may carry the message as its payload: a string is
        sent as queued, ``{"text", "mode"}`` says how (``queue``, ``steer``
        or ``interrupt``). It wins over ``user_prompt`` for that run. It
        rides the pulse rather than a data wire because a changed input
        stops a running node, and a message typed during a run is not a
        stop. Idle, any mode starts a run; during one, a queued message
        waits for it to end and a steer or interrupt joins it. Text that
        starts with ``@worker`` goes to that running worker instead.

        ``stop`` is the press-twice Stop: after the current action, and
        a second time (or a ``"force"`` payload) now.
        """
        if port_name == "stop":
            if self._is_computing:
                if payload == "force":
                    self._stop.press_force()
                else:
                    self._stop.press()
            return
        if port_name == "clear":
            self.clear_context()
            return
        text, mode = _message(payload)
        if text and self._is_computing and self._to_worker(text, mode):
            return
        if self._is_computing:
            if text:
                self._control.post(text, mode)
                self._widget_core.push_display("status", {
                    MODE_QUEUE: "Message queued: sent when this run ends.",
                    MODE_STEER: "Steering: joins the run at its next step.",
                }.get(mode, "Interrupting: cutting the current action…"))
            else:
                # A bare pulse (an upstream `done`): run again after this.
                self._run_queued = True
            return
        self._next_prompt = text or None
        self.execute()

    def _to_worker(self, text: str, mode: str) -> bool:
        """``@name message``: post to that running worker; False if none."""
        if not text.startswith("@"):
            return False
        name, _, rest = text[1:].partition(" ")
        rest = rest.strip()
        if not name or not rest:
            return False
        # A worker has no next run: a queued message steers it.
        mode = MODE_STEER if mode == MODE_QUEUE else mode
        stack, hit = list(self._control.children()), 0
        while stack:
            kid = stack.pop()
            if kid.label == name:
                kid.post(rest, mode)
                hit += 1
            stack.extend(kid.children())
        if hit:
            self._widget_core.push_display(
                "status", f"Sent to worker '{name}' ({mode}).")
        return bool(hit)

    # ── Execution ───────────────────────────────────────────────────

    def _request_stop(self, force: bool) -> None:
        """Stop the run: after the current action, or (force) now."""
        if not self._is_computing:
            return
        if force:
            self.cancel_compute()
            self._widget_core.push_display("status", "Force stopping…")
            return
        self._control.request_stop(force=False)
        self._stop.mark_stopping()
        self._widget_core.push_display(
            "status", "Stopping after the current action…")

    def _sync_stop_menu(self) -> None:
        running = bool(self._is_computing)
        self._act_stop.setEnabled(running and not self._stop.stopping)
        self._act_force.setEnabled(running)

    def cancel_compute(self) -> None:
        """Stop now, including a run that is blocked on a decision (D38).

        The loop's generator is inside a single ``next()`` while the gate
        waits and is polling nothing, so the stop flag alone would not be
        read until an answer arrived. Stop therefore cancels the seam
        *directly*; every waiter denies and the run unwinds normally. The
        engine is stopped directly too: during a tool batch the loop
        yields nothing, so a stop raised from the next event would land
        only after the batch.
        """
        self._control.request_stop(force=True)
        engine = self._engine
        if engine is not None:
            engine.request_stop()
        seam = self._seam
        if seam is not None:
            seam.cancel("the user stopped the run")
        self.decision_settled.emit("")
        super().cancel_compute()

    def execute(self) -> None:
        if self._is_computing:
            # The button is Stop while a run is on, Force stop while that
            # stop is pending.
            self._stop.press()
            return
        # A new run: the last one's stop is over, and whatever waited for
        # it (queued messages, steers that came too late) is this run's
        # prompt, ahead of a message that came with the pulse.
        self._control.reset()
        waiting = self._control.take_queued()
        if waiting:
            if self._next_prompt:
                waiting.append({"text": self._next_prompt})
            self._next_prompt = merge_texts(waiting)
        self._run_queued = False
        self._stop.set_running(True)
        self._widget_core.push_display("status", "Running…")
        super().execute()

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._show_run_button()
        if self._last_run_ok:
            # Edge-trigger downstream agents / sinks in the network.
            self.pulse("done", payload=True)

    def _cleanup_after_worker(self, skip_results: bool = False) -> None:
        # The keyword is not decoration: Weave calls this with
        # `skip_results=True` when a result arrives for a cancelled or
        # disabled node, and an override without it raises TypeError on
        # exactly the path that is already going wrong.
        stopped = self._control.stop_level() > STOP_NONE
        self._show_run_button()
        super()._cleanup_after_worker(skip_results=skip_results)
        # What arrived during the run, in the order that makes sense:
        # a new chat first, then the prompt that starts it.
        if self._clear_pending:
            self.clear_context()
        waiting = len(self._control.pending())
        if stopped:
            # A stop holds the queue rather than dropping it: the messages
            # stay listed (removable), and Start sends them.
            self._run_queued = False
            self._control.reset()
            if waiting:
                self._widget_core.push_display(
                    "status", f"Stopped — {waiting} message(s) waiting; "
                              "Start sends them.")
        elif waiting or self._run_queued:
            QTimer.singleShot(0, self.execute)

    def _show_run_button(self) -> None:
        self._stop.set_running(False)

    def cleanup(self) -> None:
        # Nothing waits on a node that is going away.
        self._run_queued = False
        self._clear_pending = False
        self._control.take_queued()
        self._stop.reset()
        self._queue_view.cleanup()
        self.cancel_compute()
        # A node deleted mid-question leaves a row nobody can answer. The
        # registry's weak reference would prune it eventually; saying so
        # now means the dock never offers the dead button at all.
        pending = self._pending_decision or {}
        if pending.get("decision_id"):
            DECISIONS.unregister(str(pending["decision_id"]))
        try:
            self.status_changed.disconnect()
        except (RuntimeError, TypeError):
            pass  # already disconnected / node partially destroyed
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
        # Any backend, not "gguf": the loop needs something that answers
        # `create_chat_completion`, and a remote endpoint answers it as
        # well as a local pool does (D45).
        if (
            not isinstance(model_handle, dict)
            or not model_handle.get("backend")
            or not ("model" in model_handle or "pool" in model_handle)
        ):
            self.compute_error.emit("No valid model connected.")
            return {"response": "Error: no valid model connected."}

        # Taken once: a prompt that came on the pulse is this run's only.
        pulse_prompt, self._next_prompt = self._next_prompt, None
        try:
            prompt = pulse_prompt or text_input(inputs, "user_prompt")
            system_text = text_input(inputs, "system_prompt")
        except ValueError as exc:
            self.compute_error.emit(f"Prompt: {exc}")
            return {"response": f"Error: prompt not readable - {exc}"}
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
            self.compute_error.emit("Empty prompt: wire one into user_prompt.")
            return {"response": "Error: empty prompt."}

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
        # The canvas's control server, when this run bound it for node-dev.
        served_weave = None
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
                # Switched and configured by the ToolBox's run_log hook.
                sink = sink_for(toolset)
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

                # The classifier, for this run only (D-S8, D38). A judge
                # that is this agent's own model refuses the run (C3).
                classifier_start = begin_classifier_run(
                    toolset, inputs.get("classifier"), run_id=run_id,
                    agent_model=model_handle,
                )
                if classifier_start.error:
                    self.compute_error.emit(classifier_start.error)
                    return {"response": f"Error: {classifier_start.error}"}
                for notice in classifier_start.notices:
                    self.status_changed.emit(notice)

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

                # Node-dev loads into and benches on a Weave through its
                # control API (P6): this canvas's, when it serves one.
                served_weave = served_control(self)
                if served_weave is not None:
                    bind_weave(toolset, served_weave)

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
                system_text, role, toolset,
                native_tools=toolset is not None
                and handle_supports_tools(model_handle),
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
            self._engine = engine
            if self._control.stop_level() > STOP_NONE:
                # Stopped between the press of Start and here.
                engine.request_stop()

            loop = AgentLoop(
                engine,
                toolset,
                max_rounds=role.max_rounds or DEFAULT_MAX_ROUNDS,
                control=self._control,
                # A full context stops being the end of the run (D24). The
                # compactor is inert until there is pressure to answer --
                # and stays inert when the backend does not report a
                # context window, because there is then no pressure to
                # measure and compacting on a hunch costs two prefills.
                # Configured by the ToolBox's auto_compact hook, adjusted
                # by the Role's; without either, only the compact_context
                # tool compacts.
                compactor=compactor_for(toolset, role),
            )

            # Build gen_params from internal defaults (no limit on tokens).
            # Sampling travels on the model wire now: the loader stamps a
            # wired Inference Settings dict onto handle["gen_params"], and
            # it wins over these fallbacks. The role still fills gaps.
            gen_params: Dict[str, Any] = {
                "temperature": 0.7,
            }

            model_gen = handle_gen_params(model_handle)
            gen_params.update(model_gen)

            if binding is not None:
                gen_params = binding.effective_gen_params(gen_params)

            self.status_changed.emit(f"Role '{role.id}' active — thinking…")

            # Subclass hook: everything the run needs is now built, and the
            # loop has not started. The Orchestrator node uses this to make
            # its workers observable and interruptible (spec D54).
            self._attach_run_observers(toolset, emit_run_event)

            final_text = ""
            run_error: Optional[str] = None
            run_started = time.time()
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
                elif isinstance(event, EventModelSwitch):
                    self.status_changed.emit(
                        f"{event.from_model} failed - falling back to "
                        f"{event.to_model}..."
                    )
                    # The failure that caused the switch has already been
                    # reported and is now recovered: leaving it in
                    # `run_error` would label a run that finished on the
                    # fallback with the primary's last words (D89).
                    run_error = None
                elif isinstance(event, EventSteered):
                    self.status_changed.emit(
                        "Interrupted — carrying on with your message…"
                        if event.interrupted else
                        "Steered — carrying on with your message…")
                    tool_turns.append({
                        "role": "steer",
                        "text": event.text,
                        "interrupted": event.interrupted,
                    })
                elif isinstance(event, EventReflection):
                    self.status_changed.emit(
                        f"Reflection retry {event.retry_count + 1}/{event.max_retries}…"
                    )
                elif isinstance(event, EventError):
                    run_error = event.error
                elif isinstance(event, EventRunResult):
                    final_text = event.text
                    outcome = event.outcome or OUTCOME_COMPLETED

            # The bill, before either exit (D90). A run that failed still
            # spent what it prefilled, so recording only on the way out
            # through success would under-report exactly the runs worth
            # looking at. Writes nothing at all when no model quoted a
            # price, which is why there is no checkbox for it.
            record_run(
                engine.spend_report(),
                run_id=run_id,
                session_id=self._session_id,
                agent=str(getattr(self, "title", "") or "agent"),
                elapsed_s=time.time() - run_started,
                outcome=outcome if not run_error else OUTCOME_ERROR,
            )

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
            self._engine = None
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
                # Unbound with the seam; says what triage let through.
                for notice in end_classifier_run(toolset):
                    self.status_changed.emit(notice)
                bind_canvas(toolset, None)
                if served_weave is not None:
                    bind_weave(toolset, None)
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
    def _compose_system_prompt(base: str, role: Any, toolset: Any,
                               native_tools: bool = False) -> str:
        """base prompt + [ROLE] block + capability/procedure blocks + tool protocol.

        Delegates to the shared, Qt-free composer so the Agent node and the
        sub-agent runner build identical system prompts.
        """
        return compose_system_prompt(base, role, toolset,
                                     native_tools=native_tools)
