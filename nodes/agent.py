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
from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtCore import Qt, QEvent, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from weave.widgetcore import WidgetCore, PortRole
from weave.node.threaded import ThreadedManualNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from weave.widgets.markdown_widget import MarkdownWidget
from weave.widgets.sync_button import SyncButton

from .silk_ports import GGUF_MODEL_TYPE, SILK_ROLE_TYPE, SILK_TOOLSET_TYPE  # noqa: F401
from ..functions.agent_loop import AgentLoop, DEFAULT_MAX_ROUNDS
from ..functions.graph_engine import GraphEngine
from ..functions.hooks import (
    HOOK_AFTER_MODEL_RESPONSE,
    HOOK_AFTER_RUN,
    HOOK_AFTER_TOOL_EXECUTE,
    HOOK_BEFORE_MODEL_REQUEST,
    HOOK_BEFORE_RUN,
    HOOK_TOOL_DENIED,
)
from ..functions.messaging import AgentMessage
from ..functions.role import DEFAULT_ROLE, RoleBinding
from ..functions.task_store import plan_changed_event  # Qt-free
from ..functions.stream_events import (
    EventChatTurn,
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

    # Worker → main-thread bridges (V6 R11.1).
    chunk_streamed = Signal(str)
    status_changed = Signal(str)

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

        container = QWidget()
        container.setLayout(layout)

        # ── Signal wiring (worker → main thread; disconnected in cleanup) ──
        self.chunk_streamed.connect(self._on_chunk_streamed)
        self.status_changed.connect(self._on_status_changed)

        # ── Mount ──
        self.set_content_widget(container)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # ── Enter-to-send ────────────────────────────────────────────────

    def eventFilter(self, obj: Any, event: QEvent) -> bool:
        if obj == self.text_prompt._text_edit and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if event.modifiers() & Qt.ShiftModifier:
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

    def _clear_context(self) -> None:
        # Release the agent's dedicated pool instance so its KV cache is
        # wiped and returned to the general idle queue.
        cached_val = self._get_cached_value("model_obj")
        if isinstance(cached_val, dict) and "pool" in cached_val:
            pool = cached_val["pool"]
            try:
                with pool._lock:
                    if self._session_id in pool._session_instances:
                        inst = pool._session_instances[self._session_id]
                        # Re-entrant lock: safe to call checkin while holding.
                        pool.checkin(
                            inst,
                            session_id=self._session_id,
                            release_session=True,
                        )
            except Exception as exc:
                log.debug(f"Clear context: pool release failed: {exc}")

        self._history.clear()
        self._widget_core.push_display("preview_display", "<i>Agent context cleared.</i>")
        self._widget_core.push_display("status", "Idle.")

    # ── Execution ───────────────────────────────────────────────────

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

    def _cleanup_after_worker(self) -> None:
        self.btn_run.set_label("Run Agent")
        super()._cleanup_after_worker()

    def cleanup(self) -> None:
        self.cancel_compute()
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

        toolset = inputs.get("toolset")
        role = inputs.get("role") or DEFAULT_ROLE

        binding: Optional[RoleBinding] = None
        event_hooks: list = []
        # Set once the per-run event emitter exists (only when a toolset is
        # wired — no toolset, no hook stream). Subclasses observe through it.
        emit_run_event: Optional[Any] = None
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

                def _emit_event(event: Any, **extra: Any) -> None:
                    """Put one typed event on the `events` port.

                    The envelope is applied here and nowhere else: run_id +
                    seq are the dedup key across graph re-evaluations, and
                    the identity pair says which node produced the line once
                    two streams are merged (spec D60.1).
                    """
                    self.emit_stream(
                        "events",
                        to_wire(event, run_id=run_id, seq=next(seq),
                                **identity, **extra),
                        throttle_ms=0,
                    )

                # Handed to subclasses (the Orchestrator node) so a worker's
                # events can be re-emitted on this node's own stream.
                emit_run_event = _emit_event

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
            )
            engine.clear_stop()

            loop = AgentLoop(
                engine,
                toolset,
                max_rounds=role.max_rounds or DEFAULT_MAX_ROUNDS,
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
                elif isinstance(event, EventReflection):
                    self.status_changed.emit(
                        f"Reflection retry {event.retry_count + 1}/{event.max_retries}…"
                    )
                elif isinstance(event, EventError):
                    run_error = event.error
                elif isinstance(event, EventRunResult):
                    final_text = event.text

            if run_error and not final_text:
                self.compute_error.emit(run_error)
                self.status_changed.emit(f"Failed: {run_error}")
                return {"response": f"Error: {run_error}"}

            self._last_run_ok = not self.is_compute_cancelled()
            self.status_changed.emit("Done." if self._last_run_ok else "Cancelled.")

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
            self._detach_run_observers(toolset)
            if toolset is not None:
                for event_name, callback in event_hooks:
                    toolset.hooks.unregister(event_name, callback)
            if binding is not None:
                binding.deactivate()

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
