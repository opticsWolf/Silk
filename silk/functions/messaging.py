# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0

AgentMessage — the typed envelope for agent-to-agent communication.

A bare ``response -> user_prompt`` string edge (or a bare tool result) carries no
provenance: the receiver cannot tell who sent it, why, or which request it
answers. :class:`AgentMessage` is the small, Qt-free envelope that makes a
hand-off self-describing — sender/recipient, a ``kind`` (task | result | error |
status | handoff), a correlation id tying a delegate → result pair, and an
``artifacts`` bag for structured payloads alongside the human-readable ``content``.

It is deliberately plain data (``to_dict`` / ``from_dict``) so it rides graph
edges as a ``dict`` (the ``agent_message`` port datatype), embeds in a tool
result, or gets logged — without pulling in PySide6. The delegation tools
(``orchestrator.py``) use it to stamp each sub-task request and its reply with a
shared ``correlation_id``; the direct message-passing path (Agent-node
inbox/outbox) uses the same shape.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

#: The recognised message kinds. ``task`` requests work; ``result`` answers it;
#: ``error`` is a failed answer; ``status`` is progress; ``handoff`` transfers
#: ownership of a conversation to another agent.
MESSAGE_KINDS = ("task", "result", "error", "status", "handoff")

#: Recipient wildcard — a broadcast to any agent listening.
BROADCAST = "*"


@dataclass
class AgentMessage:
    """A single self-describing message between two agents."""

    content: str
    sender: str = ""
    recipient: str = BROADCAST
    kind: str = "task"
    correlation_id: str = ""
    parent_id: Optional[str] = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        # A message always belongs to a correlation thread; default it to its
        # own id so a first message opens a new thread that replies can join.
        if not self.correlation_id:
            self.correlation_id = self.id

    def reply(
        self,
        content: str,
        *,
        sender: str = "",
        kind: str = "result",
        artifacts: Optional[dict[str, Any]] = None,
    ) -> "AgentMessage":
        """Build a response in the same correlation thread, threaded off this one.

        The reply's recipient is this message's sender, and it inherits the
        ``correlation_id`` so a delegate call and its answer stay paired.
        """
        return AgentMessage(
            content=content,
            sender=sender or self.recipient,
            recipient=self.sender or BROADCAST,
            kind=kind,
            correlation_id=self.correlation_id,
            parent_id=self.id,
            artifacts=dict(artifacts or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "parent_id": self.parent_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "kind": self.kind,
            "content": self.content,
            "artifacts": dict(self.artifacts),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentMessage":
        """Rebuild from a ``to_dict`` payload (or a partial hand-authored dict).

        Only ``content`` is truly required; everything else falls back to a
        sensible default, so a loosely-formed edge payload still parses.
        """
        msg = cls(
            content=str(d.get("content", "")),
            sender=str(d.get("sender", "")),
            recipient=str(d.get("recipient", BROADCAST) or BROADCAST),
            kind=str(d.get("kind", "task") or "task"),
            correlation_id=str(d.get("correlation_id", "") or ""),
            parent_id=d.get("parent_id"),
            artifacts=dict(d.get("artifacts") or {}),
        )
        # Preserve identity/time when they were supplied (round-trip fidelity).
        if d.get("id"):
            msg.id = str(d["id"])
        if not d.get("correlation_id"):
            msg.correlation_id = msg.id
        if d.get("ts") is not None:
            msg.ts = float(d["ts"])
        return msg

    def context_header(self) -> str:
        """A one-line preface injecting provenance into a receiving prompt.

        Lets a plain text-in agent still learn who is asking and in what mode,
        e.g. ``[Message from orchestrator - task]``.
        """
        who = self.sender or "unknown"
        return f"[Message from {who} - {self.kind}]"
