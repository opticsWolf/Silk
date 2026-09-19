# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Silk Model Fallback -- a second model, behind the first (D89).

Takes two model handles and emits one, carrying the order to try them in.
Wires anywhere a loader or an endpoint wires, because what it emits *is*
a model handle: the Agent downstream sees one model and runs the way it
always has, right up to the moment the primary stops answering.

Why a node rather than a field on the endpoint: fallback is a
relationship between two models, and a relationship belongs on the
canvas. You can see which model backs which, a local GGUF behind a paid
gateway is one wire, and chaining two of these nodes gives three deep
without a list widget nobody wants to edit.

What it costs is stated plainly in the status line, because the trade is
real: **a chain is as capable as its weakest link.** Native tool calling
survives only if both models have it, the context window is the smaller
of the two, and a ``cost=`` budget can only bind if both quote a price.
The transport and the denominator are fixed before the first request and
the conversation is written in them -- they cannot be renegotiated
half-way through a run just because a different model picked it up.
"""

from typing import Any, ClassVar, Dict, List, Optional

from PySide6.QtWidgets import QFormLayout, QLabel

from weave.widgetcore import WidgetCore, PortRole
from weave.node.base import ActiveNode
from weave.node import VerticalSizePolicy
from weave.registry import register_node
from weave.logger import get_logger

from .silk_ports import MODEL_HANDLE_TYPE  # noqa: F401
from ..functions.model_fallback import (
    build_chain, chain_can_price, chain_context_length, chain_of,
    chain_supports_tools, describe_chain,
)

log = get_logger("SilkModelFallbackNode")


@register_node
class SilkModelFallbackNode(ActiveNode):
    """Two model handles in, one chained handle out."""
    # Weave declares `_widget_core` as `WidgetCoreLike` -- the subset the
    # *dataflow engine* relies on. A node uses the widget-facing whole
    # (`register_widget`, `push_display`, `apply_port_value`), which is
    # the concrete `WidgetCore` the base class assigns. The narrowing is a
    # declaration for the typechecker, not a runtime change (G9).
    _widget_core: WidgetCore

    node_class: ClassVar[str] = "Silk AI"
    node_subclass: ClassVar[str] = "Loaders"
    node_name: ClassVar[Optional[str]] = "Model Fallback"
    node_description: ClassVar[Optional[str]] = (
        "Puts one model behind another: when the primary fails for good, "
        "the run continues on the fallback instead of ending."
    )
    node_tags: ClassVar[Optional[List[str]]] = [
        "silk", "llm", "fallback", "failover", "resilience", "chain",
        "model", "inference",
    ]
    node_icon: ClassVar[Optional[str]] = "node"
    vertical_size_policy: ClassVar[VerticalSizePolicy] = VerticalSizePolicy.FIT
    node_state_api = 1
    node_version = 1     # bump on any state-shape change (G20)

    def __init__(self, title: str = "Model Fallback", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)

        # -- Ports --
        # Named for the roles rather than numbered: "which one is tried
        # first" is the only thing this node decides, and `model_1` would
        # hide it behind a convention the canvas cannot show.
        self.add_input("primary", datatype="model_handle")
        self.add_input("fallback", datatype="model_handle")
        self.add_output("model_obj", datatype="model_handle")

        # -- Layout & WidgetCore --
        form = QFormLayout()
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(4)
        self._widget_core = WidgetCore(layout=form)
        self._widget_core.set_node(self)

        self._label_chain = QLabel("Nothing wired.")
        self._label_chain.setWordWrap(True)
        form.addRow("Chain:", self._label_chain)
        self._widget_core.register_widget(
            "chain", self._label_chain, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        self._label_status = QLabel("")
        self._label_status.setWordWrap(True)
        self._label_status.setToolTip(
            "What the chain can do is the intersection of what its "
            "members can do -- native tool calling only if both have it, "
            "the smaller context window, and a price only if both quote "
            "one. The run is written in those before the first request."
        )
        form.addRow("Info:", self._label_status)
        self._widget_core.register_widget(
            "status", self._label_status, role=PortRole.DISPLAY,
            datatype="str", add_to_layout=False,
        )

        #: Display text staged on the worker thread, pushed on the main
        #: one -- a node's panel mirror is a different widget object, so a
        #: direct setText leaves it showing the previous chain (WV401).
        self._sync_chain: str = "Nothing wired."
        self._sync_status: str = ""

        self.set_content_widget(self._widget_core)
        if hasattr(self._widget_core, "patch_proxy"):
            self._widget_core.patch_proxy()

    # -- Main thread --------------------------------------------------

    def on_evaluate_finished(self) -> None:
        super().on_evaluate_finished()
        self._widget_core.push_display("chain", self._sync_chain)
        self._widget_core.push_display("status", self._sync_status)

    # -- Worker thread ------------------------------------------------

    @staticmethod
    def describe_capabilities(chain: List[Dict[str, Any]]) -> str:
        """What the chain can do, as the sentence the status line shows.

        Written as the *intersection*, and phrased so a lost capability
        reads as a consequence rather than a fault: someone who wired a
        fence-only local model behind a native-tools gateway has not made
        a mistake, they have made a trade, and the line should let them
        see it and decide.
        """
        if len(chain) < 2:
            return ""
        parts = [
            "native tool calling"
            if chain_supports_tools(chain)
            else "tool fences (not every model here takes a tools field)"
        ]
        window = chain_context_length(chain)
        parts.append(
            f"context {window:,} (the smallest in the chain)"
            if window else "context unknown"
        )
        parts.append(
            "priced" if chain_can_price(chain)
            else "unpriced -- a cost budget cannot bind on this chain"
        )
        return " - ".join(parts)

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        handle = build_chain(inputs.get("primary"), inputs.get("fallback"))
        if handle is None:
            self._sync_chain = "Nothing wired."
            self._sync_status = (
                "Wire a model into `primary`, and the one to fall back to "
                "into `fallback`."
            )
            return {"model_obj": None}

        chain = chain_of(handle)
        self._sync_chain = describe_chain(chain)
        if len(chain) < 2:
            # One usable model, which is a working graph and not an
            # error: the fallback is unwired, or upstream failed to
            # connect and said so on its own status line. Passing the
            # survivor through beats failing a run that has a model.
            self._sync_status = (
                "Only one model reached this node -- it is passed through "
                "unchanged, with nothing behind it."
            )
        else:
            self._sync_status = self.describe_capabilities(chain)
        log.info(f"Model chain: {self._sync_chain}")
        return {"model_obj": handle}
