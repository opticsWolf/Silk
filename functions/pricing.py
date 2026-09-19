# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

What a run costs, when the endpoint says (D88).

A price is a fact about someone else's service, so Silk does not keep a
table of them: a hardcoded price that has gone stale is worse than no
price, because it produces a number a person will believe. The price is
read from the endpoint at connect time -- OpenRouter advertises one per
model on ``/models`` -- and carried on the model handle beside the client
it prices.

``None`` is a first-class answer. A local server has no price to give,
and a gateway may not advertise one; the run is then unpriced, which is
different from free. `UsageLimits` refuses to start a run whose cost cap
cannot bind, rather than letting a ceiling sit there doing nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = ["ModelPrice", "cost_of", "price_from_handle", "price_from_spec"]


@dataclass(frozen=True)
class ModelPrice:
    """Per-token prices, in the currency the provider quotes.

    Stored per *token* rather than per million, because that is the shape
    the arithmetic wants and the shape OpenRouter publishes. Display code
    multiplies back up.
    """

    input_per_token: float = 0.0
    output_per_token: float = 0.0
    currency: str = "USD"
    #: Where the number came from, for a status line that has to be
    #: believable: "openrouter" is the provider's own quote, "manual" is
    #: what someone typed.
    source: str = ""

    def is_free(self) -> bool:
        """Whether the provider quoted zero -- which it does, for some."""
        return not (self.input_per_token or self.output_per_token)

    def per_million(self) -> tuple[float, float]:
        """The pair as prices are usually written: in/out per 1M tokens."""
        return (self.input_per_token * 1e6, self.output_per_token * 1e6)

    def describe(self) -> str:
        """``$0.08/$0.50 per 1M tokens`` -- the way a person compares them."""
        if self.is_free():
            return "free"
        pin, pout = self.per_million()
        return f"${pin:,.2f}/${pout:,.2f} per 1M tokens"


def cost_of(price: Optional[ModelPrice], input_tokens: int = 0,
            output_tokens: int = 0) -> Optional[float]:
    """What those tokens cost, or ``None`` when the price is unknown.

    ``None`` rather than 0.0, and the distinction is the point: zero is a
    claim that the run was free, which is true for a local model and
    false for a metered one that simply did not advertise.
    """
    if price is None:
        return None
    return (max(0, int(input_tokens)) * price.input_per_token
            + max(0, int(output_tokens)) * price.output_per_token)


def price_from_spec(spec: Any, source: str = "") -> Optional[ModelPrice]:
    """Read a price out of one ``/models`` entry, or ``None``.

    OpenRouter quotes ``pricing.prompt`` / ``pricing.completion`` as
    per-token decimal *strings*. Anything unreadable is ``None``: a
    malformed quote must not become a confident zero.
    """
    if not isinstance(spec, dict):
        return None
    pricing = spec.get("pricing")
    if not isinstance(pricing, dict):
        return None
    raw_in = pricing.get("prompt")
    raw_out = pricing.get("completion")
    if raw_in is None or raw_out is None:
        return None
    try:
        prompt = float(raw_in)
        completion = float(raw_out)
    except (TypeError, ValueError):
        return None
    if prompt < 0 or completion < 0:
        return None
    return ModelPrice(input_per_token=prompt, output_per_token=completion,
                      source=source or "endpoint")


def price_from_handle(handle: Any) -> Optional[ModelPrice]:
    """The price a model handle carries, or ``None``.

    The handle stores a plain dict so it stays serialisable alongside
    everything else on it; this is the one place that shape is read.
    """
    if not isinstance(handle, dict):
        return None
    raw: Dict[str, Any] = handle.get("pricing") or {}
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        return ModelPrice(
            input_per_token=float(raw.get("input", 0.0)),
            output_per_token=float(raw.get("output", 0.0)),
            currency=str(raw.get("currency") or "USD"),
            source=str(raw.get("source") or ""),
        )
    except (TypeError, ValueError):
        return None
