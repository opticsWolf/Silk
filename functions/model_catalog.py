# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Model catalogues -- a provider's model list, kept on disk.

The generic Model Endpoint asks ``/models`` on every connect and takes
the context length on trust or not at all. For a provider Silk knows by
name (OpenCode Zen, OpenCode Go) that is both too much and too little:

* **Too much**, because the list changes a few times a month, not per
  keystroke. A dropdown that needs a network round trip before it can
  show anything is a dropdown that is empty on a train. So the list is
  kept in a JSON file the node reads when it opens, and refreshed when
  someone presses the button.
* **Too little**, because OpenCode's ``/v1/models`` is the bare OpenAI
  shape (``id``, ``object``, ``owned_by``) -- it names models but says
  nothing about their context window. The window comes from
  `models.dev <https://models.dev>`_, the open catalogue OpenCode itself
  is built on, which carries ``limit.context`` for every model on both
  plans. The endpoint stays the authority on *which* models exist (it is
  what your key can actually reach); models.dev only annotates them.

The file also holds what a person chose: a context length held below the
model's maximum is stored per model, so it survives a refresh, a restart
and a different graph using the same model.

**Fallbacks, in order.** Endpoint and models.dev both answer: the
endpoint's list, annotated. Only models.dev answers (no key yet, or the
listing is down): its list, so the dropdown is still useful before a key
is set. Only the endpoint answers: its list, every context length unknown
-- and said to be unknown, never guessed (D25). Neither: the refresh
fails, and the file on disk is left exactly as it was.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from weave.logger import get_logger

from .model_endpoint import list_model_specs

__all__ = [
    "CATALOGS",
    "CATALOG_DIR",
    "Catalog",
    "CatalogModel",
    "ModelCatalog",
    "catalog_for",
    "fetch_models_dev",
]

log = get_logger("SilkModelCatalog")

#: Where catalogue files live. Beside ``secrets.json`` on purpose: both
#: are per-user facts about providers, neither belongs in a graph.
CATALOG_DIR = Path.home() / ".weave" / "silk" / "model_catalogs"

#: models.dev's whole catalogue, every provider, one JSON document.
MODELS_DEV_URL = "https://models.dev/api.json"

#: The file format. Bump when the shape changes; an older file is then
#: read for its overrides and its model list rebuilt on the next refresh.
CACHE_VERSION = 1

FETCH_TIMEOUT_S = 20.0

#: Which wire format an AI SDK package speaks. models.dev names the
#: package per provider, and per model where a gateway mixes them (Zen
#: routes Claude to ``/messages`` and GPT to ``/responses``). Silk speaks
#: only chat completions, so anything else is flagged rather than hidden:
#: OpenCode may still accept it there, and the person can try.
_NPM_API: Dict[str, str] = {
    "@ai-sdk/openai-compatible": "chat",
    "@ai-sdk/anthropic": "messages",
    "@ai-sdk/openai": "responses",
    "@ai-sdk/google": "gemini",
}


@dataclass(frozen=True)
class Catalog:
    """One provider Silk knows by name."""

    key: str            #: cache file stem, and the handle's ``provider``
    label: str
    base_url: str
    models_dev_id: str  #: the provider's key in models.dev's catalogue
    credential: str     #: the usual credential *name* (D22)
    metered: bool       #: requests are billed per token (a price matters)
    note: str = ""


CATALOGS: Dict[str, Catalog] = {
    c.key: c for c in (
        Catalog(
            "opencode", "OpenCode Zen", "https://opencode.ai/zen/v1",
            "opencode", "OPENCODE_API_KEY", metered=True,
            note="Pay as you go; the model name is worth reading twice.",
        ),
        Catalog(
            "opencode-go", "OpenCode Go", "https://opencode.ai/zen/go/v1",
            "opencode-go", "OPENCODE_API_KEY", metered=False,
            note="Subscription: usage limits, not a per-token bill.",
        ),
    )
}


def catalog_for(key: str) -> Catalog:
    """The catalogue by key. Unknown keys are a programming error."""
    return CATALOGS[key]


@dataclass(frozen=True)
class CatalogModel:
    """What the node needs to know about one model."""

    id: str
    name: str = ""
    context_max: int = 0        #: 0 = unknown
    output_max: int = 0
    api: str = "chat"           #: chat | messages | responses | gemini
    tool_call: Optional[bool] = None
    cost_input: Optional[float] = None    #: USD per million tokens
    cost_output: Optional[float] = None

    @property
    def speaks_chat(self) -> bool:
        return self.api == "chat"

    def to_json(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"name": self.name, "context_max": self.context_max,
                               "output_max": self.output_max, "api": self.api}
        if self.tool_call is not None:
            row["tool_call"] = self.tool_call
        if self.cost_input is not None and self.cost_output is not None:
            row["cost"] = {"input": self.cost_input, "output": self.cost_output}
        return row

    @classmethod
    def from_json(cls, model_id: str, row: Any) -> "CatalogModel":
        row = row if isinstance(row, dict) else {}
        cost = row.get("cost") if isinstance(row.get("cost"), dict) else {}
        tool = row.get("tool_call")
        return cls(
            id=model_id,
            name=str(row.get("name") or ""),
            context_max=_int(row.get("context_max")),
            output_max=_int(row.get("output_max")),
            api=str(row.get("api") or "chat"),
            tool_call=tool if isinstance(tool, bool) else None,
            cost_input=_float(cost.get("input")),
            cost_output=_float(cost.get("output")),
        )


def _int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


# ── fetching ─────────────────────────────────────────────────────────────

def fetch_models_dev(timeout: float = FETCH_TIMEOUT_S) -> Dict[str, Any]:
    """models.dev's catalogue. Raises ``RuntimeError`` on any failure."""
    request = urllib.request.Request(
        MODELS_DEV_URL, headers={"User-Agent": "silk-model-catalog"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"models.dev answered {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"could not reach models.dev: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"models.dev did not answer with JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("models.dev answered something that is not a catalogue")
    return payload


def models_from_models_dev(payload: Dict[str, Any],
                           provider_id: str) -> Dict[str, CatalogModel]:
    """One provider's models out of the models.dev catalogue."""
    provider = payload.get(provider_id)
    if not isinstance(provider, dict):
        return {}
    default_api = _NPM_API.get(str(provider.get("npm") or ""), "chat")
    rows = provider.get("models")
    if not isinstance(rows, dict):
        return {}
    found: Dict[str, CatalogModel] = {}
    for model_id, row in rows.items():
        if not isinstance(row, dict) or not isinstance(model_id, str):
            continue
        limit = row.get("limit") if isinstance(row.get("limit"), dict) else {}
        cost = row.get("cost") if isinstance(row.get("cost"), dict) else {}
        override = row.get("provider") if isinstance(row.get("provider"), dict) else {}
        api = _NPM_API.get(str(override.get("npm") or ""), default_api)
        tool = row.get("tool_call")
        found[model_id] = CatalogModel(
            id=model_id,
            name=str(row.get("name") or ""),
            context_max=_int(limit.get("context")),
            output_max=_int(limit.get("output")),
            api=api,
            tool_call=tool if isinstance(tool, bool) else None,
            cost_input=_float(cost.get("input")),
            cost_output=_float(cost.get("output")),
        )
    return found


def merge_sources(
    endpoint: Optional[List[str]],
    annotated: Optional[Dict[str, CatalogModel]],
) -> Tuple[Dict[str, CatalogModel], str]:
    """The model list both sources agree on, and a word on where it came from.

    ``None`` means that source failed; an empty list is an answer.
    """
    if endpoint is not None and endpoint:
        meta = annotated or {}
        models = {name: meta.get(name, CatalogModel(id=name)) for name in endpoint}
        source = "endpoint + models.dev" if annotated is not None else (
            "endpoint only (models.dev unreachable: context lengths unknown)")
        return models, source
    if annotated:
        return dict(annotated), (
            "models.dev only (the endpoint did not list models)")
    return {}, "nothing"


# ── the cache ────────────────────────────────────────────────────────────

class ModelCatalog:
    """One provider's catalogue file: the model list and the overrides.

    Reads are cheap and local; the one network operation is `refresh`,
    which the node runs off the GUI thread. Every write goes through a
    temp file and a rename, so a crash mid-write leaves the old file, and
    a lock, so the refresh thread and a slider release on the GUI thread
    do not interleave.

    File shape::

        {
          "version": 1,
          "provider": "opencode-go",
          "base_url": "https://opencode.ai/zen/go/v1",
          "fetched_at": 1730570000.0,
          "source": "endpoint + models.dev",
          "models": {"kimi-k2.5": {"name": "...", "context_max": 262144,
                                    "output_max": 65536, "api": "chat",
                                    "tool_call": true}},
          "context_overrides": {"kimi-k2.5": 131072},
          "selected": "kimi-k2.5"
        }
    """

    _locks: Dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, catalog: Catalog, directory: Optional[Path] = None) -> None:
        self.catalog = catalog
        self.path = Path(directory or CATALOG_DIR) / f"{catalog.key}.json"
        with self._locks_guard:
            self._lock = self._locks.setdefault(str(self.path), threading.Lock())
        self.models: Dict[str, CatalogModel] = {}
        self.overrides: Dict[str, int] = {}
        self.selected: str = ""
        self.fetched_at: float = 0.0
        self.source: str = ""
        self.load()

    # ── reading ──

    def load(self) -> bool:
        """Read the file, if there is one. ``False`` when there is not."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        except (OSError, ValueError) as exc:
            log.warning(f"Could not read {self.path}: {exc}")
            return False
        if not isinstance(data, dict):
            return False
        raw_overrides = data.get("context_overrides")
        if isinstance(raw_overrides, dict):
            self.overrides = {str(k): _int(v) for k, v in raw_overrides.items()
                              if _int(v) > 0}
        self.selected = str(data.get("selected") or "")
        if data.get("version") != CACHE_VERSION:
            # Overrides are the person's; the list is ours to rebuild.
            return False
        rows = data.get("models")
        if isinstance(rows, dict):
            self.models = {str(k): CatalogModel.from_json(str(k), v)
                           for k, v in rows.items()}
        self.fetched_at = float(data.get("fetched_at") or 0.0)
        self.source = str(data.get("source") or "")
        return bool(self.models)

    def model_ids(self) -> List[str]:
        return sorted(self.models)

    def model(self, model_id: str) -> Optional[CatalogModel]:
        return self.models.get(model_id)

    def context_max(self, model_id: str) -> int:
        model = self.models.get(model_id)
        return model.context_max if model else 0

    def context_for(self, model_id: str) -> int:
        """The context length to use: the stored choice, else the maximum.

        A stored choice above a (since-shrunk) maximum is clamped -- the
        catalogue is the better witness of what the model accepts.
        """
        maximum = self.context_max(model_id)
        chosen = self.overrides.get(model_id, 0)
        if chosen and maximum:
            return min(chosen, maximum)
        return chosen or maximum

    def describe_age(self, now: Optional[float] = None) -> str:
        if not self.fetched_at:
            return "never refreshed"
        minutes = int(((now or time.time()) - self.fetched_at) // 60)
        if minutes < 1:
            return "refreshed just now"
        if minutes < 120:
            return f"refreshed {minutes} min ago"
        hours = minutes // 60
        if hours < 48:
            return f"refreshed {hours} h ago"
        return f"refreshed {hours // 24} days ago"

    # ── writing ──

    def set_context(self, model_id: str, value: int) -> None:
        """Store a chosen context length. At (or above) the max, forget it.

        "The maximum" is then a live default rather than a frozen number,
        so a model whose window grows is used at its new size.
        """
        if not model_id:
            return
        maximum = self.context_max(model_id)
        value = _int(value)
        if value <= 0 or (maximum and value >= maximum):
            self.overrides.pop(model_id, None)
        else:
            self.overrides[model_id] = value
        self.save()

    def set_selected(self, model_id: str) -> None:
        if model_id and model_id != self.selected:
            self.selected = model_id
            self.save()

    def save(self) -> None:
        payload = {
            "version": CACHE_VERSION,
            "provider": self.catalog.key,
            "base_url": self.catalog.base_url,
            "fetched_at": self.fetched_at,
            "source": self.source,
            "models": {k: self.models[k].to_json() for k in sorted(self.models)},
            "context_overrides": dict(sorted(self.overrides.items())),
            "selected": self.selected,
        }
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=self.path.parent,
                                           prefix=f".{self.path.name}.")
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                os.replace(tmp, self.path)
            except OSError as exc:
                log.warning(f"Could not write {self.path}: {exc}")

    # ── the network ──

    def refresh(self, headers: Optional[Dict[str, str]] = None,
                timeout: float = FETCH_TIMEOUT_S) -> str:
        """Ask both sources, rewrite the file, and say what happened.

        Raises ``RuntimeError`` only when *neither* source answered; the
        file is then untouched, so a refresh on a train costs nothing.
        """
        endpoint: Optional[List[str]] = None
        endpoint_error = ""
        try:
            endpoint = sorted(list_model_specs(
                self.catalog.base_url, headers=headers, timeout=timeout))
        except RuntimeError as exc:
            endpoint_error = str(exc)
        annotated: Optional[Dict[str, CatalogModel]] = None
        dev_error = ""
        try:
            annotated = models_from_models_dev(
                fetch_models_dev(timeout), self.catalog.models_dev_id)
        except RuntimeError as exc:
            dev_error = str(exc)

        models, source = merge_sources(endpoint, annotated)
        if not models:
            reasons = "; ".join(r for r in (endpoint_error, dev_error) if r)
            raise RuntimeError(
                f"No model list for {self.catalog.label}"
                + (f": {reasons}" if reasons else " (both sources listed none)")
            )
        self.models = models
        self.source = source
        self.fetched_at = time.time()
        self.save()
        unknown = sum(1 for m in models.values() if not m.context_max)
        detail = f"{len(models)} model(s) from {source}"
        if unknown and annotated is not None:
            detail += f", {unknown} without a known context length"
        if endpoint_error and endpoint is None:
            log.info(f"{self.catalog.label} listing failed: {endpoint_error}")
        return detail
