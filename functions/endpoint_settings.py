# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

Per-provider endpoint memory -- what a person chose, kept on disk.

The Model Endpoint node (and the Endpoint/Model split) remember one
thing per provider: the base URL as last edited, the credential *name*,
which model was picked last, which context lengths were held below a
model's maximum, and the model list the provider's own ``/models``
answered with last time. A person who switches from OpenRouter to a
local server and back finds everything where they left it.

Only names and URLs live here -- never key values (D22). The file sits
beside ``secrets.json`` for the same reason catalogues do: per-user
facts about providers, none of which belong in a graph.

The shape mirrors ``model_catalog.ModelCatalog`` on purpose (same
atomic writes, same override semantics) minus models.dev: a generic
OpenAI-compatible endpoint is the only source, and its rows carry a
``context_length`` when the gateway bothers to say one. ``tool_call``
comes from ``supported_parameters`` on the same row.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from weave.logger import get_logger

from .model_endpoint import list_model_specs

__all__ = [
    "ENDPOINT_DIR",
    "SETTINGS_VERSION",
    "EndpointModel",
    "EndpointSettings",
]

log = get_logger("SilkEndpointSettings")

#: One file per provider, next to the catalogues and the secrets.
ENDPOINT_DIR = Path.home() / ".weave" / "silk" / "model_endpoints"

#: The file format. Bump when the shape changes; an older file is still
#: read for its overrides and choices, which are the person's.
SETTINGS_VERSION = 1

REFRESH_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class EndpointModel:
    """What one ``/models`` row said about one model."""

    id: str
    context_max: int = 0      #: 0 = the gateway did not say
    tool_call: Optional[bool] = None

    def to_json(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"context_max": self.context_max}
        if self.tool_call is not None:
            row["tool_call"] = self.tool_call
        return row

    @classmethod
    def from_json(cls, model_id: str, row: Any) -> "EndpointModel":
        row = row if isinstance(row, dict) else {}
        try:
            context_max = int(row.get("context_max") or 0)
        except (TypeError, ValueError):
            context_max = 0
        tool = row.get("tool_call")
        return cls(
            id=model_id,
            context_max=context_max if context_max > 0 else 0,
            tool_call=tool if isinstance(tool, bool) else None,
        )


def _tool_call_from_spec(spec: Any) -> Optional[bool]:
    """What the row says about the ``tools`` parameter.

    A listed ``supported_parameters`` without ``"tools"`` is a no; a row
    that says nothing about parameters says nothing (None), which leaves
    the checkbox -- and ``connect``'s own fallback -- alone.
    """
    if not isinstance(spec, dict):
        return None
    params = spec.get("supported_parameters")
    if not isinstance(params, (list, tuple)):
        return None
    return "tools" in params


class EndpointSettings:
    """One provider's remembered endpoint.

    Reads are cheap and local; the one network operation is `refresh`,
    which nodes run off the GUI thread. Every write goes through a temp
    file and a rename, so a crash mid-write leaves the old file, and a
    lock, so a refresh thread and a GUI-thread edit do not interleave.
    """

    _locks: Dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, provider_key: str,
                 directory: Optional[Path] = None) -> None:
        self.provider = str(provider_key or "").strip().lower() or "custom"
        self.path = Path(directory or ENDPOINT_DIR) / f"{self.provider}.json"
        with self._locks_guard:
            self._lock = self._locks.setdefault(str(self.path), threading.Lock())
        self.base_url: str = ""
        self.credential: str = ""
        self.models: Dict[str, EndpointModel] = {}
        self.overrides: Dict[str, int] = {}
        self.max_overrides: Dict[str, int] = {}
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
        self.base_url = str(data.get("base_url") or "").strip()
        self.credential = str(data.get("credential") or "").strip()
        raw_overrides = data.get("context_overrides")
        if isinstance(raw_overrides, dict):
            self.overrides = {str(k): _positive_int(v)
                              for k, v in raw_overrides.items()
                              if _positive_int(v) > 0}
        raw_max = data.get("max_overrides")
        if isinstance(raw_max, dict):
            self.max_overrides = {str(k): _positive_int(v)
                                  for k, v in raw_max.items()
                                  if _positive_int(v) > 0}
        self.selected = str(data.get("selected") or "")
        if data.get("version") != SETTINGS_VERSION:
            # The person's choices survive a format change; the listing
            # is ours to rebuild on the next refresh.
            return False
        rows = data.get("models")
        if isinstance(rows, dict):
            self.models = {str(k): EndpointModel.from_json(str(k), v)
                           for k, v in rows.items()}
        self.fetched_at = float(data.get("fetched_at") or 0.0)
        self.source = str(data.get("source") or "")
        return bool(self.models)

    def model_ids(self) -> List[str]:
        return sorted(self.models)

    def model(self, model_id: str) -> Optional[EndpointModel]:
        return self.models.get(model_id)

    def context_max(self, model_id: str) -> int:
        entry = self.models.get(model_id)
        return entry.context_max if entry else 0

    def tool_call(self, model_id: str) -> Optional[bool]:
        entry = self.models.get(model_id)
        return entry.tool_call if entry else None

    def max_for(self, model_id: str) -> int:
        """The ceiling for the model: the stored max, else the detected one.

        A stored max is the person's explicit claim and wins over a
        detected window; an equal one is never stored (see `set_max`),
        so the live detected value stays the default.
        """
        return self.max_overrides.get(model_id, 0) or self.context_max(model_id)

    def context_for(self, model_id: str) -> int:
        """The context length to use: the stored choice, else the maximum.

        A stored choice above a (since-shrunk) maximum is clamped -- the
        listing is the better witness of what the model accepts.
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

    def set_base_url(self, base_url: str) -> None:
        url = str(base_url or "").strip()
        if url != self.base_url:
            self.base_url = url
            self.save()

    def set_credential(self, credential: str) -> None:
        name = str(credential or "").strip()
        if name != self.credential:
            self.credential = name
            self.save()

    def set_selected(self, model_id: str) -> None:
        if model_id and model_id != self.selected:
            self.selected = model_id
            self.save()

    def set_max(self, model_id: str, value: int) -> None:
        """Store a manual ceiling. At the detected window, forget it.

        Mirrors `set_context`: the detected window is the live default,
        and only a deviation from it is worth keeping.
        """
        if not model_id:
            return
        value = _positive_int(value)
        detected = self.context_max(model_id)
        if value <= 0 or (detected and value == detected):
            self.max_overrides.pop(model_id, None)
        else:
            self.max_overrides[model_id] = value
        self.save()

    def set_context(self, model_id: str, value: int) -> None:
        """Store a chosen context length. At (or above) the max, forget it.

        "The maximum" is then a live default rather than a frozen number,
        so a model whose window grows is used at its new size.
        """
        if not model_id:
            return
        maximum = self.context_max(model_id)
        value = _positive_int(value)
        if value <= 0 or (maximum and value >= maximum):
            self.overrides.pop(model_id, None)
        else:
            self.overrides[model_id] = value
        self.save()

    def save(self) -> None:
        payload = {
            "version": SETTINGS_VERSION,
            "provider": self.provider,
            "base_url": self.base_url,
            "credential": self.credential,
            "fetched_at": self.fetched_at,
            "source": self.source,
            "models": {k: self.models[k].to_json() for k in sorted(self.models)},
            "max_overrides": dict(sorted(self.max_overrides.items())),
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

    def refresh(self, base_url: str,
                headers: Optional[Dict[str, str]] = None,
                timeout: float = REFRESH_TIMEOUT_S) -> str:
        """Ask ``/models``, rewrite the file, and say what happened.

        Raises ``RuntimeError`` when the listing fails; the file is then
        untouched, so a refresh on a train costs nothing. Choices and
        overrides are never cleared by a refresh -- a model that
        disappears from the listing keeps its stored choice in case it
        comes back under the same name.
        """
        specs = list_model_specs(base_url, headers=headers, timeout=timeout)
        self.models = {
            name: EndpointModel(
                id=name,
                context_max=_positive_int(
                    row.get("context_length")) if isinstance(row, dict) else 0,
                tool_call=_tool_call_from_spec(row),
            )
            for name, row in specs.items()
        }
        if not self.models:
            raise RuntimeError(
                f"No model list for {self.provider}: the endpoint listed "
                f"none -- type the model name the provider documents.")
        self.source = base_url
        self.fetched_at = time.time()
        self.save()
        unknown = sum(1 for m in self.models.values() if not m.context_max)
        detail = f"{len(self.models)} model(s) from {base_url}"
        if unknown:
            detail += f", {unknown} without an advertised context length"
        return detail


def _positive_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0
