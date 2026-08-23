# -*- coding: utf-8 -*-
"""JSON preset persistence for silk nodes — Qt-free.

Presets are stored per kind as a single JSON document under the user's
home directory (``~/.weave/presets/<kind>.json``) and validated through
pydantic on load: individual invalid entries are skipped (logged), never
raised, so one corrupt preset can't take the whole store down.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Generic, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

from weave.logger import get_logger

log = get_logger("SilkPresets")

PRESET_DIR = Path.home() / ".weave" / "presets"

FORMAT_VERSION = 1


class ToolSetPreset(BaseModel):
    """Persisted state of a Silk ToolSet node's selection."""

    name: str
    checked_tools: list[str] = Field(default_factory=list)


class RolePreset(BaseModel):
    """Persisted state of a Silk Role node."""

    name: str
    role_id: str = "role"
    instructions: str = ""
    allow_all: bool = False
    checked_tools: list[str] = Field(default_factory=list)
    max_risk: str = ""  # "" = no ceiling
    max_rounds: int = 16
    # Hook *names* from the catalog — behavior is never serialized.
    hooks: list[str] = Field(default_factory=list)
    # Per-hook config values, validated against the catalog entry's
    # config_model when the hooks are built.
    hook_configs: dict[str, dict] = Field(default_factory=dict)


class InferenceSettingsPreset(BaseModel):
    """Persisted state of an Inference Settings node."""

    name: str
    temperature: float = 0.7
    use_max_tokens: bool = False
    max_tokens: int = 1024
    stop_strings: str = ""
    top_k: int = 40
    use_top_p: bool = True
    top_p: float = 0.95
    use_min_p: bool = False
    min_p: float = 0.0
    use_repeat_penalty: bool = False
    repeat_penalty: float = 1.0
    use_presence_penalty: bool = False
    presence_penalty: float = 0.0
    use_frequency_penalty: bool = False
    frequency_penalty: float = 0.0


PresetT = TypeVar("PresetT", bound=BaseModel)


class PresetStore(Generic[PresetT]):
    """Load/save named presets of one pydantic model type.

    The on-disk document is ``{"version": 1, "presets": [ {...}, ... ]}``.
    All mutating operations rewrite the file immediately.
    """

    def __init__(
        self,
        kind: str,
        model: Type[PresetT],
        directory: Optional[Path] = None,
    ) -> None:
        self.kind = kind
        self.model = model
        self.path = (directory or PRESET_DIR) / f"{kind}.json"
        self._presets: dict[str, PresetT] = {}
        self.reload()

    # -- persistence -----------------------------------------------------

    def reload(self) -> None:
        """Re-read the store from disk, validating every entry."""
        self._presets.clear()
        if not self.path.is_file():
            return
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(f"Preset store '{self.path}' unreadable: {exc}")
            return
        for raw in document.get("presets", []):
            try:
                preset = self.model.model_validate(raw)
            except ValidationError as exc:
                log.warning(
                    f"Skipping invalid {self.kind} preset "
                    f"{raw.get('name', '?')!r}: {exc}"
                )
                continue
            self._presets[preset.name] = preset

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "version": FORMAT_VERSION,
            "presets": [
                self._presets[name].model_dump() for name in sorted(self._presets)
            ],
        }
        self.path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- access ----------------------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._presets)

    def get(self, name: str) -> Optional[PresetT]:
        return self._presets.get(name)

    def upsert(self, preset: PresetT) -> None:
        self._presets[preset.name] = preset
        self._flush()

    def remove(self, name: str) -> bool:
        if name not in self._presets:
            return False
        del self._presets[name]
        self._flush()
        return True
