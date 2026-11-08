# -*- coding: utf-8 -*-
"""JSON preset persistence for silk nodes — Qt-free.

Presets are stored per kind as a single JSON document under the user's
home directory (``~/.weave/presets/<kind>.json``) and validated through
pydantic on load: individual invalid entries are skipped (logged), never
raised, so one corrupt preset can't take the whole store down.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, ClassVar, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

from weave.logger import get_logger

log = get_logger("SilkPresets")

PRESET_DIR = Path.home() / ".weave" / "presets"

FORMAT_VERSION = 1


class ToolSetPreset(BaseModel):
    """Persisted state of a Silk ToolSet node's selection."""

    name: str
    checked_tools: list[str] = Field(default_factory=list)
    allow_all: bool = False


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


class ClassifierPreset(BaseModel):
    """Persisted state of a Silk Classifier node (D-S8).

    The node's whole form is a :class:`~.classifier.ClassifierSpec`, kept as
    a plain dict here so this module stays importable without the classifier
    (validated when the node applies it). Secrets are never in it: the
    credential is stored by *name* (D22).
    """

    name: str
    spec: dict = Field(default_factory=dict)


class EnsemblePreset(BaseModel):
    """Persisted state of a Classifier Ensemble node.

    The node's whole form is an :class:`~.classifier.EnsembleSpec`, kept
    as a plain dict here so this module stays importable without the
    classifier (validated when the node applies it).
    """

    name: str
    spec: dict = Field(default_factory=dict)


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
            self._presets[preset.name] = preset  # type: ignore[attr-defined]

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
        # Every preset model carries a `name`; the TypeVar's bound
        # (BaseModel) is what cannot say so.
        self._presets[preset.name] = preset  # type: ignore[attr-defined]
        self._flush()

    def remove(self, name: str) -> bool:
        if name not in self._presets:
            return False
        del self._presets[name]
        self._flush()
        return True


class EmbedderModelPreset(BaseModel):
    """Remembered settings for one embedding model file (Embedder Loader).

    Written on every successful load, re-applied when the same file is
    picked again. ``name`` is the resolved absolute path (the store key).
    """

    name: str
    filename: str = ""
    file_size: int = 0
    device: str = "cpu"  # "cpu" | "gpu"
    n_ctx: int = 2048
    n_threads: int = 8
    n_batch: int = 512
    use_mmap: bool = True


class EmbedderModelStore(PresetStore[EmbedderModelPreset]):
    """Per-model settings memory for the Embedder Loader node.

    Qt-free: the node owns one on the GUI thread and writes from the
    worker after a successful load; a lock makes that safe. Lookup is by
    resolved path, falling back to filename + size so a moved file
    still matches.
    """

    _preset_model: ClassVar[type[EmbedderModelPreset]] = EmbedderModelPreset

    def __init__(self, directory: Optional[Path] = None) -> None:
        super().__init__("embedder_models", EmbedderModelPreset, directory)
        self._lock = threading.Lock()

    @staticmethod
    def key_for(path_str: str) -> str:
        try:
            return str(Path(path_str).expanduser().resolve())
        except OSError:
            return str(path_str)

    def lookup(self, path_str: str) -> Optional[EmbedderModelPreset]:
        """The remembered settings for *path_str*, or ``None``."""
        key = self.key_for(path_str)
        with self._lock:
            hit = self._presets.get(key)
            if hit is not None:
                return hit
            try:
                size = Path(path_str).stat().st_size
            except OSError:
                return None
            name = Path(path_str).name
            for preset in self._presets.values():
                if preset.filename == name and preset.file_size == size:
                    return preset
            return None

    def remember(self, path_str: str, **settings: Any) -> EmbedderModelPreset:
        """Store the current settings for *path_str* (rewrites the JSON)."""
        key = self.key_for(path_str)
        try:
            stat = Path(path_str).stat()
            size, filename = stat.st_size, Path(path_str).name
        except OSError:
            size, filename = 0, Path(path_str).name
        preset = self._preset_model(
            name=key, filename=filename, file_size=size, **settings
        )
        with self._lock:
            self.upsert(preset)
        return preset


class DecisionModelPreset(EmbedderModelPreset):
    """Remembered settings for one decision-model file (Classifier Loader).

    Same fields as the embedder's (device/ctx/threads/batch/mmap); a
    distinct model so validation and the on-disk document stay separate.
    """


class GGUFModelPreset(EmbedderModelPreset):
    """Remembered settings for one chat model file (GGUF Loader).

    Written on every successful load, re-applied when the same file is
    picked again. Same key rules as the embedder's (resolved path, then
    filename + size); only the on-disk document differs.
    """

    gpu_layers: int = -1
    n_ctx: int = 4096
    n_threads: int = -1
    seed: int = -1
    flash_attn: bool = False
    type_k: str = "f16"
    type_v: str = "f16"
    embedding: bool = False
    enable_mmproj: bool = False
    mmproj_path: str = ""
    enable_mtp: bool = False
    mtp_draft_tokens: int = 2


class GGUFModelStore(EmbedderModelStore):
    """Per-model settings memory for the GGUF Loader node.

    Same lookup rules as the embedder's; only the on-disk document
    differs (``gguf_models.json``) so loaders never trade settings.
    """

    _preset_model: ClassVar[type[EmbedderModelPreset]] = GGUFModelPreset

    def __init__(self, directory: Optional[Path] = None) -> None:
        PresetStore.__init__(self, "gguf_models", GGUFModelPreset,
                             directory)
        self._lock = threading.Lock()


class DecisionModelStore(EmbedderModelStore):
    """Per-model settings memory for the Classifier Loader node.

    Same lookup rules as the embedder's (resolved path, then filename +
    size); only the on-disk document differs (``decision_models.json``)
    so the two loaders never trade settings.
    """

    _preset_model: ClassVar[type[EmbedderModelPreset]] = DecisionModelPreset

    def __init__(self, directory: Optional[Path] = None) -> None:
        PresetStore.__init__(self, "decision_models", DecisionModelPreset,
                             directory)
        self._lock = threading.Lock()
