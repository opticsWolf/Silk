## Presets

`functions/presets.py` — Qt-free JSON persistence for node configuration. A
`PresetStore` per kind writes `~/.weave/presets/<kind>.json`
(`FORMAT_VERSION = 1`); presets are pydantic models (`ToolSetPreset`,
`RolePreset`, `InferenceSettingsPreset`). Loading is defensive: a missing or
unparseable file is logged and treated as empty, and individual entries that
fail validation are skipped (logged), never raised — a stale or hand-edited
file can't break the graph. The `preset_bar.py` widget uses it to
save/restore selections on the `Silk ToolBox`, `Silk Role`, and
`Inference Settings` nodes.

