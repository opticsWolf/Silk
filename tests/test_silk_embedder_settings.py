# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Tests for the embedder loader's per-model settings memory."""

import json

from silk.functions.presets import EmbedderModelStore


def test_remember_lookup_roundtrip(tmp_path):
    store = EmbedderModelStore(directory=tmp_path)
    model = tmp_path / "qwen3-embed.gguf"
    model.write_bytes(b"\0" * 1024)

    assert store.lookup(str(model)) is None

    store.remember(str(model), device="cpu", n_ctx=2048, n_threads=8,
                   n_batch=512, use_mmap=True)

    hit = store.lookup(str(model))
    assert hit is not None
    assert hit.device == "cpu"
    assert hit.n_ctx == 2048
    assert hit.file_size == 1024
    assert hit.filename == "qwen3-embed.gguf"

    # Persisted: a fresh store over the same directory finds it.
    assert EmbedderModelStore(directory=tmp_path).lookup(str(model)) is not None


def test_lookup_follows_moved_file(tmp_path):
    store = EmbedderModelStore(directory=tmp_path)
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    model = old_dir / "embed.gguf"
    model.write_bytes(b"\0" * 512)
    store.remember(str(model), device="gpu")

    # Same file contents under a new path: filename + size still match.
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    moved = new_dir / "embed.gguf"
    moved.write_bytes(b"\0" * 512)

    hit = store.lookup(str(moved))
    assert hit is not None
    assert hit.device == "gpu"


def test_different_size_does_not_match(tmp_path):
    store = EmbedderModelStore(directory=tmp_path)
    model = tmp_path / "embed.gguf"
    model.write_bytes(b"\0" * 512)
    store.remember(str(model))

    other = tmp_path / "other"
    other.mkdir()
    bigger = other / "embed.gguf"
    bigger.write_bytes(b"\0" * 1024)

    assert store.lookup(str(bigger)) is None


def test_corrupt_entry_skipped(tmp_path):
    document = {"version": 1, "presets": [
        {"filename": "no-name.gguf"},  # missing required `name`
        {"name": "x", "n_ctx": "not-an-int"},  # wrong type
    ]}
    (tmp_path / "embedder_models.json").write_text(json.dumps(document))

    store = EmbedderModelStore(directory=tmp_path)
    assert store.names() == []
