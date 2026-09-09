# -*- coding: utf-8 -*-
"""Tests for the lightweight GGUF metadata probe (silk GGUF loader)."""

import struct

import pytest

from silk.functions.gguf_meta import (
    GGUF_MAGIC,
    pack_header,
    pack_kv_int,
    read_gguf_meta,
)


def _pack_kv_str(key: str, value: str) -> bytes:
    kb = key.encode("utf-8")
    vb = value.encode("utf-8")
    return (struct.pack("<Q", len(kb)) + kb
            + struct.pack("<IQ", 8, len(vb)) + vb)


def _pack_kv_f32_array(key: str, values) -> bytes:
    kb = key.encode("utf-8")
    return (struct.pack("<Q", len(kb)) + kb
            + struct.pack("<IIQ", 9, 6, len(values))
            + b"".join(struct.pack("<f", v) for v in values))


def test_reads_context_and_layers(tmp_path):
    blob = pack_header(tensor_count=0, kv_count=4)
    blob += _pack_kv_str("general.name", "tiny-test")
    blob += pack_kv_int("llama.context_length", 8192)
    blob += _pack_kv_f32_array("llama.rope.scaling", [1.0, 2.0])
    blob += pack_kv_int("llama.block_count", 32)
    p = tmp_path / "model.gguf"
    p.write_bytes(blob)

    meta = read_gguf_meta(str(p))
    assert meta.context_length == 8192
    assert meta.block_count == 32


def test_early_exit_never_reads_tensor_table(tmp_path):
    # Both keys first, then a KV with a deliberately corrupt value type.
    # Early exit means the parser must succeed without ever touching it.
    blob = pack_header(tensor_count=999, kv_count=3)
    blob += pack_kv_int("qwen2.context_length", 32768)
    blob += pack_kv_int("qwen2.block_count", 48)
    kb = b"corrupt.key"
    blob += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 0xDEAD)
    p = tmp_path / "model.gguf"
    p.write_bytes(blob)

    meta = read_gguf_meta(str(p))
    assert meta.context_length == 32768
    assert meta.block_count == 48


def test_missing_keys_yield_none(tmp_path):
    blob = pack_header(tensor_count=0, kv_count=1)
    blob += _pack_kv_str("general.name", "no-limits")
    p = tmp_path / "model.gguf"
    p.write_bytes(blob)

    meta = read_gguf_meta(str(p))
    assert meta.context_length is None
    assert meta.block_count is None


def test_uint64_values(tmp_path):
    kb = b"llama.context_length"
    blob = pack_header(tensor_count=0, kv_count=1)
    blob += struct.pack("<Q", len(kb)) + kb + struct.pack("<IQ", 10, 131072)
    p = tmp_path / "model.gguf"
    p.write_bytes(blob)

    assert read_gguf_meta(str(p)).context_length == 131072


def test_rejects_non_gguf(tmp_path):
    p = tmp_path / "junk.gguf"
    p.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ValueError, match="not a GGUF file"):
        read_gguf_meta(str(p))


def test_rejects_truncated(tmp_path):
    blob = pack_header(tensor_count=0, kv_count=2)
    blob += pack_kv_int("llama.context_length", 4096)
    # Second promised KV never arrives.
    p = tmp_path / "model.gguf"
    p.write_bytes(blob)
    with pytest.raises(ValueError, match="truncated"):
        read_gguf_meta(str(p))


def test_rejects_unsupported_version(tmp_path):
    # Big-endian GGUF reads as a huge LE version number -> ValueError,
    # which routes callers to the gguf-package fallback.
    p = tmp_path / "model.gguf"
    p.write_bytes(GGUF_MAGIC + struct.pack(">I", 3) + b"\x00" * 32)
    with pytest.raises(ValueError, match="unsupported GGUF version"):
        read_gguf_meta(str(p))
