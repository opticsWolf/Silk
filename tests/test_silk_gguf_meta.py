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


# ── the chat template, and the protocol it decides ───────────────────────

def _tool_template() -> str:
    """The shape a tool-aware template has: given tools, renders calls."""
    return ("{% if tools %}{{ tools | tojson }}{% endif %}"
            "{% for m in messages %}{% if m.tool_calls %}"
            "{{ m.tool_calls }}{% endif %}{% endfor %}")


def test_a_tool_aware_template_is_detected(tmp_path):
    blob = pack_header(tensor_count=0, kv_count=3)
    blob += pack_kv_int("qwen3.context_length", 32768)
    blob += pack_kv_int("qwen3.block_count", 48)
    blob += _pack_kv_str("tokenizer.chat_template", _tool_template())
    p = tmp_path / "m.gguf"
    p.write_bytes(blob)

    assert read_gguf_meta(str(p)).supports_tools is True


def test_a_plain_chat_template_is_not_tool_aware(tmp_path):
    blob = pack_header(tensor_count=0, kv_count=3)
    blob += pack_kv_int("qwen3.context_length", 4096)
    blob += pack_kv_int("qwen3.block_count", 32)
    blob += _pack_kv_str(
        "tokenizer.chat_template",
        "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}")
    p = tmp_path / "m.gguf"
    p.write_bytes(blob)

    assert read_gguf_meta(str(p)).supports_tools is False


def test_no_template_at_all_is_unknown_not_false(tmp_path):
    """An embedding model or a vision projector carries none.

    None and False are different answers: False is "read it, it cannot",
    None is "there was nothing to read". Both end up on the fence
    protocol, but only one of them is a statement about the model.
    """
    blob = pack_header(tensor_count=0, kv_count=2)
    blob += pack_kv_int("bert.context_length", 512)
    blob += pack_kv_int("bert.block_count", 12)
    p = tmp_path / "m.gguf"
    p.write_bytes(blob)

    assert read_gguf_meta(str(p)).supports_tools is None


def test_half_a_marker_is_not_a_tool_template(tmp_path):
    """"tools" turns up in prose; the conservative read is the cheap one.

    A false positive costs a run -- a server handed a `tools` field its
    template cannot render refuses the request outright -- while a false
    negative only falls back to fences.
    """
    blob = pack_header(tensor_count=0, kv_count=1)
    blob += _pack_kv_str(
        "tokenizer.chat_template",
        "{# these tools are not that kind of tools #}{{ messages }}")
    p = tmp_path / "m.gguf"
    p.write_bytes(blob)

    assert read_gguf_meta(str(p)).supports_tools is False


def test_an_enormous_template_is_capped_not_swallowed(tmp_path):
    """A corrupt length field must not become a memory spike."""
    from silk.functions.gguf_meta import _TEMPLATE_CAP

    template = _tool_template() + ("x" * (_TEMPLATE_CAP + 4096))
    blob = pack_header(tensor_count=0, kv_count=2)
    blob += _pack_kv_str("tokenizer.chat_template", template)
    blob += pack_kv_int("llama.block_count", 32)
    p = tmp_path / "m.gguf"
    p.write_bytes(blob)

    meta = read_gguf_meta(str(p))
    assert meta.supports_tools is True, "the markers are near the top"
    assert meta.block_count == 32, (
        "and the stream is still positioned for the next KV: a capped read "
        "must skip the remainder, not leave the parser inside the string"
    )


def test_an_unreadable_later_kv_costs_the_hint_not_the_limits(tmp_path):
    """The values already read outlive a KV the parser cannot walk.

    Before the chat template was wanted, this loop stopped as soon as it
    had context and layers, so a later exotic KV was never reached. Now
    that it scans on, that KV must not retroactively break a file the
    probe used to read fine.
    """
    blob = pack_header(tensor_count=999, kv_count=3)
    blob += pack_kv_int("qwen2.context_length", 32768)
    blob += pack_kv_int("qwen2.block_count", 48)
    kb = b"corrupt.key"
    blob += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 0xDEAD)
    p = tmp_path / "m.gguf"
    p.write_bytes(blob)

    meta = read_gguf_meta(str(p))
    assert (meta.context_length, meta.block_count) == (32768, 48)
    assert meta.supports_tools is None, "unknown, and the fence protocol"


def test_a_file_that_is_unreadable_from_the_start_still_raises(tmp_path):
    """Salvage is for a tail, not for a file that was never parseable."""
    blob = pack_header(tensor_count=0, kv_count=1)
    kb = b"corrupt.key"
    blob += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 0xDEAD)
    p = tmp_path / "m.gguf"
    p.write_bytes(blob)

    with pytest.raises(ValueError):
        read_gguf_meta(str(p))
