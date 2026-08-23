# -*- coding: utf-8 -*-
"""Lightweight GGUF metadata probe.

Reads ONLY the GGUF header and metadata KV section, stopping as soon as the
wanted keys are found.  It never reaches the tensor-info table, so probing a
multi-GB model costs a few KB of sequential reads instead of a full metadata
+ tensor-table parse (which is what ``gguf.GGUFReader`` does on construction,
including an mmap of the whole file).

Qt-free on purpose: shared by the loader node's background probe and headless
tests.
"""

import struct
from dataclasses import dataclass
from typing import BinaryIO, Optional, Tuple

GGUF_MAGIC = b"GGUF"

# GGUF value-type id -> (struct format, byte size) for fixed-size scalars.
_SCALAR_FMT = {
    0: ("<B", 1),   # uint8
    1: ("<b", 1),   # int8
    2: ("<H", 2),   # uint16
    3: ("<h", 2),   # int16
    4: ("<I", 4),   # uint32
    5: ("<i", 4),   # int32
    6: ("<f", 4),   # float32
    7: ("<?", 1),   # bool
    10: ("<Q", 8),  # uint64
    11: ("<q", 8),  # int64
    12: ("<d", 8),  # float64
}
_FLOAT_TYPES = frozenset({6, 12})
_T_STRING = 8
_T_ARRAY = 9


@dataclass(frozen=True)
class GGUFMeta:
    """The two values the loader UI needs for spinbox clamping."""
    context_length: Optional[int] = None
    block_count: Optional[int] = None


def _read(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError("truncated GGUF file")
    return data


def _read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read(f, 4))[0]


def _read_len(f: BinaryIO, version: int) -> int:
    # GGUF v1 used uint32 for counts and string lengths; v2+ uses uint64.
    if version == 1:
        return _read_u32(f)
    return struct.unpack("<Q", _read(f, 8))[0]


def _read_key(f: BinaryIO, version: int) -> str:
    return _read(f, _read_len(f, version)).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int, version: int) -> Optional[int]:
    """Consume one KV value; return it only for integer scalars.

    Strings, floats and arrays are skipped via seek — the probe only cares
    about integer metadata, but must advance the stream correctly past
    everything else.
    """
    if vtype in _SCALAR_FMT:
        fmt, size = _SCALAR_FMT[vtype]
        val = struct.unpack(fmt, _read(f, size))[0]
        return None if vtype in _FLOAT_TYPES else int(val)
    if vtype == _T_STRING:
        f.seek(_read_len(f, version), 1)
        return None
    if vtype == _T_ARRAY:
        elem_type = _read_u32(f)
        count = _read_len(f, version)
        if elem_type in _SCALAR_FMT:
            f.seek(_SCALAR_FMT[elem_type][1] * count, 1)
        elif elem_type == _T_STRING:
            for _ in range(count):
                f.seek(_read_len(f, version), 1)
        else:
            raise ValueError(f"unsupported GGUF array element type {elem_type}")
        return None
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_meta(path: str) -> GGUFMeta:
    """Parse header + KV pairs of a GGUF file; early-exit once both keys hit.

    Raises ValueError for non-GGUF/truncated files and for unsupported
    versions (e.g. big-endian GGUF, whose version field reads as a huge
    little-endian number) — callers fall back to ``gguf.GGUFReader`` then.
    """
    with open(path, "rb") as f:
        if _read(f, 4) != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        version = _read_u32(f)
        if version not in (1, 2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        _read_len(f, version)  # tensor_count (unused)
        kv_count = _read_len(f, version)

        ctx: Optional[int] = None
        layers: Optional[int] = None
        for _ in range(kv_count):
            key = _read_key(f, version)
            val = _read_value(f, _read_u32(f), version)
            if val is None:
                continue
            if key.endswith(".context_length"):
                ctx = val
            elif key.endswith(".block_count"):
                layers = val
            if ctx is not None and layers is not None:
                break  # never reads the remaining KVs or the tensor table
        return GGUFMeta(context_length=ctx, block_count=layers)


def pack_kv_int(key: str, value: int) -> bytes:
    """Encode one uint32 KV pair (v2/v3 layout). Test/tooling helper."""
    kb = key.encode("utf-8")
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<II", 4, value)


def pack_header(tensor_count: int, kv_count: int, version: int = 3) -> bytes:
    """Encode a GGUF v2/v3 file header. Test/tooling helper."""
    return GGUF_MAGIC + struct.pack("<IQQ", version, tensor_count, kv_count)


def extract_reader_int(field: object) -> Optional[int]:
    """Extract an int from a ``gguf.GGUFReader`` field (fallback path)."""
    try:
        val = field.parts[-1]  # type: ignore[attr-defined]
        if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
            return int(val[0])
        return int(val)
    except Exception:
        return None


def read_gguf_meta_fallback(path: str) -> Tuple[Optional[int], Optional[int]]:
    """Full ``gguf.GGUFReader`` parse with early field-loop exit.

    Slow (parses the whole tensor table) — only used off the GUI thread when
    the manual parser rejects the file (exotic version / big-endian).
    """
    import gguf  # local import: optional dependency

    reader = gguf.GGUFReader(path)
    max_ctx: Optional[int] = None
    layers: Optional[int] = None
    for key, field in reader.fields.items():
        if key.endswith(".context_length"):
            max_ctx = extract_reader_int(field)
        elif key.endswith(".block_count"):
            layers = extract_reader_int(field)
        if max_ctx is not None and layers is not None:
            break
    return max_ctx, layers
