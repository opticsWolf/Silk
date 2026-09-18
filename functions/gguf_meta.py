# -*- coding: utf-8 -*-
"""Lightweight GGUF metadata probe.

Reads ONLY the GGUF header and metadata KV section, stopping as soon as the
wanted keys are found.  It never reaches the tensor-info table, so probing a
multi-GB model costs a scan of the KV section instead of a full metadata
+ tensor-table parse (which is what ``gguf.GGUFReader`` does on construction,
including an mmap of the whole file).

Cost is bounded by that section, not by the file: measured 0.1 ms for a
projector with no template, 74-185 ms for real 4-16 GB instruct models, and
the same for a 16 GB model as for a 4 GB one.  The chat template sits near
the end of the KV section, after the tokenizer's arrays, so asking for
``supports_tools`` is what turns a few-KB read into that scan -- which is
why the loader's probe runs on a daemon thread.

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


#: Markers of a tool-aware chat template. Both must appear: a template
#: that renders tool *calls* and is handed a ``tools`` list is one that was
#: written for structured calling. Either alone is too easy to hit by
#: accident -- "tools" turns up in prose, and some templates mention
#: ``tool_calls`` only to skip over a role they do not otherwise support.
_TOOL_MARKERS = ("tools", "tool_calls")

#: A template longer than this is not read past the cut. The signal is in
#: the control flow near the top, and an unbounded read from a file we are
#: only probing is how a corrupt length field becomes a memory spike.
_TEMPLATE_CAP = 256 * 1024


@dataclass(frozen=True)
class GGUFMeta:
    """What the loader can learn without loading the model.

    ``context_length`` and ``block_count`` clamp the spinboxes.
    ``supports_tools`` decides a protocol: True puts the agent's tools in
    the request's ``tools`` field, False asks for them in a text fence.
    ``None`` means the file carried no chat template at all -- an embedding
    model or a vision projector -- and is not the same answer as False.
    """
    context_length: Optional[int] = None
    block_count: Optional[int] = None
    supports_tools: Optional[bool] = None


class UnsupportedValue(ValueError):
    """A KV whose value type this parser cannot walk past.

    Distinct from a truncated or non-GGUF file: the file is intact, we
    simply do not know how wide this value is, so the scan cannot
    continue -- but whatever was read before it is still good. A
    truncated file is broken and stays an ordinary ValueError, because
    the caller's answer to that is to fall back to `gguf.GGUFReader`.
    """


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


def _read_string(f: BinaryIO, version: int) -> str:
    """Read one string value, capped. The stream is left past it either way."""
    size = _read_len(f, version)
    if size > _TEMPLATE_CAP:
        head = _read(f, _TEMPLATE_CAP)
        f.seek(size - _TEMPLATE_CAP, 1)
        return head.decode("utf-8", errors="replace")
    return _read(f, size).decode("utf-8", errors="replace")


def template_supports_tools(template: str) -> bool:
    """Whether a chat template was written for structured tool calling.

    A heuristic, and deliberately the conservative one: it reads the
    template rather than the model's name, and it wants both markers. A
    false positive costs a run -- a server handed a ``tools`` field it
    cannot render refuses the request -- while a false negative only falls
    back to the fence protocol, which works everywhere.
    """
    low = template.lower()
    return all(marker in low for marker in _TOOL_MARKERS)


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
            raise UnsupportedValue(
                f"unsupported GGUF array element type {elem_type}")
        return None
    raise UnsupportedValue(f"unknown GGUF value type {vtype}")


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
        tools: Optional[bool] = None
        for _ in range(kv_count):
            try:
                key = _read_key(f, version)
                vtype = _read_u32(f)
                if vtype == _T_STRING and key.endswith("chat_template"):
                    tools = template_supports_tools(_read_string(f, version))
                    if ctx is not None and layers is not None:
                        break
                    continue
                val = _read_value(f, vtype, version)
            except UnsupportedValue:
                # The spinbox limits are worth more than the protocol hint.
                # Before the template was wanted this loop stopped as soon as
                # it had them and never saw a later exotic KV; now that it
                # scans on, a value type we cannot walk must cost the hint
                # alone rather than the values already in hand. A *truncated*
                # file is not salvaged this way -- it is broken, and the
                # caller's answer to that is the `gguf` package.
                if ctx is None and layers is None:
                    raise
                break
            if val is None:
                continue
            if key.endswith(".context_length"):
                ctx = val
            elif key.endswith(".block_count"):
                layers = val
            if ctx is not None and layers is not None and tools is not None:
                break  # never reads the remaining KVs or the tensor table
        return GGUFMeta(context_length=ctx, block_count=layers,
                        supports_tools=tools)


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
