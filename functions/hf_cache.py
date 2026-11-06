# -*- coding: utf-8 -*-
"""HuggingFace hub-cache reader (stdlib only, never touches the network).

A hub cache (``.../hub``) holds one dir per cached repo::

    models--org--name/
        blobs/                  content-addressed payloads
        refs/main               the pinned commit hash (a text file)
        snapshots/<commit>/     repo-relative paths (links or copies)

``scan`` turns that layout into repo ids plus their weight files, so a
node can offer them without ``huggingface_hub`` installed -- listing a
local cache needs no Hub client, and on an air-gapped box that is the
whole point. If downloads are ever wanted, *that* is when the library
earns its place (as an optional import, never a hard dependency).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#: Suffixes that count as model weights for the file picker.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".onnx",
                   ".pt", ".pth", ".ckpt")

#: ``model-00001-of-00004.safetensors`` -- shards sort behind whole files.
_SHARD_RE = re.compile(r"-\d+-of-\d+", re.IGNORECASE)


@dataclass
class HFCachedFile:
    """One file inside a snapshot: name, snapshot path, byte size."""

    name: str
    path: str
    size: int = 0

    @property
    def is_shard(self) -> bool:
        return bool(_SHARD_RE.search(self.name))


@dataclass
class HFCachedRepo:
    """One ``models--*`` dir: display id, dir, revision, weight files."""

    repo_id: str
    repo_dir: str
    revision: Optional[str] = None          # commit hash, refs/main preferred
    via_ref: bool = False                   # True when refs/main pinned it
    files: List[HFCachedFile] = field(default_factory=list)
    loose: bool = False                     # True: plain folder, no hub layout


def default_cache_dir() -> Tuple[str, str]:
    """The hub dir by huggingface_hub's documented precedence.

    Returns ``(path, source)``: the first *existing* candidate wins, else
    the platform default with ``" (missing)"`` so the caller can say so.
    """
    home = os.path.expanduser("~")
    xdg = os.environ.get("XDG_CACHE_HOME") or ""
    candidates = [
        ("env:HF_HUB_CACHE", os.environ.get("HF_HUB_CACHE") or ""),
        ("env:HUGGINGFACE_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE") or ""),
        ("env:HF_HOME", os.path.join(os.environ.get("HF_HOME") or "", "hub")),
        ("env:XDG_CACHE_HOME",
         os.path.join(xdg, "huggingface", "hub") if xdg else ""),
        ("default", os.path.join(home, ".cache", "huggingface", "hub")),
    ]
    for source, path in candidates:
        if path and os.path.isdir(path):
            return os.path.normpath(path), source
    return os.path.normpath(candidates[-1][1]), "default (missing)"


def find_cache_dir(explicit: Optional[str] = None) -> Tuple[str, str]:
    """Explicit path wins; blank falls back to :func:`default_cache_dir`."""
    explicit = (explicit or "").strip()
    if explicit:
        return os.path.normpath(os.path.expanduser(explicit)), "explicit"
    return default_cache_dir()


def candidate_cache_dirs() -> List[Tuple[str, str]]:
    """Every configured hub-cache location that exists, precedence order.

    Several roots can coexist (e.g. ``HF_HUB_CACHE`` set while the
    default ``~/.cache`` is still populated); :func:`default_cache_dir`
    only reports the winner. The browser offers all of these so a
    second cache is one dropdown pick away. Deduped: one env var can
    alias another (``HF_HOME/hub`` often *is* the default).
    """
    home = os.path.expanduser("~")
    xdg = os.environ.get("XDG_CACHE_HOME") or ""
    candidates = [
        ("env:HF_HUB_CACHE", os.environ.get("HF_HUB_CACHE") or ""),
        ("env:HUGGINGFACE_HUB_CACHE",
         os.environ.get("HUGGINGFACE_HUB_CACHE") or ""),
        ("env:HF_HOME", os.path.join(os.environ.get("HF_HOME") or "", "hub")),
        ("env:XDG_CACHE_HOME",
         os.path.join(xdg, "huggingface", "hub") if xdg else ""),
        ("default", os.path.join(home, ".cache", "huggingface", "hub")),
    ]
    found: List[Tuple[str, str]] = []
    seen = set()
    for source, path in candidates:
        normed = os.path.normcase(os.path.normpath(path)) if path else ""
        if normed and normed not in seen and os.path.isdir(path):
            seen.add(normed)
            found.append((os.path.normpath(path), source))
    return found


def parse_repo_id(dirname: str) -> Optional[str]:
    """``models--org--name`` -> ``org/name``; anything else -> None."""
    if not dirname.startswith("models--"):
        return None
    return dirname[len("models--"):].replace("--", "/")


def _read_ref(repo_dir: str) -> Optional[str]:
    for ref in ("refs/main", "refs/heads/main"):
        try:
            with open(os.path.join(repo_dir, ref), "r",
                      encoding="utf-8") as fh:
                commit = fh.read().strip().split()[0]
        except (OSError, IndexError):
            continue
        if commit and os.path.isdir(os.path.join(repo_dir, "snapshots", commit)):
            return commit
    return None


def _newest_snapshot(repo_dir: str) -> Optional[str]:
    snaps = os.path.join(repo_dir, "snapshots")
    try:
        entries = [e for e in os.scandir(snaps) if e.is_dir()]
    except OSError:
        return None
    if not entries:
        return None
    entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
    return entries[0].name


def resolve_snapshot(repo_dir: str) -> Tuple[Optional[str], bool]:
    """The commit to read: ``refs/main`` first, else the newest snapshot."""
    commit = _read_ref(repo_dir)
    if commit is not None:
        return commit, True
    return _newest_snapshot(repo_dir), False


def snapshot_files(repo_dir: str, commit: str) -> List[HFCachedFile]:
    """Weight files of one snapshot, largest first (shards behind)."""
    snap = os.path.join(repo_dir, "snapshots", commit)
    try:
        names = sorted(os.listdir(snap))
    except OSError:
        return []
    out = []
    for name in names:
        if not name.lower().endswith(WEIGHT_SUFFIXES):
            continue
        path = os.path.join(snap, name)
        try:
            if not os.path.isfile(path):
                continue
            size = os.path.getsize(path)
        except OSError:
            continue
        out.append(HFCachedFile(name=name, path=path, size=size))
    out.sort(key=lambda f: (f.is_shard, -f.size, f.name))
    return out


def scan_repos(hub_dir: str,
               include_loose: bool = True) -> List[HFCachedRepo]:
    """Every ``models--*`` dir with a readable snapshot, sorted by id.

    With ``include_loose`` (default), plain subfolders holding weight
    files are listed too -- a downloads shelf or a vendor tree scans the
    same way as the hub cache, flagged ``loose`` with no revision.
    """
    try:
        entries = sorted(os.listdir(hub_dir))
    except OSError:
        return []
    repos = []
    for dirname in entries:
        repo_id = parse_repo_id(dirname)
        if repo_id is None:
            continue
        repo_dir = os.path.join(hub_dir, dirname)
        commit, via_ref = resolve_snapshot(repo_dir)
        if commit is None:
            continue
        files = snapshot_files(repo_dir, commit)
        if not files:
            continue
        repos.append(HFCachedRepo(repo_id=repo_id, repo_dir=repo_dir,
                                  revision=commit, via_ref=via_ref,
                                  files=files))
    if include_loose:
        repos.extend(scan_loose_weights(hub_dir))
    repos.sort(key=lambda r: r.repo_id.lower())
    return repos


#: Hub-internals and hub-layout dirs a loose scan never descends into.
_LOOSE_PRUNE = frozenset({"blobs", "snapshots", "refs"})


def scan_loose_weights(root: str, max_depth: int = 4,
                       max_files: int = 2000) -> List[HFCachedRepo]:
    """Plain subfolders holding weight files, one repo per folder.

    ``models--*`` dirs belong to :func:`scan_repos` (listed, never
    descended -- a blob store is big and extensionless anyway), as do
    hidden dirs and hub internals. The repo id is the folder's path
    relative to *root* (``/``-separated), or ``"(root)"`` for weights
    sitting directly in *root*. Bounded: at most *max_depth* folders
    deep, at most *max_files* files, so a stray drive root can neither
    hang the scan nor flood the picker.
    """
    if max_files <= 0 or max_depth < 0:
        return []
    groups: dict = {}
    total = 0
    stop = False
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        dirnames[:] = sorted(
            d for d in dirnames
            if not (d.startswith("models--") or d.startswith(".")
                    or d in _LOOSE_PRUNE)
            and depth + 1 <= max_depth)
        for name in sorted(filenames):
            if not name.lower().endswith(WEIGHT_SUFFIXES):
                continue
            path = os.path.join(dirpath, name)
            try:
                if not os.path.isfile(path):
                    continue
                size = os.path.getsize(path)
            except OSError:
                continue
            groups.setdefault(dirpath, []).append(
                HFCachedFile(name=name, path=path, size=size))
            total += 1
            if total >= max_files:
                stop = True
                break
        if stop:
            break
    repos = []
    for dirpath, files in groups.items():
        rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
        repo_id = "(root)" if rel == "." else rel
        files.sort(key=lambda f: (f.is_shard, -f.size, f.name))
        repos.append(HFCachedRepo(repo_id=repo_id, repo_dir=dirpath,
                                  files=files, loose=True))
    repos.sort(key=lambda r: r.repo_id.lower())
    return repos


def describe_size(size: int) -> str:
    """``4212345678`` -> ``"3.9 GB"`` for the file picker."""
    value, unit = float(size), "B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            break
        value /= 1024
    return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"


def hub_repo_dir(hub_dir: str, repo_id: str) -> str:
    """The ``models--org--name`` dir for a repo id (created on demand)."""
    path = os.path.join(hub_dir, "models--" + repo_id.replace("/", "--"))
    os.makedirs(os.path.join(path, "blobs"), exist_ok=True)
    os.makedirs(os.path.join(path, "snapshots"), exist_ok=True)
    return path


def write_ref_main(repo_dir: str, commit: str) -> None:
    """Point ``refs/main`` at a commit so scanners resolve the snapshot."""
    refs = os.path.join(repo_dir, "refs")
    os.makedirs(refs, exist_ok=True)
    with open(os.path.join(refs, "main"), "w", encoding="utf-8") as fh:
        fh.write(commit + "\n")


def link_blob(repo_dir: str, sha256: str, src: str) -> bool:
    """Hardlink the snapshot file into ``blobs/<sha256>`` (fidelity for
    other cache readers); ``False`` when linking is impossible."""
    if not (sha256 and os.path.isfile(src)):
        return False
    dest = os.path.join(repo_dir, "blobs", sha256)
    if os.path.exists(dest):
        return True
    try:
        os.link(src, dest)
        return True
    except OSError:
        return False
