# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

Minimal HuggingFace Hub client (stdlib only).

Listing, searching and downloading need no Hub library: the search and
repo endpoints are plain JSON over HTTPS, and file downloads are plain
``GET``s that follow redirects to the CDN. Downloads resume across
failures (``Range`` + ``.partial`` + ``.resume.json`` sidecar), verify
against the Hub's own integrity channel (``X-Linked-ETag``/``ETag`` or
the ``blobs`` listing), and never send the token anywhere except
``huggingface.co`` -- a redirect handler strips ``Authorization`` the
moment the host changes, so a CDN never sees it.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

HUB_ORIGIN = "https://huggingface.co"
API = HUB_ORIGIN + "/api"
CHUNK = 1024 * 1024          # per-recv cap; recvs return what is
                              # available (read1), so Cancel is checked
                              # between network round trips, not per MiB
TIMEOUT_S = 30.0
RETRIES = 3
RETRY_BACKOFF_S = (1.0, 2.0, 4.0)

_PROGRESS = Callable[[int, int], None]      # (downloaded, total); total 0 = unknown
_CANCELLED = Callable[[], bool]


class HubError(Exception):
    """A Hub failure with a machine-readable ``kind`` for the status line."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind  # auth | not_found | network | hash | cancelled | api


@dataclass
class HubFile:
    """One downloadable repo file."""

    name: str
    size: Optional[int] = None
    sha256: Optional[str] = None


# ── Auth-safe HTTP ───────────────────────────────────────────────────
class _AuthStrippingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop credentials the moment a redirect leaves the Hub.

    ``urllib`` follows ``/resolve/`` to CDN-signed URLs; the signature
    *is* the authorization there, so the Bearer token must never travel
    with it (CDN logs are not our logs).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        out = super().redirect_request(req, fp, code, msg, headers, newurl)
        if out is not None:
            old_host = urllib.parse.urlsplit(req.full_url).hostname
            new_host = urllib.parse.urlsplit(out.full_url).hostname
            if old_host != new_host:
                out.remove_header("Authorization")
                out.remove_header("Cookie")
        return out


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_AuthStrippingRedirectHandler)


def _request(url: str, token: Optional[str] = None,
             headers: Optional[Dict[str, str]] = None) -> urllib.request.Request:
    req = urllib.request.Request(url, headers=dict(headers or {}))
    if token and urllib.parse.urlsplit(url).hostname == urllib.parse.urlsplit(
            HUB_ORIGIN).hostname:
        req.add_header("Authorization", f"Bearer {token}")
    return req


def _read_json(url: str, token: Optional[str] = None) -> Any:
    try:
        with _opener().open(_request(url, token), timeout=TIMEOUT_S) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise HubError("auth",
                           f"{exc.code} from the Hub -- check the token, and for "
                           f"gated repos accept the license on the site first.") from exc
        if exc.code == 404:
            raise HubError("not_found", f"Not on the Hub: {url}") from exc
        raise HubError("network", f"Hub HTTP {exc.code}: {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError,
            http.client.HTTPException) as exc:
        raise HubError("network", f"Hub unreachable: {exc}") from exc


# ── Search + listing ─────────────────────────────────────────────────
def search_models(query: str, *, sort: str = "downloads", limit: int = 50,
                  token: Optional[str] = None) -> List[Dict[str, Any]]:
    """``GET /api/models?search=…``; entries keep id, stats and siblings."""
    params = urllib.parse.urlencode({
        "search": query, "sort": sort, "direction": "-1", "limit": max(1, limit),
    })
    data = _read_json(f"{API}/models?{params}", token)
    out = []
    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        out.append({
            "id": entry["id"],
            "likes": entry.get("likes", 0),
            "downloads": entry.get("downloads", 0),
            "pipeline_tag": entry.get("pipeline_tag"),
            "siblings": [s.get("rfilename", "") for s in entry.get("siblings", [])
                         if isinstance(s, dict)],
        })
    return out


def repo_files(repo_id: str, *, revision: str = "main",
               token: Optional[str] = None) -> Tuple[List[HubFile], str]:
    """All files of one repo revision; ``(files, pinned_revision)``.

    ``?blobs=true`` adds per-file ``lfs.oid`` (sha256) and size; without
    it sizes/hashes come from the download headers instead.
    """
    url = f"{API}/models/{repo_id}/revision/{urllib.parse.quote(revision, safe='')}"
    try:
        data = _read_json(url + "?blobs=true", token)
    except HubError as exc:
        if exc.kind == "not_found":
            data = _read_json(f"{API}/models/{repo_id}", token)
        else:
            raise
    siblings = data.get("siblings", []) if isinstance(data, dict) else []
    files = []
    for s in siblings:
        if not isinstance(s, dict) or not s.get("rfilename"):
            continue
        lfs = s.get("lfs") or {}
        files.append(HubFile(name=s["rfilename"],
                             size=s.get("size") or (lfs.get("size")),
                             sha256=(lfs.get("oid") or None)))
    sha = data.get("sha") if isinstance(data, dict) else None
    return files, (sha or revision)


def filter_files(files: List[HubFile], kind: str) -> List[HubFile]:
    """``any`` | ``safetensors`` | ``gguf`` -- by file suffix."""
    if kind == "safetensors":
        return [f for f in files if f.name.endswith(".safetensors")]
    if kind == "gguf":
        return [f for f in files if f.name.endswith(".gguf")]
    return list(files)


#: Small files a weight download usually needs alongside the tensors.
CONFIG_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                "generation_config.json", "special_tokens_map.json",
                "chat_template.jinja")


def with_companions(all_files: List[HubFile], selected: List[HubFile],
                    *, include_config: bool = False) -> List[HubFile]:
    """The selection plus what it needs to load.

    Shard indexes (``*.index.json``) always join weight downloads -- a
    sharded set without its index is unloadable. Config/tokenizer files
    join only on request (the node's checkbox, default on): kilobytes
    that are harmless for GGUF and load-bearing for safetensors.
    """
    names = {f.name for f in selected}
    out = list(selected)
    shards_queued = any("-of-" in f.name
                          and f.name.endswith((".safetensors", ".bin"))
                          for f in selected)
    for f in all_files:
        if f.name in names:
            continue
        if f.name.endswith(".index.json") and shards_queued:
            names.add(f.name)
            out.append(f)
        elif include_config and f.name in CONFIG_FILES:
            names.add(f.name)
            out.append(f)
    return out


def repo_refs(repo_id: str, *,
            token: Optional[str] = None) -> Dict[str, List[str]]:
    """``GET /api/models/{id}/refs``; ``{"branches": [...], "tags": [...]}``.

    Names only, branches in API order then tags. Empty (not an error)
    when the Hub has no refs to show -- callers fall back to the typed
    revision, and the file listing reports the real problem if any.
    """
    try:
        data = _read_json(f"{API}/models/{repo_id}/refs", token)
    except HubError:
        return {"branches": [], "tags": []}
    if not isinstance(data, dict):
        return {"branches": [], "tags": []}

    def _names(key: str) -> List[str]:
        out: List[str] = []
        for item in data.get(key) or []:
            name = item.get("name") if isinstance(item, dict) else None
            if name and name not in out:
                out.append(str(name))
        return out

    return {"branches": _names("branches"), "tags": _names("tags")}


def resolve_url(repo_id: str, revision: str, filename: str) -> str:
    """The CDN-redirecting download URL for one file."""
    return (f"{HUB_ORIGIN}/{repo_id}/resolve/"
            f"{urllib.parse.quote(revision, safe='')}/{filename}")


# ── Resumable download ───────────────────────────────────────────────
def _sidecar_paths(dest: str) -> Tuple[str, str]:
    return dest + ".partial", dest + ".resume.json"


def _read_sidecar(meta_path: str) -> Dict[str, Any]:
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _etag_of(headers: Any) -> Optional[str]:
    for key in ("X-Linked-ETag", "ETag"):
        tag = headers.get(key)
        if not tag:
            continue
        tag = tag.strip().strip('"')
        if len(tag) == 64 and "-" not in tag and all(
                c in "0123456789abcdefABCDEF" for c in tag):
            return tag.lower()      # LFS oid: the file's sha256
    return None                     # S3 multipart tags are not hashes


def resume_state(dest: str, url: str) -> Dict[str, Any]:
    """What a previous attempt left behind: ``{resumable, downloaded,
    total}``. ``resumable`` needs the same URL, a known total and room
    left -- anything else means start fresh. A full-size partial is
    reported (not resumable) so the caller can adopt it."""
    partial, meta_path = _sidecar_paths(dest)
    out = {"resumable": False, "downloaded": 0, "total": 0,
           "sha256": None}
    try:
        have = os.path.getsize(partial)
    except OSError:
        return out
    meta = _read_sidecar(meta_path)
    total = int(meta.get("total") or 0)
    out["sha256"] = meta.get("sha256")
    if meta.get("url") == url and total > 0 and 0 < have < total:
        out.update(resumable=True, downloaded=have, total=total)
    elif have >= total > 0:
        out.update(downloaded=have, total=total)   # complete-ish: verify below
    return out


def _adopt_complete(partial: str, meta_path: str, dest: str,
                    sha256: Optional[str]) -> Optional[bool]:
    """Adopt a fully-written partial (verify + rename).

    Returns whether the hash was verified, or ``None`` when adoption is
    impossible (missing/corrupt partial -- the caller starts fresh).
    """
    try:
        if not os.path.isfile(partial):
            return None
    except OSError:
        return None
    if sha256 and _sha256_of(partial) != sha256:
        for path in (partial, meta_path):
            try:
                os.remove(path)
            except OSError:
                pass
        return None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".",
                    exist_ok=True)
        os.replace(partial, dest)
    except OSError:
        return None
    try:
        os.remove(meta_path)
    except OSError:
        pass
    return bool(sha256)


def download_file(url: str, dest: str, *, expected_size: Optional[int] = None,
                  sha256: Optional[str] = None, token: Optional[str] = None,
                  progress: Optional[_PROGRESS] = None,
                  is_cancelled: Optional[_CANCELLED] = None,
                  retries: int = RETRIES) -> Dict[str, Any]:
    """GET ``url`` to ``dest`` with resume, retries, cancel and verify.

    Returns ``{path, bytes, resumed, sha256_verified}``. Raises
    :class:`HubError` (``cancelled`` keeps the partial + sidecar so a
    later call resumes; anything terminal explains itself).
    """
    is_cancelled = is_cancelled or (lambda: False)
    partial, meta_path = _sidecar_paths(dest)
    state = resume_state(dest, url)
    if (state["total"] and state["downloaded"] == state["total"]
            and not os.path.exists(dest)):
        # A previous run wrote every byte but never finalized (crash or
        # cancel between the last write and the rename): verify + adopt
        # instead of re-downloading.
        verified = _adopt_complete(partial, meta_path, dest,
                                   sha256 or state.get("sha256"))
        if verified is not None:
            return {"path": dest, "bytes": state["downloaded"],
                    "resumed": True, "sha256_verified": verified}
    have = state["downloaded"] if state["resumable"] else 0
    if not state["resumable"]:
        for path in (partial, meta_path):
            try:
                os.remove(path)
            except OSError:
                pass
    tag = sha256
    total = expected_size or state.get("total") or 0

    def attempt() -> Tuple[int, Optional[str]]:
        nonlocal have, total, tag
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            resp = _opener().open(_request(url, token, headers),
                                  timeout=TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                raise _Restart(str(exc)) from exc  # remote shrank: start over
            if exc.code in (401, 403):
                raise HubError("auth",
                               f"{exc.code} -- check the token, and for gated repos "
                               f"accept the license on the site first.") from exc
            if exc.code == 404:
                raise HubError("not_found", f"Nothing at {url}") from exc
            raise HubError("network", f"Hub HTTP {exc.code}: {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ConnectionError,
                http.client.HTTPException) as exc:
            if is_cancelled():
                # A stalled socket tripped the timeout after Cancel:
                # report the cancel, keep the partial for resume.
                raise HubError("cancelled",
                               f"Cancelled at {have}/{total or '?'} bytes "
                               f"-- resume to continue.") from exc
            raise HubError("network", f"Hub unreachable: {exc}") from exc
        with resp:
            status = getattr(resp, "status", 200)
            if have and status == 200:
                raise _Restart("server ignored Range: restarting from zero")
            if status == 416:
                raise _Restart("range unsatisfiable: restarting from zero")
            if status not in (200, 206):
                raise HubError("network", f"Hub HTTP {status}: {url}")
            tag = tag or _etag_of(resp.headers)
            total = int(resp.headers.get("Content-Range", "").rsplit("/", 1)[-1]
                        or resp.headers.get("Content-Length") or total or 0)
            try:
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump({"url": url, "total": total,
                               "sha256": tag}, fh)
                with open(partial, "ab" if have else "wb") as fh:
                    # read1, not read: read() loops until the buffer is full
                    # (or EOF), so one stalled chunk would wedge Cancel
                    # behind the whole socket timeout. read1 returns what
                    # is available after a single recv; EOF still ends the
                    # loop, silence still trips the timeout into a resume.
                    read1 = getattr(resp.fp, "read1", None)
                    while True:
                        if is_cancelled():
                            raise HubError("cancelled",
                                           f"Cancelled at {have}/{total or '?'} bytes "
                                           f"-- resume to continue.")
                        chunk = read1(CHUNK) if read1 else resp.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        if progress is not None:
                            progress(have, total)
            except OSError as exc:
                raise HubError("network", f"Local write failed: {exc}") from exc
            if total and have != total:
                # Clean EOF before the promised bytes (truncated read):
                # retryable -- the next attempt resumes from `have`.
                raise HubError("network",
                               f"Short read: {have}/{total} bytes -- resuming.")
        return have, tag

    last: Optional[Exception] = None
    for i in range(max(1, retries)):
        try:
            have, tag = attempt()
            break
        except _Restart as exc:
            have, total, tag = 0, expected_size or 0, sha256
            for path in (partial, meta_path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            last = exc
            continue
        except HubError as exc:
            if exc.kind in ("cancelled", "auth", "not_found", "hash"):
                raise
            last = exc
            time.sleep(RETRY_BACKOFF_S[min(i, len(RETRY_BACKOFF_S) - 1)])
    else:
        raise HubError("network", f"Gave up after {retries} tries: {last}")

    if total and have != total:
        raise HubError("network",
                       f"Short read: {have}/{total} bytes -- resume to continue.")
    digest: Optional[str] = None
    if tag:
        digest = _sha256_of(partial)
        if digest != tag:
            for path in (partial, meta_path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            raise HubError("hash",
                           f"Checksum mismatch for {os.path.basename(dest)} -- "
                           f"partial discarded, download again.")
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    os.replace(partial, dest)
    try:
        os.remove(meta_path)
    except OSError:
        pass
    return {"path": dest, "bytes": have, "resumed": state["resumable"],
            "sha256_verified": bool(tag and digest == tag)}


class _Restart(Exception):
    """Internal: discard the partial and try once from zero."""


def _sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
