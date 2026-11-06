# -*- coding: utf-8
"""The Hub client: search, listing, resume, integrity, auth scope.

A fake Hub (threaded HTTP) serves bytes with real ``Range`` semantics,
so resume/restart/cancel are exercised, not mocked. Qt node tests live
at the bottom with in-fixture imports, so this file still collects
where PySide6 is missing.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import socket
import threading
import urllib.parse

import pytest

from silk.functions import hf_hub
from silk.functions.hf_hub import (
    HubError, download_file, filter_files, repo_files, repo_refs,
    resolve_url, resume_state, search_models,
)

CONTENT = bytes((i % 251 for i in range(256 * 1024)))
SHA = hashlib.sha256(CONTENT).hexdigest()
TOKEN = "test-token-not-a-secret"


class FakeHubHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FakeHub/1"

    def log_message(self, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        cfg = self.server.cfg
        parts = urllib.parse.urlsplit(self.path)
        cfg["requests"].append({"path": parts.path,
                                "range": self.headers.get("Range"),
                                "auth": self.headers.get("Authorization")})
        if parts.path == "/api/models":
            self._json([
                {"id": "org/gguf-model", "likes": 10, "downloads": 5000,
                 "pipeline_tag": "text-generation",
                 "siblings": [{"rfilename": "model.gguf"},
                              {"rfilename": "config.json"}]},
                {"id": "org/other", "likes": 1, "downloads": 5},
            ])
        elif parts.path == "/api/models/org/gguf-model/revision/main":
            self._json({"sha": "abc123",
                        "siblings": [
                            {"rfilename": "model.gguf",
                             "lfs": {"oid": SHA, "size": len(CONTENT)}},
                            {"rfilename": "config.json"}]})
        elif parts.path == "/api/models/org/plain":
            self._json({"siblings": [{"rfilename": "model.gguf"}]})
        elif parts.path == "/api/models/org/gguf-model/refs":
            self._json({"branches": [{"name": "main"}, {"name": "dev"}],
                        "tags": [{"name": "v1"}]})
        elif parts.path.startswith("/api/models/"):
            self._json({"message": "nope"}, status=404)
        elif parts.path == "/r":
            host = f"localhost:{self.server.cfg['cdn_port']}"
            self.send_response(302)
            self.send_header("Location", f"http://{host}/cdn/f.bin")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif parts.path == "/cdn/f.bin":
            self._bytes(CONTENT, "cdn")
        elif "/resolve/" in parts.path:
            self._bytes(CONTENT, "resolve")
        else:
            self._json({"message": "nope"}, status=404)

    def _bytes(self, data: bytes, kind: str):
        cfg = self.server.cfg
        if cfg.get("mode") == "no-range":
            pass                                        # fall through: 200
        elif self.headers.get("Range"):
            try:
                start = int(self.headers["Range"].split("=")[1].split("-")[0])
            except (IndexError, ValueError):
                start = 0
            if start >= len(data):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range",
                             f"bytes {start}-{len(data) - 1}/{len(data)}")
            self.send_header("Content-Length", str(len(data) - start))
            self.send_header("X-Linked-ETag", cfg.get("etag", SHA))
            self.end_headers()
            self._send_body(data[start:], kind)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Linked-ETag", cfg.get("etag", SHA))
        self.end_headers()
        self._send_body(data, kind)

    def _send_body(self, data: bytes, kind: str):
        cfg = self.server.cfg
        try:
            if kind == "resolve" and cfg.get("failures_left", 0) > 0:
                cfg["failures_left"] -= 1
                half = len(data) // 2
                self.wfile.write(data[:half])
                self.wfile.flush()
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.connection.close()
                return
            if cfg.get("mode") == "stall":
                self.wfile.write(data[:32768])
                self.wfile.flush()
                while True:                      # hold open: the client must
                    threading.Event().wait(5)     # cancel or time out
                return
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture()
def hub(monkeypatch):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHubHandler)
    server.daemon_threads = True
    server.cfg = {"requests": [], "mode": "normal", "failures_left": 0,
                  "etag": SHA, "cdn_port": 0}
    cdn = http.server.ThreadingHTTPServer(("localhost", 0), FakeHubHandler)
    cdn.daemon_threads = True
    cdn.cfg = server.cfg
    server.cfg["cdn_port"] = cdn.server_address[1]
    t1 = threading.Thread(target=server.serve_forever, daemon=True)
    t2 = threading.Thread(target=cdn.serve_forever, daemon=True)
    t1.start()
    t2.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(hf_hub, "HUB_ORIGIN", base)
    monkeypatch.setattr(hf_hub, "API", base + "/api")
    yield server
    server.shutdown()
    cdn.shutdown()


@pytest.fixture()
def workdir(tmp_path):
    out = tmp_path / "dl"
    out.mkdir()
    return out


# ── Search + listing ────────────────────────────────────────────────
def test_search_parses_hits_and_keeps_siblingless_entries(hub):
    hits = search_models("gguf", token=TOKEN)
    assert [h["id"] for h in hits] == ["org/gguf-model", "org/other"]
    assert hits[0]["siblings"] == ["model.gguf", "config.json"]
    assert hits[1]["siblings"] == []


def test_filter_files_by_kind():
    from silk.functions.hf_hub import HubFile
    files = [HubFile("m.gguf", 3), HubFile("m.safetensors", 5),
             HubFile("config.json", 1)]
    assert [f.name for f in filter_files(files, "gguf")] == ["m.gguf"]
    assert [f.name for f in filter_files(files, "safetensors")] == ["m.safetensors"]
    assert len(filter_files(files, "any")) == 3


def test_repo_files_reads_blobs_hashes_and_pins_revision(hub):
    files, pinned = repo_files("org/gguf-model", token=TOKEN)
    assert pinned == "abc123"
    gguf = next(f for f in files if f.name == "model.gguf")
    assert (gguf.size, gguf.sha256) == (len(CONTENT), SHA)
    config = next(f for f in files if f.name == "config.json")
    assert (config.size, config.sha256) == (None, None)


def test_repo_files_falls_back_when_blobs_are_missing(hub):
    files, pinned = repo_files("org/plain", token=TOKEN)
    assert pinned == "main" and [f.name for f in files] == ["model.gguf"]


def test_repo_refs_names_branches_tags(hub):
    refs = repo_refs("org/gguf-model", token=TOKEN)
    assert refs == {"branches": ["main", "dev"], "tags": ["v1"]}


def test_repo_refs_empty_when_missing(hub):
    assert repo_refs("org/nope", token=TOKEN) == {"branches": [],
                                                   "tags": []}


# ── Static guard: every self-call resolves (spawn-class safety) ──
def test_no_dangling_self_calls():
    """An edit that drops/renames a method while callers remain is a plain
    AttributeError at spawn -- invisible to ruff and headless tests, which
    never instantiate the node. This parses the node source and requires
    every ``self.x()`` inside each class body to be a method of that
    class or a known inherited/framework call."""
    import ast as _ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "nodes" / "hf_download.py")
    tree = _ast.parse(src.read_text())
    allow = {
        "SilkHFDownloadNode": {
            # ThreadedManualNode / BaseControlNode API.
            "add_input", "add_output", "cancel_compute", "emit_stream",
            "report_progress", "set_content_widget",
        },
        "_SharedCombo": set(),
        "_SharedEditCombo": {"setEditable", "setInsertPolicy"},  # QWidget
    }
    checked = set()
    for cls in [n for n in _ast.walk(tree)
                 if isinstance(n, _ast.ClassDef)]:
        if cls.name not in allow:
            continue
        checked.add(cls.name)
        defined = {n.name for n in cls.body
                   if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
        called = set()
        for n in _ast.walk(cls):
            if (isinstance(n, _ast.Call)
                    and isinstance(n.func, _ast.Attribute)
                    and isinstance(n.func.value, _ast.Name)
                    and n.func.value.id == "self"):
                called.add(n.func.attr)
        dangling = called - defined - allow[cls.name]
        assert not dangling, f"{cls.name}: dangling self-calls {dangling}"
    assert checked == set(allow), f"guard missed classes: {checked}"


def test_missing_repo_is_not_found_not_network(hub):
    with pytest.raises(HubError) as exc:
        repo_files("org/nope", token=TOKEN)
    assert exc.value.kind == "not_found"


def test_resolve_url_shape():
    assert resolve_url("org/m", "main", "f.gguf").endswith(
        "/org/m/resolve/main/f.gguf")


# ── Download ────────────────────────────────────────────────────────
def test_full_download_verifies_and_cleans_up(hub, workdir):
    dest = str(workdir / "model.gguf")
    url = resolve_url("org/gguf-model", "main", "model.gguf")
    out = download_file(url, dest, expected_size=len(CONTENT),
                        sha256=SHA, token=TOKEN)
    assert out == {"path": dest, "bytes": len(CONTENT),
                   "resumed": False, "sha256_verified": True}
    with open(dest, "rb") as fh:
        assert fh.read() == CONTENT
    assert not os.path.exists(dest + ".partial")
    assert not os.path.exists(dest + ".resume.json")


def test_interrupted_download_resumes_where_it_left_off(hub, workdir):
    dest = str(workdir / "model.gguf")
    url = resolve_url("org/gguf-model", "main", "model.gguf")
    half = len(CONTENT) // 2
    with open(dest + ".partial", "wb") as fh:
        fh.write(CONTENT[:half])
    with open(dest + ".resume.json", "w") as fh:
        json.dump({"url": url, "total": len(CONTENT), "sha256": SHA}, fh)
    out = download_file(url, dest, token=TOKEN)
    assert out["resumed"] is True and out["sha256_verified"] is True
    ranges = [r["range"] for r in hub.cfg["requests"]
              if "/resolve/" in r["path"]]
    assert ranges and ranges[0] == f"bytes={half}-"
    with open(dest, "rb") as fh:
        assert fh.read() == CONTENT


def test_dropped_connection_retries_with_resume(hub, workdir):
    hub.cfg["failures_left"] = 1
    dest = str(workdir / "model.gguf")
    url = resolve_url("org/gguf-model", "main", "model.gguf")
    out = download_file(url, dest, expected_size=len(CONTENT),
                        sha256=SHA, token=TOKEN)
    assert out["sha256_verified"] is True
    with open(dest, "rb") as fh:
        assert fh.read() == CONTENT
    ranges = [r["range"] for r in hub.cfg["requests"]
              if "/resolve/" in r["path"]]
    assert len(ranges) == 2 and ranges[1] is not None
    assert ranges[1].startswith("bytes=") and not ranges[1].endswith("0-")


def test_server_that_ignores_range_restarts_from_zero(hub, workdir):
    hub.cfg["mode"] = "no-range"
    dest = str(workdir / "model.gguf")
    url = resolve_url("org/gguf-model", "main", "model.gguf")
    with open(dest + ".partial", "wb") as fh:
        fh.write(CONTENT[:100])
    out = download_file(url, dest, token=TOKEN)
    assert out["resumed"] is False
    with open(dest, "rb") as fh:
        assert fh.read() == CONTENT


def test_stale_sidecar_starts_fresh(hub, workdir):
    dest = str(workdir / "model.gguf")
    url = resolve_url("org/gguf-model", "main", "model.gguf")
    with open(dest + ".partial", "wb") as fh:
        fh.write(CONTENT[:100])
    with open(dest + ".resume.json", "w") as fh:
        json.dump({"url": "http://other/rev", "total": len(CONTENT)}, fh)
    download_file(url, dest, token=TOKEN)
    first = next(r for r in hub.cfg["requests"] if "/resolve/" in r["path"])
    assert first["range"] is None
    with open(dest, "rb") as fh:
        assert fh.read() == CONTENT


def test_hash_mismatch_discards_everything(hub, workdir):
    hub.cfg["etag"] = "0" * 64
    dest = str(workdir / "model.gguf")
    with pytest.raises(HubError) as exc:
        download_file(resolve_url("org/gguf-model", "main", "model.gguf"),
                      dest, token=TOKEN)
    assert exc.value.kind == "hash"
    assert not os.path.exists(dest) and not os.path.exists(dest + ".partial")


def test_cancel_keeps_a_resumable_partial(hub, workdir, monkeypatch):
    monkeypatch.setattr(hf_hub, "TIMEOUT_S", 3.0)
    hub.cfg["mode"] = "stall"               # headers + 32 KiB, then silence
    dest = str(workdir / "model.gguf")
    url = resolve_url("org/gguf-model", "main", "model.gguf")
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()   # cancel lands mid-stall

    with pytest.raises(HubError) as exc:
        download_file(url, dest, token=TOKEN, is_cancelled=stop.is_set)
    assert exc.value.kind == "cancelled"
    state = resume_state(dest, url)
    assert state["resumable"] is True and state["downloaded"] > 0


def test_with_companions_adds_index_for_shards_config_on_request():
    from silk.functions.hf_hub import HubFile, with_companions
    shards = [HubFile("m-00001-of-00002.safetensors", 10),
              HubFile("m-00002-of-00002.safetensors", 10)]
    index = HubFile("m.safetensors.index.json", 1)
    config = HubFile("config.json", 1)
    tok = HubFile("tokenizer.json", 1)
    gguf = HubFile("model.gguf", 100)
    everything = shards + [index, config, tok, gguf]
    out = with_companions(everything, list(shards))
    assert [f.name for f in out] == [s.name for s in shards] + \
        ["m.safetensors.index.json"]
    out = with_companions(everything, list(shards), include_config=True)
    assert [f.name for f in out] == [s.name for s in shards] + \
        ["m.safetensors.index.json", "config.json", "tokenizer.json"]
    out = with_companions(everything, [gguf])
    assert [f.name for f in out] == ["model.gguf"], \
        "a lone GGUF drags in neither index nor config"
    out = with_companions(everything, [gguf], include_config=True)
    assert [f.name for f in out] == \
        ["model.gguf", "config.json", "tokenizer.json"]


def test_token_stays_on_the_hub_host(hub, workdir):
    dest = str(workdir / "redir.bin")
    download_file(f"{hf_hub.HUB_ORIGIN}/r", dest, token=TOKEN)
    seen = {r["path"]: r["auth"] for r in hub.cfg["requests"]}
    assert seen["/r"] == f"Bearer {TOKEN}"
    assert seen["/cdn/f.bin"] is None


# ── Layout handoff: downloads land where the browser looks ─────────
def test_completed_hub_download_is_visible_to_the_browser(hub, tmp_path):
    from silk.functions.hf_cache import (
        hub_repo_dir, link_blob, scan_repos, write_ref_main,
    )
    dest = str(tmp_path / "model.gguf")
    download_file(resolve_url("org/gguf-model", "main", "model.gguf"), dest,
                  sha256=SHA, token=TOKEN)
    repo_dir = hub_repo_dir(str(tmp_path / "hub"), "org/gguf-model")
    snap = os.path.join(repo_dir, "snapshots", "abc123")
    os.makedirs(snap, exist_ok=True)
    final = os.path.join(snap, "model.gguf")
    os.replace(dest, final)
    write_ref_main(repo_dir, "abc123")
    assert link_blob(repo_dir, SHA, final) is True
    repos = scan_repos(str(tmp_path / "hub"))
    assert [r.repo_id for r in repos] == ["org/gguf-model"]
    assert repos[0].via_ref is True and repos[0].files[0].size == len(CONTENT)


def test_default_cache_dir_follows_hf_precedence(monkeypatch, tmp_path):
    from silk.functions.hf_cache import find_cache_dir
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    custom = tmp_path / "hub"
    custom.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(custom))
    path, source = find_cache_dir()
    assert path == str(custom) and source == "env:HUGGINGFACE_HUB_CACHE"
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE")
    path, source = find_cache_dir("  ")
    assert path.endswith(os.path.join(".cache", "huggingface", "hub"))
    assert source == "default (missing)"


# ── Qt wiring (Weave env; imports local so collection works anywhere) ──
@pytest.fixture()
def node():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from silk.nodes.hf_download import SilkHFDownloadNode
    return SilkHFDownloadNode()


def _ports(node_ports):
    return {p.name: p.datatype for p in node_ports}


def test_file_filter_narrows_without_reselecting(node):
    from silk.functions.hf_hub import HubFile
    node._files = [HubFile("model.gguf", 100),
                   HubFile("model.safetensors", 90),
                   HubFile("config.json", 1), HubFile("tokenizer.json", 2)]
    node._pinned_rev = "main"
    node._populate_file_combo()
    assert node._combo_file.count() == 4
    node._combo_file.setCurrentIndex(node._combo_file.findData("model.gguf"))
    node._file_filter.setText("json")
    assert node._combo_file.count() == 3  # config, tokenizer + kept pick
    assert node._combo_file.currentData() == "model.gguf"
    assert "showing 3 of 4" in node._sync_status
    node._file_filter.setText("")
    assert node._combo_file.count() == 4
    assert node._combo_file.currentData() == "model.gguf"


def test_mirror_write_updates_folder_enablement(node):
    # Mirror edits arrive via bindings with widget signals held -- the
    # folder enablement must still follow (on_ui_change, not cascades).
    assert node._target_folder.isEnabled() is False  # hub default
    node._widget_core.apply_port_value("target", "custom")
    assert node._target_folder.isEnabled() is True
    node._widget_core.apply_port_value("target", "hub")
    assert node._target_folder.isEnabled() is False


def test_results_combo_is_registered_transient_and_shared(node):
    from silk.nodes.hf_download import _SharedCombo
    assert isinstance(node._combo_results, _SharedCombo)
    assert isinstance(node._combo_file, _SharedCombo)
    assert _SharedCombo.__mirror__.shared_model is True
    assert node._combo_results.model() is not node._combo_file.model()
    binding = node._widget_core._bindings["results"]
    assert binding.policy.persist is False
    assert binding.policy.restore is False
    assert binding.policy.triggers_eval is True


def test_mirror_result_pick_fills_repo(node, monkeypatch):
    calls = []
    monkeypatch.setattr(node, "_fetch_files", lambda: calls.append(1))
    hits = [{"id": "a/b", "downloads": 3, "likes": 4},
            {"id": "c/d", "downloads": 5, "likes": 6}]
    node._on_search_done((node._seq, hits))
    assert node._combo_results.count() == 2
    assert node._repo.text() == "a/b"
    assert calls == [1]
    # Mirror picks arrive as binding writes (widget signals held).
    node._widget_core.apply_port_value(
        "results", node._combo_results.itemText(1), user_edit=False)
    assert node._repo.text() == "c/d"
    assert calls == [1, 1]


def test_repo_edit_schedules_debounced_fetch(node):
    assert node._fetch_timer.isActive() is False
    node._widget_core.apply_port_value("repo", "x/y", user_edit=False)
    assert node._fetch_timer.isActive() is True
    node._fetch_timer.stop()


def test_fetch_debounce_rearms_while_computing(node):
    node._is_computing = True
    try:
        node._on_fetch_debounce()
        assert node._fetch_timer.isActive() is True
    finally:
        node._is_computing = False
        node._fetch_timer.stop()


def test_revision_combo_is_editable_shared_and_preserves_pins(node):
    from silk.functions.hf_hub import HubFile
    from silk.nodes.hf_download import _SharedEditCombo
    assert isinstance(node._revision, _SharedEditCombo)
    assert node._revision.isEditable() is True
    assert _SharedEditCombo.__mirror__.shared_model is True
    node._widget_core.apply_port_value("repo", "org/gguf-model",
                                       user_edit=False)
    node._revision.setEditText("aabbccddeeff00112233445566778899aabbccdd")
    node._on_files_done((node._seq, ("org/gguf-model",
                                     [HubFile("model.gguf", 10)], "sha1",
                                     {"branches": ["main"], "tags": []},
                                     "aabbccddeeff00112233445566778899aabbccdd")))
    assert "main" in [node._revision.itemText(i)
                        for i in range(node._revision.count())]
    assert node._revision.currentText().startswith("aabbccddee")


def test_typo_revision_resolves_to_the_branches_default(node):
    from silk.functions.hf_hub import HubFile
    node._widget_core.apply_port_value("repo", "org/gguf-model",
                                       user_edit=False)
    node._revision.setEditText("mian")
    node._on_files_done((node._seq, ("org/gguf-model",
                                     [HubFile("model.gguf", 10)], "sha1",
                                     {"branches": ["master"], "tags": []},
                                     "master")))
    assert node._revision.currentText() == "master"


def test_fetch_clears_the_stale_text_filter(node, monkeypatch):
    import threading as threading_mod
    started = []

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            started.append(True)
    monkeypatch.setattr(threading_mod, "Thread", DummyThread)
    node._file_filter.setText("gguf")
    node._widget_core.apply_port_value("repo", "org/other",
                                       user_edit=False)
    node._fetch_files()
    assert node._file_filter.text() == ""
    assert started == [True]


def test_repopulate_without_changes_emits_no_model_signals(node):
    # Mirror-loop killer: an identical listing must not touch the model.
    from silk.functions.hf_hub import HubFile
    node._widget_core.apply_port_value("repo", "org/gguf-model",
                                       user_edit=False)
    node._on_files_done((node._seq, ("org/gguf-model",
                                     [HubFile("model.gguf", 10)], "sha1",
                                     {"branches": ["main"], "tags": []},
                                     "main")))
    resets = []
    node._combo_file.model().modelReset.connect(lambda: resets.append(1))
    node._populate_file_combo()
    node._populate_revision_combo({"branches": ["main"], "tags": []},
                                  "main")
    assert resets == []


def test_admit_only_publishes_on_change(node, monkeypatch):
    calls = []
    monkeypatch.setattr(node._widget_core, "apply_port_value",
                        lambda *a, **k: calls.append(a))
    node._admit("file", "a.gguf")
    node._admit("file", "a.gguf")
    node._admit("file", "b.gguf")
    assert [c[1] for c in calls] == ["a.gguf", "b.gguf"]


def test_status_push_only_on_change(node, monkeypatch):
    pushes = []
    monkeypatch.setattr(node._widget_core, "push_display",
                        lambda *a: pushes.append(a))
    node._set_status("hello")
    node._set_status("hello")
    assert pushes == [("status", "hello")]


def test_search_button_is_registered_and_forwardable(node):
    from weave.widgets.sync_button import SyncButton
    binding = node._widget_core._bindings.get("btn_search")
    assert binding is not None and binding.widget is node._btn_search
    assert SyncButton.__mirror__.action_signal == "clicked"


def test_mirror_kind_pick_filters_the_file_list(node):
    # Mirror picks arrive as binding writes (data values, not labels).
    from silk.functions.hf_hub import HubFile
    node._widget_core.apply_port_value("repo", "org/mixed",
                                       user_edit=False)
    node._on_files_done((node._seq, ("org/mixed",
                                     [HubFile("m.gguf", 90),
                                      HubFile("m.safetensors", 80),
                                      HubFile("config.json", 1)], "sha1",
                                     {"branches": ["main"], "tags": []},
                                     "main")))
    assert node._combo_file.count() == 3
    node._widget_core.apply_port_value("kind", "gguf", user_edit=False)
    assert node._combo_file.count() == 1
    assert node._combo_file.currentData() == "m.gguf"
    node._widget_core.apply_port_value("kind", "any", user_edit=False)
    assert node._combo_file.count() == 3


def test_mirror_file_pick_resolves_by_data(node):
    from silk.functions.hf_hub import HubFile
    node._widget_core.apply_port_value("repo", "org/mixed",
                                       user_edit=False)
    node._on_files_done((node._seq, ("org/mixed",
                                     [HubFile("m.gguf", 90),
                                      HubFile("m.safetensors", 80)], "sha1",
                                     {"branches": ["main"], "tags": []},
                                     "main")))
    node._widget_core.apply_port_value("file", "m.safetensors",
                                       user_edit=False)
    assert node._combo_file.currentData() == "m.safetensors"


def test_node_ports_and_buttons(node):
    assert _ports(node.inputs) == {"repo": "string", "revision": "string",
                                   "file": "string", "target": "string",
                                   "target_folder": "dirpath"}
    assert _ports(node.outputs) == {"file_path": "filepath",
                                    "download_info": "dict"}
    assert node._btn_download.text() == "Download"
    assert node._btn_resume.text() == "Resume"
    assert node._btn_cancel.text() == "Cancel"
    assert node._btn_open.text() == "Open repo"
    assert node._progress.maximum() == 1000
    assert node._btn_resume.isEnabled() is False
    assert node._btn_cancel.isEnabled() is False
    assert node._btn_open.isEnabled() is False


def test_folder_history_roundtrips_through_state(node, tmp_path):
    from weave.widgets.path_history_picker import PathHistoryPicker
    assert isinstance(node._target_folder, PathHistoryPicker)
    node._target_folder.set_history([str(tmp_path / "hub-a"),
                                     str(tmp_path / "hub-b")])
    saved = node.get_state()
    assert saved["target_history"] == [str(tmp_path / "hub-a"),
                                        str(tmp_path / "hub-b")]
    node._target_folder.set_history([])
    node.restore_state(saved)
    assert node._target_folder.history() == [str(tmp_path / "hub-a"),
                                             str(tmp_path / "hub-b")]
    node._repo.setText("org/gguf-model")
    node._refresh_buttons()
    assert node._btn_open.isEnabled() is True


def test_compute_without_a_repo_needs_no_network(node):
    out = node.compute({})
    assert out["file_path"] is None
    assert "repo id" in node._sync_status
