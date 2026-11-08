# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

W8 seam: the knowledge-graph node rides okfgraph 0.12's loader surface.

Offline (a stdlib stub stands in for ``okf serve``, the same idiom the
okfgraph repo's own seam tests use). Three claims.

**The spec tells the server what to run.** The external-embedder rows and
the re-embed escape ride ``OKFGRAPH_*`` env on launch (no CLI spelling to
maintain), the vector width rides ``--embedding-dim``, and the crate
default inherits nothing new.

**The ledger's inverse lane is an ``Embedder``.** An OkfGraph handle is
accepted by ``embedder_for`` and answers vectors from ``POST /ops/embed``
-- the same pinned engine the graph's search index lives in, one model
host instead of two. The optional-half rule is inherited (the first state
that fails disables it, once, out loud).

**The re-embed flow is the node's only relaunch.** ``OkfGraph.repinned``
boots pin-exempt once, re-embeds, then re-opens pin-clean; the escape
never outlives the server that did the re-pin.
"""
from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from silk.functions.embeddings import Embedder, embedder_for
from silk.functions.okf_graph import (
    LADDER_DIMS, OkfClient, OkfError, OkfEmbedder, OkfGraph, OkfSpec,
    serve_file_for,
)


class _Stub(BaseHTTPRequestHandler):
    """Health, /ops/embed (vectors keyed by text), /ops/reembed (the 0.12
    report shape), /ops/pin_info (the pin view)."""
    vectors: dict[str, list[float]] = {}

    def _reply(self, data: dict):
        body = json.dumps({"ok": True, "op": self.path, "data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._reply({"url": "http://stub", "pid": 1, "db_path": "stub.db",
                         "ops": ["embed", "reembed", "pin_info"]})
            return
        self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path == "/shutdown":
            self._reply({})
        elif self.path.endswith("/ops/embed"):
            texts = body.get("texts") or []
            dims = []
            for t in texts:
                if t not in _Stub.vectors:
                    _Stub.vectors[t] = [float(len(t) % 10), 1.0, 0.5]
                dims.append(_Stub.vectors[t])
            self._reply({"vectors": dims, "dim": 3, "model_id": "stub/model-a",
                         "task": "Document", "renormalised": False,
                         "honours_dimensions": True, "counter": "heuristic"})
        elif self.path.endswith("/ops/reembed"):
            self._reply({"reembedded": {"concepts": 2, "chunks": 1},
                         "seconds": 1.2, "precision": None,
                         "pins_after": {"pinned": {"model_id": "stub/model-b",
                                                   "embedding_dim": 512},
                                        "mismatch": []}})
        elif self.path.endswith("/ops/pin_info"):
            self._reply({"pinned": {"model_id": "stub/model-b", "precision": None,
                                    "embedding_dim": 512},
                         "configured": {"model_id": "stub/model-b"},
                         "mismatch": [], "empty": False,
                         "concepts": 2, "chunks": 1})
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


@pytest.fixture()
def stub_url(tmp_path):
    _Stub.vectors = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    info = serve_file_for(str(tmp_path / "stub.db"))
    info.parent.mkdir(exist_ok=True)
    info.write_text(json.dumps({
        "url": f"http://127.0.0.1:{port}", "pid": 0, "started_at": 0,
        "shutdown_token": "t",
    }), encoding="utf-8")
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_spec_translates_the_ext_rows_and_width():
    crate = OkfSpec(db_path="kb.db")
    assert crate.launch_env() == {}            # the default lane: no env
    # device is spelled even at its default; the width only when it moves
    assert crate.settings_args()[:2] == ["--db-path", "kb.db"]
    assert "--embedding-dim" not in crate.settings_args()
    assert OkfSpec(db_path="k", embedding_dim=256).settings_args()[-2:] \
        == ["--embedding-dim", "256"]
    ext = OkfSpec(db_path="kb.db", backend="external",
                  embed_family="nomic-v1.5",
                  endpoint_url="http://127.0.0.1:8931",
                  embedding_dim=256)
    env = ext.launch_env()
    assert env["OKFGRAPH_BACKEND"] == "external"
    assert env["OKFGRAPH_ENDPOINT_URL"] == "http://127.0.0.1:8931"
    assert env["OKFGRAPH_EMBED_FAMILY"] == "nomic-v1.5"
    assert "OKFGRAPH_ALLOW_REPIN" not in env
    args = ext.settings_args()
    assert "--embedding-dim" in args and "256" in args
    assert "--embedding-dim" not in OkfSpec(db_path="k.db").settings_args()
    repin = replace(ext, allow_repin=True).launch_env()
    assert repin["OKFGRAPH_ALLOW_REPIN"] == "1"
    assert 768 in LADDER_DIMS


def test_embedder_for_accepts_a_graph_handle(stub_url):
    spec = OkfSpec(db_path="stub.db", model_id="stub/model-a")
    graph = OkfGraph(spec, stub_url)           # attached: the stub owns it
    emb = embedder_for(graph)
    assert isinstance(emb, OkfEmbedder)
    assert isinstance(emb, Embedder)
    vec = emb.embed("ledger line one")
    assert len(vec) == 3 and all(isinstance(x, float) for x in vec)
    assert emb.dim == 3                        # the width rides the first vector
    assert emb.model_id == "stub/model-b"      # straight from the pin view
    assert graph.owned is False                # attached: close() is a no-op


def test_okfembedder_disables_itself_when_the_op_fails(stub_url, monkeypatch):
    spec = OkfSpec(db_path="stub.db", model_id="stub/model-a")
    graph = OkfGraph(spec, stub_url)

    def broken(op, timeout=0.0, **kw):
        raise OkfError("BAD_VALUE", "no engine mounted", remedy="boot first")
    monkeypatch.setattr(graph.client, "call", broken)
    emb = embedder_for(graph)
    assert emb.embed("anything") is None       # failure = silence, once
    assert emb.enabled is False
    assert emb.embed("again") is None          # stays disabled


def test_ops_embed_wire_shape_and_reembed_report(stub_url):
    client = OkfClient(stub_url)
    out = client.call("embed", texts=["alpha", "beta"], task="Document")
    assert out["dim"] == 3 and len(out["vectors"]) == 2
    assert out["counter"] == "heuristic"
    report = client.call("reembed", dry_run=False)
    assert report["reembedded"] == {"concepts": 2, "chunks": 1}
    pinned = report["pins_after"]["pinned"]
    assert pinned["model_id"] == "stub/model-b"
    sys = client.call("pin_info")
    assert sys["pinned"]["embedding_dim"] == 512
    with pytest.raises(OkfError):
        client.call("nonexistent_op")
