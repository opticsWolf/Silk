# -*- coding: utf-8 -*-
"""Turns and runs as memory, and the ``recall`` tool over them (§17, D66).

Two claims are under test. The first is that compaction stops being
destructive: a compacted round is superseded, not deleted, so *what did
the model actually see at round 7* stays answerable after the context was
squeezed -- which is the thing history-on-the-node could never do (D24/D25,
I11). The second is that ``recall`` reaches past the run it is called in;
memory that only spans the current prompt is called scrollback.
"""
from __future__ import annotations

import hashlib
import math

import pytest

from silk.functions import ledger as ledger_mod
from silk.functions.embeddings import (
    Embedder, embedder_for, model_name,
)
from silk.functions.ledger import (
    BACKEND_ENV, BACKEND_LEDGER, DISTRIBUTION, HistoryLedger,
    KIND_RUN, LedgerRegistry, TaskLedger, _related, _run_cid,
    history_path, ledger_path, open_history, open_task_store,
    requested_backend,
)

pytestmark = pytest.mark.skipif(
    not ledger_mod.available(),
    reason=f"the {DISTRIBUTION} extra is not installed",
)


@pytest.fixture
def registry():
    reg = LedgerRegistry()
    yield reg
    reg.close_all()


@pytest.fixture
def history(registry, tmp_path):
    return HistoryLedger(tmp_path, registry=registry)


@pytest.fixture
def conversation(history):
    history.start_run("r1", agent="researcher", session="s1",
                      goal="ship the parser")
    history.record_turn("r1", index=0, role="user",
                        text="how should the lexer handle nested quotes?")
    history.record_turn("r1", index=1, role="assistant",
                        text="the lexer keeps a quote stack",
                        tools=["read_file"], files=["src/lex.py"])
    history.record_turn("r1", index=2, role="assistant",
                        text="the parser consumes those tokens",
                        tools=["read_file", "write_file"],
                        files=["src/parse.py"])
    return history


# ── a run is a graph, not a log line ─────────────────────────────────────


def test_a_run_remembers_its_turns_in_order(conversation):
    turns = conversation.turns("r1")
    assert [(t["index"], t["role"]) for t in turns] == [
        (0, "user"), (1, "assistant"), (2, "assistant")]
    assert conversation.run("r1")["turns"] == 3


def test_what_a_run_touched_is_a_traversal(conversation):
    assert conversation.touched("r1") == ["src/lex.py", "src/parse.py"]
    assert conversation.used("r1") == ["read_file", "write_file"]


def test_a_tool_used_twice_is_still_one_fact(conversation):
    """Re-asserting an open edge is refused, and rightly (Doctrine III)."""
    conversation.record_turn("r1", index=3, role="assistant", text="again",
                             tools=["read_file"], files=["src/lex.py"])
    assert conversation.used("r1") == ["read_file", "write_file"], (
        "using a tool a second time is the same edge, not a second one"
    )


def test_the_run_and_its_agent_are_linked(conversation, registry, tmp_path):
    db = conversation._db()
    from silk.functions.ledger import EDGE_BY_AGENT, _agent_id

    agents = _related(db, _run_cid("r1"), EDGE_BY_AGENT)
    assert [node.id for node in agents] == [_agent_id("researcher")], (
        "D60's identity plumbing and the ledger's keys are the same thing"
    )


def test_a_turn_without_a_start_run_still_lands(history):
    """A crash before start_run must not lose the turn that followed."""
    history.record_turn("orphan", index=0, role="user", text="hello")
    assert [t["index"] for t in history.turns("orphan")] == [0]
    assert history.run("orphan")["kind"] == KIND_RUN


def test_a_finished_run_supersedes_its_status(conversation):
    conversation.finish_run("r1", status="finished", summary="shipped")
    run = conversation.run("r1")
    assert (run["status"], run["summary"]) == ("finished", "shipped")
    assert run["started_at"], "and the start is still there to read"


# ── compaction: supersession, not deletion (D24/D25, I11) ────────────────


def test_compaction_hides_rounds_without_destroying_them(conversation):
    conversation.compacted("r1", dropped=[0, 1], kept=1,
                           rationale="context pressure")

    assert [t["index"] for t in conversation.turns(
        "r1", include_superseded=False)] == [2], (
        "what the model sees now is the un-compacted tail"
    )
    assert [t["index"] for t in conversation.turns("r1")] == [0, 1, 2], (
        "but the dropped rounds stay addressable -- which is the whole "
        "reason history moved off the node (§17)"
    )
    dropped = conversation.turns("r1")[0]
    assert "nested quotes" in dropped["text"], (
        "and readable, not just countable"
    )


def test_the_compaction_itself_is_on_the_record(conversation):
    conversation.compacted("r1", dropped=[0], kept=2, rationale="too long")
    events = conversation.compactions("r1")
    assert len(events) == 1
    assert (events[0]["dropped"], events[0]["rationale"]) == ([0], "too long")


def test_two_compactions_are_two_events(conversation):
    conversation.compacted("r1", dropped=[0], kept=2, rationale="first")
    conversation.compacted("r1", dropped=[1], kept=1, rationale="second")
    assert [e["rationale"] for e in conversation.compactions("r1")] == [
        "first", "second"], "the second does not overwrite the first"
    assert [t["index"] for t in conversation.turns(
        "r1", include_superseded=False)] == [2]


# ── recall ───────────────────────────────────────────────────────────────


def test_recall_finds_a_turn_by_its_words(conversation):
    hits = conversation.recall("lexer")
    assert hits and hits[0]["run_id"] == "r1"
    assert "lexer" in hits[0]["text"], "the hit carries the text, not just an id"
    assert {h["index"] for h in hits} == {0, 1}, (
        "both turns that said 'lexer' come back; ranking decides the "
        "order, recall decides the set"
    )


def test_recall_reaches_across_runs(conversation):
    """Memory that stops at the current run is scrollback, not memory."""
    conversation.start_run("r2", agent="builder", goal="write the emitter")
    conversation.record_turn("r2", index=0, role="assistant",
                             text="the emitter walks the parser's tree")
    runs = {hit["run_id"] for hit in conversation.recall("emitter", top_k=10)}
    assert runs == {"r2"}
    assert {h["run_id"] for h in conversation.recall("parser", top_k=10)} == {
        "r1", "r2"}, "one query, both runs"


def test_recall_finds_a_compacted_turn(conversation):
    conversation.compacted("r1", dropped=[0, 1], kept=1, rationale="pressure")
    assert conversation.recall("nested quotes"), (
        "the point of the ledger: the agent can look up what it was made "
        "to forget"
    )


def test_recall_can_search_runs_instead_of_turns(conversation):
    hits = conversation.recall("ship the parser", kinds=(KIND_RUN,))
    assert [h["id"] for h in hits] == [_run_cid("r1")]


def test_an_empty_query_searches_nothing(conversation):
    assert conversation.recall("   ") == []


def test_recall_respects_top_k(conversation):
    assert len(conversation.recall("the", top_k=1)) <= 1


# ── placement and the seam ───────────────────────────────────────────────


def test_history_and_tasks_are_separate_files(tmp_path):
    assert history_path(tmp_path) != ledger_path(tmp_path), (
        "one Write Actor each: a chatty turn writer must not queue behind "
        "a plan read (and a graph may keep its plan while dropping memory)"
    )
    assert history_path(tmp_path).parent == tmp_path


def test_the_backend_is_chosen_by_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    assert requested_backend() == "sqlite", (
        "plan discovery is still file-shaped (T4, D58), so the flip is a "
        "separate change; the ledger is opt-in until then"
    )
    assert type(open_task_store(tmp_path)).__name__ == "SqliteTaskStore"

    monkeypatch.setenv(BACKEND_ENV, BACKEND_LEDGER)
    assert isinstance(open_task_store(tmp_path), TaskLedger)


def test_a_missing_extra_falls_back_loudly(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(ledger_mod, "_macrame", None)
    monkeypatch.setattr(ledger_mod, "_IMPORT_ERROR", None)
    monkeypatch.setenv(BACKEND_ENV, BACKEND_LEDGER)

    with caplog.at_level("WARNING"):
        store = open_task_store(tmp_path)
    assert type(store).__name__ == "SqliteTaskStore"
    assert any(DISTRIBUTION in record.getMessage() for record in caplog.records), (
        "D66: the graph degrades to today's behaviour loudly -- one line, "
        "never silently"
    )
    assert open_history(tmp_path) is None, (
        "and history has nothing to degrade to, so it says no rather than "
        "answering 'nothing happened'"
    )


def test_the_tools_never_import_macrame():
    """D66's seam: only functions/ledger.py knows the library exists."""
    from pathlib import Path

    from silk.functions.tools import recall_tool

    source = Path(recall_tool.__file__).read_text(encoding="utf-8")
    assert "import macrame" not in source
    assert "from ..ledger import" in source


# ── where memory lives (§22 q7) ──────────────────────────────────────────


class _Sandbox:
    """The two attributes recall reads off a FileToolSandbox."""

    def __init__(self, root, allowed):
        self.root_dir = root
        self.allowed_paths = list(allowed)


def _box():
    from silk.functions.tool_box import ToolBox

    return ToolBox(None, {"agent_id": "ag"})


def _remember(root, registry, run, text):
    ledger = HistoryLedger(root, registry=registry)
    ledger.start_run(run, agent="a", session="s", goal=text)
    ledger.record_turn(run, index=0, role="assistant", text=text)
    return ledger


def _call_recall(box, **kw):
    import asyncio
    import json
    from types import SimpleNamespace

    call = SimpleNamespace(id="c1", function=SimpleNamespace(
        name="recall", arguments=json.dumps(kw)))
    out = asyncio.run(box.execute_tool_calls_async([call]))
    return json.loads(out[0]["content"])


def test_memory_reaches_every_root_the_box_was_given(registry, tmp_path):
    """§22 q7: no per-user store -- memory follows the sandbox roots.

    Crossing projects is a wire the graph author drew (a second folder on
    the ToolBox), never a default, and every hit says which root it came
    from so one project's turn can never be read as another's.
    """
    from silk.functions.tools.recall_tool import attach_recall_tool

    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    _remember(here, registry, "r1", "the lexer keeps a quote stack")
    _remember(there, registry, "r2", "the lexer was rewritten in rust")

    box = _box()
    attach_recall_tool(box, _Sandbox(here, [here, there]))
    body = _call_recall(box, query="lexer", top_k=10)

    assert body["ok"]
    roots = {hit["root"] for hit in body["hits"]}
    assert len(body["hits"]) == 2 and len(roots) == 2, (
        "both roots remembered something about the lexer"
    )
    assert all(hit["root"] for hit in body["hits"]), (
        "a hit with no provenance is one project's memory wearing "
        "another's name"
    )


def test_a_root_that_was_not_given_is_not_remembered(registry, tmp_path):
    from silk.functions.tools.recall_tool import attach_recall_tool

    here, elsewhere = tmp_path / "here", tmp_path / "elsewhere"
    here.mkdir()
    elsewhere.mkdir()
    _remember(here, registry, "r1", "the lexer keeps a quote stack")
    _remember(elsewhere, registry, "r2", "the lexer belongs to another client")

    box = _box()
    attach_recall_tool(box, _Sandbox(here, [here]))
    body = _call_recall(box, query="lexer", top_k=10)

    assert [hit["run_id"] for hit in body["hits"]] == ["r1"], (
        "file access is the ceiling for memory too (I6): a root this box "
        "cannot read is a root it cannot remember"
    )


def test_a_root_with_no_memory_is_not_given_one(tmp_path):
    """Opening a ledger creates a file; a folder nobody asked to remember
    anything about should stay as it was found."""
    from silk.functions.tools.recall_tool import memory_roots

    here, fresh = tmp_path / "here", tmp_path / "fresh"
    here.mkdir()
    fresh.mkdir()

    assert memory_roots(_Sandbox(here, [here, fresh]), here) == []
    assert not history_path(fresh).exists()


def test_the_working_root_is_not_read_twice(registry, tmp_path):
    from silk.functions.tools.recall_tool import memory_roots

    here = tmp_path / "here"
    here.mkdir()
    _remember(here, registry, "r1", "the lexer keeps a quote stack")

    sandbox = _Sandbox(here, [here, here / "..", here])
    assert [str(root) for root in memory_roots(sandbox, here)] == []


def test_one_unreadable_root_does_not_lose_the_rest(registry, tmp_path):
    """Another project's memory is a bonus, never a dependency."""
    from silk.functions.tools import recall_tool

    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    _remember(here, registry, "r1", "the lexer keeps a quote stack")
    _remember(there, registry, "r2", "the lexer was rewritten in rust")

    class _Broken:
        root = there

        def recall(self, *_a, **_kw):
            raise RuntimeError("that disk is gone")

    box = _box()
    recall_tool.attach_recall_tool(box, _Sandbox(here, [here]))
    box._history_ledgers_read = (_Broken(),)

    body = _call_recall(box, query="lexer", top_k=10)
    assert body["ok"] and [hit["run_id"] for hit in body["hits"]] == ["r1"]


# -- the vector half of §17 -----------------------------------------------
#
# recall shipped as FTS5 because FTS5 needs no model. With an embedder the
# same call becomes a hybrid search, and the point of the exercise is the
# query whose words are not the words that were written down.


class _WordVectors(Embedder):
    """A deterministic stand-in: one bucket per word, cosine-comparable.

    Not a language model -- it cannot know that "tokenizer" and "lexer"
    are related. What it *can* do is put a document's words in the same
    place every time, which is all these tests need to tell a vector hit
    from a keyword hit.
    """

    DIM = 16

    def __init__(self, name="wordvec", fail=False):
        super().__init__(name)
        self.fail = fail
        self.calls = 0

    def _vector(self, text):
        self.calls += 1
        if self.fail:
            raise RuntimeError("no model here")
        buckets = [0.0] * self.DIM
        for word in set(text.lower().split()):
            buckets[int(hashlib.sha256(word.encode()).hexdigest(), 16)
                    % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in buckets)) or 1.0
        return [x / norm for x in buckets]


def test_a_wired_embedder_makes_recall_hybrid(registry, tmp_path):
    history = HistoryLedger(tmp_path, registry=registry,
                            embedder=_WordVectors())
    history.start_run("r1", agent="a")
    history.record_turn("r1", index=0, role="user",
                        text="the lexer chokes on nested quotes")
    history.record_turn("r1", index=1, role="assistant",
                        text="the build cache was stale")

    hits = history.recall("lexer")
    assert hits and hits[0]["id"] == "turn:r1:0"
    assert hits[0]["via"] == "both", (
        "the words and the meaning agreed, and the hit says so -- a hit "
        "only one arm found is a different kind of hit"
    )


def test_meaning_finds_what_the_words_missed(registry, tmp_path):
    """The whole reason for the vector half."""
    history = HistoryLedger(tmp_path, registry=registry,
                            embedder=_WordVectors())
    history.start_run("r1", agent="a")
    history.record_turn("r1", index=0, role="assistant",
                        text="nested quotes break the lexer")

    plain = HistoryLedger(tmp_path, registry=registry)
    assert plain.recall("plunge") == [], (
        "a word that appears nowhere finds nothing by keyword"
    )
    vector_hits = history.recall("plunge")
    assert vector_hits and vector_hits[0]["via"] == "vector", (
        "the vector arm still ranks the corpus; the hit says which arm "
        "found it so a weak match is legible as one"
    )


def test_memory_without_an_embedder_is_the_keyword_search_it_was(
        registry, tmp_path):
    history = HistoryLedger(tmp_path, registry=registry)
    history.start_run("r1", agent="a")
    history.record_turn("r1", index=0, role="user", text="the lexer again")
    hits = history.recall("lexer")
    assert hits and hits[0]["via"] == "keyword"
    assert history.embedder is None


def test_an_embedder_that_fails_costs_the_vector_and_not_the_turn(
        registry, tmp_path):
    """A turn is a fact; its vector is an index entry (§17)."""
    broken = _WordVectors(fail=True)
    history = HistoryLedger(tmp_path, registry=registry, embedder=broken)
    history.start_run("r1", agent="a")
    history.record_turn("r1", index=0, role="user",
                        text="written despite the broken embedder")

    assert not broken.enabled, "it disabled itself rather than retrying"
    hits = history.recall("broken")
    assert hits and hits[0]["via"] == "keyword", (
        "the turn was remembered and is still findable by keyword"
    )
    assert broken.calls == 1, (
        "one attempt, not one per turn -- a dead embedder must not cost "
        "a request per write for the rest of the run"
    )


def test_a_reader_that_never_wrote_can_still_search_vectors(
        registry, tmp_path):
    """Another root's memory, or this one in a later session."""
    writer = HistoryLedger(tmp_path, registry=registry,
                           embedder=_WordVectors())
    writer.start_run("r1", agent="a")
    writer.record_turn("r1", index=0, role="user", text="the lexer chokes")

    reader = HistoryLedger(tmp_path, registry=registry,
                           embedder=_WordVectors())
    hits = reader.recall("lexer")
    assert hits and hits[0]["via"] == "both", (
        "the model is registered on the read path too, or a fresh session "
        "would silently lose the index it wrote yesterday"
    )


def test_an_embedding_model_name_is_a_legal_table_name():
    """Macrame accepts [a-z][a-z0-9_]* up to 48 chars, and model files are
    not spelled like that."""
    assert model_name("Qwen3-Embedding-0.6B-Q8_0.gguf") == (
        "qwen3_embedding_0_6b_q8_0_gguf"
    )
    assert model_name("123") == "e_123", "a name must start with a letter"
    assert model_name("") == "embedding"
    assert len(model_name("x" * 100)) <= 48


def test_an_embedder_is_only_built_from_a_model_that_could_serve_one():
    class _Model:
        def create_embedding(self, text):
            return {"data": [{"embedding": [0.5, 0.5]}]}

    assert embedder_for(None) is None
    assert embedder_for({"backend": "openai"}) is None
    assert embedder_for({"backend": "gguf"}) is None, (
        "a handle with neither a model nor a pool serves nothing"
    )
    made = embedder_for({"backend": "gguf", "model": _Model()})
    assert made is not None and made.embed("hello") == [0.5, 0.5]
    assert made.dim == 2, "the width is learned, never declared"


def test_a_vector_is_read_out_of_whatever_shape_the_model_returns():
    class _Rows:
        def embed(self, text):
            return [[1.0, 0.0], [0.0, 1.0]]   # per-token rows, one embedding

    made = embedder_for({"backend": "gguf", "model": _Rows()})
    assert made.embed("x") == [1.0, 0.0]


def test_a_model_that_cannot_embed_disables_itself_quietly():
    class _ChatOnly:
        def create_embedding(self, text):
            raise RuntimeError("this server has no embedding support")

    made = embedder_for({"backend": "gguf", "model": _ChatOnly()})
    assert made.embed("x") is None, "no exception reaches the caller"
    assert not made.enabled
    assert made.embed("y") is None, "and it does not try again"


def test_every_root_is_indexed_by_the_same_model(registry, tmp_path):
    """One embedder for the box, not one per root.

    Two roots indexed by two models would be two incomparable rankings
    merged into one list, which is a worse answer than either.
    """
    from silk.functions.tools.recall_tool import attach_recall_tool

    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    _remember(here, registry, "r1", "the lexer keeps a quote stack")
    _remember(there, registry, "r2", "the lexer was rewritten in rust")

    shared = _WordVectors()
    box = _box()
    attach_recall_tool(box, _Sandbox(here, [here, there]), embedder=shared)

    assert box._history_ledger.embedder is shared
    others = box._history_ledgers_read
    assert others and all(led.embedder is shared for led in others)


def test_recall_without_an_embedder_is_unchanged(registry, tmp_path):
    """The port adds a capability; it never changes the one that was there."""
    from silk.functions.tools.recall_tool import attach_recall_tool

    root = tmp_path / "root"
    root.mkdir()
    _remember(root, registry, "r1", "the lexer keeps a quote stack")

    box = _box()
    attach_recall_tool(box, _Sandbox(root, [root]))
    assert box._history_ledger.embedder is None
    body = _call_recall(box, query="lexer")
    assert body["ok"] and body["hits"]
    assert body["hits"][0]["via"] == "keyword"
