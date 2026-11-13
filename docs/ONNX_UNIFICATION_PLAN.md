# ONNX unification plan: embroider, bobine, okfgraph, Silk

Status: plan, 2026-11-10. Branches `dev_silk` in embroider, bobine and
OKFgraph were cut from `origin/main` (embroider 0.14.1, bobine 0.8.0,
okfgraph 0.12.1) and have since merged back for the releases below.

Progress (2026-11-11): steps 1–3 of section 5 are released, each tagged
on `main` (bobine `v0.8.1`, embroider `v0.15.0`, okfgraph `v0.13.0`) and
published to PyPI (embroider and bobine also to crates.io). bobine 0.8.1
still builds against the embroider 0.14 crate; 0.15 leaves every call it
makes unchanged.

- **bobine 0.8.1 (B1) released**: the extras, CI, README, quickref, site and
  CHANGELOG all moved to ORT 1.31.1.
- **embroider 0.15 (E1–E8) released**, with these deviations from the plan:
  - `describe()` lives on the Python `Model` that `embroider.open` returns, not on each native class.
  - `space` ends in `VECTOR_CONTRACT` (`v1`), not the release, so a release that leaves vectors alone keeps every graph valid.
  - E6 stays on Pillow (okf's exact pixels) instead of the Rust `image` crate.
  - `locate_runtime` lives in Rust (`embroider::diag::locate_runtime`), runs on every open (CPU included) and works on all three platforms; the Python function wraps it. bobine's B2 can call the Rust function directly.
  - Every model class reports the precision of the weights that loaded, Laya and 2B included; a request for an unshipped tier reads `fp32`.
- **Measured:** host and in-process vectors are equal at `abs=0` across processes on CPU (open question 1, CPU half).
- **okfgraph 0.13 (O5, O4, O1, O2, O7) released**, with these deviations from the plan:
  - The space pin ignores `max_length`: stored texts are chunked below it, so a host with another ceiling serves the same vectors.
  - A session that can't name its space (a cold or older engine) warns on a space-pinned graph instead of refusing.
  - Pre-space graphs adopt a space only when their precision pin matched before the open; a graph a declared engine filled is never adopted.
  - The rerank probe also makes `rerank_max_tokens` come from the host.
  - O7 adds a `lanes()` op beside `pin_info`; `/health` carries it and `okf doctor` prints `embed_lane` / `rerank_lane` info rows.
  - `uv.lock` pins the published embroider 0.15.0 and bobine 0.8.1.
- **okfgraph 0.13.1 (setup and CLI fixes)**: skills ship in the wheel with
  `okf setup skills`; several files per `okf ingest --kind pdf`; global
  flags before the command; missing db folders created; help names the
  default paths; README install table; `pdf` extra `bobine>=0.8.1`.

Progress (2026-11-13): steps 1–4 of section 5 are released. Every library
change Silk's steps depend on is on PyPI; only bobine B6 (CLI, skill) is
deferred, and Silk needs nothing from it.

- **okfgraph 0.14.0 (O3, image lane)**: `image_backend=embroider|external`
  with `image_endpoint_url`; an external image host must be
  `embroider.serve --kind vision` (identity from `/v1/models`: kind, dim,
  precision, space, text partner), which resizes with
  `embroider.prepare_image`; the image space is pinned
  (`embedding_image_space`). New error codes `EXTERNAL_UNAVAILABLE` and
  `CONVERSION_FAILED`. A declared (non-embroider) image lane falls back to
  captions with one warning.
- **bobine 0.9.0 / 0.9.1 (B2–B5)**: `ConverterConfig(device, precision)` on
  the shared vocabulary, exported as `bobine.DEVICES` / `bobine.PRECISIONS`;
  a reusable `HybridConverter`; `bobine.serve`. B6 deferred.
- **okfgraph 0.14.1 (O6)**: `pdf_device` / `pdf_precision` pass through to
  bobine; one warm converter per settings key, released after
  `pdf_idle_seconds` (default 300) and on close; doctor's `pdf_converter`
  row.
- **okfgraph 0.14.2**: a graph `okf-mcp` / `okf serve` created on this boot
  warns on every MCP result while it is empty (`/health` `created`, doctor
  `db_created`); `--pdf-path` takes folders and `*`/`?` patterns. Design
  note for GGUF images: see section 7.
- **Floors (S6) set in Weave's pyproject** (Weave owns them): extras
  `knowledge-cpu` / `knowledge-gpu` = `okfgraph[cpu|gpu]>=0.14.2` +
  `embroider>=0.15`, and `documents` = `bobine>=0.9.1`. The floor is 0.14.2,
  not the 0.13 that S6 first named: S2's image input needs 0.14.0, S4's
  converter device needs 0.14.1. Silk's own extras still read
  `okfgraph>=0.12` / `bobine>=0.8`; they are names, not Weave's floors.
- **Next: Silk**, in section 5's order: S1 (embed and rerank on
  `embroider.serve`) → S2 embedder and reranker → S5 → S3 (with Weave's
  `node_deprecated` flag) → S2 images + S1 Vision → S1 Decider → S4.
  Open questions 2, 3 and 5 are decided (section 6).

## 1. Goal

One way to load, serve, identify and wire an ONNX model, whichever package
runs it:

- **One vocabulary**: `model_id`, `device` (`auto|cpu|cuda`), `precision`
  (`auto|fp32|fp16|int8`; no `q8` artifact exists), `cache_dir` (a Hugging Face hub cache),
  `revision`. The same words in embroider, bobine, okfgraph and every Silk
  node row.
- **One runtime**: one `onnxruntime` binary per environment, found by one
  resolver (embroider's), never by importing the Python `onnxruntime`
  module.
- **One host**: embroider ships the model server (`python -m embroider.serve`).
  Silk's ONNX Loader runs it, and okfgraph either talks to it over HTTP or,
  when nothing is wired, falls back to embroider in its own process.
- **One identity**: a model in a host describes itself (`GET /v1/models`)
  with everything a graph pin needs, so okfgraph checks a hosted model as
  strictly as its own, and a graph can move between "okf runs it" and
  "Silk hosts it" without re-embedding.
- **The spec nodes go away**: Silk KG Embedder (ONNX) and Silk KG Reranker
  (ONNX) are deprecated. The Knowledge Graph's inputs take an ONNX Loader,
  a GGUF loader, or nothing (okf's internal embroider).

## 2. What each package does today

### embroider 0.14.1 (Rust core, PyO3, native module only)

| Area | Today |
|---|---|
| Model classes | `JinaV5` (text embed, Jina + EmbeddingGemma), `JinaV5Vision`, `GemmaVision`, `Reranker`, `Decider` (GLiClass), `LayaDecider`, `Decider2B`, `Tokenizer` |
| Open signature | shared: `open(model_id, revision, cache_dir, truncate_dim, device, max_length, precision, cpu_arena)`, plus `open_files(...)` for air-gapped use |
| Registry | `available_models()`; kinds `embed`, `rerank`, `decide`, `decide-laya`, `decide-2b`; `register_model`, `register_task`, `load_registry_file` |
| Runtime | `runtime_info()`, `configure_runtime()`, `prepare_cuda_runtime()`; reads `ORT_DYLIB_PATH` |
| Cache | `cache_info`, `cache_info_files` (offline) |
| Rust seams used by others | `SessionPolicy`, `apply_providers`, `cache_info_files`, `fetch_repo_file`, `parse_owner_name`, `report()`, `cuda_available()` (bobine uses all of these) |

**Missing:**

- **No server.** Every consumer that wants a model out of process writes its own; Silk's is `functions/embroider_server.py`.
- **No uniform factory.** Each kind is a separate class with its own call names (`encode_batch`, `score_batch`, `decide`, `encode_image`), and nothing like `open(kind, ...)` covers them all.
- **No self-description.** There is no single `describe()` giving model, kind, landed device, landed precision, dim, ladder, max_length, revision and contract version. Consumers piece it together from `dim()`, `used_cuda()` and `precision()`.
- **No identity string for a vector space**, so okfgraph pins model and precision separately and cannot pin a hosted model at all.
- **`Tokenizer` gives counts but not token ids**, so a `/tokenize` built on it can only return placeholder tokens.
- **No `vision` kind in the registry listing** a host could serve.
- **The image resize contract lives in okfgraph** (`image_assets._prepare_vision`), not next to the models it belongs to.
- **No Python package layer** (`module-name = "embroider"`, no `python-source`), so Python helpers such as the server or a locator have nowhere to live.

### bobine 0.8.0 (Rust core, PyO3, `python/bobine/__init__.py`)

| Area | Today |
|---|---|
| Models | TexTeller (formula OCR), PP-DocLayout, PP-OCRv5 det+rec, SLANet tables, POTATR; bobine owns pre- and post-processing |
| Sessions | built through embroider's Rust `SessionPolicy` + `apply_providers`; hub fetch via `embroider::fetch_repo_file`; `model_status()` returns embroider `CacheReport`s |
| Device | `ConverterConfig.ort_providers` plus six per-pass provider lists, plus `ModelPrecision` and `ModelQuantization` enums |
| Runtime | `_ensure_ort_dylib()` **imports the Python `onnxruntime`** to find the DLL, then `_warm_gpu_dlls` |
| Extras | `cpu = onnxruntime==1.29.0`, `gpu = onnxruntime-gpu==1.29.0` |

**Missing:**

- **Version drift.** The extras pin ORT 1.29.0 while embroider and okfgraph pin 1.31.1, and the README still claims a shared runtime. Installing `bobine[cpu]` next to `okfgraph[cpu]` therefore fails or downgrades.
- **A second runtime resolver.** It imports `onnxruntime`, the same import that puts a second runtime into a host process. okfgraph has a third copy.
- **Different words for the same settings.** bobine speaks provider lists, not `device`/`precision`. Silk's Converter Settings node translates them by hand (`functions/bobine_convert.py`).
- **No persistent host.** Silk runs one child process per conversion (`bobine_worker.py`), so the layout and OCR models load again for every file.

### okfgraph 0.12.1 (Python, runs in `okf serve`)

| Area | Today |
|---|---|
| Text embed | `backend = embroider` (`JinaV5.open` in-process, `LazyRustEncoder`) or `external` (`ExternalEmbedEngine`, OpenAI `/embeddings`) |
| Rerank | `rerank_backend = embroider` (`Reranker.open`) or `external` (`/v1/rerank`, counts from `/tokenize`) |
| Images | in-process `JinaV5Vision` / `GemmaVision` only, plus okf's own resize contract |
| Ingest | bobine in-process (`components/converters.py`) |
| Pins | `embedding_model` (MetaText) and `embedding_precision` (Meta), checked on the crate lane; **the external lane pins the model only**; image pins on the vision lane |
| Runtime | `resolve_ort_dylib()` imports `onnxruntime` (the third resolver) |

**Missing:**

- **The external embedder treats every host as an anonymous server.**
  - It needs a declared `embed_family`, prefix table, pooling and counter.
  - It sends no `task`, so a host that applies its own prompts (embroider: Jina `Query:`/`Document:`, Gemma's prompts) can't tell queries from documents.
  - Its alternative is to prefix the text itself, which would double the prefix on an embroider host.
- **External precision is never pinned**, so an fp16 host can write into an fp32 graph undetected.
- **The external rerank lane** also needs a declared family and max_tokens that an embroider host could report itself.
- **No external image lane**, so images always need embroider inside okf.
- **No shared identity**, so a graph built in-process can't be reopened through a host of the same model, even though the vectors would be identical.

### Silk (today, after c252ad9)

| Lane | Process | Finds the runtime with |
|---|---|---|
| Knowledge Graph (`okf serve`) | okf's | okf's `resolve_ort_dylib` |
| Silk KG Embedder / Reranker (ONNX) | none (a spec okf loads) | n/a |
| ONNX Loader (embed / rerank) | `functions/embroider_server.py` | okf's resolver, imported |
| Document Converter (bobine) | one child per call | bobine's `_ensure_ort_dylib` |
| Classifier Loader (GGUF Laya, `/v1/systemone`) | llama.cpp server | n/a |

- **The same model can be loaded up to three times:** by okf internally, by the ONNX Loader, and by the GGUF Embedder Loader.
- **The KG refuses an ONNX Loader on `embedder`**, because okf has no safe way to pin it.

## 3. Target design

### 3.1 The host wire (embroider owns it, documented in `docs/serve.md`)

| Route | Kinds | Body / answer |
|---|---|---|
| `GET /health` | all | `{"status":"ok"}` |
| `GET /v1/models` | all | `{"data":[describe()]}`, the identity (3.2) |
| `POST /v1/embeddings` | embed | OpenAI shape. `task` (`query`/`document`, or `input_type`) is honoured and **the host applies the model's prompts**; a client never prefixes. `dimensions` is accepted only when it equals the opened dim, otherwise 400 (vectors stay bit-exact). |
| `POST /v1/rerank` | rerank | `{query, documents, top_n}` → `{results:[{index, relevance_score}]}` (the llama.cpp/TEI shape okf already reads) |
| `POST /v1/images/embeddings` | vision | `{image: base64 PNG/JPEG, task?}` → OpenAI shape. **The host resizes** to the slot's contract (moved from okf). |
| `POST /v1/systemone` | decide-* | the llama.cpp shape Silk's `functions/systemone.py` already speaks, so the Classifier node takes an ONNX decider unchanged |
| `POST /tokenize` | all text kinds | `{content}` → `{tokens:[real ids], count}` |

### 3.2 Identity (`describe()`)

```json
{"id": "google/embeddinggemma-2", "object": "model", "kind": "embed",
 "engine": "embroider", "engine_version": "0.15.0", "contract": "0.15",
 "revision": "<resolved sha>", "device": "cpu", "precision": "fp32",
 "dim": 512, "native_dim": 768, "ladder": [128, 256, 512, 768],
 "max_length": 8192, "tuning": "text_embed",
 "space": "embroider/google/embeddinggemma-2@<sha>:fp32:512:8192:text_embed:c0.15"}
```

- **`space`** is the vector-space identity: same `space`, same vectors.
- **Exact or refused.** okf pins it; a hosted model and okf's in-process model with equal `space` are interchangeable, and unequal `space` fails exactly like `MODEL_PIN_MISMATCH` does today.
- **Rerank and decide kinds** carry the same fields, minus dim and ladder. They are never pinned, only reported.

### 3.3 Fallback rule (part (e))

| KG input | okf runs |
|---|---|
| `embedder` ← ONNX Loader (embed) | `backend=external`, identity probed, `space` pinned, `task` sent, no prefixes |
| `embedder` ← GGUF Embedder / GGUF Loader (embedding) | `backend=external`, as today (declared family, prefixes, `/tokenize` counts) |
| `embedder` unwired | `backend=embroider`: okf's own in-process embroider, using the graph's pin (or okf's default model for a new graph) |
| `reranker` ← ONNX Loader / Reranker Loader | `rerank_backend=external` (identity probed when it is embroider) |
| `reranker` unwired | okf's own setting (default: off, or embroider in-process if okf's config names a model) |
| `image_model` ← ONNX Loader (vision) | `image_backend=external` (new) |
| `image_model` unwired | okf's in-process vision slot with its `auto` pairing |

## 4. Work per package

Order matters: embroider first, then okfgraph and bobine, which require the
new embroider; Silk last. Release names are suggestions.

### 4.1 embroider 0.15 (`dev_silk`)

- **E1 Python package layer.**
  - Change: switch maturin to `python-source = "python"` with `module-name = "embroider._native"`, and add `python/embroider/__init__.py` to re-export everything. This is the layout bobine already uses.
  - Must hold: `import embroider` keeps working, every public name stays where it is, and the `.pyi` stubs are updated.
  - Test: an import-surface test listing every public name.
- **E2 One runtime locator.**
  - Add `embroider.locate_runtime() -> {"dylib", "source", "version"}`. It looks for the `onnxruntime` / `onnxruntime-gpu` distribution with `importlib.util.find_spec` and a glob of `capi/`, **never importing it**, and sets `ORT_DYLIB_PATH` when unset.
  - Move bobine's and okf's resolution order here: explicit env first, then a pip install, then the OS path (report only).
  - Fold `_warm_gpu_dlls` into the existing `prepare_cuda_runtime()`.
  - Test: one test per platform branch, like okf's `resolve_ort_dylib` tests, moved over.
- **E3 `describe()` and `space`.**
  - Add `.describe()` on every model class, returning the 3.2 dict. Add `embroider.space_id(model_id, revision, precision, dim, max_length, tuning)` as a pure function.
  - The revision is the resolved snapshot sha, read from the hub cache, never "main".
  - Test: `describe()` on a fake-opened model, plus the `space` format pinned in a golden.
- **E4 Uniform factory.**
  - Add `embroider.open(kind, model_id, **opts)`. It dispatches on the registry kind (`embed`, `vision`, `rerank`, `decide`, `decide-laya`, `decide-2b`) and returns the model object.
  - Add a small `Model` protocol: `describe()`, `count_tokens()`, plus the kind's one call.
  - `available_models()` gains a `vision` kind row for the vision slots, and an `open_kwargs` hint per row (which `open` arguments apply).
- **E5 `Tokenizer.encode(text) -> list[int]`** (and on every model's tokenizer), so `/tokenize` returns real ids.
- **E6 Image preparation.**
  - Add `embroider.prepare_image(data: bytes, model_id) -> (rgb_bytes, h, w)`. It decodes PNG/JPEG with the Rust `image` crate, already a transitive dependency to check, otherwise it becomes a feature, then applies the slot's resize contract.
  - okf's `_prepare_vision` moves here, so host and okf resize identically.
  - Test: the existing `fixtures/vision/*` goldens run through `prepare_image`.
- **E7 The server, `python -m embroider.serve`.**
  - What: Silk's `embroider_server.py`, generalised to every kind, as `python/embroider/serve.py`, using the standard library only (`http.server.ThreadingHTTPServer`, one model lock).
  - Arguments: `--kind --model --revision --device --precision --cache-dir --dim --max-length --host --port [--files model.onnx tokenizer.json]`.
  - Routes: 3.1. Before binding, it calls `locate_runtime()` and pops its own directory from `sys.path`.
  - Exit codes: 2 for bad arguments, 3 when the model fails to open, with the message on stderr.
  - Tests: HTTP tests with a fake model (port Silk's `test_silk_onnx_models.py` host tests), plus one live test per kind, skipped when the model is not cached.
  - Docs: `docs/serve.md`.
- **E8 Docs and compat.** Add the 0.15 row to `docs/compat.md` (ORT 1.31.1, okfgraph 0.13, bobine 0.9), a quickref section, and a CHANGELOG entry.

**Risk.** E7's exactness rests on the host opening the model exactly as okf
does: same `JinaV5.open` arguments and the same `SessionPolicy`. ORT's
intra-op thread count can change the float summation order. Add a parity
test: okf's crate lane and the host embed the same 20 texts, and the
results are compared at `abs=0`. If they differ, `space` gains the thread
count, or the host pins threads to okf's setting.

### 4.2 okfgraph 0.13 (`dev_silk`, requires `embroider>=0.15,<0.16`)

- **O1 Identity probe on the external embedder.**
  - When `backend=external`, `ExternalEmbedEngine` first calls `GET /v1/models`. If the answer says `engine == "embroider"`:
    - Take `family`, `native_dim`, `ladder`, `max_tokens`, `precision` and `space` from the identity. `embed_family` and `embed_prefixes` become optional, and if given they must agree.
    - Send `task` and no prefixes (prefixes off for this engine).
    - Count with `/tokenize` → `count`.
    - Pin `space` (O4) as well as model and precision.
  - Any other host keeps the 0.12 declared contract exactly.
  - Setting: `embed_identity = auto|declared` (default `auto`: probe, and treat a 404 as declared).
  - Tests: a fake embroider host and a fake llama.cpp host, both run through the same graph.
- **O2 The same probe on the external rerank lane**: `rerank_family` and `rerank_max_tokens` become optional for an embroider host, with counts from `/tokenize`.
- **O3 External image lane.**
  - Settings: `image_backend = embroider|external` and `image_endpoint_url`.
  - `ExternalVisionEngine` posts the original image bytes to `/v1/images/embeddings`; the host resizes.
  - The image pins take `space` from the host's identity, and `image_assets` calls `embroider.prepare_image` for the in-process lane (E6).
- **O4 `embedding_space` pin.**
  - Add a MetaText key `embedding_space` (and `image_space`). It is set by whichever lane opens first: in-process from `describe()`, hosted from `/v1/models`.
  - Old graphs carry only model and precision. On first open they adopt `space` when model and precision agree, and they are never rewritten otherwise.
  - Result: a graph moves between in-process and a host with no re-embed when `space` matches, and is refused with a clear message when it doesn't, for example "pinned fp16, host landed fp32 on CPU".
- **O5 Runtime**: `resolve_ort_dylib()` delegates to `embroider.locate_runtime()`, keeping the current function as a shim for one release.
- **O6 Ingest vocabulary**: `converters.py` passes bobine `device`/`precision` (B3) instead of provider lists, and `BobineConverter` reuses one bobine converter per settings key (B4's in-process API) so the layout/OCR models stop reopening per file, with a bounded lifetime: the CLI holds it only for the files of one `okf ingest --kind pdf --pdf-path A B …` (0.13.1) and frees it when the command ends; `okf serve` frees it after a set idle time (a setting; never held for the server's whole life).
- **O7 Reporting**: `okf doctor` and the serve status show, per lane, `backend`, `endpoint`, the identity, and the pinned `space`. The Silk KG's Info line reads this.
- **Docs**: settings reference, CHANGELOG, and the compat row.

### 4.3 bobine 0.9 (`dev_silk`)

- **B1 Fix the runtime pin now** (worth its own 0.8.1): the extras become `onnxruntime==1.31.1` / `onnxruntime-gpu[cuda,cudnn]==1.31.1`, and the README text follows. Also check that bobine's lock already resolves embroider 0.14.1 at 1.31.1.
- **B2 Runtime locator**: `_ensure_ort_dylib` and `_warm_gpu_dlls` are replaced by a native call to embroider's Rust locator and `prepare_cuda_dirs()`, so `import bobine` no longer imports `onnxruntime`. `bobine.ORT_DYLIB_PATH` stays.
- **B3 Shared vocabulary.**
  - `ConverterConfig(device="auto|cpu|cuda", precision="auto|fp32|fp16|int8")`. Device maps to `ort_providers` through `embroider::DeviceReq`; precision maps onto `ModelPrecision` / `ModelQuantization`. **No `q8`**: bobine's enums are `ModelPrecision{Fp32,Fp16}` + `ModelQuantization{Fp32,Int8}` and no artifact in the stack ships q8, so bobine refuses the word by name rather than pretending. B3 also exposes `bobine.DEVICES` / `bobine.PRECISIONS` so okfgraph's settings validate against the source instead of a duplicate tuple (okfgraph 0.14.1 §E).
  - Per-pass provider lists stay as the expert override and win when set.
  - `describe()` on `HybridConverter` returns the landed provider and precision per pass.
- **B4 Persistent host, `python -m bobine.serve`.**
  - Routes: `POST /v1/convert` (`{path, output_dir, settings}`; NDJSON progress lines, then the result), `GET /v1/models` (`model_status()` plus the landed devices), and `GET /health`.
  - One `HybridConverter` per settings key stays warm, which removes Silk's per-call model reload.
  - The same reuse in process: today `bobine.ingest_document` builds a new `HybridConverter` per call, and the reusable Python `HybridConverter` can't stage images, count pages or report progress. Add `HybridConverter.ingest_document(path, output_dir, on_page=, should_continue=)` with the full pipeline, so a caller (okfgraph O6) holds one converter across files, and a way to release it (dropping the object frees the ONNX sessions) so the caller sets its lifetime. B4's host evicts a warm converter after an idle timeout the same way.
  - Cancel is a per-job route, or killing the host stays the hard stop.
- **B5 Tests**: the vocabulary mapping table, the locator without `onnxruntime` imported (assert `"onnxruntime" not in sys.modules`), and B4's HTTP with a text file (no models).
- **B6 CLI and skill (deferred, after 0.9)**: bobine ships neither today; the
  only entry points are the Python API and B4's host.
  - A `bobine` console script: `convert FILE... [--output-dir] [--device]
    [--precision] [--routing-mode]`, many files in one process on one warm
    `HybridConverter`; `models` (status, offline) and `serve` (B4).
  - A harness-neutral `skills/bobine-cli/SKILL.md`, shipped inside the
    wheel, following the pattern okfgraph already ships: its `pyproject.toml` force-includes `skills/*/SKILL.md` (since 0.13.1) and `okf setup skills` copies them into an agent's skill directory.

### 4.4 Silk (Weave `dev_gui`, after the releases)

- **S1 ONNX Loader on embroider's server.**
  - `EmbroiderPool` runs `[python, -m, embroider.serve, ...]`, and `functions/embroider_server.py` is deleted.
  - Roles: **Embedding**, **Reranker**, **Vision** (new, output `image_model`), and **Decider** (new: embroider's Laya / GLiClass / 2B on `/v1/systemone`, output `model_obj` with role `classifier`, for the Classifier node).
  - Status shows `describe()`: landed device and precision, dim, and `space`.
- **S2 Knowledge Graph wiring (3.3).**
  - `embedder` accepts the ONNX Loader. The `wiring_problem` refusal goes, and an embroider handle maps to `backend=external` with `embed_identity=auto`.
  - A new `image_model` input takes the ONNX Loader in the Vision role.
  - When unwired, the KG sets no backend env and okf uses its in-process embroider. The KG's Info names the lane and `space`, and warns on a pin it can't meet, for example "graph pinned fp16; the loader landed fp32 on CPU, pick CUDA".
  - okf serve restarts when a lane's wiring changes, as it already does for the GGUF lane.
- **S3 Deprecate the spec nodes.**
  - Silk KG Embedder (ONNX) and Silk KG Reranker (ONNX) keep loading and working for one release. The KG still reads their wires as okf in-process settings. Their description and title say "deprecated: use ONNX Loader, or leave the input unwired".
  - Weave has no flag to hide a node from the palette, so add one (`node_deprecated: ClassVar[bool]`): the palette skips the node, scenes still load it, and the linter warns on new uses. That is a small change in Weave.
  - Remove the spec nodes in the release after.
  - Optional: a canvas action "Replace with ONNX Loader" that copies model, device, precision, cache and width.
- **S4 Converter Settings on the shared vocabulary**: its Device and Precision rows match the ONNX Loader's, `bobine_convert.clean_settings` passes them through (B3), and the worker can move to a warm `bobine.serve` pool (B4) behind the same `ChildServerPool`.
- **S5 Runtime status in one place**: a small `functions/onnx_runtime.py` asks `embroider.runtime_info()` in a child process (never in Weave's process) and caches the answer. Every ONNX node's tooltip and error text uses it, for example "no onnxruntime: pip install okfgraph[cpu]".
- **S6 Pins and floors** (done 2026-11-13): Weave's pyproject raises the floors (`embroider>=0.15`, `okfgraph>=0.14.2`, `bobine>=0.9.1`). Weave owns the floors; Silk and Veil don't.
- **S7 Tests**:
  - The KG maps each of the 3.3 rows onto okf env.
  - The ONNX Loader runs each role against a fake host.
  - The spec nodes still load an old scene and warn.
  - A live end-to-end test, skipped unless the model is cached: an unwired KG builds a graph in-process, then the same graph is reopened through an ONNX Loader with the same model, and search works with no re-embed.

## 5. Sequence

1. **bobine 0.8.1**: B1 only (runtime pin). Small, removes a real install conflict.
2. **embroider 0.15**: E1 → E2 → E3 → E5 → E4 → E7 → E6 → E8. E7 can start once E1 lands; E6 is only needed for vision.
3. **okfgraph 0.13**: O5 → O4 → O1 → O2 → O7, then O3 (vision) and O6 (after bobine 0.9).
4. **bobine 0.9**: B2 → B3 → B4 → B5. B6 (CLI, skill) follows in a later release.
5. **Silk**: S1 (embed and rerank on `embroider.serve`) → S2 embedder and reranker → S5 → S3 → S2 images and S1 Vision → S1 Decider → S4.
6. **Weave**: S6 floors (done 2026-11-13, ahead of step 5) and the `node_deprecated` flag, together with S3.

Each step ends green on its own: embroider `cargo test` + pytest, okf's suite, bobine's suite, and Silk's suite plus Weave's seam tests.

## 6. Open questions (all closed, 2026-11-13)

1. **Bit-exactness across processes** (see the 4.1 risk): if ORT threads change vectors, either `space` includes threads or the host matches okf's thread setting. Measure before O4 ships.
   **Closed:** host and in-process vectors measured equal at `abs=0` on CPU; okfgraph 0.13 shipped the space pin on that basis.
2. **Which `ChildServerPool` owns a model used by two nodes?** Today each loader starts its own host. A process-wide Silk registry keyed by `space` would share one host per model; worth it once the KG and a ToolSet use the same embedder.
   **Decided: no registry; sharing is wiring.** One loader node owns one lazy host, and its output already feeds the KG and a ToolSet at once. A registry that merges two loader nodes would need answers to who stops the shared host and what happens when one node changes its device. Instead, a loader's status warns when another loader already holds the same model, device, precision and width ("already loaded by <node>; wire that one instead"). Revisit only if duplicate loaders prove common.
3. **Rerank when unwired**: should okf's in-process reranker stay available as the unwired fallback (today it needs okf config), or should Silk say "no reranker" plainly?
   **Decided: no reranker wired, no reranking**, as `knowledge_graph.py` already does ("No reranker wired: searches are not reranked."). A hidden in-process reranker would spend memory and time the user never chose. After S3 removes the spec nodes, the ONNX Loader in the Reranker role is the only way to rerank on embroider; the deprecated Silk KG Reranker's note says so.
4. **E1 is a packaging change** to a published wheel. It needs one release candidate built on all three platforms before okfgraph depends on it.
   **Closed:** embroider 0.15.0 released on all three platforms; okfgraph 0.13+ depends on it.
5. **GPU memory**: the host defaults to CPU (leaving VRAM to chat models), but a graph pinned fp16 needs CUDA. Should the KG suggest the device from the pin, or should the ONNX Loader read the pin?
   **Decided: the KG checks and suggests; the loader keeps the user's device.** The loader can't know which graph it serves (it may feed several), and VRAM is the user's call, so it stays on its `cpu` default. The KG already reads the graph's pins (`pin_info`) and the loader's settings, and the loader starts lazily, so before anything starts the KG predicts the precision the loader would land on. On a mismatch it refuses with the fix instead of starting a host okf would reject: "graph pinned fp16; this loader would land fp32 on CPU — set its Device to cuda, or rebuild the graph." Part of S2; an unwired KG is unaffected (okf adopts the graph's pins in-process).

## 7. Images on a GGUF graph (after this plan)

A graph whose text lane is a GGUF Embedder Loader (llama.cpp) has no image
lane today: okfgraph refuses a declared image identity and falls back to
captions. The earlier idea of pairing it with an embroider vision model is
dropped. llama-server embeds images itself on `/v1/embeddings` when loaded
with `--mmproj`, so text and images can come from **one** multimodal GGUF
model, in one space by construction. okfgraph's design note:
`OKFgraph/docs/plan-declared-image-identity.md`.

Gates, in order:

1. **Spike on a current llama.cpp build** (no code): a multimodal
   embedding GGUF + mmproj under `--embedding --mmproj --pooling mean
   --alias <model_id>`. Confirm the image request shape on
   `/v1/embeddings`, the `multimodal` capability and the alias on
   `/v1/models`, distinct vectors for distinct images (llama.cpp #13666),
   text→image retrieval that makes sense, and how a text+image element
   composes. Pick the model and check its license.
2. **okfgraph** (likely 0.15.0): `image_identity = "declared"`, valid only
   when the image model is the graph's text model; checks at open (name,
   capability, two-image probe, same truncation and L2 normalization);
   model/precision pins keyed on the declared id, no space pin.
3. **Silk**: the GGUF Embedder Loader starts llama-server with `--mmproj`,
   `--embedding`, `--pooling` and `--alias`, and fetches the mmproj with
   the model. The KG's `image_model` input accepts a GGUF loader only when
   it is the same loader that feeds `embedder`, mapping to
   `image_backend=external`, `image_identity=declared` and the same host
   URL. Anything else keeps captions.

A graph built on a text-only GGUF cannot gain images: its text vectors
come from another model. Images need a graph built on the multimodal GGUF.
