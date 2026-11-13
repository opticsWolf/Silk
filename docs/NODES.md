# Silk Node Reference

Every Silk node searches under the **Silk AI** category, in one of four
submenus: **Loaders**, **Agents**, **Display** and **Configuration**. Ports
are listed as they are registered; `exec` ports are trigger pulses.

*(The category was plain "AI" until 2026-09-19, and the read-only surfaces
— Hook Monitor, Plan Viewer, Task Hub, Chat Log Display — sat under Weave's
own **Display** category. Both were changed so a Silk install adds one
menu, not entries scattered through a host's.)*

## Model

### GGUF Loader — `nodes/gguf_loader.py` (`GGUFLNode`)
Loads a GGUF model via llama.cpp into the shared pool (thread-safe; ejects
the previous model when re-run).

| Direction | Port | Type |
|---|---|---|
| in | `model_path` | `filepath` |
| in | `prompt_cache` | `filepath` |
| out | `model_obj` | `model_handle` |
| out | `pool_info` | `dict` (live pool stats) |

The handle carries `supports_tools` when the GGUF's own chat template was
written for structured tool calling — the probe reads the template and
looks for it being handed a `tools` list *and* rendering `tool_calls`.
That is what puts the agent on the native protocol instead of the text
fence. A file with no chat template at all (an embedding model, a vision
projector) is "unknown" rather than "no", and unknown uses fences.

**Group Requests by Agent** (advanced mode, on by default) is D47
mechanism A. One server holds one resident KV context, so two agents
taking turns overwrite each other's prompt cache and each round
re-prefills everything past the system prompt: measured at 74.8% prefix
reuse for one conversation against **0.4%** for two running at once, and
the same work in 1.1 s against 4.5 s. With grouping on, two conversations
reuse exactly what one does. It changes only the *order* of a queue that
already exists -- the server serialises every request anyway (D43) -- so
it costs no throughput, only a short wait at a switch, and that wait is
capped by what a lost prefix was measured to cost. Turn it off if your
prompts are small enough that prefill does not matter; the Pool Monitor's
*Affinity* row reports the wait it costs beside the reuse it buys.

### Model Endpoint — `nodes/model_endpoint.py` (`SilkModelEndpointNode`)
A model that runs somewhere else, wired where the loader would go (D45).
Emits the same `model_handle`, so Agent, Worker and the ToolBox's
embedding input take it without knowing the difference.

| Direction | Port | Type |
|---|---|---|
| in | `provider` | `string` (preset key) |
| in | `base_url` | `string` |
| in | `model` | `string` |
| in | `credential` | `string` (a **name**, never a value) |
| in | `context_length` | `int` (0 = unknown) |
| in | `supports_tools` | `bool` (native tool calling, off) |
| out | `model_obj` | `model_handle` |

Presets: LM Studio, llama.cpp server, Ollama, vLLM, LiteLLM proxy,
OpenRouter, and **Custom** for anything else that serves
`/v1/chat/completions` — a hosted gateway, your own proxy, a colleague's
box. A preset only fills *empty* fields, so "this provider, my host" is
one edit rather than a re-type.

**What the endpoint already knows, it fills in.** The `/models` request
that lists the models also carries, on a gateway that says so, each
model's price, its context window and whether it takes a `tools` field.
The node reads all three from that one response: the price goes on the
handle (and is what a `cost=` budget is measured against — see D88), and
the other two fill the fields *if they are empty*. A value you typed
always wins, because you can see a proxy in the way that a catalogue
cannot. LM Studio and llama.cpp quote no price, which the status line
says plainly rather than implying the run is free.

### OpenCode Zen / OpenCode Go — `nodes/opencode_endpoint.py` (`SilkOpenCodeZenNode`, `SilkOpenCodeGoNode`)
The Model Endpoint, specialised for OpenCode's two plans: Zen
(`https://opencode.ai/zen/v1`, pay as you go) and Go
(`https://opencode.ai/zen/go/v1`, subscription). Same `model_handle` out.

| Direction | Port | Type |
|---|---|---|
| in | `model` | `string` (dropdown, editable) |
| in | `context_length` | `int` (slider; 0 = unknown) |
| in | `credential` | `string` (default `OPENCODE_API_KEY`) |
| in | `supports_tools` | `bool` (auto-on when the catalogue says so) |
| out | `model_obj` | `model_handle` |

**The model list is a file.** `~/.weave/silk/model_catalogs/opencode.json`
and `opencode-go.json`, read when the node opens. The refresh button
rewrites it off the GUI thread; the first node of a kind on a machine with
no file refreshes once by itself. OpenCode's `/v1/models` names the models
(it is the authority on what your key reaches) but carries no context
window, so the windows come from [models.dev](https://models.dev), the
catalogue OpenCode itself is built on. If models.dev is unreachable the
list still loads with every window *unknown*; if the endpoint is, the
models.dev list is used, so the dropdown works before a key is set. If
neither answers, the file is left as it was.

**The context slider tops out at the model's window** and starts there.
Holding it lower stores that value for the model in the same file
(`context_overrides`), so it survives a refresh and is the default next
time anyone picks that model; dragging it back to the maximum forgets the
override. A model OpenCode documents on `/messages` or `/responses`
rather than `/chat/completions` (Claude and GPT on Zen, MiniMax on Go) is
listed but flagged in the status line, since Silk speaks chat completions
only. Zen's per-token price from models.dev goes on the handle; Go is a
subscription and quotes none.

**Native tool calling** is a checkbox, and it is off. On, the agent's
tools go in the request's `tools` field; off, they go in a text fence.
The two failure modes are not symmetric — a server that does not accept
`tools` refuses the whole request, while fences merely cost accuracy on a
model that could have done better — so the protocol that works everywhere
is the one you get without asking. Most hosted gateways support native;
small local models often do not. Verified live against LM Studio.

Three behaviours worth knowing:

- **It asks before it answers.** Connecting probes `/models`, so a wrong
  URL or an unset key shows up here rather than as a failed agent run
  three nodes downstream. One model advertised and none chosen means that
  one is used; several means it asks, because picking one of nine on a
  metered gateway is picking someone's bill. A model the endpoint does not
  list is still used — gateways route unlisted aliases — and logged.
- **The credential is a name** (D22). The field holds the name of an
  environment variable or of an entry in `~/.weave/silk/secrets.json`,
  resolved at connect time into the request headers. The status line says
  the name resolved; it never shows the value, and neither does the handle,
  so a saved graph stays shareable.
- **An unknown context window stays unknown.** Most endpoints do not
  advertise one. Compaction (D25) needs a real denominator, so 0 means
  "off" rather than a guess that would summarise too early or overflow.

**Why there is no litellm dependency.** OpenRouter, LM Studio, vLLM,
Ollama and litellm's own proxy all speak the OpenAI chat API, so one HTTP
client reaches all of them. For providers that do not (Anthropic, Gemini,
Bedrock), run the **litellm proxy** and point this node at it: the
provider-specific translation stays in a process that specialises in it,
and Silk's model layer stays one wire format wide. An Unsloth fine-tune is
an ordinary model once vLLM or llama.cpp is serving it.

### Model Fallback — `nodes/model_fallback.py` (`SilkModelFallbackNode`)
One model behind another (D89). When the primary fails for good — the
server is gone, the key was revoked, the model was retired, or three
spaced retries all came back 503 — the run continues on the fallback
instead of ending.

| Direction | Port | Type |
|---|---|---|
| in | `primary` | `model_handle` (tried first) |
| in | `fallback` | `model_handle` (tried when the primary fails) |
| out | `model_obj` | `model_handle` (the chain, as one handle) |

What it emits *is* a model handle, so the Agent downstream sees one model
and runs the way it always has. Wire two of these in series for a chain
three deep; the flattening is done here, so nothing downstream recurses.
The same model wired into both ports is deduplicated — falling back to
the model that just failed cannot help.

**A chain is as capable as its weakest link**, and the status line says
so, because the trade is real:

| | rule | why |
|---|---|---|
| native tool calling | only if **every** member has it | the transport is chosen once, before the first request, and the system prompt is written to match it |
| context window | the **smallest** known one | compaction plans its cuts against it, and a conversation grown to fit the primary must still fit whatever catches it |
| price | only if **every** member quotes one | a cost cap that stops binding when the cheap fallback takes over is not a cap (D88) |

So a fence-only local model behind a native-tools gateway puts the whole
chain on fences. That is a trade, not a mistake — the status line states
it and lets you decide, rather than making it quietly.

**What a switch does not do.** It does not spend a round (`max_rounds`
bounds the model's reasoning, and being handed a dead server is not a
thought), it does not restart the conversation, and it does not happen
after tokens have already reached you — a second model continuing over a
partial answer would splice two voices into one turn. It also never
happens on a context overflow, which is compaction's (D40): the next
model's window is no larger. The fallback starts with a retry budget of
its own.

**What it does do** is yield an `EventModelSwitch` on the `events` port,
and the Agent node's status line says which model took over. The answer
from that point comes from a different model at a different price; a run
that finished on the cheap fallback should not look identical to one that
finished on the paid primary.

## Knowledge

### Knowledge Graph — `nodes/knowledge_graph.py` (`SilkKnowledgeGraphNode`)
Serves one OKFgraph knowledge graph (`okf serve`, in its own process) and
hands it to ToolBoxes, whose `knowledge` tool group gives agents search,
read, traverse and, by access level, ingest and curation.

| Direction | Port | Type |
|---|---|---|
| in | `roots` | `dirpath_list` (where to find or create `.okfgraph/okfgraph.db`) |
| in | `embedder` | `model_handle` (the Embedder Loader's `model_obj`, or the ONNX Loader as Embedding) |
| in | `onnx_embedder` | `onnx_embedder` (Silk KG Embedder (ONNX)'s `embedder_spec`) |
| in | `image_model` | `image_model` (the ONNX Loader as Vision) |
| in | `reranker` | `reranker_model` (Silk KG Reranker (ONNX), Reranker Loader, or ONNX Loader as Reranker) |
| out | `knowledge` | `okf_graph` |

The node has **no model fields**: the models come on the wires, and the
**Embedder** row (`embed_source`: onnx / gguf, with the Re-embed button)
picks which wire embeds; the other is ignored.

| Embedder | Lane in okf | What the node passes |
|---|---|---|
| gguf, Embedder Loader wired | external (`/v1/embeddings` on Silk's llama.cpp server) | the server URL, pin `gguf/<file stem>` (`+p<6 hex>` with declared prefixes), the query/document prefixes, and from the GGUF header: family, pooling, width; exact token counts from the server's `/tokenize` |
| gguf, ONNX Loader wired | external (`/v1/embeddings` on embroider's host) | the host URL and the model id; nothing is declared, because okf asks the host who it is (`embed_identity=auto`) and pins the space it names. The loader's Width must equal the graph's Vector dim |
| onnx, `onnx_embedder` wired | embroider, in okf's process | model id, device, precision, cache folder, image model and its precision |
| onnx, nothing wired | embroider | the ONNX model the graph was last embedded with; `google/embeddinggemma-2` (Silk's default, `DEFAULT_EMBED_MODEL`) for a new graph |

With nothing wired, okf finds the model itself: embroider looks it up in
the Hugging Face hub cache (`~/.cache/huggingface/hub`, or the ONNX
Embedder's cache folder) and downloads it on first use. The Info line
says which it is (`okf model-info`, offline): "Model files: … in <cache>"
or "Not downloaded yet: … the first ingest or search fetches it". Not
downloaded is a notice, not an error. A graph this node has not served
before that was embedded with another embroider model (okf's old default,
jina v5 small, say) keeps that model ("Kept …") instead of re-embedding.

**The node turns red** when it cannot serve a graph that searches: a
wiring problem, a gguf Embedder row with nothing wired, okf failing to
start, a pin it cannot read, or a graph embedded with other settings.
The Info line says which; the wire still carries `None` (or the
mismatched graph), and the next evaluation that serves one clears it. No
database chosen yet is not red.

The Embedder Loader is not loaded by hand: its server starts when this
node asks for its address, which is when the graph is served.

What a GGUF embedder is, okf is told from its file, never from a field:
the **family** is `nomic-v1.5` when the model is one (okf then adds its
`search_query:` / `search_document:` prefixes and the 64–768 ladder),
otherwise the model's own name; the
**pooling** is the file's `<arch>.pooling_type`, which is what llama.cpp
pools by; the **width** is the file's `<arch>.embedding_length` (Vector
dim applies to ONNX models and nomic only). A model wider than 1024 is
refused, as is a chat model on `embedder`.

**Prefixes.** Retrieval embedders are trained to read a query and a
passage differently, told apart by a prefix (`query: ` / `passage: ` for
e5, an instruction before the query for bge, mxbai, arctic and
Qwen3-Embedding). okf adds only the prefixes it is declared, so Silk
declares them from the model's name (`functions/embed_prefixes.py`);
the Embedder Loader's **Prefixes** field overrides that (`auto`, `none`,
or `Query=...|Document=...`). A declared table tags the pin
(`gguf/<stem>+p<6 hex>`): changing the prefixes asks for a re-embed, as
changing the model does. A graph embedded before Silk declared them is
served without them ("Embedded without the query/document prefixes…")
until Re-embed is pressed; nothing re-embeds unasked. nomic keeps okf's
own table and its untagged pin.

**Token counts** are exact: Silk's llama.cpp server
(`functions/llama_server.py`) answers llama-server's `POST /tokenize`,
and okf counts chunk lengths with it (`server-tokenize`) instead of
estimating from the file.

Unplugging a loader never re-embeds. On the ONNX lane the graph keeps its
model; a graph embedded with a wired model stops serving and waits for it
to be wired again (or for the Embedder row to be set to onnx, which
re-embeds with the ONNX model).

**Images** come from the ONNX Loader in the Vision role, on
`image_model` (okfgraph 0.14): okf embeds them on its external image lane,
so no vision ONNX loads beside the chat model. The host names the text
model whose vectors it shares, and that must be the graph's own text
model; the node refuses a mismatched pair (red, with both names) before
okf does. The image host must be open at the graph's width. Images are
pinned by okf on the first one it embeds; with no image wire, okf uses
its own vision slot when the text model has a partner, else captions.

**Reranking is on exactly when a reranker is wired.** Then every agent
search is reranked (`rank='cross'` on concepts, `rerank` on chunks; the
`hub` and `ppr` graph walks keep their order); with none wired no search
is, and okf's default cross-encoder is never loaded. The reranker writes
no vectors, so changing it restarts the server and never re-embeds.
An ONNX Reranker is loaded by okf itself (embroider); a Reranker Loader's
GGUF model is reached on okf's external rerank lane (okfgraph 0.12):
the node starts the loader's `/v1/rerank` server, passes its address,
the model's family, the loader's context as the pair ceiling
(`rerank_max_tokens`) and `rerank_count = server-tokenize`, so okf counts
pairs with the server's own `/tokenize`. `reranker_model` also carries
the plain external form (any `/v1/rerank` host by URL and family).

### Reranker Loader — `nodes/reranker_loader.py` (`RerankerLoaderNode`) *(Silk AI / Loaders)*
A GGUF reranker for the Knowledge Graph — a cross-encoder with a
classification head (bge-reranker-v2-m3) or a Qwen3-Reranker — served on
`POST /v1/rerank` by `functions/rerank_server.py` in its own process
(`functions/rerank_pool.py`). The Python `llama_cpp.server` has no rerank
endpoint, so Silk scores pairs with llama.cpp directly: rank pooling for
a file with a head, and for a Qwen3 reranker converted without one,
Qwen3's own prompt and P("yes"). The pair is built like `llama-server`
builds it (the GGUF's rerank template, else `[BOS] q [EOS] [SEP] doc
[EOS]`); a pair longer than the context keeps the query and cuts the
document.

| Direction | Port | Type |
|---|---|---|
| in | `model_path` | `filepath` |
| out | `reranker` | `reranker_model` (`runtime: gguf`) |

Fields: model path, Device (CPU / GPU), Context (the pair ceiling, also
told to okf), CPU threads, mmap. Like the Embedder Loader it is **loaded
on demand**: the handle goes out at once and the server starts when the
Knowledge Graph serves a graph with it; Unload stops it. A server that
will not start turns the Knowledge Graph red with the reason.

### Silk KG Embedder (ONNX) / Silk KG Reranker (ONNX) — `node_deprecated`, kept for saved scenes — `nodes/onnx_loader.py` (`SilkOnnxEmbedderNode`, `SilkOnnxRerankerNode`) *(Silk AI / Knowledge)*
**Deprecated (Silk 2026-11):** hidden from the add-node palette, still
loaded by saved scenes, removed one release later. Use the ONNX Loader
instead — its **Embedding** handle on the Knowledge Graph's `embedder`,
its **Reranker** handle on `reranker` — or leave the input unwired (okf
then embeds in-process with the graph's own pin, and with no reranker
wired no search is reranked). Both nodes keep working for this release.

Choose the embroider model (id, device, precision, cache folder) a
Knowledge Graph embeds or reranks with, and put the choice on a wire.
These are **specifications, not models**: nothing loads in Weave. A second
ONNX Runtime in Weave's process can take the interpreter down, so the
Knowledge Graph node passes the specification to `okf serve`, which loads
it with embroider, and the graph's vectors and pin stay okf's. Only the
Knowledge Graph reads these wires; for an ONNX model other nodes or your
own code can call, use the ONNX Loader below.

| Node | Output | Type |
|---|---|---|
| Silk KG Embedder (ONNX) | `embedder_spec` | `onnx_embedder` |
| Silk KG Reranker (ONNX) | `reranker_spec` | `reranker_model` (`runtime: embroider`) |

Until Silk 2026-11 they were called ONNX Embedder and ONNX Reranker, sat
under Loaders, and their outputs were `onnx_embedder` and `reranker`.
Scenes saved then load with their wires on the renamed outputs
(`node_renamed_ports`).

The model lists are embroider's registry and editable, so a model a newer
embroider gained can be typed in. embroider 0.14 adds the English
`cross-encoder/ms-marco-MiniLM-L-6-v2` beside the multilingual default.

Silk KG Embedder (ONNX)'s **Precision** offers only what the chosen model ships:
int8 is jina nano's alone. Asked for one it lacks, the node turns red
instead of okf failing at start.

**Images** and **Image prec.** choose how okf embeds images for
image-content search (`OKFGRAPH_IMAGE_MODEL_ID`, `OKFGRAPH_IMAGE_PRECISION`).
An image model's vectors only compare against its text partner, so each
text model has at most one:

| Text model | Image model |
|---|---|
| `google/embeddinggemma-2` | `google/embeddinggemma-2-vision` |
| `jinaai/jina-embeddings-v5-text-nano-retrieval` | `jina-v5-omni-nano-retrieval-vision` |
| `jinaai/jina-embeddings-v5-text-small-retrieval` | none: images are searched by caption |

`auto` is okf's own pairing (the table). A partner that does not match is
greyed out, and wiring one anyway turns the node red. Image weights come
in fp32 and fp16 only, and image search needs a vector width of 768 or
less. okf pins the image model on the first image it embeds; the
Knowledge Graph's Info says which one embeds images.

### ONNX Loader — `nodes/onnx_model_loader.py` (`OnnxLoaderNode`) *(Silk AI / Loaders)*
An embroider model as a real model: it runs in its own process
(`python -m embroider.serve`, run by `functions/embroider_pool.py`), and
the handle on the wire speaks HTTP to it, in the shapes the GGUF servers
answer. A consumer cannot tell it from its GGUF twin. **Role** picks what
the model is, and fills one of four outputs (the others carry nothing):

| Role | Output | Type | Serves |
|---|---|---|---|
| Embedding | `model_obj` | `model_handle` (`backend: onnx`) | `POST /v1/embeddings` |
| Reranker | `reranker` | `reranker_model` (`runtime: onnx`) | `POST /v1/rerank` |
| Vision | `image_model` | `image_model` (`runtime: onnx`) | `POST /v1/images/embeddings` |
| Decision | `classifier` | `classifier_model` (`backend: systemone`) | `POST /systemone` |

Every host answers `GET /health` and `GET /v1/models` (embroider's
`describe()`: landed device and precision, width, vector space, contract);
the text kinds also answer `POST /tokenize` (an exact count). Fields:
Role, Model (the role's models from embroider's registry, editable; a
typed id survives a role change), Device (CPU by default, leaving VRAM to
the chat model), Precision, Width (embedders and image models: the
model's Matryoshka steps, `native` for its own), Cache.

- **As an embedder** it works wherever `functions.embeddings.embedder_for`
  takes a handle: a ToolSet's `embedding_model`, memory recall, your own
  nodes. `/v1/embeddings` takes `task` (`Document`, the default, or
  `Query`; `input_type` `query` / `document` too), because the retrieval
  models embed a query and a passage differently.
- **As a reranker** it works on the Knowledge Graph's `reranker` port, which

  hands it to okf's external rerank lane like the Reranker Loader's
  (pair ceiling and family from the registry, exact counts from
  `/tokenize`), and on any `/v1/rerank` client.
- **As a vision model** it names the text model whose vectors it shares
  (`text_partner`): image vectors compare only against that model, so a
  graph's text lane must be it.
- **As a decision model** it answers typed questions (choice, score,
  noul) on `/systemone`, for the Classifier and Ensemble nodes. The
  bands table is keyed per model, so a decision model the Classifier has
  not been calibrated for shows up as such.
- **The Knowledge Graph's model too.** On `embedder` the Knowledge Graph
  takes this handle onto okf's external lane (`embed_identity=auto`), so
  the graph's vectors are still okf's to make, with the model this node
  runs; on `image_model` likewise for images. Knot a graph to a loader,
  not to Silk KG Embedder (ONNX), when the model should be one process.

**Loaded on demand**, like the GGUF loaders: the handle goes out at once,
the process starts the first time something uses the model, and Unload
stops it. **The runtime is asked once, in a child.** Weave must not import
`onnxruntime` (a second one takes the interpreter down), so
`functions/onnx_runtime.py` runs `embroider.runtime_info()` in a `python
-c` child the first time a node evaluates and caches the answer: the Info
line ends with "ONNX Runtime 1.31.1, CUDA ready" (or "CPU only"), the
Device tooltip gains it, and a missing runtime, or a Device row asking for
CUDA on a CPU-only install, stops the evaluation with one sentence
(`okfgraph[gpu]`, or pick CPU) instead of failing later inside okf. If the
same model is also chosen with Silk KG Embedder (ONNX), it is in memory
twice: once in okf's process, once in this one.

## Documents

Documents to Markdown with [bobine](https://pypi.org/project/bobine/)
(Silk's `documents` extra): PDF (formulas as LaTeX, table structure, OCR
for scans), Office files including the legacy formats, JATS XML articles
and text. **bobine runs in a child process per call**
(`functions/bobine_worker.py`): `import bobine` loads ONNX Runtime, which
must not run in Weave's process. That also makes Cancel exact. The PDF
model routes need an `onnxruntime` install in the environment; without
one, PDFs keep to the text layer and the nodes say so. Without bobine the
nodes say how to install it and output nothing.

### Converter Settings — `nodes/document_nodes.py` (`SilkConverterSettingsNode`) *(Silk AI / Documents)*
bobine's options on a `bobine_settings` wire, as plain data: PDF route
(`never` text layer only · `surgical` + formula OCR · `auto` + layout
and OCR where needed · `always` every page), device, precision, formula
weights (int8, CPU only), table backend, images (link / embed), extract
images, detect headings, formula and page DPI, OCR language, model cache.
**Check models** reports which models are downloaded and whether an ONNX
Runtime is there to run them; it downloads nothing. Unwired, the
converters use bobine's defaults.

### Document Converter — `nodes/document_nodes.py` (`SilkDocumentConverterNode`) *(Silk AI / Documents)*
A file, or every file bobine reads in a folder (optionally with
subfolders), to `<output>/<stem>/<stem>.md` with its images and, for
workbooks, a CSV per sheet. Output empty: a `markdown` folder beside the
source. Runs on **Convert**, with page progress and **Cancel** (files
already done stay); a folder keeps going past a file that fails and
reports it.

| Direction | Port | Type |
|---|---|---|
| in | `settings` | `bobine_settings` |
| in | `source` / `folder` / `out_dir` | `filepath` / `dirpath` / `dirpath` |
| out | `markdown` | `string` (a folder's files joined, each under a `<!-- name -->` marker) |
| out | `md_path` | `filepath` (the first file's) |
| out | `output_dir` | `dirpath` |
| out | `report` | `dict` (per file: paths, pages, images, data files; failures; settings) |

### Excel Converter / JATS Converter — `nodes/document_nodes.py` (`SilkExcelConverterNode`, `SilkJatsConverterNode`) *(Silk AI / Documents)*
One file each, converted when the path changes. **Excel** (xlsx, xlsm,
xls, xlsb, ods): `markdown` (a table per sheet), `sheets`, `csv_by_sheet`
and the cell `json`. **JATS** (.xml / .nxml, takes `settings` and an
output folder): `markdown` with LaTeX math, `md_path`, `title` and an
`info` dict of formula, figure and table counts.

## Tool assembly

The ToolBox, ToolSet, MCP Aggregator and Role bodies are groups of rows —
settings, the tool tree, the focused tool's details — with a splitter
handle between each group. Drag a handle to give the tree or the details
the room; the node keeps the height you set, and the pane sizes are saved
with the graph. A group's tall widget (tree, editor, details, hooks)
takes the group's spare height, and Hooks and Details carry their label
above them. The groups run settings, hooks, tools, then details. The
ToolBox, ToolSet and Role trees each have a Category dropdown to show one
category at a time.

Keep the node short and drag it wide, and once the groups fit side by
side at their preferred widths they turn side by side; narrow it again
or make it tall, and they stack.

### Silk ToolBox — `nodes/toolbox.py`
The registry of **all** tools an agent network may use: sandbox roots (hard
ceiling), toolchain packs, per-tool selection and details.

| Direction | Port | Type |
|---|---|---|
| in | `sandbox_roots` | `dirpath_list` |
| in | `toolchains` | `toolchains` |
| in | `mcp` | `mcp_servers` |
| in | `plan` | `silk_plan` |
| in | `embedding_model` | `model_handle` |
| out | `toolbox` | `silk_toolbox` |
| out | `root_paths` | `dirpath_list` |

**Sandbox roots are wired, never typed.** There is no picker on the node:
a `Folder` node for one root, a `Folder List` for several — the
`dirpath` → `dirpath_list` cast wraps the single case on connection, so
both fit the one port. The roots are the hard ceiling of the whole graph,
and a ceiling that can also be set inside the node is a ceiling you
cannot read off the canvas. Unwired, the node builds nothing and says so.

**The tool tree is the only selector.** There used to be a row of group
checkboxes above it — File Read, File Write, Ripgrep, Task Planning and
the rest — and they asked the same question the ticks already answered,
in a way that could disagree with them. They are gone. A group is
attached because something in it is ticked, and that is the whole rule
(`groups_for`). Everything else falls out of the same selection:

* the sandbox is **writable** because a write tool is ticked
  (`WRITING_GROUPS`), not because a separate switch says so;
* **plugin authoring** is on because a `suite_tools` tool is ticked, and
  that is what adds `~/.weave/plugins` to the sandbox as a writable root;
* **graph authoring** mounts because a graph tool is ticked *and* the
  whitelist behind its gear names at least one class;
* **node-dev** promotes into the folder behind its gear (saved in the
  graph). Empty is Weave's user plugin folder. Weave remembers any other
  folder on the first promotion into it and loads it at every start
  (`weave.plugin.remember_root`; `forget_root` stops that). The sign-off
  request names the folder.

**The tree is populated before anything is wired.** It is seeded from a
static catalog (`functions/tool_groups.py`) built once by running every
attacher against a throwaway sandbox and recording what it registered —
nothing is executed, only described. The root is usually the thing being
wired *because* tools are wanted, so a tree that filled in only after the
root arrived was a tree that could not be set up first. `file_read` and
`ripgrep` start ticked; writing, planning, memory, graph authoring and
plugin authoring are each a different request, and start off.

**Unticking is reversible.** The tree lists what is *offered*, which is
the static catalog plus whatever the last build added dynamically — not
what survived the narrowing. Reading it off the finished box is what used
to make an unticked tool vanish from the tree altogether: the box no
longer had it, so the catalog no longer mentioned it, so there was
nothing left to tick back on. For the same reason a group with nothing
ticked keeps its row.

The narrowing is the **last entry of the build recipe**, not something
the node applies to its own output, so every ToolSet derived from this
box replays it. Without that a derived set would come back holding tools
its ToolBox had been told to drop.

A tool the tree has never shown cannot have been unticked, so the node
remembers which ones it has offered (`seen_tools`, saved with the graph).
The static tools are seen at construction; a toolchain pack or an MCP
server's tools are seen when they first arrive, and arrive **ticked** —
otherwise they would be pruned on the same evaluation that created them,
and wiring a toolchain would appear to do nothing.

Toolchain tools are ordinary tools: the packs register on the box like
every other attacher and appear in the same tree under their own
categories (`code`, `lint`, `build`), so they are ticked, filtered and
inherited exactly like the file tools.

**Some groups carry settings, and they sit on the group's own row.** A
category row with a ⚙ opens its configuration on double-click, or from
the row's context menu — the same affordance the Hooks list uses. Only
graph authoring has one today (*Allowed node classes…*), and it is there
rather than in a field of its own precisely because it is not a
preference: it is the grant those tools run under, and a grant belongs
beside the tools it governs.

**Hooks start on.** Every hook in the catalog is ticked by default except
the four that can refuse a call or stop to ask — `tool_approval`,
`signoff`, `tool_budget`, `task_audit` (`GATING_HOOKS`). Observation is
what people wish they had switched on *after* a run, and a hook that was
never ticked leaves nothing to go back to; a gate you did not choose is
how people learn to switch hooks off wholesale. Each default degrades on
its own when a dependency is missing — `remember` without the `ledger`
extra logs that it did not attach rather than failing the build.

Wiring a `Silk Plan` node into `plan` names the plan the task tools work
on. Left unwired, they discover the newest plan under the sandbox root —
which is how several agents share one plan, and why two unrelated plans in
one root used to find each other (D23).

Ticking **`recall`** mounts memory search: keyword search
over the turns and runs remembered in this sandbox root's history ledger,
including ones from earlier sessions and ones compaction dropped (§17,
D66). It needs the `ledger` extra (`pip install macrame-db`); without it
the tool registers and says so rather than quietly returning nothing.

**Placeable nodes** is the graph-authoring grant (§18, D71), and it lives
behind the ⚙ on the tree's `graph` row. Tick the graph tools
(`list_placeable_nodes`, `describe_graph`, `list_node_settings`,
`place_node`, `connect`, `set_node_value`, `disconnect`, `remove_node`)
*and* name at least one node class there, and the pack mounts. Leave the
whitelist empty — the default — and the pack stays out of the prompt
entirely rather than mounting eight tools that refuse everything; the
status line says so, because ticked-but-ungranted is a half-finished
setup rather than a safe one. No agent fed by this ToolBox can build
graph at all until a class is named. Every edit an agent makes
goes onto the canvas's own undo stack, so one Ctrl+Z takes back one tool
call; destructive calls reach only what that run itself placed, and no
mutation may touch the agent, its tool chain, or anything upstream of it
(D72, D73). The list travels in the saved graph and in presets: it carries
no secret and no filesystem authority.

**Plugin authoring** (§19) is the `plugins` category of the tree: ticking
`list_suites`, `load_suite`, `reload_suite` or `request_relaunch` lets
the agent write node suites into `~/.weave/plugins` and load them into
the running session. It adds
that directory to the sandbox as the only writable root (unless file
writing is already on). Every load asks you, every time,
and shows you the diff of what this run wrote — no Role, preset or
grant can pre-approve it, because importing runs that code with the
full authority of the Weave process. The tools mount, and the load verb
is what asks; unlike graph authoring there is no separate grant to name
first, because the answer is given per load and never in advance. A state-version finding
(WV520–WV522) stops the load before you are even asked: it means saved
graphs would not survive it. Weave core and Silk stay read-only.

Which backend stores the plan is the environment's business, not the
node's: `SILK_TASK_BACKEND=ledger` puts it on the Macrame ledger, the
default keeps the SQLite store, and both answer the same protocol — the
Plan Viewer, the Task Hub and the sign-off flow cannot tell which
answered.

### Silk ToolSet — `nodes/toolset.py`
Selects a subset of ToolBox tools for an agent, with optional per-toolset
sandbox permissions and named presets.

| Direction | Port | Type |
|---|---|---|
| in | `toolbox` | `silk_toolbox` |
| in | `permissions` | `file_permissions` |
| out | `toolset` | `silk_toolset` |
| out | `permissions` | `file_permissions` |

**Allow All Tools** takes every tool the connected ToolBox offers, including
tools that appear later (a toolchain wired upstream). The tree shows them
all ticked and locks; your own ticks come back when you untick it. Presets
save it.

The knowledge graph access level sits behind the gear on the tree's
knowledge row (double-click or right-click it), as on the ToolBox; the row
is there while the connected ToolBox offers knowledge tools.

### Silk MCP Server — `nodes/mcp_server.py`
Connects to one MCP server and shares the **live session** with every
downstream agent — one handshake per server, not per agent (D19). The
credential field holds the *name* of an environment variable or an entry in
`~/.weave/silk/secrets.json`; no secret is ever stored in the graph (D22).
Servers chain through `mcp_in` like toolchains, and every tool is prefixed
with the server id (D21).

| Direction | Port | Type |
|---|---|---|
| in | `mcp_in` | `mcp_servers` |
| in | `server_id` | `string` |
| in | `transport` | `string` |
| in | `command` | `string` |
| in | `args` | `string` |
| in | `url` | `string` |
| in | `credential` | `string` |
| out | `mcp` | `mcp_servers` |

### Silk MCP Aggregator — `nodes/mcp_aggregator.py`
Checkbox tree over every server on the wire: a category row is a server, a
leaf is one tool (D20). Unchecking records an exclusion — it never closes a
session, because the sessions belong to the MCP nodes and a toggle should
not cost a handshake.

| Direction | Port | Type |
|---|---|---|
| in | `mcp_in` | `mcp_servers` |
| out | `mcp` | `mcp_servers` |

### Silk Plan — `nodes/task.py` *(Silk AI / Agents)*
Names the plan agents work on, so the store never has to guess which plan a
root means (D23). Lists the plans that already exist under the root with
their goal and open-task count; `(new plan)` plus a name creates one at a
path you can find again, `(newest under root)` keeps shared discovery.

Shown as *Silk Plan*; it was called *Silk Task*, though it makes no task. A
**plan** is a goal, its acceptance criteria and a list of **tasks**, stored
as one `plan-*.db` under a sandbox root; this node says which plan. The
class is still `SilkTaskNode`, which is what saved graphs and the
graph-authoring whitelist refer to.

| Direction | Port | Type |
|---|---|---|
| in | `root` | `string` |
| in | `root_paths` | `dirpath_list` (the folder list that feeds the ToolBox's `sandbox_roots`) |
| in | `plan_choice` | `string` |
| in | `plan_name` | `string` |
| out | `plan` | `silk_plan` |

Wire `plan` to the ToolBox's `plan` and to the Plan Viewer's `plan_ref`.
Feed `root_paths` from the same **Folder List** that feeds the ToolBox's
`sandbox_roots` — not from the ToolBox's `root_paths`. The ToolBox takes
this node's `plan`, so that wire closes a loop, and the engine breaks a loop
with the last cached value: on a first evaluation the plan is named under
no root at all.

### Toolchain — `nodes/toolchain.py`
Configures a set of external toolchains — Python interpreters/venvs, ruff,
mypy, radon, maturin, cargo — as structured tools for the agent.

| Direction | Port | Type |
|---|---|---|
| in | `toolchains` | `toolchains` |
| out | `toolchains` | `toolchains` |

## Agent configuration

### Silk Role — `nodes/role.py`
Declarative agent configuration: persona instructions plus a
**hard-enforced** tool selection, downselected from the connected toolset.

| Direction | Port | Type |
|---|---|---|
| in | `toolset` | `silk_toolset` |
| in | `instructions` | `string` |
| in | `permissions` | `file_permissions` |
| out | `role` | `silk_role` |
| out | `permissions` | `file_permissions` |

**Allow All Tools** grants every tool of the connected toolset, never more:
the toolset is built with only its own tools, so a role cannot reach past
it. The tree shows them all ticked and locks. Your own ticks stay underneath and come back when
you untick it. Max Risk still applies to every tool, Allow All or not.

**Max Risk** is a ceiling, not a selection: tools above it are never
granted, ticked or not. The tree shows them unticked and greyed, with the
reason in the tooltip, and it wins over Allow All. Your ticks on them are
kept and return when the ceiling is raised.

### Inference Settings — `nodes/inference_settings.py`
Builds a `gen_params` dict (sampling / generation parameters) from UI
controls; save and recall configurations as named presets.

| Direction | Port | Type |
|---|---|---|
| out | `gen_params` | `dict` |

## Execution

### Silk Prompt — `nodes/prompt.py` *(Silk AI / Agents)*
The chat box for an agent or orchestrator. Three icon buttons: Send, Stop,
New chat. Send and Stop are split buttons; their arrows hold the other
modes.

| Key | Sends as | What happens on a running agent |
|---|---|---|
| Enter | queue | waits; the next run's prompt |
| Ctrl+Enter | steer | joins this run at its next step |
| Ctrl+Shift+Enter | interrupt | cuts the current action, carries on with it |

Shift+Enter starts a new line; Ctrl+Up / Ctrl+Down step through earlier
messages (kept with the graph). `@name message` goes to the running worker
`name` of an Orchestrator. A queued message travels as the plain text, a
steer or an interrupt as `{"text", "mode"}`.

Stop is press-twice: the first press stops after the current action; for
a few seconds the button then pulses red, and a second press stops now.

| Direction | Port | Type |
|---|---|---|
| out | `chat` | `exec` → the agent's `run` (payload: the message, Stop or New chat) |
| out | `prompt` | `string` (the last message sent) |

**One wire.** Send, Stop and New chat all travel on `chat`, told apart by
the payload (`functions/run_control.py`): a message is the text or
`{"text", "mode"}`, Stop is `{"action": "stop", "force": bool}`, New chat
`{"action": "clear"}`. A graph saved with the old `send` / `stop` /
`new_chat` outputs (and the agent's `stop` / `clear` inputs) loads with
those wires merged onto `chat → run` (`node_renamed_ports`).

The message rides the pulse, so `chat → run` is the whole wiring.
It does not go over a data wire because a changed input stops a running
node; this way a message sent mid-run is queued as the next one. `prompt`
is there for anything else that wants the text — wiring it into the
agent's `user_prompt` too works while idle, but then a send during a run
stops that run.

```
[Silk Prompt] ─chat──────▶ [Silk Agent] .run
[Silk Agent]  ─events────▶ [Chat Log Display] .event
```

**A question from the agent shows under the box.** When an agent the
`chat` wire reaches blocks on a decision, the Prompt node shows it below
the text box — the classifier's advice when one is wired, a reason field,
and the same four answers — and answers through the agent's own handler,
like the Decision Inbox does (`widgets/decision_strip.py`). Hidden while
nothing waits.

### Silk Agent — `nodes/agent.py`
The autonomous tool-calling agent: wires model + toolset + role into the
Qt-free `AgentLoop`. Exec `run`/`done` ports let agents chain into networks.
This is the node that **runs an agent where it sits**; to describe an agent
for an Orchestrator to delegate to instead, use **Silk Worker** below.

| Direction | Port | Type |
|---|---|---|
| in | `model_obj` | `model_handle` |
| in | `toolset` | `silk_toolset` |
| in | `role` | `silk_role` |
| in | `system_prompt` | `string` |
| in | `user_prompt` | `string` |
| in | `inbox` | `agent_message` |
| in | `run` | `exec` (messages, Stop and New chat) |
| in | `permissions` | `file_permissions` |
| in | `budget` | `string` |
| in | `classifier` | `silk_classifier` |
| out | `response` | `string` |
| out | `outbox` | `agent_message` |
| out | `events` | `dict` |
| out | `done` | `exec` |

`events` is the one typed stream (spec D2/D3): run lifecycle, model rounds,
tool calls and results, denials, plan snapshots, chat turns and decisions,
each carrying `type`, `ts`, `run_id`, `seq` and the agent identity.
Consumers filter by `type`.

**The node is controls, not a chat window.** One icon row: Run/Stop (one
toggle) and Clear context, next to the status. The prompt arrives from
outside — a **Silk Prompt** node, or any text on `user_prompt` — and the
reply is read downstream (`events` → Chat Log Display, or `response`).
The `run` input carries all three buttons, by payload:

- A message (or a bare pulse) starts a run. The pulse may carry the
  message as its payload (the Prompt node's `chat` does); it wins over
  `user_prompt` for that run. During a run the message is filed by its
  mode (below) instead of stopping the run; a bare `run` pulse runs again
  after it.
- `{"action": "stop"}` stops the run, press-twice like the button: the
  first after the current action, a second (or `"force": true`) now.

**What waits on the run** is listed under the icon row, hidden while
empty. Each message has its mode as an icon — click to re-file it — and a
remove button. The modes (`functions/run_control.py`):

| Mode | When it reaches the run | What it cuts |
|---|---|---|
| queue | when the run ends: the next run's prompt (several are merged) | nothing |
| steer | at the next step — after the tool batch, or instead of ending on an answer | nothing |
| interrupt | now | the model stream at the next token; tools not yet started. A tool already running finishes; its result is kept |

The agent's own button is Start → Stop → (pulsing red) Force stop; its
arrow has both stops. A soft stop finishes the current action, takes no
next one, and ends the run `stopped`. A stop holds the queue rather than
dropping it: the messages stay listed and Start sends them, ahead of any
new one. Steers that missed the run are filed as queued.
- `{"action": "clear"}` forgets the conversation (the pool session and the history),
  then emits `context.cleared` on `events`, which starts a wired Chat Log
  over. During a run it waits for the run to end. The plan, the knowledge
  graph and always-allow grants are not conversation and are kept.

**File access is a port, not a hidden handle** (spec D16-D18). Whatever
reaches `permissions` — wired straight here, or inherited down the
ToolSet → Role → Agent chain — narrows the toolset's sandbox for this run
and only this run, and is restored afterwards. The two sources compose by
narrowing, so adding a wire can only reduce access (I6); a malformed grant
grants nothing rather than falling back to the wider one.

**A blocked agent also shows on the canvas and in the Decision Inbox**
(D59): the node switches its pulse to a heartbeat while waiting, and
`DecisionInboxDock.attach(main_window, pinned=True)` puts a dock in the
window's right sidebar listing every waiting agent with the same four
answers, the classifier's advice and a reason field (a Lace dock: the host
window needs its `dock_manager`, which a Weave host sets up). The example
apps attach it whenever Silk is loaded. It cannot be closed — it moves,
floats and pins — so a waiting agent has nowhere to hide. Answering there
calls this node's own handler — the dock owns no seam.

**A reason is kept.** The card, the inbox and the Prompt strip each take an
optional reason with the answer. It reaches the model with a denial, is
written to the decision log (`~/.weave/silk/decisions.jsonl`, or
`$SILK_DECISION_LOG`), and is read back as `prior_decisions` the next time
the classifier triages that tool; `findings_gate` can file it into a wired
Knowledge Graph as a rule, dogma, evidence or decision.

**The approval prompt is part of this node** (spec D48/I12). When a gated
tool call blocks, the question appears in the node itself — Deny, Allow
once, Allow this run, Always allow — and the held call resumes with the
answer. It is not a separate node because the run is inside `compute()`,
where no graph channel can reach it; the `decision.request` /
`decision.response` pair on `events` is a *mirror* for a monitor, never
the way the answer arrives (D59). Stop cancels the seam directly, so a run
waiting on a decision unwinds at once instead of waiting out the timeout.

**A placed node can also be configured** (§22 q9). `list_node_settings`
reads a node's widgets — name, type, current value, and why a port is not
writable — and `set_node_value` writes one, as one undo step. Values only
(text, number, boolean): a model, a toolset or a permissions object comes
from a connection. Display widgets are readable but never writable;
internal ones — a String Input's text box, an Int Input's number — are
writable unless the node grants authority (any Silk node, or one that
outputs tools, permissions, roots, a model or a plan), which is what keeps
a `Plugin authoring` checkbox out of an agent's reach.

**Which nodes an agent may edit is the node's Agent access** (right-click
a node, **Agent access**). A node an agent placed stays that agent's in
full, across runs and saves; any other node gives agents no access until
you pick a level: *Connect* (wire into its inputs), *Configure* (also
change its settings), *Full* (also remove it). Wiring *from* a node's
outputs never needs access. A level you pick applies to every agent and
beats ownership, so *No access* takes back a node an agent placed;
*Default* returns it to the owner. Each change is one undo step. Nothing
opens the agent's own path — itself, and whatever feeds it.

**An approved plugin comes back by itself, until it is edited** (§22 q10).
Approving a `load_suite` pins the SHA-256 of every importable file in that
suite, and the next start loads it without asking while the digests match.
Any edit, and the next load asks again with the diff; a suite that crashed
a start loses its pin entirely and must be approved by hand (§22 q11). The
pins are listed and revocable in the same dock as the grants.

**"Always allow" can be taken back** (§22 q1). A durable grant lives in
`~/.weave/silk/grants.json`, keyed by resolved project root, and
`GrantManagerDock.attach(main_window)` opens a dock that lists every one of
them with Revoke per grant and Revoke all per project (a Lace dock, like the
Decision Inbox, so the host window needs its `dock_manager`). It re-reads the file
each refresh, so a revocation takes effect on the next gated call in every
window; it can only remove, never grant. Run-scoped grants are not listed —
they end with the run.

**A priced run leaves a line in the ledger** (D90). When the endpoint
quoted a price, the run appends one line to `~/.weave/silk/spend.jsonl`
— when, how long, which models, tokens each way, how much, and how it
ended — attributed *per model*, so a run that fell back (D89) says which
model owns which part of the bill. There is no checkbox: a run against a
model that quotes nothing writes nothing, so a purely local graph never
acquires the file. Content never appears in it, a failed run is recorded
too (it was still prefilled), and a ledger that cannot be written is
logged once and ignored rather than failing the run.

`functions/cost_ledger.summarize(days=7)` answers "what did this week
cost me", with the per-model breakdown that makes the number actionable:
a total says you spent eleven dollars, the breakdown says nine of them
went to one model you could have put a cheaper one behind.

### Silk Worker — `nodes/worker.py`
Describes a named worker (model + toolset + role) that a Silk Orchestrator
can delegate to. Chain several to build a `silk_workers` roster.

| Direction | Port | Type |
|---|---|---|
| in | `model_obj` | `model_handle` |
| in | `toolset` | `silk_toolset` |
| in | `role` | `silk_role` |
| in | `description` | `string` |
| in | `budget` | `string` |
| in | `workers_in` | `silk_workers` |
| out | `workers` | `silk_workers` |

**This node runs nothing on its own.** Its `workers` output must reach a
Silk Orchestrator's `workers` input; nothing else consumes the type, so a
Worker with no Orchestrator downstream does exactly nothing.

That is the whole difference from the **Silk Agent** node above, which runs
where it sits — hence `run`/`done` and a `response` output, none of which a
Worker has. What a Worker has instead is a **name** (an orchestrator's model
addresses it by name, not by wire) and a **speciality** (advertised through
`list_workers` so the model can choose it). Neither means anything for an
Agent node.

It could not be a node that runs, either: `delegate` spawns workers inside
a single tool call, on a worker thread, possibly several at once, so a
worker has to be data the orchestrator carries rather than a box the engine
schedules.

*Renamed from "Silk Agent Spec"* (class `SilkAgentSpecNode`, ports
`agents_in`/`agents`, type `silk_agents`), which read like configuration
for an Agent node — the one thing it is not. Saved graphs using the old
node are not migrated.

### Silk Orchestrator — `nodes/orchestrator.py`
A Silk Agent that delegates self-contained sub-tasks to a roster of worker
agents (`delegate` / `delegate_parallel`). Takes the standard agent inputs
plus:

| Direction | Port | Type |
|---|---|---|
| in | `workers` | `silk_workers` |
| in | `max_depth` | `int` (spin box, default 2) |

`max_depth` is how deep delegation may nest: `1` lets the orchestrator call
workers but stops a worker from sub-delegating, `2` allows one further hop.
Cycles are refused at any depth. An upstream connection overrides the spin
box.

Clear context clears the orchestrator's own conversation only: a worker
runs in a fresh, isolated context on every delegation, so it has none to
clear.

**Workers are steered and stopped like the orchestrator.** Every
delegation runs under a child of the orchestrator's run control, so while
a worker runs it is listed under the icon row (`⇢ name`), with its own
press-twice Stop and its own waiting messages. Send `@name message` from a
Prompt node to reach it: a plain send steers it (a worker has no next run
to queue for), Ctrl+Shift+Enter interrupts it. A stop on the orchestrator
reaches every worker below it, soft or forced.

## Observability & human gates

### Hook Monitor — `nodes/hook_monitor.py` *(Silk AI / Display)*
Graph-native observability sink for the Agent's `events` stream: rolling log
of everything a run says, with per-type and per-tool counters.

| Direction | Port | Type |
|---|---|---|
| in | `event` | `dict` |
| out | `counts` | `dict` |

### Plan Viewer — `nodes/plan_viewer.py` *(Silk AI / Display)*
Display **and** graph-composition surface for the agent task tracker: shows
the current plan (goal, task tree, course corrections) rendered in the app's
markdown style.

| Direction | Port | Type |
|---|---|---|
| in | `root` | `string` |
| in | `plan_ref` | `silk_plan` |
| in | `plan` | `dict` |
| in | `event` | `dict` |
| out | `plan_json` | `dict` |
| out | `plan_text` | `string` |
| out | `plan_html` | `string` |

Sources are tried in order: an explicit `plan` snapshot, then `plan_ref`,
then `root`. The reference outranks the root because a root only says
*where* to look, and looking picks the newest plan there.

No input is required; one source plus `event` is the usual wiring:

- **A named plan:** `Silk Plan.plan` → `plan_ref`. The reference carries
  its root, so `root` stays unwired.
- **No Silk Plan node:** the folder path → `root`; the newest plan there
  is shown.
- **Live updates:** an Agent's `events` → `event`. Only `plan.summary`
  events are read, and their snapshot is shown as is — it outranks
  `plan_ref` — so give each agent its own viewer, or use the Task Hub for
  several.
- `plan` is for a snapshot you already hold; most graphs leave it unwired.

### Task Hub — `nodes/task_hub.py` *(Silk AI / Display)*
The multi-agent progress board (D58). Scans **every** `plan-*.db` under the
graph's sandbox roots and renders one section per plan, tasks grouped by
lane, with `claimed_by` as the per-task agent badge — the field the store
has always recorded and no view has ever shown.

| Direction | Port | Type |
|---|---|---|
| in | `roots` | `dirpath_list` (wire `Silk ToolBox.root_paths`) |
| in | `event` | `dict` (any agent's `events`; counted, never answered) |
| in | `refresh` | `exec` (a timer pulse or any agent's `done`) |
| out | `plans_json` | `dict` |
| out | `pending` | `int` |

`pending` is how many agents are blocked on a decision right now. That is
all `event` is for: it reads `decision.request` / `decision.response` and
the end of a run, and does **not** redraw the board. The board re-reads the
plan files when `refresh` pulses or `roots` changes, so wire an agent's
`done` (or a timer) to `refresh` to see it move. The ToolBox's `root_paths`
is the right source for `roots` here: the hub feeds nothing back, so there
is no loop. The hub may **count** decisions; only the asking node — or its
dock mirror — may answer one (D59). There are no Approve/Reject buttons here: D31–D33 deleted parked
sign-off, so a task change is decided during the turn, not held in a row.

### Chat Log Display — `nodes/chat_display.py` *(Silk AI / Display)*
Sink that continuously appends chat turns to a running log, rendering the
thread as markdown/HTML.

| Direction | Port | Type |
|---|---|---|
| in | `event` | `dict` | (the Agent's `events` stream; keeps `chat.turn`) |

A `context.cleared` event (the agent's Clear context, or a Prompt node's
New chat) empties the log: a new chat starts on a new page. A steer or
an interrupt that joined a turn shows inside it, as a user line at the
point the run took it (`run.steered` on `events`).

### Pool Monitor — `nodes/pool_monitor.py` *(Silk AI / Display)*
Live snapshot of GGUF pool state (active/idle instances, capacity). Wire any
agent's `done` port to `refresh` for updates without polling.

Two rows are worth reading together, because the second exists for the
first:

- **Prefix reuse** — `reuse · contention · prefill · n measured` (D41,
  D47). Before any request it says *no requests measured yet* rather than
  0%, because those lead to opposite decisions. Reuse at 0% in every
  shape usually means the backend refuses partial KV removal; see
  [docs/prefix_reuse_measurement.md](prefix_reuse_measurement.md).
- **Affinity** — what the request queue is doing about it: the share of
  requests that followed one from the same conversation, the average wait
  that grouping cost, and how many hold windows paid off against how many
  elapsed unused. If the hold stops paying it suppresses itself and says
  so.

| Direction | Port | Type |
|---|---|---|
| in | `model_obj` | `model_handle` |
| in | `refresh` | `exec` |
| out | `pool_status` | `dict` |

## Example graph

```
[Silk MCP Server] ─mcp─▶ [Silk MCP Aggregator] ─mcp─┐
                                                    ▼
[Silk Plan] ─plan──────────────────────────────────┐
                                                    ▼
[Toolchain] ─toolchains─▶ [Silk ToolBox] ◀──sandbox_roots ─ [Folder List]
                              │ silk_toolbox
                              ▼
                         [Silk ToolSet] ◀──permissions (optional)
                              │ silk_toolset
                     ┌────────┴────────┐
                     ▼                 ▼
              [Silk Role]         [Silk Worker] ─agents─▶ [Silk Orchestrator]
                     │ silk_role              ▲ (worker bundles)
                     ▼                        │
  [GGUF Loader] ─model_handle─▶ [Silk Agent] ───┘
       │                        │
       └─ pool_info             ├─ events ─┬─▶ [Hook Monitor]
                                │          ├─▶ [Chat Log Display]
                                │          ├─▶ [Task Hub] ◀─root_paths─ [Silk ToolBox]
                                │          └─▶ [Plan Viewer]
                                └─ done (exec) ─▶ [Pool Monitor] .refresh
                                                  [Task Hub] .refresh
```

The plan wiring on its own, because the obvious version is a loop:

```
[Folder List] ─folders─┬─▶ [Silk ToolBox] .sandbox_roots
                       └─▶ [Silk Plan] .root_paths      (not ToolBox.root_paths)
[Silk Plan] ─plan──────┬─▶ [Silk ToolBox] .plan
                       └─▶ [Plan Viewer] .plan_ref
[Silk Agent] ─events───┬─▶ [Plan Viewer] .event        (live redraw)
                       └─▶ [Task Hub] .event           (pending count only)
[Silk Agent] ─done─────▶ [Task Hub] .refresh           (redraws the board)
```
