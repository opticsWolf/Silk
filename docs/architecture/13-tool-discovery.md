## Tool discovery and search

`functions/tool_search.py` — **`ToolSearch`** backs deferred tool discovery
for `DeferredCapability`. It indexes registered tools and capabilities and
supports pluggable strategies:

- `keywords` (default) — local keyword-overlap ranking.
- `bm25` — BM25-based ranking.
- `regex` — regex match.
- a **custom callable** `(queries, tools) -> names`.

`max_results` caps output; `search_capabilities` returns matching
capabilities (so a deferred capability can be surfaced to the model as a
candidate to load).

