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

### `search_tools` — the model-facing half (D4–D6)

`functions/tool_discovery.py`. The index above existed for a long time
without anything model-facing calling it: the only discovery tool an agent
could see was `load_capability`, which takes an id and no query — you can
only load what you already know the name of.

**One tool, not two (D5).** `search_tools(query, category=None,
capability=None, limit=6)` returns *individual* tools, ranked, each hit
carrying `name`, `description`, the full `parameters` schema, and its
`category` / `tags` / `risk` / `capability` where those exist. The schema is
what makes one tool enough: the model can call what it just found, so there
is no companion load tool and the always-present surface stays one tool
wide. `build_system_prompt` adds a one-line `[TOOL DISCOVERY]` section
saying so, because the tools an agent was *not* given are exactly the ones
it cannot see, and a missing tool otherwise reads as a missing capability.

**Discovery obeys the role gate (I8).** `ToolSearch.permits` is the filter,
and the ToolBox installs its own `role_permits` into it at construction. It
has to live in `ToolSearch` rather than in the discovery module, because
every path into the index must obey it — including `search_capabilities`,
which drops a capability whose every tool the active role forbids. A
capability declaring no tools at all is instructions or hooks and stays.
Custom strategies are gated on their *answer*: they may return whatever
they like, so trusting them with the corpus is not the same as trusting
their result.

**Per-tool deferral (D6).** `ToolBox.defer_tools(names)` marks a tool
*registered and dispatchable but not advertised*: `get_tool_schemas` skips
it, `search_tools` still finds it, and the role gate treats it exactly like
any other tool. Deferral is about prompt space, never about permission.
`undefer_tools` reverses it — deliberately, and never mid-run.

**Auto-load at dispatch (D6).** `register_capability` indexes a deferred
capability's tools *by name* without loading them, so they are findable
before anything has loaded them; `_tool_capabilities` records which
capability provides which name. When the model calls a tool that is
indexed but not registered, `execute_tool_calls_async` calls
`tool_discovery.autoload`, which loads the owning capability and lets the
call proceed. The role gate runs after, on the loaded tool, so auto-loading
can only ever save a round trip — it can never widen what an agent may do.
A load that fails comes back as a structured `autoload_failed` error naming
the way forward.

**Loading does not re-advertise (I11).** An auto-loaded tool joins
`deferred_tools` rather than the advertisement. If loading re-advertised,
the schema block at the head of the prompt would change mid-run and the KV
prefix would be invalid for every remaining round — a full prefill (D41).
The model does not need the advertisement; it has the schema from its
search. The spec permits either this or a deliberate, counted invalidation,
and forbids only silent recomposition. `load_capability` remains the
explicit path that does re-advertise, because the model asked for it.

### Ranking

`bm25` is Okapi BM25 over tool names and descriptions, with `k1 = 1.5` and
`b = 0.75` (closes **G2**, which had it aliased to `keywords` behind a
TODO). Names are tokenized as identifiers — `read_file` becomes `read`,
`file` — because the name carries most of the signal. What the ranking buys
over keyword overlap is the idf factor: in a tool corpus, words like "file"
or "list" appear in half the descriptions and should count for almost
nothing, while a term appearing in one tool should decide the ordering.
`keywords` remains the default strategy and the no-index fallback.

