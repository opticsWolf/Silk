# Silk — Codebase Deep Dive

*2026-09-09 (expanded same-day revision). Scope: the whole tree as of `d8e3bbb` on `dev` —
`functions/` (+ `tools/`), `nodes/`, `widgets/`, `__init__.py` — 106 files, ~35,900 LOC.
Method: static graph index (CodeRadar: 274 classes, 1,520 functions, 3,116 call edges),
smell/scaffolding/dead-code scans, then **manual verification of every load-bearing claim**
— full reads of the core machinery (`agent_loop.py`, `approval.py`, `decision_seam.py`,
`blocking_seam.py`, `tool_box.py` dispatch path, `model_pool.py`, `self_modify.py`),
AST cross-passes, import-graph analysis, `ruff check`, docs↔code reconciliation.
Complements [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) (Weave-fit review,
2026-08-31): that one asks "does Silk use Weave correctly"; this one asks "what is the
code, how is it built, and where does it hurt".*

---

# Part I — Shape

## 1. What Silk is

A local-first agentic runtime exposed as Weave canvas nodes. The repo root **is** the
`silk` package, linked into a Weave checkout as a git submodule at `weave/plugins/silk`.
It embeds GGUF local-LLM agents into a visual graph: a llama.cpp model pool (one shared
`llama_cpp.server` subprocess), a sandboxed tool system with hard role enforcement, an
autonomous tool-calling loop, multi-agent orchestration with budget nesting, a
task/sign-off system with human approval gates, MCP client sessions, graph authoring by
the agent, and a bitemporal task/history ledger.

Importing the package registers all nodes: `__init__.py` → `import_node_tree(__name__)`
→ `NODE_REGISTRY`, then runs `log_contract()` (the Weave-internals contract check) and
logs `version_string()` once.

## 2. The tree

| Layer | LOC | Qt? | Character |
|---|---|---|---|
| `functions/` (runtime) | 22,874 | **no** (verified: zero PySide6 imports) | the agent runtime |
| `functions/tools/` | 4,858 | no | sandboxed file/search/task/toolchain tools |
| `nodes/` (Weave wrappers) | 6,183 | yes | thin shells over the runtime |
| `widgets/` (Qt docks) | 1,952 | yes | decision inbox, grant manager, node whitelist |

Largest files: `ledger.py` 1,434 · `task_store.py` 1,216 · `mcp_toolset.py` 1,186 ·
`capabilities.py` 1,111 · `tool_box.py` 1,103 · `nodes/agent.py` 1,098 · `hooks.py`
1,038 · `hook_catalog.py` 862 · `toolset.py` 781 · `tools/file_read.py` 758 ·
`stream_events.py` 754 · `agent_loop.py` 661 · `gguf_loader.py` 638 · `model_pool.py`
620 · `self_modify.py` 612.

Note on recency: the 2026-09 pull (63 commits over five days, +20.5k/−1.7k lines, 39 of
them on 2026-09-02 alone) built most of the heaviest surface — `ledger.py`,
`compaction.py`, `approval.py`+`decision_*`, `mcp_*`, `self_modify.py`+`graph_author.py`,
`grants.py`/`file_grants.py`, `prefix_*`. Much of this review looks at days-old code.

## 3. Node roster (17, matching NODES.md 17 sections exactly)

`GGUFLNode` (loader) · `SilkToolBoxNode` · `SilkToolSetNode` · `SilkMCPServerNode` ·
`SilkMCPAggregatorNode` · `SilkTaskNode` · `ToolchainNode` · `SilkRoleNode` ·
`AgentInferenceSettingsNode` · `SilkAgentNode` · `SilkAgentSpecNode` ·
`SilkOrchestratorNode` · `SilkHookMonitorNode` · `SilkPlanViewerNode` ·
`SilkTaskHubNode` · `ChatDisplayNode` · `PoolMonitorNode`.

Naming caveat: `SilkMCPServerNode` **hosts no socket** — it owns one *client session* to
one external MCP server and shares it downstream ("one handshake per server, not per
agent"). The name reads as if Silk serves MCP; it connects out.

## 4. Core execution flow

```
SilkAgentNode.compute()        (nodes/agent.py — wiring hub, 454 LOC)
  └─ AgentLoop.run()          sync generator, yields typed events
       ├─ select_transport()  native tool_calls or ```tool_call fences
       ├─ bind_stop()         toolbox can ask "did the user stop?" (G8)
       └─ _run_rounds()
            ├─ _maybe_compact(REASON_PRESSURE)        context pressure seam (D24)
            ├─ usage_limits.check_request() / check_input_tokens()   (G7: separate gates)
            ├─ engine.stream_response()               one model request
            │    └─ truncation detection (missing finish_reason, D43)
            ├─ transport.extract_calls()              fences or native tool_calls
            ├─ usage_limits.reserve_tool_calls(len)   claim whole batch (D52.4)
            └─ asyncio.run(toolbox.execute_tool_calls_async(calls))   fresh loop per batch
                 ├─ autoload()                        load-then-run, never widen (D6)
                 ├─ role_permits()                    hard gate at dispatch
                 ├─ _validate_args()                  pydantic; wrap_tool_validate middleware
                 ├─ sequential/parallel split; per-call stop checks (G8)
                 └─ _safe_execute() → wrap_tool_execute chain
                      ├─ approval_floor (D82, fail-closed)
                      ├─ approval gate (D30/D31, outermost by force, I10)
                      ├─ change tracking (D77), spill (§22), remember (§17)
                      └─ innermost: asyncio.wait_for + to_thread for sync tools
```

Every failure is a structured, model-visible result; tools never raise across the loop
boundary (invariant I1, verified at `tool_box.py::_safe_execute` and the loop's
`_args_as_dict`/`_correct_schema` handling).

# Part II — Subsystem deep dives

## 5. The loop (`agent_loop.py`, 661 LOC)

A synchronous generator, deliberately not a coroutine; each tool batch gets its own
short-lived `asyncio.run` (thread-model doc §16, verified). Verified behaviours:

- **Stop semantics (G8):** `bind_stop`/unbind in `finally`; a stop mid-round with work
  left yields `OUTCOME_STOPPED`; a completed answer is never relabelled stopped.
- **Budget stops (G7/G13):** the four gates (request / input tokens / output tokens /
  tool calls) are separate `try` blocks so `limit_type` is exact; every budget exit still
  yields `EventRunResult` with `outcome=usage_limited` and carries the text produced so
  far — the fix G13 documents, present in code.
- **Compaction seam (D24/D25):** pressure compaction before each request; overflow
  compaction (forced, at most once per run via `_overflow_compacted`) only on a
  *classified* overflow; a raising compactor degrades to no-compactor behaviour.
- **Model failures (D40):** `classify_model_error` → `EventError.kind`; only overflow is
  retried; a dead server ends the run rather than spending a prefill failing twice.
- **Truncation (D43):** a stream ending without `finish_reason` fails the round with an
  actionable message (shared-server `interrupt_requests` warning) — and the pool turns
  that flag off at source (§8).
- **Reflection:** at most ONE consolidated retry nudge per round (fan-out cannot spam the
  transcript or burn the retry budget), gated by a run-wide retry counter; nudges carry
  `correct_schema` extracted from structured errors.

Assessment: the highest-value 300 lines in the repo. Complexity is high (CC 26) because
recovery, budget, compaction and stop semantics genuinely meet here; the helpers
(`_recover_or_stop`, `_stopped_by_budget`, `_maybe_compact`, `_model_failed`) are the
right seams and keep the main body readable.

## 6. Tool dispatch (`tool_box.py`)

`execute_tool_calls_async` (149 LOC, CC 15): validate → role-gate → partition
sequential/parallel → gather → per-call stop checks for sequential. Verified details:

- **Autoload (D6):** a discovered-but-unloaded tool is loaded and run in the same
  dispatch rather than costing a round trip; the role gate runs *after* load, so this
  cannot widen permission.
- **Role boundary is hard:** denied calls are refused at dispatch even if the model
  hallucinated a fence for an unadvertised tool (I4: invisible at advertisement *and*
  refused at dispatch — both halves present).
- **Validation:** pydantic `model_validate_json` one-step; `wrap_tool_validate`
  middleware can repair or refuse; refusals are results, not crashes.
- **requires_approval fail-closed (D82/G1):** a flagged tool in a box with no floor does
  not run — the old G1 TODO site, now a real check plus the `approval_floor` middleware.
- **Timeouts:** both sync (`asyncio.to_thread` + `wait_for`) and async paths bounded.

The `ToolSet` layer (`toolset.py`) is a composable decorator family — 12 wrapper classes
(`Filtered`, `Prefixed`, `Renamed`, `ApprovalRequired`, `DeferredLoading`,
`IncludeReturnSchemas`, …) each implementing the same 8-method interface. That is
structural duplication *by design* (uniform wrappers over one protocol); it is also what
the clone scanner would flag if it didn't time out. `build_toolset` rebuilds an
independent ToolBox per agent rather than view-wrapping a shared one (design rule) —
workers never fight over one `RoleBinding`.

## 7. Approval & the decision machinery (`approval.py`, `decision_seam.py`, `blocking_seam.py`, `decision_registry.py`, `grants.py`)

The human-in-the-loop path, verified end-to-end:

- **One gate, two domains (D30/D31):** task changes (`add/complete/complete_final/rescope/goal`)
  and tool calls (names or risk bands) resolve in one `wrap_tool_execute` middleware;
  where both apply, **stricter wins**. `complete_final` is derived from the store
  (a completion that closes the plan is its own change type).
- **Monotonic guard (I10):** `make_outermost` forces the gate to position zero and
  re-forces on later registration — a middleware registered ahead could otherwise answer
  a call the gate never sees.
- **Order of consultation:** run grant → durable grant → policy → human. Grants only
  *skip* a question, never create one.
- **Late-bound seam (D38):** policy snapshotted at attach; the *who to ask* binds per run
  (`bind_run_seam`, context-manager `run_seam`); unbound ⇒ deny-with-count (D36), and the
  first headless refusal logs once, the rest are counted (`headless_refusals`, q1d).
- **The waiter (`blocking_seam.py`):** the general mechanism for both seams (human answer
  D49; main-thread canvas D70). The ordering rule — *commit outcome under lock, then set
  the event; waiter re-reads under lock* — makes Stop / timeout / answer / no-answerer /
  transport-error five distinguishable wakeups. Per-request `threading.Event`s (no
  shared broadcast), idempotent `commit` (D42's fifth race), `cancel()` called *directly*
  by the node's stop handler because nothing polls the loop while a worker blocks (D38).
- **`DriveGate`** checkpoints (`ask/wait/resolve/wake`) exist purely to make the race
  catalog deterministic in tests — testability designed in, not hoped for.
- **Grants:** durable `GrantStore` (JSON, per-project, allow-only, revocation = deletion)
  and run-scoped `RunGrants`; "don't ask again" from a prompt becomes the right scope.
- **Fail-closed everywhere:** no path through the seam produces consent nobody gave.

Assessment: this is the hardest subsystem in the repo (threading + Qt + protocol design)
and it is handled with unusual rigour. The 31-CC `attach_approval_gate` is big because it
is a closure over two policy domains plus the refusal/grant/floor interactions; the
complexity is *domain* complexity, not accidental.

## 8. Model layer (`model_pool.py`, `gguf_loader.py`, `graph_engine.py`)

Verified:

- **One shared `llama_cpp.server` subprocess**, configured via a generated JSON config
  file (no CLI flag guessing), bound to `127.0.0.1` on an ephemeral port, stderr captured
  to a file that doubles as the **prefix-reuse meter** (D41/D47: `PrefixMeter` + `LogDrain`
  read the backend's own cache reports).
- **`interrupt_requests: False`** with the D43 rationale in-line: the server's default
  truncates an in-flight stream with a well-formed `[DONE]` when a second request arrives,
  which is indistinguishable from a clean finish; the GraphEngine's missing-finish-reason
  check stays anyway (a remote backend can truncate for its own reasons).
- **`OpenAICompatClient`** (ex-`OpenAIClientMock`, renamed — G11's fix) is the whole
  client surface: `create_chat_completion` (+SSE generator that owns the response
  lifecycle), `tokenize` (~4 chars/token approximation, honest), `reset`. Credential is a
  *name* (D22), resolved at connect, held only in memory.
- **Serialization made visible (§22 q1c):** one server ⇒ fan-out is correct-but-serial;
  `_flight_begin/end` counts queueing, logs once, then counts silently.
- **Shutdown registry:** the pool registers itself with Weave's shutdown registry so a
  quit or node ejection frees VRAM even if no node calls `cleanup` (with force-kill tier).
- **Sessions:** `_bound_sessions` is a *set* (a counter once measured requests-ever and
  lied); `release_session` is the public "Clear Context" (G17's fix, documented in-line).
- **GraphEngine** wraps the client: stop flags, native-tools arming, prefix-metered
  requests, `sibling()` for workers, `replace_history_prefix` for compaction (D24/D25).

G6 (pool has no recovery when the server dies) is the one remaining open gap here, and
the tracker says so honestly.

## 9. Self-modification & graph authoring (`self_modify.py`, `graph_author.py`, `import_reach.py`, `suite_pins.py`, `nodes/graph_canvas.py`)

The agent's write-verb safety chain, verified against the D76–D81 rules:

1. **Write scope (D76):** agent-authored suites live only in `~/.weave/plugins/`;
   `check_suite` refuses shipped/builtin suites with an explanation the model can act on.
2. **Load is always approved (D77):** `ALWAYS_APPROVE` is a floor no preset/grant can
   lower (distinct from I6 ceilings); the approval request carries the **diff of this
   run's writes** (`ChangeSet` snapshots before/after around each write-tool middleware,
   256 KB/400-line/40-file caps so a human reads rather than scrolls) plus a file listing
   with sizes/mtimes and a `why` that states the actual danger ("import runs with full
   process authority; the sandbox that constrained writing it does not apply to
   importing it").
3. **The linter is the code review (D78):** `weave_lint --format json` in a subprocess,
   `WV520-522` (state-versioning) are hard stops, and a check that cannot run fails
   **closed** (`ran=False` → refuse).
4. **Quarantine feedback (D81):** a suite that crashed a start gets a quarantine fact
   written where the next run reads it, its human pin withdrawn (`suite_pins` pins are
   SHA-256 digests of the exact bytes approved — "approving pins these exact bytes"), and
   a task added to the plan so a human sees it.
5. **Import reach reported (G21):** `import_reach.py` names writable roots that sit on
   the import path — the honest residue: a sandbox root on the import path is deferred
   process authority, redeemable with one `load_suite`. Reported, not silent.
6. **Graph edits (D71–D73):** whitelist (`narrowed()` intersection, I6), `RunScope`
   (may only take apart what this run placed), `check_self_modification` refuses edits to
   the agent's own upstream execution path, and canvas mutations go through
   `CanvasAuthor` (worker → main-thread `MainThreadCall` → undoable commands, q9).

## 10. Task system & ledger (`task_store.py`, `ledger.py`, `signoff.py`, `task_board.py`, `plan_discovery.py`)

- `Plan/Task/Goal/Deviation` dataclasses with canonical JSON snapshots
  (`plan_to_json`/`plan_from_json`) so viewers render plans received on ports.
- **Backends:** `SqliteTaskStore` (always) and `TaskLedger` (macrame-db, optional
  `ledger` extra) behind `open_task_store`; `plan_discovery` finds plans without knowing
  which backend wrote them (T4).
- **The ledger is append-only** (`_put` supersedes, never rewrites — Doctrine III) and
  `load(as_of)` answers "what did the plan look like at 14:00" as one read. Revision
  commits are read-check-assert under a decision lock (D64); `claim_task` is the compound
  decision the lock exists for.
- **HistoryLedger** records runs/turns/compactions as concepts with identity edges and
  answers `recall` (hybrid keyword+embedding search over the ledger's memory, with `_via`
  saying which arm found each hit).
- **Sign-off:** per-change-type policy (`agent`/`human`), presets (`auto/completions/
  final/strict`), deviations *held* not applied — applied only on sign-off (design rule).

## 11. MCP stack (`mcp_session.py`, `mcp_toolset.py`, `mcp_reach.py`, `nodes/mcp_server.py`, `nodes/mcp_aggregator.py`)

- **D19–D22 shape:** a node owns one live session per server (own event loop per
  session, `_serve`); the session is handed downstream as data; several MCP nodes reach
  one ToolBox; credentials are names resolved at connect (`resolved_headers`), never
  persisted.
- **Namespacing (D21):** every tool from a server carries the server's prefix.
- `mcp_reach.py` classifies advertised tools that look like filesystem access and says so
  in one status line — the same "reported, not silent" honesty as `import_reach`.
- `mcp_toolset.py` (1,186 LOC) carries a vendored-shaped transport/SDK shim (stdio
  subprocess, resources, prompts, prompts templates, error mapping). It is the largest
  file in `functions/` and the one place that feels like an imported dependency living in
  the tree rather than a Silk subsystem — a candidate for extraction upstream or behind a
  thin adapter if `mcp` SDK coverage grows.

## 12. Events & observability (`stream_events.py`, `event_format.py`, `event_sink.py`, monitors)

- One canonical typed vocabulary (19 `Event*` dataclasses) with `to_wire`/`wire_kind`
  and an `EventBuilder`; content-light by rule (`EventModelResponse` carries length, not
  text; `EventStart` carries a system-prompt *length* — §22 q3).
- `RunSink`: one JSONL file per run (T7), opened on first event, pruning old files,
  **redaction by construction** (`redact()` keeps shape and sizes, never text) — the
  "observability is content-free" design rule, already implemented rather than pending.
- Monitors count and render; the decision *inbox* dock answers through the asking node,
  never around it (D59), and `DecisionRegistry` tracks open requests session-scoped with
  weak node references.

## 13. Hooks & middleware (`hooks.py`, `hook_catalog.py`)

- `emit`: FIFO for `before_*`, LIFO for `after_*` (Pydantic-AI semantics); hook
  exceptions are swallowed (a hook must not break a run); per-tool applicability binding
  lives in the registry (D13), not in each hook's body.
- `emit_middleware`: index-based chain over a stable handler list — a middleware may call
  `handler()` more than once (retry is advertised) and each call re-runs the *remaining*
  chain (the old shared-pop version silently skipped middlewares); context travels with
  the call on pass-through (`_next(**{**kw, **overrides})`) — the exact mechanism that
  used to silently disable the D77/D82 floors when a pass-through dropped `tool_name`.
- `hook_catalog.py`: selectable, pydantic-configured hooks (signoff, tool approval,
  spill, remember, tool budget, task audit, redact-secrets, timing, usage meter,
  log-tool-calls) with inert factories wired store-aware by `attach_catalog_hooks` —
  catalog UI and runtime wiring share one vocabulary.

## 14. Multi-agent (`orchestrator.py`, `subagent.py`, `usage_limits.py`, `agent_spec.py`)

- `run_subagent` is the Qt-free drive loop shared by node and orchestrator: fresh
  history (no context leak), own pool session (own KV cache), own `RoleBinding` per
  worker (independent rebuilt ToolBoxes — concurrency-safe by construction).
- **Budgets nest (D26/T3):** `nest(shared, own)` makes worker caps a `SubBudget`
  *inside* the orchestrator's shared cap (never beside it); a worker's `SubBudget` of the
  same parent is returned unchanged on re-entry. The whole fan-out shares one
  `UsageLimits` and `reserve_tool_calls` claims a batch before dispatch (D52.4).
- Delegation depth is port-or-spinbox with `_spill_delegation` handling oversized
  replies; worker events re-emit on the orchestrator's stream content-lightly.

# Part III — Cross-cutting audits

## 15. Layering audit — **holds**

- `functions/` contains **zero** PySide6/PyQt imports (grep-verified; the only matches
  are docstring prose). The Qt-free claim is real: a full agent run works headless.
- `functions/` *does* import `weave.logger` in 29 modules plus `weave.engine.*` in 6 —
  Silk is Weave-independent-of-Qt, not Weave-independent. That is by design and policed
  by `weave_contract.py` (D83/G20): the list of internals Silk reaches into, each with a
  reason, checked at import; a moved seam says so by name in the load log, and the
  Weave-side test fails. A finding never blocks a load.
- Dependency guards degrade with named, actionable messages
  (`server_missing_deps_message()` names the missing pip extra and verifies the server's
  *third-party* deps, not just module presence — the `[server]` extra ships files
  without fastapi).
- Node-layer type coverage: G9 closed ("type the node layer, and fix the three live bugs
  it found").

## 16. Coupling & import graph

Intra-`functions/` fan-in (module dependencies, relative imports resolved):
`tool_box` 17 · `tools/file_sandbox` ~13 · `hooks` 11 · `stream_events` 5 ·
`usage_limits` 4 · `task_store` 4 · `self_modify` 4 · `capabilities` 4 · `grants` 4 ·
`decision_seam` 3 · `protocols` 3.

Fan-out (most coupled): `hook_catalog` 10 · `tool_box` 9 · `agent_loop` 8 ·
`approval` 8 · `subagent` 6 · `approval_floor` 6.

`tool_box` is the dependency hub **by design** — it is the substrate everything registers
on, and its imports flow *inward* (it imports floors/floors-import it back). Five
apparent import cycles resolve as:

| Cycle | Edge nature | Verdict |
|---|---|---|
| `approval → tool_box` | top-level (approval.py:78) | real edge, loads cleanly |
| `tool_box → approval_floor` | **deferred** (inside functions, x2) | deliberate seam, no cycle at load |
| `approval_floor → approval` | top-level | real edge |
| `tool_box → tools/tool_loader` | top-level + deferred | real edge |
| `self_modify ↔ suite_pins` | both **deferred** | lazy, harmless |
| `task_store → plan_discovery` | deferred; `plan_discovery → task_store` top-level | harmless |

One genuine wrinkle: `approval_floor → approval` is top-level while `tool_box →
approval_floor` is deferred — the floor sits between two top-level edges of the gate
subsystem. It works (Python loads it), but a future refactor that promotes one of the
deferred imports could turn this into a real import-order bug. The tests would catch it
only in Weave's tree.

## 17. Concurrency audit — **strong, with designed-in testability**

- Thread model (doc §16) matches the code exactly: sync generator loop; one fresh
  `asyncio.run` per tool batch; sync tools via `to_thread`; `wait_for` timeouts;
  `gather` for parallel, declared-`sequential` one at a time.
- The human-decision block: request goes *out* on the hook/emit path (thread-safe), the
  answer comes *back* through the `DecisionSeam` waiter; node calls `seam.cancel()`
  directly on Stop because nothing polls mid-`next()` (D38/G8). Ordering rule enforced
  in `BlockingSeam.commit`/`submit` (write-under-lock, then wake; re-read under lock).
- `DriveGate` checkpoints make the five-way race catalog deterministic in tests —
  production cost is one dict lookup.
- Lock inventory is small and purposeful: `file_locks.py` (per-path locks + a
  readers-writer `_RootGate` over nested roots), `blocking_seam`, `ledger` registry,
  `mcp_session`, `model_pool` (`RLock` over process/session state; separate
  `_prefix_lock` for the meter), `usage_limits`, `prefix_stats`.
- The one `except BaseException` in the tree (`tools/file_write.py:116`) is *correct*:
  temp-file cleanup on any exit path then `raise` — part of the
  `mkstemp → fsync → os.replace` atomic write, with `expected_sha256` preconditions
  compared inside the write's own lock (§22 q8).

## 18. Security & containment posture

The design's own honesty is its best feature (D36 fail-closed, "reported not silent"):

- **File sandbox:** everything resolved via `Path.resolve()` before comparison
  (`file_sandbox.py`), roots/allowed/denied/writable modes resolved at construction and
  narrowed monotonically (`FileGrants.narrow` — a chain of narrowings is monotonically
  decreasing, I6); hierarchical mode map, extension checks, `_assert_safe` escapes check.
- **Process authority is admitted:** `run_python` and the toolchain family execute *real
  processes* (python `-I`, cwd = sandbox root, risk=`high`, 60 s timeout; cargo/maturin/
  ruff/mypy/radon similarly). Containment for these is the role gate + approval floors +
  risk metadata, **not** an OS sandbox. `toolchains.py` says so in its module docstring,
  and G21 keeps the residue REPORTED with both halves built (`import_reach`,
  `mcp_reach`). Silk cannot narrow this; it documents it.
- **Secrets:** names only in graphs/presets/servers (`MCPServerSpec` "no secret ever
  lands here"); values resolved at connect from env or `~/.weave/silk/secrets.json`,
  never written back, only used for headers; a missing credential raises with the path
  to put it (D22). A `redact_secrets` middleware masks secret-looking tool results; the
  JSONL sink keeps shapes and sizes only.
- **Approval surfaces:** decision requests render tool args + risk + change type +
  project root; load requests render diffs (capped) + file listings + the pin
  consequence ("approving also pins these exact bytes"); the grant manager dock can
  revoke any durable grant or suite pin.

## 19. Error-handling census

One `except BaseException` (verified correct, above). Broad `except Exception` clusters
where classification is the job: `model_pool` (10 — dependency probes, shutdown, log
tail), `mcp_toolset` (6), `ledger` (6), `hooks` (6), `agent_loop` (6 — compactor,
transport, engine probes). Elsewhere the failure-as-data rule keeps `except` counts low
and specific. `ruff check .`: **all checks passed** at the repo's own config
(py312, line 100).

## 20. Docs ↔ code cross-check — **unusually faithful**

- `NODES.md`: 17 node sections = 17 node classes, grouped as Model / Tool assembly /
  Agent configuration / Execution / Observability & human gates. ✔
- `TOOLS.md`: module→tool table matches the registered inventory (file read/write/
  manipulate, ripgrep, task tracker's ten plan verbs, toolchains' ten commands; support
  modules named). ✔
- `OPEN_TOPICS.md`: 16 of 21 gaps CLOSED with code citations; G15 landed as measurement
  (`prefix_stats` reading the server's own logs — nothing added to the model path); G20
  partly closed (`weave_contract.py` exists and runs); G21 REPORTED with the mitigation
  named as mitigation. The one place the docs hedge ("test suite runs from the Weave
  root — `python -m pytest tests -q`; invariant fixtures in `tests/test_silk_invariants.py`")
  is accurate: **this repo has no tests/ directory** — the suite lives in the Weave tree.
- `17-invariants.md` maps 1:1 onto enforcement sites verified above (I1 dispatch shape,
  I2 after_run-exactly-once, I3 loop-never-executes-tools, I4 double-sided role gate,
  I5 store-reads-never-mutate).
- `16-thread-model.md` matches the code, including the exact seam/waiter mechanics.

## 21. Git analytics (last 63 commits, 2026-08-30 → 09-03)

Cadence: 7 + 6 + 2 + **39** + 24. Churn leaders: `DESIGN_SPEC_DRAFT.md` +2,522,
`ledger.py` +1,434, `OPEN_TOPICS.md` +965/−233, `nodes/agent.py` +618/−78,
`self_modify.py` +612. The work was one sustained design-sprint; the spec (D-numbers now
into the 80s) advanced in lockstep with the code — every module cites the decisions it
implements, which is what makes a review of days-old code possible at all.

# Part IV — Health & recommendations

## 22. Smell findings (credible subset)

Tooling caveats first: **1,301 of 1,737 graph findings are `dead-code` false positives**
— verified against `agent_loop.py:508` calling `ToolBox.execute_tool_calls_async` through
a duck-typed attribute and the dynamic `NODE_REGISTRY`; `affected()` probes return 0 as
expected. The protocol-seam style defeats the reachability analysis *by design*. The
clone detector times out at this size (three attempts). An independent AST pass
reproduced the complexity ranking (raw CC numbers run higher — e.g. `compute` ≈71 vs 49
— the ranking is the robust signal).

Real findings, with file attribution verified by AST:

| Location | CC | LOC | Note |
|---|---|---|---|
| `nodes/agent.py` · `SilkAgentNode.compute` | 49 | 454 | Brain Method (WMC 107); wiring hub |
| `nodes/toolbox.py` · `SilkToolBoxNode.compute` | 23 | 161 | Brain Method (WMC 35) |
| `functions/tools/graph_authoring.py` · `attach_graph_tools` | 33 | 311 | largest `attach_*` |
| `functions/approval.py` · `attach_approval_gate` | 31 | 187 | 19 returns; domain-heavy |
| `functions/agent_loop.py` · `AgentLoop._run_rounds` | 26 | 289 | Brain Method (WMC 49), by design |
| `functions/stream_events.py` · `format_event` | 24 | — | event formatting |
| `functions/tools/ripgrep_tool.py` · `_search_ripgrep_impl` | 21 | 112 | |
| `functions/subagent.py` · `run_subagent` | 17 | 122 | |
| `functions/tool_box.py` · `execute_tool_calls_async` | 15 | 149 | |

Other long methods ≥100 LOC: `SilkAgentNode.__init__` 218, `GGUFLNode.__init__` 207,
`AgentInferenceSettingsNode.__init__` 203, `SilkToolBoxNode.__init__` 194,
`attach_suite_tools` 187, `attach_file_read_tools` 177, `attach_orchestrator_tools`
166, `attach_remember_hook` 141, `attach_file_write_tools` 122, `ToolBox.register` 107,
`attach_recall_tool` 104, `GGUFModelPool.__init__` 101.

Histogram (false-positive rule excluded): long-parameter-list 118 · long-method 89 ·
data-class 66 (mostly by-design dataclasses) · deep-nesting 59 (max 5) ·
excessive-returns 40 · high-cyclomatic 37 · too-many-fields 23 (config objects) ·
brain-method 4.

The concentration pattern: **`attach_*_tools` factories and node `compute()`/`__init__`
wiring**. Each new seam thickens the same bodies. The data-structure layer
(ledger/task_store/grants) and the seam layer (blocking_seam/decision_seam) are clean.

## 23. Verdict

**Architecture: excellent — and verifiably so.** The layering rule holds under grep;
the thread model doc matches the code line-for-line; failure semantics are designed
(fail-closed, named causes, model-visible results); observability is content-free by
construction; the docs cite code and the code cites decisions; ruff is clean; the test
suite lives where the repo says it does (Weave root) and pins the invariants.

**Watch items, priority order (unchanged from v1, now with deeper evidence):**

1. **`SilkAgentNode.compute()`** — 454 LOC, CC 49, WMC 107. Extract per-concern wiring;
   the 218-LOC `__init__` sheds with it. Highest-risk body in the repo.
2. **The `attach_*_tools` family** (311/238/187/187/177/166 LOC, CC up to 33). A shared
   registration skeleton (schema → gate → register → wire) cuts all six at once and
   would surface latent inconsistencies between them.
3. **`AgentLoop._run_rounds`** — keep new logic out of the main body; the helpers are
   already the right seams.
4. **Task-store signatures** — 7–8-param bundles → the existing dataclasses.
5. **`mcp_toolset.py`** — the one module that reads like vendored SDK surface; consider
   extraction or a thin adapter before it grows.
6. **Process, not code:** never run automated dead-code cleanup against this analyzer's
   reachability verdicts without grep + runtime verification; the codebase's best quality
   (protocol seams) is exactly what defeats the analysis.
7. **Minor:** the `approval ↔ approval_floor` top-level/deferred edge mix — pin it with
   an import-order test before a refactor promotes a deferred import.

# Part V — Method & reproducibility

- Index: CodeRadar codegraph — 106 files / 274 classes / 1,520 functions / 3,116 call
  edges; smells at normal strictness; scaffolding, dead-code and `affected()` probes;
  semantic search. Clone scan timed out at min_lines 15 and 30.
- Verification: full reads of `agent_loop.py`, `approval.py`, `decision_seam.py`,
  `blocking_seam.py`, `model_pool.py`, `self_modify.py`, plus the `tool_box.py`
  dispatch region and `hooks.py` middleware chain; AST passes for method LOC (≥80) and
  approximate CC; import-graph analysis with relative imports resolved and cycle edges
  classified top-level vs deferred; `ruff check .`; greps for Qt/subprocess/secrets/
  threading; docs cross-checks (NODES.md 17=17, TOOLS.md inventory, OPEN_TOPICS G/T
  status, invariants §17, thread model §16).
