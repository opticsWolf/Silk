## Thread model

- **The agent loop is a synchronous generator.** It is driven from the GUI
  thread (or any thread) and yields events as they happen; it is not
  itself async.
- **Tool batches get their own event loop.** Each round's tool batch runs
  via `asyncio.run(toolbox.execute_tool_calls_async(...))` — a fresh,
  short-lived loop per batch, which keeps the sync/async boundary simple.
  Inside that batch: sync executables are pushed off the loop via
  `asyncio.to_thread` and never block it; both sync and async execution is
  wrapped in `asyncio.wait_for` for timeouts; independent calls run under
  `asyncio.gather`, and tools flagged `sequential` run one at a time, so a
  mutation that can't tolerate concurrency is safe by declaration.
- **Model requests are streaming generators** — the loop pulls deltas as
  they arrive, so the UI stays live and a stop is honoured at the next token
  boundary.
- **The task store is its own concern** — SQLite with optimistic revision
  checks, so concurrent agent runs on one plan resolve conflicts by
  returning a `Conflict` rather than corrupting state.

