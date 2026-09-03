# Measuring prompt-prefix reuse

*Spec D41/D47, Phase 1 item 6. Closes the measurement half of G15.*

Silk's context design rests on a number nobody has measured: how much of a
request's prompt the backend already had in its KV cache. Every mechanism in
D47 -- session-affine routing, `LlamaCache`, prefix-stable rendering -- is a
way of protecting that number, and none of them should be built before it is
known. **"Do nothing" is a permitted, and quite likely, outcome.**

## What is instrumented

Nothing was added to the model path. `verbose` is already forwarded to the
spawned `llama_cpp.server`, and the pool already captures the server's stderr
to a file (it needs it to explain a failed start). That file already carries,
per request:

```
Llama.generate: 512 prefix-match hit, remaining 8 prompt tokens to eval
llama_perf_context_print: prompt eval time =  210.11 ms /     8 tokens
llama_perf_context_print:       total time = 1980.44 ms /   140 tokens
```

`functions/prefix_stats.py` reads that file forward from wherever it last
stopped. `GGUFModelPool.begin_request` / `end_request` bracket each request,
and `GraphEngine.stream_response` calls them around its stream, tagging the
sample with the session id (D46). Attribution is sequential because the
server is: `llama_outer_lock` serialises every request (D43/D53), so lines
that appear between begin and end belong to the request in between.

Both pool hooks are optional and both are called defensively -- a
measurement must never be able to fail a run.

## Reading the numbers

| Metric | Definition | What it decides |
|---|---|---|
| **Reuse rate** | matched / (matched + evaluated), summed over requests | whether reuse is being lost at all |
| **Contention rate** | requests whose immediate predecessor came from a different session, over requests that have a predecessor | whether the loss is interleaving or prefix instability |
| **Prefill share** | prompt-eval time / total time, summed | whether any of this is worth building |

Anything the log did not say is reported as `None`, never as zero. A reuse
rate of 0.0 and an unknown reuse rate lead to opposite decisions.

## How to run it

**Live, on the canvas.** Wire a **Pool Monitor** to the GGUF loader's
`model_obj` and read the *Prefix reuse* row:

```
Prefix reuse:  reuse 93.2%  ·  contention 25.0%  ·  prefill 11.4%  ·  12/12 measured
```

Before any request it reads `— (no requests measured yet)`, which is the
point: an unmeasured metric must not render as a zero, because 0% and
"nobody looked" lead to opposite decisions. Wire an agent's `done` port to
the monitor's `refresh` to see it move as a run proceeds.

**Live, from code.** The pool snapshot carries the same report under
`prefix_reuse`, and the GGUF loader node already streams that snapshot on its
`pool_info` port, so anything watching the pool sees the three numbers
without further wiring:

```python
pool.prefix_report()
# {'requests': 12, 'measured_requests': 12, 'reuse_rate': 0.93,
#  'contention_rate': 0.25, 'prefill_share': 0.11, ...}
```

`pool.reset_prefix_stats()` clears the window between experiments.

**Offline, from a captured log.** The server log path is
`<tempdir>/silk-llama-*.json.log`. Contention is not recoverable this way --
nothing in the file says whose request a line was -- so it reports unknown:

```bash
python -m weave.plugins.silk.functions.prefix_stats /path/to/silk-llama-XXXX.json.log
```

## The two runs to capture

D47 asks for both, because they answer different halves of the question:

1. **A single multi-round run.** One agent, several tool rounds. Reuse here
   is what I11 (prefix stability) buys; a low rate means the prompt is not
   byte-stable across rounds, and no amount of routing will fix it.
2. **An orchestrator fan-out.** Two or more workers delegated in one graph.
   Reuse here is what interleaving costs; the difference between run 1 and
   run 2 *is* the contention term.

## The decision rule (D47), applied in order

1. **Prefill share under ~15%** — do nothing. None of A/B/C is worth its
   cost.
2. **Reuse low, contention high** — the loss is interleaving. Session-affine
   routing (A) or `LlamaCache` (B, at the cost model in D44).
3. **Reuse low, contention low** — the prefix itself is unstable. Fix the
   rendering (I11); no cache mechanism substitutes for it.
