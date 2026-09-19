# Measuring prompt-prefix reuse

*Spec D41/D47, Phase 1 item 6. Closes the measurement half of G15.*

> **Taken 2026-09-19. The result is in [The numbers](#the-numbers) below
> and the decision is spec D91: mechanism A, session affinity, built and
> on by default.** The rest of this page is the procedure, which is still
> worth running — the threshold in D47's first clause is read off *your*
> workload, and a different model, GPU or prompt size can move it.

Silk's context design rested, until 2026-09-19, on a number nobody had
measured: how much of a
request's prompt the backend already had in its KV cache. Every mechanism in
D47 -- session-affine routing, `LlamaCache`, prefix-stable rendering -- is a
way of protecting that number, and none of them should be built before it is
known. **"Do nothing" is a permitted, and quite likely, outcome.**

## What is instrumented

Nothing was added to the model path. `verbose` is forwarded to the spawned
`llama_cpp.server`, and the pool already captures the server's stderr to a
file (it needs it to explain a failed start). That file carries, per
request:

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

**One correction, 2026-09-19.** The GGUF loader node hard-coded
`verbose: False`, so on the canvas the server never wrote the lines this
reads and `prefix_report()` said "nothing measured" forever -- while the
same code worked perfectly from a script, which is why nobody caught it.
That is why the numbers below took until now to exist. `verbose` is now
on: about six lines per request into a file the pool already keeps, and
no prompt or completion text is logged.

## Reading the numbers

| Metric | Definition | What it decides |
|---|---|---|
| **Reuse rate** | matched / (matched + evaluated), summed over requests | whether reuse is being lost at all |
| **Contention rate** | requests whose immediate predecessor came from a different session, over requests that have a predecessor | whether the loss is interleaving or prefix instability |
| **Prefill share** | prompt-eval time / total time, summed | whether any of this is worth building |

Anything the log did not say is reported as `None`, never as zero. A reuse
rate of 0.0 and an unknown reuse rate lead to opposite decisions.

**A timing that cannot be true is also "did not say".** The first request
after a model loads prints `prompt eval time = 0.11 ms / 24 tokens` —
218,000 tokens a second on hardware that does 550 — because that first
prefill is booked against the load rather than the request. Left in, it
would put a near-zero prefill on the single largest prefill of a run,
which is the one number D47's first clause reads. The rate is dropped and
the token counts, which are real, are kept.

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

## The numbers

Taken 2026-09-19 on gemma-4-E4B-it-Q4_K_M, RTX 3090, `n_ctx` 16384,
80-token answers, two conversations of four rounds each driven from two
threads (the way `delegate_parallel` drives them):

| Prompt | Shape | Reuse | Contention | Prefill | Wall |
|---|---|---|---|---|---|
| ~230 tok | one session | 71.1% | 0% | 26.0% | 0.4 s |
| ~230 tok | two at once | 10.3% | 100% | 39.9% | 0.6 s |
| ~1600 tok | one session | 74.8% | 0% | 33.1% | 1.1 s |
| ~1600 tok | two at once | **0.4%** | 100% | 74.1% | **4.5 s** |
| ~4250 tok | one session | 74.9% | 0% | 66.4% | 1.9 s |
| ~4250 tok | two at once | **0.2%** | 100% | 80.3% | **11.4 s** |

Reuse is scale-free, as a token ratio should be. Prefill share is not,
and that is the trap: **a first capture at 200-token prompts read 5% and
would have fired clause 1 (do nothing).** Prefill share is a statement
about the ratio of prompt to answer, and an agent round is the inverse
shape of a chat turn — schemas, instructions and history in, a sentence
out. Read the threshold off the size you actually run at.

With session affinity on (spec D91), the same runs:

| Prompt | Reuse | Contention | Wall | vs. off |
|---|---|---|---|---|
| ~230 tok | 70.9% | 14.3% | 0.7 s | *slower by 0.1 s* |
| ~1600 tok | **74.8%** | 14.3% | **1.9 s** | 2.4x faster |
| ~4250 tok | **74.9%** | 14.3% | **3.8 s** | 3.0x faster |

Two conversations reuse exactly what one does. The first row is the
honest cost, left visible: below the size where prefill matters, the hold
window costs more than it saves — which is why the window is capped by
the measured cost of the prefix it protects.

### If reuse is 0% in *every* shape

Check the log for `prefix-match found but partial kv removal not
supported, re-evaluating full prompt`. On some architectures (observed on
Qwen3.5-4B) llama.cpp finds the prefix and re-evaluates the whole thing
anyway. That is a property of the backend, not of scheduling, and no
amount of affinity, `LlamaCache` or extra backends changes it. The answer
is a different model.

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

## What happened when the rule was applied

Clause 2, on 2026-09-19: **session affinity** (mechanism A), built as
`functions/session_affinity.py` and on by default. It groups the queue by
conversation instead of by arrival — a release opens a short window in
which the outgoing session keeps its claim, because the same
conversation's next request does not exist yet when its round ends, and a
queue that admits whoever is waiting therefore admits the *other* one
every time.

The window is bounded three ways, all visible on the Pool Monitor's
**Affinity** row:

- **priced** — it never exceeds `full_prefill_ms`, what a lost prefix was
  measured to cost on this server, so a cheap prefix cannot buy an
  expensive wait;
- **leashed** — three windows that elapse unused suppress it, and an
  arrival that *would* have been in time restores it, so recovery is free;
- **fair** — after `max_streak` consecutive grants a conversation with
  somebody waiting yields a slot.

Turn it off with **Group Requests by Agent** on the GGUF loader (advanced
mode). The reason to is latency in a graph whose prompts are small enough
that prefill does not matter — which the Affinity row will tell you,
because it reports the wait it is costing you next to the reuse it is
buying.
