
## The classifier — triage, hooks and the `classify` tool

A typed classifier (hosted Jev, or a local judge model) is a **graph resource on
a port**, like a model (D-S8). The Silk Classifier node outputs a
`silk_classifier` handle; the Agent node takes it next to `model_obj`, and the
Orchestrator passes it on to its workers for the length of a delegation. Unwired,
nothing changes: every decision stays with the user.

### Three consumers, one of which can change what runs

| Consumer | Installed as | Purpose key | Can it change what runs? |
|---|---|---|---|
| Approval gate | a step **inside** the gate, after grants | `gate` | Only at the `triage` level, in `live` mode, inside the bands calibrated for the model that answered |
| Classifier hooks (`task_drift`, `prose_question`, `findings_gate`) | catalog entries, Role- or ToolBox-selected | `hook` | `task_drift` bounces a `task_add` once; the others file things. Never approve or deny |
| `classify` tool | registered for a run while the port is wired | `tool` | No: advisory results, own cache namespace, never read by the gate |

### `functions/triage.py` and the `triage` level

`SIGNOFF_LEVELS` is `agent < triage < human`; "the stricter answer wins" ranks
by it. `SignoffConfig` offers `triage` per change type, `ToolApprovalConfig`
offers `triage_bands` and `triage_tools`; `human` entries always win, and naming
a tool `triage` never lowers a band gated `human`.

In the gate (`functions/approval.py`) the order is: floors (D77, D82) → grants →
**classifier** → seam. With no classifier bound, or mode `off`, a `triage` call
*is* a `human` call (`effective_level`). Otherwise `triage_approval` asks the
five approval questions about a state built from the plan and the call's own
arguments — never from the transcript — and:

- at `triage`, `live`, all bands met for that model → the call runs without
  asking and is recorded on the toolset (`triaged_calls`, summarised at run end);
- otherwise → the question goes to the human with `priority`, `suggestion`,
  `why` and the verdict in `DecisionRequest.detail`.

The classifier never denies. A `complete_final` at `triage` also needs the
completion check to accept; a rejected check only adds `suggestion: deny` and
the unmet criteria to the card. A worker without a seam can pass a triage-level
call this way; every other gated call it makes is still refused (D36).

### `functions/classifier.py` — one question format, a chain of backends

Questions and answers use Jev's `/v1/systemone` shape everywhere
(`validate_questions`, `check_answers`). `ChainClassifier` tries hosted Jev
(pinned version, redacted state, only with the project's egress opt-in), then a
local judge (`classifier_local.pool_scorer`: a *second* GGUF model scoring fixed
options by log-probability), then nothing. A circuit breaker rests a failing
link. The cache is keyed by purpose. `ClassifierSpec` is the node's whole form;
`ClassifierHandle.for_run` snapshots it per run (D38).

### `functions/classifier_binding.py` — per run, like the seam

`begin_classifier_run` binds the run's view on the toolset, registers `classify`
while wired, refuses a run whose judge is the agent's own model, and says once
when the policy uses `triage` with nothing to triage with.
`end_classifier_run` unbinds, removes a port-registered `classify`, and reports
how many calls triage allowed. `inherit_run_classifier` /
`restore_run_classifier` are the orchestrator's push/restore (D52.2).

### `ToolboxBound` hooks

A hook map is built before any toolbox exists. A hook that needs one (the
classifier hooks read the run's classifier, plan and identity) is declared as
`ToolboxBound(make)`; `register_hook_map` calls `make(registry.owner)` when it
registers it. `ToolBox` lends its registry itself as the owner.
`prose_question` needs the work ledger and `findings_gate` decision records and a
knowledge sink; until a toolbox carries those (`_work_ledger`, `_decision_log`,
`_knowledge`), they are selectable and do nothing.

### Calibration

Modes: `off`, `shadow` (default: score and record, never skip), `live`. The
thresholds table holds bands **per model id**; a model without a row — a new Jev
version, another judge file — answers but never skips. Nothing switches itself
on.
