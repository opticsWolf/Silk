## Usage, reflection, and output validation

Three independent guardrails on a run:

- **`functions/usage_limits.py`** — `UsageLimits` caps a *run-wide* budget:
  output tokens, input tokens, model requests, and tool calls. Exceeding any
  cap yields a `USAGE_LIMIT` event and ends the run cleanly (not an error —
  a controlled stop). `functions/usage.py`'s `UsageStats` is the plain
  accumulator that records the actuals.
- **`functions/reflection.py`** — `ReflectionConfig` drives self-correction:
  when a tool returns a *retryable* validation error (or output validation
  fails), the loop injects a reflection prompt and retries, within a bounded
  retry budget. Non-retryable error types (e.g. `role_denied`) are not
  retried — the model is told the denial is final.
- **`functions/output_schema.py`** — `OutputSchema` (built via
  `from_model(BaseModel)` or `from_dict`) validates the model's *final*
  answer against a schema and `build_instruction()` injects the required
  shape into the system prompt. `OutputValidator.validate_with_reflection`
  couples it to the reflection loop.

`functions/run_context.py`'s `RunContext` is the typed bag carried through a
run: `engine`, `deps` (e.g. `db_pool`, `user_session`), `usage`,
`usage_limits`, `model_settings`, `run_step`, and the loaded/available
capability + tool name sets — what a dynamic capability description or tool
can see about the current run.

