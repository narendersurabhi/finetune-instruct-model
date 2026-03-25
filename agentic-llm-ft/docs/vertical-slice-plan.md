# Vertical Slice Implementation Plan

## Objective
Deliver one end-to-end, reproducible vertical slice that proves the repository can:
1. load and validate a dataset,
2. run a minimal LoRA fine-tuning job,
3. evaluate the checkpoint with task metrics + validation loss/perplexity,
4. run inference on held-out prompts,
5. execute the simulated agent loop with deterministic tools,
6. persist artifacts that enable CI/ops verification.

## Vertical Slice Definition
The slice is "thin" in scope but complete across layers:
- **Data:** canonical small dataset only.
- **Training:** short single-run configuration intended for fast feedback.
- **Evaluation:** one configured eval dataset and metrics output.
- **Inference:** one deterministic smoke prompt set.
- **Agent Runtime:** one trace-producing scenario with tools.
- **Ops:** scriptable commands and artifact checks.

## Success Criteria
A vertical slice is complete when all criteria below are met in one runbook:
- Dataset validation passes with no schema errors.
- Training completes and writes a checkpoint artifact.
- Eval emits JSON metrics including `exact_match`, `tool_call_precision`, and non-null `validation_loss`/`perplexity` when checkpoint loading is enabled.
- Inference emits structured output for at least one prompt.
- Agent runtime produces a non-empty trace log with at least one tool execution.
- Commands are documented and repeatable from any working directory.

## Implementation Phases

### Phase 1 — Baseline audit and deterministic config freeze
- Add a dedicated config override file for slice defaults (small epochs/steps, stable seed, limited samples).
- Confirm entry points resolve configs via Hydra from any CWD.
- Document exact commands and expected artifact paths.

**Deliverables**
- `configs/experiments/vertical_slice.yaml`.
- Runbook section in README or dedicated docs page.

### Phase 2 — Data and schema gate
- Use existing dataset normalization/validation path as the first gate.
- Add a small, explicitly versioned eval subset if needed for deterministic checks.
- Add/extend tests that fail fast on malformed examples.

**Deliverables**
- Optional tiny eval fixture dataset.
- Test updates for dataset + prompt rendering integrity.

### Phase 3 — Minimal training path
- Configure a low-cost LoRA run profile for fast completion.
- Ensure tokenizer/chat template path respects `max_seq_len` and pad token behavior.
- Persist checkpoint under a deterministic output directory.

**Deliverables**
- Documented training command with overrides.
- Saved checkpoint artifact path convention.

### Phase 4 — Checkpointed evaluation
- Run `scripts/run_eval.py` against the training output using `eval.checkpoint_path`.
- Validate both task metrics and model-based metrics (`validation_loss`, `perplexity`).
- Save metrics artifact in a stable location.

**Deliverables**
- Eval command + sample metrics schema.
- Metric artifact contract for automation.

### Phase 5 — Inference + agent runtime smoke
- Run inference command with vertical-slice checkpoint.
- Run simulated agent loop on one scenario that triggers tool usage.
- Save outputs/traces for post-run inspection.

**Deliverables**
- Inference output artifact.
- Agent trace artifact with deterministic rows.

### Phase 6 — CI-oriented automation
- Add a `make` target (or script wrapper) to orchestrate the full slice.
- Gate completion with a compact verification checklist.
- Keep runtime bounded for developer feedback loops.

**Deliverables**
- `make vertical-slice` (or equivalent).
- CI-ready checklist with pass/fail conditions.

## Proposed Task Breakdown (Backlog)
1. Create vertical slice experiment config.
2. Add docs runbook with expected outputs.
3. Add/refresh tiny deterministic eval fixture.
4. Add smoke test for eval metric keys + non-null validation metrics when checkpoint is provided.
5. Add script/Make target for end-to-end orchestration.
6. Add artifact verifier utility (optional) for CI assertions.

## Risks and Mitigations
- **Risk:** environment-dependent model downloads slow execution.
  - **Mitigation:** support pre-downloaded/local model path and tiny sample sizes for slice mode.
- **Risk:** non-determinism in generation and tool output.
  - **Mitigation:** fixed seeds, deterministic tool stubs, bounded decoding params.
- **Risk:** brittle artifact paths in different CWDs.
  - **Mitigation:** centralize output roots in config and reference absolute/normalized paths.

## Out of Scope (for this slice)
- Full hyperparameter sweeps.
- Multi-model benchmarking matrix.
- Production deployment infrastructure.
- Long-horizon agent trajectories.

## Exit Checklist
- [ ] Vertical slice config committed.
- [ ] Runbook commands documented.
- [ ] End-to-end command completes successfully.
- [ ] Metrics + inference + trace artifacts generated and validated.
- [ ] Lightweight automated check added to prevent regressions.
