# Vertical Slice Runbook

This runbook implements the phased rollout from `docs/vertical-slice-plan.md`.

## Phase Tracker

| Phase | Goal | Status | Owner Action |
|---|---|---|---|
| 1 | Deterministic config + artifact paths | ✅ Implemented | Use `configs/experiments/vertical_slice.yaml` |
| 2 | Data/schema gate | ✅ Implemented | Validate deterministic eval fixture and tests |
| 3 | Minimal LoRA training | ✅ Implemented | Run training command below |
| 4 | Checkpointed eval | ✅ Implemented | Run eval with checkpoint-based metrics |
| 5 | Inference + agent trace | ✅ Implemented | Run inference smoke and inspect trace |
| 6 | CI automation + checks | ✅ Implemented | Run `make vertical-slice` then `make vertical-slice-verify` |

## Artifact Contract

All vertical-slice outputs are rooted at:

- `outputs/vertical-slice/checkpoints/lora`
- `outputs/vertical-slice/eval/metrics.json`
- `outputs/vertical-slice/eval/predictions.jsonl`
- `outputs/vertical-slice/inference/agent_traces.jsonl`

## Phase Commands

Run commands from repo root (`agentic-llm-ft/`).

### Phase 1 — Baseline deterministic profile

```bash
python scripts/train.py --config-name experiments/vertical_slice --help
```

### Phase 2 — Data and schema gate

```bash
pytest tests/test_dataset.py tests/test_prompt_rendering.py tests/test_vertical_slice_eval.py
```

### Phase 3 — Minimal training path

```bash
python scripts/train.py --config-name experiments/vertical_slice
```

### Phase 4 — Checkpointed evaluation

```bash
python run_eval.py --config-name experiments/vertical_slice
```

Expected metrics keys include:
- `exact_match`
- `tool_call_precision`
- `validation_loss`
- `perplexity`

### Phase 5 — Inference + agent runtime smoke

```bash
python scripts/inference.py --config-name experiments/vertical_slice inference.mode=agent inference.prompt="What's the weather in Seattle?"
```

### Phase 6 — End-to-end automation and verification

```bash
make vertical-slice
make vertical-slice-verify
```

## CI Recommendation

Use `make vertical-slice-verify` as the pass/fail gate after running the slice. The verifier checks required artifacts, non-empty trace/prediction logs, and non-null checkpoint-based eval metrics.
