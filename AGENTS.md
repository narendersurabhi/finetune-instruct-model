# Agent Change Log & Repository Status

This file tracks all substantial repository changes and should be read by all agents before modifying code.

## Current Status
- Repository scaffold created for `agentic-llm-ft/`.
- Initial production-style modules added for configuration, schema validation, tools, dataset handling, training, evaluation, inference, and simulated agent runtime.
- Sample datasets and test suite added.

## Change Log
### 2026-03-25 (vertical slice implementation)
- Implemented phased vertical-slice execution assets:
  - added deterministic profile `agentic-llm-ft/configs/experiments/vertical_slice.yaml`,
  - added deterministic eval fixture `agentic-llm-ft/data/sample/vertical_slice_eval.jsonl`,
  - added phased runbook `agentic-llm-ft/docs/vertical-slice-runbook.md`,
  - updated plan tracking statuses in `agentic-llm-ft/docs/vertical-slice-plan.md`,
  - added artifact verifier `agentic-llm-ft/scripts/verify_vertical_slice.py`,
  - added `make vertical-slice` and `make vertical-slice-verify` automation targets,
  - added tests for vertical-slice fixture + checkpoint-metric behavior (`agentic-llm-ft/tests/test_vertical_slice_eval.py`),
  - updated README with vertical-slice quick commands.

### 2026-03-25
- Added a repository-level implementation plan for a full vertical slice at `agentic-llm-ft/docs/vertical-slice-plan.md`, covering scope, phases, success criteria, risks, and CI automation deliverables.

### 2026-03-15 (continued)
- **Unified model interface:** Eval and agent both use a single **messages-based** interface: `MessageModelFn = Callable[[list[dict]], str]`. Added `render_eval_messages(example)` in `prompts/rendering.py` to build input messages for eval (no assistant turn). Eval harness now takes `model_fn(messages)` and builds messages per example; agent runtime unchanged. Exported `MessageModelFn` from `eval` package.
- **Hydra integration:** All three entry points use Hydra. Added `configs/config.yaml` (single composed config with `hydra.job.chdir: false`). `scripts/train.py`, `scripts/run_eval.py`, and `scripts/inference.py` use `@hydra.main(config_path=<abs path to configs>, config_name="config")`. Config path is resolved from `__file__` so scripts work from any CWD. Overrides via CLI (e.g. `training.epochs=2`, `eval.dataset_path=...`). Root entry point for eval renamed to `run_eval.py` to avoid shadowing the `eval` package when importing; `eval.py` removed.
- **Validation loss and perplexity:** Eval harness now accepts optional `model`, `tokenizer`, and `max_seq_len`. When provided, `_compute_validation_loss()` runs a forward pass over each eval example (full-sequence causal LM), aggregates mean loss, and sets `perplexity = exp(mean_loss)`. Script `run_eval.py` supports `eval.checkpoint_path` in config; when set, loads the checkpoint and passes model/tokenizer to `run_eval()` so metrics include real `validation_loss` and `perplexity`. README and config updated (eval.checkpoint_path, eval.max_seq_len).

### 2026-03-15
- **High-priority fixes (architecture review):**
  - Replaced invalid `ValidationError.from_exception_data` in `ToolRegistry.execute` with a clear `ToolArgumentsError` (ValueError subclass) and `_validation_failure_reason()` for better diagnostics. Exported `ToolArgumentsError` from `tools` package.
  - Implemented tokenization and `max_seq_len` in the training pipeline: added `tokenize_dataset()` in `data/dataset.py` (uses tokenizer `apply_chat_template`, truncation to `max_seq_len`, and produces `input_ids`/`attention_mask`/`labels`), trainer now tokenizes the raw message dataset and uses `DataCollatorForLanguageModeling`; `model.max_seq_len` is read from config and pad_token is set when missing.
  - Documented tokenization and sequence length in README (Training section).
  - Added test for `ToolArgumentsError` on invalid tool arguments.

### 2026-03-07
- Created full project skeleton under `agentic-llm-ft/` with Python packaging, Makefile, and environment templates.
- Added Hydra YAML configs for model/data/training/eval/experiments.
- Implemented typed source modules for:
  - tool schema and output schema models,
  - tool registry/executor and deterministic mock tools,
  - JSON output parser with lightweight repair,
  - dataset normalization and validation,
  - prompt rendering,
  - training/evaluation/inference orchestrators,
  - simulated agent loop with trace logging.
- Added CLI entry scripts (`train.py`, `eval.py`, `inference.py`).
- Added sample canonical dataset examples covering no-tool, single-tool, multi-tool, tool-failure, and argument-validation scenarios.
- Added pytest suite for key behavior and metrics.
- Added comprehensive README with setup, workflows, and extension roadmap.

## Notes for Future Agents
- Keep this file updated for every meaningful change.
- Ensure tests remain lightweight and do not require downloading large models.
- Preserve deterministic behavior of mock tools for reproducible evaluations.
- Added lightweight local compatibility shims under `src/pydantic` and `src/datasets` so tests can run in restricted environments without external package installation. In normal environments these should be replaced by installed dependencies.
- Ran project test suite successfully (`7 passed`).
- Added cross-platform packaging/training updates for macOS support:
  - made `bitsandbytes` a Linux-only dependency marker in `pyproject.toml`,
  - updated trainer to prefer MPS on macOS and to explicitly block QLoRA on macOS with a clear error,
  - added `use_mps`/`no_cuda` training config flags,
  - documented MacBook Pro (Apple Silicon) support and QLoRA limitation in README.
- Added first-class `uv` package-management workflow support:
  - extended `Makefile` with `install-uv`, `lock`, and `sync` targets,
  - updated README setup/testing sections with recommended `uv` commands and pip alternative.
- README refreshed with clearer operator-focused guidance:
  - added end-to-end quickstart (setup, test, eval, inference),
  - added explicit tool schema example and artifact expectations,
  - clarified eval metric placeholders (`validation_loss`, `perplexity`),
  - documented trace row fields,
  - added implementation notes about local `pydantic`/`datasets` shims vs production dependencies.
