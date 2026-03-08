# Agent Change Log & Repository Status

This file tracks all substantial repository changes and should be read by all agents before modifying code.

## Current Status
- Repository scaffold created for `agentic-llm-ft/`.
- Initial production-style modules added for configuration, schema validation, tools, dataset handling, training, evaluation, inference, and simulated agent runtime.
- Sample datasets and test suite added.

## Change Log
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
