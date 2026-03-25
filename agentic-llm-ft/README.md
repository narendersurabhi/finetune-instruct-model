# agentic-llm-ft

Production-grade Python foundation for fine-tuning open-source LLMs for **reasoning**, **planning**, and **strict tool-calling** in agentic AI systems.

> This repository is an engineering **starter foundation**: data contracts, tool schema validation,
> agent loop simulation, and evaluation harness are implemented so teams can iterate quickly toward
> real model training and real tool APIs.

## Features

- Canonical dataset format for tool-aware supervised fine-tuning.
- Structured assistant output schema:

```json
{
  "plan": ["string"],
  "tool_calls": [{"name": "string", "arguments": {}}],
  "final_answer": "string or null"
}
```

- Tool abstraction layer with strict argument validation:
  - `ToolSpec`
  - `ToolRegistry`
  - `ToolExecutor`
- Deterministic mock tools for local development:
  - `get_weather`
  - `get_stock_price`
  - `calculator`
  - `search_docs`
  - `retrieve_faq`
  - `calendar_lookup`
- JSON output parser with repair and schema validation.
- Simulated agent runtime loop with trace logging.
- Evaluation harness for tool selection + argument correctness.
- LoRA + optional QLoRA training scaffolding.

---

## Configuration (Hydra)

A single config file `configs/config.yaml` drives train, eval, and inference. Override any key from the CLI:

```bash
python train.py training.epochs=2 model.max_seq_len=2048
python run_eval.py eval.dataset_path=data/my/eval.jsonl eval.checkpoint_path=checkpoints/lora
python inference.py inference.mode=batch inference.input=prompts.txt
```

`hydra.job.chdir` is set to `false` so relative paths in the config are resolved from the directory you run the command in (typically the repo root).

---

## Supported Base Models

Configured via YAML:

- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Llama-3-8B-Instruct`

---

## End-to-End Quickstart

```bash
# 1) environment
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra dev

# 2) run tests
uv run pytest

# 3) run evaluation harness on sample data (config in configs/config.yaml; override with Hydra)
uv run python run_eval.py
# Or: uv run python run_eval.py eval.dataset_path=data/sample/eval.jsonl eval.output_dir=outputs/eval

# 4) run agent loop inference
uv run python inference.py
# Or: uv run python inference.py inference.mode=agent inference.prompt="What's the weather in Seattle?"
```

After step (3), inspect:
- `outputs/eval/metrics.json`
- `outputs/eval/predictions.jsonl`
- `outputs/eval/tool_call_analysis.json`

---


## Vertical Slice (Phased)

Use the dedicated deterministic profile in `configs/experiments/vertical_slice.yaml` to run the full thin-but-complete slice.

```bash
# run all phases (train -> eval -> inference)
make vertical-slice

# verify CI artifact contract
make vertical-slice-verify
```

Detailed phased runbook: `docs/vertical-slice-runbook.md`.

## Repository Layout

```text
agentic-llm-ft/
  configs/
  data/sample/
  scripts/
  src/
  tests/
  train.py
  run_eval.py
  inference.py
```

---

## Setup

### Recommended: `uv` (fast, reproducible)

If you are starting fresh, **yes, use `uv`**. It is faster than plain pip/venv and gives better lockfile-driven reproducibility.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra dev
```

Optional lockfile workflow:

```bash
uv lock
uv sync --frozen --extra dev
```

### Alternative: pip/venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Optional:

```bash
pip install -e .[wandb]
```

### macOS (Apple Silicon / MacBook Pro)

Yes — the repository can run on modern MacBook Pro systems (including Apple Silicon generations) for
data processing, tests, evaluation harness, and LoRA fine-tuning workflows.

- Recommended backend: PyTorch `mps` for LoRA training.
- QLoRA (`bitsandbytes` 4-bit) is Linux-focused and not supported in this project on macOS.
- `pyproject.toml` pins `bitsandbytes` only on Linux so installation on macOS is cleaner.

For Mac training, keep:

```yaml
training:
  qlora: false
  use_mps: true
```

---

## Dataset Format

Each JSONL row uses canonical fields:

```json
{
  "id": "example_1",
  "system_prompt": "...",
  "user_prompt": "...",
  "available_tools": [...],
  "assistant_plan": [...],
  "assistant_tool_calls": [...],
  "tool_results": [...],
  "assistant_final": "..."
}
```

Validation checks include:
- unknown tool names
- invalid JSON arguments
- missing final answers
- malformed tool results

### Tool Schema Example

```json
{
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"},
      "unit": {"type": "string"}
    },
    "required": ["city"]
  }
}
```

Tools are registered and validated by `ToolRegistry`, then executed by `ToolExecutor`.

---

## Training

### LoRA

```bash
python scripts/train.py --config configs/experiments/base.yaml
```

### QLoRA

Set `training.qlora: true` and `compute_dtype` in your YAML.

Training supports:
- gradient checkpointing
- gradient accumulation
- bf16/fp16
- max sequence length through model config
- checkpoint resume
- optional W&B logging

**Tokenization and sequence length:** The training pipeline converts each example’s `messages` (system/user/assistant) into token ids using the tokenizer’s `apply_chat_template`, then truncates to `model.max_seq_len` from your config. The tokenized dataset has `input_ids`, `attention_mask`, and `labels`; batching and padding are handled by `DataCollatorForLanguageModeling`. Set `model.max_seq_len` in your experiment YAML (e.g. `configs/experiments/base.yaml`).

---

## Evaluation

Config is in `configs/config.yaml` (Hydra). Override from the CLI:

```bash
python run_eval.py
# Override paths/metrics:
python run_eval.py eval.dataset_path=data/sample/eval.jsonl eval.output_dir=outputs/eval
# With a checkpoint to compute validation_loss and perplexity:
python run_eval.py eval.checkpoint_path=checkpoints/lora
```

Outputs:
- `metrics.json`
- `predictions.jsonl`
- `tool_call_analysis.json`

Metrics include:
- `validation_loss` (placeholder)
- `perplexity` (placeholder)
- `tool_selection_accuracy`
- `tool_argument_validity_rate`
- `tool_schema_compliance`
- `tool_call_precision`
- `tool_call_recall`
- `final_answer_correctness`

`validation_loss` and `perplexity` are currently placeholders in the local harness and should be
wired to model-forward evaluation when integrating a real checkpoint pipeline.

---

## Inference

### Agent loop mode

```bash
python scripts/inference.py --mode agent --prompt "What's the weather in Seattle?"
```

### Batch mode

```bash
python scripts/inference.py --mode batch --input data/sample/batch_prompts.txt
```

### Interactive mode

```bash
python scripts/inference.py --mode interactive
```

Agent traces are logged to:

```text
outputs/{run_id}/agent_traces.jsonl
```

(For local scripts, run directory is defined by CLI `--out`.)

Trace rows include:
- `raw_model_output`
- `parsed_json`
- `tool_calls`
- `tool_results`
- `final_answer`

---

## Structured Output Contract

- **Pre-tool responses**:
  - `final_answer = null`
- **Final responses**:
  - `tool_calls = []`

`src/agent/output_parser.py` repairs malformed JSON minimally, parses it, and validates against Pydantic schema.

---

## Testing

```bash
uv run pytest
```

Or with pip environment:

```bash
pytest
```

Tests are lightweight and do not download large models.

---

## Implementation Notes

- For constrained/offline environments, this repo currently includes lightweight local compatibility
  shims under `src/pydantic` and `src/datasets` to keep tests runnable without full package installs.
- For production deployment, prefer standard upstream dependencies (`pydantic`, `datasets`) and remove
  shims once environment packaging is stable.

---

## Future Extensions

This baseline is designed to extend to:
- DPO/RLHF
- LangGraph orchestration
- RAG-oriented data generation
- real API tools
- OpenAI-style function calling
