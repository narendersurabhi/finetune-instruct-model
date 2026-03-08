# agentic-llm-ft

Production-grade Python foundation for fine-tuning open-source LLMs for **reasoning**, **planning**, and **strict tool-calling** in agentic AI systems.

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

## Supported Base Models

Configured via YAML:

- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Llama-3-8B-Instruct`

---

## Repository Layout

```text
agentic-llm-ft/
  configs/
  data/sample/
  scripts/
  src/
  tests/
  train.py
  eval.py
  inference.py
```

---

## Setup

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

---

## Evaluation

```bash
python scripts/eval.py --dataset data/sample/eval.jsonl --out outputs/eval
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
pytest
```

Tests are lightweight and do not download large models.

---

## Future Extensions

This baseline is designed to extend to:
- DPO/RLHF
- LangGraph orchestration
- RAG-oriented data generation
- real API tools
- OpenAI-style function calling
