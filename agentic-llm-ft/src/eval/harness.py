from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from agent import OutputParser
from data import load_examples
from eval.metrics import EvalRecord, compute_metrics
from prompts import render_eval_messages, render_training_messages
from schemas import DatasetExample
from tools import ToolRegistry

# Unified model interface: both eval and agent use messages in, text out.
MessageModelFn = Callable[[list[dict[str, str]]], str]

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _compute_validation_loss(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    examples: list[DatasetExample],
    max_seq_len: int,
) -> tuple[float, float]:
    """Compute mean validation loss and perplexity over examples (full-sequence causal LM)."""
    import torch

    model.eval()
    total_loss = 0.0
    n = 0
    for ex in examples:
        messages = render_training_messages(ex)
        tokenizer_output = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            truncation=True,
            max_length=max_seq_len,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        if tokenizer_output.dim() == 1:
            tokenizer_output = tokenizer_output.unsqueeze(0)
        input_ids = tokenizer_output.to(model.device)
        labels = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
        n += 1
    if n == 0:
        return float("nan"), float("nan")
    mean_loss = total_loss / n
    perplexity = float(__import__("math").exp(mean_loss))
    return mean_loss, perplexity


def run_eval(
    dataset_path: Path,
    model_fn: MessageModelFn,
    registry: ToolRegistry,
    output_dir: Path,
    *,
    model: "PreTrainedModel | None" = None,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
    max_seq_len: int = 4096,
) -> dict[str, Any]:
    examples = load_examples(dataset_path)
    parser = OutputParser()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[EvalRecord] = []
    predictions: list[dict] = []
    analyses: list[dict] = []

    for ex in examples:
        messages = render_eval_messages(ex)
        raw = model_fn(messages)
        parsed = parser.parse(raw)
        predicted_tools = {c.name for c in parsed.tool_calls}
        target_tools = {c["name"] for c in ex.assistant_tool_calls}

        valid_calls = 0
        for call in parsed.tool_calls:
            if registry.validate_arguments(call.name, call.arguments):
                valid_calls += 1

        record = EvalRecord(
            target_tools=target_tools,
            predicted_tools=predicted_tools,
            valid_argument_calls=valid_calls,
            total_predicted_calls=len(parsed.tool_calls),
            schema_compliant_calls=valid_calls,
            final_answer_correct=parsed.final_answer == ex.assistant_final,
        )
        records.append(record)
        predictions.append({"id": ex.id, "raw": raw, "parsed": parsed.model_dump()})
        analyses.append(
            {
                "id": ex.id,
                "target_tools": sorted(target_tools),
                "predicted_tools": sorted(predicted_tools),
                "valid_calls": valid_calls,
                "total_calls": len(parsed.tool_calls),
            }
        )

    validation_loss: float | None = None
    perplexity: float | None = None
    if model is not None and tokenizer is not None:
        validation_loss, perplexity = _compute_validation_loss(
            model, tokenizer, examples, max_seq_len
        )

    metrics: dict[str, Any] = {
        "validation_loss": validation_loss,
        "perplexity": perplexity,
        **compute_metrics(records),
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")
    (output_dir / "tool_call_analysis.json").write_text(
        json.dumps(analyses, indent=2), encoding="utf-8"
    )
    return metrics
