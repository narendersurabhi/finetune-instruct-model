from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from agent import OutputParser
from data import load_examples
from eval.metrics import EvalRecord, compute_metrics
from tools import ToolRegistry

ModelFn = Callable[[str], str]


def run_eval(dataset_path: Path, model_fn: ModelFn, registry: ToolRegistry, output_dir: Path) -> dict:
    examples = load_examples(dataset_path)
    parser = OutputParser()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[EvalRecord] = []
    predictions: list[dict] = []
    analyses: list[dict] = []

    for ex in examples:
        raw = model_fn(ex.user_prompt)
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

    metrics = {
        "validation_loss": None,
        "perplexity": None,
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
