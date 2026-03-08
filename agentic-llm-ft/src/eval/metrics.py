from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalRecord:
    target_tools: set[str]
    predicted_tools: set[str]
    valid_argument_calls: int
    total_predicted_calls: int
    schema_compliant_calls: int
    final_answer_correct: bool


def compute_metrics(records: list[EvalRecord]) -> dict[str, float]:
    tp = sum(len(r.target_tools & r.predicted_tools) for r in records)
    fp = sum(len(r.predicted_tools - r.target_tools) for r in records)
    fn = sum(len(r.target_tools - r.predicted_tools) for r in records)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    selection_accuracy = sum(r.target_tools == r.predicted_tools for r in records) / max(len(records), 1)
    arg_validity = sum(r.valid_argument_calls for r in records) / max(
        sum(r.total_predicted_calls for r in records), 1
    )
    schema_compliance = sum(r.schema_compliant_calls for r in records) / max(
        sum(r.total_predicted_calls for r in records), 1
    )
    final_correct = sum(r.final_answer_correct for r in records) / max(len(records), 1)

    return {
        "tool_selection_accuracy": selection_accuracy,
        "tool_argument_validity_rate": arg_validity,
        "tool_schema_compliance": schema_compliance,
        "tool_call_precision": precision,
        "tool_call_recall": recall,
        "final_answer_correctness": final_correct,
    }
