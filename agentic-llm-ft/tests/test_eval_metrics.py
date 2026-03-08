from eval.metrics import EvalRecord, compute_metrics


def test_evaluation_metrics() -> None:
    records = [
        EvalRecord(
            target_tools={"get_weather"},
            predicted_tools={"get_weather"},
            valid_argument_calls=1,
            total_predicted_calls=1,
            schema_compliant_calls=1,
            final_answer_correct=True,
        )
    ]
    metrics = compute_metrics(records)
    assert metrics["tool_selection_accuracy"] == 1.0
    assert metrics["tool_argument_validity_rate"] == 1.0
