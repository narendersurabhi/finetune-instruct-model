from __future__ import annotations

import json
from pathlib import Path

from eval import run_eval
from tools import build_default_registry


def test_vertical_slice_eval_fixture_loads() -> None:
    rows = [
        json.loads(line)
        for line in Path("data/sample/vertical_slice_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert rows[0]["id"] == "vs_weather"


def test_run_eval_writes_required_metrics_with_checkpoint_mode(monkeypatch, tmp_path) -> None:
    def fake_validation_loss(*args, **kwargs):
        return 0.42, 1.52

    monkeypatch.setattr("eval.harness._compute_validation_loss", fake_validation_loss)

    def model_fn(messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"].lower()
        if "weather" in prompt:
            return json.dumps(
                {
                    "plan": ["Use tool"],
                    "tool_calls": [{"name": "get_weather", "arguments": {"city": "Seattle"}}],
                    "final_answer": "Seattle is sunny and 22°C.",
                }
            )
        return json.dumps(
            {
                "plan": ["Answer directly"],
                "tool_calls": [],
                "final_answer": "Hello! It's great to meet you.",
            }
        )

    metrics = run_eval(
        dataset_path=Path("data/sample/vertical_slice_eval.jsonl"),
        model_fn=model_fn,
        registry=build_default_registry(),
        output_dir=tmp_path,
        model=object(),
        tokenizer=object(),
        max_seq_len=128,
    )

    assert "exact_match" in metrics
    assert "tool_call_precision" in metrics
    assert metrics["validation_loss"] == 0.42
    assert metrics["perplexity"] == 1.52
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "predictions.jsonl").exists()
