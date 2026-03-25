#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_METRIC_KEYS = {
    "exact_match",
    "tool_call_precision",
    "validation_loss",
    "perplexity",
}


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"[vertical-slice] verification failed: {message}")


def _count_jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify vertical-slice artifacts.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/vertical-slice"),
        help="Artifact root for the vertical slice.",
    )
    args = parser.parse_args()

    root = args.root
    checkpoint_dir = root / "checkpoints" / "lora"
    metrics_path = root / "eval" / "metrics.json"
    predictions_path = root / "eval" / "predictions.jsonl"
    traces_path = root / "inference" / "agent_traces.jsonl"

    _ensure(root.exists(), f"artifact root does not exist: {root}")
    _ensure(checkpoint_dir.exists(), f"checkpoint directory missing: {checkpoint_dir}")
    _ensure(metrics_path.exists(), f"metrics artifact missing: {metrics_path}")
    _ensure(predictions_path.exists(), f"predictions artifact missing: {predictions_path}")
    _ensure(traces_path.exists(), f"agent trace artifact missing: {traces_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_METRIC_KEYS - set(metrics.keys()))
    _ensure(not missing, f"metrics.json missing required keys: {missing}")
    _ensure(metrics.get("validation_loss") is not None, "validation_loss is null")
    _ensure(metrics.get("perplexity") is not None, "perplexity is null")

    _ensure(_count_jsonl_rows(predictions_path) > 0, "predictions.jsonl has no rows")
    _ensure(_count_jsonl_rows(traces_path) > 0, "agent_traces.jsonl has no rows")

    print("[vertical-slice] verification passed")


if __name__ == "__main__":
    main()
