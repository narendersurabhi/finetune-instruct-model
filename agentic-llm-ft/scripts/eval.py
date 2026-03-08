#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval import run_eval
from tools import build_default_registry


class EchoModel:
    def __call__(self, prompt: str) -> str:
        if "weather" in prompt.lower():
            return json.dumps(
                {
                    "plan": ["Use weather tool"],
                    "tool_calls": [{"name": "get_weather", "arguments": {"city": "Seattle"}}],
                    "final_answer": "Weather retrieved.",
                }
            )
        return json.dumps({"plan": ["Answer directly"], "tool_calls": [], "final_answer": "OK"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on tool-calling metrics")
    parser.add_argument("--dataset", default="data/sample/eval.jsonl")
    parser.add_argument("--out", default="outputs/eval")
    args = parser.parse_args()

    metrics = run_eval(
        dataset_path=Path(args.dataset),
        model_fn=EchoModel(),
        registry=build_default_registry(),
        output_dir=Path(args.out),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
