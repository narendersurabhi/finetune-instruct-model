#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference import run_agent_inference
from tools import build_default_registry


def run_interactive(out_dir: Path) -> None:
    registry = build_default_registry()
    while True:
        prompt = input("User> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        answer = run_agent_inference(registry, prompt, out_dir)
        print(f"Assistant> {answer}")


def run_batch(input_path: Path, out_dir: Path) -> None:
    registry = build_default_registry()
    lines = [ln.strip() for ln in input_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    outputs = [{"prompt": p, "answer": run_agent_inference(registry, p, out_dir)} for p in lines]
    print(json.dumps(outputs, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference in interactive, batch, or agent modes")
    parser.add_argument("--mode", choices=["interactive", "batch", "agent"], default="agent")
    parser.add_argument("--input", default="data/sample/batch_prompts.txt")
    parser.add_argument("--out", default="outputs/inference")
    parser.add_argument("--prompt", default="What's the weather in Seattle?")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if args.mode == "interactive":
        run_interactive(out_dir)
    elif args.mode == "batch":
        run_batch(Path(args.input), out_dir)
    else:
        answer = run_agent_inference(build_default_registry(), args.prompt, out_dir)
        print(answer)


if __name__ == "__main__":
    main()
