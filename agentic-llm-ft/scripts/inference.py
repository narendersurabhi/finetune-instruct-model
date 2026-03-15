#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from inference import run_agent_inference
from tools import build_default_registry

_CONFIGS_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def run_interactive(out_dir: Path, system_prompt: str) -> None:
    registry = build_default_registry()
    while True:
        prompt = input("User> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        answer = run_agent_inference(registry, prompt, out_dir, system_prompt=system_prompt)
        print(f"Assistant> {answer}")


def run_batch(input_path: Path, out_dir: Path, system_prompt: str) -> None:
    registry = build_default_registry()
    lines = [ln.strip() for ln in input_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    outputs = [
        {"prompt": p, "answer": run_agent_inference(registry, p, out_dir, system_prompt=system_prompt)}
        for p in lines
    ]
    print(json.dumps(outputs, indent=2))


@hydra.main(config_path=_CONFIGS_DIR, config_name="config", version_base=None)
def main(cfg) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    inf_cfg = cfg.get("inference", {})
    out_dir = Path(inf_cfg.get("output_dir", "outputs/inference"))
    system_prompt = inf_cfg.get("system_prompt", "You are an agentic assistant.")
    mode = inf_cfg.get("mode", "agent")
    prompt = inf_cfg.get("prompt", "What's the weather in Seattle?")
    input_path = Path(inf_cfg.get("input", "data/sample/batch_prompts.txt"))

    if mode == "interactive":
        run_interactive(out_dir, system_prompt)
    elif mode == "batch":
        run_batch(input_path, out_dir, system_prompt)
    else:
        answer = run_agent_inference(
            build_default_registry(), prompt, out_dir, system_prompt=system_prompt
        )
        print(answer)


if __name__ == "__main__":
    main()
