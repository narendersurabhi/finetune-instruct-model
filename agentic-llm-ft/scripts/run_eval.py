#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from eval import run_eval
from tools import build_default_registry

_CONFIGS_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def _last_user_content(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


class EchoModel:
    """Stub model using the unified messages-based interface."""

    def __call__(self, messages: list[dict]) -> str:
        prompt = _last_user_content(messages)
        if "weather" in prompt.lower():
            return json.dumps(
                {
                    "plan": ["Use weather tool"],
                    "tool_calls": [{"name": "get_weather", "arguments": {"city": "Seattle"}}],
                    "final_answer": "Weather retrieved.",
                }
            )
        return json.dumps({"plan": ["Answer directly"], "tool_calls": [], "final_answer": "OK"})


def _load_model_and_tokenizer(checkpoint_path: str | None, model_name: str | None):
    if not checkpoint_path:
        return None, None
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = str(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
    model.eval()
    return model, tokenizer


@hydra.main(config_path=_CONFIGS_DIR, config_name="config", version_base=None)
def main(cfg) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    eval_cfg = cfg.get("eval", {})
    dataset_path = Path(eval_cfg.get("dataset_path", "data/sample/eval.jsonl"))
    output_dir = Path(eval_cfg.get("output_dir", "outputs/eval"))
    max_seq_len = int(eval_cfg.get("max_seq_len", 4096))
    checkpoint_path = eval_cfg.get("checkpoint_path")
    model_name = cfg.get("model", {}).get("name")

    model, tokenizer = _load_model_and_tokenizer(checkpoint_path, model_name)

    metrics = run_eval(
        dataset_path=dataset_path,
        model_fn=EchoModel(),
        registry=build_default_registry(),
        output_dir=output_dir,
        max_seq_len=max_seq_len,
        model=model,
        tokenizer=tokenizer,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
