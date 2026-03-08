from __future__ import annotations

from peft import LoraConfig


def build_lora_config(r: int, alpha: int, dropout: float, target_modules: list[str]) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
