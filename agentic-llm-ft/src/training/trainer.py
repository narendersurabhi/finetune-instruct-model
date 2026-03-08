from __future__ import annotations

from pathlib import Path
from typing import Any

from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from data import build_hf_dataset
from training.lora import build_lora_config
from tools import ToolRegistry


def run_sft(cfg: dict[str, Any], registry: ToolRegistry) -> None:
    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quant_cfg = None
    if cfg["training"].get("qlora", False):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=cfg["training"].get("compute_dtype", "bfloat16"),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map="auto",
    )

    lora = build_lora_config(
        r=cfg["training"]["lora_r"],
        alpha=cfg["training"]["lora_alpha"],
        dropout=cfg["training"]["lora_dropout"],
        target_modules=cfg["training"]["target_modules"],
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()

    dataset = build_hf_dataset(Path(cfg["data"]["train_path"]), registry)

    args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        bf16=cfg["training"].get("bf16", False),
        fp16=cfg["training"].get("fp16", False),
        logging_steps=cfg["training"].get("logging_steps", 10),
        save_steps=cfg["training"].get("save_steps", 200),
        resume_from_checkpoint=cfg["training"].get("resume_from_checkpoint"),
        report_to=["wandb"] if cfg["training"].get("wandb", False) else [],
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train(resume_from_checkpoint=cfg["training"].get("resume_from_checkpoint"))
    trainer.save_model(cfg["training"]["output_dir"])
