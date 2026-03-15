from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from data import build_hf_dataset, tokenize_dataset
from training.lora import build_lora_config
from tools import ToolRegistry


def run_sft(cfg: dict[str, Any], registry: ToolRegistry) -> None:
    model_name = cfg["model"]["name"]
    max_seq_len = cfg["model"].get("max_seq_len", 4096)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_macos = platform.system() == "Darwin"
    use_qlora = cfg["training"].get("qlora", False)
    if use_qlora and is_macos:
        raise ValueError(
            "QLoRA with bitsandbytes is not supported on macOS. Use LoRA on MPS/CPU instead."
        )

    use_mps = is_macos and cfg["training"].get("use_mps", True)
    device_map = "mps" if use_mps else "auto"

    quant_cfg = None
    if use_qlora:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=cfg["training"].get("compute_dtype", "bfloat16"),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map=device_map,
    )

    lora = build_lora_config(
        r=cfg["training"]["lora_r"],
        alpha=cfg["training"]["lora_alpha"],
        dropout=cfg["training"]["lora_dropout"],
        target_modules=cfg["training"]["target_modules"],
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()

    raw_dataset = build_hf_dataset(Path(cfg["data"]["train_path"]), registry)
    dataset = tokenize_dataset(raw_dataset, tokenizer=tokenizer, max_seq_len=max_seq_len)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        bf16=cfg["training"].get("bf16", False),
        fp16=cfg["training"].get("fp16", False),
        no_cuda=cfg["training"].get("no_cuda", False),
        use_mps_device=use_mps,
        logging_steps=cfg["training"].get("logging_steps", 10),
        save_steps=cfg["training"].get("save_steps", 200),
        resume_from_checkpoint=cfg["training"].get("resume_from_checkpoint"),
        report_to=["wandb"] if cfg["training"].get("wandb", False) else [],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=cfg["training"].get("resume_from_checkpoint"))
    trainer.save_model(cfg["training"]["output_dir"])
