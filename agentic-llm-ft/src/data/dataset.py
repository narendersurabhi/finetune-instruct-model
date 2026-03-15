from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Dataset

from data.validators import validate_example
from prompts.rendering import render_training_messages
from schemas import DatasetExample
from tools import ToolRegistry

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def load_examples(path: Path) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(DatasetExample.model_validate(json.loads(line)))
    return examples


def build_hf_dataset(path: Path, registry: ToolRegistry) -> Dataset:
    examples = load_examples(path)
    records = []
    for ex in examples:
        validate_example(ex, registry)
        records.append({"id": ex.id, "messages": render_training_messages(ex)})
    return Dataset.from_list(records)


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: "PreTrainedTokenizerBase",
    max_seq_len: int,
) -> Dataset:
    """
    Tokenize a dataset whose rows have a "messages" key (list of dicts with "role" and "content").
    Uses the tokenizer's chat template, truncates to max_seq_len, and returns input_ids,
    attention_mask, and labels (same as input_ids for causal LM SFT).
    """

    def tokenize_row(example: dict[str, Any]) -> dict[str, Any]:
        tokenizer_output = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            truncation=True,
            max_length=max_seq_len,
            add_generation_prompt=False,
        )
        if hasattr(tokenizer_output, "tolist"):
            token_ids = tokenizer_output.tolist()
        else:
            token_ids = list(tokenizer_output)
        attention_mask = [1] * len(token_ids)
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": token_ids.copy(),
        }

    return dataset.map(
        tokenize_row,
        remove_columns=dataset.column_names,
        desc="tokenize",
    )
