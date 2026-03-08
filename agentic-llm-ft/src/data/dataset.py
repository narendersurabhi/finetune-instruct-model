from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from data.validators import validate_example
from prompts.rendering import render_training_messages
from schemas import DatasetExample
from tools import ToolRegistry


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
