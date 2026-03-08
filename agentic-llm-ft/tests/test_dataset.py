from pathlib import Path

from data import build_hf_dataset
from tools import build_default_registry


def test_dataset_normalization() -> None:
    ds = build_hf_dataset(Path("data/sample/train.jsonl"), build_default_registry())
    assert len(ds) >= 5
    assert "messages" in ds.features
