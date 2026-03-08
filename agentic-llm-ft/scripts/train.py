#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from config.loader import load_config
from tools import build_default_registry
from training import run_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA/QLoRA model for agentic tool-calling")
    parser.add_argument("--config", default="configs/experiments/base.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    run_sft(cfg, registry=build_default_registry())


if __name__ == "__main__":
    main()
