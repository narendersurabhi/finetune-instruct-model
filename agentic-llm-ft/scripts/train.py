#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from tools import build_default_registry
from training import run_sft

_CONFIGS_DIR = str(Path(__file__).resolve().parent.parent / "configs")


@hydra.main(config_path=_CONFIGS_DIR, config_name="config", version_base=None)
def main(cfg) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    run_sft(cfg, registry=build_default_registry())


if __name__ == "__main__":
    main()
