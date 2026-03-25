PYTHON ?= python3
UV ?= uv

.PHONY: install install-uv lock sync test train eval infer format vertical-slice vertical-slice-verify

install:
	$(PYTHON) -m pip install -e .[dev]

install-uv:
	$(UV) sync --extra dev

lock:
	$(UV) lock

sync:
	$(UV) sync --frozen --extra dev

test:
	pytest

train:
	$(PYTHON) scripts/train.py

eval:
	$(PYTHON) run_eval.py

infer:
	$(PYTHON) scripts/inference.py

format:
	ruff check . --fix

vertical-slice:
	$(PYTHON) scripts/train.py --config-name experiments/vertical_slice
	$(PYTHON) run_eval.py --config-name experiments/vertical_slice
	$(PYTHON) scripts/inference.py --config-name experiments/vertical_slice inference.mode=agent inference.prompt="What's the weather in Seattle?"

vertical-slice-verify:
	$(PYTHON) scripts/verify_vertical_slice.py --root outputs/vertical-slice
