PYTHON ?= python3
UV ?= uv

.PHONY: install install-uv lock sync test train eval infer format

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
	$(PYTHON) scripts/eval.py

infer:
	$(PYTHON) scripts/inference.py

format:
	ruff check . --fix
