PYTHON ?= python3

.PHONY: install test train eval infer format

install:
	$(PYTHON) -m pip install -e .[dev]

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
