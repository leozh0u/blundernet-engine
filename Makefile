PYTHON ?= python3
VENV ?= .venv

.PHONY: setup build lint test check clean
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -r requirements.txt pytest ruff
	$(MAKE) build

build:
	$(VENV)/bin/python cpp/setup.py build_ext --inplace

lint:
	$(VENV)/bin/ruff check .

test:
	$(VENV)/bin/python -m pytest -q

check: lint test

clean:
	rm -rf build cpp/build .pytest_cache .ruff_cache
	find src tests scripts cpp -type d -name __pycache__ -prune -exec rm -rf {} +
