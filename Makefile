SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python
MUTATE_PATHS ?= src/mini_rsa
PYTEST ?= $(PYTHON) -m pytest

.PHONY: help quality lint format type import-smoke test gui security mutation mutation-clean

help:
	@echo "Available targets:"
	@echo "  make quality        Run linting, typing, security, import smoke, and unit tests"
	@echo "  make lint           Run Ruff lint + format and Black style checks"
	@echo "  make type           Run mypy over src/mini_rsa, tests, and rsa.py"
	@echo "  make security       Run bandit and pip-audit"
	@echo "  make import-smoke   Import all public modules to catch circular dependencies"
	@echo "  make test           Run pytest (-m 'not gui') with coverage gate"
	@echo "  make gui            Run GUI-marked pytest suite (requires PyQt6/xvfb on CI)"
	@echo "  make mutation       Run coverage + mutmut on $(MUTATE_PATHS)"
	@echo "  make mutation-clean Remove cached mutmut/Hypothesis artifacts"
	@echo "  make codex-pipeline Run the full automation script (JSON summary in codex-report.json)"

quality: lint type security import-smoke test

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m black --check .

type:
	$(PYTHON) -m mypy src/mini_rsa tests rsa.py

security:
	$(PYTHON) -m bandit -r src/mini_rsa rsa.py
	@if [ -z "$$SKIP_PIP_AUDIT" ]; then \
		$(PYTHON) -m pip_audit -r requirements.txt; \
	else \
		echo "Skipping pip-audit (SKIP_PIP_AUDIT set)"; \
	fi

import-smoke:
	$(PYTHON) -m pytest tests/unit/test_imports.py

test:
	$(PYTHON) -m pytest -m "not gui" --randomly-seed=1 --cov=mini_rsa --cov-report=xml --cov-fail-under=90

gui:
	QT_QPA_PLATFORM=$${QT_QPA_PLATFORM:-offscreen} $(PYTEST) -m gui

mutation:
	$(PYTHON) -m pytest -q --cov=mini_rsa
	@mut_paths="$${MUTATE_PATHS:-src/mini_rsa}"; \
	echo "Running mutmut on $$mut_paths"; \
	$(PYTHON) -m mutmut run --CI --use-coverage --paths-to-mutate $$mut_paths --tests-dir tests | tee mutmut.log
	$(PYTHON) -m mutmut results | tee mutmut.results || true

mutation-clean:
	rm -rf mutants/.hypothesis mutants/mutmut.log mutants/mutmut.results mutants/mutmut-stats.json

codex-pipeline:
	@SKIP_MUTATION=$(SKIP_MUTATION) \
	SKIP_HOOKS=$(SKIP_HOOKS) \
	SKIP_PIP_AUDIT=$(SKIP_PIP_AUDIT) \
	MUTATE_PATHS="$(MUTATE_PATHS)" \
	OUTPUT_PATH="$(OUTPUT)" \
	$(PYTHON) scripts/run_quality.py
