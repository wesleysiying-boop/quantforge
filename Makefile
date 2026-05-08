.PHONY: help install dev test lint fmt typecheck cov clean run-example dashboard

help:
	@echo "quantforge — common tasks"
	@echo "  make install      install runtime deps"
	@echo "  make dev          install dev deps (lint, test, type)"
	@echo "  make test         run tests"
	@echo "  make cov          run tests with coverage"
	@echo "  make lint         ruff check"
	@echo "  make fmt          ruff format"
	@echo "  make typecheck    mypy strict"
	@echo "  make run-example  run the SMA crossover backtest"
	@echo "  make dashboard    launch Streamlit results dashboard"
	@echo "  make clean        remove build artifacts"

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[all]"

test:
	pytest

cov:
	pytest --cov-report=html

lint:
	ruff check .

fmt:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src

run-example:
	python examples/01_sma_crossover.py

dashboard:
	streamlit run src/quantforge/viz/dashboard.py

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
