# Compatible with Linux and macOS

.PHONY: help install run lint format clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies via uv"
	@echo "  make run       - Run the game"
	@echo "  make lint      - Run code checks (ruff + mypy)"
	@echo "  make format    - Auto-format code via ruff"
	@echo "  make clean     - Remove cache and temporary files"

install:
	uv sync

run:
	uv run main.py

lint:
	@echo "Running Linter (Ruff)..."
	uv run ruff check .
	@echo "Running Type Checker (MyPy)..."
	uv run mypy .

format:
	uv run ruff format .

clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +