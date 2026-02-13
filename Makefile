# VDD Makefile
# Convenience commands for development and reproducibility

.PHONY: install install-dev test lint format experiments paper clean help

# Default target
help:
	@echo "VDD Development Commands"
	@echo "========================"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install package"
	@echo "  make install-dev  - Install with dev dependencies"
	@echo "  make install-all  - Install all dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linter"
	@echo "  make format       - Format code"
	@echo ""
	@echo "Experiments:"
	@echo "  make experiments  - Run all experiments"
	@echo "  make exp-core     - Run core experiments (1-8)"
	@echo "  make exp-extended - Run extended experiments (9-15)"
	@echo "  make exp-realworld- Run real-world experiments (16-20)"
	@echo ""
	@echo "Paper:"
	@echo "  make paper        - Generate paper PDF"
	@echo "  make paper-all    - Generate all paper formats"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove generated files"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

# Development
test:
	pytest tests/ -v --cov=src/vdd --cov-report=term-missing

lint:
	ruff check src/ experiments/ tests/

format:
	black src/ experiments/ tests/
	ruff check --fix src/ experiments/ tests/

# Experiments
experiments:
	python run_experiments.py --all

exp-core:
	python run_experiments.py --core

exp-extended:
	python run_experiments.py --extended

exp-realworld:
	python run_experiments.py --realworld

exp-list:
	python run_experiments.py --list

# Paper generation
paper:
	tectonic paper_v4.tex

paper-all:
	tectonic paper_v4.tex
	pandoc paper_v4.tex -o paper_v4.docx
	pandoc paper_v4.tex -o paper_v4.html --standalone
	pandoc paper_v4.tex -o paper_v4.md

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .coverage
	rm -rf src/vdd/__pycache__ src/vdd/**/__pycache__
	rm -rf experiments/__pycache__ tests/__pycache__
	rm -rf *.egg-info build dist
	rm -f *.aux *.log *.out *.toc *.bbl *.blg

clean-results:
	rm -f results/*.png results/*.json results/*.csv results/*.txt

# Quick verification
verify:
	@echo "Verifying VDD installation..."
	python -c "from vdd.drift_detection import EmbeddingDistanceDetector; print('✓ Drift detection')"
	python -c "from vdd.memory import VDDMemoryBank; print('✓ Memory bank')"
	python -c "from vdd.retrieval import VDDRetriever; print('✓ Retriever')"
	@echo "All imports successful!"
