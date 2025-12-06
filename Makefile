# Credit Risk MLOps - Makefile
# 
# Common commands for development and operations.
# Usage: make <command>

.PHONY: help install dev-install download preprocess train run test lint clean docker-build docker-up docker-down report backup

# Default target
help:
	@echo "Credit Risk MLOps - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install production dependencies"
	@echo "  dev-install    Install with dev dependencies"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  download       Download UCI dataset"
	@echo "  preprocess     Run preprocessing pipeline"
	@echo "  train          Train models with MLflow"
	@echo "  pipeline       Run full DVC pipeline"
	@echo ""
	@echo "API:"
	@echo "  run            Start FastAPI server locally"
	@echo "  test           Run all tests"
	@echo "  lint           Run linter"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-up      Start all services"
	@echo "  docker-down    Stop all services"
	@echo ""
	@echo "Monitoring:"
	@echo "  report         Generate drift report"
	@echo "  backup         Backup PostgreSQL database"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          Remove temporary files"

# ============================================
# Setup
# ============================================
install:
	pip install uv
	uv pip install --system .

dev-install:
	pip install uv
	uv pip install --system -e ".[dev]"

# ============================================
# Data Pipeline
# ============================================
download:
	python src/data/download.py

preprocess:
	python src/data/preprocess.py

train:
	python src/models/train.py

pipeline:
	dvc repro

# ============================================
# API
# ============================================
run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=src

lint:
	ruff check src/ tests/

# ============================================
# Docker
# ============================================
docker-build:
	docker build -t credit-risk-api -f docker/Dockerfile .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

# ============================================
# Monitoring
# ============================================
report:
	python scripts/generate_report.py

backup:
	python scripts/backup_db.py

# ============================================
# Cleanup
# ============================================
clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf src/__pycache__ src/**/__pycache__
	rm -rf tests/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
