SHELL := /usr/bin/env bash
.PHONY: help deps ollama redis neo4j python setup run clean purge

BACKEND ?= ollama
MODEL_NAME ?=

help:
	@echo "HARMONY-X — Unified Experiment Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  deps     Install system dependencies (Homebrew, Python 3.11+)"
	@echo "  ollama   Install Ollama"
	@echo "  redis    Install & start Redis"
	@echo "  neo4j    Install & start Neo4j"
	@echo "  python   Create UV virtual env & install Python packages"
	@echo "  env      Create .env template"
	@echo "  setup    Run all of the above (full fresh-server setup)"
	@echo "  run      Run the experiment (set BACKEND=openrouter or BACKEND=ollama)"
	@echo "  clean    Remove logs, DBs, caches, and output artifacts"
	@echo "  purge    Remove everything (including venv, configs)"
	@echo ""
	@echo "Variables:"
	@echo "  BACKEND=ollama      Victim backend (ollama|openrouter)"
	@echo "  MODEL_NAME=         Override victim model name in config"

UNAME_S := $(shell uname -s)

brew:
ifeq ($(UNAME_S),Darwin)
	@which brew >/dev/null 2>&1 || { \
		echo "[...] Installing Homebrew..."; \
		/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
	}
	@echo "[OK]   Homebrew $$(brew --version | head -1)"
endif

deps: brew
ifeq ($(UNAME_S),Darwin)
	@which python3 >/dev/null 2>&1 || { echo "[...] Installing Python via Homebrew..."; brew install python@3.11; }
endif
	@python3 --version
	@which uv >/dev/null 2>&1 || { echo "[...] Installing UV..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	@echo "[OK]   UV $$(uv --version 2>/dev/null || echo 'installed')"

ollama:
	@which ollama >/dev/null 2>&1 || { \
		echo "[...] Installing Ollama..."; \
		curl -fsSL https://ollama.com/install.sh | sh; \
	}
	@echo "[OK]   Ollama installed"

redis: brew
ifeq ($(UNAME_S),Darwin)
	@which redis-server >/dev/null 2>&1 || brew install redis
	@redis-cli ping >/dev/null 2>&1 || { \
		echo "[...] Starting Redis..."; \
		brew services start redis 2>/dev/null || redis-server --daemonize yes >/tmp/redis.log 2>&1; \
		sleep 2; \
	}
	@echo "[OK]   Redis $$(redis-cli ping 2>/dev/null)"
endif

neo4j: brew
ifeq ($(UNAME_S),Darwin)
	@which neo4j >/dev/null 2>&1 || brew install neo4j
	@neo4j status 2>/dev/null | grep -q "is running" || { \
		echo "[...] Starting Neo4j..."; \
		brew services start neo4j 2>/dev/null || neo4j start 2>/dev/null; \
		sleep 5; \
	}
	@echo "[OK]   Neo4j $$(neo4j status 2>/dev/null | head -1)"
endif

python:
	@uv sync
	@echo "[OK]   Python packages installed (uv sync)"

env:
	@test -f ../.env || { \
		echo '# HARMONY-X environment' > ../.env; \
		echo 'OPENROUTER_API_KEY=sk-your-key-here' >> ../.env; \
		echo 'GEMMA_API_KEY=' >> ../.env; \
		echo 'HX_REDIS_URL=redis://localhost:6379/0' >> ../.env; \
		echo 'HX_NEO4J_URI=bolt://localhost:7687' >> ../.env; \
		echo 'HX_NEO4J_USER=neo4j' >> ../.env; \
		echo 'HX_NEO4J_PASSWORD=password' >> ../.env; \
		echo 'HARMFUL_CSV=../prompt.csv' >> ../.env; \
		echo "[OK]   Created ../.env template"; \
	}

setup: deps redis neo4j env python
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  All done! Run:"
	@echo "    make run BACKEND=ollama"
	@echo "    make run BACKEND=openrouter"
	@echo "═══════════════════════════════════════════════════════════════"

run:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  HARMONY-X Experiment (BACKEND=$(BACKEND))"
	@echo "═══════════════════════════════════════════════════════════════"
	@cd .. && uv run experiments/run_experiment.py \
		--backend $(BACKEND) \
		$(MODEL_NAME:%=--model-name %)

clean:
	@rm -rf logs/* __pycache__ .venv *.db logv2.log
	@rm -rf ../outputs/campaign/*
	@echo "[OK]   Cleaned artifacts"

purge: clean
	@rm -rf .venv configs
	@echo "[OK]   Purged"
