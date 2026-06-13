# ===========================================================================
#  HARMONY-X — MeetAISmall Experiment Makefile
#  Fresh-server setup: installs everything from scratch.
#  Targets: deps, ollama, redis, neo4j, setup, run, clean
# ===========================================================================

SHELL := /usr/bin/env bash
.PHONY: help deps ollama redis neo4j python setup run clean purge

help:
	@echo "HARMONY-X — MeetAISmall Experiment"
	@echo ""
	@echo "Targets:"
	@echo "  deps     Install system dependencies (Homebrew, Python 3.11+)"
	@echo "  ollama   Install Ollama + pull MeetSolanki/MeetAISmall:latest model"
	@echo "  redis    Install & start Redis"
	@echo "  neo4j    Install & start Neo4j"
	@echo "  python   Create UV virtual env & install Python packages"
	@echo "  env      Create .env template"
	@echo "  setup    Run all of the above (full fresh-server setup)"
	@echo "  run      Run the experiment"
	@echo "  clean    Remove logs, DBs, caches, and output artifacts"
	@echo "  purge    Remove everything (including venv, configs)"

# ── Detect OS ──
UNAME_S := $(shell uname -s)

# ── Install Homebrew (macOS) ──
brew:
ifeq ($(UNAME_S),Darwin)
	@which brew >/dev/null 2>&1 || { \
		echo "[...] Installing Homebrew..."; \
		/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
	}
	@echo "[OK]   Homebrew $$(brew --version | head -1)"
endif

# ── System deps ──
deps: brew
ifeq ($(UNAME_S),Darwin)
	@which python3 >/dev/null 2>&1 || { echo "[...] Installing Python via Homebrew..."; brew install python@3.11; }
endif
	@python3 --version
	@which uv >/dev/null 2>&1 || { echo "[...] Installing UV..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	@echo "[OK]   UV $$(uv --version 2>/dev/null || echo 'installed — reload shell or add ~/.cargo/bin to PATH')"

# ── Ollama ──
ollama:
	@which ollama >/dev/null 2>&1 || { \
		echo "[...] Installing Ollama..."; \
		curl -fsSL https://ollama.com/install.sh | sh; \
	}
	@echo "[OK]   Ollama installed"
	@echo "[...] Pulling MeetSolanki/MeetAISmall:latest (this may take a while)..."
	@ollama pull MeetSolanki/MeetAISmall:latest
	@echo "[OK]   Model MeetSolanki/MeetAISmall:latest ready"

# ── Redis ──
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

# ── Neo4j ──
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

# ── Python virtual env (UV) ──
python:
	@uv venv .venv 2>/dev/null; echo "[OK]   Virtual env created at .venv"
	@source .venv/bin/activate && uv pip install -r requirements.txt
	@echo "[OK]   Python packages installed ($$(.venv/bin/python3 -c 'import yaml,neo4j,redis,requests,httpx,numpy; print("all OK")' 2>&1))"

# ── Create .env template ──
env:
	@test -f ../.env || { \
		echo '# HARMONY-X environment' > ../.env; \
		echo 'GEMMA_API_KEY=' >> ../.env; \
		echo 'HX_REDIS_URL=redis://localhost:6379/0' >> ../.env; \
		echo 'HX_NEO4J_URI=bolt://localhost:7687' >> ../.env; \
		echo 'HX_NEO4J_USER=neo4j' >> ../.env; \
		echo 'HX_NEO4J_PASSWORD=password' >> ../.env; \
		echo 'HARMFUL_CSV=../prompt.csv' >> ../.env; \
		echo "[OK]   Created ../.env template — edit GEMMA_API_KEY if needed"; \
	}

# ── Full fresh-server setup ──
setup: deps ollama redis neo4j python env
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  All done! Run the experiment with:"
	@echo "    make run"
	@echo "  or directly:"
	@echo "    source .venv/bin/activate && python run_experiment.py"
	@echo "═══════════════════════════════════════════════════════════════"

# ── Run experiment ──
run:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  HARMONY-X — MeetAISmall Experiment"
	@echo "═══════════════════════════════════════════════════════════════"
	@bash run_exp.sh

# ── Clean artifacts ──
clean:
	@rm -rf logs/* outputs/* __pycache__ .venv *.db logv2.log
	@rm -f ../.env
	@echo "[OK]   Cleaned artifacts"

# ── Purge everything ──
purge: clean
	@rm -rf .venv configs
	@echo "[OK]   Purged — directory is bare"
	@ls -la
