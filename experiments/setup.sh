#!/usr/bin/env bash
# ===========================================================================
#  HARMONY-X — CodeLlama-7B Setup Script
#  Uses UV for Python environment management.
#  Idempotent — safe to run multiple times.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$HERE/.." && pwd)"

echo "═══════════════════════════════════════════════════════════════"
echo "  HARMONY-X — Unified Environment Setup"
echo "  Python:   UV virtual env (uv sync)"
echo "═══════════════════════════════════════════════════════════════"

# ── 1. System dependencies ──
echo ""
echo "── Step 1/5: System dependencies ──"

if [[ "$(uname -s)" == "Darwin" ]]; then
    if ! command -v brew &>/dev/null; then
        echo "[...] Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    echo "[OK]   Homebrew $(brew --version | head -1)"
fi

if ! command -v python3 &>/dev/null; then
    echo "[...] Installing Python 3.11..."
    if [[ "$(uname -s)" == "Darwin" ]]; then
        brew install python@3.11
    else
        echo "[FAIL] Please install Python 3.11+ manually"
        exit 1
    fi
fi
echo "[OK]   Python $(python3 --version)"

# ── 2. UV (Python package manager) ──
echo ""
echo "── Step 2/5: UV package manager ──"
if ! command -v uv &>/dev/null; then
    echo "[...] Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "[OK]   UV $(uv --version 2>/dev/null || echo 'installed')"

# ── 3. Ollama + CodeLlama-7B model ──
echo ""
echo "── Step 3/5: Ollama + codellama:7b ──"
if ! command -v ollama &>/dev/null; then
    echo "[...] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo "[OK]   Ollama installed"

# Start Ollama if not running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "[...] Starting Ollama server..."
    ollama serve >/tmp/ollama.log 2>&1 &
    sleep 3
    for i in $(seq 1 15); do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
fi
echo "[OK]   Ollama server ready"

# Pull model
if ! ollama list 2>/dev/null | grep -q "codellama:7b"; then
    echo "[...] Pulling codellama:7b (this may take a while)..."
    ollama pull codellama:7b
fi
echo "[OK]   Model codellama:7b available"

# ── 4. Redis ──
echo ""
echo "── Step 4/5: Redis ──"
if ! command -v redis-server &>/dev/null; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        brew install redis
    else
        echo "[WARN] Please install redis-server manually"
    fi
fi
if ! redis-cli ping 2>/dev/null | grep -q "PONG"; then
    echo "[...] Starting Redis..."
    if [[ "$(uname -s)" == "Darwin" ]] && command -v brew &>/dev/null; then
        brew services start redis 2>/dev/null || true
    fi
    redis-server --daemonize yes >/tmp/redis.log 2>&1 || true
    sleep 2
fi
echo "[OK]   Redis $(redis-cli ping 2>/dev/null)"

# ── 5. Python virtual env + dependencies (UV) ──
echo ""
echo "── Step 5/5: Python environment (UV) ──"
cd "$PROJECT_DIR"

# Single-command uv setup
uv sync

# Verify key packages
echo "[...] Verifying installations..."
uv run python3 -c "
import yaml, neo4j, redis, requests, httpx, numpy
print('[OK]   All core packages importable')
"

# ── Done ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Run:   cd $HERE && make run BACKEND=ollama"
echo "  Or:    cd $HERE && make run BACKEND=openrouter"
echo "  Or:    uv run python experiments/run_experiment.py --backend ollama"
echo "═══════════════════════════════════════════════════════════════"
