#!/usr/bin/env bash
# ===========================================================================
#  HARMONY-X — Llama-3-8B-Instruct Setup Script
#  Uses UV for Python environment management.
#  No Ollama — victim calls OpenRouter API directly.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$HERE/.." && pwd)"

echo "═══════════════════════════════════════════════════════════════"
echo "  HARMONY-X — Llama-3-8B-Instruct Environment Setup"
echo "  Victim:   meta-llama/llama-3-8b-instruct (via OpenRouter)"
echo "  Python:   UV virtual env"
echo "═══════════════════════════════════════════════════════════════"

# ── 1. System dependencies ──
echo ""
echo "── Step 1/4: System dependencies ──"
if ! command -v python3 &>/dev/null; then
    echo "[FAIL] Please install Python 3.11+"
    exit 1
fi
echo "[OK]   Python $(python3 --version)"

# ── 2. UV (Python package manager) ──
echo ""
echo "── Step 2/4: UV package manager ──"
if ! command -v uv &>/dev/null; then
    echo "[...] Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "[OK]   UV $(uv --version 2>/dev/null || echo 'installed')"

# ── 3. Redis ──
echo ""
echo "── Step 3/4: Redis ──"
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

# ── 4. Python virtual env + dependencies (UV) ──
echo ""
echo "── Step 4/4: Python environment (UV) ──"
cd "$HERE"

if [ ! -d ".venv" ]; then
    uv venv .venv
    echo "[OK]   Created .venv"
fi

source .venv/bin/activate
uv pip install -r requirements.txt

python3 -c "
import yaml, neo4j, redis, requests, httpx, numpy
print('[OK]   All core packages importable')
"

# ── Done ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Ensure OPENROUTER_API_KEY is set in $(dirname "$PROJECT_DIR")/.env:"
echo "    OPENROUTER_API_KEY=sk-..."
echo ""
echo "  Activate env:  source $HERE/.venv/bin/activate"
echo "  Run:           cd $HERE && python run_experiment.py"
echo "═══════════════════════════════════════════════════════════════"
