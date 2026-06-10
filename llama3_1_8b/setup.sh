#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing Ollama ==="
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed"
fi

echo ""
echo "=== Pulling Llama-3.1-8B model ==="
ollama pull llama3.1:8b

echo ""
echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Run the experiment with:"
echo "  python llama3_1_8b/run_experiment.py"
