.PHONY: help server server-pull local

# ── EMP+T Experiment (DAN template on RMCBench) ──────────────────────────

help:
	@echo "Usage:"
	@echo "  make server     Run EMP+T on codellama:7b + codellama:34b via Ollama"
	@echo "  make local      Run EMP+T on 6 models via OpenRouter"

server-pull:
	ollama pull codellama:7b
	ollama pull codellama:34b

server: server-pull
	python3 experiments/emp_t_experiment.py --where server

local:
	python3 experiments/emp_t_experiment.py --where local

