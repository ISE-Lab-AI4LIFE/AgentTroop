# HARMONY-X

Automated safety layer reverse-engineering for LLMs.

## Structure

```
HARMONY-X/
├── victim/                          Victim implementations
│   ├── openrouter/victim.py         OpenRouter API victim
│   └── ollama/victim.py             Local Ollama victim
├── experiments/                     Experiment runner code
│   ├── run_experiment.py            Unified entry point
│   ├── run_exp.sh                   Shell runner (services + experiment)
│   ├── Makefile                     Build targets
│   ├── setup.sh                     Fresh-server setup
│   ├── configs/
│   │   ├── openrouter_experiment_config.yaml
│   │   └── ollama_experiment_config.yaml
│   └── run_guided_asr.py, run_adversarial_asr.py, ...
├── outputs/campaign/                Experiment outputs (.db, .json)
├── data/                            Shared data files
│   └── benign_prompts.csv
├── adapters/                        Base classes & victim factory
│   ├── base_victim.py
│   └── victim_factory.py
├── agents/                          RL agents
├── core/                            Core pipeline
├── evaluation/                      Evaluators (RQ0, RQ1, RQ2, ASR)
├── llm/                             LLM client (OpenRouter)
├── knowledge/                       Knowledge stores
├── orchestration/                   Orchestrator
├── pyproject.toml                   Project config (uv)
└── .env                             Environment variables
```

## Quick Start (fresh server)

```bash
# 1. Clone
git clone <repo> && cd HARMONY-X

# 2. One-command setup — installs uv, Python deps, Redis, Neo4j, Ollama, .env
bash experiments/setup.sh

# 3. Run experiment
make -C experiments run BACKEND=ollama
```

## Setup Details

### Prerequisites

- macOS (Linux support: manual service installation)
- Python 3.10+

### Single-command setup

```bash
bash experiments/setup.sh
```

This runs:
1. Homebrew + Python 3.11+ (if missing)
2. UV package manager (if missing)
3. Redis (install & start)
4. Neo4j (install & start)
5. `.env` template creation
6. `uv sync` — creates `.venv`, installs all dependencies

### Manual setup (if you prefer step-by-step)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Start services
brew services start redis
brew services start neo4j     # or: neo4j start

# Create .env
cat > .env << EOF
OPENROUTER_API_KEY=sk-your-key-here
HX_REDIS_URL=redis://localhost:6379/0
HX_NEO4J_URI=bolt://localhost:7687
HX_NEO4J_USER=neo4j
HX_NEO4J_PASSWORD=password
HARMFUL_CSV=prompt.csv
EOF
```

## Running Experiments

### Via Makefile (recommended)

```bash
# Ollama victim (local, e.g. llama3.1:8b, codellama:7b, meetai-small)
make -C experiments run BACKEND=ollama

# OpenRouter victim (API, e.g. meta-llama/llama-3.2-3b-instruct)
make -C experiments run BACKEND=openrouter MODEL_NAME="meta-llama/llama-3.2-3b-instruct"

# Custom seed count
make -C experiments run BACKEND=ollama  # edit NUM_SEEDS in run_exp.sh or pass --num-seeds

# Custom config
make -C experiments run BACKEND=openrouter  # edit CONFIG_FILE in run_exp.sh or pass --config
```

### Via uv run (no Makefile)

```bash
# Ollama
uv run experiments/run_experiment.py --backend ollama

# OpenRouter
uv run experiments/run_experiment.py \
    --backend openrouter \
    --model-name "meta-llama/llama-3.2-3b-instruct"

# With custom seed count
uv run experiments/run_experiment.py --backend ollama --num-seeds 100

# With prior campaign for RQ2 transfer evaluation
uv run experiments/run_experiment.py --backend ollama --prior-campaign <campaign_id>
```

### Via shell script (with service management)

```bash
# Full pipeline with Redis/Neo4j checks
bash experiments/run_exp.sh --backend ollama

# With custom config
bash experiments/run_exp.sh \
    --backend openrouter \
    --model-name "meta-llama/llama-3.2-3b-instruct" \
    --num-seeds 50
```

## Adding a New Victim Model

### Ollama (local)

1. Pull the model: `ollama pull <model_name>`
2. Run with the new model name:

```bash
uv run experiments/run_experiment.py \
    --backend ollama \
    --model-name "<model_name>"
```

The config default in `experiments/configs/ollama_experiment_config.yaml` will be used.
To persist the model name as default, edit `ollama_experiment_config.yaml`:

```yaml
victim:
  ollama_url: "http://localhost:11434"
  model_name: "<model_name>"    # change here
  temperature: 0.0
  max_tokens: 150
```

### OpenRouter (API)

1. Set `OPENROUTER_API_KEY` in `.env`
2. Run with the new model name:

```bash
uv run experiments/run_experiment.py \
    --backend openrouter \
    --model-name "<openrouter-model-slug>"
```

To persist, edit `experiments/configs/openrouter_experiment_config.yaml`:

```yaml
victim:
  model_name: "<openrouter-model-slug>"   # change here
  temperature: 0.0
  max_tokens: 150
```

**No new directory needed** — just supply `--model-name`.

## Outputs

All experiment results go to `outputs/campaign/<campaign_id>/`:

```
outputs/campaign/<campaign_id>/
├── <campaign_id>_episodic.db         Episodic memory database
├── final_program.json                 Best learned defense program
├── final_theory.json                  Learned theories
├── interventions_history.json         All intervention attempts
├── hypotheses_history.json            Experiment summary
├── sde_evidence.json                  Semantic discovery evidence
└── evaluation/
    └── evaluation_report.json         RQ0, RQ1, RQ2, ASR results
```

## Evaluation Reports

After each experiment, an evaluation report is saved to `outputs/campaign/<campaign_id>/evaluation/evaluation_report.json` containing:

| Metric | Description |
|--------|-------------|
| **ASR** | Baseline attack success rate on raw harmful prompts |
| **Harmony ASR** | ASR through Red Team Agent refinement pipeline |
| **Adversarial ASR** | ASR using the learned program to craft prompts |
| **RQ0** | Program accuracy on held-out test set |
| **RQ1** | Intervention efficiency (program prediction accuracy) |
| **RQ2** | Transfer speed vs. prior campaign |

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `OPENROUTER_API_KEY not set` | Missing `.env` | `cp .env.template .env` and add key |
| `Neo4j connection refused` | Neo4j not running | `brew services start neo4j` |
| `Redis connection refused` | Redis not running | `brew services start redis` |
| `ollama: command not found` | Ollama not installed | `brew install ollama` or run `setup.sh` |
| ASR = 0, all REFUSE | Invalid model name / API key | Check `OPENROUTER_API_KEY` and `--model-name` |
