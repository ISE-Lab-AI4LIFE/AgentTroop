# AgentTroop

**Response-Aware Multi-Agent Attacks for Eliciting Malicious Code**

AgentTroop is a **response-aware multi-agent framework** for conducting adaptive adversarial attacks against code-generating LLMs. Rather than treating each interaction as an isolated attempt, AgentTroop continuously learns from victim responses, maintains explicit hypotheses about the target's defensive behavior, and refines future attack strategies accordingly.

## How It Works

AgentTroop maintains a **Bayesian Version Space** $\mathcal{V}$ containing candidate defense programs $\{d_i\}$, each representing a possible explanation of the victim's response. A belief distribution is maintained over these candidates and continuously updated through Bayesian inference:

$$P(d_i | e_t) = \frac{P(e_t | d_i) P(d_i)}{\sum_{d_j \in \mathcal{D}} P(e_t | d_j) P(d_j)}$$

At each iteration, the framework designs a **probing prompt** (intervention) to discriminate among alternative defense programs, executes it against the victim, and updates beliefs. Interventions are selected via **Expected Free Energy (EFE)** minimization to maximize information gain about the victim's decision boundary:

$$G(I) = \underbrace{\mathbb{E}_{o \sim P(o|I)} \left[ D_{KL}[b_{t+1} \| b_t] \right]}_{\text{epistemic value}}$$

## Architecture

AgentTroop uses 5 LLM-based agents coordinated through the shared Version Space:

```
┌──────────────────────────────────────────────────┐
│                 Orchestrator Agent                │
│  Schedules agents, maintains Version Space,       │
│  Bayesian belief update, convergence check        │
└──────┬──────────┬──────────┬──────────┬──────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Cognitive │ │Strategist│ │Researcher│ │Red Team  │
│ Agent    │ │ Agent    │ │ Agent    │ │ Agent    │
├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤
│Detect    │ │Identify  │ │Synthesize│ │Generate  │
│behavioral│ │competing │ │refined   │ │executable│
│incon-    │ │defense   │ │defense   │ │probing   │
│sistencies│ │programs  │ │programs  │ │prompts   │
│→ generate│ │→ formulate│ │→ inject  │ │via       │
│hypotheses│ │inter-    │ │into V    │ │jailbreak │
│& programs│ │ventions  │ │          │ │techniques│
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

| Agent | Role |
|-------|------|
| **Orchestrator** | Collects interaction traces, schedules agents, maintains Version Space $\mathcal{V}$, performs Bayesian belief updates, detects convergence |
| **Cognitive Agent** | Analyzes interaction traces, identifies behavioral inconsistencies (same base prompt, different outcomes under transformation), generates defense hypotheses |
| **Strategist Agent** | Identifies competing defense programs via posterior belief, performs symbolic analysis to find distinguishing conditions, formulates intervention objectives |
| **Researcher Agent** | Periodically consolidates interaction evidence (every $N$ interventions), performs evolutionary synthesis to generate new candidate programs, injects them into Version Space |
| **Red Team Agent** | Refines probing prompts using jailbreak transformations (21 techniques via UCB bandit selection), semantic reframing, role-playing, and prompt engineering |

## Core Loop

```
Seed Prompts → Behavioral Variants → Victim Probing →
Anomaly Detection → Hypothesis Generation → Version Space Init →
[Main Loop]
  Competing Program Selection → Intervention Design (EFE) →
  Red Team Refinement → Victim Probing → Outcome Classification →
  Belief Update → Convergence Check (entropy & accuracy)
  [Every N: Researcher Synthesis → Inject New Programs]
```

## Key Components

| Module | Purpose |
|--------|---------|
| `core/primitive.py` | 92 primitives: 27 Predicates, 38 Transforms, 27 Classifiers for building defense programs |
| `core/program.py` | AST-based defense program representation (IfThenElse, And, Or, Not, Threshold) |
| `core/executor.py` | ProgramExecutor — evaluates defense programs against prompts |
| `core/jailbreak.py` | 21 jailbreak techniques with templates (DAN, GCG, hex_injection, persona, etc.) and UCB1 bandit selection |
| `inference/version_space.py` | Bayesian Version Space: top-K candidate programs, posterior beliefs, entropy, information gain |
| `inference/efe.py` | Expected Free Energy computation for intervention selection |
| `inference/belief_updater.py` | Bayesian belief update with soft likelihood ($\varepsilon = 0.1$) |
| `agents/cognitive.py` | Behavioral anomaly detection (group-by-base-prompt, entropy scoring, adaptive percentile) and hypothesis generation |
| `agents/strategist.py` | Disagreement-driven intervention design (Δ scoring → EFE rescore → semantic rescore), program prediction, technique selection |
| `agents/researcher.py` | Evolutionary synthesis (pop=100, gen=30, mut=0.2, cross=0.7) and program verification |
| `agents/red_team.py` | Jailbreak-aware probing prompt refinement |
| `orchestration/orchestrator.py` | 6-phase loop coordination, belief management, convergence detection |
| `synthesis/` | Evolutionary synthesizer (genetic programming), verifier |
| `evaluation/` | Evaluators for RQ0 (FA), RQ1 (PBA, NI), RQ2 (MR), RQ3 (ablation) |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Attack Success Rate (ASR)** | % of harmful prompts that bypass safety mechanisms |
| **Malicious Rate (MR)** | % of successful jailbreak responses containing actionable malicious code |
| **Final Accuracy (FA)** | Synthesized program agreement with victim on unseen prompts |
| **Peak Balanced Accuracy (PBA)** | Best balanced accuracy on harmful + benign validation set |
| **Number of Interventions (NI)** | Interventions required to reach PBA |

## Installation

### Prerequisites

- Python 3.10+
- [UV](https://docs.astral.sh/uv/) (Python package manager)
- Redis (for session memory)
- Ollama (for local victims) or OpenRouter / OpenAI API key (for cloud victims)
- Neo4j (optional, for scientific memory and causal graph)

### Step 1: System Dependencies

```bash
# macOS
brew install redis
brew install neo4j    # optional
brew install ollama   # for local victims

# Ubuntu/Debian
sudo apt install redis-server
# Download Neo4j from https://neo4j.com/download/
# Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Python Environment

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repo-url> && cd HARMONY_X
uv sync
```

Or use the automated setup script:
```bash
bash experiments/setup.sh
```

### Step 3: Environment Variables

Copy `.env.template` to `.env` and fill in your API keys:

```bash
cp .env.template .env
```

Required variables in `.env`:

| Variable | Required for | Description |
|----------|-------------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter backend | API key from https://openrouter.ai/keys |
| `OPENAI_API_KEY` | OpenAI backend / agentic LLM | API key from https://platform.openai.com/api-keys |
| `REPLICATE_API_TOKEN` | Replicate backend | API token from https://replicate.com/account/api-tokens |

Optional variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HX_REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `HX_NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `HX_NEO4J_USER` | `neo4j` | Neo4j username |
| `HX_NEO4J_PASSWORD` | `password` | Neo4j password |
| `HARMFUL_CSV` | `prompt.csv` | Path to RMCBench harmful prompts CSV |

### Step 4: Pull Victim Models (Ollama only)

```bash
ollama pull codellama:7b
ollama pull llama3.1:8b
ollama pull phi4
ollama pull qwen
ollama pull deepseek-r1:8b
```

### Step 5: Start Services

```bash
brew services start redis        # Redis (session memory)
brew services start neo4j        # Neo4j (optional, scientific memory)
ollama serve                     # Ollama (only if using local victims)
```

## Experiment Configuration

Configuration is managed via YAML files in `configs/` and `experiments/configs/`:

### Core Config Files

| File | Purpose |
|------|---------|
| `configs/experiment_config.yaml` | Default experiment config |
| `configs/config_quick_test.yaml` | Quick debug config (1 seed, 25 iterations, limited transforms) |
| `configs/config_test_fix.yaml` | Fix validation config |
| `experiments/configs/ollama_experiment_config.yaml` | Ollama backend config |
| `experiments/configs/openrouter_experiment_config.yaml` | OpenRouter backend config |
| `experiments/configs/openai_experiment_config.yaml` | OpenAI backend config |

### Key Configuration Parameters

```yaml
# Orchestrator
orchestrator:
  max_iterations: 50              # Max main loop iterations
  max_interventions: 500          # Max total interventions
  synthesis_interval: 3           # Run Researcher every N interventions
  entropy_convergence_threshold: 0.1  # Stop when H < threshold
  accuracy_threshold: 0.85        # Stop when program accuracy >= threshold

# Cognitive Agent
cognitive:
  anomaly_threshold: 0.15         # Minimum anomaly score
  anomaly_selection:
    method: "percentile"          # "percentile" or "threshold"
    percentile: 85                # Top percentile for anomaly selection

# Strategist Agent
strategist:
  max_chain_depth: 4              # Max transform chain depth
  max_candidates_heuristic: 120   # Max heuristic candidates
  num_trials: 4                   # Prediction trials for non-deterministic classifiers

# Researcher / Synthesis
synthesis:
  mode: "evolutionary"
  evolutionary:
    population_size: 150          # GA population
    generations: 50               # GA generations
    mutation_rate: 0.25           # Mutation probability
    crossover_rate: 0.75          # Crossover probability

# Victim (Ollama)
victim:
  ollama_url: "http://localhost:11434"
  model_name: "llama3.1:8b"
  temperature: 0.0
  max_tokens: 150

# Victim (OpenRouter)
victim:
  model_name: "meta-llama/llama-3.2-3b-instruct"
  temperature: 0.0
  max_tokens: 150
```

## Running Experiments

### Quick Test (Debug Mode)

```bash
# Toy victim (deterministic, no API calls)
python experiments/run_experiment.py \
    --config configs/config_quick_test.yaml \
    --backend ollama \
    --model-name "" \
    --num-seeds 3
```

### Full Experiment

```bash
# Via Ollama (local)
python experiments/run_experiment.py \
    --backend ollama \
    --config experiments/configs/ollama_experiment_config.yaml \
    --model-name "codellama:7b" \
    --num-seeds 50

# Via OpenRouter (API)
python experiments/run_experiment.py \
    --backend openrouter \
    --config experiments/configs/openrouter_experiment_config.yaml \
    --model-name "meta-llama/llama-3.1-8b-instruct" \
    --num-seeds 50

# Use all RMCBench prompts
python experiments/run_experiment.py \
    --backend openrouter \
    --full \
    --model-name "deepseek/deepseek-v3.2"
```

### Using the Shell Script

```bash
# Default (Ollama, llama3.1:8b)
bash experiments/run_exp.sh

# With options
bash experiments/run_exp.sh \
    --backend openrouter \
    --model-name "meta-llama/llama-3.1-8b-instruct" \
    --num-seeds 100 \
    --config experiments/configs/openrouter_experiment_config.yaml
```

### Experiment CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/<backend>_experiment_config.yaml` | Path to config YAML |
| `--num-seeds` | 100 | Number of seed prompts for initial variants |
| `--full` | False | Use ALL prompts from CSV (overrides --num-seeds) |
| `--backend` | `ollama` | Victim backend: `ollama`, `openrouter`, `openai`, `replicate` |
| `--agentic-backend` | `openai` | Backend for agent LLMs (cognitive, red-team, judge) |
| `--model-name` | from config | Override victim model name |
| `--num-asr` | 40 | Number of prompts for ASR evaluation (`full` = all) |
| `--num-variants` | 5 | Number of variants per prompt in ASR evaluation |
| `--max-techniques` | 0 | Limit jailbreak techniques (0 = all 21) |
| `--judge-backend` | same as agentic | Backend for judge LLM |
| `--judge-model` | backend default | Model for judge LLM |
| `--prior-campaign` | None | Prior campaign ID for transfer evaluation (RQ3) |
| `--ablation-strategist` | False | Disable Strategist (random probing baseline) |
| `--ablation-cognitive` | False | Disable Cognitive LLM (fallback hypotheses only) |

### System Prompt Conditioning

```bash
# Inject generic task-oriented system prompt into victim
python experiments/run_with_system_prompt.py \
    --model-name "meta-llama/llama-3.1-8b-instruct" \
    --num-seeds 50

# With free-form system prompt
python experiments/run_with_system_prompt.py \
    --model-name "meta-llama/llama-3.1-8b-instruct" \
    --free
```

## Ablation Studies

```bash
# Run all ablations with toy victim
python -m experiments.ablation.run_all

# Run individual ablations
python -m experiments.ablation.no_synthesis
python -m experiments.ablation.random_probing
python -m experiments.ablation.no_scientific_memory
python -m experiments.ablation.no_llm
```

### Ablation Modes

| Mode | Flag | What changes |
|------|------|-------------|
| **No Strategist** | `--ablation-strategist` | Random pair selection, identity-only interventions |
| **No Cognitive LLM** | `--ablation-cognitive` | Keyword-only fallback hypotheses (no LLM) |
| **No Synthesis** | `no_synthesis` wrapper | No evolutionary synthesis — VS only has compiled programs |
| **No Scientific Memory** | `no_scientific_memory` wrapper | Disables Neo4j graph store |
| **Random Probing** | `random_probing` wrapper | Random intervention design (no VS-driven selection) |

## Evaluation

After a campaign completes, run evaluation:

```bash
python experiments/run_evaluation.py \
    --campaign <campaign_id> \
    --program-id <best_program_id> \
    --output-dir evaluation/reports

# Example
python experiments/run_evaluation.py \
    --campaign deepseek_deepseek_v3_2_20260622_230237 \
    --num-test-prompts 50 \
    --accuracy-threshold 0.85
```

### Evaluation CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--campaign` | required | Campaign ID to evaluate |
| `--program-id` | best program | Specific program ID for RQ0 evaluation |
| `--num-test-prompts` | 50 | Number of held-out test prompts |
| `--accuracy-threshold` | 0.85 | Threshold for RQ0/RQ1 |
| `--judge` | `llm` | Judge type: `rule` or `llm` |
| `--llm-model` | `gemma-4-31b-it` | LLM model for LLMJudge |
| `--baseline-campaign` | None | Prior campaign for RQ1 random probing comparison |
| `--transfer-threshold` | 0.9 | Transfer accuracy threshold for RQ3 |
| `--prompt-csv` | `prompt.csv` | Path to RMCBench harmful prompts |

## Output Structure

```
outputs/campaign/<campaign_id>/
├── <campaign_id>_episodic.db       # SQLite — all interaction traces
├── final_program.json               # Best discovered defense program
├── final_theory.json                # Abstracted safety theory
├── hypotheses_history.json          # Generated hypothesis summaries
├── interventions_history.json       # All probing attempts (187+ episodes)
├── version_space.json               # VS state: candidates, entropy, posterior history
├── sde_evidence.json                # Semantic boundary evidence
├── technique_stats.json             # UCB technique selection statistics
├── evaluation/
│   └── evaluation_report.json       # ASR, MR, FA, PBA, NI results
```

## Dataset

The project uses **RMCBench** (Benchmarking Large Language Models' Resistance to Malicious Code) — the first benchmark specifically designed to measure LLM resistance to malicious code generation, published at IEEE/ACM ASE 2024. It contains **473 seed prompts** across two evaluation scenarios:

| Scenario | Description | Levels |
|----------|-------------|--------|
| **Text-to-Code** | Natural language descriptions requesting malicious code generation | Level 1: explicit keywords; Level 2: implicit (metaphorical) |
| **Code-to-Code** | Code translation or completion of malicious code snippets | Code Translation, Code Completion |

**Scope:**
- **11 malware categories**: Viruses, Worms, Trojan horses, Spyware, Adware, Ransomware, Rootkits, Phishing, Vulnerability Exploitation, Network attacks, Others
- **9 programming languages**: C, C++, C#, Go, Java, PHP, Python, HTML/JavaScript, Bash
- **28.71%** average refusal rate across 11 popular LLMs, highlighting the challenge of safety alignment

The full dataset (473 prompts × variants) has **38,168 rows** in `prompt.csv` (columns: `pid,category,task,level,description,level,prompt,malicious functionality,...,language,...,code to be completed,...`).

Benign prompts for balanced accuracy evaluation: `data/benign_prompts.csv`.

## Supported Victims

- **Ollama** (local): codellama:7b/13b/34b, llama3.1:8b, phi4, qwen, deepseek-r1:8b, meetai-small
- **OpenRouter** (API): 50+ models via unified API (GPT-4o, Claude, Gemini, DeepSeek, LLaMA, etc.)
- **OpenAI** (API): GPT-4o-mini, GPT-4o
- **Replicate** (API): Various open models
- **Toy victims** (testing/diagnostic): KeywordFilter, LengthFilter, RegexVictim, ThresholdVictim

## Research Contributions

1. **Novel approach**: Response-aware multi-agent framework combining coordinated agent reasoning, Bayesian hypothesis maintenance, and hypothesis-guided intervention generation
2. **Empirical evaluation**: Comprehensive evaluation against proprietary and open-source LLMs (GPT-4o-mini, Gemini-2.5-Flash, Phi-4, LLaMA-3.1-8B, DeepSeek-V3.2, Qwen2.5-72B, CodeLlama, MiMo-2.5-Pro) using the RMCBench benchmark
3. **Open science**: Full replication package with code and data

## BibTeX

```bibtex
@article{agenttroop2026,
  title={The Wolf Pack: Response-Aware Multi-Agent Attacks for Eliciting Malicious Code},
  author={Anonymous Authors},
  year={2026},
  journal={arXiv preprint}
}
```
