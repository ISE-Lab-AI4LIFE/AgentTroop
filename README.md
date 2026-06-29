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
| **Cognitive Agent** | Analyzes interaction traces, identifies behavioral inconsistencies (semantically similar prompts with different refusal rationales), generates defense hypotheses |
| **Strategist Agent** | Identifies competing defense programs via posterior belief, performs symbolic analysis to find distinguishing conditions, formulates intervention objectives |
| **Red Team Agent** | Generates executable probing prompts using jailbreak transformations, semantic reframing, role-playing, and prompt engineering |
| **Researcher Agent** | Periodically consolidates interaction evidence, synthesizes refined defense programs, injects new candidates into Version Space |

## Core Loop

```
Interaction → Behavioral Analysis → Hypothesis Generation → 
Version Space Update → Competing Program Selection → 
Intervention Design (EFE) → Probing → Belief Update → Convergence?
```

## Key Components

| Module | Purpose |
|--------|---------|
| `core/primitive.py` | 92 primitives: 27 Predicates, 38 Transforms, 27 Classifiers for building defense programs |
| `core/program.py` | AST-based defense program representation (IfThenElse, And, Or, Not, Threshold) |
| `core/executor.py` | ProgramExecutor — evaluates defense programs against prompts |
| `inference/version_space.py` | Bayesian Version Space: top-K candidate programs, posterior beliefs, entropy, information gain |
| `inference/efe.py` | Expected Free Energy computation for intervention selection |
| `agents/cognitive.py` | Behavioral inconsistency detection and hypothesis generation |
| `agents/strategist.py` | Disagreement-driven intervention design and execution |
| `agents/researcher.py` | Defense program synthesis and verification |
| `agents/red_team.py` | Jailbreak-aware probing prompt generation |
| `orchestration/orchestrator.py` | 6-phase loop coordination, belief management, convergence detection |
| `synthesis/` | Multi-strategy synthesis: evolutionary (genetic programming), neural-guided, fitness-guided, CVC5 SMT |
| `sde/` | Semantic Discovery Engine — continuous-score boundary estimation for classifier primitives |
| `evaluation/` | Evaluators for RQ1 (ASR, MR), RQ2 (FA, PBA, NI), RQ3 (ablation) |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Attack Success Rate (ASR)** | % of harmful prompts that bypass safey mechanisms |
| **Malicious Rate (MR)** | % of successful jailbreak responses containing actionable malicious code |
| **Final Accuracy (FA)** | Synthesized program agreement with victim on unseen prompts |
| **Peak Balanced Accuracy (PBA)** | Best balanced accuracy on harmful + benign validation set |
| **Number of Interventions (NI)** | Interventions required to reach PBA |

## Quick Start

```bash
# Install dependencies
uv sync

# Start services
brew services start redis
brew services start neo4j

# Set up environment
cp .env.template .env   # edit OPENROUTER_API_KEY

# Run against a victim (Ollama)
make -C experiments run BACKEND=ollama

# Run against a victim (OpenRouter)
make -C experiments run BACKEND=openrouter MODEL_NAME="meta-llama/llama-3.2-3b-instruct"
```

## Output Structure

```
outputs/campaign/<campaign_id>/
├── <campaign_id>_episodic.db       # All interaction traces (prompt, response, outcome)
├── final_program.json               # Best discovered defense program
├── final_theory.json                # Abstracted safety theory
├── interventions_history.json       # All probing attempts
├── sde_evidence.json                # Semantic boundary evidence
└── evaluation/
    └── evaluation_report.json       # ASR, MR, FA, PBA, NI results
```

## Research Contributions

1. **Novel approach**: Response-aware multi-agent framework combining coordinated agent reasoning, Bayesian hypothesis maintenance, and hypothesis-guided intervention generation
2. **Empirical evaluation**: Comprehensive evaluation against proprietary and open-source LLMs (GPT-4o-mini, Gemini-2.5-Flash, Phi-4, LLaMA-3.1-8B, DeepSeek-V3.2, Qwen2.5-72B, CodeLlama, MiMo-2.5-Pro) using the RMCBench benchmark
3. **Open science**: Full replication package with code and data

## Supported Victims

- **Ollama** (local): llama3.1:8b, codellama:7b/13b/34b, phi4, qwen, deepseek, meetai-small
- **OpenRouter** (API): 50+ models via unified API
- **Toy victims** (testing): KeywordFilter, LengthFilter, RegexVictim, ThresholdVictim, etc.

## BibTeX

```bibtex
@article{agenttroop2026,
  title={The Wolf Pack: Response-Aware Multi-Agent Attacks for Eliciting Malicious Code},
  author={Anonymous Authors},
  year={2026},
  journal={arXiv preprint}
}
```
