# HARMONY-X Comprehensive Analysis & Experimental Readiness Report

## 1. System Architecture Overview

```
                    ┌──────────────────────────────────────────────────┐
                    │              Orchestrator V2                      │
                    │  6-phase POMDP loop (run / async_run)             │
                    └─────┬───────┬───────┬───────┬───────┬────────────┘
                          │       │       │       │       │
              ┌───────────┘       │       │       │       └───────────┐
              ▼                   ▼       ▼       ▼                   ▼
    CognitiveAgent        Strategist    Researcher  RedTeamAgent     SDE
    (Gemma LLM)           Agent         (CVC5)      (LLM refiner)  Engine
    │                     │             │           │               │
    │  detect_anomalies() │             │           │               │
    │  generate_hypotheses│             │           │               │
    └─────────┬───────────┘             │           │               │
              │ select_hypothesis_pair()│           │               │
              │ design_intervention()   │           │               │
              │ execute_intervention()  │           │               │
              └─────────┬───────────────┘           │               │
                        │                           │               │
                        ▼                           ▼               ▼
                   ┌──────────────────────────────────────────────────┐
                   │           OllamaVictim (llama3.1:8b)             │
                   │           http://localhost:11434/api/generate    │
                   └──────────────────────────────────────────────────┘
```

### 1.1 Module Stack (10 layers)

| Layer | Directory | Lines | Primary Role |
|-------|-----------|-------|--------------|
| **Core** | `core/` | ~2,500 | 77 primitives, Program AST, Intervention, Executor, Grammar |
| **Agents** | `agents/` | ~6,200 | Cognitive (Gemma LLM), Strategist, Researcher (CVC5), RedTeam (LLM refiner) |
| **Orchestration** | `orchestration/` | ~2,800 | 6-phase POMDP loop, SurrogatePolicyModel, CounterfactualLearner |
| **Inference** | `inference/` | ~2,500 | VersionSpace (1022 lines), BayesianBeliefUpdater, POMDP, EFE |
| **Knowledge** | `knowledge/` | ~3,500 | L1 Episodic (SQLite), L2 Session (Redis), L3 Strategy, L4 Defense (Neo4j), L5 Semantic (FAISS+SQLite), L6 Scientific (Neo4j), KnowledgeManager |
| **Synthesis** | `synthesis/` | ~1,500 | CVC5Synthesizer, GrammarExporter, Verifier, Preprocessor, HypothesisOptimizer |
| **Evaluation** | `evaluation/` | ~2,000 | RQ0–RQ3, ASR, Adversarial ASR, LLM Judge, RuleBasedJudge |
| **Adaptation** | `adapters/` + `llama3_1_8b/` | ~1,200 | BaseVictim, OllamaVictim (188 lines), VictimFactory, toy victims |
| **SDE** | `sde/` | ~2,000 | SemanticDiscoveryEngine, BoundaryEstimator, ConceptDiscovery, EmbeddingStore |
| **LLM** | `llm/` | ~500 | OpenRouterClient, GemmaVictim, get_default_client |

### 1.2 Data Flow (Probe → Attack)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1-2: RECONNAISSANCE (Cognitive Agent)                             │
│                                                                         │
│  1. seed_episodic_memory():                                             │
│     - Load harmful + benign prompts from CSV (default: 120/80 split)    │
│     - MultiTierSeedStrategy: 8 semantic + 4 structural + 3 encoding     │
│       variants per prompt = 15 variants/base                             │
│     - Call victim.respond() on each variant → Outcome{0,1}              │
│     - Store episodes in EpisodicMemory (SQLite)                         │
│                                                                         │
│  2. detect_anomalies():                                                 │
│     - Compare prompt pairs with same base but different transforms      │
│     - Large outcome difference → signals defense mechanism              │
│                                                                         │
│  3. generate_hypotheses():                                              │
│     - Uses Gemma LLM with structured template                           │
│     - Produces structural hypotheses about the safety mechanism         │
│     - warm_start_from_scientific_memory() for prior theories            │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3-4: INTERVENTION (Strategist Agent + RedTeamAgent)               │
│                                                                         │
│  Loop (max_iterations=50, max_interventions=1000):                      │
│                                                                         │
│  1. select_hypothesis_pair()                                            │
│     - Picks pair with maximum VersionSpace disagreement                 │
│     - Falls back to belief entropy                                     │
│                                                                         │
│  2. design_intervention()                                               │
│     - Creates Intervention(base_prompt, transforms[], metadata)         │
│     - Transforms discriminate between hypotheses                        │
│                                                                         │
│  3. [NEW] RedTeam LLM Refinement                                        │
│     - if red_team is not None AND phase > 2:                            │
│       → maybe_refine_intervention(intervention)                         │
│     - LLM receives: original prompt + template-expanded variant         │
│       + technique metadata (jailbreak technique name, category,         │
│       complexity)                                                       │
│     - Returns improved prompt (harder to detect)                        │
│     - Graceful fallback on LLM failure (403/429)                        │
│                                                                         │
│  4. execute_intervention(victim)                                        │
│     - Sends prompt to Ollama /api/generate                              │
│     - Returns 0 (ACCEPT) or 1 (REFUSE) via 62 refusal patterns         │
│                                                                         │
│  5. _update_belief_from_observation()                                   │
│     - Bayesian posterior update on VersionSpace                         │
│                                                                         │
│  6. store_intervention() → EpisodicMemory                               │
│                                                                         │
│  7. [every synthesis_interval=3]                                        │
│     _synthesize_and_update_version_space()                              │
│                                                                         │
│  8. evaluate_on_holdout()                                               │
│     - Reweight posterior by holdout accuracy                            │
│                                                                         │
│  9. Convergence check:                                                  │
│     - Entropy < 0.1 for 5 cycles AND holdout ≥ 0.8 with gap ≤ 0.1      │
│     - OR holdout accuracy ≥ 0.9 with gap ≤ 0.1                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 5-6: SYNTHESIS (Researcher Agent)                                 │
│                                                                         │
│  1. CVC5Synthesizer.synthesize(episodes):                               │
│     - Enumerates program space via CVC5 SMT solver                     │
│     - Produces top-K candidates for VersionSpace                        │
│     - Heuristic fallback if CVC5 returns 0 programs                     │
│     - Type diversity guarantee (≥3 per core type)                      │
│                                                                         │
│  2. verify_and_store():                                                 │
│     - Verifies program against victim                                   │
│     - Stores in DefenseProgramStore (Neo4j)                             │
│     - Abstracts theory → ScientificMemory                               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ POST-PIPELINE: EVALUATION                                               │
│                                                                         │
│  - RQ0: Program accuracy on held-out test set                           │
│  - RQ1: Intervention efficiency (prediction accuracy vs. budget)        │
│  - ASR: Attack Success Rate (raw prompts bypassing safety)              │
│  - Adversarial ASR: Program-guided prompt crafting success rate         │
│  - RQ2: Explanation score (human annotation, Fleiss' Kappa)            │
│  - RQ3: Transfer speed (speedup vs. prior campaign)                    │
│  - SDE evidence export                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Experiment Pipeline Configuration

### 2.1 prompt.csv Analysis

| Metric | Value |
|--------|-------|
| Total CSV rows | 473 |
| Unique prompts | 470 |
| Duplicates | 3 prompts appear twice each |
| Unique tasks | 3 (text-to-code generation, code completion, code translation) |
| Schema | pid, category, task, level description, level, prompt, malicious functionality, malicious categories, ... |
| Default load (--num-seeds 50) | 30 harmful + 20 benign → ~450 seed episodes |
| Full load (--full) | 473 harmful → ~7,095 seed episodes |

### 2.2 Seed Strategy

| Tier | Count/Prompt | Total (473 prompts) | Transforms |
|------|-------------|---------------------|------------|
| T1: Semantic | 8 | 3,784 | add_role_play (6 roles), add_prefix (5 prefixes), 3 grammar shifts, 3 perspective shifts |
| T2: Structural | 4 | 1,892 | wrap_code_block, format_as_json, add_markdown (3), add_prefix (3), add_ignore_filter_token (2) |
| T3: Encoding | 3 | 1,419 | leet_speak, unicode_obfuscate, insert_typos, reverse_text, pig_latin |
| Baseline (no transform) | 1 | 473 | Plain prompt |
| **Total** | **16** | **~7,568** | |

T3 cap: 30% max of total variants enforced at seeding time.

### 2.3 Experiment Config (experiment_config.yaml)

| Section | Key Parameters |
|---------|---------------|
| **Async** | `use_async: true` |
| **Orchestrator** | max_iterations=50, max_interventions=1000, accuracy_threshold=0.90, synthesis_interval=3 |
| **Cognitive** | anomaly_threshold=0.1, llm_model="gemma" |
| **Strategist** | use_llm=true, max_chain_depth=3, max_candidates_llm=20 |
| **Researcher** | max_depth=4, beam_width=200, use_cvc5=true, cvc5_path=/opt/homebrew/bin/cvc5 |
| **Victim** | model="llama3.1:8b", temperature=0.0, max_tokens=100 |
| **Red Team** | population_size=10, refinement_rounds=3, mutation_rate=0.3, enable_self_eval=false |
| **Transforms** | disabled: [] (all 77 primitives enabled) |

---

## 3. Gap Fixes Applied

### Gap 1: RedTeamAgent NOT wired into pipeline ❌ → ✅

**Root cause**: `run_experiment.py` (lines 424–444) created the Orchestrator without passing `red_team_agent=red_team`. Orchestrator's `self.red_team = None` → refinement code in `run()` (line 398) and `async_run()` (line 703) never executed.

**Fix applied** (`llama3_1_8b/run_experiment.py`):
- Added `from agents.red_team import RedTeamAgent` (line 332)
- Creates `RedTeamAgent` with config-driven params (lines 359–377):
  ```python
  red_team = RedTeamAgent(
      victim=victim,
      llm_client=llm,
      episodic_memory=episodic,
      scientific_memory=scientific,
      population_size=rt_cfg.get("population_size", 10),
      refinement_rounds=rt_cfg.get("refinement_rounds", 3),
      ...
  )
  ```
- Passes `red_team_agent=red_team` to Orchestrator (line 451)

**Result**: Every non-exploratory intervention in Phase 3+ (i.e. after reconnaissance) passes through `RedTeamAgent.maybe_refine_intervention()` → LLM refines the prompt with jailbreak technique metadata before sending to the victim.

### Gap 2: Seed strategy references deleted transforms ❌ → ✅

**Root cause**: `TIER3_TRANSFORMS` (seed_strategy.py:115–123) contained `rot13` and `base64` which were removed from `core/primitive.py` during encoding primitive cleanup. These silently failed (exception caught → returns original prompt unchanged), reducing effective T3 seeding.

**Fix applied** (`llama3_1_8b/seed_strategy.py`):
- Removed `{"name": "rot13", ...}` and `{"name": "base64", ...}` entries
- Remaining T3 transforms (5): `leet_speak`, `unicode_obfuscate`, `insert_typos`, `reverse_text`, `pig_latin`

**Result**: All T3 transforms now successfully apply to seed prompts. No silent failures.

### Gap 3: run_exp.sh default config path wrong ❌ → ✅

**Root cause**: `run_exp.sh` defaulted to `CONFIG_FILE="$SCRIPT_DIR/config.yaml"` which didn't exist. Since the script always passes `--config "$CONFIG_FILE"`, it would override `run_experiment.py`'s correct default (`configs/experiment_config.yaml`) and trigger `FileNotFoundError`.

**Fix applied** (`llama3_1_8b/run_exp.sh`):
- Changed default to `CONFIG_FILE="$PROJECT_DIR/configs/experiment_config.yaml"`

**Result**: Shell script and Python default now point to the same valid config file.

### Gap 4: No red_team section in YAML configs ❌ → ✅

**Root cause**: The YAML configs had no `red_team:` section, meaning `run_experiment.py` would fall back to hardcoded defaults when creating the RedTeamAgent. The config system was incomplete.

**Fix applied** (all 4 config files):
- Added `red_team:` section with `population_size`, `refinement_rounds`, `mutation_rate`, `crossover_rate`, `max_chain_depth`, `enable_self_eval`
- Cleaned up references to deleted encoding transforms in `disabled:` lists

**Result**: Full parameterization of the Red Team Agent through YAML configs.

---

## 4. Experimental Readiness Assessment

### 4.1 Prerequisites

| Service | Required | Default Endpoint | Status |
|---------|----------|-----------------|--------|
| Ollama | Yes | http://localhost:11434 | Auto-started by run_exp.sh |
| Redis | Yes | redis://localhost:6379 | Auto-started by run_exp.sh |
| Neo4j | Yes | bolt://localhost:7687 | Auto-started by run_exp.sh |
| CVC5 | Yes | /opt/homebrew/bin/cvc5 | Checked at synthesis time |
| Python 3.10+ | Yes | — | Pre-flight check |
| .env file | Yes | API keys | Pre-flight check |

### 4.2 Execution Options

| Command | Seeds | Episodes | Use Case |
|---------|-------|----------|----------|
| `bash run_exp.sh --num-seeds 30` | 30 | ~450 | Quick smoke test |
| `bash run_exp.sh --num-seeds 50` | 50 | ~750 | Standard development |
| `bash run_exp.sh --num-seeds 200 --full` | 200 (harmful) + 80 (benign) | ~4,200 | Full experiment |
| `bash run_exp.sh --full` | 473 harmful + 80 benign | ~8,300 | Full RMC-Bench |
| `bash run_exp.sh --num-seeds 5 --config ../configs/config_quick_test.yaml` | 5 | ~75 | Validation run |

### 4.3 Pipeline Components Active

| Component | Status | Lines |
|-----------|--------|-------|
| Orchestrator 6-phase loop | ✅ Active | 2,292 |
| CognitiveAgent (Gemma LLM) | ✅ Active | 1,574 |
| StrategistAgent (LLM + heuristic) | ✅ Active | 2,352 |
| ResearcherAgent (CVC5 synthesis) | ✅ Active | 937 |
| **RedTeamAgent (LLM refiner)** | **✅ NEWLY WIRED** | 1,335 |
| SDE Engine (semantic discovery) | ✅ Patched in | 2,000+ |
| SurrogatePolicyModel | ✅ Active | — |
| CounterfactualLearner | ✅ Active | — |
| VersionSpace (1022 lines) | ✅ Active | 1,022 |
| BayesianBeliefUpdater | ✅ Active | — |
| CVC5Synthesizer | ✅ Active | — |
| Evaluation (RQ0–RQ3, ASR) | ✅ Active | 2,000+ |
| KnowledgeManager + 6 stores | ✅ Active | 3,500+ |

### 4.4 Red Team LLM Refinement Integration Points

| Location | File | Line | Phase Guard | Effect |
|----------|------|------|-------------|--------|
| Orchestrator.run() | orchestrator.py | 398–399 | `self.phase.value > 2` | Sync sync execution |
| Orchestrator.async_run() | orchestrator.py | 703–704 | `self.phase.value > 2` | Async execution |

The refinement check guards against Phase 1–2 reconnaissance (hypothesis generation). All Phase 3+ interventions (actual attack prompts targeting the victim) are LLM-refined.

### 4.5 Known Limitations

1. **Ollama latency**: Each victim call takes 200–500ms. With 1,000 interventions + ~7,500 seed episodes, a full run can take 30–90 minutes.
2. **CVC5 availability**: Falls back to heuristic synthesis if CVC5 binary is not at configured path.
3. **OpenRouter API key required**: Red Team LLM refinement uses OpenRouter. Without valid `OPENROUTER_API_KEY` in `.env`, refinement silently falls back to template-only mode.

---

## 5. File Manifest of Modified Files

| File | Changes |
|------|---------|
| `llama3_1_8b/run_experiment.py` | +RedTeamAgent import, +RedTeamAgent creation (lines 359-377), +red_team_agent= param in Orchestrator (line 451) |
| `llama3_1_8b/seed_strategy.py` | Removed rot13, base64 from TIER3_TRANSFORMS |
| `llama3_1_8b/run_exp.sh` | Fixed default config path → `configs/experiment_config.yaml` |
| `configs/experiment_config.yaml` | Added red_team section |
| `configs/config_quick_test.yaml` | Added red_team section, cleaned transform disabled list |
| `configs/config_validate.yaml` | Added red_team section |
| `configs/config_test_fix.yaml` | Added red_team section, cleaned transform disabled list |
