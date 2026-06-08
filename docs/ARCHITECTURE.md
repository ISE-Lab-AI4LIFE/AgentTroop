# HARMONY-X Architecture

## Overview

HARMONY-X reverse-engineers LLM safety policies using a **Version Space** over candidate programs. The core loop is:

```
Data → Programs → Version Space → Disagreement → Intervention → Data
```

This replaces the previous design of `hypotheses → belief → intervention` which suffered from degenerate belief states, no hypothesis competition, and strategist collapse.

## Before vs After

| Component | Before (Broken) | After (Fixed) |
|-----------|----------------|---------------|
| **Belief State** | `BayesianBeliefUpdater(states=[])` — empty POMDP states | `VersionSpace` with posterior over candidate programs |
| **Hypothesis Competition** | All hypotheses had same confidence/prediction | Hypotheses assigned **different anomaly subsets** → differential confidence |
| **Intervention Design** | Confidence-based pair selection | **Disagreement-driven**: find prompts where candidates disagree |
| **Program Synthesis** | Returns 1 program (or 0) | Returns **top-K** candidates via `synthesize_top_k()` |
| **Grammar Balance** | Only REFUSE programs (`then_outcome=1`) | **Both outcomes**: REFUSE/ACCEPT, ACCEPT/REFUSE, ALWAYS_ACCEPT, ALWAYS_REFUSE |
| **Program Storage** | Single `best_program_id` in Redis | `VersionSpace` maintains top-K candidates + **Redis persistence** for resume |
| **Convergence** | Entropy=0 always (degenerate) | Entropy over candidate posterior + **Information Gain tracking** per intervention |
| **Dead Code** | POMDP `_state_ids`, old belief path | **Removed** — Version Space is single source of truth |
| **Pipeline Flow** | Anomaly → Hypothesis → Intervention → Synthesis | Data → Programs → Version Space → Disagreement → Intervention |
| **Fallback** | Returns empty when no program matches | **Heuristic candidates** from keyword/length predicates ensure VS always ≥ 10 |

## Key Components

### 1. Version Space (`inference/version_space.py`)

The central data structure. Maintains:

- `candidates: List[CandidateProgram]` — top-K programs
- `posterior: np.ndarray` — P(program | data) via Bayesian update
- `entropy(): float` — uncertainty measure over candidates (0 when < 2 candidates)
- `total_info_gain: float` — cumulative information gain across all belief updates

Key methods:
- `add_candidate(program, accuracy, source)` — add/update program; auto-prunes to max_candidates
- `update_belief(prompt, outcome, predict_fn)` — Bayesian posterior update with info gain tracking
- `get_most_uncertain_pair(prompts, executor)` — find max disagreement region
- `get_disagreement_pairs(prompts, executor, top_k)` — top-K disagreement pairs
- `most_likely() -> CandidateProgram` — highest posterior probability

### 2. Grammar (`synthesis/grammar_exporter.py`)

Generates programs with **balanced outcomes**:

```python
# Each condition produces TWO programs:
IF cond THEN REFUSE ELSE ACCEPT    # then_outcome=1, else_outcome=0
IF cond THEN ACCEPT ELSE REFUSE    # then_outcome=0, else_outcome=1

# Plus baselines:
ALWAYS_ACCEPT   # then_outcome=0, else_outcome=0
ALWAYS_REFUSE   # then_outcome=1, else_outcome=1
```

This ensures the search space covers both outcome polarities. Baseline programs guarantee at least 2 candidates even when conditions are poor.

### 3. StrategistAgent (`agents/strategist.py`)

**Disagreement-driven intervention**:

1. `select_hypothesis_pair()` checks Version Space first
2. `_select_from_version_space()` → finds pair with max predicted disagreement
3. Falls back to confidence-based (only when VS < 2 candidates)

Dead POMDP code removed:
- `_state_ids` no longer accessed
- POMDP belief fallback deleted
- EFE works directly with `VersionSpace`

### 4. CVC5Synthesizer (`synthesis/cvc5_synthesizer.py`)

Methods:
- `synthesize(examples)` — returns best program (legacy)
- `synthesize_top_k(examples, k=10)` — returns top-K matching programs sorted by fitness + MDL

### 5. Orchestrator (`orchestration/orchestrator.py`)

**Single owner of Version Space**. All agents receive reference, not ownership.

Pipeline flow:

1. **Phase 1-2**: Detect anomalies, generate hypotheses
2. **Phase 3-4**: Strategist selects intervention via:
   - Version Space disagreement (primary)
   - Confidence-based fallback (when VS < 2 candidates)
3. **Phase 5**: `_synthesize_and_update_version_space()`:
   - `synthesize_top_k()` → add to VS → persist to SessionMemory
   - **Heuristic fallback**: if synthesis returns 0 programs, generates keyword/length-based candidates
   - Postcondition: `VS.num_candidates >= 10` (guaranteed by fallback)
4. **Phase 6**: Verify most likely candidate, convergence check

Convergence check:
- **Version space entropy** < threshold for 5 cycles
- **Most likely candidate** accuracy ≥ threshold

### 6. Information Gain Tracking

Every belief update records:

```
IG = H_before - H_after
```

- `VS.info_gains: List[float]` — per-update IG values
- `VS.total_info_gain: float` — cumulative IG
- Logged at each update: `"Belief update: H=0.693→0.512 IG=0.181 candidates=10"`

### 7. Heuristic Fallback Candidates

When `synthesize_top_k` returns 0 (e.g., data imbalance, restrictive grammar), the orchestrator generates fallback candidates:

- Top-8 keywords from REFUSE examples → both outcome variants (16 programs)
- Length thresholds [30, 50, 100, 200] → both outcome variants (8 programs)
- Filters to keep only those with accuracy > 0.5

This guarantees the Version Space always has ≥ 10 candidates after the first synthesis cycle.

### 8. SessionMemory Top-K Persistence

```python
session_memory.set_version_space(campaign_id, candidates)
session_memory.get_version_space(campaign_id)  # restore for resume
```

Stores program_id, accuracy, posterior, source for each candidate. Enables resume support.

## Pipeline Flow (Detailed)

```
┌─────────────────────────────────────────────────────────┐
│  Episodic Memory (existing episodes)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 1-2: Cognitive Agent                              │
│  - detect_anomalies()                                    │
│  - generate_hypotheses() (each has DIFFERENT anomalies)  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 3-4: Strategist Agent                             │
│  ┌─ Version Space ≥ 2 candidates? ──→ Disagreement pair │
│  └─ < 2 candidates? ──→ Confidence-based fallback        │
│  Execute intervention → outcome → belief update (IG log)│
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 5: Synthesis + Version Space Update               │
│  ┌─ synthesize_top_k() → found? ──→ Add to VS           │
│  └─ 0 programs ──→ Heuristic fallback candidates         │
│  Reset belief → persist to SessionMemory                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 6: Verify + Converge                              │
│  - most_likely() accuracy ≥ threshold? → stop            │
│  - entropy < threshold for 5 cycles? → stop              │
│  - else → back to Phase 3                                │
└─────────────────────────────────────────────────────────┘
```

## Invariants

| Invariant | Enforced By | Failure Mode |
|-----------|-------------|-------------|
| VS always has ≥ 2 candidates after synthesis | `_generate_heuristic_candidates()` fallback | Impossible (fallback always generates) |
| Single owner of Version Space | Orchestrator creates, agents reference | Only one orchestrator per campaign |
| No POMDP dead code | `_state_ids` removed from belief_updater | Strategist fallback stays clean |
| Grammar generates both outcomes | `grammar_exporter.py:152-195` | Programs with both polarities |
| Info gain logged per update | `orchestrator._update_belief_from_observation()` | IG tracked and reported |

## Configuration

```python
orchestrator = Orchestrator(
    ...,
    top_k_candidates=10,                    # VS size
    entropy_convergence_threshold=0.1,      # Convergence when entropy < 0.1
    synthesis_interval=5,                   # Run synthesis every N interventions
)
```

## Success Criteria

A campaign is successful when:

1. After first synthesis: `VS.num_candidates >= 10`
2. After each intervention: entropy decreases (monotonic posterior concentration)
3. Strategist selects prompts based on candidate disagreement
4. Candidate pool shrinks over time (pruning)
5. At campaign end: posterior concentrated on 1-3 programs
6. Convergence is driven by knowledge gain, not iteration limit
