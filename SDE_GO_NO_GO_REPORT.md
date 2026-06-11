# SDE Integration Go/No-Go Report

**Date:** 2026-06-11
**Prepared for:** HARMONY-X structural pipeline integration decision

---

## Executive Summary

**Recommendation: NO-GO** for production integration in current form.
**Recommended path:** Architectural redirection — SDE concepts should become first-class predicates for CVC5 + Version Space, not a parallel scoring/reranking system.

---

## Evidence

### Phase C: Ablation Study (15 runs across 3 configs)

| Metric | A (no SDE) | B (evidence) | C (seeding) |
|---|---|---|---|
| test_accuracy_mean | 0.80 | **1.00** | 0.50 |
| test_accuracy_stdev | 0.27 | 0.00 | 0.00 |
| exact_match_rate | 0.60 | **1.00** | 0.00 |
| interventions_mean | 17.8 | 16.4 | 16.6 |
| time_mean | 22.6s | 43.4s | 29.4s |
| rerank_rate_mean | 0.0% | 0.0% | 0.0% |
| sel_change_rate_mean | 0.0% | 0.0% | 0.0% |
| n_seeded_mean | 0 | 0 | 5 |

### Phase A: Integration Audit Findings

| Finding | Impact |
|---|---|
| `_rescore_with_semantic` adds FLAT bonus to all candidates — cannot change relative ranking | Zero decision influence |
| `_seed_semantic_hypotheses` is dead code — never called from any pipeline path | Zero decision influence |
| `MultiDimensionalBoundaryEstimator` is passive (data collected, never consumed) | Zero decision influence |
| `PromptEmbeddingStore` is passive (data sink only) | Zero decision influence |
| `SemanticConceptDiscovery` is passive (logging only) | Zero decision influence |

### Phase E: Semantic Benchmark Validation

| Victim | Accuracy | Converged |
|---|---|---|
| InstructionalOnly | 0.8800 | Yes |
| InstructionalAndHarmful | 0.7143 | Yes |
| JailbreakOnly | 0.9524 | Yes |
| HarmfulOrJailbreak | 0.8065 | Yes |
| MixedSymbolicSemantic | 0.8065 | Yes |

Semantic victims converge (71-95% accuracy) but have no influence on structural decisions.

### Phase D: Structural Safety Guarantee

- `semantic_enabled=False` by default when no SDE engine is connected
- SYMBOLIC mode enforces zero semantic influence
- 10 regression tests pass for disabled-SDE behavior
- Structural pipeline unchanged: 1.000 accuracy, 15 interventions, CVC5 exact match

---

## Why NO-GO

1. **Zero Decision Influence (0.0% rerank/selection rate across 15 runs)** — The flat bonus design in `_rescore_with_semantic` cannot change intervention selection. This is a mathematical certainty, not a tuning issue.

2. **Config C (seeding) degrades performance** — From 0.80 → 0.50 accuracy (-37.5%). Seeding semantic hypotheses into Version Space introduces hypotheses that don't fit the structural problem, wasting iterations.

3. **Config B improvement may be noise** — At n=5, the +0.20 improvement over Config A is not statistically significant (A stdev=0.27). Config B also takes 2× longer (43s vs 23s).

4. **All 5 SDE components are passive or dead code** — The system appears active (instantiated, methods called) but cannot influence any decision. This represents ~500 lines of code with zero behavioral impact.

5. **Architectural mismatch** — SDE's flat bonus + passive estimator pattern cannot integrate meaningfully with Version Space's delta-EFE ranking and CVC5 synthesis.

---

## Recommended Path Forward

1. **Concepts as first-class predicates** — Instead of flat bonuses, translate `SemanticConceptDiscovery` concepts into CVC5 predicates consumed by the synthesizer. This would let semantic knowledge directly influence which programs are synthesized.

2. **Per-proposition rescoring** — Replace `_rescore_with_semantic` flat bonus with per-candidate score adjustment based on how each candidate prompt relates to learned semantic boundaries.

3. **Active estimator wiring** — Connect `MultiDimensionalBoundaryEstimator` output to intervention candidate ranking, e.g., as a diversity penalty or boundary-proximity bonus.

4. **Pipeline seeding** — Call `_seed_semantic_hypotheses` from `Orchestrator.run()` after each synthesis cycle, passing concepts to Version Space for hypothesis initialization.

5. **Structural regression gate** — Maintain `semantic_enabled=False` default and all 10 safety regression tests. SDE must never degrade structural performance.
