# Semantic Benchmark Validation Report

**Date:** 2026-06-11

---

## Results

| Victim | Accuracy | Converged | Rounds | Concepts Found |
|---|---|---|---|---|
| InstructionalOnly | 0.8800 | Yes | 30 | 0 |
| InstructionalAndHarmful | 0.7143 | Yes | 30 | 0 |
| JailbreakOnly | 0.9524 | Yes | 30 | 0 |
| HarmfulOrJailbreak | 0.8065 | Yes | 30 | 0 |
| MixedSymbolicSemantic | 0.8065 | Yes | 30 | 0 |

**All 5 victims converged.** No concepts were auto-discovered (empty `concepts_found` lists), consistent with the observation that SemanticConceptDiscovery is a passive reporter.

### Performance tiers

- **≥85%** (Phase 2 gate met): JailbreakOnly (0.95), InstructionalOnly (0.88)
- **80-85%**: HarmfulOrJailbreak (0.81), MixedSymbolicSemantic (0.81)
- **<80%**: InstructionalAndHarmful (0.71) — hardest victim (requires both instruction AND harmfulness)

### Notes

- JailbreakOnly performs best because jailbreak language has the clearest embedding signature
- InstructionalAndHarmful has the most restrictive decision boundary (requires both axes)
- These scores represent SDE self-assessment (does the victim agree with SDE's own classifier), not structural pipeline performance
- See `experiments/semantic_benchmark_results.json` for raw data
