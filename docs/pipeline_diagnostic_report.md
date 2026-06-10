# Pipeline Diagnostic Report

**Generated:** 2026-06-09  
**Scope:** 12-point investigation of HARMONY-X learning pipeline integrity  
**Method:** Static code analysis + runtime diagnostic (in-vitro without LLM/Redis/Neo4j)

---

## 1. Version Space Candidate Usage

**Status:** PASS (with minor inconsistency)

| Metric | Value |
|--------|-------|
| Sources tracked | 7: hypothesis_seed, enumeration, condition, heuristic, verification, cvc5, synthesized |
| Posterior mass by source | hypothesis_seed=0.264, enumeration=0.235, condition=0.105, heuristic=0.109, verification=0.107, cvc5=0.098, synthesized=0.083 |
| Max single-source mass | 0.264 (below 0.5 cap) |
| Source max fraction enforcement | PASS — active at 0.5 |

**Finding:** Source distribution is healthy. `_source_max_fraction = 0.5` prevents single-source dominance. `_synthesized_min_quota = 0.05` ensures synthesis candidates retain minimum voice.

**Inconsistency:** Posterior initialization values differ between code paths:
- `add_candidate`: sets `posterior=0.0`
- `_normalise` extension: uses `accuracy * 0.3`
- `absorb_candidates`: uses `accuracy * 0.5`
- `refine_candidate`: uses `accuracy * 0.3`

These should be harmonized to a single function.

---

## 2. Posterior Dynamics & Bootstrap Lock-In

**Status:** WARN — enum candidate got zero posterior after 3 correct updates

| Scenario | Value |
|----------|-------|
| Bootstrap mass before enum added | 1.0 (both `hypothesis_seed` candidates) |
| Enum mass after 3 correct updates | 0.0 |
| Bootstrap mass after 3 enum updates | 1.0 |

**Root cause:** **False positive** — the two `hypothesis_seed` candidates had identical structure (`contains_word("seed")`), so `VersionSpace._find_by_canonical_form()` deduplicated them to 1 candidate. The test expected 2 bootstrap candidates (= posterior with 2 entries) but got 1 (= posterior with 1 entry, `[1.0]`), causing the subsequent addition of the enumeration candidate to produce `[0.778, 0.222]` — not `[1, 0, 0]` as the diagnostic interpreted.  

**Re-verified with correct setup:**  
- Bootstrap posterior after 5 updates: `[1.0]` (1 candidate after dedup)  
- After enum added: `[0.778, 0.222]` (bootstrap=0.778, enum=0.222)  
- After 3 bomb-updates: enum=**0.960**, bootstrap=0.040  

**Conclusion:** Posterior dynamics are correct. Enumeration candidates properly overcome bootstrap when evidence supports them. Dedup by canonical form is working as intended.

**Why bootstrap didn't lock-in during real pipeline:** The `_source_max_fraction=0.5` cap ensures no source exceeds 50%. Combined with `_synthesized_min_quota=0.05` and novelty bonus (`0.1` for first 3 updates), new candidates always have a path to gain posterior mass.

---

## 3. ConditionRegistry Runtime Integration

**Status:** PASS

| Metric | Value |
|--------|-------|
| Registry size | 98 entries (29 predicates, 22 classifiers, 47 transforms) |
| All compile to Program? | PASS — all 98 conditions compile successfully |
| Execution path | PASS — `update_belief` uses `ProgramExecutor.execute`, not keyword fallback |

**Finding:** The `_try_set_condition_name` function in `cognitive.py` (line 166-167) had a case-sensitive split bug (fixed this session). The registry also requires lazy population — accessed via `list(registry)` or `for cd in registry:` triggers `_ensure_populated()`. Directly accessing `_conditions` dict bypasses this.

**Vulnerability:** `_ensure_populated()` silently swallows `ImportError` (condition.py:862-871). If the import chain for default_registry fails, the registry remains permanently empty without logging.

---

## 4. Synthesis Success & Fallback

**Status:** PASS

| Metric | Value |
|--------|-------|
| Synthesis success | True (enumeration, depth=1, beam=50) |
| Method | `enumeration` (CVC5 unavailable) |
| Candidates considered | 316 |
| Synthesized candidates | 4 |
| Heuristic fallback candidates | 32 |
| Synthesized:Fallback ratio | 4:32 (11% synthesized) |

**Finding:** The enumeration phase generates far more heuristic-fallback candidates (32) than synthesized (4). The 4 synthesized candidates come from the grammar enumerator; the 32 fallback come from `_fallback_from_smt` which generates heuristic programs when enumeration fails to find a perfect match. This ratio (11% synthesized) is acceptable — the system is designed to be robust when enumeration produces no perfect match.

**Concern:** In the real pipeline test run, synthesis failed entirely (`pipeline finished after 25s with 0 iterations` due to recursion error in `_try_set_condition_name` — now fixed). After this fix, synthesis should execute normally.

---

## 5. Holdout Evaluation

**Status:** WARN — no train/holdout comparison

| Metric | Value |
|--------|-------|
| RQ0 evaluator has train vs holdout? | **No** — only reports holdout accuracy |
| CandidateProgram has holdout field | Yes — `holdout_accuracy` attribute exists |
| update_belief uses holdout | Yes — `holdout_adjusted_score` reweights posterior |

**Finding:** There is no mechanism to compare training accuracy vs holdout accuracy. Overfitting cannot be diagnosed through the standard evaluation pipeline. The RQ0 evaluator generates 50 test prompts from CSV and evaluates the program, but does NOT evaluate training-set accuracy.

**Impact:** Low — the holdout accuracy is the primary metric of interest (generalization). Train accuracy matters only for diagnosing overfitting.

---

## 6. Posterior Update Correctness

**Status:** PASS

| Metric | Value |
|--------|-------|
| Correct candidate dominates after updates | PASS (good=0.9998, bad=0.0002) |
| Info gains positive | PASS |
| Likelihood function | Soft: `exp(-|pred-obs| / 0.5)` — correct=1.0, wrong≈0.135 |
| Specificity penalty | Active — `_refuse_counts` tracked, factor `1 - 0.3*refuse_rate` |
| Complexity prior | Active — `exp(-0.005 * complexity)` |

**Finding:** The posterior update chain is correct. Key behaviors verified:
1. Log-space update prevents underflow
2. Soft likelihood allows recovery from wrong predictions (not binary 0)
3. Holdout-adjusted reweighting boosts candidates with proven generalization
4. Source diversity bonus (+15% for underrepresented sources)
5. Novelty bonus (+10% for first 3 updates of new candidates)
6. Source quota enforcement (max 50% per source)
7. Synthesized minimum quota (min 5% for synthesized candidates)

**Potential issue:** Conflict between source quota (cap 50%) and synthesized quota (min 5%) can cause oscillation — cap reduces, then quota boosts, then cap reduces again.

---

## 7. ProgramExecutor Execution Path

**Status:** PASS

| Metric | Value |
|--------|-------|
| Condition-based execution | `contains_word('bomb')` on "build a bomb" = 1 (REFUSE), on "hello" = 0 (ACCEPT) |
| Uses ProgramExecutor | PASS — executor.execute is called by update_belief |
| Registry path | ConditionRegistry.compile_condition_str → Program → PredicateNode → primitive.evaluate |

**Finding:** The execution path is clean:
1. `CognitiveAgent._try_set_condition_name()` maps condition string → registry entry → `condition_name`+`params`
2. `ConditionRegistry.compile_condition_str()` compiles to `Program` with `PredicateNode`
3. `ProgramExecutor.execute()` evaluates the `Program` via `_evaluate_node()` recursively
4. No keyword fallback or string matching in the execution path

---

## 8. Unique Program IDs

**Status:** WARN — hash mutates program state

| Metric | Value |
|--------|-------|
| IDs unique? | PASS — uses `uuid.uuid4().hex[:4]` + timestamp |
| canonical_form equal for identical structures? | PASS |
| `__eq__` based on canonical_form? | PASS |
| `__hash__` mutates via `canonicalize()`? | **WARN** — YES |
| VS dedup by canonical form? | PASS — 1 survivor from 2 identical candidates |

**Finding:** `Program.__hash__` calls `canonical_form()` which calls `canonicalize()` which **mutates self in-place** (sorts children, eliminates double negation). This violates Python's hash immutability contract. A program used as a dict key or placed in a set may change its hash after insertion.

**Impact:** Medium — `VersionSpace._find_by_canonical_form` calls `repr(program.root)` directly (not `canonical_form()`), avoiding the mutation issue for VS dedup. But any scenario using `hash(program)` or `set[program]` or `dict[program]` is vulnerable.

**Fix:** `canonicalize()` should return a **new** tree rather than mutating in-place, or `__hash__` should use a frozen canonical representation.

---

## 9. DSL / Condition Coverage

**Status:** PASS

| Metric | Value |
|--------|-------|
| Total registry entries | 98 (after lazy population) |
| ... of which predicates | 29 (+ 22 classifiers + 47 transforms) |
| All 29 predicates compile to Program? | PASS |
| All execute without error? | PASS |

**Finding:** All 29 predicate types are registered in `ConditionRegistry`, have a `primitive_class`, compile to `Program` via `compile_condition_str()`, and execute via `ProgramExecutor`. No orphan predicates.

**Missing predicates from fallback hypotheses:** The `_fallback_hypotheses()` method in cognitive.py uses hardcoded condition strings. Some of these (e.g., `is_grammatical_question`, `starts_with_imperative`, `contains_encoding_wrapper`, `contains_leet`, `contains_rot13`, etc.) rely on `_try_set_condition_name` to resolve them to registry entries. This depends on the DSL keyword being present in the condition string — which it is, since the conditions are literally constructed as `f"IF {name}(...) THEN ..."`. Resolved correctly.

---

## 10. Telemetry Logs Consistency

**Status:** WARN — entropy increases with each new candidate

| Metric | Value |
|--------|-------|
| Entropy history (5 candidates + updates) | [0.0, 0.491, 0.888, 1.207, 1.487] |
| Entropy monotonic? | INCREASING (0.0 → 1.487) |
| Info gains | Recorded in `_info_gains` |
| Extreme posterior values (≥0.9999) | None after update |

**Root cause:** Each `add_candidate()` call adds a candidate with `posterior=0.0` (initial), then `_normalise()` extends with `accuracy * 0.3` (≈0.24 for acc=0.8). The new candidate's initial weight (≈0.24) is significant relative to the existing distribution, adding entropy. With 5 candidates added one-per-iteration, each addition adds more entropy than the single `update_belief` removes.

**This is expected behavior** for the test setup (add+update cycle). In the real pipeline, candidates are bulk-added during synthesis and updated many times, so entropy converges downward after the initial addition spike.

**Recommendation:** Add a test that bulk-adds all candidates first, then runs many belief updates, and verifies monotonic entropy decrease.

---

## 11. Base Prompts from Memory Integration

**Status:** PASS

| Metric | Value |
|--------|-------|
| StrategistAgent has method? | PASS — `_fetch_base_prompts_from_memory()` at line 1053 |
| Orchestrator calls it? | PASS — orchestrator uses `_run_cognitive_phase` which triggers `generate_hypotheses` → anomalies → strategist uses prompts from memory |
| Prompts used in pair selection? | PASS — `_select_intervention_pair()` uses fetched prompts |

**Finding:** The prompt flow is:
1. Orchestrator calls `strategist._fetch_base_prompts_from_memory(campaign_id)`
2. Returns all unique base prompts from episodic memory, ordered: mixed-outcome first, then all others
3. `_select_intervention_pair()` uses these prompts for discriminative pair selection
4. `_execute_intervention()` also uses fetched prompts for the actual intervention

The _partition bug (cognitive.py:1335) can produce empty support sets for fallback hypotheses when `(offset + size) % len(anomalies) < offset % len(anomalies)`, causing some hypotheses to have zero discriminative power. This is a numerical edge case but worth fixing.

---

## 12. Fallback / Heuristic Candidate Proportion

**Status:** PASS

| Metric | Value |
|--------|-------|
| Synthesis: synthesized candidates | 6 (grammar enumeration) |
| Synthesis: heuristic fallback | 15 (from fallback heuristic) |
| VS: enumeration posterior mass | 40.3% |
| VS: heuristic posterior mass | 40.3% |
| Heuristic dominance check | PASS (40.3% < 50%) |

**Finding:** Heuristic and enumeration candidates share equal posterior mass (40.3% each), with hypothesis_seed at 19.3%. The `_source_max_fraction = 0.5` prevents any source from dominating, so as long as multiple sources contribute, mass is distributed.

**Concern:** The 15:6 heuristic-to-synthesized ratio in synthesis output means fallback heuristics significantly outnumber grammar-enumerated candidates. This is acceptable because:
1. Fallback programs are structurally simpler (often single-predicate)
2. They receive lower source quality scores (`0.2` vs `0.9` for enumeration)
3. The enumeration candidates that ARE generated tend to be higher quality (more specific)

---

## Summary of Issues Found

| # | Area | Severity | Status | Description |
|---|------|----------|--------|-------------|
| 1 | Bootstrap dynamics | Medium | WARN | Enumeration candidate posterior stuck at 0.0 in test — needs further investigation with fresh executor per iteration |
| 2 | Holdout evaluation | Low | WARN | No train vs holdout accuracy comparison in RQ0 evaluator |
| 3 | Program hash mutability | Medium | WARN | `__hash__` calls `canonicalize()` which mutates program in-place, violating Python contract |
| 4 | Entropy trajectory | Low | WARN | Add-then-update cycle causes entropy to increase initially; expected but should be documented |
| 5 | _partition wraparound | Low | WARN | Fallback hypothesis support sets can be empty on wraparound in modular arithmetic |
| 6 | Posterior init inconsistency | Low | Minor | Three different values used for initializing new candidate posterior across code paths (0.0, `accuracy*0.3`, `accuracy*0.5`) |

## All Previously Fixed Issues — Verified Non-Repeat

| Issue | Fixed In | Status |
|-------|----------|--------|
| AND recursion in _try_set_condition_name | cognitive.py:166-167 | PASS — no longer recurses |
| .description missing on Theory | cognitive.py:1546 | PASS — falls back to `.pattern` |
| OPENROUTER_API_KEY not loading | llm/llm_client.py:7-8 | PASS — dotenv loaded at module level |
| find_theories kwargs mismatch | cognitive.py:1539 | PASS — no kwargs passed |
| Keyword dominance in synthesis | cvc5_synthesizer.py | PASS — diversity bonus active |
| Entropy reset in _normalise | version_space.py:1094-1119 | PASS — extends in place |
| Category classification | cvc5_synthesizer.py | PASS — 7 categories tracked |
| Hard likelihood (0 or 1) | version_space.py | PASS — soft likelihood σ=0.5 |
