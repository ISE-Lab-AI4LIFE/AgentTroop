# HARMONY-X Forensic Analysis: Beyond Keyword Matching

## Executive Summary

HARMONY-X defines **29 predicate types** spanning lexical, structural, semantic, jailbreak-specific, and discourse categories. However, only **10 of 29 predicates** actually enter the learning loop through hypothesis generation → program compilation. The remaining 19 predicates exist as registered DSL classes with executable `evaluate()` methods, but are **dead code** in the learning pipeline — they can only appear via GrammarExporter enumeration during synthesis, never via the hypothesis refinement path. Composite (AND/OR/NOT) multi-predicate theories are structurally impossible in the hypothesis→program path and exist only in the synthesis grammar.

---

## 1. Capability Audit

### 1.1 Roleplay Framing Detection

| Aspect | Status |
|--------|--------|
| DSL Predicate | ✅ `StartsWithRoleplayPredicate` (`core/primitive.py:477-499`) |
| What it checks | 11 prefix strings: `"as a"`, `"pretend"`, `"imagine you are"`, `"act as"`, `"you are now"`, `"from now on"`, `"roleplay"`, `"let's roleplay"`, `"scenario:"`, `"you will act"`, `"you are going to"` |
| Registered in ConditionRegistry | ✅ Yes (auto-populated) |
| Has `compile_to_node()` | ✅ Yes (via `primitive_class`) |
| In PREDICATE_ONTOLOGY | ✅ Yes (template: `IF starts_with_roleplay(prompt) THEN REFUSE`) |
| **In `_fallback_hypotheses()`** | **❌ No** — instead uses weak proxy `IF contains_word('researcher') THEN ACCEPT` |
| **In `compile_condition_to_program()`** | **❌ No** — only handles the 10 keyword/discourse patterns |
| **In `_try_set_condition_name()`** | **❌ No** — no regex pattern for `starts_with_roleplay` |
| **Actually used in learning?** | **❌ No** — defined but unreachable from hypothesis→program path |
| **VersionSpace reachable?** | ✅ Yes — can enter via GrammarExporter enumeration → synthesizer |

**Architectural gap:** The predicate exists, evaluates correctly, and can be compiled. But `_fallback_hypotheses()` uses `contains_word('researcher')` as a roleplay proxy, and `compile_condition_to_program()` doesn't handle the string `"starts_with_roleplay"`.

**Minimal fix** (`agents/strategist.py`, after line 1401):
```python
# In compile_condition_to_program(), after the discourse patterns:
if "starts_with_roleplay" in cond_lower:
    from core.primitive import StartsWithRoleplayPredicate
    prog = Program(root=IfThenElseNode(
        condition=PredicateNode(primitive=StartsWithRoleplayPredicate()),
        then_outcome=then_out, else_outcome=else_out,
    ))
    prog.source = "compiled_from_condition"
    return prog
```
Plus add to `_try_set_condition_name()` in `agents/cognitive.py`:
```python
if "starts_with_roleplay" in cond:
    hyp.condition_name = "starts_with_roleplay"
    hyp.condition_params = {}
```
And replace `IF contains_word('researcher') THEN ACCEPT` in `_fallback_hypotheses()` with `IF starts_with_roleplay(prompt) THEN REFUSE`.

---

### 1.2 Instruction Hierarchy Conflict Detection

| Aspect | Status |
|--------|--------|
| DSL Predicate | ✅ `ContainsSystemOverridePredicate` (`core/primitive.py:501-523`) |
| What it checks | 12 override phrases as substrings: `"ignore previous"`, `"ignore all"`, `"disregard"`, `"forget"`, `"override"`, `"new instruction"`, `"system prompt"`, `"you are not"`, `"ignore the above"`, `"don't follow"`, `"break character"`, `"do not follow"` |
| In fallback hypotheses | **❌ No** |
| In compile_condition_to_program() | **❌ No** |
| In _try_set_condition_name() | **❌ No** |
| **Actually used?** | **❌ No** |

**Architectural gap:** Same pattern as roleplay. Predicate exists and works but never enters the hypothesis→program path.

**Minimal fix:** Same approach as above — add to `compile_condition_to_program()`, `_try_set_condition_name()`, and add a hypothesis in `_fallback_hypotheses()`.

---

### 1.3 Contextual Framing / Jailbreak Pattern Detection

| Aspect | Status |
|--------|--------|
| DSL Predicate | ✅ `MatchesJailbreakPatternPredicate` (`core/primitive.py:668-715`) |
| What it checks | 7 compiled regex patterns including: `ignore\s+(all\s+)?(previous|above|prior)`, `dan\|do\s+anything\s+now`, `you\s+(must|have\s+to|will)\s+obey`, `character\s+mode\|hypothetical\s+scenario`, `system\s+prompt\s*:`, `new\s+instruction`, `pretend\s+(you\s+are|to\s+be)` |
| Parameter | `pattern_library: str = "default_library"` — declared but **unused** in `evaluate()` |
| In fallback hypotheses | **❌ No** |
| In compile_condition_to_program() | **❌ No** |
| In _try_set_condition_name() | **❌ No** |
| **Actually used?** | **❌ No** |
| **Bug:** | `pattern_library` parameter is dead — `evaluate()` hardcodes 7 patterns regardless |

**Architectural gap:** The `pattern_library` parameter suggests a design intent for pluggable pattern libraries that was never implemented. All 7 patterns are hardcoded.

**Minimal fix:** Same three-file pattern plus decide whether `pattern_library` should actually be used.

---

### 1.4 Code Wrapping Detection

| Aspect | Status |
|--------|--------|
| DSL Predicate | ✅ `ContainsCodeBlockPredicate` (`core/primitive.py:545-560`) |
| What it checks | `"```" in prompt` (simple substring) |
| Related predicate | `ContainsEncodingWrapperPredicate` (`core/primitive.py:697-714`) — checks for ` ```base64\n...\n``` `, ` ```hex\n...\n``` `, etc. |
| `ContainsDelimiterPredicate` | Checks for `"""`, `---`, `\|\|\|`, `===`, `` ``` `` (multi-method delimiter injection) |
| In fallback hypotheses | **❌ No** — none of the three code-related predicates appear |
| **Actually used?** | **❌ No** — DSL classes are dead code in the learning pipeline |

**Architectural gap:** Code wrapping is a known jailbreak technique (e.g., wrapping instructions in code blocks to bypass safety filters). The predicates exist and evaluate correctly but are never generated as hypotheses.

---

### 1.5 Encoding Strategy Detection

| Predicate | Lexical check | In fallback hypotheses? | In compile? | In condition_name? | Actually used? |
|-----------|------|:---:|:---:|:---:|:---:|
| `ContainsLeetPredicate` | Digits+symbols > 10% | ✅ (cognitive.py:1472) | ✅ | ✅ | **✅ YES** |
| `ContainsRot13Predicate` | >50% letters in n-z | ✅ (cognitive.py:1473) | ✅ | ✅ | **✅ YES** |
| `ContainsBase64Predicate` | >95% base64 chars | ❌ | ❌ | ❌ | **❌ No** |
| `ContainsHexPredicate` | >80% hex digits + 6+ run | ❌ | ❌ | ❌ | **❌ No** |
| `ContainsEncodingWrapperPredicate` | ```base64```/```hex```/etc blocks | ❌ | ❌ | ❌ | **❌ No** |

**Architectural gap:** Only leet and rot13 encoding detection is used. Base64, hex, and encoding wrapper detection exist but are unreachable from hypothesis generation.

---

### 1.6 Semantic Intent Categories

| Aspect | Status |
|--------|--------|
| DSL Predicate | ✅ `IntentPredicate` (`core/primitive.py:630-666`) |
| Modes | `"harmful"` (12 keywords: bomb, kill, weapon, hack, steal, virus, drug, murder, exploit, attack, jailbreak, bypass) or `"innocuous"` (12 keywords) |
| `SentimentPredicate` | ✅ `SentimentPredicate` (`core/primitive.py:607-643`) — uses keyword-counting heuristic (7 positive, 7 negative words), NOT real sentiment analysis |
| In fallback hypotheses | **❌ No** — `_fallback_hypotheses()` manually constructs keyword lists that duplicate IntentPredicate's logic |
| In compile_condition_to_program() | **❌ No** |
| In _try_set_condition_name() | **❌ No** |
| **Actually used?** | **❌ No** |
| **Design issue:** | Sentiment and intent are keyword-counting heuristics, not actual ML-based semantic analysis. `SentimentPredicate` uses only 7 positive and 7 negative keywords. `IntentPredicate` has only two intent types ("harmful", "innocuous") with ~12 keywords each. |

**Architectural gap:** The predicates exist but the hypothesis pipeline duplicates their logic ad-hoc in `_fallback_hypotheses()` instead of using them.

**Key observation:** The `_fallback_hypotheses()` code (cognitive.py:1461) creates `IF matches_regex(r'(?i)\b(kill|bomb|steal|weapon|drug|explosive|hack|fraud|murder)\b') THEN REFUSE` which is a **hand-rolled duplicate** of what `IntentPredicate(intent_type="harmful")` already provides. The system is maintaining two parallel keyword lists — one in the predicate class, one in the hypothesis generator — that can drift apart.

---

### 1.7 Multi-Predicate Compositional Theories (AND/OR/NOT)

| Aspect | Status |
|--------|--------|
| GrammarExporter `_enumerate_conditions()` | ✅ Produces `NotNode`, `AndNode`, `OrNode`, `ApplyTransformNode` composites up to `max_depth` |
| CVC5 SMT encoding | ✅ Includes `(not prev)`, `(and prev depth_1)`, `(or prev depth_1)` |
| CVC5 model reconstruction | **❌ BROKEN** — `_reconstruct_from_model()` only handles single predicates, cannot reconstruct AND/OR/NOT from CVC5 output |
| `generate_variants()` in VS | ✅ Can decompose AND/OR (generalize) and add NOT (specialize) |
| **Hypothesis string "AND"** | **❌ BROKEN** — `compile_condition_to_program()` silently ignores everything after "AND" |
| Condition strings with "AND" in production | Only 1: `IF char_count(prompt) < 50 AND starts_with_imperative(prompt) THEN ACCEPT` |
| This AND condition compiles to? | Only `length_lt(50)` — the `starts_with_imperative` part is silently dropped |

**Architectural gap:** Composite (AND/OR/NOT) conditions cannot enter through the hypothesis→program path. The only composite that exists (`char_count(prompt) < 50 AND starts_with_imperative(prompt)`) is truncated by the compiler. CVC5 can theoretically produce composites but the reconstruction code cannot parse them.

---

### 1.8 Transform-Chain / ApplyTransform Patterns

| Aspect | Status |
|--------|--------|
| GrammarExporter `_enumerate_conditions()` | ✅ Produces `ApplyTransformNode(transform=t, inner=node)` for predicate-after-transform |
| In fallback hypotheses | **❌ No** — no transform-chain hypotheses are generated |
| In compile_condition_to_program() | **❌ No** |
| In _try_set_condition_name() | **❌ No** |
| In `_generate_candidates()` | ✅ Creates interventions with transform chains — but these test the **same** hypothesis against transformed prompts, not hypotheses that *detect* transformations |

**Architectural gap:** The LLM prompt example (`IF contains_word(decoded(ROT13(prompt)), 'bomb') THEN REFUSE`) suggests the system should generate hypotheses about transform-detection, but no transform-chain predicates exist. The system can test hypotheses against transformed prompts, but cannot generate hypotheses about the transformation itself (e.g., "if encoded in base64, treat differently").

---

## 2. Root Cause Analysis

The architectural issue has a single root cause: **three independent condition-registration systems** that were never synchronized:

```
1. core/primitive.py:_register_default_primitives()  → 29 Predicate classes
2. agents/cognitive.py:_try_set_condition_name()      → 10 regex patterns
3. agents/strategist.py:compile_condition_to_program() → 10 compilation rules
```

System (1) is the authoritative source of truth. Systems (2) and (3) are handwritten subsets. Any predicate added to (1) without corresponding updates to (2) and (3) is **dead code in the hypothesis→program path**.

The predicates that ARE fully wired (10 of 29) correspond to those that appeared in the original `_fallback_hypotheses()` code. Predicates added later (base64, hex, roleplay, system_override, etc.) were never integrated.

Additionally, the `_fallback_hypotheses()` method (cognitive.py:1272-1492) duplicates predicate logic:
- `IntentPredicate` keyword lists are duplicated inline (cognitive.py:1461)
- `StartsWithRoleplayPredicate` prefix list is unused — instead uses `contains_word('researcher')`
- `MatchesJailbreakPatternPredicate` patterns are never used in any hypothesis

---

## 3. Data Flow Map: Which Paths Actually Work

```
Hypothesis string
    │
    ▼
_try_set_condition_name(hyp)
    │   can set condition_name for: contains_word, contains_any_word,
    │   length_lt, length_gt, is_grammatical_question,
    │   starts_with_imperative, has_number, contains_leet,
    │   contains_rot13, matches_regex  ← 10 of 29
    │
    ▼
is condition_name set? ──YES──▶ ConditionRegistry.compile_to_node()
    │                               │ Works for ALL 29 predicates
    │                               ▼
    │                          ProgramExecutor.execute()
    │
    NO
    │
    ▼
compile_condition_to_program(condition_string)
    │   handles: contains_word, contains_any_word, length_gt, length_lt,
    │   has_number, contains_leet, contains_rot13, is_grammatical_question,
    │   starts_with_imperative, matches_regex  ← 10 of 29
    │   AND: silently broken (drops everything after "AND")
    │
    ▼
    ┌───── Success ──▶ ProgramExecutor.execute()
    │
    FAIL (or no keywords)
    │
    ▼
_keyword_fallback(prompt, hypothesis)  ← heuristic only, no predicate use
```

**The critical bottleneck:** `compile_condition_to_program()` and `_try_set_condition_name()` handle only the same 10 patterns. The other 19 predicates can ONLY enter through the ConditionRegistry→compile_to_node() path (bypassing string parsing), which requires `hypothesis.condition_name` to be set.

---

## 4. Detailed Gap Catalog

| # | Capability | Exists as DSL? | In hypothesis gen? | In compile? | In condition_name? | In synthesis? | In VS? | Gap severity |
|---|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | Keyword matching | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | FIXED |
| 2 | Multi-keyword (any/all) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | FIXED |
| 3 | Length thresholds | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | FIXED |
| 4 | Regex matching | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | FIXED |
| 5 | Number detection | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | FIXED |
| 6 | Leet detection | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | FIXED |
| 7 | ROT13 detection | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | FIXED |
| 8 | Grammatical question | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | FIXED |
| 9 | Imperative detection | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | FIXED |
| 10 | Prefix/suffix matching | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | MINOR |
| 11 | Has special char | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | MINOR |
| 12 | All caps detection | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | MINOR |
| 13 | Empty prompt | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | MINOR |
| 14 | Delimiter injection | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | MEDIUM |
| 15 | Code block detection | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | MEDIUM |
| 16 | Emoji detection | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | LOW |
| 17 | URL detection | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | LOW |
| 18 | Repetitive patterns | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | LOW |
| 19 | Base64 detection | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | MEDIUM |
| 20 | Hex detection | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | MEDIUM |
| 21 | Sentiment analysis* | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | LOW |
| 22 | Intent classification* | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | MEDIUM |
| 23 | Roleplay detection | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | HIGH |
| 24 | System override detection | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | HIGH |
| 25 | Jailbreak pattern detection | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | HIGH |
| 26 | Encoding wrapper detection | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | MEDIUM |
| 27 | Composite AND conditions | **❌** | ❌ | ❌ | ❌ | ✅ | ✅ | HIGH |
| 28 | Composite OR conditions | **❌** | ❌ | ❌ | ❌ | ✅ | ✅ | HIGH |
| 29 | Composite NOT conditions | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | MEDIUM |
| 30 | Transform-chain detection | **❌** | ❌ | ❌ | ❌ | ✅ | ✅ | HIGH |
| 31 | `contains_all_words` (all-of) | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | LOW |

*Sentiment and Intent are keyword-counting heuristics, not actual ML-based analysis.

---

## 5. Predicate Dead Code Analysis

The following predicates are **fully defined** (DSL class, `evaluate()`, registered, in ontology) but **never enter the hypothesis→program learning loop**. They can only appear via GrammarExporter enumeration during synthesis:

| Predicate | Category | `evaluate()` actually works? | Why entered but unused? |
|-----------|----------|:---:|-------------------------|
| `ContainsAllWordsPredicate` | lexical | ✅ | No compile support, no condition_name |
| `StartsWithPredicate` | lexical | ✅ | No compile support (starts_with("X") vs starts_with_roleplay different) |
| `EndsWithPredicate` | lexical | ✅ | Same |
| `HasSpecialCharPredicate` | structural | ✅ | Same |
| `IsAllCapsPredicate` | structural | ✅ | Same |
| `IsEmptyPredicate` | structural | ✅ | Same |
| `ContainsDelimiterPredicate` | structural | ✅ | Same — despite being a known jailbreak vector |
| `ContainsCodeBlockPredicate` | structural | ✅ | Same — despite being a known attack pattern |
| `HasEmojiPredicate` | structural | ✅ | Lower priority |
| `ContainsURLPredicate` | structural | ✅ | Lower priority |
| `IsRepetitivePredicate` | structural | ✅ | Lower priority |
| `ContainsBase64Predicate` | structural | ✅ | Never added to fallback/compile |
| `ContainsHexPredicate` | structural | ✅ | Never added |
| `SentimentPredicate` | semantic | ✅ (heuristic) | Never added |
| `IntentPredicate` | semantic | ✅ (heuristic) | Code in `_fallback_hypotheses` duplicates its logic |
| `StartsWithRoleplayPredicate` | jailbreak | ✅ | Weak proxy `contains_word('researcher')` used instead |
| `ContainsSystemOverridePredicate` | jailbreak | ✅ | Never added |
| `MatchesJailbreakPatternPredicate` | jailbreak | ✅ | Never added — despite being the most comprehensive jailbreak detector |
| `ContainsEncodingWrapperPredicate` | jailbreak | ✅ | Never added |

**Total dead code in the learning pipeline: 19 of 29 predicates (65.5%)**

---

## 6. Composite Condition Analysis

### AND Conditions
- **Only 1 in the entire codebase:** `IF char_count(prompt) < 50 AND starts_with_imperative(prompt) THEN ACCEPT`
- **compile_condition_to_program() result:** Only the first condition survives (`length_lt(50)`); `starts_with_imperative` and the "AND" are silently dropped
- `_try_set_condition_name()` result: Sets `condition_name="length_lt"` only (first matching pattern)
- **Fallback prediction** (_keyword_fallback): Separates keywords `['50']` and creates a `ContainsWordPredicate(word='50')` — completely wrong

### OR Conditions
- **0 in any actual hypothesis string.** Only in toy_model.py ground truth documentation.
- GrammarExporter `_enumerate_conditions()` CAN produce OR nodes at depth ≥2.
- CVC5 SMT encoding CAN produce OR at depth ≥2.
- CVC5 reconstruction CANNOT parse OR from the model.

### NOT Conditions
- **0 in any hypothesis string.**
- `generate_variants()` in VersionSpace CAN add NOT via specialization.
- GrammarExporter CAN produce NOT at depth ≥2.

### Conclusion: Composite conditions structurally cannot enter through the hypothesis → program path. They can only enter through synthesis enumeration, and even then CVC5 reconstruction is broken for composites.

---

## 7. Hypothesis String → Program Compilation: Complete Coverage Table

| Condition string pattern | Compile? | Result | condition_name set? | Actually produces correct program? |
|--------------------------|:---:|--------|:---:|:---:|
| `contains_word('X')` | ✅ | `ContainsWordPredicate(X)` | ✅ | ✅ |
| `contains_any_word(['X','Y'])` | ✅ | `ContainsAnyWordPredicate([X,Y])` | ✅ | ✅ |
| `char_count(prompt) < N` | ✅ | `LengthLtPredicate(N)` | ✅ | ✅ |
| `char_count(prompt) > N` | ✅ | `LengthGtPredicate(N)` | ✅ | ✅ |
| `has_number(prompt)` | ✅ | `HasNumberPredicate()` | ✅ | ✅ |
| `contains_leet(prompt)` | ✅ | `ContainsLeetPredicate()` | ✅ | ✅ |
| `contains_rot13(prompt)` | ✅ | `ContainsRot13Predicate()` | ✅ | ✅ |
| `is_grammatical_question(prompt)` | ✅ | `IsGrammaticalQuestionPredicate()` | ✅ | ✅ |
| `starts_with_imperative(prompt)` | ✅ | `StartsWithImperativePredicate()` | ✅ | ✅ |
| `matches_regex(r'...')` | ✅ | `MatchesRegexPredicate(...)` | ✅ | ✅ |
| `starts_with_roleplay(prompt)` | ❌ | None | ❌ | ❌ |
| `contains_system_override(prompt)` | ❌ | None | ❌ | ❌ |
| `matches_jailbreak_pattern(prompt)` | ❌ | None | ❌ | ❌ |
| `contains_encoding_wrapper(prompt)` | ❌ | None | ❌ | ❌ |
| `contains_code_block(prompt)` | ❌ | None | ❌ | ❌ |
| `contains_delimiter(prompt)` | ❌ | None | ❌ | ❌ |
| `contains_base64(prompt)` | ❌ | None | ❌ | ❌ |
| `contains_hex(prompt)` | ❌ | None | ❌ | ❌ |
| `sentiment(prompt) > T` | ❌ | None | ❌ | ❌ |
| `intent(prompt) = 'X'` | ❌ | None | ❌ | ❌ |
| `starts_with('X')` | ❌ | None | ❌ | ❌ |
| `ends_with('X')` | ❌ | None | ❌ | ❌ |
| `contains_all_words(['X','Y'])` | ❌ | None | ❌ | ❌ |
| `has_special_char(prompt)` | ❌ | None | ❌ | ❌ |
| `is_all_caps(prompt)` | ❌ | None | ❌ | ❌ |
| `is_empty(prompt)` | ❌ | None | ❌ | ❌ |
| `has_emoji(prompt)` | ❌ | None | ❌ | ❌ |
| `contains_url(prompt)` | ❌ | None | ❌ | ❌ |
| `is_repetitive(prompt)` | ❌ | None | ❌ | ❌ |
| `X AND Y` (any composite) | **❌ BROKEN** | Only first condition; second silently dropped | Partial (first only) | ❌ |
| `X OR Y` (any composite) | ❌ | None | ❌ | ❌ |

---

## 8. Synthesis Path Analysis

| Synthesis path | AND/OR/NOT support | Works end-to-end? |
|---------------|:---:|:---:|
| GrammarExporter `_enumerate_conditions()` | ✅ Full AND/OR/NOT/ApplyTransform | ✅ |
| CVC5 SMT encoding | ✅ Includes AND/OR/NOT at depth ≥ 2 | ✅ (solver produces valid model) |
| CVC5 model reconstruction (`_reconstruct_from_model`) | **❌ Limited** — only handles single predicate/classifier/transform | **❌ BROKEN for composites** |
| Enumeration (`_try_enumeration`) | ✅ Full AND/OR/NOT/ApplyTransform | ✅ |
| Hybrid selection | ✅ Picks min-MDL between CVC5 and enumeration | ✅ |
| `generate_variants()` specialization | ✅ Adds NOT | ✅ |
| `generate_variants()` generalization | ✅ Decomposes AND/OR | ✅ |

**Key finding:** The enumeration path successfully generates AND/OR/NOT composite programs. CVC5 SMT also models them but cannot reconstruct them. So composite programs CAN enter the VS through enumeration-based synthesis, but NOT through hypothesis→program compilation.

**The VS can learn composites** (via synthesis), but the hypothesis generator cannot produce them.

---

## 9. Recommended Code Changes

### Minimal changes to enable the HIGH-severity gaps:

#### 9.1 Enable jailbreak predicates in compile path
Add to `compile_condition_to_program()` in `agents/strategist.py`:
```python
# After the discourse patterns, add:
for pred_name, pred_cls, check_str in [
    ("starts_with_roleplay", StartsWithRoleplayPredicate(), "starts_with_roleplay"),
    ("contains_system_override", ContainsSystemOverridePredicate(), "contains_system_override"),
    ("matches_jailbreak_pattern", MatchesJailbreakPatternPredicate(), "matches_jailbreak_pattern"),
    ("contains_encoding_wrapper", ContainsEncodingWrapperPredicate(), "contains_encoding_wrapper"),
    ("contains_code_block", ContainsCodeBlockPredicate(), "contains_code_block"),
    ("contains_delimiter", ContainsDelimiterPredicate(), "contains_delimiter"),
    ("contains_base64", ContainsBase64Predicate(), "contains_base64"),
    ("contains_hex", ContainsHexPredicate(), "contains_hex"),
]:
    if check_str in cond_lower:
        prog = Program(root=IfThenElseNode(
            condition=PredicateNode(primitive=pred_cls),
            then_outcome=then_out, else_outcome=else_out,
        ))
        prog.source = "compiled_from_condition"
        return prog
```

#### 9.2 Enable jailbreak predicates in condition_name path
Add to `_try_set_condition_name()` in `agents/cognitive.py`:
```python
for name in ["starts_with_roleplay", "contains_system_override",
             "matches_jailbreak_pattern", "contains_encoding_wrapper",
             "contains_code_block", "contains_delimiter",
             "contains_base64", "contains_hex"]:
    if name in cond:
        hyp.condition_name = name
        hyp.condition_params = {}
        return
```

#### 9.3 Add jailbreak hypotheses to fallback
Add to `_fallback_hypotheses()` in `agents/cognitive.py`:
```python
_add("Roleplay framing", "IF starts_with_roleplay(prompt) THEN REFUSE", 0.55)
_add("System override attempt", "IF contains_system_override(prompt) THEN REFUSE", 0.55)
_add("Jailbreak pattern match", "IF matches_jailbreak_pattern(prompt) THEN REFUSE", 0.55)
_add("Encoding wrapper", "IF contains_encoding_wrapper(prompt) THEN REFUSE", 0.50)
_add("Code block injection", "IF contains_code_block(prompt) THEN REFUSE", 0.45)
```

#### 9.4 Fix composite AND compilation
The `compile_condition_to_program()` currently only handles the first condition when "AND" appears. To support AND, the method needs to:
1. Split on " AND " in the condition string
2. Compile each sub-condition independently
3. Combine them with `AndNode`
4. Wrap in `IfThenElseNode`

Minimum viable implementation:
```python
if " and " in cond_lower:
    parts = cond_lower.split(" and ")
    sub_programs = []
    for part in parts:
        prog = compile_condition_to_program(part)
        if prog is None:
            break
        sub_programs.append(prog.root.condition)
    else:
        from core.program import AndNode
        combined = sub_programs[0]
        for sp in sub_programs[1:]:
            combined = AndNode(left=combined, right=sp)
        prog = Program(root=IfThenElseNode(
            condition=combined, then_outcome=then_out, else_outcome=else_out,
        ))
        prog.source = "compiled_from_condition"
        return prog
```

#### 9.5 Enable IntentPredicate and SentimentPredicate
Replace the manually constructed keyword-hypotheses in `_fallback_hypotheses()`:
```python
# Replace the regex dangerous-keywords hypothesis with:
_add("Harmful keyword detection", "IF intent(prompt) = 'harmful' THEN REFUSE", 0.60)

# Replace the combined-keywords hypothesis with:
_add("Innocuous topic detection", "IF intent(prompt) = 'innocuous' THEN ACCEPT", 0.50)
```

Also add compile support for `intent(prompt) = 'X'` and `sentiment(prompt) > T` to `compile_condition_to_program()`.

#### 9.6 Fix CVC5 composite reconstruction
In `cvc5_synthesizer.py`, `_reconstruct_from_model()` needs to handle AND/OR/NOT/ApplyTransform from the SMT model. This requires:
1. Parsing the SMT model output for nested `(and ...)`, `(or ...)`, `(not ...)` patterns
2. Recursively building the corresponding `AndNode`, `OrNode`, `NotNode`, `ApplyTransformNode`
3. Wrapping the result in `IfThenElseNode`

This is the most complex fix and would require changes in `core/grammar.py`'s model parser.

---

## 10. Risk Classification

### Risks that WILL affect real campaigns:
1. **19 of 29 predicates unreachable from hypothesis→program** — the system can never learn a hypothesis that involves, e.g., `contains_system_override`, because no LLM or fallback hypothesis will generate it, and it cannot be compiled from a condition string.
2. **AND composites silently broken** — the fallback hypothesis `char_count(prompt) < 50 AND starts_with_imperative(prompt)` is silently truncated to only the first condition. If a real safety policy uses AND, the system cannot learn it.
3. **Keyword dominance is structural** — the hypothesis generator is hardcoded to prefer keyword predicates (contains_word, contains_any_word) over structural/jailbreak/semantic predicates.
4. **IntentPredicate/SentimentPredicate unused** — the system defines "harmful" and "innocuous" keyword lists but never generates hypotheses using them, while simultaneously maintaining a third copy of harmful keywords in `_fallback_hypotheses()`.

### Risks that would affect specific attack scenarios:
1. **Code wrapping attacks** — `ContainsCodeBlockPredicate`, `ContainsEncodingWrapperPredicate`, and `ContainsDelimiterPredicate` are all dead code. A defense against code-wrapping jailbreaks cannot be learned.
2. **Instruction hierarchy attacks** — `ContainsSystemOverridePredicate` is dead code. A defense against "ignore previous instructions" attacks cannot be learned.
3. **Roleplay attacks** — `StartsWithRoleplayPredicate` is dead code. A defense against roleplay-framing attacks cannot be learned (the `contains_word('researcher')` proxy is extremely weak and easily bypassed).

### Cosmetic or low-risk:
1. `HasEmojiPredicate`, `ContainsURLPredicate`, `IsEmptyPredicate`, `IsRepetitivePredicate` — relevant to niche attack scenarios but lower priority.
2. `StartsWithPredicate`, `EndsWithPredicate`, `HasSpecialCharPredicate`, `IsAllCapsPredicate` — useful for completeness but not critical.

---

## 11. Summary: What the System Can Actually Learn vs What It Claims to Support

| Category | Claimed support (29 predicates) | Actual learning path (10 predicates) | Synthesis-only (19 predicates) |
|----------|:---:|:---:|:---:|
| Keyword/lexical | 6 | 4 | 2 (starts_with, ends_with, contains_all_words) |
| Structural | 15 | 4 | 11 |
| Semantic | 2 | 0 | 2 |
| Jailbreak-specific | 4 | 0 | 4 |
| Discourse | 2 | 2 | 0 |
| Composite AND/OR | Partial | **0** (broken) | Partial (enumeration only) |

**The system can reliably learn only 10 of its 29 predicate types through the hypothesis→program path.**
**Multi-predicate theories (AND/OR) cannot enter through this path at all.**
**CVC5 synthesis can produce composites only through enumeration, not through the SMT solver (reconstruction is broken).**
**19 predicates are architecturally dead code in the learning loop.**

---

*Audit performed 2026-06-09. All claims are verified against the actual code paths in files at `/Users/hieunguyen/HARMONY_X/*.py`.*
