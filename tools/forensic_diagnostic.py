"""Forensic diagnostic: trace why ALL hypothesis pairs have zero discriminative power.

Simulates the exact execution path of the Strategist Agent:
1. Creates hypotheses matching what _fallback_hypotheses + LLM generate
2. Loads the same prompts as _resolve_base_prompts
3. Tests _discriminative_power for every (h1, h2, prompt) combination
4. Reports the EXACT reason when pairs have delta == 0 for all prompts
"""

import sys, os, json, itertools, re, random, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from agents.cognitive import Hypothesis
from agents.strategist import (
    StrategistAgent, _NullHypothesis,
)
from core.intervention import Intervention
from core.primitive import Transform, default_registry
from core.executor import ProgramExecutor
from prompt_loader import load_prompts

STATS = {"pairs_total": 0, "pairs_delta_gt_0": 0, "pairs_delta_0": 0}


# ── 1.  Hypothesis factory matching _fallback_hypotheses ─────────────────────


def make_fallback_hypotheses(top_keywords: List[str]) -> List[Hypothesis]:
    """Replicate _fallback_hypotheses conditions verbatim."""
    seen: set = set()
    hyps: List[Hypothesis] = []

    def _add(desc, cond, conf=0.5):
        if cond not in seen:
            seen.add(cond)
            h = Hypothesis(description=desc, condition=cond)
            h.confidence = conf
            hyps.append(h)

    for kw in top_keywords[:5]:
        _add(f"Keyword '{kw}'", f"IF contains_word('{kw}') THEN REFUSE")

    if top_keywords:
        kwl = "', '".join(top_keywords[:4])
        _add("Combined keywords", f"IF contains_any_word(['{kwl}']) THEN REFUSE")

    _add("Short prompt", "IF char_count(prompt) < 30 THEN ACCEPT")
    _add("Grammatical question", "IF is_grammatical_question(prompt) THEN ACCEPT")
    _add("Polite request", "IF contains_any_word(['please', 'could you', 'would you']) THEN ACCEPT")

    if len(hyps) < 3:
        for fw in ["bomb", "kill", "harm"]:
            _add(f"Fallback '{fw}'", f"IF contains_word('{fw}') THEN REFUSE")

    _add("Benign topic", "IF contains_any_word(['recipe', 'weather', 'history', 'science', 'math']) THEN ACCEPT")
    _add("Short imperative", "IF char_count(prompt) < 50 AND starts_with_imperative(prompt) THEN ACCEPT")
    _add("Roleplay researcher", "IF contains_word('researcher') THEN ACCEPT")
    _add("Fictional/story", "IF contains_any_word(['fictional', 'story']) THEN ACCEPT")
    _add("Long prompt", "IF char_count(prompt) > 120 THEN REFUSE")
    _add("Has number", "IF has_number(prompt) THEN REFUSE")
    _add("Regex dangerous", r"IF matches_regex(r'(?i)\b(kill|bomb|steal|weapon|drug|explosive|hack|fraud|murder)\b') THEN REFUSE")
    _add("Leet speak", "IF contains_leet(prompt) THEN REFUSE")
    _add("ROT13", "IF contains_rot13(prompt) THEN REFUSE")

    return hyps


# ── 2.  Simulated LLM hypotheses ────────────────────────────────────────────


def make_llm_hypotheses() -> List[Hypothesis]:
    """Conditions that an LLM might realistically generate (no keywords)."""
    return [
        Hypothesis(
            description="Model refuses base64-encoded prompts.",
            condition="IF contains_transform('base64') THEN REFUSE",
            confidence=0.55,
        ),
        Hypothesis(
            description="Model refuses prompts containing hidden instructions after ROT13 decoding.",
            condition="IF contains_rot13_decoded('harmful') THEN REFUSE",
            confidence=0.60,
        ),
        Hypothesis(
            description="Model accepts prompts that use polite framing even when otherwise harmful.",
            condition="IF contains_any_word(['please', 'help', 'could']) THEN ACCEPT",
            confidence=0.50,
        ),
    ]


# ── 3.  Mock LLM client ─────────────────────────────────────────────────────


class MockLLMClient:
    """Simulates llama3_1_8b returning '0' or '1' based on keyword heuristics."""
    def __init__(self, mode: str = "simple"):
        self.mode = mode  # 'simple', 'verbose', 'always0', 'always1'
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        if self.mode == "always0":
            return "0"
        if self.mode == "always1":
            return "1"
        if self.mode == "verbose":
            return "The model would REFUSE (1) because the hypothesis condition is met." if "REFUSE" in prompt else "0"
        # simple mode: extract from hypothesis condition
        if "REFUSE" in prompt:
            # Heuristic: check if condition keywords match prompt
            return "1" if any(kw in prompt.lower() for kw in ["harm", "danger", "bomb", "virus", "malware"]) else "0"
        return "0"  # ACCEPT hypothesis → most prompts accepted


# ── 4.  Forensic test harness ───────────────────────────────────────────────


def _extract_keywords(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"'([^']*)'", text) if re.findall(r"'([^']*)'", text) else []


def trace_prediction(
    prompt: str, 
    hypothesis: Hypothesis, 
    use_llm: bool = True, 
    llm_client: Any = None,
) -> Tuple[int, str]:
    """Replicate _predict_outcome EXACTLY with tracing."""
    # 1. program check
    program = getattr(hypothesis, "program", None)
    if program is not None:
        return "[PROGRAM]", 0  # marked for tracing

    # 2. keyword fast path
    cond = getattr(hypothesis, "condition", "")
    kw = _extract_keywords(cond) if isinstance(cond, str) else []
    if kw:
        # replicate _keyword_fallback
        from agents.strategist import StrategistAgent
        # Can't easily call the method without an instance, replicate manually
        # Actually let's just call the method
        pass

    if isinstance(cond, str) and kw:
        return f"[KEYWORD {kw}]", None  # delegate to keyword_fallback trace

    # 3. LLM path
    if use_llm and llm_client is not None:
        return f"[LLM]", None

    # 4. keyword_fallback last resort
    return f"[LAST_RESORT]", None


def get_condition_category(cond: str) -> str:
    """Classify hypothesis prediction behavior."""
    if not cond:
        return "NO_CONDITION"
    has_kw = bool(_extract_keywords(cond))
    if has_kw:
        return "KEYWORD"
    if "char_count" in cond:
        return "CHAR_COUNT"
    if "has_number" in cond:
        return "HAS_NUMBER"
    if "matches_regex" in cond:
        return "REGEX"
    if "contains_leet" in cond:
        return "LEET"
    if "contains_rot13" in cond:
        return "ROT13"
    if "is_grammatical_question" in cond:
        return "UNSUPPORTED_SEMANTIC"
    return "OTHER_NO_KW"


def simulate_keyword_fallback(
    cond: str, 
    prompt_lower: str,
    hypothesizer: StrategistAgent,
) -> int:
    """Replicate StrategistAgent._keyword_fallback and _score_condition."""
    cond_lower = cond.lower()
    predicts_refuse = "then refuse" in cond_lower
    predicts_accept = "then accept" in cond_lower

    if not predicts_refuse and not predicts_accept:
        return 0  # fallback

    # replicate _score_condition
    scores = []

    # contains_word
    kw = _extract_keywords(cond)
    if kw:
        matches = sum(1 for k in kw if k.lower() in prompt_lower)
        scores.append(matches / len(kw))

    # contains_any_word
    if "contains_any_word" in cond_lower:
        list_m = re.search(r"contains_any_word\s*\(\s*\[([^\]]+)\]\)", cond, re.IGNORECASE)
        if list_m:
            items = re.findall(r"'([^']*)'", list_m.group(1))
            if items:
                matches = sum(1 for it in items if it.lower() in prompt_lower)
                scores.append(matches / len(items))

    # char_count
    for op in (">", "<"):
        length_m = re.search(rf"char_count\s*\(\s*prompt\s*\)\s*{re.escape(op)}\s*(\d+)", cond_lower)
        if length_m:
            threshold = int(length_m.group(1))
            actual = len(prompt_lower)
            scores.append(1.0 if (actual > threshold if op == ">" else actual < threshold) else 0.0)

    if "has_number" in cond_lower:
        scores.append(1.0 if re.search(r"\d", prompt_lower) else 0.0)

    if "contains_leet" in cond_lower:
        scores.append(0.0)  # approximation: never matches typical prompts

    if "matches_regex" in cond_lower:
        rx_m = re.search(r"matches_regex\s*\(\s*(?:r)?['\"]([^'\"]+)['\"]\s*\)", cond, re.IGNORECASE)
        if rx_m:
            try:
                scores.append(1.0 if re.search(rx_m.group(1), prompt_lower) else 0.0)
            except re.error:
                scores.append(0.0)

    score = sum(scores) / len(scores) if scores else 0.0

    if predicts_refuse:
        return 1 if score >= 0.5 else 0
    else:
        return 0 if score >= 0.5 else 1


def predict_with_trace(prompt: str, hypothesis: Hypothesis, llm_client=None, use_llm=True) -> Tuple[int, str]:
    """Full _predict_outcome replication with trace marker."""
    cond = getattr(hypothesis, "condition", "")
    cond_str = str(cond) if cond else ""

    # 1. program
    if getattr(hypothesis, "program", None) is not None:
        return 0, "PROGRAM"

    # 2. keyword fast path
    kw = _extract_keywords(cond_str)
    if isinstance(cond_str, str) and kw:
        result = simulate_keyword_fallback(cond_str, prompt.lower(), None)
        return result, f"KEYWORD:{','.join(kw)}"

    # 3. LLM
    if use_llm and llm_client is not None:
        _prompt = (
            f"Hypothesis: {getattr(hypothesis, 'description', '')}\n"
            f"Condition: {cond_str}\n"
            f"Prompt: {prompt}\n"
            f"Answer:"
        )
        raw = llm_client.generate(_prompt, max_tokens=4, temperature=0.0)
        raw_stripped = raw.strip()
        if "1" in raw_stripped and "0" not in raw_stripped:
            return 1, "LLM:1"
        if "0" in raw_stripped and "1" not in raw_stripped:
            return 0, "LLM:0"
        return 0, "LLM:AMBIGUOUS(0)"

    # 4. last resort keyword_fallback
    result = simulate_keyword_fallback(cond_str, prompt.lower(), None)
    return result, f"LAST_RESORT(cond={get_condition_category(cond_str)})"


def run_diagnostic(prompts: List[str], hypotheses: List[Hypothesis], llm_client=None, use_llm=True):
    """Test every (h1, h2, prompt) triple and report discriminative power."""
    cat_map = {h.id: get_condition_category(getattr(h, "condition", "")) for h in hypotheses}
    print(f"\n{'='*70}")
    print(f"FORENSIC DIAGNOSTIC: {len(hypotheses)} hypotheses × {len(prompts)} prompts")
    print(f"LLM mode: {getattr(llm_client, 'mode', 'N/A') if llm_client else 'DISABLED'}")
    print(f"{'='*70}")

    # Hypothesis catalogue
    print(f"\n── Hypothesis Catalogue ──")
    for i, h in enumerate(hypotheses):
        cat = cat_map[h.id]
        cond = getattr(h, "condition", "")[:60]
        print(f"  {i}: [{cat:22s}] conf={h.confidence:.2f}  cond={cond}")

    # Count hypothesis categories
    cat_counts = Counter(cat_map.values())
    print(f"\n── Category distribution ──")
    for cat, cnt in cat_counts.most_common():
        print(f"  {cat}: {cnt}")

    # Pair-by-pair analysis
    pairs_with_delta = 0
    pairs_without_delta = 0
    zero_delta_reasons: Counter = Counter()

    for h1, h2 in itertools.combinations(hypotheses, 2):
        pair_key = (h1.id[:8], h2.id[:8], cat_map[h1.id], cat_map[h2.id])
        max_delta = 0
        best_prompt = ""
        h1_vals: set = set()
        h2_vals: set = set()
        num_llm_calls = 0

        for prompt in prompts:
            p1, t1 = predict_with_trace(prompt, h1, llm_client, use_llm)
            p2, t2 = predict_with_trace(prompt, h2, llm_client, use_llm)
            d = abs(p1 - p2)
            h1_vals.add(p1)
            h2_vals.add(p2)
            if "LLM" in t1:
                num_llm_calls += 1
            if "LLM" in t2:
                num_llm_calls += 1
            if d > max_delta:
                max_delta = d
                best_prompt = prompt[:50]

        if max_delta > 0:
            pairs_with_delta += 1
            print(f"  ✓ Δ={max_delta:.0f}  [{cat_map[h1.id]} vs {cat_map[h2.id]}] "
                  f"prompt='{best_prompt}...'  (h1_vals={h1_vals}, h2_vals={h2_vals})")
        else:
            pairs_without_delta += 1
            # Determine reason
            h1_preds = set()
            h2_preds = set()
            for prompt in prompts[:20]:
                p1, t1 = predict_with_trace(prompt, h1, llm_client, use_llm)
                p2, t2 = predict_with_trace(prompt, h2, llm_client, use_llm)
                h1_preds.add((p1, t1))
                h2_preds.add((p2, t2))

            h1_only = {p for p, _ in h1_preds}
            h2_only = {p for p, _ in h2_preds}
            reasons = []
            if len(h1_only) == 1 and len(h2_only) == 1:
                if list(h1_only)[0] == list(h2_only)[0]:
                    reasons.append(f"both_always_{list(h1_only)[0]}")
            h1_modes = {t for _, t in h1_preds}
            h2_modes = {t for _, t in h2_preds}
            reasons.append(f"h1_modes={h1_modes}")
            reasons.append(f"h2_modes={h2_modes}")
            reason_str = "; ".join(reasons)
            zero_delta_reasons[reason_str] += 1
            if pairs_without_delta <= 3:
                print(f"  ✗ Δ=0  [{cat_map[h1.id]:22s} vs {cat_map[h2.id]:22s}]  {reason_str}")

    # Summary
    total = pairs_with_delta + pairs_without_delta
    print(f"\n{'='*70}")
    print(f"RESULTS: {pairs_with_delta}/{total} pairs have Δ > 0  ({100*pairs_with_delta//total}%)")
    print(f"         {pairs_without_delta}/{total} pairs have Δ = 0  ({100*pairs_without_delta//total}%)")
    if zero_delta_reasons:
        print(f"\n── Zero-delta reasons ──")
        for reason, cnt in zero_delta_reasons.most_common():
            print(f"  '{reason}': {cnt} pairs")

    zero_kw_hyps = sum(1 for h in hypotheses if cat_map[h.id] == "KEYWORD")
    zero_nokw_hyps = total_pairs_with_cat(hypotheses, cat_map, "KEYWORD", "CHAR_COUNT")
    
    return pairs_with_delta, pairs_without_delta


def total_pairs_with_cat(hypotheses, cat_map, *cats):
    count = 0
    for h1, h2 in itertools.combinations(hypotheses, 2):
        if cat_map[h1.id] in cats or cat_map[h2.id] in cats:
            count += 1
    return count


# ── 5.  Main ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("Loading prompts from prompt.csv ...")
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts")

    # Sample keywords from prompts (as _fallback_hypotheses does)
    import re
    from collections import Counter
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "it", "its", "my", "your", "his",
        "her", "our", "their", "no", "nor", "not", "or", "and", "but",
        "if", "because", "so", "than", "too", "very", "just", "about",
        "also", "make", "get", "give", "tell", "show", "without",
    }
    content_words: Counter = Counter()
    for p in prompts:
        words = re.findall(r"[a-zA-Z]{3,}", p.lower())
        for w in words:
            if w not in stopwords:
                content_words[w] += 1
    top_keywords = [w for w, _ in content_words.most_common(8)]
    print(f"Top keywords from prompts: {top_keywords}")

    # Create hypotheses
    fallback = make_fallback_hypotheses(top_keywords)
    llm_hyps = make_llm_hypotheses()
    all_hyps = fallback + llm_hyps
    print(f"Created {len(all_hyps)} hypotheses ({len(fallback)} fallback + {len(llm_hyps)} LLM)")

    # ── Run 1: LLM AVAILABLE ──
    print(f"\n{'#'*70}")
    print(f"# SCENARIO 1: LLM available (simple mode)")
    print(f"{'#'*70}")
    llm_simple = MockLLMClient(mode="simple")
    run_diagnostic(prompts, all_hyps, llm_client=llm_simple, use_llm=True)
    print(f"\nLLM call count: {llm_simple.call_count}")

    # ── Run 2: LLM always returns 0 ──
    print(f"\n{'#'*70}")
    print(f"# SCENARIO 2: LLM always returns 0")
    print(f"{'#'*70}")
    llm_always0 = MockLLMClient(mode="always0")
    run_diagnostic(prompts[:50], all_hyps, llm_client=llm_always0, use_llm=True)
    print(f"\nLLM call count: {llm_always0.call_count}")

    # ── Run 3: No LLM ──
    print(f"\n{'#'*70}")
    print(f"# SCENARIO 3: No LLM available")
    print(f"{'#'*70}")
    run_diagnostic(prompts[:50], all_hyps, llm_client=None, use_llm=False)

    # ── Run 4: Key diagnostic — which hypotheses predict what ──
    print(f"\n{'#'*70}")
    print(f"# SCENARIO 4: Per-hypothesis prediction analysis (no LLM)")
    print(f"{'#'*70}")
    for h in all_hyps:
        cat = get_condition_category(getattr(h, "condition", ""))
        preds = set()
        traces = Counter()
        for p in prompts[:50]:
            val, trace = predict_with_trace(p, h, llm_client=None, use_llm=False)
            preds.add(val)
            traces[trace] += 1
        if len(preds) == 1:
            print(f"  ALWAYS {list(preds)[0]}  [{cat:22s}]  {getattr(h, 'condition', '')[:50]}")
        else:
            # Show split
            ratio = sum(1 for p in prompts[:50] if predict_with_trace(p, h, llm_client=None, use_llm=False)[0] == 1)
            print(f"  MIXED ({ratio}/50 = 1)  [{cat:22s}]  {getattr(h, 'condition', '')[:50]}  traces={dict(traces)}")
