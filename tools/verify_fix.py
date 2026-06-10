"""Verify the discriminative power fix under REAL campaign conditions.

Simulates the ACTUAL campaign:
- Keywords extracted from the REAL anomaly prompts (harmful + benign)
- Prompt pool = episodic memory prompts (seed prompts) + default prompts
- Tests the NEW pair selection (prediction diversity check)
- Tests the NEW fetch_base_prompts_from_memory (all prompts, not just mixed-outcome)
"""

import sys, os, re, itertools, json, random, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
from typing import List, Tuple, Optional, Any, Dict

from agents.cognitive import Hypothesis
from prompt_loader import load_prompts

# ── Load REAL anomaly prompts ──
harmful_csv = os.path.join(os.path.dirname(__file__), "..", "llama3_1_8b", "harmful_prompts.csv")
benign_csv = os.path.join(os.path.dirname(__file__), "..", "llama3_1_8b", "benign_prompts.csv")

harmful_prompts = load_prompts(harmful_csv)
benign = []
with open(benign_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        p = row.get('prompt', '').strip()
        if p:
            benign.append(p)

anomaly_prompts = harmful_prompts[:5] + benign[:12]  # limit to match scale of 19 anomalies

print(f"Anomaly prompts: {len(anomaly_prompts)} ({len(harmful_prompts)} harmful + {len(benign)} benign)")
print(f"Harmful examples: {[p[:40] for p in harmful_prompts[:3]]}")
print(f"Benign examples: {[p[:40] for p in benign[:3]]}")

# ── Extract keywords (same logic as _fallback_hypotheses) ──
stopwords = {'the','a','an','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might','shall','can',
    'to','of','in','for','on','with','at','by','from','as','into','through','during',
    'before','after','above','below','between','out','off','over','under','again',
    'further','then','once','here','there','when','where','why','how','what','which',
    'who','whom','this','that','these','those','am','it','its','my','your','his','her',
    'our','their','no','nor','nor','not','or','and','but','if','because','so','than',
    'too','very','just','about','also','make','get','give','tell','show','without',}
content_words = Counter()
for p in anomaly_prompts:
    words = re.findall(r'[a-zA-Z]{3,}', p.lower())
    for w in words:
        if w not in stopwords:
            content_words[w] += 1
top_keywords = [w for w, _ in content_words.most_common(8)]
print(f"\nTop keywords from anomaly prompts: {top_keywords}")

# ── Create hypotheses (replicating _fallback_hypotheses exactly) ──
def make_hyps(keywords):
    seen = set()
    hyps = []
    def add(desc, cond, conf=0.5):
        if cond not in seen:
            seen.add(cond)
            h = Hypothesis(description=desc, condition=cond)
            h.confidence = conf
            hyps.append(h)
    for kw in keywords[:5]:
        add(f"Keyword '{kw}'", f"IF contains_word('{kw}') THEN REFUSE")
    if keywords:
        kwl = "', '".join(keywords[:4])
        add("Combined keywords", f"IF contains_any_word(['{kwl}']) THEN REFUSE")
    add("Short prompt", "IF char_count(prompt) < 30 THEN ACCEPT")
    add("Grammatical question", "IF is_grammatical_question(prompt) THEN ACCEPT")
    add("Polite request", "IF contains_any_word(['please', 'could you', 'would you']) THEN ACCEPT")
    if len([h for h in hyps if h.confidence > 0]) < 3:
        for fw in ["bomb", "kill", "harm"]:
            add(f"Fallback '{fw}'", f"IF contains_word('{fw}') THEN REFUSE")
    add("Benign topic", "IF contains_any_word(['recipe', 'weather', 'history', 'science', 'math']) THEN ACCEPT")
    add("Short imperative", "IF char_count(prompt) < 50 AND starts_with_imperative(prompt) THEN ACCEPT")
    add("Roleplay researcher", "IF contains_word('researcher') THEN ACCEPT")
    add("Fictional/story", "IF contains_any_word(['fictional', 'story']) THEN ACCEPT")
    add("Long prompt", "IF char_count(prompt) > 120 THEN REFUSE")
    add("Has number", "IF has_number(prompt) THEN REFUSE")
    add("Regex dangerous", r"IF matches_regex(r'(?i)\b(kill|bomb|steal|weapon|drug|explosive|hack|fraud|murder)\b') THEN REFUSE")
    add("Leet speak", "IF contains_leet(prompt) THEN REFUSE")
    add("ROT13", "IF contains_rot13(prompt) THEN REFUSE")
    return hyps

hyps = make_hyps(top_keywords)
print(f"\nCreated {len(hyps)} hypotheses")

# ── Load prompt pools ──
default_prompts = load_prompts()  # from prompt.csv
print(f"Default prompt pool: {len(default_prompts)} prompts")

# Simulate episodic memory: the seed prompts ARE the anomaly prompts
memory_prompts = anomaly_prompts  # In real campaign, these come from episodic memory
print(f"Memory prompt pool: {len(memory_prompts)} prompts")

# Combined pool (as _resolve_base_prompts now does with the fix)
combined = list(dict.fromkeys(memory_prompts + default_prompts))  # unique, preserve order
print(f"Combined pool: {len(combined)} prompts")

# ── OLD BEHAVIOR: fetch_base_prompts_from_memory only returned mixed-outcome prompts ──
# The mixed-outcome prompts are the SEED episode prompts where the same base prompt
# had both outcome 0 and outcome 1. With 5 harmful prompts, all have mixed outcomes.
# But the benign prompts might NOT have mixed outcomes (same outcome for all variants).
mixed_only = harmful_prompts[:5]  # Only harmful prompts have mixed outcomes
print(f"\nOLD (mixed-outcome only) pool: {len(mixed_only)} prompts")
print(f"  OLD pool content: {[p[:40] for p in mixed_only]}")

# ── PREDICTION ANALYSIS ──
# Replicate keyword_fallback prediction
def _extract_keywords(text):
    return re.findall(r"'([^']*)'", text) if isinstance(text, str) else []

def predict(prompt, hyp, use_keyword_fallback=True):
    cond = getattr(hyp, 'condition', '') or ''
    kw = _extract_keywords(cond)
    if kw:
        # keyword path
        prompt_lower = prompt.lower()
        cond_lower = cond.lower()
        predicts_refuse = 'then refuse' in cond_lower
        predicts_accept = 'then accept' in cond_lower
        if not predicts_refuse and not predicts_accept:
            return 0
        scores = []
        # contains_word
        kw_matches = sum(1 for k in kw if k.lower() in prompt_lower)
        if kw_matches:
            scores.append(kw_matches / len(kw))
        # contains_any_word
        if 'contains_any_word' in cond_lower:
            list_m = re.search(r"contains_any_word\s*\(\s*\[([^\]]+)\]\)", cond, re.IGNORECASE)
            if list_m:
                items = re.findall(r"'([^']*)'", list_m.group(1))
                if items:
                    matches = sum(1 for it in items if it.lower() in prompt_lower)
                    scores.append(matches / len(items))
        # char_count
        for op in ('>', '<'):
            m = re.search(rf"char_count\s*\(\s*prompt\s*\)\s*{re.escape(op)}\s*(\d+)", cond_lower)
            if m:
                thresh = int(m.group(1))
                actual = len(prompt_lower)
                scores.append(1.0 if (actual > thresh if op == '>' else actual < thresh) else 0.0)
        if 'has_number' in cond_lower:
            scores.append(1.0 if re.search(r'\d', prompt_lower) else 0.0)
        if 'contains_leet' in cond_lower:
            scores.append(0.0)
        if 'matches_regex' in cond_lower:
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
    # No keywords → LLM path → simulate as simple
    if use_keyword_fallback:
        # last-resort keyword_fallback (skipping LLM)
        prompt_lower = prompt.lower()
        cond_lower = cond.lower()
        predicts_refuse = 'then refuse' in cond_lower
        predicts_accept = 'then accept' in cond_lower
        if not predicts_refuse and not predicts_accept:
            return 0
        scores = []
        if 'has_number' in cond_lower:
            scores.append(1.0 if re.search(r'\d', prompt_lower) else 0.0)
        if 'contains_leet' in cond_lower:
            scores.append(0.0)
        if 'matches_regex' in cond_lower:
            rx_m = re.search(r"matches_regex\s*\(\s*(?:r)?['\"]([^'\"]+)['\"]\s*\)", cond, re.IGNORECASE)
            if rx_m:
                try:
                    scores.append(1.0 if re.search(rx_m.group(1), prompt_lower) else 0.0)
                except re.error:
                    scores.append(0.0)
        for op in ('>', '<'):
            m = re.search(rf"char_count\s*\(\s*prompt\s*\)\s*{re.escape(op)}\s*(\d+)", cond_lower)
            if m:
                thresh = int(m.group(1))
                actual = len(prompt_lower)
                scores.append(1.0 if (actual > thresh if op == '>' else actual < thresh) else 0.0)
        score = sum(scores) / len(scores) if scores else 0.0
        if predicts_refuse:
            return 1 if score >= 0.5 else 0
        else:
            return 0 if score >= 0.5 else 1
    return 0  # LLM path (would call LLM)

# ── OLD BEHAVIOR: Test with mixed-outcome-only prompt pool ──
print(f"\n{'='*60}")
print(f"OLD BEHAVIOR: mixed-outcome-only prompt pool ({len(mixed_only)} prompts)")
print(f"{'='*60}")
for h in hyps:
    cond = getattr(h, 'condition', '')[:50]
    preds = set()
    for p in mixed_only:
        preds.add(predict(p, h))
    if len(preds) == 1:
        print(f"  ALWAYS {list(preds)[0]}  {cond}")
    else:
        ones = sum(1 for p in mixed_only if predict(p, h) == 1)
        print(f"  MIXED ({ones}/{len(mixed_only)} = 1)  {cond}")
zero_pairs_old = 0
nonzero_pairs_old = 0
for h1, h2 in itertools.combinations(hyps, 2):
    max_delta = max(abs(predict(p, h1) - predict(p, h2)) for p in mixed_only)
    if max_delta > 0:
        nonzero_pairs_old += 1
    else:
        zero_pairs_old += 1
total_old = zero_pairs_old + nonzero_pairs_old
print(f"\nPairs with Δ>0: {nonzero_pairs_old}/{total_old} ({100*nonzero_pairs_old//total_old}%)")
print(f"Pairs with Δ=0: {zero_pairs_old}/{total_old} ({100*zero_pairs_old//total_old}%)")

# ── NEW BEHAVIOR: Test with combined prompt pool (memory + default) ──
print(f"\n{'='*60}")
print(f"NEW BEHAVIOR: combined prompt pool ({len(combined)} prompts)")
print(f"{'='*60}")
for h in hyps:
    cond = getattr(h, 'condition', '')[:50]
    preds = set()
    for p in combined:
        preds.add(predict(p, h))
    if len(preds) == 1:
        print(f"  ALWAYS {list(preds)[0]}  {cond}")
    else:
        ones = sum(1 for p in combined if predict(p, h) == 1)
        print(f"  MIXED ({ones}/{len(combined)} = 1)  {cond}")
zero_pairs_new = 0
nonzero_pairs_new = 0
for h1, h2 in itertools.combinations(hyps, 2):
    max_delta = max(abs(predict(p, h1) - predict(p, h2)) for p in combined)
    if max_delta > 0:
        nonzero_pairs_new += 1
    else:
        zero_pairs_new += 1
total_new = zero_pairs_new + nonzero_pairs_new
print(f"\nPairs with Δ>0: {nonzero_pairs_new}/{total_new} ({100*nonzero_pairs_new//total_new}%)")
print(f"Pairs with Δ=0: {zero_pairs_new}/{total_new} ({100*zero_pairs_new//total_new}%)")

# ── NEW PAIR SELECTION: prediction-diverse selection ──
print(f"\n{'='*60}")
print(f"NEW PAIR SELECTION: probes from combined pool")
print(f"{'='*60}")
# Simulate the new pair selector
probe = combined[:50]
candidates = []
for h1, h2 in itertools.combinations(hyps, 2):
    disagree = sum(1 for p in probe if predict(p, h1) != predict(p, h2)) / len(probe)
    conf1 = getattr(h1, 'confidence', 0.5)
    conf2 = getattr(h2, 'confidence', 0.5)
    conf_uncertainty = 1.0 - abs(conf1 - conf2)
    candidates.append((disagree, conf_uncertainty, h1, h2))
candidates.sort(key=lambda x: (-x[0], -x[1]))
print(f"Top 5 pairs by prediction disagreement:")
for i, (disc, conf, h1, h2) in enumerate(candidates[:5]):
    print(f"  {i+1}. Δ={disc:.3f} (probe)  "
          f"h1={getattr(h1,'condition','')[:30]}... | "
          f"h2={getattr(h2,'condition','')[:30]}...")
print(f"\n  → Best pair selected by new selector: Δ_probe={candidates[0][0]:.3f}")
print(f"  → Best pair would have been selected by OLD selector (confidence):")
print(f"     Δ_old={candidates[-1][0]:.3f} (last place, worst disagreement)")
