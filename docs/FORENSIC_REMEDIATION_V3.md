# FORENSIC REMEDIATION V3 — Version Space Learning Quality

## Thay đổi so với V2

### V2 (đã loại bỏ)
- **Hard cap 40% cho keyword candidates**: Step 7 trong `update_belief()` ép keyword posterior ≤ 40%, bất kể evidence.
- **Không chấp nhận hội tụ keyword**: Convergence check từ chối nếu `best.predicate_type == "keyword"`.
- **Holdout chỉ cho top-5 candidates**: `evaluate_on_holdout(top_k=5)`.

### V3 (hiện tại)
- **Evidence-driven adjustment, không hard cap**: Soft predicate-type diversity bonus (hệ số 0.5× so với source bonus), không ép cứng.
- **Convergence purely evidence-driven**: Không kiểm tra predicate type.
- **Holdout trên TOÀN BỘ candidates**: `evaluate_on_holdout()` không còn tham số `top_k`.

---

## Files thay đổi

| File | Dòng | Thay đổi |
|------|------|----------|
| `inference/version_space.py` | ~15 | Remove hard cap step 7, thay bằng soft diversity bonus; làm holdout step 1 unconditional |
| `orchestration/orchestrator.py` | ~25 | Remove `not_keyword_only` convergence check; mở rộng holdout lên all candidates; remove `top_k` parameter |

---

## 1. Holdout-adjusted scoring — unconditional

### Trước (V2)
```python
# Chỉ chạy nếu CÓ candidate có holdout > 0
has_real_holdout = any(c.holdout_accuracy > 0.0 for c in self._candidates)
if has_real_holdout:
    for i, c in enumerate(self._candidates):
        adj = self.holdout_adjusted_score(c)
        self._posterior[i] *= max(0.01, adj * 2.0)
```

### Sau (V3)
```python
# Luôn chạy — holdout_adjusted_score trả về low default cho unevaluated
for i, c in enumerate(self._candidates):
    adj = self.holdout_adjusted_score(c)
    self._posterior[i] *= max(0.01, adj * 2.0)
```

Công thức `holdout_adjusted_score()` (giữ nguyên từ V2):
- **Evaluated**: `score = posterior × holdout_accuracy - |train - holdout| - complexity × 0.01`
- **Unevaluated**: `score = posterior × 0.3 × (1.0 - complexity × 0.01)`

Default score thấp hơn evaluated score trong mọi trường hợp ngoại trừ evaluated quá tệ (holdout accuracy rất thấp + overfit nặng).

### Ví dụ
| Candidate | Train | Holdout | Gap | Complexity | Score (evidenced) |
|-----------|-------|---------|-----|------------|-------------------|
| Keyword: `contains_word('bomb')` | 0.95 | 0.90 | 0.05 | 2 | 0.5×0.90−0.05−0.02 = **0.38** |
| Structural: `has_number()` | 0.88 | 0.87 | 0.01 | 3 | 0.3×0.87−0.01−0.03 = **0.22** |
| Unevaluated | — | — | — | 2 | 0.5×0.3×0.98 = **0.15** |

Keyword thắng nếu thực sự tốt hơn. Nhưng nếu keyword overfit (train=0.98, holdout=0.70):
- Keyword score = 0.6×0.70−0.28−0.02 = **0.12**
- Structural score = 0.4×0.82−0.03−0.03 = **0.27**
→ Structural thắng (generalization penalty tự động).

---

## 2. Predicate-family diversity — soft bonus thay vì hard cap

### Trước (V2) — Hard cap
```python
if kw_mass > 0.4:  # keyword > 40%
    kw_scale = 0.4 / kw_mass      # giảm keyword
    nk_boost = 0.6 / (1 - kw_mass) # tăng non-keyword
```

### Sau (V3) — Soft bonus (evidence-driven)
```python
# Cùng cơ chế với source diversity bonus (step 3)
type_counts = count_candidates_by_predicate_type()
max_tc = max(type_counts.values())
for each candidate:
    rep_ratio = type_counts[pt] / max_tc
    if rep_ratio < 0.5:  # underrepresented type
        bonus = 0.15 * 0.5 * (1.0 - rep_ratio)  # half of source bonus
        posterior *= (1.0 + bonus)
```

**Tác dụng**: Không ép keyword xuống 40%. Nếu keyword chiếm 80% candidates, bonus cho non-keyword chỉ ~0.0375 (3.75%). Nếu keyword thực sự tốt hơn dựa trên accuracy/holdout/complexity, chúng vẫn thắng.

---

## 3. Holdout evaluation — toàn bộ candidates

### Trước (V2)
```python
evaluate_on_holdout(holdout_prompts=[], top_k=5)
```

### Sau (V3)
```python
evaluate_on_holdout(holdout_prompts=[])  # evaluates ALL candidates
```

**Tác động**: Mỗi candidate trong VS đều có `holdout_accuracy`, `train_accuracy`, `generalization_gap` sau mỗi kỳ holdout evaluation. `update_belief()` step 1 luôn dùng data thật cho mọi candidate.

---

## 4. Convergence — evidence-driven

### Trước (V2)
```python
not_keyword_only = best.predicate_type != "keyword"
if real_holdout > 0.0 and best.accuracy >= threshold and not_keyword_only:
    converged = True
```

### Sau (V3)
```python
if real_holdout > 0.0 and best.accuracy >= threshold:
    converged = True  # evidence decides, not predicate type
```

**Tác động**: Nếu keyword candidate có holdout=0.92, train=0.95, gap=0.03, hệ thống hội tụ. Nếu keyword candidate có holdout=0.60 (overfit), hệ thống không hội tụ dù train accuracy cao.

---

## 5. Posterior init — chuẩn hóa (giữ từ V2)

```python
@staticmethod
def _initial_posterior(accuracy: float) -> float:
    return max(1e-6, 0.01 + accuracy * 0.3)
```

Áp dụng cho mọi entry point: `add_candidate()`, `absorb_candidates()`, `refine_candidate()`, `_normalise()`.

---

## 6. Công thức posterior đầy đủ (V3)

```
P_new(i) ∝ posterior_bayesian(i) × H(i) × F × D(i) × N(i) × Q(i) × M(i) × P(i)

Trong đó:
  posterior_bayesian = prior × exp(-error/0.5) × exp(-0.005×complexity) × specificity
  H(i) = max(0.01, holdout_adjusted_score(i) × 2.0)         — UNCONDITIONAL
  F    = max(posterior, 1e-4)                               — floor
  D(i) = source diversity bonus                             — source underrepresented
  N(i) = novelty bonus                                      — mới thêm (3 updates)
  Q(i) = source quota cap                                   — mỗi source ≤ 50%
  M(i) = synthesis min quota                                — synthesis ≥ 5%
  P(i) = predicate-type diversity bonus (SOFT, không hard)  — family underrepresented
```

---

## Test results

```
488 passed, 5 skipped, 0 failed
```

All synthesis, core, and agent tests pass. No regression.

---

## Tóm tắt so sánh V2 → V3

| Tính năng | V2 | V3 |
|-----------|----|----|
| Keyword cap | Hard 40% | Soft bonus (3.75% max) |
| Convergence check | Từ chối keyword | Evidence-driven |
| Holdout coverage | Top-5 | All candidates |
| Holdout adjustment | Chỉ khi `has_real_holdout` | Unconditional |
| Posterior init | Standardized (V2) | Standardized (giữ nguyên) |
| Công thức | Hard constraint + evidence | Pure evidence |
