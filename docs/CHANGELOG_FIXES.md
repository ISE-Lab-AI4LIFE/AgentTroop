# Changelog — Fixes & Integration

## 1. Wire RedTeamAgent vào pipeline

**File**: `llama3_1_8b/run_experiment.py`

- Thêm import `RedTeamAgent` (dòng 332)
- Tạo `RedTeamAgent` instance với config-driven params (dòng 359–377): victim, llm_client, episodic_memory, scientific_memory, population_size, refinement_rounds
- Truyền `red_team_agent=red_team` vào `Orchestrator()` (dòng 451)

**Trước**: `self.red_team = None` → refinement code ở orchestrator.py (lines 398, 703) không bao giờ chạy.

**Sau**: Mọi intervention Phase 3+ trong `run()` và `async_run()` đều qua `RedTeamAgent.maybe_refine_intervention()`.

---

## 2. Fix Intervention.final_prompt setter

**File**: `core/intervention.py`

- `final_prompt` là `@property` read-only (gọi `apply()`)
- `RedTeamAgent.maybe_refine_intervention()` cố gắng set `intervention.final_prompt = refined` → crash
- **Fix**: Thêm `@final_prompt.setter` (dòng 23–24)

---

## 3. Xoá transform không tồn tại khỏi seed_strategy T3

**File**: `llama3_1_8b/seed_strategy.py`

**Trước** (4/5 transforms không tồn tại trong registry):
```
leet_speak ❌  unicode_obfuscate ❌  insert_typos ✅
reverse_text ❌  pig_latin ❌
```

**Sau** (5/5 tồn tại):
```
insert_typos ✅  add_zero_width_chars ✅  html_encode ✅
random_case ✅  url_encode ✅
```

---

## 4. Fix run_exp.sh config path

**File**: `llama3_1_8b/run_exp.sh`

`CONFIG_FILE="$SCRIPT_DIR/config.yaml"` → `CONFIG_FILE="$PROJECT_DIR/configs/experiment_config.yaml"`

File `config.yaml` trong `llama3_1_8b/` không tồn tại → script luôn fail nếu không truyền `--config`.

---

## 5. Thêm red_team section vào YAML configs

**Files**: `configs/*.yaml` (4 files)

Thêm section:
```yaml
red_team:
  population_size: 10
  refinement_rounds: 3
  mutation_rate: 0.3
  crossover_rate: 0.5
  max_chain_depth: 3
  enable_self_eval: false
```

---

## 6. Dọn config disabled lists

Xoá tham chiếu đến transforms đã xoá khỏi core/primitive.py:
- `rot13`, `base64`, `morse_code`, `binary_encode`, `hex_encode`
- `caesar_cipher`, `atbash_cipher`, `vigenere_cipher`, `rail_fence_cipher`
- `leet_speak`, `reverse_text`, `pig_latin`, `unicode_obfuscate`

**Files affected**: `configs/config_quick_test.yaml`, `configs/config_test_fix.yaml`

---

## Files changed (8 files)

| File | Type | Summary |
|------|------|---------|
| `llama3_1_8b/run_experiment.py` | Fix | Wire RedTeamAgent |
| `core/intervention.py` | Fix | Add setter for final_prompt |
| `llama3_1_8b/seed_strategy.py` | Fix | Replace broken T3 transforms |
| `llama3_1_8b/run_exp.sh` | Fix | Config path |
| `configs/experiment_config.yaml` | Config | Add red_team section |
| `configs/config_quick_test.yaml` | Config | Add red_team + clean disabled |
| `configs/config_validate.yaml` | Config | Add red_team section |
| `configs/config_test_fix.yaml` | Config | Add red_team + clean disabled |

## prompt.csv verification

- **473** entries, **470** unique harmful prompts
- 3 tasks: text-to-code generation, code completion, code translation
- Default split: 30 harmful + 20 benign → ~750 seed episodes
- `--full`: 473 harmful + 80 benign → ~8,300 seed episodes
