# Synthesis & Verification Module

> **Researcher Agent**: Tổng hợp chương trình từ ví dụ và kiểm chứng bằng can thiệp.  
> Cập nhật: 10 điểm cải tiến synthesis core, real classifier, beam width, disk cache, free thresholds, verbose, transform chains.  
> Cập nhật lần cuối: 6/6/2026.

## Tổng quan

Module `synthesis/` cung cấp pipeline cho Researcher Agent:
1. **GrammarExporter** — liệt kê không gian chương trình
2. **CVC5Synthesizer** — tổng hợp chương trình (enumeration + CVC5 fallback)
3. **ProgramVerifier** — kiểm chứng trên can thiệp thực tế

```
Examples → GrammarExporter → enumerated programs (sorted by complexity, fitness-scored)
              ↓
    CVC5Synthesizer.synthesize_with_stats()
       ├── _try_enumeration()     ← beam search (beam_width=200) + memory/disk cache
       │   └── _fitness_score()   ← sắp xếp programs theo accuracy → beam
       │   └── _matches_all()     ← 10 trials / example, max_errors từ allow_error_rate
       └── _try_cvc5()            ← SMT-LIB model parser (use_free_thresholds)
              ↓
         Program + SynthesisStats
              ↓
    ProgramVerifier.verify(exclude_prompts=..., verbose=False)
              ↓
         VerificationReport
              ↓
    abstract_theory() → Theory → Scientific Memory
    store_verified_program() → DefenseProgramRecord → DefenseProgramStore
```

---

## GrammarExporter

**File:** `synthesis/grammar_exporter.py`

### Chức năng

| Method | Mô tả |
|--------|-------|
| `get_primitives()` | Lấy catalog từ `PrimitiveRegistry` hoặc `OntologyMemory` |
| `get_parameterized_primitives(examples)` | Extract keywords từ REFUSE examples, tạo parameterized predicates |
| `enumerate_conditions(max_depth, examples, max_programs)` | Sinh condition nodes (depth 1→max_depth) |
| `enumerate_programs(max_depth, examples, max_programs)` | Wrap conditions → `Program`, sorted by `complexity()` |
| `export_to_smtlib(examples, output_file, max_depth)` | Xuất SMT-LIB 2.6 (declare-fun cho predicates/classifiers/transforms) |

### Grammar operators

| Depth | Operators | Ví dụ |
|-------|-----------|-------|
| 1 | `PredicateNode`, `ThresholdNode` | `ContainsWord("bomb")`, `toxicity_score > 0.5` |
| 2+ | `NotNode`, `AndNode`, `OrNode`, `ApplyTransformNode` | `NOT contains_word("hello")`, `rot13(contains_word("bomb"))` |

### Keyword extraction

Từ các example `outcome == 1`, extract chữ cái ≥ 3 ký tự, tạo `ContainsWordPredicate` tương ứng.

### Beam search

`max_programs` giới hạn số programs mỗi depth. Mặc định 0 = unlimited (nhưng internal limit 2000).

### Complexity sorting

`enumerate_programs` trả về programs sorted by `p.complexity()` (node count) — Occam priority.

### Free variable thresholds (SMT-LIB)

`export_to_smtlib(use_free_thresholds=True)` thay vì hardcode `(> (toxicity_score x) 0.5)`, khai báo free variable:
```
(declare-fun threshold_toxicity_score () Real)
(assert (and (<= 0 threshold_toxicity_score) (<= threshold_toxicity_score 1)))
```
Model CVC5 chọn threshold tối ưu, `_extract_thresholds_from_model()` đọc và clamp [0, 1].

### Guard chống combinatorial explosion

`_enumerate_conditions` có `limit = max_programs if max_programs > 0 else 2000` — không bao giờ sinh quá 2000 nodes absolute.

---

## CVC5Synthesizer

**File:** `synthesis/cvc5_synthesizer.py`

### Constructors parameters

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `cvc5_path` | `"cvc5"` | Path đến CVC5 binary |
| `timeout` | `30` | Timeout cho CVC5 (giây) |
| `max_depth` | `3` | Depth tối đa ban đầu |
| `allow_error_rate` | `0.0` | Noise tolerance (0..1) |
| `beam_width` | `200` | Beam search limit; 0 = unlimited (fallback 500) |
| `use_cache` | `True` | Cache `_matches_all` results in memory |
| `cache_path` | `None` | Disk cache path (pickle), tự động load/save |

### Methods

| Method | Return | Mô tả |
|--------|--------|-------|
| `synthesize(examples, registry, ontology)` | `Optional[Program]` | Entry point (convenience) |
| `synthesize_with_stats(examples, registry, ontology)` | `Tuple[Optional[Program], SynthesisStats]` | Entry point + stats |
| `_try_enumeration(examples, exporter, max_errors, stats, depth)` | `Optional[Program]` | Enumeration với beam search + memory/disk cache + fitness scoring |
| `_try_cvc5(examples, exporter, max_errors)` | `Optional[Program]` | CVC5 subprocess; syntax error → fallback |
| `_parse_cvc5_output(output, exporter, examples, max_errors)` | `Optional[Program]` | Parse SMT-LIB model (sat/unsat/model) |
| `_parse_smt_model(model_text)` | `Dict[str, Any]` | Extract function defs bằng balanced-parentheses parser |
| `_extract_thresholds_from_model(model_predicates)` | `Dict[str, float]` | Đọc threshold từ model CVC5, clamp [0, 1] |
| `_build_from_model(model_predicates, exporter, thresholds)` | `Optional[Program]` | Build Program từ model predicates + thresholds |
| `_fitness_score(program, examples, executor)` | `float` | Accuracy (correct / len(examples)) |
| `synthesize_from_episodes(episodic_memory, campaign_id, experiment_id, ...)` | `Tuple[Optional[Program], SynthesisStats]` | Đọc Episodic Memory → examples |
| `abstract_theory(program, model_family, conditions, provenance)` | `Theory` | Trích xuất Theory pattern từ Program (AND/OR/NOT/nested) |
| `store_verified_program(defense_store, program, name, confidence, provenance)` | `str` | Lưu Program vào DefenseProgramStore |

### SynthesisStats

```python
@dataclass
class SynthesisStats:
    duration_ms: float           # Thời gian chạy (ms)
    depth_used: int              # Depth cuối cùng đã thử
    programs_tried: int          # Số programs đã kiểm tra
    programs_skipped_cache: int  # Số programs skip từ cache
    cvc5_used: bool              # Đã dùng CVC5?
    enumeration_found: bool      # Enumeration tìm thấy lời giải?
    allow_error_rate: float      # Noise tolerance
    max_errors: int              # Số lỗi tối đa cho phép
    errors_actual: int           # Số lỗi thực tế của program
    beam_width: int              # Beam width đang dùng
```

### Non-deterministic handling

`_matches_all` chạy **10 trials** mỗi example, yêu cầu **tất cả 10 phải match** (không như cũ: "ít nhất 1 phải match"). Loại bỏ gần như hoàn toàn nhiễu từ classifier.

`ToxicityScoreClassifier` và `SentimentClassifier` giờ dùng TextBlob (nếu có) + keyword heuristic — **deterministic**, thay `random.random()` cũ.

### Auto depth increase + combinatorial guard

Khi `_try_enumeration` thất bại ở depth hiện tại, tự động tăng depth lên 1 và thử lại, tối đa `max_depth + 2`.

Nếu `beam_width=0` và `depth > max_depth + 1`, tự động skip (tránh sinh 1.6M nodes ở depth 4+).

### Beam width

`beam_width` (default 200) giới hạn số programs kiểm tra mỗi depth. Cơ chế:
1. `enumerate_conditions` giới hạn tuyệt đối 2000 nodes qua `limit`
2. `_try_enumeration` lấy `max_p = beam_width if beam_width > 0 else 500`
3. Nếu `beam_width > 0`, tính `_fitness_score` cho `beam_width * 2` programs, chọn top `beam_width`
4. Programs được sắp xếp theo `complexity()` trước fitness scoring

### Disk cache

`cache_path` parameter cho phép persist `_matches_all` results qua pickle:
```python
synth = CVC5Synthesizer(cache_path="/tmp/synth_cache.pkl")
```
Auto load khi khởi tạo, auto save sau mỗi depth. Cache key = `hash(canonical_form, tuple(examples))`.

### Free variable thresholds

`export_to_smtlib(use_free_thresholds=True)` khai báo free Real variable cho mỗi classifier threshold thay vì hardcode 0.5. CVC5 tự chọn threshold tối ưu. `_extract_thresholds_from_model` đọc và clamp về [0, 1].

### Fitness scoring

`_fitness_score()` = `correct / len(examples)`. Dùng trong beam search để ưu tiên programs có accuracy cao nhất.

### Verbose logging (verifier)

`ProgramVerifier.verify(verbose=True)` log từng intervention prompt + expected + actual outcome.

---

## ProgramVerifier

**File:** `synthesis/verifier.py`

### Methods

| Method | Return | Mô tả |
|--------|--------|-------|
| `verify(program, num_test_interventions, accuracy_threshold, exclude_prompts, verbose)` | `VerificationReport` | Kiểm chứng; `verbose=True` log từng intervention |
| `_default_intervention_generator(victim, n)` | `List[str]` | Sinh prompts: 20 base + 4 transforms + role-prefix variants |
| `_generate_suggestions(failures, program)` | `List[str]` | FP/FN analysis |

### Intervention generation

Mỗi base prompt sinh:
1. Original prompt
2. `Rot13Transform`, `Base64DecodeTransform`, `ToLowercaseTransform`, `RemovePunctuationTransform`
3. Role-prefix variants: *"As a researcher, ..."*, *"In an academic context, ..."*, etc.
4. Synonym variants (khi `content_words >= 3`)

### exclude_prompts

```python
report = verifier.verify(
    program,
    num_test_interventions=20,
    accuracy_threshold=0.9,
    exclude_prompts={"bomb", "kill"},  # training examples
)
```

---

## Test Coverage

| File | Tests | Coverage |
|------|-------|----------|
| `tests/synthesis/test_grammar_exporter.py` | 18 | Core enumeration + SMT-LIB |
| `tests/synthesis/test_cvc5_synthesizer.py` | 16 | Core synthesis |
| `tests/synthesis/test_cvc5_synthesizer_extended.py` | 32 | Free thresholds, transform chains, real classifier, beam width, verbose, disk cache, abstract theory, episodic integration, fitness scoring |
| `tests/synthesis/test_verifier.py` | 18 | Verification + suggestions + verbose |
| **TOTAL** | **84 (1 skipped if no CVC5)** | ✅ All pass |

---

## Kiến trúc & Design Decisions

### Enumeration-first strategy
CVC5 chưa cài đặt. Enumeration với keyword extraction + complexity sorting hoạt động tốt depth ≤ 3, beam_width=200.

### Balanced-parentheses SMT parser
Thay vì regex non-greedy dễ sai, `_parse_smt_model` dùng depth counter để parse function bodies chính xác.

### Noise tolerance
`allow_error_rate` cho phép synthesis thư giãn khi dữ liệu có nhiễu. `_matches_all` kiểm tra bằng counter errors thay vì binary pass/fail.

### Occam complexity
`enumerate_programs` sort theo node count, synthesis trả về program đầu tiên thỏa mãn (đơn giản nhất).

### Beam width
Default 200. Kết hợp fitness scoring: tính accuracy cho `beam_width × 2` programs, chọn top `beam_width`. Tránh combinatorial explosion ở depth ≥ 3.

### Disk cache
Persist `_matches_all` results qua pickle. Giúp tái sử dụng kết quả giữa các lần chạy (synthesis, verify).

### Free variable thresholds
Dùng `NRA` logic + free Real variables cho classifier thresholds thay vì hardcode. CVC5 chọn threshold tối ưu.

### Episodic → Synthesis pipeline
`synthesize_from_episodes` cho phép Researcher Agent đọc trực tiếp từ SQLite, không cần transform thủ công.

### Verifier deterministic + edge variants
`_default_intervention_generator` deterministic (không shuffle), thêm RemovePunctuationTransform + role prefixes + synonym variants. `verbose=True` log chi tiết.

### Real deterministic classifiers
`ToxicityScoreClassifier` và `SentimentClassifier` dùng TextBlob + keyword heuristic — deterministic, không còn `random.random()`.

---

## Dependencies

| Package | Bắt buộc? | Ghi chú |
|---------|-----------|---------|
| Python 3.10+ | ✅ | Type hints, dataclasses |
| `core` (internal) | ✅ | Program, Primitive, Executor |
| `knowledge` (internal) | Optional | Episodic Memory, Scientific Memory, Defense Store |
| CVC5 binary | ❌ Optional | Fallback khi enumeration thất bại |

---

## Future Work (Post-12-improvements)

1. **Cài đặt CVC5** — test `_parse_cvc5_output` với binary thật (đã có test skeleton, skip nếu không có binary)
2. **Victim-adaptive generation** — sinh can thiệp dựa trên behavior profile
3. **Causal verification** — kiểm chứng causal effect thay vì accuracy
4. **Incremental synthesis** — dùng verified program làm seed để tổng hợp program phức tạp hơn
5. **LLM-guided enumeration** — dùng LLM để ưu tiên programs triển vọng
6. **Integration tests với Neo4j thật** — DefenseStore + ScientificMemory (đã có test, skip khi không có Neo4j)
7. **Documentation** — `docs/synthesis_usage.md` cho người dùng cuối
