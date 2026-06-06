# HARMONY-X — Danh sách thành phần & Tiến độ

## 6 Tầng bộ nhớ (Hierarchical Memory)

| Tầng | Tên | Database | File thực thi | Tests | Trạng thái |
|------|-----|----------|---------------|-------|------------|
| L1 | Episodic Memory | SQLite | `knowledge/episodic/episodic.py` | 63 tests | ✅ Hoàn thành |
| L2 | Session Memory | Redis (cache) | — | — | ⬜ Chưa xây dựng |
| L3 | Strategy Memory | Neo4j / file | — | — | ⬜ Chưa xây dựng |
| L4 | Defense Program Store | Neo4j | `knowledge/defense_store.py` | 25 tests | ✅ Hoàn thành |
| L5 | Ontology Memory | Neo4j | `knowledge/ontology_memory.py` | 19 tests | ✅ Hoàn thành |
| L6 | Scientific Memory | Neo4j | `knowledge/scientific_memory.py` | 47 tests | ✅ Hoàn thành |

### ✅ L1 — Episodic Memory
- Lưu raw intervention data: prompt, response, outcome, timestamp, intervention_id
- SQLite database, hỗ trợ checksum, soft-delete, provenance, annotation, transformation trace
- 63 tests, hoạt động độc lập

### ✅ L4 — Defense Program Store (`knowledge/defense_store.py`)
- Lưu các chương trình $\Pi$ đã được xác nhận dưới dạng AST tree trong Neo4j
- Chuyển đổi 2 chiều giữa `core.program.Program` (AST) ↔ Neo4j graph nodes/edges
- Labels: `DefenseProgram` (metadata), `ASTNode` (cấu trúc cây), `PrimitiveNode` (primitive reference)
- Relationships: `HAS_ROOT`, `LEFT`, `RIGHT`, `CHILD`, `CONDITION`, `INNER`, `PRIMITIVE`, `NEXT_VERSION`
- Versioning: mỗi lần save tạo version mới, get trả về latest; hỗ trợ `get(pid, version=N)`
- CRUD: save, get, delete; find_by_confidence, find_by_primitive, list_program_ids
- update_confidence tạo version mới
- Export/import JSON (có/sans lịch sử)
- 25 tests (6 unit + 19 Neo4j)

### ✅ L5 — Ontology Memory (`knowledge/ontology_memory.py`)
- Catalog các primitive generic (predicate, transform, classifier, policy) trong Neo4j
- Label `OntologyPrimitive` với unique constraint trên `name`
- Kiểm tra loại primitive hợp lệ: predicate, transform, classifier, policy
- CRUD: save_primitive, get_primitive, delete_primitive, list_primitives, find_primitives
- `sync_to_registry()`: đồng bộ từ `core.primitive.default_registry` (8 built-in primitives)
- Export/import JSON
- 19 tests (5 unit + 14 Neo4j)

### ✅ L6 — Scientific Memory (`knowledge/scientific_memory.py`)
- Lưu abstract safety theories dạng pattern + conditions + confidence + provenance
- Neo4j với composite unique constraint `(id, version)`
- Versioning, dynamic condition properties, pattern CONTAINS search
- Auto-compact background thread, export/import
- 47 tests

---

## Agent Layer

| Agent | File | Trạng thái |
|-------|------|------------|
| Orchestrator | — | ⬜ Chưa xây dựng |
| Cognitive Agent | — | ⬜ Chưa xây dựng |
| Strategist Agent | — | ⬜ Chưa xây dựng |
| Researcher Agent | — | ⬜ Chưa xây dựng |

### ⬜ Orchestrator
- Điều phối vòng lặp 6 phase: phát hiện bất thường → sinh giả thuyết → thiết kế can thiệp → thực thi → tổng hợp → kiểm chứng
- Quản lý proposal queue, checkpoint, state machine

### ⬜ Cognitive Agent
- Phát hiện bất thường từ Episodic Memory, sinh giả thuyết cấu trúc bằng LLM

### ⬜ Strategist Agent
- Thiết kế can thiệp tối ưu để phân biệt giả thuyết

### ⬜ Researcher Agent
- Tổng hợp chương trình bằng CVC5, kiểm chứng bằng can thiệp
- Trích xuất abstract theory → Scientific Memory
- Ghi vào Defense Program Store, Ontology Memory

---

## Core Modules (Foundation)

| Module | File | Tests | Trạng thái |
|--------|------|-------|------------|
| Program AST | `core/program.py` | — | ✅ Hoàn thành |
| Primitive types | `core/primitive.py` | — | ✅ Hoàn thành |
| Type definitions | `core/types.py` | — | ✅ Hoàn thành |
| Semantic Memory (vector) | `knowledge/semantic_memory.py` | 53 tests | ✅ Hoàn thành |
| Synthesis — Grammar Exporter | `synthesis/grammar_exporter.py` | 18 tests | ✅ Hoàn thành |
| Synthesis — CVC5 Synthesizer | `synthesis/cvc5_synthesizer.py` | 48 tests (16 base + 32 ext) | ✅ Hoàn thành |
| Synthesis — Program Verifier | `synthesis/verifier.py` | 18 tests (14 base + 4 ext) | ✅ Hoàn thành |

### ✅ Core Program (`core/program.py`)
- Dataclasses: `Program`, `ProgramFragment`, `Policy`, `PolicyTemplate`
- Node types: `PredicateNode`, `TransformNode`, `ClassifierNode`, `ThresholdNode`, `ApplyTransformNode`, `AndNode`, `OrNode`, `NotNode`, `IfThenElseNode`
- Serialization: `to_dict()` / `from_dict()` cho mỗi node

### ✅ Core Primitive (`core/primitive.py`)
- `Primitive` base class + `Predicate`, `Transform`, `Classifier`
- `PrimitiveRegistry` singleton
- Built-in primitives: `ContainsWordPredicate`, `LengthGtPredicate`, `MatchesRegexPredicate`, `Rot13Transform`, `Base64DecodeTransform`, `ToLowercaseTransform`, `RemovePunctuationTransform`, `ToxicityScoreClassifier`, `SentimentClassifier`
- `ToxicityScoreClassifier` dùng TextBlob (nếu có) + keyword heuristic (`bomb`, `kill`, …), deterministic
- `SentimentClassifier` dùng TextBlob (nếu có) + keyword heuristic (`bad`, `terrible`, … → `good`, `great`, …)

### ✅ Semantic Memory (`knowledge/semantic_memory.py`)
- Vector store with FAISS index + SQLite metadata
- Hybrid search (vector similarity + keyword), episode integration
- 53 tests

---

### ✅ Synthesis Module (`synthesis/`)
- **GrammarExporter** (`synthesis/grammar_exporter.py`): Enumerate program space, extract keywords từ REFUSE examples, export SMT-LIB 2.6
- **CVC5Synthesizer** (`synthesis/cvc5_synthesizer.py`): Synthesis bằng enumeration (primary) + CVC5 subprocess (optional fallback), `build_simple_program()` utility
- **ProgramVerifier** (`synthesis/verifier.py`): Verify program bằng victim interventions, sinh `VerificationReport` với accuracy/failures/suggestions
- **`synthesize_with_stats()`** — trả về `SynthesisStats` (duration, depth, programs_tried, cache_hits, v.v.)
- **`allow_error_rate`** — noise tolerance: cho phép sai số `N * rate` mẫu
- **`beam_width`** (default 200) — beam search: giới hạn số programs mỗi depth; 0 = unlimited (fallback 500)
- **Auto depth increase** — tự động tăng depth khi không tìm thấy lời giải (tối đa max_depth+2), skip nếu beam_width=0 và depth > max_depth+1
- **Guard chống combinatorial explosion**: `_enumerate_conditions` giới hạn 2000 nodes absolute; beam_width=200 mặc định
- **Disk cache** — `cache_path` parameter; load/save `_matches_all` results qua pickle; `_save_cache()` tự động persist
- **Memory cache** — cache kết quả `_matches_all` theo canonical hash (`_compute_hash`)
- **Complexity sorting** — enumeration trả về program đơn giản nhất (node count), kết hợp fitness scoring
- **`_parse_cvc5_output`** — balanced-parentheses SMT-LIB model parser
- **`_parse_smt_model`** — trích xuất function definitions từ S‑expression
- **`_extract_thresholds_from_model`** — đọc threshold từ model CVC5, clamp [0, 1]
- **`export_to_smtlib(use_free_thresholds=True)`** — dùng free variable thresholds (NRA logic) thay vì hardcode 0.5
- **Fitness scoring** — `_fitness_score()` tính accuracy trên examples, dùng trong beam search
- **`synthesize_from_episodes()`** — đọc Episodic Memory, tự động tạo examples
- **`abstract_theory()`** — trích xuất `Theory` pattern từ program (AND/OR/NOT/Nested), lưu vào Scientific Memory
- **`store_verified_program()`** — lưu program đã verify vào Defense Program Store
- **Edge intervention generation** — verifier sinh thêm role‑prefix variants, RemovePunctuationTransform, synonym variants
- **`exclude_prompts`** — loại trừ training prompts khỏi test set
- **Verbose logging** — verifier nhận `verbose=True`, log từng intervention + outcome
- Non-deterministic primitive handling: mỗi example chạy 10 lần trong `_matches_all` (tất cả phải match)
- **Real ToxicityScoreClassifier** — TextBlob (sentiment polarity + keyword boost) hoặc keyword heuristic (`bomb`, `kill`, … × 0.25)
- **SentimentClassifier** — TextBlob (polarity→score) hoặc keyword heuristic (positive/negative word ratio)
- 80 tests (16 base + 32 extended + 18 verifier + 15 grammar_exporter), tất cả pass (1 skipped nếu không có CVC5 binary)

---

## Infrastructure

| Thành phần | Mục đích | Trạng thái |
|------------|----------|------------|
| Neo4j | Graph database cho L4, L5, L6, Causal Graph | ✅ Đang chạy |
| Chroma | Vector embedding | ✅ Có (thay thế bằng FAISS trong Semantic Memory) |
| Redis | Proposal queue + Session cache | ⬜ Chưa tích hợp |
| CVC5 | SMT solver cho program synthesis | 🔶 Optional (synthesis hoạt động bằng enumeration) |

---

## Causal Graph

| Thành phần | Trạng thái |
|------------|------------|
| Causal Defense Graph (Neo4j) | ⬜ Chưa xây dựng |

---

## Giai đoạn phát triển (Implementation Plan)

| Giai đoạn | Trọng tâm | Trạng thái |
|-----------|-----------|------------|
| 1 | Hạ tầng cơ bản: Neo4j, Episodic/Session memory, Knowledge Manager | 🔶 Đang thực hiện (thiếu Session, Redis) |
| 2 | Cognitive Agent: phát hiện bất thường, sinh giả thuyết | ⬜ Chưa bắt đầu |
| 3 | Strategist Agent: thiết kế can thiệp heuristic | ⬜ Chưa bắt đầu |
| 4 | Researcher Agent: CVC5, grammar, program synthesis | 🔶 Đang thực hiện (synthesis module done, thiếu integration với Agent layer) |
| 5 | Orchestrator: vòng lặp 6 phase, checkpoint, proposal queue | ⬜ Chưa bắt đầu |
| 6 | Scientific Memory: trích xuất lý thuyết, chuyển giao | ✅ Hoàn thành (phần lưu trữ), ⬜ Thiếu phần trích xuất |

---

## Thống kê hiện tại

| Hạng mục | Số lượng |
|----------|----------|
| Tổng số memory layers đã xây dựng | 4 / 6 (L1, L4, L5, L6) |
| Tổng số tests (knowledge layer) | 207 |
| Tổng số tests (synthesis module) | 80 |
| **Tổng số tests (toàn bộ)** | **402** |
| Tổng số files Python (knowledge + synthesis) | 8 modules |
| Neo4j labels đã định nghĩa | `Theory`, `DefenseProgram`, `ASTNode`, `PrimitiveNode`, `OntologyPrimitive` |
| Neo4j constraints | Unique constraints trên Theory(id,version), DefenseProgram(id,version), ASTNode(node_id), PrimitiveNode(name), OntologyPrimitive(name) |
