# HARMONY-X — Danh sách thành phần & Tiến độ

## 6 Tầng bộ nhớ (Hierarchical Memory)

| Tầng | Tên | Database | File thực thi | Tests | Trạng thái |
|------|-----|----------|---------------|-------|------------|
| L1 | Episodic Memory | SQLite | `knowledge/episodic/episodic.py` | 63 tests | ✅ Hoàn thành |
| L2 | Session Memory | Redis (cache) | `knowledge/session_memory.py` | 15 tests | ✅ Hoàn thành |
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
| Orchestrator | `orchestration/__init__.py` | ✅ Hoàn thành (15 tests) |
| Cognitive Agent | `agents/cognitive.py` | ✅ Hoàn thành (80 tests) |
| Strategist Agent | `agents/strategist.py` | ✅ Hoàn thành (62 tests) |
| Researcher Agent | `agents/researcher.py` | ✅ Hoàn thành (40 tests) |

### ✅ Orchestrator (`orchestration/__init__.py`)
- Điều phối vòng lặp 6 phase: phát hiện bất thường → sinh giả thuyết → thiết kế can thiệp → thực thi → tổng hợp → kiểm chứng
- State machine (`OrchestratorPhase` enum) với checkpoint support
- Quản lý hypothesis pairs cho Strategist, pipeline end‑to‑end cho Researcher
- Kiểm tra hội tụ: dừng sớm khi có program verified + không còn anomalies
- 15 tests (mock tất cả dependencies)

### ✅ Cognitive Agent (`agents/cognitive.py`)
- Phát hiện bất thường từ Episodic Memory (so sánh outcome theo nhóm prompt gốc)
- Sinh giả thuyết cấu trúc bằng LLM (GEMMA qua LLMClient) với fallback mặc định
- Ước lượng độ tin cậy (Laplace smoothing) dựa trên số lượng anomalies hỗ trợ
- Data classes: `Anomaly`, `Hypothesis` với auto UUID, serialization to_dict
- Tích hợp GrammarExporter để lấy primitive catalog
- 80 tests (mock LLM, memory, grammar exporter)

### ✅ Strategist Agent (`agents/strategist.py`)
- Thiết kế can thiệp tối ưu để phân biệt giả thuyết cạnh tranh
- Dùng `Intervention` (base prompt + transforms) từ `core/intervention.py`
- Hỗ trợ cả text-based hypotheses (cognitive, với keyword extraction) và Program AST (executor)
- Hybrid mode: heuristic local search + LLM-guided transform suggestion
- Transform chain support: kết hợp nhiều transforms (cấu hình `max_chain_depth`)
- Tự động lấy base prompts từ Episodic Memory (dựa trên campaign_id)
- Xử lý non‑deterministic classifier với `num_trials` + majority vote
- Clamp `intervention_budget` trong [1, 1000] với cảnh báo
- `max_candidates_heuristic` / `max_candidates_llm` riêng biệt
- Logging metrics: candidates count, avg Δ
- Ghi kết quả can thiệp vào Episodic Memory
- Primitive cache với `refresh_primitive_cache()` + OntologyMemory hook
- 62 tests (mock executor, memory, grammar exporter, LLM client)

### ✅ Researcher Agent (`agents/researcher.py`)
- **`agents/researcher.py`** — `ResearcherAgent` class
- **Pipeline 5 bước:** synthesis → verification → store program → abstract theory → store theory
- **`run_reverse_engineering_pipeline()`** — entry point end‑to‑end, trả về dict kết quả
- Không dùng LLM, chỉ dùng `synthesis` module + `knowledge` memory layers
- Constructor tự động tạo `CVC5Synthesizer` (max_depth=3, beam_width=200) và `ProgramExecutor`
- Hỗ trợ `allow_error_rate`, `experiment_id`, `exclude_prompts`, `verbose`
- 40 tests (mock tất cả dependencies)

---

## Core Modules (Foundation)

| Module | File | Tests | Trạng thái |
|--------|------|-------|------------|
| Program AST | `core/program.py` | — | ✅ Hoàn thành |
| Primitive types | `core/primitive.py` | 200 tests (92 primitives) | ✅ Hoàn thành |
| Type definitions | `core/types.py` | — | ✅ Hoàn thành |
| Semantic Memory (vector) | `knowledge/semantic_memory.py` | 53 tests | ✅ Hoàn thành |
| Researcher Agent | `agents/researcher.py` | 17 tests | ✅ Hoàn thành |
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
- **92 built-in primitives**: 27 predicates + 38 transforms + 27 classifiers
- All primitives use `@dataclass` with `__post_init__` for metadata
- Toxicity/Sentiment classifiers use TextBlob (nếu có) + keyword heuristics
- 200 unit tests covering all 92 primitives + registry completeness

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
| Redis | Proposal queue + Session cache | ✅ Đã tích hợp (Knowledge Manager) |
| CVC5 | SMT solver cho program synthesis | 🔶 Optional (synthesis hoạt động bằng enumeration) |

---

## Causal Graph

| Thành phần | Trạng thái |
|------------|------------|
| Causal Defense Graph (Neo4j) | ✅ Đã xây dựng (`knowledge/causal_graph.py`, ~30 tests) |

---

## Giai đoạn phát triển (Implementation Plan)

| Giai đoạn | Trọng tâm | Trạng thái |
|-----------|-----------|------------|
| 1 | Hạ tầng cơ bản: Neo4j, Episodic/Session memory, Knowledge Manager | ✅ Đã hoàn thành (Redis + Session Memory + Knowledge Manager + Causal Graph) |
| 2 | Cognitive Agent: phát hiện bất thường, sinh giả thuyết | ✅ Hoàn thành (`agents/cognitive.py`, 80 tests) |
| 3 | Strategist Agent: thiết kế can thiệp heuristic | ✅ Hoàn thành (`agents/strategist.py`, 30 tests) |
| 4 | Researcher Agent: CVC5, grammar, program synthesis, pipeline end‑to‑end | ✅ Hoàn thành (`agents/researcher.py`, 40 tests) |
| 5 | Orchestrator: vòng lặp 6 phase, checkpoint, proposal queue | ✅ Hoàn thành (`orchestration/__init__.py`, 15 tests) |
| 6 | Scientific Memory: trích xuất lý thuyết, chuyển giao | ✅ Hoàn thành (phần lưu trữ), ⬜ Thiếu phần trích xuất |

---

## Thống kê hiện tại

| Hạng mục | Số lượng |
|----------|----------|
| Tổng số memory layers đã xây dựng | 6 / 6 (L1-L6 + Causal Graph) |
| Tổng số tests (knowledge layer) | 207 + 30 (causal) + 25 (manager) + 15 (session) = ~277 |
| Tổng số tests (synthesis module) | 80 |
| Tổng số tests (primitive module) | 200 |
| Tổng số tests (researcher agent) | 40 |
| Tổng số tests (cognitive agent) | 80 |
| Tổng số tests (strategist agent) | 62 |
| Tổng số tests (orchestrator) | 15 |
| **Tổng số tests (toàn bộ)** | **~739** (chưa kể causal + manager) |
| Tổng số primitives | **92** (27 predicate + 38 transform + 27 classifier) |
| Tổng số files Python (knowledge + synthesis + agents + core) | 12 modules |
| Neo4j labels đã định nghĩa | `Theory`, `DefenseProgram`, `ASTNode`, `PrimitiveNode`, `OntologyPrimitive`, `CausalNode` |
| Neo4j constraints | Unique constraints trên Theory(id,version), DefenseProgram(id,version), ASTNode(node_id), PrimitiveNode(name), OntologyPrimitive(name), CausalNode(id) |
