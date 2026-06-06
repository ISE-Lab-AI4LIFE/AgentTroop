# CHECKPOINT — HARMONY-X Knowledge Layer

> **Tổng quan toàn bộ hệ thống tại thời điểm hiện tại.**  
> Dùng làm mốc để cập nhật các phiên sau.
>
> - **Mốc 1 (7/6):** Episodic Memory (L1) hoàn chỉnh + Scientific Memory (L6) khởi tạo
> - **Mốc 2 (8/6):** Semantic Memory (L5) production-ready + Auto-Compact
> - **Mốc 3 (8/6):** Synthesis module — GrammarExporter + CVC5Synthesizer + ProgramVerifier
> - **Mốc 4 (6/6):** Synthesis hardening — beam width, disk cache, free thresholds, real classifiers, guard explosion
> - **Mốc 5 (6/6):** Researcher Agent — pipeline end‑to‑end, 5 bước, 17 tests
> - **Mốc 6 (6/6):** 92 primitives (27+38+27) — 200 tests, primitive_catalog.md, 4 bug fixes
> - **Mốc 7 (6/6):** Cognitive Agent — anomaly detection, LLM hypothesis generation, 56 tests
> - **Mốc 8 (6/6):** Strategist Agent + Orchestrator — intervention design, 6-phase loop, hybrid heuristic/LLM, 796 total tests

---

## Mốc 1 — 7/6/2026

### 1.1. Episodic Memory (L1)

**File:** `knowledge/episodic/episodic.py` — SQLite backend, 63 tests.

**Tính năng đã hoàn thiện:**
- **CRUD:** `save_episode` (upsert), `get_episode`, `delete_episode` (soft), `hard_delete_episode`, `restore_episode`, `episode_exists`
- **Query:** theo campaign, experiment, session, timerange, parent episode — tất cả qua `EpisodeFilter` composite
- **Indexes:** strategy_name, agent_name, hypothesis_id, campaign_id, experiment_id, victim_name, outcome, session_id, parent_episode_id, created_at
- **Evidence:** `EpisodeEvidence` với hypothesis FK, `get_evidence_for_hypothesis`, `get_evidence_for_episode`
- **Annotations:** key-value metadata linh hoạt
- **Checksum:** SHA-256 tự động, `verify_episode_checksum`, `find_corrupted_episodes`
- **TransformationTrace + TransformStep:** ghi lại lịch sử biến đổi prompt
- **Provenance:** git hash tự động, `Provenance` dataclass
- **Export/Import/Snapshot:** JSONL campaign, snapshot với evidence
- **Reproducibility:** `reconstruct_campaign` chronological generator
- **Persistence:** test với file DB thật, batch 50k records
- **Dataclasses:** `Episode` (17 fields), `InterventionRecord` (13 fields), `EpisodeEvidence`, `EpisodeFilter`, `Provenance`, `TransformationTrace`, `TransformStep`
- **Kết quả:** 63/63 tests pass

### 1.2. Scientific Memory (L6) — Khởi tạo

**File:** `knowledge/scientific_memory.py` — Neo4j backend.

**Tính năng:**
- **Theory dataclass:** auto-ID (`thr_<hex>`), clamp confidence/version, dict roundtrip
- **CRUD + Versioning:** `save_theory` tạo version mới (tăng dần), `get_theory` (latest), `get_theory_version` (specific)
- **Dynamic conditions:** conditions lưu thành dynamic Cypher properties, filtering tại Cypher level (không filter Python)
- **NEXT_VERSION chain:** mỗi version là node riêng, liên kết bằng relationship
- **`find_theories_by_pattern`:** Cypher `CONTAINS` substring search, case-sensitive/insensitive
- **Export/Import:** `export_theories(include_history)`, `import_theories(overwrite_existing, include_history)` — hỗ trợ version history
- **Schema:** `(id, version)` UNIQUE constraint, indexes trên confidence + condition keys
- **Kết quả:** 39/39 tests pass (khi có Neo4j), 5 unit tests pass (không Neo4j)

### 1.3. Documentation

- `knowledge/SCIENTIFIC_MEMORY.md` — Hướng dẫn cài đặt Neo4j, schema, API, kiến trúc

---

## Mốc 2 — 8/6/2026

### 2.1. Semantic Memory (L5) — Production-ready

**File:** `knowledge/semantic_memory.py` — FAISS (primary) + SQLite (metadata).

#### Backend
| Backend | Khi có FAISS | Khi không có FAISS |
|---------|-------------|-------------------|
| Vector search | `faiss.IndexFlatIP` (inner product) | numpy cosine similarity |
| Keyword search | SQLite FTS5 (BM25) | SQL `LIKE` + TF scoring |
| Dim | Auto-detect từ vector đầu tiên | 384 (default) |

#### API

**CRUD:**
- `add_embedding(episode_id, content_type, content, embedding) → str`
- `add_embeddings_batch(embeddings: List[StoredEmbedding]) → List[str]` — batch insert 1 transaction
- `get_embedding(id) → Optional[StoredEmbedding]`
- `delete_embedding(id) → bool`, `delete_by_episode(episode_id) → int`, `delete_all() → int`

**Search:**
- `search_by_embedding(query_vector, top_k, content_type_filter, min_similarity)` — FAISS/numpy
- `search_by_text(text, ...)` — sentence-transformers → `search_by_embedding`
- **`hybrid_search(text, top_k, content_type_filter, keyword_weight, vector_weight)`** — union keyword + vector, normalize, combine

**Episode Integration:**
- `add_from_episode(episode)` — tự động tạo 3 embeddings: prompt, response, summary
- `sync_episode(episode_id)` — kiểm tra EpisodicMemory, tạo/xoá embedding tương ứng
- `auto_sync_episodes` flag — validation khi add

**Export/Import:**
- Legacy: `.json` file (single file)
- Enhanced: directory với `metadata.json` + `faiss.index`
- `import_(path)` — tự động detect format

**Stats:**
- `count() → int`, `get_stats() → Dict`

#### Kiến trúc FAISS

```
SemanticMemory
├── SQLite
│   ├── embeddings table (BLOB float32)
│   └── embeddings_fts (FTS5 virtual table)
├── FAISS IndexFlatIP (in-memory)
│   └── _faiss_ids: List[str] (position → embedding ID)
└── Cơ chế dirty flag:
    - delete → _faiss_dirty = True
    - search → rebuild nếu dirty
```

#### Logging
- `logging.getLogger(__name__)` — warning khi FAISS/ST không available
- `_HAS_FAISS`, `_HAS_ST` flags

#### Kết quả: 51/51 tests pass

| Nhóm | Số test |
|------|---------|
| Dataclass tests | 2 |
| CRUD | 7 |
| FAISS | 4 |
| Batch insert | 4 |
| Search (embedding) | 5 |
| Keyword search (FTS + LIKE) | 5 |
| Hybrid search | 4 |
| Search by text | 2 |
| Episode integration | 4 |
| Export/Import | 4 |
| Stats | 2 |
| Context manager | 1 |
| FAISS fallback | 2 |
| FTS fallback | 1 |
| Edge cases | 5 |

### 2.2. Scientific Memory (L6) — Auto-Compact

**Nâng cấp trên `knowledge/scientific_memory.py`:**

#### Methods mới

| Method | Mô tả |
|--------|-------|
| `compact_theory(theory_id, keep_versions=10) → int` | Giữ N version mới nhất, xoá phần còn lại |
| `compact_all(keep_versions=10) → Dict[str, int]` | Compact tất cả theory |
| `compact_if_needed(keep_versions, max_versions_before_compact) → Dict[str, int]` | Tự động trigger khi vượt ngưỡng |
| `compact_older_than(days, keep_versions=1) → Dict[str, int]` | Xoá version cũ hơn N ngày |
| `get_version_stats(theory_id) → Dict` | Monitoring: total, oldest/newest version, estimated size |

#### Background Auto-Compact

```python
memory.set_auto_compact_enabled(
    enabled=True,
    keep_versions=10,
    check_interval_minutes=60,
)
memory.disable_auto_compact()
```

- `threading.Timer` daemon thread
- Định kỳ chạy `compact_if_needed` với `max_versions_before_compact = keep_versions * 2`
- Tự động stop khi `close()`
- Lock an toàn (`_auto_compact_lock`)

#### Logging
- `logging.getLogger(__name__)`
- `compact_theory`: log số version xoá + thời gian
- `compact_all`, `compact_if_needed`, `compact_older_than`: log summary
- Auto-compact thread: log error

#### Indexes mới
- `updated_at` — cho `compact_older_than`
- `created_at` — cho time-based queries

#### Kết quả: 47/47 tests pass (39 cũ + 8 mới)

| Test mới | Mô tả |
|----------|-------|
| `test_compact_if_needed_triggers_when_over_threshold` | 25 versions, threshold 20 → compact 15 |
| `test_compact_if_needed_noop_when_under_threshold` | 5 versions, threshold 20 → noop |
| `test_compact_if_needed_multiple_theories` | 3 theories, 2/3 triggers |
| `test_compact_older_than` | Backdated 31-35 ngày, keep 1 → xoá 4 |
| `test_compact_older_than_keeps_minimum` | 3 versions, keep 2 → xoá 1 |
| `test_get_version_stats` | 7 versions → stats đúng |
| `test_get_version_stats_nonexistent` | Theory không tồn tại → 0 |
| `test_auto_compact_enable_disable` | Toggle flag + keep_versions |

### 2.3. Documentation

- `knowledge/SEMANTIC_AND_AUTOCOMPACT.md` — Full docs:
  - Semantic Memory: FAISS, FTS, hybrid search, batch, episode sync, export/import
  - Auto-Compact: compact_if_needed, compact_older_than, background thread, stats
  - Test coverage: 161 tests (63 + 47 + 51)

### 2.4. Module Exports

`knowledge/__init__.py` exports:
```python
# Episodic (L1)
EpisodicMemory, Episode, EpisodeEvidence, EpisodeFilter,
InterventionRecord, Provenance

# Scientific Memory (L6)
ScientificMemory, Theory

# Semantic Memory (L5)
SemanticMemory, StoredEmbedding
```

## Mốc 5 — 6/6/2026 (Researcher Agent)

### 5.1. Researcher Agent (`agents/researcher.py`)

**Class:** `ResearcherAgent` — pipeline end‑to‑end cho reverse engineering.

**Constructor tự động tạo:**
- `CVC5Synthesizer(max_depth=3, beam_width=200, timeout=30, use_cache=True)`
- `ProgramExecutor(default_registry)`

**Phương thức:**

| Method | Mô tả |
|--------|-------|
| `synthesize_from_campaign(campaign_id, experiment_id, allow_error_rate)` | Đọc episodes → synthesis → stats |
| `verify_program(program, victim, num_test, threshold, exclude_prompts, verbose)` | Verify với victim |
| `store_program(program, name, confidence, provenance, status)` | Lưu vào DefenseProgramStore |
| `abstract_theory(program, model_family, conditions, provenance)` | Trích xuất Theory |
| `store_theory(theory)` | Lưu vào ScientificMemory |
| `run_reverse_engineering_pipeline(campaign_id, victim, ...)` | Pipeline 5 bước end‑to‑end |

**Pipeline 5 bước:**
1. `synthesize_from_campaign` → nếu không có program, dừng
2. `verify_program` → accuracy + verified flag
3. `store_program` → program_id (status=confirmed/draft)
4. `abstract_theory` → Theory pattern từ program
5. `store_theory` → theory_id

**Xử lý lỗi:** Exception trong pipeline → log + return dict `success=False`.

**Kết quả: 17/17 tests pass**

| Test class | Số test |
|------------|---------|
| TestSynthesizeFromCampaign | 4 |
| TestVerifyProgram | 2 |
| TestStoreProgram | 2 |
| TestAbstractTheory | 2 |
| TestStoreTheory | 1 |
| TestPipeline | 5 |
| TestDefaultSynthesizer | 1 |

**Files:**
- `agents/__init__.py` — export `ResearcherAgent`
- `agents/researcher.py` — implementation
- `tests/agents/test_researcher.py` — 17 tests
- `docs/researcher_agent.md` — documentation

### 5.2. Cập nhật system test count

- **Synthesis:** 84 tests (1 skipped)
- **Knowledge layer:** 207 tests
- **Researcher Agent:** 17 tests
- **Total:** **419 tests** (+17 từ Researcher Agent)

### 5.3. Ghi chú

- Agent **không dùng LLM** — chỉ dùng synthesis module + memory layers.
- Có thể test độc lập với mock: `python -m pytest tests/agents/test_researcher.py -v`
- `verify_program` tạo `ProgramVerifier` mới mỗi lần gọi (victim‑specific).
- Pipeline tự động dừng nếu synthesis không tìm được program.


## Mốc 7 — 6/6/2026 (Cognitive Agent)

### 7.1. Cognitive Agent (`agents/cognitive.py`)

**Class:** `CognitiveAgent` — phát hiện bất thường và sinh giả thuyết cấu trúc.

**Constructor:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episodic_memory` | **required** | Source of episode data |
| `llm_client` | auto (env vars) | LLM for hypothesis generation |
| `ontology_memory` | `None` | Optional primitive catalog |
| `grammar_exporter` | auto (`default_registry`) | Primitive catalog provider |
| `anomaly_threshold` | `0.2` | Min outcome diff to flag anomaly |
| `base_prompts` | `DEFAULT_BASE_PROMPTS` (26 prompts) | Base prompts for grouping |

**Phương thức:**

| Method | Mô tả |
|--------|-------|
| `detect_anomalies(campaign_id, experiment_id)` | Đọc episodes → group by base prompt → detect outcome differences |
| `generate_hypotheses(anomalies, prior_hypotheses)` | LLM prompt → parse JSON → Hypothesis[] |
| `estimate_confidence(hypothesis, anomalies)` | Laplace smoothing: `(supporting + 1) / (total + 2)` |

**Data classes:**

- `Anomaly` — `id`, `base_prompt`, `transform_names`, `outcome_original/transformed`, `difference`, `episode_id_original/transformed`, `timestamp`
- `Hypothesis` — `id`, `description`, `condition`, `confidence`, `supporting_anomaly_ids`, `created_at`

**Logic phát hiện bất thường:**
1. Lọc episodes theo `EpisodeFilter(campaign_id, experiment_id)`
2. Bỏ qua episode có `outcome is None`
3. Gom nhóm theo `intervention.prompt`
4. Trong mỗi nhóm, nếu có từ 2 outcomes khác nhau → tạo Anomaly cho mỗi cặp

**Logic sinh giả thuyết:**
1. Tạo prompt template với mô tả anomalies + primitive catalog
2. Gọi `LLMClient.generate(prompt, temperature=0.0)`
3. Parse JSON response (direct / markdown code block / regex fragment)
4. Fallback: hypothesis mặc định (keyword filter + decode cipher) khi LLM fails
5. Giới hạn 5 hypotheses

**Kết quả: 39/39 tests pass**

### 7.4. Cải tiến (10 điểm, Mốc 7.1)

Sau khi hoàn thiện Cognitive Agent, đã áp dụng 10 cải tiến:

| # | Cải tiến | Mô tả |
|---|----------|-------|
| 1 | **Tích hợp `prior_hypotheses` vào prompt LLM** | Prior hypotheses được đưa vào prompt, LLM được yêu cầu tinh chỉnh, cải thiện hoặc đề xuất mới, tránh trùng lặp |
| 2 | **Phát hiện transform chain anomaly** | `_detect_transform_chain_anomalies()`: phát hiện trường hợp chỉ khi kết hợp nhiều transforms mới thay đổi outcome |
| 3 | **Tùy chỉnh `base_prompts` qua file cấu hình** | `load_base_prompts(path)` hỗ trợ JSON và YAML; tham số `base_prompts_path` trong constructor |
| 4 | **Logging chi tiết parse LLM response** | Mỗi parsing strategy ghi `logger.debug` kết quả (thành công/thất bại, số lượng items) |
| 5 | **`experiment_id` kết hợp `campaign_id`** | Test mới kiểm tra cả hai filter hoạt động đồng thời |
| 6 | **Weighted confidence** | `estimate_confidence` dùng weighted Laplace: anomalies có `difference` lớn hơn được trọng số cao hơn |
| 7 | **Validation `anomaly_threshold`** | Clamp `[0, 1]` kèm warning log nếu ngoài khoảng |
| 8 | **Anomaly store callback** | Constructor nhận `anomaly_store: Callable[[List[Anomaly]], None]`; lỗi callback không làm gián đoạn detect |
| 9 | **Unit test `prior_hypotheses` có nội dung** | `test_prior_hypotheses_included_in_prompt` kiểm tra prior hypotheses xuất hiện trong prompt |
| 10 | **Tài liệu** | `docs/cognitive.md` — tài liệu đầy đủ: thiết kế, API, tests, so sánh Researcher Agent |

### 7.5. Cập nhật test count

| Khu vực | Tests cũ | Tests mới |
|---------|----------|-----------|
| Constructor | 4 | 7 (+3: clamping, anomaly store) |
| load_base_prompts | — | 5 (mới) |
| detect_anomalies (pairwise) | 11 | 11 |
| Transform chain | — | 4 (mới) |
| Anomaly store | — | 3 (mới) |
| generate_hypotheses | 6 | 7 (+1: prior_hypotheses in prompt) |
| estimate_confidence | 5 | 5 (sửa weighted) |
| Pipeline | 2 | 2 |
| Data class serialisation | 2 | 2 |
| LLM parsing | 2 | 2 |
| **Total** | **39** | **56** |

| Area | Tests |
|------|-------|
| Knowledge layer | 207 |
| Synthesis module | 84 (1 skipped CVC5) |
| Primitive tests | 200 |
| Researcher Agent | 40 |
| Cognitive Agent | 56 |
| **Total** | **~690** |

### 7.6. Files updated

| File | Change |
|------|--------|
| `agents/cognitive.py` | Thêm: `load_base_prompts()`, `_detect_transform_chain_anomalies()`, weighted confidence, threshold clamping, anomaly_store, prior_hypotheses prompt, parse logging |
| `tests/agents/test_cognitive.py` | 80 tests (56 cũ + 24 mới: base_prompts validation, anomaly store queue, persist, LLM retry, primitive cache, get_anomalies type, logging format) |
| `docs/cognitive.md` | Tài liệu đầy đủ (thay thế `docs/cognitive_agent.md`) |
| `TODO.md` | Cập nhật test counts |
| `CHECKPOINT.md` | Mốc 7 mở rộng |
| `CHECKPOINT.md` | Mốc 7 added |


## Mốc 6 — 6/6/2026 (92 primitives + full tests)

### 6.1. Mở rộng lên 92 primitives

**Thay đổi trong `core/primitive.py`:**
- **Predicates:** 27 (thêm 19 mới: ContainsAnyWord, ContainsAllWords, LengthLt, StartsWith, EndsWith, HasNumber, HasSpecialChar, IsAllCaps, ContainsLeet, ContainsRot13, ContainsBase64, ContainsHex, IsEmpty, StartsWithRoleplay, ContainsSystemOverride, ContainsDelimiter, ContainsCodeBlock, HasEmoji, ContainsURL, Sentiment, Intent, MatchesJailbreakPattern, ContainsEncodingWrapper, IsRepetitive)
- **Transforms:** 38 (thêm 30 mới: Base64Decode, ToLowercase, ToUppercase, LeetSpeak, ReverseText, PigLatin, MorseCode, AddPrefix, AddSuffix, WrapCodeBlock, InsertTypos, WordShuffle, AddMarkdown, AddZeroWidthChars, UnicodeObfuscate, HtmlEncode, URLEncode, QuotedPrintable, BinaryEncode, HexEncode, RemoveVowels, Boustrophedon, AtbashCipher, CaesarCipher, VigenereCipher, RailFenceCipher, RemoveWhitespace, InsertSynonyms, EscapeQuotes, FormatAsJson, AddRolePlay, Truncate, PadToLength, RandomCase, CharacterSubstitution)
- **Classifiers:** 27 (thêm 22 mới: IntentScore, ObscurityScore, LengthScore, RepetitionScore, EntropyScore, LanguageScore, JailbreakLikelihood, ContainsBlacklistedWord, SpecialCharRatio, DigitRatio, UpperCaseRatio, PunctuationRatio, WhitespaceRatio, UniqueTokenRatio, Gpt2Perplexity, EncodingDetection, RefusalSimilarity, HarmfulnessSimilarity, CodeLikelihood, JsonLikelihood, SqlLikelihood, PromptInjectionLikelihood, RoleplayLikelihood, AdversarialSuffixScore, PersuasionScore)

### 6.2. Bug fixes

| Bug | Fix |
|-----|-----|
| `ContainsRot13Predicate` always returned True (used full alphabet) | Changed check to n-z half only with 50% threshold |
| `ContainsBase64Predicate` failed on simple base64 strings | Simplified to `ratio > 0.95` check |
| `QuotedPrintableTransform` didn't encode non-ASCII chars | Added `ord(c) < 128` guard |
| `to_dict()` failed on dataclass primitives (no version_id attr) | Changed to `getattr(self, ..., default)` |
| Comment said 38 transforms but only 34 registered | Added 4 new transforms (Truncate, PadToLength, RandomCase, CharacterSubstitution) |

### 6.3. Unit tests — 200 tests for 92 primitives

**File:** `tests/core/test_primitive.py` — 200 tests covering all 92 primitives.

| Test group | Count |
|------------|-------|
| Predicate tests (27 classes) | ~72 tests |
| Transform tests (38 classes) | ~88 tests |
| Classifier tests (27 classes) | ~33 tests |
| Registry tests | 7 tests |
| **Total** | **200 tests** |

### 6.4. Documentation

- `docs/primitive_catalog.md` — full table of all 92 primitives with parameters and descriptions

### 6.5. System test count update

| Area | Tests |
|------|-------|
| Knowledge layer | 207 |
| Synthesis module | 80 (84 - 1 skipped CVC5) |
| Primitive tests | 200 |
| Researcher Agent | 17 |
| **Total** | **~614** |

### 6.6. Files changed/created

| File | Change |
|------|--------|
| `core/primitive.py` | 92 primitives, 4 new transforms, 3 bug fixes, to_dict fix |
| `tests/core/test_primitive.py` | 200 tests created |
| `docs/primitive_catalog.md` | Created |
| `TODO.md` | Updated test counts, primitive stats |
| `CHECKPOINT.md` | Mốc 6 added |


## Mốc 4 — 6/6/2026 (synthesis hardening + real classifiers)

### 4.1. `_matches_all` — sửa lỗi non-deterministic

**Vấn đề:** `_matches_all` cũ yêu cầu "ít nhất 1/10 trial match" — classifier non-deterministic vẫn pass dễ dàng.

**Fix:** Yêu cầu **tất cả 10 trials phải match**. Loại bỏ gần như hoàn toàn nhiễu.

### 4.2. ToxicityScoreClassifier thật

**Vấn đề:** Dùng `random.random()` → không phù hợp synthesis.

**Fix:** Dùng TextBlob (sentiment polarity + keyword boost) nếu có, fallback keyword heuristic (`bomb`, `kill`, … × 0.25). Deterministic.

### 4.3. SentimentClassifier (mới)

`SentimentClassifier` dùng TextBlob (polarity→score) hoặc keyword heuristic (positive/negative word ratio). Đã đăng ký vào `default_registry`.

### 4.4. Beam width + guard combinatorial explosion

| Vấn đề | Fix |
|--------|-----|
| `beam_width=0` (default cũ) → enumeration sinh 1.6M nodes ở depth 4 | Default `beam_width=200` |
| `_enumerate_conditions` không giới hạn | Internal `limit=2000` absolute |
| Auto depth tăng vô hạn | Skip nếu `beam_width=0` và `depth > max_depth + 1` |
| Fitness scoring thiếu | `_fitness_score()` + top-k beam selection |

### 4.5. Disk cache

Tham số `cache_path`: persist `_matches_all` results qua pickle. Load khi init, save sau mỗi depth. Key = `hash(canonical_form, tuple(examples))`.

### 4.6. Free variable thresholds (SMT-LIB)

`export_to_smtlib(use_free_thresholds=True)`: dùng `(declare-fun threshold_toxicity_score () Real)` + assert [0, 1] thay vì hardcode `(> (toxicity_score x) 0.5)`. `_extract_thresholds_from_model()` đọc và clamp.

### 4.7. Verbose logging (verifier)

`ProgramVerifier.verify(verbose=True)`: log từng intervention prompt + expected + actual outcome.

### 4.8. Fitness scoring + beam selection

`_fitness_score()` = `correct / len(examples)`. Trong `_try_enumeration`:
1. Lấy `beam_width * 2` programs từ enumeration
2. Tính fitness cho từng program
3. Sort theo `(-score, complexity())`
4. Giữ top `beam_width`

### 4.9. Transform chains + extended tests

- `ApplyTransformNode` chains ở depth 3+: `rot13(to_lowercase(contains_word("bomb")))`
- `_nodes_at_transform_depth` hỗ trợ nested ApplyTransformNode
- 32 extended tests: free thresholds, beam width, disk cache, verbose, real classifier, transform chains, abstract theory, episodic integration

### 4.10. Kết quả tests

| File | Tests cũ | Tests mới |
|------|----------|-----------|
| `test_cvc5_synthesizer.py` | 16 | 16 (giữ nguyên) |
| `test_cvc5_synthesizer_extended.py` | 28 | 32 |
| `test_verifier.py` | 14 | 18 |
| `test_grammar_exporter.py` | 18 | 18 (giữ nguyên) |
| **Synthesis TOTAL** | **82** | **84 (1 skipped)** |
| **System TOTAL** | **397** | **402** |

### 4.11. Files đã thay đổi

| File | Thay đổi |
|------|----------|
| `synthesis/cvc5_synthesizer.py` | beam_width, disk cache, fitness score, free thresholds, guard explosion, extract_thresholds |
| `synthesis/grammar_exporter.py` | `_enumerate_conditions` limit 2000, free threshold param, use_free_thresholds trong SMT |
| `synthesis/verifier.py` | `verbose` parameter |
| `core/primitive.py` | ToxicityScoreClassifier thật, SentimentClassifier mới |
| `tests/synthesis/test_cvc5_synthesizer_extended.py` | 32 tests mới |
| `SYNTHESIZER_AND_VERIFIER.md` | Cập nhật docs |
| `TODO.md` | Cập nhật test counts, primitives |
| `CHECKPOINT.md` | Mốc 4 mới |


## Mốc 3 — 8/6/2026

### 3.1. Synthesis Module (`synthesis/`)

**Files:**
- `synthesis/grammar_exporter.py` — Grammar enumeration + SMT-LIB export (*xem Mốc 4 cho cập nhật: limit 2000, free thresholds*)
- `synthesis/cvc5_synthesizer.py` — Program synthesis (enumeration + optional CVC5) (*xem Mốc 4: beam_width, disk cache, fitness scoring, guard*)
- `synthesis/verifier.py` — Program verification against victim (*xem Mốc 4: verbose logging*)

**GrammarExporter (`synthesis/grammar_exporter.py`):**

| Method | Mô tả |
|--------|-------|
| `get_parameterized_primitives(examples)` | Trích xuất keywords từ REFUSE examples, tạo `ContainsWordPredicate` tương ứng |
| `enumerate_conditions(max_depth)` | Sinh tất cả AST condition nodes (depth 1: predicate/threshold, depth 2+: not/and/or/apply) |
| `enumerate_programs(max_depth)` | Bọc conditions vào `IfThenElseNode` → `Program` |
| `export_to_smtlib(examples, output_file)` | Xuất SMT-LIB 2.6 cho CVC5 |

**Keyword Extraction:**
- Lọc example có `outcome == 1` (REFUSE)
- Tách từ, lọc chữ cái ≥ 3 ký tự
- Tạo `ContainsWordPredicate(word=k)` cho mỗi keyword
- Giảm không gian tìm kiếm từ vô hạn → O(k)

**CVC5Synthesizer (`synthesis/cvc5_synthesizer.py`):**

| Method | Mô tả |
|--------|-------|
| `synthesize(examples, primitive_registry, ontology_memory)` | Entry point: thử enumeration → CVC5 fallback |
| `_try_enumeration(examples, exporter)` | Duyệt enumerated programs, kiểm tra từng chương trình |
| `_try_cvc5(examples, exporter)` | Gọi CVC5 subprocess với file SMT-LIB tạm |
| `_matches_all(program, examples, executor, num_trials=10)` | Kiểm tra program khớp tất cả examples (10 lần mỗi example để loại nhiễu từ non-deterministic classifier) |

**Non-deterministic handling:**
- `ToxicityScoreClassifier` dùng `random.random()` → không phù hợp synthesis
- `_matches_all` chạy mỗi example **10 lần** để đảm bảo chương trình deterministic
- Xác suất random program pass: *extremely* low

**ProgramVerifier (`synthesis/verifier.py`):**

| Method | Mô tả |
|--------|-------|
| `verify(program, num_test_interventions, accuracy_threshold)` | Sinh N can thiệp, so sánh victim outcome vs program outcome |
| `_default_intervention_generator(victim, n)` | 20 base prompts + 3 transforms (ROT13, base64, lowercase) — deterministic |
| `_generate_suggestions(failures, program)` | Phân tích false positive/false negative |

**VerificationReport:**
```python
@dataclass
class VerificationReport:
    program: Program
    accuracy: float
    failures: List[Tuple[str, Outcome, Outcome]]
    suggestions: List[str]
    verified: bool
    num_tested: int
    num_correct: int
    def to_dict(self) -> dict
```

**Kết quả: 48/48 tests pass** (thời điểm 8/6)

| File tests | Số lượng |
|------------|----------|
| `tests/synthesis/test_grammar_exporter.py` | 18 |
| `tests/synthesis/test_cvc5_synthesizer.py` | 16 |
| `tests/synthesis/test_verifier.py` | 14 |
| **TOTAL** | **48** |

### 3.2. Module Exports

`knowledge/__init__.py` được mở rộng:
```python
# Defense Program Store (L4)
DefenseProgramRecord, DefenseProgramStore

# Ontology Memory (L5)
OntologyPrimitive, OntologyMemory
```

### 3.3. Sửa lỗi

- **`_matches_all`**: Từ 1 → 10 trials mỗi example để loại non-deterministic classifiers
- **`_default_intervention_generator`**: Xoá `random.shuffle` để deterministic
- **`test_verify_wrong_program`**: Tăng `num_test_interventions` 5→12 để bao gồm "bomb" prompts

### 3.4. Cải tiến synthesis module (10 điểm)

**1. CVC5 integration hoàn chỉnh:**
- `_parse_cvc5_output` balanced-parentheses SMT-LIB model parser (không dùng regex đơn giản)
- `_parse_smt_model` trích xuất function definitions từ S‑expression
- Auto depth increase khi enumeration thất bại (tối đa max_depth+2)
- Timeout + xử lý lỗi subprocess

**2. Mở rộng grammar:**
- ClassifierNode + ThresholdNode với THRESHOLD_CANDIDATES
- ApplyTransformNode chain (depth 2+)
- AndNode/OrNode/NotNode logic operators
- Ontology Memory integration qua `get_primitives()` → `_get_from_ontology()`

**3. Tích hợp L5/L6:**
- `abstract_theory()` — trích xuất Theory pattern từ Program, tạo Theory dataclass
- `store_verified_program()` — lưu Program vào DefenseProgramStore

**4. Noise tolerance:**
- Tham số `allow_error_rate` (0..1) trong constructor
- `_matches_all` chấp nhận `max_errors` errors

**5. Tối ưu enumeration:**
- `max_programs_per_depth` — beam search
- Caching `_matches_all` theo canonical hash
- Complexity sorting: programs.sort(key=lambda p: p.complexity())

**6. Episodic Memory integration:**
- `synthesize_from_episodes(episodic_memory, campaign_id, experiment_id)` — tự động đọc episodes, tạo examples

**7. Occam complexity:**
- `enumerate_programs` sắp xếp theo node count
- Trả về chương trình đơn giản nhất (complexity())

**8. Edge intervention generation:**
- Thêm RemovePunctuationTransform, role‑prefix variants
- `exclude_prompts` parameter trong `verify()` để loại trừ training set

**9. Stats & logging:**
- `SynthesisStats` dataclass (duration, depth, programs_tried, cache_hits, cvc5_used, errors, v.v.)
- `synthesize_with_stats()` trả về (Program, SynthesisStats)

**10. Integration tests:**
- 34 extended tests covering tất cả tính năng mới (mock memory/store)
- 82 synthesis tests total, 397 system-wide
- *(Mốc 4 mở rộng: 84 synthesis tests, 402 system-wide)*

### 3.5. Documentation

- `SYNTHESIZER_AND_VERIFIER.md` — Full docs: API, kiến trúc, design decisions, future work

---

## Tổng kết

### File structure

```
knowledge/
├── __init__.py                    # Exports (5 modules: L1, L4, L5, L6 + Semantic)
├── episodic/
│   └── episodic.py                # L1 — SQLite (63 tests)
├── defense_store.py               # L4 — Neo4j (25 tests)
├── ontology_memory.py             # L5 — Neo4j (19 tests)
├── scientific_memory.py           # L6 — Neo4j (47 tests)
├── semantic_memory.py             # L5 — FAISS + SQLite (53 tests)
├── SCIENTIFIC_MEMORY.md
└── SEMANTIC_AND_AUTOCOMPACT.md

synthesis/
├── __init__.py
├── grammar_exporter.py            # Grammar enumeration + SMT-LIB (18 tests) — updated Mốc 4
├── cvc5_synthesizer.py            # Program synthesis (16 tests) — updated Mốc 4
└── verifier.py                    # Program verification (18 tests) — updated Mốc 4

tests/knowledge/
├── test_episodic.py               # 63 tests
├── test_defense_store.py          # 25 tests (6 unit + 19 Neo4j)
├── test_ontology_memory.py        # 19 tests (5 unit + 14 Neo4j)
├── test_scientific_memory.py      # 47 tests
└── test_semantic_memory.py        # 53 tests

tests/synthesis/
├── test_grammar_exporter.py       # 18 tests
├── test_cvc5_synthesizer.py       # 16 tests
├── test_cvc5_synthesizer_extended.py  # 32 tests — extended features
└── test_verifier.py               # 18 tests
```

### Test results

| Module | Backend | Tests | Trạng thái |
|--------|---------|-------|-----------|
| Episodic Memory (L1) | SQLite | 63 | ✅ PASS |
| Defense Program Store (L4) | Neo4j | 25 | ✅ PASS |
| Ontology Memory (L5) | Neo4j | 19 | ✅ PASS |
| Scientific Memory (L6) | Neo4j | 47 | ✅ PASS |
| Semantic Memory (L5) | FAISS + SQLite | 53 | ✅ PASS |
| Grammar Exporter | Python | 18 | ✅ PASS |
| CVC5 Synthesizer | Python | 48 (16 base + 32 ext) | ✅ PASS (1 skipped nếu không có CVC5 binary) |
| Program Verifier | Python | 18 | ✅ PASS |
| **TOTAL** | | **402** | ✅ **ALL PASS** |

### Dependencies

| Package | Priority | Used by | Fallback |
|---------|----------|---------|---------|
| `neo4j` | Required | Scientific Memory | — |
| `numpy` | Required | Semantic Memory | — |
| `faiss-cpu` | **Primary** | Semantic Memory (vector search) | numpy cosine similarity (⚠️ warning) |
| `sentence-transformers` | **Primary** | Semantic Memory (text→embedding) | keyword search (⚠️ warning) |
| `scikit-learn` | Required | `adapters/toy_victims/neural.py` | — |
| `networkx` | Required | `evaluation/structural_recovery.py` | — |
| CVC5 binary | Optional | Synthesis (CVC5Synthesizer) | Enumeration path (⚠️ warning log) |

---

## Mốc 8 — 6/6/2026 (Strategist Agent + Orchestrator)

### 8.1. Strategist Agent (`agents/strategist.py`)

**Class:** `StrategistAgent` — thiết kế và thực thi can thiệp tối ưu để phân biệt giả thuyết.

**Constructor:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episodic_memory` | **required** | Ghi kết quả can thiệp vào L1 |
| `executor` | auto (`default_registry`) | Đánh giá hypothesis dạng Program AST |
| `llm_client` | `None` | LLM cho hybrid mode & outcome prediction |
| `grammar_exporter` | auto (`default_registry`) | Lấy danh sách transforms |
| `primitive_registry` | `default_registry` | Fallback registry |
| `intervention_budget` | 50 | Số transforms tối đa thử mỗi prompt |
| `use_llm` | `True` | Bật LLM-guided generation |
| `temperature` | 0.7 | Nhiệt độ LLM |
| `max_prompt_length` | 2000 | Giới hạn độ dài prompt |

**Phương thức:**

| Method | Mô tả |
|--------|-------|
| `select_hypothesis_pair(hypotheses)` | Chọn cặp có độ bất định cao nhất `1 - |conf₁ - conf₂|` |
| `design_intervention(h1, h2, base_prompts)` | Heuristic local search + LLM-guided, chọn can thiệp có `|pred₁ - pred₂|` lớn nhất |
| `execute_intervention(intervention, victim)` | Gửi prompt → victim, trả về Outcome |
| `store_intervention(intervention, outcome, campaign_id, h1, h2, ...)` | Tạo Episode, ghi vào Episodic Memory |
| `run_intervention_round(hypotheses, victim, campaign_id, ...)` | End-to-end: select → design → execute → store |
| `evaluate_discriminative_power(intervention, h1, h2)` | Tính Δ(I; Π₁, Π₂) |
| `refresh_primitive_cache()` | Xoá primitive cache |

**Kiến trúc xử lý hypothesis:**
- Duck-typed: chấp nhận `agents.cognitive.Hypothesis` (text-based) và `core.hypothesis.Hypothesis` (Program-based)
- `_predict_outcome`: ProgramExecutor → LLM → keyword extraction (regex `'...'`) → default ACCEPT

**Hybrid intervention design:**
- Heuristic local search: thử identity + từng transform, tính Δ, chọn max
- LLM-guided (khi `use_llm=True` và `llm_client` khả dụng): gợi ý transforms
- Nếu LLM lỗi: fallback heuristic
- Early return khi Δ=1.0 (perfect discrimination)

**Kết quả: 62/62 tests pass**

| Test class | Số test |
|------------|---------|
| TestConstructor | 3 |
| TestSelectHypothesisPair | 4 |
| TestPredictOutcome | 5 |
| TestDiscriminativePower | 3 |
| TestDesignIntervention | 5 |
| TestExecuteIntervention | 2 |
| TestStoreIntervention | 2 |
| TestRunInterventionRound | 3 |
| TestLlmGuidedIntervention | 3 |
| TestRefreshPrimitiveCache | 2 |
| TestEdgeCases | 4 |
| TestApplyTransformName | 2 |
| TestTransformChain | 5 |
| TestTransformChainCustomDepth | 3 |
| TestBudgetClamping | 3 |
| TestNonDeterministic | 4 |
| TestBasePromptsFromMemory | 4 |
| TestCandidateLimits | 3 |
| TestAutoInvalidate | 2 |

### 8.2. Orchestrator (`orchestration/__init__.py`)

**Class:** `Orchestrator` — điều phối vòng lặp 6-phase.

**Constructor:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cognitive_agent` | **required** | CognitiveAgent instance |
| `strategist_agent` | **required** | StrategistAgent instance |
| `researcher_agent` | **required** | ResearcherAgent instance |
| `episodic_memory` | **required** | Central L1 store |
| `max_iterations` | 10 | Số vòng lặp tối đa |
| `convergence_threshold` | 0.05 | Ngưỡng hội tụ |

**State machine (`OrchestratorPhase`):** `IDLE → ANOMALY_DETECTION → HYPOTHESIS_GENERATION → INTERVENTION_DESIGN → INTERVENTION_EXECUTION → PROGRAM_SYNTHESIS → VERIFICATION_AND_STORE → CONVERGED`

**Pipeline flow:**

```
run_pipeline(campaign_id, victim, experiment_id)
  │
  ├── Phase 1-2: Cognitive Agent
  │     ├── detect_anomalies(campaign_id, experiment_id)
  │     └── generate_hypotheses(anomalies, prior_hypotheses)
  │
  ├── Phase 3-4: Strategist Agent
  │     ├── design_intervention(h1, h2) cho top 3 hypothesis pairs
  │     └── execute_intervention(intervention, victim)
  │
  ├── Phase 5-6: Researcher Agent
  │     └── run_reverse_engineering_pipeline(campaign_id, victim)
  │
  └── Check convergence → tiếp tục hoặc dừng
```

**Hội tụ:** Dừng khi có `program_id` từ Researcher AND không còn anomalies mới.

**Kết quả: 15/15 tests pass**

| Test class | Số test |
|------------|---------|
| TestConstructor | 3 |
| TestRunPipeline | 6 |
| TestCheckConvergence | 4 |
| TestPhaseStrategist | 2 |

### 8.3. Cập nhật test count

| Khu vực | Tests |
|---------|-------|
| Knowledge layer | 207 |
| Synthesis module | 84 (1 skipped CVC5) |
| Primitive tests | 200 |
| Researcher Agent | 40 |
| Cognitive Agent | 80 |
| Strategist Agent | 62 |
| Orchestrator | 15 |
| **Total** | **796** (5 skipped) |

### 8.4. Files created/updated

| File | Change |
|------|--------|
| `agents/strategist.py` | Mới: StrategistAgent class (~225 dòng) |
| `agents/__init__.py` | Thêm export `StrategistAgent` |
| `orchestration/__init__.py` | Mới: Orchestrator + OrchestratorPhase |
| `tests/agents/test_strategist.py` | Mới: 62 tests |
| `tests/orchestration/test_orchestrator.py` | Mới: 15 tests |
| `docs/strategist_agent.md` | Mới: tài liệu |
| `TODO.md` | Cập nhật agent layer, test counts |
| `CHECKPOINT.md` | Mốc 8 |

### 8.5. Orchestrator architecture

```
Orchestrator
├── Phase 1-2: Cognitive Agent
│   ├── detect_anomalies → List[Anomaly]
│   └── generate_hypotheses → List[Hypothesis]
├── Phase 3-4: Strategist Agent
│   ├── design_intervention(h1, h2) → Intervention
│   └── execute_intervention(I, victim) → Episode → EpisodicMemory
├── Phase 5-6: Researcher Agent
│   └── run_reverse_engineering_pipeline → Program + Theory
└── Convergence check
```

1. **Cognitive Agent** đọc Episodic Memory từ các vòng trước (nếu có)
2. **Strategist Agent** dùng hypotheses text-based để thiết kế can thiệp
3. Kết quả can thiệp được ghi vào **Episodic Memory** (cùng campaign)
4. **Researcher Agent** đọc tất cả episodes (cũ + mới) cho synthesis
5. Vòng lặp tiếp tục cho đến khi hội tụ hoặc hết `max_iterations`

### 8.6. Ghi chú

- `orchestration/__init__.py` hiện là file duy nhất trong `orchestration/` (single module pattern).
- StrategistAgent dùng duck-typing cho hypotheses → không cần import cụ thể, hỗ trợ nhiều loại hypothesis.
- Keyword extraction trong `_predict_outcome` dùng regex đơn giản `r"'([^']*)'"` — phù hợp condition pattern `contains_word('bomb')`.
- `intervention_budget` giới hạn số transforms thử mỗi prompt để tránh combinatorial explosion. Giá trị được clamp [1, 1000] với cảnh báo.
- `_check_convergence` dừng pipeline sớm khi có program + không còn anomalies.
- **Cognitive Agent fixes (6/6):** (1) `_validate_base_prompts()` — kiểm tra non-empty, ≤1000 ký tự. (2) Logging format dùng `fallback=%s` động. (3) `AnomalyStore.get_anomalies()` → `List[Anomaly]` thay vì `List[Dict]`. (4) Sửa bug `PRAGMA table_info` dùng `d[0]` thay vì `d[1]`. (5) Thêm 24 tests (base_prompts validation, anomaly store queue, persist, LLM retry, primitive cache, get_anomalies type, logging format). (6) Cập nhật docs AnomalyStore schema, primitive cache invalidation.
- **Strategist Agent improvements (10 items, 6/6):** (1) Transform chain support (`max_chain_depth`). (2) Base prompts từ Episodic Memory. (3) Logging metrics (candidates, avg Δ). (4) Non‑deterministic classifier handling (`num_trials`). (5) Clamp `intervention_budget` [1, 1000]. (6) Auto-fetch prompts từ campaign. (7) LLM prompt template documentation. (8) `max_candidates_heuristic` / `max_candidates_llm` riêng. (9) 8 transform chain tests. (10) OntologyMemory auto-invalidate hook.

### Ghi chú cho phiên sau

1. **Nếu thêm tính năng mới:** cập nhật CHECKPOINT.md với mốc mới.
2. **Test commands:**
    ```bash
    python -m pytest tests/ -v                           # Tất cả (796 tests)
    python -m pytest tests/agents/ -v                    # Agent layer (182 tests)
    python -m pytest tests/agents/test_strategist.py -v          # Strategist (62 tests)
    python -m pytest tests/orchestration/test_orchestrator.py -v # Orchestrator (15 tests)
    python -m pytest tests/agents/test_cognitive.py -v          # Cognitive (80 tests)
    python -m pytest tests/knowledge/test_episodic.py -v          # L1 — SQLite
    python -m pytest tests/knowledge/test_defense_store.py -v     # L4 — cần Neo4j
    python -m pytest tests/knowledge/test_ontology_memory.py -v   # L5 — cần Neo4j
    python -m pytest tests/knowledge/test_scientific_memory.py -v # L6 — cần Neo4j
    python -m pytest tests/knowledge/test_semantic_memory.py -v   # L5 FAISS
   ```
3. **Neo4j** phải chạy trên `localhost:7687` để test L4, L5, L6.
4. **FAISS** và **sentence-transformers** là optional; code tự fallback.
5. **CVC5** không bắt buộc; synthesis dùng enumeration path.
6. **Tests non-deterministic đã được xử lý:** `_matches_all` chạy 10 trials mỗi example, **yêu cầu ALL 10 match**.
7. **Beam width mặc định 200.** Nếu synthesis không tìm thấy lời giải, thử tăng `beam_width` hoặc dùng `beam_width=0` + giới hạn depth manually.
8. **Real classifiers** (`ToxicityScoreClassifier`, `SentimentClassifier`) deterministic, không cần lo lắng non-deterministic noise.
