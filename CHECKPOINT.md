# CHECKPOINT — HARMONY-X Knowledge Layer

> **Tổng quan toàn bộ hệ thống tại thời điểm hiện tại.**  
> Dùng làm mốc để cập nhật các phiên sau.
>
> - **Mốc 1 (7/6):** Episodic Memory (L1) hoàn chỉnh + Scientific Memory (L6) khởi tạo
> - **Mốc 2 (8/6):** Semantic Memory (L5) production-ready + Auto-Compact

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

---

## Tổng kết

### File structure

```
knowledge/
├── __init__.py                    # Exports (3 modules)
├── episodic/
│   └── episodic.py                # L1 — SQLite (63 tests)
├── scientific_memory.py           # L6 — Neo4j (47 tests)
├── semantic_memory.py             # L5 — FAISS + SQLite (51 tests)
├── SCIENTIFIC_MEMORY.md
└── SEMANTIC_AND_AUTOCOMPACT.md

tests/knowledge/
├── test_episodic.py               # 63 tests
├── test_scientific_memory.py      # 47 tests
└── test_semantic_memory.py        # 51 tests
```

### Test results

| Module | Backend | Tests | Trạng thái |
|--------|---------|-------|-----------|
| Episodic Memory (L1) | SQLite | 63 | ✅ PASS |
| Scientific Memory (L6) | Neo4j | 47 | ✅ PASS |
| Semantic Memory (L5) | FAISS + SQLite | 51 | ✅ PASS |
| **TOTAL** | | **161** | ✅ **ALL PASS** |

### Dependencies

| Package | Required? | Used by |
|---------|-----------|---------|
| `neo4j` | ✅ Required | Scientific Memory |
| `numpy` | ✅ Required | Semantic Memory |
| `faiss-cpu` | ❌ Optional | Semantic Memory (vector search) |
| `sentence-transformers` | ❌ Optional | Semantic Memory (text→embedding) |

### Ghi chú cho phiên sau

1. **Nếu thêm tính năng mới:** cập nhật CHECKPOINT.md với mốc mới.
2. **Test commands:**
   ```bash
   python -m pytest tests/knowledge/ -v                 # Tất cả
   python -m pytest tests/knowledge/test_semantic_memory.py -v   # L5
   python -m pytest tests/knowledge/test_scientific_memory.py -v # L6 (cần Neo4j)
   python -m pytest tests/knowledge/test_episodic.py -v          # L1
   ```
3. **Neo4j** phải chạy trên `localhost:7687` để test L6.
4. **FAISS** và **sentence-transformers** là optional; code tự fallback.
