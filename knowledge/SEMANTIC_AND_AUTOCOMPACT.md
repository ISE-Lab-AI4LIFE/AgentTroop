# Semantic Memory (L5) & Auto-Compact (L6) — HARMONY-X

## Tổng quan

- **Semantic Memory (L5):** Lưu trữ vector embedding với backend **FAISS** (ưu tiên) + SQLite (metadata). Cho phép tìm kiếm ngữ nghĩa (cosine similarity), tìm kiếm keyword (FTS5/LIKE), và hybrid (kết hợp cả hai).
- **Auto-Compact (L6):** Cơ chế dọn dẹp version history trong Scientific Memory (Neo4j) — tự động kích hoạt theo ngưỡng, theo thời gian, hoặc background thread định kỳ.

---

## Semantic Memory (L5)

### File cấu trúc

```
harmony-x/
├── knowledge/
│   ├── __init__.py                        # Export tất cả module
│   ├── semantic_memory.py                 # Module chính (FAISS + SQLite)
│   ├── scientific_memory.py              # Scientific Memory (Neo4j) + auto-compact
│   ├── SCIENTIFIC_MEMORY.md
│   └── SEMANTIC_AND_AUTOCOMPACT.md       # Tài liệu này
└── tests/
    └── knowledge/
        ├── test_episodic.py              # 63 tests
        ├── test_scientific_memory.py     # 47 tests (39 cũ + 8 mới)
        └── test_semantic_memory.py       # 51 tests (17 cũ + 34 mới)
```

### Yêu cầu

| Thư viện | Mức ưu tiên | Mục đích | Fallback |
|----------|------------|----------|----------|
| `numpy` | **Bắt buộc** | Vector operations, BLOB I/O | — |
| `faiss-cpu` | **Primary** | Vector index (tăng tốc search ~100x) | numpy cosine similarity (⚠️ warning) |
| `sentence-transformers` | **Primary** | Text → embedding (`search_by_text`, `hybrid_search`) | keyword search (⚠️ warning) |

**Primary =** được cài đặt mặc định và là phương án chạy chính. Nếu không có, code tự fallback kèm warning ra terminal.

### Lớp dữ liệu: `StoredEmbedding`

```python
@dataclass
class StoredEmbedding:
    id: str               # emb_<hex>
    episode_id: str       # FK → Episodic Memory
    content_type: str     # "prompt" | "response" | "summary" | custom
    content: str          # Text gốc
    embedding: List[float]  # Dense vector (float32)
    created_at: float     # Unix timestamp
    metadata: Dict        # Optional metadata
```

### Lớp `SemanticMemory`

**Khởi tạo:**

```python
from knowledge.semantic_memory import SemanticMemory

sm = SemanticMemory(":memory:")                          # In-memory (test)
sm = SemanticMemory("/tmp/vectors.db")                   # File-based
sm = SemanticMemory("vectors.db", auto_sync_episodes=True)  # Auto-sync với Episodic
```

#### CRUD

| Method | Mô tả |
|--------|-------|
| `add_embedding(episode_id, content_type, content, embedding, metadata) → str` | Thêm một embedding. |
| `add_embeddings_batch(embeddings: List[StoredEmbedding]) → List[str]` | Batch insert trong 1 transaction (nhanh hơn loop). |
| `get_embedding(emb_id) → Optional[StoredEmbedding]` | Lấy theo id. |
| `delete_embedding(emb_id) → bool` | Xoá một embedding. |
| `delete_by_episode(episode_id) → int` | Xoá tất cả embedding của một episode. |
| `delete_all() → int` | Xoá toàn bộ. |

#### Search

| Method | Mô tả |
|--------|-------|
| `search_by_embedding(query_embedding, top_k=10, content_type_filter=None, min_similarity=0.5) → List[Tuple[StoredEmbedding, float]]` | Tìm bằng vector. Dùng FAISS nếu có, fallback numpy. |
| `search_by_text(text, top_k=10, content_type_filter=None, min_similarity=0.5) → List[Tuple[StoredEmbedding, float]]` | Embed text → `search_by_embedding`. Cần sentence-transformers. |
| `hybrid_search(text, top_k=10, content_type_filter=None, keyword_weight=0.3, vector_weight=0.7) → List[Tuple[StoredEmbedding, float]]` | Kết hợp keyword (FTS5/LIKE) + vector (FAISS). |

**Hybrid Search** chi tiết:
1. Keyword search → BM25 (FTS5) hoặc TF (LIKE fallback) → top_k*2 candidates
2. Vector search → FAISS → top_k*2 candidates
3. Union candidates, normalize keyword score (min-max), combine với vector score
4. `combined = (keyword_weight * kw_norm + vector_weight * vec) / (keyword_weight + vector_weight)`

#### Episode Integration

| Method | Mô tả |
|--------|-------|
| `add_from_episode(episode, embedding_model=None) → List[str]` | Tự động tạo 3 embeddings: prompt, response, summary. |
| `sync_episode(episode_id) → Dict[str, int]` | Đồng bộ với EpisodicMemory: tạo summary nếu chưa có, xoá nếu episode đã bị xoá. |
| `auto_sync_episodes` (constructor flag) | Khi `True`, mỗi `add_embedding` kiểm tra episode_id tồn tại. |

#### Export / Import

| Method | Mô tả |
|--------|-------|
| `export(path)` | Legacy: nếu path kết thúc `.json` → ghi JSON. Mới: tạo thư mục chứa `metadata.json` + `faiss.index`. |
| `import_(path)` | Load từ JSON hoặc thư mục. |

#### Stats

| Method | Mô tả |
|--------|-------|
| `count() → int` | Tổng số embeddings. |
| `get_stats() → Dict` | Stats: total, dimension, FAISS availability, content type distribution. |

### Fallback Behaviour

Cả hai thư viện `faiss-cpu` và `sentence-transformers` đều được import bằng `try/except ImportError`:

```python
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    logger.warning("FAISS not installed — falling back to numpy")
```

Khi không có FAISS → `search_by_embedding` dùng numpy full-scan (chậm hơn nhưng vẫn đúng).

Khi không có sentence-transformers:
- `search_by_text` → fallback sang keyword search (FTS5/LIKE)
- `hybrid_search` → fallback sang keyword-only search
- `add_from_episode` → raise `RuntimeError` (cần model để tạo embedding)

### FAISS Architecture

```
SemanticMemory
├── SQLite (metadata)
│   ├── embeddings table (id, episode_id, content_type, content, embedding BLOB, ...)
│   └── embeddings_fts (FTS5 virtual table cho keyword search)
│
└── FAISS Index (in-memory)
    └── IndexFlatIP (inner product = cosine similarity cho normalized vectors)
        └── _faiss_ids: List[str]  # position → embedding ID mapping
```

- Vector được L2-normalize trước khi add vào FAISS.
- Khi delete, đặt `_faiss_dirty = True`; search trigger rebuild.
- Khi rebuild, đọc tất cả vectors từ SQLite, normalize, add vào FAISS mới.

### Schema SQLite

```sql
CREATE TABLE IF NOT EXISTS embeddings (
    id           TEXT PRIMARY KEY,
    episode_id   TEXT NOT NULL,
    content_type TEXT NOT NULL,
    content      TEXT NOT NULL,
    embedding    BLOB,              -- numpy.float32 raw bytes
    created_at   REAL NOT NULL,
    metadata     TEXT DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_fts
USING fts5(id UNINDEXED, content, content_type UNINDEXED);
```

---

## Auto-Compact (Scientific Memory — L6)

### Vấn đề

Mỗi lần `save_theory()` tạo một version mới. Với các theory được cập nhật nhiều lần, version history phình to, làm chậm query và tốn dung lượng Neo4j.

### Methods

| Method | Mô tả |
|--------|-------|
| `compact_theory(theory_id, keep_versions=10) → int` | Giữ N version mới nhất, xoá phần còn lại. |
| `compact_all(keep_versions=10) → Dict[str, int]` | Compact tất cả theory. |
| `compact_if_needed(keep_versions=10, max_versions_before_compact=20) → Dict[str, int]` | Tự động compact các theory vượt ngưỡng. |
| `compact_older_than(days, keep_versions=1) → Dict[str, int]` | Xoá version cũ hơn N ngày. |
| `get_version_stats(theory_id) → Dict` | Trả về total_versions, oldest/newest version, estimated_size_bytes. |

### Auto-Compact Background Thread

```python
memory.set_auto_compact_enabled(
    enabled=True,
    keep_versions=10,
    check_interval_minutes=60,  # Kiểm tra mỗi 60 phút
)
# ...
memory.disable_auto_compact()
```

- Dùng `threading.Timer` (daemon thread).
- Định kỳ chạy `compact_if_needed` với `max_versions_before_compact = keep_versions * 2`.
- Tự động dừng khi `close()` được gọi.

### Logging

Sử dụng `logging.getLogger(__name__)`:
- `compact_theory`: log số version đã xoá và thời gian chạy.
- `compact_all`, `compact_if_needed`, `compact_older_than`: log tổng quan.
- Auto-compact background: log error nếu exception.

---

## Tests

### Kết quả hiện tại: **161 tests — tất cả pass**

| Module | Số test cũ | Số test mới | Tổng |
|--------|-----------|-------------|------|
| Episodic Memory (L1) | 63 | 0 | 63 |
| Scientific Memory (L6) — Neo4j | 39 | 8 | 47 |
| Semantic Memory (L5) — FAISS | 17 | 34 | 51 |

### Chạy tests

```bash
# Tất cả knowledge tests
python -m pytest tests/knowledge/ -v

# Chỉ semantic memory (không cần Neo4j)
python -m pytest tests/knowledge/test_semantic_memory.py -v

# Chỉ auto-compact tests
python -m pytest tests/knowledge/test_scientific_memory.py -k "compact" -v

# Chỉ episodic tests
python -m pytest tests/knowledge/test_episodic.py -v
```

### Semantic test coverage (51 tests)

| Nhóm | Số test | Mô tả |
|------|---------|-------|
| `TestStoredEmbeddingDataclass` | 2 | auto ID, dict roundtrip |
| `TestCRUD` | 7 | add/get, nonexistent, delete, delete_by_episode, delete_all |
| `TestFAISS` | 4 | init, search order, rebuild after delete, empty after delete_all |
| `TestBatchInsert` | 4 | basic batch, retrievable, empty list, FAISS sync |
| `TestSearch` | 5 | top-k, filter, min_similarity, empty DB, zero vector |
| `TestKeywordSearch` | 5 | FTS match, filter, no match, empty query, LIKE fallback |
| `TestHybridSearch` | 4 | results, filter, empty DB, zero weights |
| `TestSearchByText` | 2 | text search, missing model |
| `TestEpisodeIntegration` | 4 | add_from_episode (2x), sync_episode noop, auto_sync flag |
| `TestExportImport` | 4 | legacy JSON, empty JSON, directory export/import, no FAISS index |
| `TestStats` | 2 | count, get_stats |
| `TestContextManager` | 1 | context manager |
| `TestFAISSFallback` | 2 | fallback search, fallback correctness |
| `TestFTSFallback` | 1 | FTS unavailable → LIKE |
| `TestEdgeCases` | 5 | zero vector norm, unit vector norm, empty hybrid, metadata default, nonexistent delete |

### Auto-compact test coverage (8 tests)

| Test | Mô tả |
|------|-------|
| `test_compact_keeps_latest_versions` | Giữ 5 version từ 20 → xoá 15 |
| `test_compact_nonexistent_returns_zero` | Theory không tồn tại |
| `test_compact_enforces_minimum_one` | keep_versions=0 → giữ 1 |
| `test_compact_when_below_threshold` | 3 versions, keep=10 → không xoá |
| `test_compact_all` | Compact tất cả (2/3 theories) |
| `test_compact_if_needed_*` | 3 tests: trigger over threshold, noop under, multiple theories |
| `test_compact_older_than` | Xoá theo thời gian (backdated timestamps) |
| `test_compact_older_than_keeps_minimum` | Giữ N version dù cũ |
| `test_get_version_stats*` | 2 tests: stats đúng, nonexistent |
| `test_auto_compact_enable_disable` | Bật/tắt background thread |

---

## Các cải thiện gần đây

### Semantic Memory

1. **FAISS backend** — Chuyển từ numpy loop sang FAISS IndexFlatIP, tăng tốc search ~100x. Fallback về numpy nếu FAISS không có.
2. **Batch insert** — `add_embeddings_batch()` dùng 1 transaction SQL + 1 FAISS add.
3. **FTS5 keyword search** — Tìm kiếm full-text với BM25. Fallback về SQL LIKE nếu FTS5 không available.
4. **Hybrid search** — Kết hợp keyword + vector với trọng số tuỳ chỉnh.
5. **Episode integration** — `add_from_episode()` tự động tạo 3 embeddings (prompt, response, summary). `sync_episode()` đồng bộ với Episodic Memory. `auto_sync_episodes` flag.
6. **Enhanced export** — Export ra thư mục chứa metadata JSON + FAISS index.
7. **34 tests mới** — FAISS, batch, hybrid, FTS, episode sync, fallback, edge cases.

### Scientific Memory

1. **compact_if_needed** — Tự động compact khi theory vượt ngưỡng `max_versions_before_compact`.
2. **compact_older_than** — Xoá version cũ hơn N ngày (dựa trên `updated_at`).
3. **get_version_stats** — Monitoring: total versions, oldest/newest, estimated size.
4. **Background auto-compact** — `set_auto_compact_enabled()` dùng `threading.Timer` daemon thread.
5. **Logging** — Ghi log số version xoá, thời gian chạy, lỗi.
6. **8 tests mới** — Trigger, older_than, stats, background thread toggle.
