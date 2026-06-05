"""Semantic Memory — FAISS + SQLite vector store for HARMONY-X.

L5 in the hierarchical memory architecture. Stores embeddings of prompts,
responses, and episode summaries, providing fast similarity search via
FAISS (falling back to numpy+SQLite when FAISS is unavailable).

Key features:
- FAISS index for approximate/ exact nearest neighbour search.
- FTS5-based keyword search (falls back to SQL LIKE).
- Hybrid search combining keyword + vector scores.
- Batch insertion for bulk imports.
- ``add_from_episode()`` to auto-embed an ``Episode`` object.
- ``sync_episode()`` to stay in sync with Episodic Memory.
- Enhanced export (directory with FAISS index + JSON metadata).

Requires: ``numpy``.  Optional: ``faiss``, ``sentence-transformers``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import faiss  # type: ignore[import-untyped]

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    _HAS_ST = True
except ImportError:
    _HAS_ST = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class StoredEmbedding:
    """An embedding stored in Semantic Memory, linked to an episode.

    Attributes:
        episode_id: Foreign-key reference to an episode in Episodic Memory.
        content_type: ``"prompt"`` | ``"response"`` | ``"summary"`` | custom.
        content: Original text that was embedded.
        embedding: Dense vector as a list of floats.
        metadata: Arbitrary key-value metadata.
        id: Auto-generated unique identifier (``emb_<hex>``).
        created_at: Unix timestamp of creation.
    """

    episode_id: str
    content_type: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"emb_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "content_type": self.content_type,
            "content": self.content,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredEmbedding":
        return cls(
            id=data.get("id", ""),
            episode_id=data.get("episode_id", ""),
            content_type=data.get("content_type", ""),
            content=data.get("content", ""),
            embedding=list(data.get("embedding", [])),
            created_at=float(data.get("created_at", time.time())),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# FAISS + SQLite store
# ---------------------------------------------------------------------------


class SemanticMemory:
    """Vector-embedding store backed by FAISS (primary) + SQLite (metadata).

    Parameters:
        db_path: Path to the SQLite database (``:memory:`` for in-memory,
            useful in tests).
        model_name: Sentence-Transformer model name.  If ``None``,
            ``search_by_text`` and ``hybrid_search`` will raise an error.
        auto_sync_episodes: When ``True``, every ``add_embedding`` checks
            that the referenced ``episode_id`` exists in Episodic Memory.
        dim: Dimensionality of the embedding vectors.  Auto-detected from
            the first embedding added when possible.
    """

    def __init__(
        self,
        db_path: str = "semantic_memory.db",
        model_name: Optional[str] = None,
        auto_sync_episodes: bool = False,
        dim: int = 384,
    ) -> None:
        self.db_path = db_path
        self._model_name = model_name
        self._model: Any = None
        self._auto_sync = auto_sync_episodes
        self._dim = dim

        parent = Path(db_path).parent
        if str(parent) != "." and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._init_fts()

        # FAISS state
        self._faiss_index: Any = None
        self._faiss_ids: List[str] = []  # position → embedding id
        self._faiss_dirty: bool = False
        if _HAS_FAISS:
            self._rebuild_faiss()
        else:
            logger.warning(
                "FAISS is not installed — falling back to numpy cosine "
                "similarity.  Install with: pip install faiss-cpu"
            )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "SemanticMemory":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                id           TEXT PRIMARY KEY,
                episode_id   TEXT NOT NULL,
                content_type TEXT NOT NULL,
                content      TEXT NOT NULL,
                embedding    BLOB,
                created_at   REAL NOT NULL,
                metadata     TEXT NOT NULL DEFAULT '{}'
            )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sm_episode "
            "ON embeddings(episode_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sm_content_type "
            "ON embeddings(content_type)"
        )
        self._conn.commit()

    def _init_fts(self) -> None:
        """Initialise FTS5 virtual table for keyword search.

        Falls back gracefully if FTS5 is not available in the SQLite build.
        """
        self._has_fts = False
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_fts "
                "USING fts5(id UNINDEXED, content, content_type UNINDEXED)"
            )
            self._has_fts = True
        except sqlite3.OperationalError:
            logger.info("FTS5 not available — using LIKE for keyword search")

    # ------------------------------------------------------------------
    # FAISS helpers
    # ------------------------------------------------------------------

    def _ensure_faiss(self, dim: Optional[int] = None) -> None:
        """Ensure the FAISS index exists, optionally with a given dimension.

        If the index already exists and *dim* does not match, the index
        is rebuilt from scratch.
        """
        if not _HAS_FAISS:
            return
        if self._faiss_index is not None:
            if dim is not None and self._faiss_index.d != dim:
                # Dimension mismatch — rebuild
                self._rebuild_faiss()
            return
        if dim is not None:
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_ids = []
        else:
            self._rebuild_faiss()

    def _rebuild_faiss(self) -> None:
        """Rebuild the FAISS index from all vectors currently in SQLite."""
        if not _HAS_FAISS:
            return
        rows = self._conn.execute(
            "SELECT id, embedding FROM embeddings"
        ).fetchall()
        if not rows:
            self._faiss_index = None
            self._faiss_ids = []
            self._faiss_dirty = False
            return

        dim = len(self._vector_from_blob(rows[0]["embedding"]))
        vectors: List[np.ndarray] = []
        ids: List[str] = []
        for row in rows:
            vec = self._vector_from_blob(row["embedding"])
            vectors.append(np.array(vec, dtype=np.float32))
            ids.append(row["id"])

        self._dim = dim
        vecs = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vecs)
        self._faiss_index = faiss.IndexFlatIP(self._dim)
        self._faiss_index.add(vecs)
        self._faiss_ids = ids
        self._faiss_dirty = False

    def _mark_faiss_dirty(self) -> None:
        self._faiss_dirty = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Lazy-load the Sentence-Transformer model."""
        if self._model is not None:
            return self._model
        if not _HAS_ST:
            raise RuntimeError(
                "sentence-transformers is not installed; "
                "pip install sentence-transformers"
            )
        name = self._model_name or "all-MiniLM-L6-v2"
        logger.info("Loading sentence-transformer model: %s", name)
        self._model = SentenceTransformer(name)  # type: ignore[assignment]
        return self._model

    @staticmethod
    def _blob_from_vector(vec: List[float]) -> bytes:
        return np.array(vec, dtype=np.float32).tobytes()

    @staticmethod
    def _vector_from_blob(blob: bytes) -> List[float]:
        return np.frombuffer(blob, dtype=np.float32).tolist()

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        arr_a = np.array(a, dtype=np.float32)
        arr_b = np.array(b, dtype=np.float32)
        denom = np.linalg.norm(arr_a) * np.linalg.norm(arr_b)
        if denom == 0.0:
            return 0.0
        return float(np.dot(arr_a, arr_b) / denom)

    @staticmethod
    def _normalize_vector(vec: List[float]) -> List[float]:
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            return vec
        return (arr / norm).tolist()

    def _row_to_embedding(self, row: sqlite3.Row) -> StoredEmbedding:
        return StoredEmbedding(
            id=row["id"],
            episode_id=row["episode_id"],
            content_type=row["content_type"],
            content=row["content"],
            embedding=self._vector_from_blob(row["embedding"]),
            created_at=float(row["created_at"]),
            metadata=json.loads(row["metadata"]),
        )

    def _check_episode_exists(self, episode_id: str) -> bool:
        """Check whether the episode exists in Episodic Memory

        Uses lazy import to avoid circular dependencies.
        Returns ``True`` when the EpisodicMemory module is unavailable
        (fail-open) so embeddings are never blocked by a missing module.
        """
        try:
            from knowledge.episodic.episodic import EpisodicMemory  # type: ignore[import-untyped]

            db_dir = os.path.dirname(os.path.abspath(self.db_path)) if self.db_path != ":memory:" else "."
            em = EpisodicMemory(db_path=os.path.join(db_dir, "episodic_memory.db"))
            exists = em.episode_exists(episode_id)
            em.close()
            return exists
        except Exception:
            return True  # fail-open

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_embedding(
        self,
        episode_id: str,
        content_type: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an embedding. Returns its id."""
        if self._auto_sync and not self._check_episode_exists(episode_id):
            raise ValueError(
                f"Episode {episode_id} does not exist in Episodic Memory "
                "(auto_sync_episodes=True)"
            )

        obj = StoredEmbedding(
            episode_id=episode_id,
            content_type=content_type,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )
        blob = self._blob_from_vector(embedding)
        self._conn.execute(
            """INSERT INTO embeddings
               (id, episode_id, content_type, content,
                embedding, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                obj.id,
                obj.episode_id,
                obj.content_type,
                obj.content,
                blob,
                obj.created_at,
                json.dumps(obj.metadata, ensure_ascii=False),
            ),
        )
        self._conn.commit()

        # Update FTS
        self._upsert_fts(obj.id, obj.content, obj.content_type)

        # Update FAISS
        if _HAS_FAISS:
            dim = len(embedding)
            self._ensure_faiss(dim=dim)
            if self._faiss_index is not None:
                vec = np.array([self._normalize_vector(embedding)], dtype=np.float32)
                self._faiss_index.add(vec)
                self._faiss_ids.append(obj.id)

        return obj.id

    def add_embeddings_batch(
        self,
        embeddings: List[StoredEmbedding],
    ) -> List[str]:
        """Insert multiple embeddings in a single transaction.

        This is significantly faster than calling ``add_embedding`` in a
        loop when importing large amounts of data.

        Returns the list of generated ids in the same order.
        """
        ids: List[str] = []

        if self._auto_sync:
            for emb in embeddings:
                if not self._check_episode_exists(emb.episode_id):
                    raise ValueError(
                        f"Episode {emb.episode_id} does not exist "
                        "(auto_sync_episodes=True)"
                    )

        with self._conn:
            for emb in embeddings:
                emb.__post_init__()
                blob = self._blob_from_vector(emb.embedding)
                self._conn.execute(
                    """INSERT INTO embeddings
                       (id, episode_id, content_type, content,
                        embedding, created_at, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        emb.id,
                        emb.episode_id,
                        emb.content_type,
                        emb.content,
                        blob,
                        emb.created_at,
                        json.dumps(emb.metadata, ensure_ascii=False),
                    ),
                )
                ids.append(emb.id)

        # Batch update FTS
        for emb in embeddings:
            self._upsert_fts(emb.id, emb.content, emb.content_type)

        # Batch update FAISS
        if _HAS_FAISS and embeddings:
            dim = len(embeddings[0].embedding)
            self._ensure_faiss(dim=dim)
            if self._faiss_index is not None:
                vecs = np.array(
                    [self._normalize_vector(e.embedding) for e in embeddings],
                    dtype=np.float32,
                )
                self._faiss_index.add(vecs)
                self._faiss_ids.extend(ids)

        return ids

    def get_embedding(self, embedding_id: str) -> Optional[StoredEmbedding]:
        row = self._conn.execute(
            "SELECT * FROM embeddings WHERE id = ?",
            (embedding_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_embedding(row)

    def delete_embedding(self, embedding_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM embeddings WHERE id = ?",
            (embedding_id,),
        )
        self._conn.commit()
        if cursor.rowcount > 0:
            # Remove from FTS
            self._delete_fts(embedding_id)
            self._mark_faiss_dirty()
            return True
        return False

    def delete_by_episode(self, episode_id: str) -> int:
        """Delete all embeddings linked to an episode. Returns count."""
        ids = self._conn.execute(
            "SELECT id FROM embeddings WHERE episode_id = ?",
            (episode_id,),
        ).fetchall()
        cursor = self._conn.execute(
            "DELETE FROM embeddings WHERE episode_id = ?",
            (episode_id,),
        )
        self._conn.commit()
        if cursor.rowcount > 0:
            for row in ids:
                self._delete_fts(row["id"])
            self._mark_faiss_dirty()
        return cursor.rowcount

    def delete_all(self) -> int:
        cursor = self._conn.execute("DELETE FROM embeddings")
        self._conn.commit()
        self._conn.execute("DELETE FROM embeddings_fts")
        self._conn.commit()
        if cursor.rowcount > 0:
            self._mark_faiss_dirty()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # FTS helpers
    # ------------------------------------------------------------------

    def _upsert_fts(self, emb_id: str, content: str, content_type: str) -> None:
        if not self._has_fts:
            return
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings_fts(id, content, content_type) "
                "VALUES (?, ?, ?)",
                (emb_id, content, content_type),
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # FTS might not be available

    def _delete_fts(self, emb_id: str) -> None:
        if not self._has_fts:
            return
        try:
            self._conn.execute(
                "DELETE FROM embeddings_fts WHERE id = ?",
                (emb_id,),
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass

    def _keyword_search(
        self,
        query_text: str,
        top_k: int = 10,
        content_type_filter: Optional[str] = None,
    ) -> List[Tuple[StoredEmbedding, float]]:
        """Search by keyword using FTS5 or LIKE fallback.

        Returns ``(StoredEmbedding, score)`` pairs.  When using FTS5 the
        score is the BM25 rank; with LIKE fallback it is a simple TF-based
        score (fraction of query terms matched).
        """
        terms = query_text.strip().split()
        if not terms:
            return []

        if self._has_fts:
            return self._keyword_search_fts(
                terms, top_k, content_type_filter
            )
        return self._keyword_search_like(terms, top_k, content_type_filter)

    def _keyword_search_fts(
        self,
        terms: List[str],
        top_k: int,
        content_type_filter: Optional[str],
    ) -> List[Tuple[StoredEmbedding, float]]:
        """Keyword search via FTS5 BM25."""
        fts_query = " OR ".join(terms)
        sql = (
            "SELECT e.*, rank "
            "FROM embeddings_fts f "
            "JOIN embeddings e ON e.id = f.id "
            "WHERE embeddings_fts MATCH ?"
        )
        params: List[Any] = [fts_query]
        if content_type_filter:
            sql += " AND e.content_type = ?"
            params.append(content_type_filter)
        sql += " ORDER BY rank LIMIT ?"
        params.append(top_k)

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return self._keyword_search_like(terms, top_k, content_type_filter)

        results: List[Tuple[StoredEmbedding, float]] = []
        for row in rows:
            emb = self._row_to_embedding(row)
            # BM25 rank is typically negative — negate for a positive score
            score = -row["rank"] if row["rank"] < 0 else 1.0 / (1.0 + row["rank"])
            results.append((emb, score))
        return results

    def _keyword_search_like(
        self,
        terms: List[str],
        top_k: int,
        content_type_filter: Optional[str],
    ) -> List[Tuple[StoredEmbedding, float]]:
        """Keyword search via SQL LIKE + simple TF scoring."""
        conditions = " OR ".join(
            "e.content LIKE ?" for _ in terms
        )
        sql = f"SELECT e.* FROM embeddings e WHERE ({conditions})"
        params: List[str] = [f"%{t}%" for t in terms]
        if content_type_filter:
            sql += " AND e.content_type = ?"
            params.append(content_type_filter)
        sql += " LIMIT ?"
        params.append(str(top_k))

        rows = self._conn.execute(sql, params).fetchall()
        results: List[Tuple[StoredEmbedding, float]] = []
        for row in rows:
            emb = self._row_to_embedding(row)
            content_lower = emb.content.lower()
            matched = sum(1 for t in terms if t.lower() in content_lower)
            score = matched / len(terms) if terms else 0.0
            results.append((emb, score))
        return results

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_numpy(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        content_type_filter: Optional[str] = None,
        min_similarity: float = 0.5,
    ) -> List[Tuple[StoredEmbedding, float]]:
        """Fallback search using numpy cosine similarity (when FAISS is
        unavailable or index needs rebuilding)."""
        if content_type_filter:
            rows = self._conn.execute(
                "SELECT * FROM embeddings WHERE content_type = ?",
                (content_type_filter,),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM embeddings").fetchall()

        scored: List[Tuple[StoredEmbedding, float]] = []
        for row in rows:
            emb = self._row_to_embedding(row)
            sim = self._cosine_similarity(query_embedding, emb.embedding)
            if sim >= min_similarity:
                scored.append((emb, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        content_type_filter: Optional[str] = None,
        min_similarity: float = 0.5,
    ) -> List[Tuple[StoredEmbedding, float]]:
        """Find stored embeddings by cosine similarity to *query_embedding*.

        Uses FAISS when available, falling back to a full numpy scan.

        Returns up to *top_k* ``(StoredEmbedding, score)`` pairs sorted
        by descending similarity.
        """
        # Filter by content_type first to narrow the search
        ids_to_exclude: List[str] = []

        if _HAS_FAISS and self._faiss_index is not None and self._faiss_index.ntotal > 0:
            if self._faiss_dirty:
                self._rebuild_faiss()
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return []
            # FAISS search on normalized vectors
            query_norm = self._normalize_vector(query_embedding)
            query_arr = np.array([query_norm], dtype=np.float32)
            k = min(top_k * 2, self._faiss_index.ntotal)
            scores_arr, indices = self._faiss_index.search(query_arr, k)
            results: List[Tuple[StoredEmbedding, float]] = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self._faiss_ids):
                    continue
                sim = float(scores_arr[0][i])
                if sim < min_similarity:
                    continue
                emb_id = self._faiss_ids[idx]
                row = self._conn.execute(
                    "SELECT * FROM embeddings WHERE id = ?",
                    (emb_id,),
                ).fetchone()
                if row is None:
                    continue
                if content_type_filter and row["content_type"] != content_type_filter:
                    continue
                results.append((self._row_to_embedding(row), sim))
            return results[:top_k]

        # Fallback
        return self._search_numpy(
            query_embedding,
            top_k=top_k,
            content_type_filter=content_type_filter,
            min_similarity=min_similarity,
        )

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        content_type_filter: Optional[str] = None,
        min_similarity: float = 0.5,
    ) -> List[Tuple[StoredEmbedding, float]]:
        """Embed *query_text* with the Sentence-Transformer model, then
        call ``search_by_embedding``."""
        model = self._get_model()
        vec = model.encode(query_text).tolist()  # type: ignore[union-attr]
        return self.search_by_embedding(
            vec,
            top_k=top_k,
            content_type_filter=content_type_filter,
            min_similarity=min_similarity,
        )

    # ------------------------------------------------------------------
    # Hybrid search (keyword + vector)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_text: str,
        top_k: int = 10,
        content_type_filter: Optional[str] = None,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> List[Tuple[StoredEmbedding, float]]:
        """Combine keyword (BM25 / LIKE) and vector (FAISS) search.

        Both score components are normalised to ``[0, 1]`` before being
        combined via weighted sum.

        Returns up to *top_k* ``(StoredEmbedding, combined_score)``
        sorted descending.
        """
        total_weight = keyword_weight + vector_weight
        if total_weight <= 0.0:
            keyword_weight, vector_weight = 0.3, 0.7
            total_weight = 1.0

        # 1. Get vector embedding for the query
        model = self._get_model()
        query_vec = model.encode(query_text).tolist()  # type: ignore[union-attr]

        # 2. Keyword candidates
        kw_results = self._keyword_search(
            query_text,
            top_k=top_k * 2,
            content_type_filter=content_type_filter,
        )

        # 3. Vector candidates
        vec_results = self.search_by_embedding(
            query_vec,
            top_k=top_k * 2,
            content_type_filter=content_type_filter,
            min_similarity=0.0,
        )

        # 4. Build score dicts
        kw_scores: Dict[str, float] = {r[0].id: r[1] for r in kw_results}
        vec_scores: Dict[str, float] = {r[0].id: r[1] for r in vec_results}

        # 5. Union of all candidate ids
        all_ids = set(kw_scores.keys()) | set(vec_scores.keys())

        # 6. For ids only found in keyword, compute vector similarity
        missing_vec = all_ids - set(vec_scores.keys())
        if missing_vec:
            for eid in missing_vec:
                emb = self.get_embedding(eid)
                if emb is not None:
                    vec_scores[eid] = self._cosine_similarity(
                        query_vec, emb.embedding
                    )

        # 7. Normalize keyword scores (min-max)
        if kw_scores:
            min_kw = min(kw_scores.values())
            max_kw = max(kw_scores.values())
            kw_range = max_kw - min_kw
            kw_norm = {
                k: (v - min_kw) / kw_range if kw_range > 0 else 1.0
                for k, v in kw_scores.items()
            }
        else:
            kw_norm = {}

        # 8. Combine
        combined: List[Tuple[StoredEmbedding, float]] = []
        for eid in all_ids:
            kw = kw_norm.get(eid, 0.0) * keyword_weight
            vec = vec_scores.get(eid, 0.0) * vector_weight
            score = (kw + vec) / total_weight
            emb = self.get_embedding(eid)
            if emb is not None:
                combined.append((emb, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    # ------------------------------------------------------------------
    # Episode integration
    # ------------------------------------------------------------------

    def add_from_episode(
        self,
        episode: Any,
        embedding_model: Optional[Any] = None,
    ) -> List[str]:
        """Auto-embed an ``Episode`` object.

        Creates three embeddings:
        - ``"prompt"`` – the intervention prompt.
        - ``"response"`` – the raw response.
        - ``"summary"`` – concatenated prompt + response.

        Args:
            episode: An ``Episode`` instance from ``knowledge.episodic``.
            embedding_model: Optional callable that takes a string and
                returns a vector.  Defaults to the instance's Sentence-
                Transformer model.

        Returns:
            List of created embedding ids ``[prompt_id, response_id, summary_id]``.
        """
        if embedding_model is None:
            embedding_model = self._get_model()

        ep_id = episode.episode_id
        prompt = episode.intervention.prompt
        response = episode.raw_response or ""
        summary = f"{prompt}\n---\n{response}" if response else prompt

        created_ids: List[str] = []

        for ct, text in [
            ("prompt", prompt),
            ("response", response),
            ("summary", summary),
        ]:
            if not text.strip():
                continue
            vec = embedding_model.encode(text).tolist()  # type: ignore[union-attr]
            eid = self.add_embedding(
                episode_id=ep_id,
                content_type=ct,
                content=text,
                embedding=vec,
            )
            created_ids.append(eid)

        return created_ids

    def sync_episode(self, episode_id: str) -> Dict[str, int]:
        """Synchronise embeddings with Episodic Memory.

        - If an episode exists in Episodic Memory but has no embeddings,
          a summary embedding is created.
        - If an episode has been removed from Episodic Memory, its
          embeddings are deleted from Semantic Memory.

        Returns a dict ``{"created": int, "deleted": int}``.
        """
        result: Dict[str, int] = {"created": 0, "deleted": 0}

        try:
            from knowledge.episodic.episodic import EpisodicMemory  # type: ignore[import-untyped]

            db_dir = os.path.dirname(os.path.abspath(self.db_path)) if self.db_path != ":memory:" else "."
            em = EpisodicMemory(db_path=os.path.join(db_dir, "episodic_memory.db"))
        except Exception:
            logger.warning("Cannot sync — EpisodicMemory unavailable")
            return result

        try:
            episode_exists = em.episode_exists(episode_id)
            existing_embeddings = self._conn.execute(
                "SELECT id FROM embeddings WHERE episode_id = ?",
                (episode_id,),
            ).fetchall()

            if not episode_exists and existing_embeddings:
                # Episode was deleted — clean up
                deleted = self.delete_by_episode(episode_id)
                result["deleted"] = deleted
            elif episode_exists and not existing_embeddings:
                # Episode exists but has no embeddings — create summary
                episode = em.get_episode(episode_id)
                if episode is not None:
                    model = self._get_model()
                    text = (
                        episode.intervention.prompt
                        or episode.raw_response
                        or ""
                    )
                    if text.strip():
                        vec = model.encode(text).tolist()  # type: ignore[union-attr]
                        self.add_embedding(
                            episode_id=episode_id,
                            content_type="summary",
                            content=text,
                            embedding=vec,
                        )
                        result["created"] = 1
        finally:
            em.close()

        return result

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export(self, path: str) -> None:
        """Export embeddings (and FAISS index when available).

        *Legacy:* If *path* ends with ``.json``, writes a single JSON file
        (no FAISS index).

        *New:* Otherwise creates a directory containing:
        - ``metadata.json`` — all embedding records.
        - ``faiss.index`` — serialised FAISS index (if FAISS is available).
        """
        p = Path(path)
        if p.suffix == ".json":
            # Legacy single-file export
            rows = self._conn.execute(
                "SELECT * FROM embeddings ORDER BY created_at"
            ).fetchall()
            data = [self._row_to_embedding(r).to_dict() for r in rows]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return

        # New directory-based export
        p.mkdir(parents=True, exist_ok=True)

        rows = self._conn.execute(
            "SELECT * FROM embeddings ORDER BY created_at"
        ).fetchall()
        data = [self._row_to_embedding(r).to_dict() for r in rows]
        with open(p / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if _HAS_FAISS:
            if self._faiss_dirty:
                self._rebuild_faiss()
            if self._faiss_index is not None and self._faiss_index.ntotal > 0:
                faiss.write_index(self._faiss_index, str(p / "faiss.index"))

    def import_(
        self,
        path: str,
    ) -> int:
        """Import embeddings from a directory or JSON file.

        If *path* is a directory, attempts to load ``metadata.json``
        (and ``faiss.index`` if present).

        If *path* is a ``.json`` file, uses the legacy single-file format.

        Returns the number of embeddings imported.
        """
        p = Path(path)

        if p.is_dir():
            return self._import_directory(p)

        # Legacy file-based import
        with open(path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        return self._import_records(raw_list)

    def _import_directory(self, directory: Path) -> int:
        """Import from an export directory (metadata.json + optional faiss.index)."""
        meta_path = directory / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {directory}"
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        count = self._import_records(raw_list)

        # Optionally load the FAISS index
        faiss_path = directory / "faiss.index"
        if _HAS_FAISS and faiss_path.exists():
            try:
                loaded = faiss.read_index(str(faiss_path))
                if loaded.ntotal > 0:
                    # Rebuild from SQLite to ensure consistency
                    self._rebuild_faiss()
                    logger.info(
                        "Loaded FAISS index with %d vectors", loaded.ntotal
                    )
                else:
                    self._rebuild_faiss()
            except Exception as exc:
                logger.warning("Could not load FAISS index: %s", exc)
                self._rebuild_faiss()
        elif _HAS_FAISS:
            self._rebuild_faiss()

        return count

    def _import_records(self, raw_list: List[Dict[str, Any]]) -> int:
        """Insert records from a parsed JSON list (upsert by id)."""
        embeddings = [StoredEmbedding.from_dict(raw) for raw in raw_list]
        if not embeddings:
            return 0
        return len(self.add_embeddings_batch(embeddings))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Total number of stored embeddings."""
        row = self._conn.execute(
            "SELECT count(*) AS cnt FROM embeddings"
        ).fetchone()
        return row["cnt"] if row else 0

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the stored embeddings.

        Includes total count, dimension, FAISS availability, and counts
        per content_type.
        """
        total = self.count()
        type_counts: Dict[str, int] = {}
        for row in self._conn.execute(
            "SELECT content_type, count(*) AS cnt FROM embeddings "
            "GROUP BY content_type"
        ):
            type_counts[row["content_type"]] = row["cnt"]

        faiss_ntotal = 0
        if _HAS_FAISS and self._faiss_index is not None:
            faiss_ntotal = self._faiss_index.ntotal
        return {
            "total_embeddings": total,
            "dimension": self._dim,
            "faiss_available": _HAS_FAISS and self._faiss_index is not None,
            "faiss_ntotal": faiss_ntotal,
            "faiss_dirty": self._faiss_dirty,
            "fts_available": self._has_fts,
            "model_name": self._model_name or "all-MiniLM-L6-v2",
            "content_types": type_counts,
        }
