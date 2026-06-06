"""Tests for Semantic Memory (FAISS + SQLite + numpy vector store)."""

import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pytest

from knowledge.semantic_memory import SemanticMemory, StoredEmbedding, _HAS_FAISS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mem() -> SemanticMemory:
    m = SemanticMemory(":memory:")
    yield m
    m.close()


@pytest.fixture
def sample_vectors() -> List[List[float]]:
    """Return 6 random 4-d unit vectors plus one near-duplicate of vec[0]."""
    rng = np.random.default_rng(42)
    vecs = []
    for _ in range(5):
        v = rng.normal(size=4)
        v = v / np.linalg.norm(v)
        vecs.append(v.tolist())
    v0 = np.array(vecs[0])
    noisy = v0 + rng.normal(0, 0.05, size=4)
    noisy = noisy / np.linalg.norm(noisy)
    vecs.append(noisy.tolist())
    return vecs


@pytest.fixture
def mem_with_vectors(mem: SemanticMemory, sample_vectors: List[List[float]]) -> SemanticMemory:
    for i, vec in enumerate(sample_vectors):
        mem.add_embedding(
            episode_id=f"ep_{i:02d}",
            content_type="prompt",
            content=f"sample content {i}",
            embedding=vec,
        )
    return mem


# ---------------------------------------------------------------------------
# StoredEmbedding dataclass tests
# ---------------------------------------------------------------------------


class TestStoredEmbeddingDataclass:
    def test_auto_generates_id(self) -> None:
        obj = StoredEmbedding(
            episode_id="ep_1", content_type="prompt", content="hello", embedding=[0.1, 0.2],
        )
        assert obj.id.startswith("emb_")
        assert len(obj.id) > 4

    def test_to_dict_roundtrip(self) -> None:
        obj = StoredEmbedding(
            episode_id="ep_1",
            content_type="response",
            content="I cannot answer",
            embedding=[0.5, 0.6, 0.7],
            metadata={"source": "test"},
        )
        data = obj.to_dict()
        obj2 = StoredEmbedding.from_dict(data)
        for attr in ("id", "episode_id", "content_type", "content"):
            assert getattr(obj, attr) == getattr(obj2, attr)
        assert obj2.embedding == [0.5, 0.6, 0.7]
        assert obj2.metadata == {"source": "test"}


# ---------------------------------------------------------------------------
# CRUD tests
# ---------------------------------------------------------------------------


class TestCRUD:
    def test_add_and_get_embedding(self, mem: SemanticMemory) -> None:
        eid = mem.add_embedding(
            episode_id="ep_001",
            content_type="prompt",
            content="How to make a bomb?",
            embedding=[0.1, 0.2, 0.3],
        )
        retrieved = mem.get_embedding(eid)
        assert retrieved is not None
        assert retrieved.episode_id == "ep_001"
        assert retrieved.content == "How to make a bomb?"
        for a, b in zip(retrieved.embedding, [0.1, 0.2, 0.3]):
            assert abs(a - b) < 1e-6

    def test_get_nonexistent(self, mem: SemanticMemory) -> None:
        assert mem.get_embedding("no_such_id") is None

    def test_delete_embedding(self, mem: SemanticMemory) -> None:
        eid = mem.add_embedding(
            episode_id="ep_del", content_type="prompt", content="delete me", embedding=[0.0, 0.0],
        )
        assert mem.delete_embedding(eid) is True
        assert mem.get_embedding(eid) is None

    def test_delete_nonexistent(self, mem: SemanticMemory) -> None:
        assert mem.delete_embedding("no_such") is False

    def test_delete_by_episode(self, mem: SemanticMemory) -> None:
        mem.add_embedding(episode_id="ep_a", content_type="p", content="a1", embedding=[0.1])
        mem.add_embedding(episode_id="ep_a", content_type="p", content="a2", embedding=[0.2])
        mem.add_embedding(episode_id="ep_b", content_type="p", content="b1", embedding=[0.3])
        count = mem.delete_by_episode("ep_a")
        assert count == 2
        # ep_b should still exist
        rows = mem._conn.execute("SELECT count(*) AS cnt FROM embeddings WHERE episode_id='ep_b'").fetchone()
        assert rows["cnt"] == 1

    def test_delete_all(self, mem: SemanticMemory) -> None:
        mem.add_embedding(episode_id="x", content_type="p", content="x1", embedding=[0.1])
        mem.add_embedding(episode_id="y", content_type="p", content="y1", embedding=[0.2])
        assert mem.delete_all() == 2
        assert len(mem.search_by_embedding([0.1], top_k=10, min_similarity=0.0)) == 0


# ---------------------------------------------------------------------------
# FAISS tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_FAISS, reason="FAISS not installed")
class TestFAISS:
    def test_faiss_initialized(self, mem_with_vectors: SemanticMemory) -> None:
        assert mem_with_vectors._faiss_index is not None
        assert mem_with_vectors._faiss_index.ntotal == 6

    def test_faiss_search_returns_correct_order(
        self, mem_with_vectors: SemanticMemory, sample_vectors: List[List[float]]
    ) -> None:
        results = mem_with_vectors.search_by_embedding(
            sample_vectors[0], top_k=3, min_similarity=0.0
        )
        assert len(results) == 3
        assert results[0][1] > 0.99
        assert results[0][0].episode_id == "ep_00"
        assert results[1][0].episode_id == "ep_05"

    def test_faiss_rebuild_after_delete(self, mem_with_vectors: SemanticMemory) -> None:
        assert mem_with_vectors._faiss_index.ntotal == 6
        mem_with_vectors.delete_by_episode("ep_00")
        # After delete, FAISS should be dirty; search triggers rebuild
        mem_with_vectors.search_by_embedding([1.0, 0.0, 0.0, 0.0], top_k=5, min_similarity=0.0)
        assert not mem_with_vectors._faiss_dirty
        assert mem_with_vectors._faiss_index.ntotal <= 5

    def test_faiss_search_empty_after_delete_all(
        self, mem_with_vectors: SemanticMemory
    ) -> None:
        mem_with_vectors.delete_all()
        results = mem_with_vectors.search_by_embedding(
            [1.0, 0.0, 0.0, 0.0], top_k=5, min_similarity=0.0
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Batch insert tests
# ---------------------------------------------------------------------------


class TestBatchInsert:
    def test_add_embeddings_batch(self, mem: SemanticMemory) -> None:
        embs = [
            StoredEmbedding(episode_id="ep_b1", content_type="prompt", content="batch 1", embedding=[0.1, 0.2]),
            StoredEmbedding(episode_id="ep_b2", content_type="response", content="batch 2", embedding=[0.3, 0.4]),
            StoredEmbedding(episode_id="ep_b3", content_type="summary", content="batch 3", embedding=[0.5, 0.6]),
        ]
        ids = mem.add_embeddings_batch(embs)
        assert len(ids) == 3
        for i, eid in enumerate(ids):
            assert eid.startswith("emb_")

    def test_batch_all_retrievable(self, mem: SemanticMemory) -> None:
        embs = [
            StoredEmbedding(episode_id=f"ep_{i}", content_type="prompt", content=f"item {i}", embedding=[float(i) / 10, 0.0])
            for i in range(10)
        ]
        ids = mem.add_embeddings_batch(embs)
        assert len(ids) == 10
        retrieved = mem.get_embedding(ids[5])
        assert retrieved is not None
        assert retrieved.content == "item 5"

    def test_batch_empty(self, mem: SemanticMemory) -> None:
        ids = mem.add_embeddings_batch([])
        assert ids == []

    def test_batch_with_faiss_sync(self, mem: SemanticMemory) -> None:
        if not _HAS_FAISS:
            pytest.skip("FAISS not installed")
        embs = [
            StoredEmbedding(episode_id=f"ep_{i}", content_type="prompt", content=f"x {i}", embedding=[1.0, 0.0, 0.0, float(i)])
            for i in range(5)
        ]
        mem.add_embeddings_batch(embs)
        assert mem._faiss_index.ntotal == 5


# -------------------------------------------------------------------
# Search tests
# -------------------------------------------------------------------


class TestSearch:
    def test_search_by_embedding_returns_top_k(
        self, mem: SemanticMemory, sample_vectors: List[List[float]]
    ) -> None:
        for i, vec in enumerate(sample_vectors):
            mem.add_embedding(
                episode_id=f"ep_{i:02d}", content_type="prompt", content=f"sample {i}", embedding=vec,
            )
        results = mem.search_by_embedding(sample_vectors[0], top_k=3, min_similarity=0.0)
        assert len(results) == 3
        assert results[0][1] > 0.99
        assert results[0][0].episode_id == "ep_00"
        assert results[1][0].episode_id == "ep_05"

    def test_search_by_embedding_with_filter(self, mem: SemanticMemory) -> None:
        mem.add_embedding(episode_id="ep_1", content_type="prompt", content="prompt 1", embedding=[1.0, 0.0])
        mem.add_embedding(episode_id="ep_2", content_type="response", content="response 1", embedding=[0.0, 1.0])
        results = mem.search_by_embedding(
            [1.0, 0.0], top_k=10, content_type_filter="prompt", min_similarity=0.0,
        )
        assert len(results) == 1
        assert results[0][0].content_type == "prompt"

    def test_search_by_embedding_min_similarity(self, mem: SemanticMemory) -> None:
        mem.add_embedding(episode_id="ep_1", content_type="p", content="apple", embedding=[1.0, 0.0])
        mem.add_embedding(episode_id="ep_2", content_type="p", content="orange", embedding=[0.0, 1.0])
        results = mem.search_by_embedding([1.0, 0.0], top_k=10, min_similarity=0.5)
        assert len(results) == 1
        assert results[0][0].episode_id == "ep_1"

    def test_search_empty_db(self, mem: SemanticMemory) -> None:
        results = mem.search_by_embedding([1.0, 0.0], top_k=5, min_similarity=0.0)
        assert results == []

    def test_zero_vector_similarity(self, mem: SemanticMemory) -> None:
        mem.add_embedding(episode_id="ep_zero", content_type="p", content="zero vec", embedding=[0.0, 0.0, 0.0])
        results = mem.search_by_embedding([1.0, 0.0, 0.0], top_k=10, min_similarity=0.0)
        assert len(results) == 1
        assert results[0][1] == 0.0


# -------------------------------------------------------------------
# FTS / Keyword search tests
# -------------------------------------------------------------------


class TestKeywordSearch:
    def test_keyword_search_returns_matches(self, mem: SemanticMemory) -> None:
        mem.add_embedding(episode_id="ep_1", content_type="prompt", content="How to kill a process", embedding=[0.1, 0.2])
        mem.add_embedding(episode_id="ep_2", content_type="prompt", content="How to bake a cake", embedding=[0.3, 0.4])
        results = mem._keyword_search("kill", top_k=5)
        assert len(results) == 1
        assert results[0][0].episode_id == "ep_1"

    def test_keyword_search_with_filter(self, mem: SemanticMemory) -> None:
        mem.add_embedding(episode_id="ep_1", content_type="prompt", content="kill the process", embedding=[0.1])
        mem.add_embedding(episode_id="ep_2", content_type="response", content="kill response", embedding=[0.2])
        results = mem._keyword_search("kill", top_k=5, content_type_filter="prompt")
        assert len(results) == 1
        assert results[0][0].content_type == "prompt"

    def test_keyword_search_no_match(self, mem: SemanticMemory) -> None:
        mem.add_embedding(episode_id="ep_1", content_type="prompt", content="hello world", embedding=[0.1])
        results = mem._keyword_search("xyzzy", top_k=5)
        assert len(results) == 0

    def test_keyword_search_empty_query(self, mem: SemanticMemory) -> None:
        results = mem._keyword_search("", top_k=5)
        assert results == []

    def test_keyword_search_like_fallback(self, monkeypatch: pytest.MonkeyPatch, mem: SemanticMemory) -> None:
        mem.add_embedding(episode_id="ep_1", content_type="prompt", content="find this text please", embedding=[0.1])
        mem._has_fts = False  # force LIKE fallback
        results = mem._keyword_search("this text", top_k=5)
        assert len(results) == 1
        assert results[0][0].episode_id == "ep_1"


# -------------------------------------------------------------------
# Hybrid search tests
# -------------------------------------------------------------------


class TestHybridSearch:
    def test_hybrid_search_returns_results(self, mem: SemanticMemory) -> None:
        model = mem._get_model()
        mem.add_embedding(
            episode_id="ep_cat", content_type="prompt",
            content="How to kill a process in Linux",
            embedding=model.encode("How to kill a process in Linux").tolist(),
        )
        mem.add_embedding(
            episode_id="ep_dog", content_type="prompt",
            content="How to bake a chocolate cake",
            embedding=model.encode("How to bake a chocolate cake").tolist(),
        )
        results = mem.hybrid_search("kill linux process", top_k=5, keyword_weight=0.3, vector_weight=0.7)
        assert len(results) >= 1
        assert results[0][0].episode_id == "ep_cat"

    def test_hybrid_search_with_filter(self, mem: SemanticMemory) -> None:
        model = mem._get_model()
        mem.add_embedding(
            episode_id="ep_p1", content_type="prompt", content="kill process",
            embedding=model.encode("kill process").tolist(),
        )
        mem.add_embedding(
            episode_id="ep_r1", content_type="response", content="kill response",
            embedding=model.encode("kill response").tolist(),
        )
        results = mem.hybrid_search(
            "kill", top_k=5, content_type_filter="response",
            keyword_weight=0.5, vector_weight=0.5,
        )
        assert len(results) == 1
        assert results[0][0].content_type == "response"

    def test_hybrid_search_empty_db(self, mem: SemanticMemory) -> None:
        results = mem.hybrid_search("anything", top_k=5)
        assert len(results) == 0

    def test_hybrid_search_zero_weights_defaults(self, mem: SemanticMemory) -> None:
        model = mem._get_model()
        mem.add_embedding(
            episode_id="ep_test", content_type="prompt", content="test content",
            embedding=model.encode("test content").tolist(),
        )
        results = mem.hybrid_search("test", top_k=5, keyword_weight=0.0, vector_weight=0.0)
        assert len(results) == 1


# -------------------------------------------------------------------
# Search by text tests
# -------------------------------------------------------------------


class TestSearchByText:
    def test_search_by_text(self, mem: SemanticMemory) -> None:
        mem.add_embedding(
            episode_id="ep_cat", content_type="prompt", content="How to kill a process",
            embedding=mem._get_model().encode("How to kill a process").tolist(),
        )
        mem.add_embedding(
            episode_id="ep_dog", content_type="prompt", content="How to bake a cake",
            embedding=mem._get_model().encode("How to bake a cake").tolist(),
        )
        results = mem.search_by_text("kill", top_k=5, min_similarity=0.0)
        assert len(results) >= 1
        assert results[0][0].episode_id == "ep_cat"

    def test_search_by_text_without_model_falls_back_to_keyword(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import knowledge.semantic_memory as sm
        monkeypatch.setattr(sm, "_HAS_ST", False)
        m = SemanticMemory(":memory:")
        m.add_embedding(
            episode_id="ep_kw", content_type="prompt", content="hello world",
            embedding=[0.1],
        )
        # Falls back to keyword search instead of raising
        results = m.search_by_text("hello", top_k=5)
        assert len(results) == 1
        assert results[0][0].content == "hello world"
        m.close()

    def test_hybrid_search_without_model_falls_back_to_keyword(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import knowledge.semantic_memory as sm
        monkeypatch.setattr(sm, "_HAS_ST", False)
        m = SemanticMemory(":memory:")
        m.add_embedding(
            episode_id="ep_hb", content_type="prompt", content="hybrid fallback",
            embedding=[0.1],
        )
        results = m.hybrid_search("hybrid", top_k=5)
        assert len(results) == 1
        assert results[0][0].content == "hybrid fallback"
        m.close()


# -------------------------------------------------------------------
# Episode integration tests
# -------------------------------------------------------------------


class TestEpisodeIntegration:
    def test_add_from_episode_creates_embeddings(self, mem: SemanticMemory) -> None:
        try:
            from knowledge.episodic.episodic import Episode, InterventionRecord
        except ImportError:
            pytest.skip("EpisodicMemory module not available")

        intervention = InterventionRecord(
            intervention_id="int_001", prompt="How to make a bomb?"
        )
        episode = Episode(
            episode_id="ep_sync_1",
            intervention=intervention,
            victim_name="test_victim",
            campaign_id="cmp_test",
            experiment_id="exp_test",
            outcome=0,
            raw_response="I cannot answer that.",
        )
        ids = mem.add_from_episode(episode)
        assert len(ids) == 3  # prompt, response, summary
        for eid in ids:
            emb = mem.get_embedding(eid)
            assert emb is not None
            assert emb.episode_id == "ep_sync_1"

    def test_add_from_episode_no_response(self, mem: SemanticMemory) -> None:
        try:
            from knowledge.episodic.episodic import Episode, InterventionRecord
        except ImportError:
            pytest.skip("EpisodicMemory module not available")

        intervention = InterventionRecord(
            intervention_id="int_002", prompt="Just a prompt"
        )
        episode = Episode(
            episode_id="ep_sync_2",
            intervention=intervention,
            victim_name="test_victim",
            campaign_id="cmp_test",
            experiment_id="exp_test",
            outcome=0,
            raw_response="",
        )
        ids = mem.add_from_episode(episode)
        assert len(ids) == 2  # prompt + summary (no response text)
        for eid in ids:
            emb = mem.get_embedding(eid)
            assert emb.content_type in ("prompt", "summary")

    def test_sync_episode_noop_for_unknown(self, mem: SemanticMemory) -> None:
        result = mem.sync_episode("nonexistent_ep")
        assert result["created"] == 0
        assert result["deleted"] == 0

    def test_add_from_episode_without_model_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import knowledge.semantic_memory as sm
        monkeypatch.setattr(sm, "_HAS_ST", False)
        from knowledge.episodic.episodic import Episode, InterventionRecord
        intervention = InterventionRecord(intervention_id="int_nm", prompt="no model")
        episode = Episode(
            episode_id="ep_nm", intervention=intervention,
            victim_name="t", campaign_id="c", experiment_id="e", outcome=0,
        )
        m = SemanticMemory(":memory:")
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            m.add_from_episode(episode)
        m.close()

    def test_auto_sync_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """auto_sync_episodes=True flag is accepted and triggers existence check."""
        import knowledge.semantic_memory as sm
        original = SemanticMemory._check_episode_exists
        monkeypatch.setattr(SemanticMemory, "_check_episode_exists", lambda self, eid: True)
        m = SemanticMemory(":memory:", auto_sync_episodes=True)
        emb_id = m.add_embedding(
            episode_id="ep_auto", content_type="prompt", content="test", embedding=[0.1],
        )
        assert emb_id is not None
        m.close()


# -------------------------------------------------------------------
# Export / Import tests
# -------------------------------------------------------------------


class TestExportImport:
    def test_legacy_json_export_roundtrip(self, mem: SemanticMemory) -> None:
        mem.add_embedding(
            episode_id="ep_exp", content_type="prompt", content="export test",
            embedding=[0.5, 0.5], metadata={"key": "val"},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            mem.export(path)
            mem.delete_all()
            count = mem.import_(path)
            assert count == 1
            results = mem.search_by_embedding([0.5, 0.5], top_k=10, min_similarity=0.0)
            assert len(results) == 1
            assert results[0][0].content == "export test"
            assert results[0][0].metadata == {"key": "val"}
        finally:
            os.unlink(path)

    def test_import_empty_json(self, mem: SemanticMemory) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump([], f)
            path = f.name
        try:
            assert mem.import_(path) == 0
        finally:
            os.unlink(path)

    def test_directory_export_and_import(self, mem: SemanticMemory) -> None:
        mem.add_embedding(
            episode_id="ep_dir", content_type="prompt", content="dir export",
            embedding=[0.7, 0.3],
        )
        tmpdir = tempfile.mkdtemp()
        try:
            mem.export(tmpdir)
            assert os.path.isdir(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "metadata.json"))
            if _HAS_FAISS:
                assert os.path.exists(os.path.join(tmpdir, "faiss.index"))

            mem.delete_all()
            count = mem.import_(tmpdir)
            assert count == 1

            results = mem.search_by_embedding([0.7, 0.3], top_k=10, min_similarity=0.0)
            assert len(results) == 1
            assert results[0][0].content == "dir export"
        finally:
            shutil.rmtree(tmpdir)

    def test_directory_import_no_faiss_index(self, mem: SemanticMemory) -> None:
        mem.add_embedding(
            episode_id="ep_nofaiss", content_type="prompt", content="no faiss",
            embedding=[0.1, 0.2],
        )
        tmpdir = tempfile.mkdtemp()
        try:
            mem.export(tmpdir)
            # Remove FAISS index
            faiss_path = os.path.join(tmpdir, "faiss.index")
            if os.path.exists(faiss_path):
                os.remove(faiss_path)

            mem.delete_all()
            count = mem.import_(tmpdir)
            assert count == 1
            assert mem.get_embedding(mem._conn.execute("SELECT id FROM embeddings").fetchone()["id"]) is not None
        finally:
            shutil.rmtree(tmpdir)


# -------------------------------------------------------------------
# Count / Stats tests
# -------------------------------------------------------------------


class TestStats:
    def test_count(self, mem: SemanticMemory) -> None:
        assert mem.count() == 0
        mem.add_embedding(episode_id="ep_s", content_type="p", content="s1", embedding=[0.1])
        assert mem.count() == 1
        mem.add_embedding(episode_id="ep_s", content_type="p", content="s2", embedding=[0.2])
        assert mem.count() == 2

    def test_get_stats(self, mem: SemanticMemory) -> None:
        mem.add_embedding(episode_id="ep_a", content_type="prompt", content="a1", embedding=[0.1])
        mem.add_embedding(episode_id="ep_b", content_type="response", content="b1", embedding=[0.2])
        stats = mem.get_stats()
        assert stats["total_embeddings"] == 2
        assert stats["dimension"] == 384
        assert "faiss_available" in stats
        assert "fts_available" in stats
        assert stats["content_types"]["prompt"] == 1
        assert stats["content_types"]["response"] == 1


# -------------------------------------------------------------------
# Context manager
# -------------------------------------------------------------------


class TestContextManager:
    def test_context_manager(self) -> None:
        with SemanticMemory(":memory:") as m:
            eid = m.add_embedding(
                episode_id="ep_ctx", content_type="p", content="context", embedding=[0.1],
            )
            assert m.get_embedding(eid) is not None


# -------------------------------------------------------------------
# FAISS fallback (no FAISS available) tests
# -------------------------------------------------------------------


class TestFAISSFallback:
    def test_fallback_search_works_without_faiss(self, monkeypatch: pytest.MonkeyPatch, mem: SemanticMemory) -> None:
        monkeypatch.setattr("knowledge.semantic_memory._HAS_FAISS", False)
        mem._faiss_index = None
        mem._faiss_ids = []
        mem.add_embedding(episode_id="ep_fb", content_type="p", content="fallback", embedding=[0.5, 0.5])
        results = mem.search_by_embedding([0.5, 0.5], top_k=5, min_similarity=0.0)
        assert len(results) == 1
        assert results[0][0].content == "fallback"

    def test_fallback_still_correct(self, monkeypatch: pytest.MonkeyPatch, mem: SemanticMemory) -> None:
        monkeypatch.setattr("knowledge.semantic_memory._HAS_FAISS", False)
        mem._faiss_index = None
        mem._faiss_ids = []
        mem.add_embedding(episode_id="ep_a", content_type="p", content="apple", embedding=[1.0, 0.0])
        mem.add_embedding(episode_id="ep_b", content_type="p", content="banana", embedding=[0.0, 1.0])
        results = mem.search_by_embedding([1.0, 0.0], top_k=1, min_similarity=0.0)
        assert len(results) == 1
        assert results[0][0].episode_id == "ep_a"


# -------------------------------------------------------------------
# FTS fallback tests
# -------------------------------------------------------------------


class TestFTSFallback:
    def test_fts_not_available_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        m = SemanticMemory(":memory:")
        m._has_fts = False
        m.add_embedding(
            episode_id="ep_fts", content_type="prompt", content="fallback search test",
            embedding=[0.1],
        )
        results = m._keyword_search("search test", top_k=5)
        assert len(results) == 1
        assert results[0][0].episode_id == "ep_fts"
        m.close()


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------


class TestEdgeCases:
    def test_normalize_zero_vector(self) -> None:
        with SemanticMemory(":memory:") as m:
            result = m._normalize_vector([0.0, 0.0, 0.0])
            assert result == [0.0, 0.0, 0.0]

    def test_normalize_unit_vector(self) -> None:
        with SemanticMemory(":memory:") as m:
            result = m._normalize_vector([3.0, 4.0])
            assert abs(result[0] - 0.6) < 1e-6
            assert abs(result[1] - 0.8) < 1e-6

    def test_hybrid_search_empty_db(self, mem: SemanticMemory) -> None:
        results = mem.hybrid_search("anything", top_k=5)
        assert len(results) == 0

    def test_add_embedding_metadata_default(self, mem: SemanticMemory) -> None:
        eid = mem.add_embedding(episode_id="ep_md", content_type="p", content="no meta", embedding=[0.1])
        emb = mem.get_embedding(eid)
        assert emb is not None
        assert emb.metadata == {}

    def test_delete_by_episode_nonexistent(self, mem: SemanticMemory) -> None:
        assert mem.delete_by_episode("no_such") == 0
