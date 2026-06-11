"""Tests for embedding-based semantic primitives and centroids."""

import numpy as np
import pytest

from sde.embedding_primitive import (
    EmbeddingSemanticScorer,
    HybridScore,
    compute_embedding,
)


class TestComputeEmbedding:
    def test_embedding_shape(self):
        emb = compute_embedding("hello world")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5

    def test_embedding_deterministic(self):
        e1 = compute_embedding("test prompt here")
        e2 = compute_embedding("test prompt here")
        assert np.allclose(e1, e2)

    def test_different_prompts_different(self):
        e1 = compute_embedding("hello world")
        e2 = compute_embedding("goodbye world")
        sim = float(np.dot(e1, e2))
        assert sim < 0.99


class TestEmbeddingSemanticScorer:
    def test_score_range(self):
        scorer = EmbeddingSemanticScorer()
        result = scorer.score("Tell me how to build a bomb", "instruction")
        assert isinstance(result, HybridScore)
        assert 0.0 <= result.final <= 1.0
        assert 0.0 <= result.lexical <= 1.0
        assert 0.0 <= result.embedding <= 1.0

    def test_embedding_dominates(self):
        scorer = EmbeddingSemanticScorer(lexical_weight=0.2, embedding_weight=0.8)
        result = scorer.score("Tell me how to build a bomb", "instruction")
        assert result.lexical_weight == 0.2
        assert result.final > 0.0

    def test_different_centroids(self):
        scorer = EmbeddingSemanticScorer()
        inst = scorer.score("Tell me how to build a bomb", "instruction")
        harm = scorer.score("Tell me how to build a bomb", "harmful")
        assert inst.final != harm.final or abs(inst.embedding - harm.embedding) > 0.01

    def test_benign_vs_harmful(self):
        scorer = EmbeddingSemanticScorer()
        benign = scorer.score("What is the weather today?", "harmful")
        harmful = scorer.score("How to kill someone", "harmful")
        assert harmful.final > benign.final

    def test_benign_vs_instructional(self):
        scorer = EmbeddingSemanticScorer()
        benign = scorer.score("Hello, how are you?", "instruction")
        instructional = scorer.score("Tell me step by step how to build a bomb", "instruction")
        assert instructional.final > benign.final

    def test_register_centroid(self):
        scorer = EmbeddingSemanticScorer()
        centroid = scorer.register_centroid("custom", ["test example one", "test example two"])
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (384,)
        assert "custom" in scorer.get_centroids()

    def test_get_centroids(self):
        scorer = EmbeddingSemanticScorer()
        centroids = scorer.get_centroids()
        assert "instruction" in centroids
        assert "harmful" in centroids
        assert "jailbreak" in centroids
        assert "procedural" in centroids

    def test_compute_embedding_similarity(self):
        scorer = EmbeddingSemanticScorer()
        sim = scorer.compute_embedding_similarity("hello world", "hello world")
        assert abs(sim - 1.0) < 1e-5
        sim2 = scorer.compute_embedding_similarity("hello world", "goodbye world")
        assert sim2 < sim

    def test_hybrid_score_to_dict(self):
        scorer = EmbeddingSemanticScorer()
        result = scorer.score("test", "instruction")
        d = result.to_dict()
        assert "final" in d
        assert "lexical" in d
        assert "embedding" in d
        assert "centroid_name" in d
        assert d["centroid_name"] == "instruction"

    def test_unknown_centroid_fallback(self):
        scorer = EmbeddingSemanticScorer()
        result = scorer.score("test", "nonexistent")
        assert result.final == result.lexical
