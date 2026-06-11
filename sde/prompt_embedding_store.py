"""Problem 3: Embedding-space diversity search for prompt generation.

Maintains a store of previously generated prompt embeddings.
Rejects prompts that are too similar to existing ones.
Enforces minimum cosine distance threshold.

Goal: maximise semantic coverage, not lexical variation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .embedding_primitive import compute_embedding

logger = logging.getLogger(__name__)


@dataclass
class StoredPrompt:
    """A prompt stored with its embedding for diversity checking.

    Attributes
    ----------
    prompt : str
    embedding : np.ndarray
    source : str
        Generation source (semantic / symbolic / hybrid).
    score_vector : Dict[str, float]
        Score vector if available.
    """
    prompt: str
    embedding: np.ndarray
    source: str = "unknown"
    score_vector: Dict[str, float] = field(default_factory=dict)


class PromptEmbeddingStore:
    """Store of prompt embeddings for diversity enforcement.

    Parameters
    ----------
    min_cosine_distance : float
        Minimum distance (1 - similarity) between prompts (default 0.15).
    store_size : int
        Max prompts to keep in memory (default 500).
    """

    def __init__(
        self,
        min_cosine_distance: float = 0.15,
        store_size: int = 500,
    ) -> None:
        self.min_cosine_distance = min_cosine_distance
        self.store_size = store_size
        self._prompts: List[StoredPrompt] = []

    def add(self, prompt: str, source: str = "unknown",
            score_vector: Optional[Dict[str, float]] = None) -> None:
        """Add a prompt to the store."""
        embedding = compute_embedding(prompt)
        self._prompts.append(StoredPrompt(
            prompt=prompt,
            embedding=embedding,
            source=source,
            score_vector=score_vector or {},
        ))
        if len(self._prompts) > self.store_size:
            self._prompts = self._prompts[-self.store_size:]

    def is_diverse(self, prompt: str) -> bool:
        """Check if a prompt is sufficiently different from stored prompts.

        Parameters
        ----------
        prompt : str

        Returns
        -------
        bool
            True if prompt is diverse enough to be novel.
        """
        if not self._prompts:
            return True
        embedding = compute_embedding(prompt)
        threshold = 1.0 - self.min_cosine_distance
        for stored in self._prompts:
            sim = float(np.dot(embedding, stored.embedding))
            if sim >= threshold:
                return False
        return True

    def get_diverse_candidates(
        self,
        candidates: List[str],
        n: int = 5,
    ) -> List[str]:
        """Filter candidates, keeping only diverse ones.

        Parameters
        ----------
        candidates : List[str]
            Raw candidate prompts.
        n : int
            Max number to return.

        Returns
        -------
        List[str]
            Diverse prompts, ordered by input order.
        """
        diverse: List[str] = []
        for c in candidates:
            if self.is_diverse(c):
                diverse.append(c)
                self.add(c, source="filtered")
            if len(diverse) >= n:
                break
        return diverse

    def size(self) -> int:
        return len(self._prompts)

    def clear(self) -> None:
        self._prompts.clear()

    def embedding_coverage(self) -> Dict[str, float]:
        """Compute coverage statistics for the store.

        Returns
        -------
        Dict with mean_similarity (self-consistency) and diversity_score.
        """
        if len(self._prompts) < 2:
            return {"mean_similarity": 0.0, "diversity_score": 1.0}
        embeddings = np.array([p.embedding for p in self._prompts])
        sims = embeddings @ embeddings.T
        n = len(self._prompts)
        upper_tri = sims[np.triu_indices(n, k=1)]
        mean_sim = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
        return {
            "mean_similarity": round(mean_sim, 4),
            "diversity_score": round(1.0 - mean_sim, 4),
        }
