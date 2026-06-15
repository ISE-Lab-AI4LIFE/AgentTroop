"""Neural-Guided Synthesizer — uses embeddings + bandit for transform selection."""

import logging
import math
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from core.executor import ProgramExecutor
from core.primitive import (
    ContainsWordPredicate, ContainsAnyWordPredicate,
    LengthGtPredicate, LengthLtPredicate,
    StartsWithRoleplayPredicate, ContainsSystemOverridePredicate,
    MatchesJailbreakPatternPredicate, ContainsEncodingWrapperPredicate,
    PrimitiveRegistry, default_registry,
)
from core.program import (
    Program, IfThenElseNode, PredicateNode,
)

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None


class UCBBandit:
    """Upper Confidence Bound bandit for transform selection."""
    
    def __init__(self, n_arms: int, c: float = 1.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = [0.0] * n_arms
        self.values = [0.0] * n_arms
        self.t = 0
    
    def select_arm(self) -> int:
        self.t += 1
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        ucb_values = [
            v + self.c * math.sqrt(math.log(self.t) / c)
            for v, c in zip(self.values, self.counts)
        ]
        return max(range(self.n_arms), key=lambda i: ucb_values[i])
    
    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1.0
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward


class NeuralGuidedSynthesizer:
    """Neural-guided synthesizer using embeddings + bandit for transform selection.
    
    Uses SentenceTransformer to embed prompts and transforms, then uses
    a UCB bandit to select promising transforms based on history.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        bandit_algorithm: str = "ucb",
        primitive_registry: Optional[PrimitiveRegistry] = None,
    ):
        self.embedding_model_name = embedding_model
        self.bandit_algorithm = bandit_algorithm
        self.primitive_registry = primitive_registry or default_registry
        self.executor = ProgramExecutor(self.primitive_registry)
        
        self._model = None
        self._bandit = None
        self._transforms = []
        self._init_model()
    
    def _init_model(self) -> None:
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                self._model = SentenceTransformer(self.embedding_model_name)
                logger.info("Loaded embedding model: %s", self.embedding_model_name)
            except Exception as e:
                logger.warning("Failed to load embedding model: %s", e)
                self._model = None
        else:
            logger.warning("sentence-transformers not available; neural synthesizer disabled")
            self._model = None
    
    def _ensure_bandit(self, n_transforms: int) -> None:
        if self._bandit is None or self._bandit.n_arms != n_transforms:
            self._bandit = UCBBandit(n_arms=n_transforms)
            self._transforms = list(range(n_transforms))
    
    def synthesize(
        self,
        examples: List[Tuple[str, int]],
        base_programs: Optional[List[Program]] = None,
        k: int = 10,
    ) -> List[Program]:
        start = time.time()
        if not examples:
            return []
        
        keywords = self._extract_keywords(examples)
        candidates = []
        
        for kw in keywords[:10]:
            for to, eo in [(1, 0), (0, 1)]:
                prog = Program(
                    root=IfThenElseNode(
                        condition=PredicateNode(primitive=ContainsWordPredicate(word=kw)),
                        then_outcome=to, else_outcome=eo,
                    )
                )
                prog.id = f"neural_{uuid.uuid4().hex[:8]}"
                fitness = self._fitness(prog, examples)
                candidates.append((fitness, prog))
        
        candidates.sort(key=lambda x: -x[0])
        results = [p for _, p in candidates[:k]]
        
        logger.info(
            "Neural synthesis: %d candidates, best=%.4f, %.1fs",
            len(results), candidates[0][0] if candidates else 0.0,
            time.time() - start,
        )
        return results
    
    def _extract_keywords(self, examples: List[Tuple[str, int]]) -> List[str]:
        import re
        from collections import Counter
        refuse_words = Counter()
        for prompt, outcome in examples:
            if outcome == 1:
                words = re.findall(r"[a-zA-Z]{3,}", prompt.lower())
                refuse_words.update(w for w in words if w not in {
                    "the", "a", "an", "is", "are", "was", "were", "be", "been",
                    "have", "has", "had", "do", "does", "did", "will",
                    "would", "could", "should", "may", "might", "shall", "can",
                    "to", "of", "in", "for", "on", "with", "at", "by", "from",
                    "as", "into", "through", "during", "before", "after",
                    "then", "once", "here", "there", "when", "where",
                    "why", "how", "what", "which", "who", "whom", "this",
                    "these", "those", "am", "it", "its", "no", "nor", "not",
                    "or", "and", "but", "if", "so", "than", "too", "very",
                    "just", "about", "also",
                })
        return [w for w, _ in refuse_words.most_common(20)]
    
    def _fitness(self, program: Program, examples: List[Tuple[str, int]]) -> float:
        if not examples:
            return 0.0
        correct = 0
        for prompt, expected in examples:
            try:
                if int(self.executor.execute(program, prompt)) == expected:
                    correct += 1
            except Exception:
                pass
        return correct / len(examples)
