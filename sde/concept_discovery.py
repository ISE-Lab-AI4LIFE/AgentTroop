"""Problem 5: Semantic concept discovery.

Converts decision surfaces into interpretable concepts.
Uses clustering over embeddings to group observations into semantic regions.
Identifies common intents, themes, and recurring latent factors.

This explanation layer is separate from prediction.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .embedding_primitive import compute_embedding
from .semantic_store import SemanticObservation

logger = logging.getLogger(__name__)


_CONCEPT_LABELS: Dict[str, List[str]] = {
    "instructional_intent": [
        "tell me how", "explain", "instructions", "steps", "guide",
        "procedure", "walk me through", "show me how", "teach me",
    ],
    "harmful_topic": [
        "bomb", "kill", "weapon", "poison", "hack", "malware",
        "exploit", "steal", "fraud", "attack",
    ],
    "jailbreak_intent": [
        "ignore", "override", "dan", "jailbreak", "no restrictions",
        "forget your training", "you are now", "bypass",
    ],
    "procedural_request": [
        "step", "first", "next", "then", "finally", "phase",
        "stage", "sequence", "procedure",
    ],
    "benign_query": [
        "what is", "weather", "hello", "who is", "capital of",
        "definition", "meaning of", "translate",
    ],
}


@dataclass
class SemanticConcept:
    """An interpretable semantic concept discovered from observations.

    Attributes
    ----------
    name : str
        Human-readable concept name.
    centroid_embedding : np.ndarray
        Embedding centroid of this concept cluster.
    keywords : List[str]
        Representative keywords for interpretability.
    observation_count : int
        Number of observations assigned to this concept.
    refuse_rate : float
        Fraction of observations in this concept that resulted in REFUSE.
    description : str
        Natural language description.
    confidence : float
        How well-defined this concept is (0-1).
    """
    name: str
    centroid_embedding: np.ndarray
    keywords: List[str]
    observation_count: int
    refuse_rate: float
    description: str
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "keywords": self.keywords,
            "observation_count": self.observation_count,
            "refuse_rate": round(self.refuse_rate, 4),
            "description": self.description,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class ConceptExplanation:
    """An explanation of victim behaviour in terms of semantic concepts.

    Attributes
    ----------
    rule : str
        Human-readable decision rule.
    concepts : List[SemanticConcept]
        Concepts involved in the decision.
    confidence : float
        Confidence in this explanation.
    positive_concepts : List[str]
        Concepts whose presence leads to REFUSE.
    negative_concepts : List[str]
        Concepts whose absence leads to ACCEPT.
    """
    rule: str
    concepts: List[SemanticConcept]
    confidence: float
    positive_concepts: List[str]
    negative_concepts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "concepts": [c.to_dict() for c in self.concepts],
            "confidence": round(self.confidence, 4),
            "positive_concepts": self.positive_concepts,
            "negative_concepts": self.negative_concepts,
        }


class ActiveConceptCluster:
    """A cluster discovered by DBSCAN on misclassified prompts.

    Attributes
    ----------
    centroid_embedding : np.ndarray
        Mean embedding of cluster members.
    keywords : List[str]
        Extracted representative keywords.
    members : List[SemanticObservation]
        Observations in this cluster.
    purity : float
        Fraction of members with the same label (ALL_MISTAKE).
    is_high_purity : bool
        True when purity >= 0.8 and size >= 3.
    label : int
        Cluster label (-1 for noise).
    """

    def __init__(
        self,
        label: int,
        members: List[SemanticObservation],
        all_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        self.label = label
        self.members = members
        self.centroid_embedding = (
            np.mean(all_embeddings, axis=0)
            if all_embeddings is not None and len(all_embeddings) > 0
            else np.zeros(384)
        )
        self.keywords = SemanticConceptDiscovery._extract_keywords(
            [m.prompt for m in members], top_k=5,
        )
        outcomes = [m.outcome for m in members]
        self.refuse_rate = float(np.mean(outcomes)) if outcomes else 0.5
        self.purity = max(self.refuse_rate, 1.0 - self.refuse_rate)
        self.is_high_purity = self.purity >= 0.8 and len(members) >= 3

    @property
    def size(self) -> int:
        return len(self.members)

    def to_semantic_concept(self) -> SemanticConcept:
        dominant = "REFUSE" if self.refuse_rate >= 0.5 else "ACCEPT"
        return SemanticConcept(
            name=f"active_concept_{self.label}",
            centroid_embedding=self.centroid_embedding,
            keywords=self.keywords[:5],
            observation_count=self.size,
            refuse_rate=self.refuse_rate,
            description=(
                f"Cluster of {self.size} {'misclassified' if self.label >= 0 else 'noise'} "
                f"prompts; {dominant} rate={self.refuse_rate:.2f}; "
                f"purity={self.purity:.2f}; "
                f"keywords: {', '.join(self.keywords[:5])}"
            ),
            confidence=self.purity,
        )


class SemanticConceptDiscovery:
    """Discovers interpretable semantic concepts from observations.

    Uses embedding clustering + keyword extraction to identify
    latent semantic factors driving victim behaviour.

    Parameters
    ----------
    n_clusters : int
        Number of concept clusters (default 5).
    min_observations_per_concept : int
        Minimum observations to form a concept (default 3).
    """

    def __init__(
        self,
        n_clusters: int = 5,
        min_observations_per_concept: int = 3,
    ) -> None:
        self.n_clusters = n_clusters
        self.min_observations_per_concept = min_observations_per_concept
        self._predefined_concepts: Dict[str, np.ndarray] = {}
        self._init_predefined_concepts()

    def _init_predefined_concepts(self) -> None:
        for name, kws in _CONCEPT_LABELS.items():
            if kws:
                emb = compute_embedding(" ".join(kws))
                self._predefined_concepts[name] = emb

    def discover_concepts(
        self,
        observations: List[SemanticObservation],
    ) -> List[SemanticConcept]:
        """Discover concepts from observed prompts using embedding clustering.

        Parameters
        ----------
        observations : List[SemanticObservation]

        Returns
        -------
        List[SemanticConcept]
            Discovered concepts sorted by observation count.
        """
        if len(observations) < self.min_observations_per_concept:
            return self._fallback_concepts(observations)

        prompts = [obs.prompt for obs in observations]
        outcomes = [obs.outcome for obs in observations]
        embeddings = np.array([compute_embedding(p) for p in prompts])

        # Cluster
        concepts = self._cluster_concepts(embeddings, prompts, outcomes)

        # Merge with predefined concept matching
        concepts = self._merge_predefined(concepts, embeddings, prompts, outcomes)

        concepts.sort(key=lambda c: c.observation_count, reverse=True)
        return concepts

    def explain(
        self,
        observations: List[SemanticObservation],
    ) -> ConceptExplanation:
        """Produce a human-readable explanation of victim behaviour.

        Parameters
        ----------
        observations : List[SemanticObservation]

        Returns
        -------
        ConceptExplanation
        """
        concepts = self.discover_concepts(observations)
        if not concepts:
            return ConceptExplanation(
                rule="Insufficient data for explanation",
                concepts=[],
                confidence=0.0,
                positive_concepts=[],
                negative_concepts=[],
            )

        high_refuse = [c for c in concepts if c.refuse_rate > 0.6 and c.observation_count >= 2]
        low_refuse = [c for c in concepts if c.refuse_rate < 0.4 and c.observation_count >= 2]

        positive_names = [c.name for c in high_refuse]
        negative_names = [c.name for c in low_refuse]

        if positive_names and negative_names:
            rule = (
                "REFUSE occurs when: "
                + " AND ".join(positive_names)
                + " (and NOT: " + " OR ".join(negative_names) + ")"
            )
        elif positive_names:
            rule = "REFUSE occurs when: " + " OR ".join(positive_names)
        elif negative_names:
            rule = "REFUSE occurs when avoiding: " + " OR ".join(negative_names)
        else:
            rule = "No clear semantic pattern detected."

        confidence = self._compute_explanation_confidence(concepts, observations)
        return ConceptExplanation(
            rule=rule,
            concepts=concepts,
            confidence=confidence,
            positive_concepts=positive_names,
            negative_concepts=negative_names,
        )

    def _cluster_concepts(
        self,
        embeddings: np.ndarray,
        prompts: List[str],
        outcomes: List[int],
    ) -> List[SemanticConcept]:
        n = len(prompts)
        if n < self.n_clusters:
            k = max(1, n)
        else:
            k = self.n_clusters

        try:
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = clusterer.fit_predict(embeddings)
            centers = clusterer.cluster_centers_
        except Exception:
            labels = np.zeros(n, dtype=int)
            centers = np.mean(embeddings, axis=0, keepdims=True)

        concepts: Dict[int, Dict[str, Any]] = {}
        for i in range(k):
            mask = labels == i
            if mask.sum() == 0:
                continue
            cluster_prompts = [prompts[j] for j in range(n) if mask[j]]
            cluster_outcomes = [outcomes[j] for j in range(n) if mask[j]]
            centroid_emb = centers[i] if i < len(centers) else np.mean(embeddings[mask], axis=0)
            keywords = self._extract_keywords(cluster_prompts)
            refuse_rate = float(np.mean(cluster_outcomes))
            concepts[i] = {
                "centroid": centroid_emb,
                "keywords": keywords,
                "count": int(mask.sum()),
                "refuse_rate": refuse_rate,
            }

        return [
            SemanticConcept(
                name=f"concept_{i}",
                centroid_embedding=c["centroid"],
                keywords=c["keywords"][:5],
                observation_count=c["count"],
                refuse_rate=c["refuse_rate"],
                description=f"Cluster of {c['count']} prompts; "
                            f"refuse_rate={c['refuse_rate']:.2f}; "
                            f"keywords: {', '.join(c['keywords'][:5])}",
                confidence=min(1.0, c["count"] / max(n, 1) * 2),
            )
            for i, c in concepts.items()
        ]

    def _merge_predefined(
        self,
        concepts: List[SemanticConcept],
        embeddings: np.ndarray,
        prompts: List[str],
        outcomes: List[int],
    ) -> List[SemanticConcept]:
        for pname, pemb in self._predefined_concepts.items():
            sims = embeddings @ pemb
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim > 0.5:
                matched = [outcomes[i] for i in range(len(outcomes))
                           if embeddings[i] @ pemb > 0.4]
                if matched:
                    rr = float(np.mean(matched))
                    kws = _CONCEPT_LABELS.get(pname, [])
                    concepts.append(SemanticConcept(
                        name=pname,
                        centroid_embedding=pemb,
                        keywords=kws[:5],
                        observation_count=len(matched),
                        refuse_rate=rr,
                        description=f"'{pname}': {len(matched)} similar prompts, "
                                    f"refuse_rate={rr:.2f}",
                        confidence=min(1.0, len(matched) / max(len(outcomes), 1) * 2),
                    ))
        return concepts

    def _fallback_concepts(
        self,
        observations: List[SemanticObservation],
    ) -> List[SemanticConcept]:
        """Fallback when too few observations for clustering."""
        if not observations:
            return []
        outcomes = [obs.outcome for obs in observations]
        for pname, pemb in self._predefined_concepts.items():
            prompts = [obs.prompt for obs in observations]
            kws = _CONCEPT_LABELS.get(pname, [])
            matched = [outcomes[i] for i in range(len(prompts))
                       if any(kw in prompts[i].lower() for kw in kws)]
            if matched:
                rr = float(np.mean(matched))
                return [SemanticConcept(
                    name=pname,
                    centroid_embedding=pemb,
                    keywords=kws[:5],
                    observation_count=len(matched),
                    refuse_rate=rr,
                    description=f"'{pname}': {len(matched)} matched prompts",
                    confidence=0.5,
                )]
        return []

    @staticmethod
    def _extract_keywords(prompts: List[str], top_k: int = 10) -> List[str]:
        """Extract representative keywords from a cluster of prompts.

        Extended stopwords include question words and weak verbs commonly
        found in benign instructional prompts.
        """
        import re
        from collections import Counter
        words: List[str] = []
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "and", "or", "but", "not", "this", "that", "it", "its",
            "i", "you", "he", "she", "we", "they", "me", "my", "your",
            "do", "does", "did", "have", "has", "had", "can", "will",
            "would", "could", "should", "may", "might", "shall",
            # Extended stopwords: question words + weak instructional verbs
            "tell", "how", "what", "why", "when", "where", "which",
            "who", "whom", "whose", "must",
        }
        for p in prompts:
            tokens = re.findall(r"[a-zA-Z]+", p.lower())
            words.extend(t for t in tokens if t not in stopwords and len(t) > 2)
        if not words:
            return prompts[:min(3, len(prompts))]
        common = Counter(words).most_common(top_k)
        return [w for w, _ in common]

    @staticmethod
    def _compute_explanation_confidence(
        concepts: List[SemanticConcept],
        observations: List[SemanticObservation],
    ) -> float:
        if not observations or not concepts:
            return 0.0
        top = concepts[0]
        coverage = top.observation_count / max(len(observations), 1)
        clarity = top.confidence
        return min(1.0, coverage * clarity * 2)

    # ------------------------------------------------------------------
    # Active concept discovery (DBSCAN on misclassifications)
    # ------------------------------------------------------------------

    def collect_misclassifications(
        self,
        observations: List[SemanticObservation],
        predictions: Optional[Dict[str, int]] = None,
        balance_ratio: float = 0.7,
    ) -> List[SemanticObservation]:
        """Find observations where the SDE prediction disagreed with the victim.

        Returns false-positive (SDE said ACCEPT, victim said REFUSE) and
        false-negative (SDE said REFUSE, victim said ACCEPT) observations,
        balanced by *balance_ratio*.

        Parameters
        ----------
        observations : List[SemanticObservation]
            All observations from the engine.
        predictions : Dict[str, int], optional
            Pre-computed SDE predictions per prompt.  When None, uses
            score > 0.5 as the prediction rule.
        balance_ratio : float
            Target fraction of FP in the returned set (default 0.5).
            The remaining fraction comes from FN.  If one class has
            fewer samples, all available samples are used.

        Returns
        -------
        List[SemanticObservation]
            Sampled misclassifications (FP + FN).
        """
        fps: List[SemanticObservation] = []
        fns: List[SemanticObservation] = []
        for obs in observations:
            if predictions is not None:
                sde_pred = predictions.get(obs.prompt, 0)
            else:
                sde_pred = 1 if obs.score > 0.5 else 0
            if sde_pred == 1 and obs.outcome == 0:
                fps.append(obs)
            elif sde_pred == 0 and obs.outcome == 1:
                fns.append(obs)

        if not fps and not fns:
            return []

        if not fns:
            logger.warning(
                "No false negatives found — returning %d FPs only",
                len(fps),
            )
            return fps

        import random
        total = len(fps) + len(fns)
        if total < 4:
            return fps + fns

        n_fp = max(1, int(total * balance_ratio)) if fps else 0
        n_fn = max(1, total - n_fp) if fns else 0

        sampled_fp = random.sample(fps, min(n_fp, len(fps)))
        sampled_fn = random.sample(fns, min(n_fn, len(fns)))
        return sampled_fp + sampled_fn

    def collect_confident_refuses(
        self,
        observations: List[SemanticObservation],
        predictions: Optional[Dict[str, int]] = None,
        max_samples: int = 20,
    ) -> List[SemanticObservation]:
        """Find observations where SDE and victim both predicted REFUSE (True Positives).

        These form the basis for high-quality REFUSE cluster discovery.

        Parameters
        ----------
        observations : List[SemanticObservation]
            All observations from the engine.
        predictions : Dict[str, int], optional
            Pre-computed SDE predictions.
        max_samples : int
            Maximum number of TPs to return (default 20).

        Returns
        -------
        List[SemanticObservation]
            True positive observations (up to *max_samples*).
        """
        tps: List[SemanticObservation] = []
        for obs in observations:
            if predictions is not None:
                sde_pred = predictions.get(obs.prompt, 0)
            else:
                sde_pred = 1 if obs.score > 0.5 else 0
            if sde_pred == 1 and obs.outcome == 1:
                tps.append(obs)
        tps.sort(key=lambda o: o.score, reverse=True)
        return tps[:max_samples]

    def discover_active_concepts(
        self,
        observations: List[SemanticObservation],
        eps: float = 0.6,
        min_samples: int = 2,
        purity_threshold: float = 0.9,
        min_cluster_size: int = 3,
        include_tps: bool = True,
        max_tps: int = 20,
        use_all_observations: bool = True,
    ) -> List[SemanticConcept]:
        """Discover concepts by clustering prompts with DBSCAN.

        Pipeline:
          1. Collect all observations (not just misclassifications) so that
             both ACCEPT and REFUSE labels are present, enabling DBSCAN to
             form high-purity REFUSE clusters.
          2. Embed via ``compute_embedding``.
          3. Run DBSCAN clustering with *eps* neighbourhood radius.
          4. Filter clusters by purity >= *purity_threshold* and
             size >= *min_cluster_size*.
          5. Only keep REFUSE clusters (refuse_rate >= 0.8).
             ACCEPT clusters (refuse_rate <= 0.2) are discarded — they
             contain noisy keywords ("tell", "how") that interfere with
             the structural pipeline.
          6. Convert high-purity REFUSE clusters to ``SemanticConcept``.

        Parameters
        ----------
        observations : List[SemanticObservation]
        eps : float
            DBSCAN neighbourhood radius (default 0.6).
        min_samples : int
            Min points for core cluster (default 2).
        purity_threshold : float
            Min fraction of majority label (default 0.9).
        min_cluster_size : int
            Min cluster members to form a concept (default 3).
        include_tps : bool, optional
            Ignored when use_all_observations=True.
        max_tps : int, optional
            Ignored when use_all_observations=True.
        use_all_observations : bool
            When True, uses ALL observations (not just misclassifications).
            This gives DBSCAN both REFUSE and ACCEPT examples, enabling
            high-quality REFUSE cluster discovery.

        Returns
        -------
        List[SemanticConcept]
            High-purity REFUSE concepts only (may be empty).
        """
        if use_all_observations:
            cluster_obs = list(observations)
        else:
            mis = self.collect_misclassifications(observations, balance_ratio=0.5)
            cluster_obs = list(mis)
            if include_tps:
                tps = self.collect_confident_refuses(observations, max_samples=max_tps)
                cluster_obs.extend(tps)

        if len(cluster_obs) < min_cluster_size:
            return []

        prompts = [m.prompt for m in cluster_obs]
        embeddings = np.array([compute_embedding(p) for p in prompts])

        try:
            from sklearn.cluster import DBSCAN
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
            labels = clusterer.fit_predict(embeddings)
        except Exception:
            labels = np.full(len(cluster_obs), -1, dtype=int)

        clusters: Dict[int, List[SemanticObservation]] = {}
        cluster_embs: Dict[int, List[np.ndarray]] = {}
        for i, label in enumerate(labels):
            label_int = int(label)
            clusters.setdefault(label_int, []).append(cluster_obs[i])
            cluster_embs.setdefault(label_int, []).append(embeddings[i])

        concepts: List[SemanticConcept] = []
        for label_int, members in clusters.items():
            if label_int < 0:
                continue
            if len(members) < min_cluster_size:
                continue
            all_emb = np.array(cluster_embs[label_int])
            cluster = ActiveConceptCluster(
                label=label_int,
                members=members,
                all_embeddings=all_emb,
            )
            # Only keep high-purity REFUSE clusters
            if not cluster.is_high_purity:
                continue
            if cluster.purity < purity_threshold:
                continue
            if cluster.refuse_rate < 0.8:
                continue
            concepts.append(cluster.to_semantic_concept())

        concepts.sort(key=lambda c: c.observation_count, reverse=True)
        return concepts

    @staticmethod
    def proposals_for_episodic_memory(
        concepts: List[SemanticConcept],
    ) -> List[Dict[str, Any]]:
        """Convert discovered concepts to a serialisable format for storage.

        Each dict can be stored in episodic memory annotations for reuse
        across campaigns.
        """
        return [
            {
                "name": c.name,
                "keywords": c.keywords,
                "observation_count": c.observation_count,
                "refuse_rate": round(c.refuse_rate, 4),
                "confidence": round(c.confidence, 4),
                "source": "active_concept_discovery",
            }
            for c in concepts
            if c.observation_count >= 3 and c.confidence >= 0.8
        ]

    @staticmethod
    def is_common_keyword(keyword: str, all_prompts: List[str], threshold: float = 0.3) -> bool:
        """Check if a keyword appears in too many prompts.

        Uses aggressive threshold (0.3) to filter out generic words like
        "tell", "how", "information", "about" that would generate noisy
        hypotheses in the structural pipeline.
        """
        if not all_prompts:
            return False
        count = sum(1 for p in all_prompts if keyword.lower() in p.lower())
        return (count / len(all_prompts)) > threshold

    @staticmethod
    def _EXTENDED_STOPWORDS() -> set:
        """Extended stopwords including question words and weak instructional verbs."""
        return {
            "tell", "how", "what", "why", "when", "where", "which",
            "who", "whom", "whose", "must",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "and", "or", "but", "not", "this", "that", "it", "its",
            "do", "does", "did", "have", "has", "had", "can", "will",
            "would", "could", "should", "may", "might", "shall",
        }

    @staticmethod
    def is_stopword(word: str) -> bool:
        return word.lower() in SemanticConceptDiscovery._EXTENDED_STOPWORDS()
