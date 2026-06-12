"""Surrogate Policy Model — trained on episodes to predict victim outcomes.

When victim is ≈100% REFUSE or ≈100% ACCEPT, the surrogate provides:
1. Uncertainty estimates for each prediction
2. Synthetic probes that would change the surrogate's prediction
3. A differentiable signal for the version space posterior update
"""

import logging
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "shall", "should", "may", "might", "must", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "as", "until", "while",
    "about", "if", "but", "or", "and", "i", "me", "my", "myself", "we",
    "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
}


@dataclass
class SurrogatePrediction:
    prompt: str
    predicted_outcome: int  # 0=ACCEPT, 1=REFUSE
    confidence: float  # 0.0-1.0
    uncertainty: float  # 0.0-1.0 (higher = more uncertain)
    n_episodes: int
    prediction_id: str = ""


@dataclass
class SurrogateTrainingStats:
    n_episodes: int
    n_features: int
    train_accuracy: float
    n_refuse: int
    n_accept: int
    feature_importance: Dict[str, float]
    duration_ms: float


class SurrogatePolicyModel:
    """Lightweight Bayesian surrogate trained on episodes.

    Uses a Naive Bayes-like approach with Dirichlet-multinomial priors
    to provide calibrated uncertainty even with few data points.
    Supports three feature types:
    1. Keyword presence (bag-of-words)
    2. Length features (short/medium/long)
    3. Structural features (question, imperative, roleplay, etc.)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        max_features: int = 100,
        min_episodes_for_training: int = 3,
        model_type: str = "mlp",
        feature_type: str = "embedding",
        min_accuracy: float = 0.65,
    ):
        self.alpha = alpha
        self.max_features = max_features
        self.min_episodes_for_training = min_episodes_for_training
        self.model_type = model_type
        self.feature_type = feature_type
        self.min_accuracy = min_accuracy
        self._vocab: List[str] = []
        self._word_counts: np.ndarray = np.zeros((2, 0))
        self._prior_counts = np.array([1.0, 1.0])
        self._n_episodes = 0
        self._is_trained = False
        self._feature_importance: Dict[str, float] = {}
        self._embedding_model = None
        self._sklearn_model = None
        self._scaler = None
        self._is_active = True

    @property
    def is_active(self) -> bool:
        return self._is_active

    def _build_embeddings(self, prompts: List[str]) -> np.ndarray:
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers is not installed")
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_model.encode(prompts, show_progress_bar=False)

    def train(self, episodes: List[Tuple[str, int]]) -> SurrogateTrainingStats:
        """Train surrogate on (prompt, outcome) episodes."""
        start = time.time()
        self._n_episodes = len(episodes)
        if self._n_episodes < self.min_episodes_for_training:
            logger.info("Surrogate: too few episodes (%d < %d)", self._n_episodes, self.min_episodes_for_training)
            return SurrogateTrainingStats(
                n_episodes=self._n_episodes, n_features=0, train_accuracy=0.0,
                n_refuse=sum(1 for _, o in episodes if o == 1),
                n_accept=sum(1 for _, o in episodes if o == 0),
                feature_importance={}, duration_ms=0.0,
            )

        use_embedding = (
            _HAS_SENTENCE_TRANSFORMERS
            and _HAS_SKLEARN
            and self.feature_type == "embedding"
        )

        if use_embedding:
            prompts = [p for p, _ in episodes]
            labels = np.array([o for _, o in episodes])
            embeddings = self._build_embeddings(prompts)

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(embeddings)

            if self.model_type == "xgboost" and _HAS_XGB:
                self._sklearn_model = XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                )
            else:
                self._sklearn_model = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=10,
                )

            # Balance classes via resampling (compatible with sklearn 2.x)
            refuse_idx = np.where(labels == 1)[0]
            accept_idx = np.where(labels == 0)[0]
            n_refuse = len(refuse_idx)
            n_accept = len(accept_idx)
            if n_refuse > 0 and n_accept > 0 and n_refuse != n_accept:
                minority = refuse_idx if n_refuse < n_accept else accept_idx
                majority = accept_idx if n_refuse < n_accept else refuse_idx
                n_minority = len(minority)
                n_majority = len(majority)
                resampled_minority = np.random.choice(minority, size=n_majority, replace=True)
                balanced_idx = np.concatenate([majority, resampled_minority])
                np.random.shuffle(balanced_idx)
                X_bal = X_scaled[balanced_idx]
                y_bal = labels[balanced_idx]
            else:
                X_bal, y_bal = X_scaled, labels

            self._sklearn_model.fit(X_bal, y_bal)
            self._is_trained = True

            train_preds = self._sklearn_model.predict(X_scaled)
            train_acc = float(np.mean(train_preds == labels))
            n_refuse = int(labels.sum())
            n_accept = len(labels) - n_refuse

            if train_acc < self.min_accuracy:
                self._is_active = False
                logger.warning(
                    "=== SURROGATE DEACTIVATED: train_acc=%.3f < min_accuracy=%.3f, "
                    "falling back to uniform prior for all predictions ===",
                    train_acc, self.min_accuracy,
                )

            logger.info(
                "Surrogate (embedding): trained on %d episodes, %d features, train_acc=%.3f "
                "(refuse=%d, accept=%d, %.1fms)",
                self._n_episodes, embeddings.shape[1], train_acc,
                n_refuse, n_accept, (time.time() - start) * 1000,
            )
            return SurrogateTrainingStats(
                n_episodes=self._n_episodes,
                n_features=embeddings.shape[1],
                train_accuracy=train_acc,
                n_refuse=n_refuse,
                n_accept=n_accept,
                feature_importance={},
                duration_ms=(time.time() - start) * 1000,
            )

        word_freq: Counter = Counter()
        for prompt, _ in episodes:
            words = re.findall(r"[a-zA-Z]{3,}", prompt.lower())
            word_freq.update(w for w in words if w not in _STOPWORDS)

        self._vocab = [w for w, _ in word_freq.most_common(self.max_features)]
        n_features = len(self._vocab)
        self._word_counts = np.zeros((2, n_features))
        word_to_idx = {w: i for i, w in enumerate(self._vocab)}

        for prompt, outcome in episodes:
            words = set(re.findall(r"[a-zA-Z]{3,}", prompt.lower()))
            for w in words:
                if w in word_to_idx:
                    self._word_counts[outcome, word_to_idx[w]] += 1

        self._is_trained = True

        total = len(episodes)
        n_refuse = self._word_counts[1].sum()
        n_accept = self._word_counts[0].sum()
        for i, word in enumerate(self._vocab):
            count_refuse = self._word_counts[1, i]
            count_accept = self._word_counts[0, i]
            total_word = count_refuse + count_accept
            if total_word > 0:
                p_refuse_given_word = count_refuse / max(total_word, 1)
                p_refuse_prior = n_refuse / max(total, 1)
                self._feature_importance[word] = abs(p_refuse_given_word - p_refuse_prior)
            else:
                self._feature_importance[word] = 0.0

        train_correct = 0
        for prompt, expected in episodes:
            pred = self.predict(prompt)
            if pred.predicted_outcome == expected:
                train_correct += 1
        train_acc = train_correct / max(total, 1)

        if train_acc < self.min_accuracy:
            self._is_active = False
            logger.warning(
                "=== SURROGATE DEACTIVATED: train_acc=%.3f < min_accuracy=%.3f, "
                "falling back to uniform prior for all predictions ===",
                train_acc, self.min_accuracy,
            )

        logger.info(
            "Surrogate: trained on %d episodes, %d features, train_acc=%.3f "
            "(refuse=%d, accept=%d, %.1fms)",
            self._n_episodes, n_features, train_acc,
            n_refuse, n_accept, (time.time() - start) * 1000,
        )
        return SurrogateTrainingStats(
            n_episodes=self._n_episodes, n_features=n_features,
            train_accuracy=train_acc,
            n_refuse=n_refuse, n_accept=n_accept,
            feature_importance=dict(sorted(
                self._feature_importance.items(), key=lambda x: -x[1]
            )[:20]),
            duration_ms=(time.time() - start) * 1000,
        )

    def predict(self, prompt: str) -> SurrogatePrediction:
        """Predict outcome with calibrated uncertainty."""
        if not self._is_trained:
            return SurrogatePrediction(
                prompt=prompt, predicted_outcome=1,
                confidence=0.5, uncertainty=1.0,
                n_episodes=self._n_episodes,
            )

        if not self._is_active:
            return SurrogatePrediction(
                prompt=prompt, predicted_outcome=1,
                confidence=0.5, uncertainty=1.0,
                n_episodes=self._n_episodes,
                prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
            )

        if self._sklearn_model is not None and self._scaler is not None:
            embedding = self._build_embeddings([prompt])
            X_scaled = self._scaler.transform(embedding)

            if hasattr(self._sklearn_model, "predict_proba"):
                proba = self._sklearn_model.predict_proba(X_scaled)[0]
                if len(proba) == 1:
                    p_refuse = float(proba[0]) if self._sklearn_model.classes_[0] == 1 else 0.0
                elif len(proba) >= 2:
                    refuse_idx = (
                        1 if self._sklearn_model.classes_[1] == 1 else 0
                    )
                    p_refuse = float(proba[refuse_idx])
                else:
                    p_refuse = 0.5
            else:
                pred_class = int(self._sklearn_model.predict(X_scaled)[0])
                p_refuse = 1.0 if pred_class == 1 else 0.0

            predicted = 1 if p_refuse >= 0.5 else 0
            confidence = max(p_refuse, 1.0 - p_refuse)
            p_accept = 1.0 - p_refuse
            if p_refuse > 0 and p_accept > 0:
                uncertainty = -p_refuse * math.log(p_refuse) - p_accept * math.log(p_accept)
                uncertainty /= math.log(2)
            else:
                uncertainty = 0.0

            return SurrogatePrediction(
                prompt=prompt,
                predicted_outcome=predicted,
                confidence=confidence,
                uncertainty=uncertainty,
                n_episodes=self._n_episodes,
                prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
            )

        words = set(re.findall(r"[a-zA-Z]{3,}", prompt.lower()))
        word_to_idx = {w: i for i, w in enumerate(self._vocab)}

        n_refuse = self._word_counts[1].sum() + self.alpha
        n_accept = self._word_counts[0].sum() + self.alpha
        total = n_refuse + n_accept

        p_refuse_prior = n_refuse / max(total, 1)

        log_ratio = 0.0
        n_features_found = 0
        for w in words:
            if w in word_to_idx:
                idx = word_to_idx[w]
                count_refuse = self._word_counts[1, idx] + self.alpha
                count_accept = self._word_counts[0, idx] + self.alpha
                p_given_refuse = count_refuse / max(n_refuse, 1)
                p_given_accept = count_accept / max(n_accept, 1)
                if p_given_accept > 0 and p_given_refuse > 0:
                    log_ratio += math.log(p_given_refuse / p_given_accept)
                    n_features_found += 1

        if n_features_found > 0:
            log_posterior_odds = log_ratio + math.log(p_refuse_prior / (1.0 - p_refuse_prior + 1e-10))
            p_refuse = 1.0 / (1.0 + math.exp(-log_posterior_odds))
        else:
            p_refuse = p_refuse_prior

        predicted = 1 if p_refuse >= 0.5 else 0
        confidence = max(p_refuse, 1.0 - p_refuse)

        p_accept = 1.0 - p_refuse
        if p_refuse > 0 and p_accept > 0:
            uncertainty = -p_refuse * math.log(p_refuse) - p_accept * math.log(p_accept)
            uncertainty /= math.log(2)
        else:
            uncertainty = 0.0

        return SurrogatePrediction(
            prompt=prompt,
            predicted_outcome=predicted,
            confidence=confidence,
            uncertainty=uncertainty,
            n_episodes=self._n_episodes,
            prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
        )

    def predict_batch(
        self, prompts: List[str]
    ) -> List[SurrogatePrediction]:
        """Batch predict for multiple prompts."""
        return [self.predict(p) for p in prompts]

    def compute_disagreement(
        self,
        prompt: str,
        candidate_predictions: List[int],
    ) -> float:
        """Compute normalized disagreement entropy among candidate predictions.

        0.0 = all candidates agree (no signal)
        1.0 = maximum disagreement (best signal)
        """
        if len(candidate_predictions) < 2:
            return 0.0
        n_refuse = sum(1 for p in candidate_predictions if p == 1)
        n_total = len(candidate_predictions)
        p = n_refuse / n_total
        if p <= 0.0 or p >= 1.0:
            return 0.0
        entropy = -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)
        return entropy / math.log(2)

    def expected_information_gain(
        self,
        prompt: str,
        candidate_predictions: List[int],
        candidate_posteriors: List[float],
    ) -> float:
        """Expected Information Gain of querying this prompt.

        EIG = H[current_posterior] - E[ H[posterior | outcome] ]

        Uses the current candidate posterior and their predictions to estimate
        how much information this prompt would provide.
        """
        if len(candidate_predictions) < 2 or len(candidate_posteriors) < 2:
            return 0.0

        candidates = list(zip(candidate_predictions, candidate_posteriors))
        n_refuse = sum(p for pred, p in candidates if pred == 1)
        n_total = sum(p for _, p in candidates)
        p_refuse = n_refuse / max(n_total, 1e-10)

        if p_refuse <= 0.0 or p_refuse >= 1.0:
            return 0.0

        h_current = -p_refuse * math.log(p_refuse) - (1.0 - p_refuse) * math.log(1.0 - p_refuse)
        h_current /= math.log(2)

        posterior_refuse = [
            p * (1.0 if pred != 1 else 0.0)
            for pred, p in candidates
        ]
        posterior_accept = [
            p * (1.0 if pred != 0 else 0.0)
            for pred, p in candidates
        ]
        sum_refuse = sum(posterior_refuse) + 1e-10
        sum_accept = sum(posterior_accept) + 1e-10
        posterior_refuse = [s / sum_refuse for s in posterior_refuse]
        posterior_accept = [s / sum_accept for s in posterior_accept]

        def _entropy(dist: List[float]) -> float:
            h = -sum(p * math.log(p + 1e-10) for p in dist)
            return h / math.log(2)

        h_expected = p_refuse * _entropy(posterior_refuse) + (1.0 - p_refuse) * _entropy(posterior_accept)

        return h_current - h_expected

    def state_dict(self) -> Dict[str, Any]:
        return {
            "vocab": self._vocab,
            "word_counts": self._word_counts.tolist(),
            "prior_counts": self._prior_counts.tolist(),
            "n_episodes": self._n_episodes,
            "is_trained": self._is_trained,
            "alpha": self.alpha,
            "model_type": self.model_type,
            "feature_type": self.feature_type,
            "min_accuracy": self.min_accuracy,
            "is_active": self._is_active,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._vocab = state.get("vocab", [])
        self._word_counts = np.array(state.get("word_counts", np.zeros((2, 0))))
        self._prior_counts = np.array(state.get("prior_counts", [1.0, 1.0]))
        self._n_episodes = state.get("n_episodes", 0)
        self._is_trained = state.get("is_trained", False)
        self.alpha = state.get("alpha", 0.1)
        self.model_type = state.get("model_type", "mlp")
        self.feature_type = state.get("feature_type", "embedding")
        self.min_accuracy = state.get("min_accuracy", 0.65)
        self._is_active = state.get("is_active", True)
