"""Problem 2: Multi-dimensional boundary estimator.

Replaces the 1D threshold model (score > θ) with a logistic regression
over all semantic dimensions.

Learn:
    P(refuse | x) = σ(w · x + b)

Exposes:
    - Feature importance (coefficient values)
    - Confidence intervals
    - Decision surface weights

Stays explainable — no black-box neural classifiers.
"""

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Feature axis labels
_FEATURE_NAMES: List[str] = [
    "instruction_score",
    "harmfulness_score",
    "procedurality_score",
    "jailbreak_score",
]


@dataclass
class MultiBoundaryEstimate:
    """Estimate from the multi-dimensional boundary model.

    Attributes
    ----------
    coefficients : Dict[str, float]
        Learned weights per feature dimension.
    intercept : float
        Bias term b.
    feature_names : List[str]
        Ordered feature names.
    num_observations : int
        Training data count.
    confidence_intervals : Dict[str, Tuple[float, float]]
        95% CI per coefficient.
    accuracy : float
        In-sample accuracy.
    decision_boundary : str
        Human-readable decision boundary expression.
    """
    coefficients: Dict[str, float]
    intercept: float
    feature_names: List[str]
    num_observations: int
    confidence_intervals: Dict[str, Tuple[float, float]]
    accuracy: float
    decision_boundary: str = ""

    def __post_init__(self) -> None:
        if not self.decision_boundary:
            self.decision_boundary = self._format_boundary()

    def _format_boundary(self) -> str:
        terms: List[str] = []
        for name in self.feature_names:
            w = self.coefficients.get(name, 0.0)
            if abs(w) > 0.01:
                terms.append(f"{w:.3f} * {name}")
        if abs(self.intercept) > 0.01:
            terms.append(f"{self.intercept:.3f}")
        if not terms:
            return "P(refuse) = σ(0.0)"
        expr = " + ".join(terms)
        return f"P(refuse) = σ({expr})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coefficients": {k: round(v, 4) for k, v in self.coefficients.items()},
            "intercept": round(self.intercept, 4),
            "num_observations": self.num_observations,
            "confidence_intervals": {
                k: (round(lo, 4), round(hi, 4))
                for k, (lo, hi) in self.confidence_intervals.items()
            },
            "accuracy": round(self.accuracy, 4),
            "decision_boundary": self.decision_boundary,
        }

    @property
    def feature_importance(self) -> Dict[str, float]:
        total = sum(abs(v) for v in self.coefficients.values())
        if total == 0:
            return {k: 0.0 for k in self.coefficients}
        return {k: abs(v) / total for k, v in self.coefficients.items()}


class MultiDimensionalBoundaryEstimator:
    """Logistic regression boundary estimator over multiple semantic dimensions.

    Learns P(refuse | instruction, harmfulness, procedurality, jailbreak)
    using sklearn's LogisticRegression with L2 regularization.

    Parameters
    ----------
    feature_names : List[str]
        Names of input features (default: instruction, harmfulness, etc.).
    C : float
        Inverse regularization strength (default 1.0).
    max_iter : int
        Maximum solver iterations (default 1000).
    min_observations : int
        Minimum observations before fitting (default 10).
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        C: float = 1.0,
        max_iter: int = 1000,
        min_observations: int = 10,
    ) -> None:
        self.feature_names = feature_names or _FEATURE_NAMES[:]
        self.C = C
        self.max_iter = max_iter
        self.min_observations = min_observations

        self._features: List[np.ndarray] = []
        self._outcomes: List[int] = []
        self._cached_estimate: Optional[MultiBoundaryEstimate] = None
        self._dirty: bool = True
        self._lock = threading.Lock()

    @property
    def num_observations(self) -> int:
        return len(self._outcomes)

    def observe(
        self,
        scores: Dict[str, float],
        outcome: int,
    ) -> None:
        """Record an observation.

        Parameters
        ----------
        scores : Dict[str, float]
            Score vector: {feature_name: score}.
        outcome : int
            1 = REFUSE, 0 = REFUSE_FAIL.
        """
        vec = np.array([scores.get(name, 0.0) for name in self.feature_names])
        with self._lock:
            self._features.append(vec)
            self._outcomes.append(1 if outcome else 0)
            self._dirty = True

    def observe_vector(
        self,
        score_vector: List[float],
        outcome: int,
    ) -> None:
        """Record observation from an ordered score vector."""
        assert len(score_vector) == len(self.feature_names)
        with self._lock:
            self._features.append(np.array(score_vector, dtype=np.float64))
            self._outcomes.append(1 if outcome else 0)
            self._dirty = True

    def estimate(self) -> MultiBoundaryEstimate:
        """Fit the logistic regression and return the estimate."""
        if not self._dirty and self._cached_estimate is not None:
            return self._cached_estimate
        with self._lock:
            est = self._fit()
            self._cached_estimate = est
            self._dirty = False
        return est

    def predict(self, scores: Dict[str, float]) -> float:
        """Predict P(refuse = 1) given a score vector.

        Parameters
        ----------
        scores : Dict[str, float]
            Score vector.

        Returns
        -------
        float
            Predicted probability of refusal in [0, 1].
        """
        est = self.estimate()
        vec = np.array([scores.get(name, 0.0) for name in self.feature_names])
        w = np.array([est.coefficients.get(n, 0.0) for n in self.feature_names])
        z = float(np.dot(vec, w) + est.intercept)
        return 1.0 / (1.0 + math.exp(-z))

    def _fit(self) -> MultiBoundaryEstimate:
        n = len(self._outcomes)
        if n < self.min_observations:
            return MultiBoundaryEstimate(
                coefficients={name: 0.0 for name in self.feature_names},
                intercept=0.0,
                feature_names=self.feature_names[:],
                num_observations=n,
                confidence_intervals={
                    name: (-1.0, 1.0) for name in self.feature_names
                },
                accuracy=0.5,
                decision_boundary="Insufficient data",
            )
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            logger.warning("sklearn not available; using heuristic fit")
            return self._heuristic_fit()

        X = np.vstack(self._features)
        y = np.array(self._outcomes)
        clf = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs",
            fit_intercept=True, random_state=42,
        )
        clf.fit(X, y)
        coefs = clf.coef_[0]
        intercept = float(clf.intercept_[0])
        accuracy = float(clf.score(X, y))

        # Bootstrap confidence intervals
        ci: Dict[str, Tuple[float, float]] = {}
        n_bootstrap = min(200, max(20, n // 2))
        bootstrap_coefs: List[np.ndarray] = []
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            Xb = X[idx]
            yb = y[idx]
            try:
                bc = LogisticRegression(
                    C=self.C, max_iter=self.max_iter,
                    solver="lbfgs", fit_intercept=True,
                )
                bc.fit(Xb, yb)
                bootstrap_coefs.append(bc.coef_[0])
            except Exception:
                continue

        if bootstrap_coefs:
            boot_arr = np.array(bootstrap_coefs)
            for i, name in enumerate(self.feature_names):
                lo = float(np.percentile(boot_arr[:, i], 2.5))
                hi = float(np.percentile(boot_arr[:, i], 97.5))
                ci[name] = (lo, hi)
        else:
            ci = {name: (float(coefs[i]) - 0.1, float(coefs[i]) + 0.1)
                  for i, name in enumerate(self.feature_names)}

        return MultiBoundaryEstimate(
            coefficients={
                name: float(coefs[i])
                for i, name in enumerate(self.feature_names)
            },
            intercept=intercept,
            feature_names=self.feature_names[:],
            num_observations=n,
            confidence_intervals=ci,
            accuracy=accuracy,
        )

    def _heuristic_fit(self) -> MultiBoundaryEstimate:
        """Fallback heuristic when sklearn is unavailable."""
        n = len(self._outcomes)
        if n < 3:
            return MultiBoundaryEstimate(
                coefficients={name: 0.0 for name in self.feature_names},
                intercept=0.0, feature_names=self.feature_names[:],
                num_observations=n,
                confidence_intervals={name: (-1.0, 1.0) for name in self.feature_names},
                accuracy=0.5,
                decision_boundary="Insufficient data",
            )
        X = np.vstack(self._features)
        y = np.array(self._outcomes)
        # Closed-form approximation via linear regression on logit
        y_clipped = np.clip(y, 0.001, 0.999)
        logit_y = np.log(y_clipped / (1.0 - y_clipped))
        try:
            X_aug = np.hstack([X, np.ones((n, 1))])
            theta = np.linalg.lstsq(X_aug, logit_y, rcond=None)[0]
            coefs = theta[:-1]
            intercept = float(theta[-1])
        except Exception:
            coefs = np.zeros(len(self.feature_names))
            intercept = 0.0
        preds = 1.0 / (1.0 + np.exp(-(X @ coefs + intercept)))
        accuracy = float(np.mean((preds > 0.5) == y))
        return MultiBoundaryEstimate(
            coefficients={
                name: float(coefs[i])
                for i, name in enumerate(self.feature_names)
            },
            intercept=intercept,
            feature_names=self.feature_names[:],
            num_observations=n,
            confidence_intervals={
                name: (float(coefs[i]) - 0.2, float(coefs[i]) + 0.2)
                for i, name in enumerate(self.feature_names)
            },
            accuracy=accuracy,
        )

    def reset(self) -> None:
        with self._lock:
            self._features.clear()
            self._outcomes.clear()
            self._cached_estimate = None
            self._dirty = True
