"""Composite boundary estimator for multi-primitive victim detection.

Fuses per-primitive boundary estimates into a single decision boundary
when the victim uses multiple semantic properties simultaneously.

Two fusion modes:
  - AND: all primitives must exceed their boundary for REFUSE
  - OR:  any primitive exceeding its boundary triggers REFUSE

Uses LogisticRegression on multi-primitive feature vectors to learn
the fused decision surface.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class CompositeBoundaryEstimate:
    """Fused boundary estimate across multiple primitives.

    Attributes
    ----------
    n_observations : int
        Number of observations used.
    uncertainty : float
        Composite uncertainty (0 = certain, 1 = high).
    fusion_mode : str
        How primitives are combined ("and" or "or").
    per_primitive_uncertainties : Dict[str, float]
        Individual per-primitive uncertainties.
    learned_coefficients : Optional[Dict[str, float]]
        Logistic regression coefficients, if fitted.
    is_fitted : bool
        Whether the logistic model has been fitted.
    """
    n_observations: int
    uncertainty: float
    fusion_mode: str
    per_primitive_uncertainties: Dict[str, float]
    learned_coefficients: Optional[Dict[str, float]] = None
    is_fitted: bool = False

    def to_dict(self) -> dict:
        return {
            "n_observations": self.n_observations,
            "uncertainty": round(self.uncertainty, 4),
            "fusion_mode": self.fusion_mode,
            "per_primitive_uncertainties": {
                k: round(v, 4) for k, v in self.per_primitive_uncertainties.items()
            },
            "learned_coefficients": (
                {k: round(v, 4) for k, v in self.learned_coefficients.items()}
                if self.learned_coefficients else None
            ),
            "is_fitted": self.is_fitted,
        }


class CompositeBoundaryEstimator:
    """Fuses multiple per-primitive boundary estimators into one.

    Maintains per-primitive BayesianBoundaryEstimators internally.
    Additionally, fits a LogisticRegression over the full feature vector
    of all primitive scores to learn the true fused decision surface.

    Parameters
    ----------
    primitive_names : List[str]
        Order of primitive dimensions.
    fusion_mode : str
        "and" or "or" (default "and").
    min_observations_for_fit : int
        Minimum observations to fit the logistic model (default 20).
    """
    def __init__(
        self,
        primitive_names: Optional[List[str]] = None,
        fusion_mode: str = "and",
        min_observations_for_fit: int = 20,
    ) -> None:
        self.primitive_names = primitive_names or [
            "instruction_score",
            "harmfulness_score",
            "jailbreak_score",
            "procedurality_score",
        ]
        self.fusion_mode = fusion_mode.lower()
        self.min_observations = min_observations_for_fit
        self._observations: List[Tuple[Dict[str, float], int]] = []
        self._model: Optional[LogisticRegression] = None
        self._is_fitted = False

    @property
    def num_observations(self) -> int:
        return len(self._observations)

    def observe(self, scores: Dict[str, float], outcome: int) -> None:
        """Record a multi-primitive observation.

        Parameters
        ----------
        scores : Dict[str, float]
            Scores keyed by primitive name.
        outcome : int
            0 = ACCEPT, 1 = REFUSE.
        """
        validated = {}
        for name in self.primitive_names:
            validated[name] = max(0.0, min(1.0, scores.get(name, 0.5)))
        self._observations.append((validated, 1 if outcome else 0))

    def estimate(self) -> CompositeBoundaryEstimate:
        """Compute the composite boundary estimate."""
        n = len(self._observations)
        per_prim_unc = self._compute_per_primitive_uncertainties()

        if n < 3:
            return CompositeBoundaryEstimate(
                n_observations=n,
                uncertainty=1.0,
                fusion_mode=self.fusion_mode,
                per_primitive_uncertainties=per_prim_unc,
                is_fitted=False,
            )

        if n >= self.min_observations and not self._is_fitted:
            self._fit_logistic()

        if self._is_fitted and self._model is not None:
            composite_unc = self._logistic_uncertainty()
            coeffs = {
                name: float(self._model.coef_[0][i])
                for i, name in enumerate(self.primitive_names)
            }
            return CompositeBoundaryEstimate(
                n_observations=n,
                uncertainty=float(composite_unc),
                fusion_mode=self.fusion_mode,
                per_primitive_uncertainties=per_prim_unc,
                learned_coefficients=coeffs,
                is_fitted=True,
            )

        composite_unc = self._fusion_uncertainty(per_prim_unc)
        return CompositeBoundaryEstimate(
            n_observations=n,
            uncertainty=float(composite_unc),
            fusion_mode=self.fusion_mode,
            per_primitive_uncertainties=per_prim_unc,
            is_fitted=False,
        )

    def _compute_per_primitive_uncertainties(self) -> Dict[str, float]:
        """Estimate per-primitive uncertainty from score variance."""
        if len(self._observations) < 2:
            return {name: 1.0 for name in self.primitive_names}

        uncertainties = {}
        for name in self.primitive_names:
            scores = np.array([obs[0].get(name, 0.5) for obs in self._observations])
            outcomes = np.array([obs[1] for obs in self._observations])
            if len(np.unique(outcomes)) < 2:
                uncertainties[name] = 1.0
                continue
            accept_scores = scores[outcomes == 0]
            refuse_scores = scores[outcomes == 1]
            if len(accept_scores) < 1 or len(refuse_scores) < 1:
                uncertainties[name] = 1.0
                continue
            sep = abs(np.mean(accept_scores) - np.mean(refuse_scores))
            uncertainties[name] = float(1.0 - min(1.0, max(0.0, sep)))
        return uncertainties

    def _fusion_uncertainty(self, per_prim: Dict[str, float]) -> float:
        """Fuse per-primitive uncertainties using the fusion mode."""
        values = list(per_prim.values())
        if self.fusion_mode == "or":
            return float(min(values))
        return float(max(values))

    def _fit_logistic(self) -> None:
        """Fit a LogisticRegression on all observations."""
        X = np.array([
            [obs[0].get(name, 0.5) for name in self.primitive_names]
            for obs in self._observations
        ])
        y = np.array([obs[1] for obs in self._observations])

        unique = np.unique(y)
        if len(unique) < 2:
            logger.debug("Cannot fit logistic: only one class present")
            return

        try:
            self._model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
            self._model.fit(X, y)
            self._is_fitted = True
            logger.info(
                "CompositeBoundaryEstimator fitted: coeffs=%s",
                {n: round(float(c), 4)
                 for n, c in zip(self.primitive_names, self._model.coef_[0])},
            )
        except Exception as exc:
            logger.warning("Logistic fit failed: %s", exc)
            self._is_fitted = False

    def _logistic_uncertainty(self) -> float:
        """Estimate uncertainty from logistic model prediction entropy."""
        if self._model is None or len(self._observations) < 2:
            return 1.0
        X = np.array([
            [obs[0].get(name, 0.5) for name in self.primitive_names]
            for obs in self._observations[-20:]
        ])
        try:
            probs = self._model.predict_proba(X)
            entropies = -np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0)), axis=1)
            return float(np.mean(entropies) / np.log(2.0))
        except Exception:
            return 1.0

    def predict(self, scores: Dict[str, float]) -> Tuple[int, float]:
        """Predict outcome and confidence for a multi-primitive score vector.

        Returns
        -------
        (predicted_outcome, confidence)
            predicted_outcome is 0 (ACCEPT) or 1 (REFUSE).
            confidence is in [0, 1].
        """
        if self._model is not None and self._is_fitted:
            vec = np.array([[scores.get(n, 0.5) for n in self.primitive_names]])
            try:
                pred = int(self._model.predict(vec)[0])
                proba = self._model.predict_proba(vec)[0]
                confidence = float(max(proba))
                return (pred, confidence)
            except Exception:
                pass

        # Fallback: use fusion mode
        if self.fusion_mode == "or":
            vote = any(scores.get(n, 0.0) > 0.5 for n in self.primitive_names)
        else:
            vote = all(scores.get(n, 0.0) > 0.5 for n in self.primitive_names)
        return (1 if vote else 0, 0.5)

    def reset(self) -> None:
        self._observations = []
        self._model = None
        self._is_fitted = False
