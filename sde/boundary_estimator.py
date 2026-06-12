"""Stage 2: Bayesian boundary estimation for semantic scores.

Learns a decision boundary P(θ | data) using a Beta-Bernoulli model.
Given semantic score s(prompt) and outcome y ∈ {0=ACCEPT, 1=REFUSE},
we model:

    y ~ Bernoulli(σ(s - θ))

where σ is the step function: y=1 iff s > θ.

The posterior over θ is approximated using a discretised Beta distribution
over candidate threshold values.

This is fundamentally different from Version Space posterior:
- Version Space:  P(program | data)   — which program is correct
- SDE:            P(θ | data)          — where is the boundary
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BoundaryEstimate:
    """The current estimate of a semantic decision boundary.

    Attributes
    ----------
    primitive_name : str
        Name of the semantic score primitive being estimated.
    posterior_mean : float
        Expected value of θ — the most likely threshold.
    posterior_std : float
        Standard deviation of θ — uncertainty (0 = certain).
    credible_interval : Tuple[float, float]
        The 95% credible interval for θ.
    evidence_weight : float
        Total number of observations contributing to this estimate.
    is_reliable : bool
        True when uncertainty is low enough to use.
    score_bound_low : float
        Lowest score seen in observations.
    score_bound_high : float
        Highest score seen in observations.
    observations : List[Tuple[float, int]]
        Raw (score, outcome) pairs that formed this estimate.
    """
    primitive_name: str
    posterior_mean: float
    posterior_std: float
    credible_interval: Tuple[float, float]
    evidence_weight: float
    is_reliable: bool
    score_bound_low: float
    score_bound_high: float
    observations: List[Tuple[float, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primitive_name": self.primitive_name,
            "posterior_mean": round(self.posterior_mean, 4),
            "posterior_std": round(self.posterior_std, 4),
            "credible_interval": [round(self.credible_interval[0], 4),
                                  round(self.credible_interval[1], 4)],
            "evidence_weight": round(self.evidence_weight, 1),
            "is_reliable": self.is_reliable,
            "score_range": [round(self.score_bound_low, 4),
                            round(self.score_bound_high, 4)],
            "num_observations": len(self.observations),
        }


# ---------------------------------------------------------------------------
# Bayesian Boundary Estimator
# ---------------------------------------------------------------------------

class BayesianBoundaryEstimator:
    """Estimates a decision threshold θ for a single semantic score primitive.

    Uses a discretised Beta-Bernoulli model:
        For each candidate threshold t, maintain Beta(α_t, β_t).
        When (score, outcome) is observed:
            If score > t and outcome = 1 (REFUSE):  α_t += 1
            If score > t and outcome = 0 (ACCEPT):   β_t += 1
            (vice versa for score <= t)
        Posterior mean = E[θ] ≈ Σ(t * P(θ=t)) over discretised grid.

    Parameters
    ----------
    primitive_name : str
        Name of the semantic score primitive.
    prior_alpha : float
        Beta prior α (default 1 = uniform).
    prior_beta : float
        Beta prior β (default 1 = uniform).
    grid_size : int
        Number of discrete threshold candidates (default 50).
    min_observations_for_reliable : int
        Observations needed before estimate is considered reliable (default 10).
    max_uncertainty_for_reliable : float
        Max posterior_std for reliability (default 0.08).
    """
    def __init__(
        self,
        primitive_name: str,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        grid_size: int = 50,
        min_observations_for_reliable: int = 10,
        max_uncertainty_for_reliable: float = 0.08,
    ) -> None:
        self.primitive_name = primitive_name
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.grid_size = grid_size
        self.min_observations = min_observations_for_reliable
        self.max_uncertainty = max_uncertainty_for_reliable

        self._observations: List[Tuple[float, int]] = []
        self._grid: np.ndarray = np.linspace(0.0, 1.0, grid_size)
        self._alphas: np.ndarray = np.full(grid_size, prior_alpha, dtype=np.float64)
        self._betas: np.ndarray = np.full(grid_size, prior_beta, dtype=np.float64)
        self._cached_estimate: Optional[BoundaryEstimate] = None
        self._dirty: bool = True
        self._direction: str = "positive"

    @property
    def num_observations(self) -> int:
        return len(self._observations)

    @property
    def direction(self) -> str:
        return self._direction

    def estimate_direction(self) -> str:
        """Compute Pearson correlation between scores and outcomes.

        Returns "positive", "negative", or "unknown".
        """
        if len(self._observations) < 5:
            return "positive"

        scores = np.array([s for s, _ in self._observations])
        outcomes = np.array([o for _, o in self._observations])

        if np.std(scores) < 1e-10 or np.std(outcomes) < 1e-10:
            return "positive"

        corr = np.corrcoef(scores, outcomes)[0, 1]
        logger.info("SDE direction: correlation=%.4f between scores and outcomes", corr)

        if abs(corr) < 0.1:
            logger.warning("SDE direction: weak correlation (%.4f), using default positive", corr)
            return "positive"

        self._direction = "positive" if corr > 0 else "negative"
        logger.info("SDE direction: detected %s (corr=%.4f)", self._direction, corr)
        return self._direction

    def observe(self, score: float, outcome: int) -> None:
        """Update boundary belief with a new (score, outcome) observation.

        Parameters
        ----------
        score : float
            Semantic score in [0, 1].
        outcome : int
            0 = ACCEPT, 1 = REFUSE.
        """
        score = max(0.0, min(1.0, float(score)))
        outcome = 1 if outcome else 0
        self._observations.append((score, outcome))

        for i, t in enumerate(self._grid):
            if score > t:
                if outcome == 1:
                    self._alphas[i] += 1.0
                else:
                    self._betas[i] += 1.0
            else:
                if outcome == 0:
                    self._alphas[i] += 1.0
                else:
                    self._betas[i] += 1.0

        self._dirty = True

        if len(self._observations) % 5 == 0:
            self.estimate_direction()

    def observe_batch(self, observations: List[Tuple[float, int]]) -> None:
        for score, outcome in observations:
            self.observe(score, outcome)

    def estimate(self) -> BoundaryEstimate:
        """Compute the current boundary estimate."""
        if not self._dirty and self._cached_estimate is not None:
            return self._cached_estimate
        return self._compute()

    def _compute(self) -> BoundaryEstimate:
        n = len(self._observations)
        if n == 0:
            result = BoundaryEstimate(
                primitive_name=self.primitive_name,
                posterior_mean=0.5,
                posterior_std=0.289,
                credible_interval=(0.0, 1.0),
                evidence_weight=0.0,
                is_reliable=False,
                score_bound_low=0.0,
                score_bound_high=1.0,
                observations=[],
            )
            self._cached_estimate = result
            self._dirty = False
            return result

        posteriors = np.zeros(self.grid_size)
        for i in range(self.grid_size):
            posteriors[i] = np.exp(
                self._alphas[i] * np.log(self._grid[i] + 1e-15) +
                self._betas[i] * np.log(1.0 - self._grid[i] + 1e-15)
            )
        posteriors = posteriors / (posteriors.sum() + 1e-15)
        mean = float(np.sum(posteriors * self._grid))
        variance = float(np.sum(posteriors * (self._grid - mean) ** 2))
        std = float(math.sqrt(max(variance, 1e-10)))

        cumulative = np.cumsum(posteriors)
        lower_idx = int(np.searchsorted(cumulative, 0.025))
        upper_idx = int(np.searchsorted(cumulative, 0.975))
        ci = (float(self._grid[min(lower_idx, self.grid_size - 1)]),
              float(self._grid[min(upper_idx, self.grid_size - 1)]))

        logger.info("SDE direction: using %s direction for %s", self._direction, self.primitive_name)
        if self._direction == "negative":
            mean = 1.0 - mean
            ci = (1.0 - ci[1], 1.0 - ci[0])

        scores = [s for s, _ in self._observations]
        evidence_weight = float(n)
        is_reliable = (
            n >= self.min_observations and
            std <= self.max_uncertainty
        )

        result = BoundaryEstimate(
            primitive_name=self.primitive_name,
            posterior_mean=mean,
            posterior_std=std,
            credible_interval=ci,
            evidence_weight=evidence_weight,
            is_reliable=is_reliable,
            score_bound_low=min(scores) if scores else 0.0,
            score_bound_high=max(scores) if scores else 1.0,
            observations=self._observations.copy(),
        )
        self._cached_estimate = result
        self._dirty = False
        return result

    def uncertainty(self) -> float:
        """Return current uncertainty (posterior std)."""
        return self.estimate().posterior_std

    def reliability(self) -> bool:
        """Return whether the estimate is reliable."""
        return self.estimate().is_reliable

    def score_region(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Return the score region worth exploring for this primitive.

        Focuses on the 95% credible interval plus margin.
        """
        est = self.estimate()
        low = max(0.0, est.credible_interval[0] - 0.15)
        high = min(1.0, est.credible_interval[1] + 0.15)
        return (low, high)

    def generate_target_scores(self, n: int = 7) -> List[float]:
        """Generate target scores for intervention design.

        Spreads points across the uncertain region near the boundary,
        with most points near the posterior mean.
        """
        est = self.estimate()
        if not est.is_reliable:
            mid = est.posterior_mean
            spacing = max(0.05, est.posterior_std * 2.0)
            low = max(0.05, mid - spacing * 2)
            high = min(0.95, mid + spacing * 2)
            targets = list(np.linspace(low, high, n))
            return [round(float(t), 4) for t in targets]

        ci_low, ci_high = est.credible_interval
        margin = max(0.05, est.posterior_std)
        low = max(0.0, ci_low - margin)
        high = min(1.0, ci_high + margin)
        targets = list(np.linspace(low, high, min(n, 9)))
        return [round(float(t), 4) for t in targets]

    def reset(self) -> None:
        self._observations = []
        self._alphas = np.full(self.grid_size, self.prior_alpha, dtype=np.float64)
        self._betas = np.full(self.grid_size, self.prior_beta, dtype=np.float64)
        self._dirty = True
        self._cached_estimate = None
