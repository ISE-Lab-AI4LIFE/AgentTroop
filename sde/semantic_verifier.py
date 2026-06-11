"""Stage 6: Semantic verification.

Verifies boundary consistency, calibration, and monotonicity.
Detects when semantic score quality degrades or collapses.

This is independent from the ProgramVerifier.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .boundary_estimator import BayesianBoundaryEstimator, BoundaryEstimate
from .score_primitives import (
    _compute_instruction_score, _compute_harmfulness_score,
    _compute_procedurality_score, _compute_jailbreak_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BoundaryConsistencyReport:
    """Report on the quality of a semantic boundary estimate.

    Attributes
    ----------
    primitive_name : str
    is_consistent : bool
        Whether the boundary is internally consistent.
    calibration_error : float
        Mean absolute calibration error (0 = perfect).
    monotonicity_score : float
        How well score predicts outcome (1 = perfect monotonic).
    collapse_detected : bool
        Whether the score primitive has collapsed to a constant.
    collapse_score : float
        Degree of collapse (0 = no collapse, 1 = fully collapsed).
    num_pass : int
        Observations consistent with boundary.
    num_fail : int
        Observations violating boundary consistency.
    details : Dict
        Additional diagnostic information.
    """
    primitive_name: str
    is_consistent: bool
    calibration_error: float
    monotonicity_score: float
    collapse_detected: bool
    collapse_score: float
    num_pass: int
    num_fail: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primitive_name": self.primitive_name,
            "is_consistent": self.is_consistent,
            "calibration_error": round(self.calibration_error, 4),
            "monotonicity_score": round(self.monotonicity_score, 4),
            "collapse_detected": self.collapse_detected,
            "collapse_score": round(self.collapse_score, 4),
            "num_pass": self.num_pass,
            "num_fail": self.num_fail,
        }


# ---------------------------------------------------------------------------
# Semantic Verifier
# ---------------------------------------------------------------------------

_SCORE_FN_MAP: Dict[str, Callable[[str], float]] = {
    "instruction_score": _compute_instruction_score,
    "harmfulness_score": _compute_harmfulness_score,
    "procedurality_score": _compute_procedurality_score,
    "jailbreak_score": _compute_jailbreak_score,
}


class SemanticVerifier:
    """Verifies the quality of semantic boundary estimates.

    Checks:
    1. Boundary consistency — observations agree with estimated threshold
    2. Calibration — P(REFUSE) vs score is monotonic
    3. Collapse detection — score stays constant across diverse prompts
    4. Score-response monotonicity — higher scores correlate with REFUSE
    """

    def __init__(
        self,
        score_functions: Optional[Dict[str, Callable[[str], float]]] = None,
        min_pass_rate: float = 0.75,
        max_calibration_error: float = 0.15,
        max_collapse_score: float = 0.15,
    ) -> None:
        self.score_functions = score_functions or _SCORE_FN_MAP
        self.min_pass_rate = min_pass_rate
        self.max_calibration_error = max_calibration_error
        self.max_collapse_score = max_collapse_score

    def verify_boundary(
        self,
        estimator: BayesianBoundaryEstimator,
        observations: Optional[List[Tuple[str, float, int]]] = None,
    ) -> BoundaryConsistencyReport:
        """Verify boundary consistency for an estimator.

        Parameters
        ----------
        estimator : BayesianBoundaryEstimator
            The boundary estimator to verify.
        observations : List[Tuple[str, float, int]], optional
            (prompt, score, outcome) triples. Uses estimator's internal
            data if not provided.

        Returns
        -------
        BoundaryConsistencyReport
        """
        est = estimator.estimate()
        if observations is None:
            obs = [(f"obs_{i}", s, o) for i, (s, o) in enumerate(est.observations)]
        else:
            obs = observations

        if len(obs) < 3:
            return BoundaryConsistencyReport(
                primitive_name=est.primitive_name,
                is_consistent=True,
                calibration_error=0.0,
                monotonicity_score=1.0,
                collapse_detected=False,
                collapse_score=0.0,
                num_pass=0,
                num_fail=0,
                details={"reason": "insufficient_observations"},
            )

        # Boundary consistency
        theta = est.posterior_mean
        num_pass = 0
        num_fail = 0
        for _, score, outcome in obs:
            predicted = 1 if score > theta else 0
            if predicted == outcome:
                num_pass += 1
            else:
                num_fail += 1
        pass_rate = num_pass / max(len(obs), 1)
        is_consistent = pass_rate >= self.min_pass_rate

        # Calibration error
        scores = np.array([s for _, s, _ in obs])
        outcomes = np.array([o for _, _, o in obs])
        calibration_error = self._compute_calibration_error(scores, outcomes)

        # Monotonicity
        monotonicity = self._compute_monotonicity(scores, outcomes)

        # Collapse detection
        collapse_score, collapse_detected = self._detect_collapse(
            est, observations=obs
        )

        return BoundaryConsistencyReport(
            primitive_name=est.primitive_name,
            is_consistent=is_consistent,
            calibration_error=round(float(calibration_error), 4),
            monotonicity_score=round(float(monotonicity), 4),
            collapse_detected=collapse_detected,
            collapse_score=round(float(collapse_score), 4),
            num_pass=num_pass,
            num_fail=num_fail,
            details={
                "pass_rate": round(float(pass_rate), 4),
                "theta": round(float(theta), 4),
                "num_observations": len(obs),
                "std": round(float(est.posterior_std), 4),
            },
        )

    def verify_primitive(
        self,
        primitive_name: str,
        prompts: List[str],
    ) -> Dict[str, Any]:
        """Verify that a semantic primitive produces non-trivial scores.

        Checks for collapse and dynamic range.

        Parameters
        ----------
        primitive_name : str
        prompts : List[str]
            Diverse set of prompts to test.

        Returns
        -------
        Dict with score statistics.
        """
        score_fn = self.score_functions.get(primitive_name)
        if score_fn is None:
            return {"error": f"Unknown primitive: {primitive_name}"}

        scores = [score_fn(p) for p in prompts]
        unique_scores = len(set(scores))
        score_range = max(scores) - min(scores) if scores else 0.0
        collapsed = unique_scores <= 1 or score_range < 0.05

        return {
            "primitive_name": primitive_name,
            "num_prompts": len(prompts),
            "scores": [round(s, 4) for s in scores],
            "score_min": round(min(scores), 4),
            "score_max": round(max(scores), 4),
            "score_range": round(score_range, 4),
            "unique_scores": unique_scores,
            "collapsed": collapsed,
            "mean_score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "std_score": round(float(np.std(scores)), 4) if scores else 0.0,
        }

    @staticmethod
    def _compute_calibration_error(
        scores: np.ndarray, outcomes: np.ndarray
    ) -> float:
        """Compute calibration error using binned score intervals."""
        if len(scores) < 5:
            return 0.0
        n_bins = min(5, len(scores) // 2)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        errors: List[float] = []
        for i in range(n_bins):
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
            if mask.sum() < 2:
                continue
            bin_refuse_rate = outcomes[mask].mean()
            bin_mid = (bin_edges[i] + bin_edges[i + 1]) / 2.0
            errors.append(abs(bin_refuse_rate - bin_mid))
        return float(np.mean(errors)) if errors else 0.0

    @staticmethod
    def _compute_monotonicity(
        scores: np.ndarray, outcomes: np.ndarray
    ) -> float:
        """Compute how monotonically outcome increases with score.

        Uses Kendall Tau rank correlation between score and outcome.
        Returns 1.0 if perfectly monotonic, 0.0 if independent.
        """
        if len(scores) < 3:
            return 1.0
        concordant = 0
        discordant = 0
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                score_diff = scores[i] - scores[j]
                outcome_diff = outcomes[i] - outcomes[j]
                if score_diff * outcome_diff > 0:
                    concordant += 1
                elif score_diff * outcome_diff < 0:
                    discordant += 1
        total = concordant + discordant
        if total == 0:
            return 0.5
        tau = (concordant - discordant) / total
        return float((tau + 1.0) / 2.0)

    @staticmethod
    def _detect_collapse(
        est: BoundaryEstimate,
        observations: Optional[List[Tuple[str, float, int]]] = None,
    ) -> Tuple[float, bool]:
        """Detect if a score primitive has collapsed.

        Collapse = all scores near 0 or all near 1.
        Returns (collapse_score, is_collapsed).
        """
        if not est.observations:
            return 0.0, False
        scores = [s for s, _ in est.observations]
        if not scores:
            return 0.0, False
        mean_score = float(np.mean(scores))
        score_std = float(np.std(scores))
        if score_std < 0.05 and (mean_score < 0.1 or mean_score > 0.9):
            collapse_score = max(0.0, 1.0 - score_std * 20)
            return collapse_score, True
        return score_std / max(mean_score, 1 - mean_score, 0.01), False
