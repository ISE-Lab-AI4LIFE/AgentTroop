"""Transfer Speed metrics — measures cross-model transfer learning efficiency.

RQ-TL-1: Cold-start queries vs warm-start queries to reach accuracy > 90%
RQ-TL-2: Entropy reduction rate with vs without scientific memory transfer.

Metrics::

    transfer_speedup = cold_queries / warm_queries
    transfer_accuracy_boost = warm_init_accuracy - cold_init_accuracy
    queries_to_90pct(victim, warm_start)  # interventions needed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TransferMetrics:
    """Metrics from a transfer learning evaluation."""

    cold_queries_to_threshold: int = 0
    warm_queries_to_threshold: int = 0
    transfer_speedup: float = 1.0
    cold_init_accuracy: float = 0.0
    warm_init_accuracy: float = 0.0
    accuracy_boost: float = 0.0
    cold_entropy_history: List[float] = field(default_factory=list)
    warm_entropy_history: List[float] = field(default_factory=list)
    cold_accuracy_history: List[float] = field(default_factory=list)
    warm_accuracy_history: List[float] = field(default_factory=list)
    queries_per_campaign: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cold_queries_to_threshold": self.cold_queries_to_threshold,
            "warm_queries_to_threshold": self.warm_queries_to_threshold,
            "transfer_speedup": round(self.transfer_speedup, 3),
            "cold_init_accuracy": round(self.cold_init_accuracy, 3),
            "warm_init_accuracy": round(self.warm_init_accuracy, 3),
            "accuracy_boost": round(self.accuracy_boost, 3),
            "queries_per_campaign": self.queries_per_campaign,
        }


def evaluate_transfer(
    cold_accuracy_curve: List[float],
    warm_accuracy_curve: List[float],
    cold_entropy: Optional[List[float]] = None,
    warm_entropy: Optional[List[float]] = None,
    accuracy_threshold: float = 0.9,
) -> TransferMetrics:
    """Compare cold-start vs warm-start accuracy curves.

    Parameters
    ----------
    cold_accuracy_curve : list of float
        Accuracy per intervention (cold start — no prior knowledge).
    warm_accuracy_curve : list of float
        Accuracy per intervention (warm start — with Scientific Memory).
    cold_entropy : list of float, optional
    warm_entropy : list of float, optional
    accuracy_threshold : float

    Returns
    -------
    TransferMetrics
    """
    metrics = TransferMetrics()

    # Queries to reach threshold
    for i, acc in enumerate(cold_accuracy_curve):
        if acc >= accuracy_threshold:
            metrics.cold_queries_to_threshold = i + 1
            break
    for i, acc in enumerate(warm_accuracy_curve):
        if acc >= accuracy_threshold:
            metrics.warm_queries_to_threshold = i + 1
            break

    metrics.cold_init_accuracy = cold_accuracy_curve[0] if cold_accuracy_curve else 0.0
    metrics.warm_init_accuracy = warm_accuracy_curve[0] if warm_accuracy_curve else 0.0
    metrics.accuracy_boost = metrics.warm_init_accuracy - metrics.cold_init_accuracy

    if metrics.warm_queries_to_threshold > 0 and metrics.cold_queries_to_threshold > 0:
        metrics.transfer_speedup = (
            metrics.cold_queries_to_threshold / metrics.warm_queries_to_threshold
        )

    if cold_entropy:
        metrics.cold_entropy_history = cold_entropy
    if warm_entropy:
        metrics.warm_entropy_history = warm_entropy

    metrics.cold_accuracy_history = cold_accuracy_curve
    metrics.warm_accuracy_history = warm_accuracy_curve
    metrics.queries_per_campaign = max(len(cold_accuracy_curve), len(warm_accuracy_curve))

    logger.info(
        "Transfer: cold=%d warm=%d speedup=%.2fx boost=%.3f",
        metrics.cold_queries_to_threshold,
        metrics.warm_queries_to_threshold,
        metrics.transfer_speedup,
        metrics.accuracy_boost,
    )
    return metrics


class TransferSpeedMetric:
    """Metric class wrapping evaluate_transfer for the evaluation pipeline.

    Provides the ``compute()`` interface expected by RQ3Evaluator.
    """

    def __init__(self, episodic_memory: Any) -> None:
        self._episodic = episodic_memory

    def compute(
        self,
        prior_campaign_id: str,
        target_campaign_id: str,
        prior_experiment_id: Optional[str] = None,
        target_experiment_id: Optional[str] = None,
        threshold: float = 0.9,
    ) -> Dict[str, Any]:
        """Compute transfer speed metrics between two campaigns.

        Builds cold-start (prior) and warm-start (target) accuracy curves
        from episodic memory, then delegates to ``evaluate_transfer()``.
        """
        cold_curve = self._build_accuracy_curve(
            prior_campaign_id, prior_experiment_id, threshold,
        )
        warm_curve = self._build_accuracy_curve(
            target_campaign_id, target_experiment_id, threshold,
        )
        metrics = evaluate_transfer(cold_curve, warm_curve, accuracy_threshold=threshold)
        return metrics.to_dict()

    def _build_accuracy_curve(
        self,
        campaign_id: str,
        experiment_id: Optional[str] = None,
        threshold: float = 0.9,
    ) -> List[float]:
        """Build accuracy curve from episodic memory for a campaign.

        Simplified: returns a mock curve when ground-truth isn't available.
        Subclasses can override with actual victim-proxy comparisons.
        """
        try:
            episodes = self._episodic.get_episodes_by_campaign(campaign_id)
            if not episodes:
                return []
            curve: List[float] = []
            for i, ep in enumerate(episodes):
                acc = 1.0 if ep.outcome == 0 else 0.0
                curve.append(sum(curve[-3:] + [acc]) / min(len(curve) + 1, 3))
            return curve
        except Exception:
            return []
