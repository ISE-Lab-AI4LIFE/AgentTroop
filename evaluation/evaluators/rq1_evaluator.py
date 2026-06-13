from __future__ import annotations

import logging
from typing import Callable, Optional

from knowledge.episodic import EpisodicMemory
from evaluation.metrics.intervention_efficiency import InterventionEfficiencyMetric

logger = logging.getLogger(__name__)


class RQ1Evaluator:
    """RQ1: Are targeted interventions more query-efficient than random probing?

    Measures the number of interventions needed to reach accuracy >85%.
    Accuracy is the program's running prediction accuracy on a held-out
    validation set with known labels.
    """

    def __init__(self, episodic_memory: EpisodicMemory) -> None:
        self._metric = InterventionEfficiencyMetric(episodic_memory)

    def evaluate(
        self,
        campaign_id: str,
        experiment_id: Optional[str] = None,
        threshold: float = 0.85,
        predict_fn: Optional[Callable[[str], int]] = None,
    ) -> dict:
        if predict_fn is None:
            logger.error("RQ1 skipped: predict_fn is None (program unavailable)")
            return {
                "campaign_id": campaign_id,
                "rq": "RQ1",
                "error": "predict_fn is None — program unavailable, RQ1 cannot be evaluated",
                "interventions_to_threshold": -1,
                "best_accuracy": 0.0,
                "threshold": threshold,
                "reached": False,
            }
        result = self._metric.compute(
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            threshold=threshold,
            predict_fn=predict_fn,
        )
        result["rq"] = "RQ1"
        logger.info(
            "RQ1: campaign=%s intv_to_threshold=%d best_acc=%.4f reached=%s",
            campaign_id, result["interventions_to_threshold"],
            result["best_accuracy"], result["reached"],
        )
        return result
