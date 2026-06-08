from __future__ import annotations

import logging
from typing import Optional

from knowledge.episodic import EpisodicMemory
from evaluation.metrics.intervention_efficiency import InterventionEfficiencyMetric

logger = logging.getLogger(__name__)


class RQ1Evaluator:
    """RQ1: Are targeted interventions more query-efficient than random probing?

    Measures the number of interventions needed to reach accuracy >85%.
    """

    def __init__(self, episodic_memory: EpisodicMemory) -> None:
        self._metric = InterventionEfficiencyMetric(episodic_memory)

    def evaluate(
        self,
        campaign_id: str,
        experiment_id: Optional[str] = None,
        threshold: float = 0.85,
        baseline_campaign_id: Optional[str] = None,
        baseline_experiment_id: Optional[str] = None,
    ) -> dict:
        result = self._metric.compute(
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            threshold=threshold,
            baseline_campaign_id=baseline_campaign_id,
            baseline_experiment_id=baseline_experiment_id,
        )
        result["rq"] = "RQ1"
        logger.info(
            "RQ1: campaign=%s intv_to_threshold=%d best_acc=%.4f reached=%s",
            campaign_id, result["interventions_to_threshold"],
            result["best_accuracy"], result["reached"],
        )
        if baseline_campaign_id:
            logger.info(
                "RQ1 baseline: campaign=%s baseline=%s improvement=%.2f",
                campaign_id,
                baseline_campaign_id,
                result.get("improvement_ratio", 0.0),
            )
        return result
