from __future__ import annotations

import logging
from typing import Optional

from knowledge.episodic import EpisodicMemory
from evaluation.metrics.transfer_speed import TransferSpeedMetric

logger = logging.getLogger(__name__)


class RQ3Evaluator:
    """RQ3: Does Scientific Memory support effective transfer between models?

    Compares intervention counts for a target campaign against a baseline
    campaign that represents learning from scratch or without prior transfer.
    """

    def __init__(self, episodic_memory: EpisodicMemory) -> None:
        self._metric = TransferSpeedMetric(episodic_memory)

    def evaluate(
        self,
        prior_campaign_id: str,
        target_campaign_id: str,
        prior_experiment_id: Optional[str] = None,
        target_experiment_id: Optional[str] = None,
        threshold: float = 0.9,
    ) -> dict:
        result = self._metric.compute(
            prior_campaign_id=prior_campaign_id,
            target_campaign_id=target_campaign_id,
            prior_experiment_id=prior_experiment_id,
            target_experiment_id=target_experiment_id,
            threshold=threshold,
        )
        result["rq"] = "RQ3"
        logger.info(
            "RQ3: prior=%s target=%s speedup=%.2f",
            prior_campaign_id, target_campaign_id, result["speedup_ratio"],
        )
        return result
