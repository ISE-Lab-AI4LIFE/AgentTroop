from __future__ import annotations

import logging
from typing import Optional

from adapters.base_victim import BaseVictim
from evaluation.judges.base import Judge
from evaluation.metrics.asr import AttackSuccessRateMetric
from evaluation.utils.test_generator import TestGenerator

logger = logging.getLogger(__name__)


class ASREvaluator:
    """ASR: Attack Success Rate of jailbreak prompts against a victim.

    Can be run before and after HARMONY-X discovers a program to measure impact.
    """

    def __init__(
        self,
        victim: BaseVictim,
        judge: Optional[Judge] = None,
        csv_path: str = "",
    ) -> None:
        self._victim = victim
        self._metric = AttackSuccessRateMetric(judge)
        self._csv_path = csv_path

    def evaluate(
        self,
        prompts: Optional[list[str]] = None,
        num_prompts: int = 50,
        judge: Optional[Judge] = None,
    ) -> dict:
        if prompts is None:
            generator = TestGenerator(self._csv_path)
            prompts = generator.generate_jailbreak_prompts(num_prompts)
        result = self._metric.evaluate(prompts, self._victim, judge)
        result["rq"] = "ASR"
        logger.info("ASR: asr=%.4f (%d/%d)", result["asr"], result["successes"], result["total"])
        return result
