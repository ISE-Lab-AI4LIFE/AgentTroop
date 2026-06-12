from __future__ import annotations

import logging
from typing import Optional

from adapters.base_victim import BaseVictim
from core.executor import ProgramExecutor
from core.program import Program
from evaluation.judges.base import Judge
from evaluation.metrics.adversarial_asr import AdversarialASRMetric
from evaluation.utils.test_generator import TestGenerator

logger = logging.getLogger(__name__)


class AdversarialASREvaluator:
    """Adversarial ASR: craft prompts with the learned program, then measure ASR.

    Uses the program to find transform chains that elicit ACCEPT from the
    victim, providing an upper bound on adversarial vulnerability.
    """

    def __init__(
        self,
        victim: BaseVictim,
        judge: Optional[Judge] = None,
        executor: Optional[ProgramExecutor] = None,
        csv_path: str = "",
    ) -> None:
        self._victim = victim
        self._judge = judge
        self._metric = AdversarialASRMetric(executor)
        self._csv_path = csv_path

    def evaluate(
        self,
        program: Program,
        num_test_prompts: int = 30,
        max_depth: int = 2,
        test_prompts: Optional[list[str]] = None,
    ) -> dict:
        if test_prompts is None:
            generator = TestGenerator(self._csv_path)
            test_prompts = generator.generate_jailbreak_prompts(num_test_prompts)

        result = self._metric.evaluate(
            prompts=test_prompts,
            program=program,
            victim=self._victim,
            judge=self._judge,
            max_depth=max_depth,
        )
        result["rq"] = "Adversarial ASR"
        result["program_id"] = program.id
        result["num_test_prompts"] = len(test_prompts)
        logger.info(
            "Adversarial ASR: asr=%.4f (%d/%d) | pre-accepted=%d/%d program=%s",
            result["adversarial_asr"],
            result["adversarial_successes"],
            result["adversarial_total"],
            result["pre_accepted_accepts"],
            result["pre_accepted_total"],
            program.id,
        )
        return result
