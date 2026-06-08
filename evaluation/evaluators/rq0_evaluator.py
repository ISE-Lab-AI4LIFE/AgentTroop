from __future__ import annotations

import logging
from typing import Optional

from core.executor import ProgramExecutor
from core.primitive import default_registry
from core.program import Program

from adapters.base_victim import BaseVictim
from evaluation.judges.base import Judge
from evaluation.metrics.program_accuracy import ProgramAccuracyMetric
from evaluation.utils.test_generator import TestGenerator

logger = logging.getLogger(__name__)


class RQ0Evaluator:
    """RQ0: Can the synthesized program predict victim responses on unseen prompts?

    Uses ProgramAccuracyMetric on a held-out test set.
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
        self._metric = ProgramAccuracyMetric(executor)
        self._csv_path = csv_path

    def evaluate(
        self,
        program: Program,
        num_test_prompts: int = 50,
        test_prompts: Optional[list[str]] = None,
    ) -> dict:
        if test_prompts is None:
            generator = TestGenerator(self._csv_path)
            test_prompts = generator.generate(num_test_prompts)

        accuracy = self._metric.evaluate(program, test_prompts, self._victim, self._judge)
        result = {
            "rq": "RQ0",
            "program_id": program.id,
            "accuracy": accuracy,
            "num_test_prompts": len(test_prompts),
            "threshold": 0.85,
            "passed": accuracy >= 0.85,
        }
        logger.info("RQ0: accuracy=%.4f passed=%s", accuracy, result["passed"])
        return result
