from __future__ import annotations

import logging
from typing import Optional

from core.executor import ProgramExecutor
from core.primitive import default_registry
from core.program import Program
from core.types import Outcome

from adapters.base_victim import BaseVictim
from evaluation.judges.base import Judge

logger = logging.getLogger(__name__)


class ProgramAccuracyMetric:
    """RQ0: Accuracy of a synthesized program on a held-out test set.

    Compares the program's prediction (via ProgramExecutor) against
    the victim's actual behaviour (via victim.respond() or a Judge).
    """

    def __init__(self, executor: Optional[ProgramExecutor] = None) -> None:
        self._executor = executor or ProgramExecutor(default_registry)

    def evaluate(
        self,
        program: Program,
        test_prompts: list[str],
        victim: BaseVictim,
        judge: Optional[Judge] = None,
    ) -> float:
        if not test_prompts:
            return 0.0
        correct = 0
        for prompt in test_prompts:
            actual_outcome = self._get_victim_outcome(prompt, victim, judge)
            try:
                predicted = self._executor.execute(program, prompt)
            except Exception:
                predicted = 0
            if predicted == actual_outcome:
                correct += 1
        accuracy = correct / len(test_prompts)
        logger.info(
            "ProgramAccuracy: %.4f (%d/%d) on %s",
            accuracy, correct, len(test_prompts), program.id,
        )
        return accuracy

    def _get_victim_outcome(
        self, prompt: str, victim: BaseVictim, judge: Optional[Judge]
    ) -> Outcome:
        if judge is not None:
            response = victim.respond(prompt)
            if isinstance(response, str):
                return judge.judge(response)
            return response
        raw = victim.respond(prompt)
        if isinstance(raw, str):
            try:
                return int(raw.strip())
            except (ValueError, AttributeError):
                return 0
        return int(raw)
