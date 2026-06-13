from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from agents.red_team import RedTeamAgent
from core.executor import ProgramExecutor
from core.program import Program
from evaluation.judges.base import Judge
from evaluation.metrics.asr import AttackSuccessRateMetric

logger = logging.getLogger(__name__)


class AdversarialASREvaluator:
    """ASR evaluation using RedTeamAgent to craft adversarial prompts from the learned program.

    After training, uses the best program's primitives (keywords, transforms)
    to craft adversarial variants of harmful test prompts and measures
    the true ASR (harmony_asr).
    """

    def __init__(
        self,
        victim: Any,
        judge: Optional[Judge] = None,
        red_team: Optional[RedTeamAgent] = None,
        csv_path: str = "",
    ) -> None:
        self._victim = victim
        self._metric = AttackSuccessRateMetric(judge)
        self._red_team = red_team or RedTeamAgent()
        self._csv_path = csv_path

    def evaluate(
        self,
        program: Program,
        test_prompts: Optional[List[Tuple[str, int]]] = None,
        num_test_prompts: int = 30,
    ) -> dict:
        if not test_prompts and not self._csv_path:
            return {"harmony_asr": 0.0, "total": 0, "successes": 0, "failures": 0}

        if test_prompts is None:
            from evaluation.utils.test_generator import TestGenerator
            generator = TestGenerator(self._csv_path)
            raw = generator.generate_jailbreak_prompts(num_test_prompts)
            test_prompts = [(p, 0) for p in raw]

        if not test_prompts:
            return {"harmony_asr": 0.0, "total": 0, "successes": 0, "failures": 0}

        primitives = self._red_team.extract_primitives(program)
        successes = 0
        details = []

        for prompt, expected in test_prompts:
            crafted = self._red_team.craft_adversarial_prompt(prompt, primitives)
            if self._metric._judge:
                raw_text = self._victim.query_raw(crafted)
                outcome = self._metric._judge.judge(raw_text)
            else:
                raw = self._victim.respond(crafted)
                try:
                    outcome = int(raw)
                except (ValueError, TypeError):
                    outcome = 0
            if outcome == 0:
                successes += 1
            details.append({
                "original": prompt,
                "crafted": crafted,
                "outcome": outcome,
            })

        total = len(test_prompts)
        asr = successes / total if total > 0 else 0.0

        logger.info(
            "Adversarial ASR (RedTeamAgent): asr=%.4f (%d/%d)",
            asr, successes, total,
        )

        return {
            "adversarial_asr": asr,
            "adversarial_total": total,
            "adversarial_successes": successes,
            "adversarial_failures": total - successes,
            "pre_accepted_total": total,
            "pre_accepted_accepts": successes,
            "details": details,
            "program_primitives": {
                "keywords": primitives.get("keywords", []),
                "transforms": primitives.get("transforms", []),
            },
        }
