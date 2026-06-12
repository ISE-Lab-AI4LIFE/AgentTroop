from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from adapters.base_victim import BaseVictim
from core.intervention import Intervention
from core.jailbreak import select_technique, apply_technique_to_intervention
from evaluation.judges.base import Judge
from evaluation.metrics.asr import AttackSuccessRateMetric

logger = logging.getLogger(__name__)


class HarmonyASREvaluator:
    """ASR evaluation that routes prompts through the full jailbreak pipeline:
    technique selection → template expansion → LLM refinement → victim execution.

    Uses LLMJudge (not the victim's internal heuristic) for classification.
    Logs all victim responses for human review.
    """

    def __init__(
        self,
        victim: BaseVictim,
        judge: Optional[Judge] = None,
        csv_path: str = "",
        red_team_agent: Any = None,
        strategist_agent: Any = None,
        num_variants: int = 1,
    ) -> None:
        self._victim = victim
        self._metric = AttackSuccessRateMetric(judge)
        self._csv_path = csv_path
        self._red_team = red_team_agent
        self._strategist = strategist_agent
        self._num_variants = num_variants
        self._used_techniques: list[str] = []

    def evaluate(
        self,
        prompts: Optional[list[str]] = None,
        num_prompts: int = 50,
        judge: Optional[Judge] = None,
    ) -> dict:
        """Run ASR evaluation through the Red Team Agent pipeline.

        For each prompt:
          1. Strategy engine selects technique → expands template
          2. Red Team LLM refines the crafted prompt
          3. Crafted prompt sent to victim (raw response via query_raw)
          4. LLMJudge classifies response as ACCEPT (0) or REFUSE (1)
          5. Full response logged for human evaluation
        """
        if prompts is None:
            from evaluation.utils.test_generator import TestGenerator
            generator = TestGenerator(self._csv_path)
            raw_prompts = generator.generate_jailbreak_prompts(num_prompts)
            prompts = [p for p in raw_prompts]

        if not prompts:
            return {"asr": 0.0, "total": 0, "successes": 0, "failures": 0, "rq": "HarmonyASR"}

        j = judge or self._metric._judge
        results: list[dict] = []

        for idx, prompt in enumerate(prompts):
            crafted, technique = self._craft_prompt(prompt)
            raw_response = self._query_raw(crafted)

            outcome = j.judge(raw_response) if j else 0

            if technique:
                from core.jailbreak import record_technique_outcome
                record_technique_outcome(technique, prompt, outcome)

            entry = {
                "index": idx,
                "goal": prompt[:120],
                "crafted_prompt": crafted,
                "victim_response": raw_response,
                "outcome": outcome,
                "label": "SUCCESS" if outcome == 0 else "REFUSE",
            }
            results.append(entry)
            logger.info(
                "[%d/%d] %s | goal=%r | response=%r",
                idx + 1, len(prompts),
                "✓" if outcome == 0 else "✗",
                prompt[:60], raw_response[:120],
            )

        successes = sum(1 for r in results if r["outcome"] == 0)
        asr = successes / len(results) if results else 0.0

        logger.info(
            "HarmonyASR: asr=%.4f (%d/%d) through RedTeamAgent pipeline",
            asr, successes, len(results),
        )

        return {
            "asr": asr,
            "total": len(results),
            "successes": successes,
            "failures": len(results) - successes,
            "details": results,
            "rq": "HarmonyASR",
        }

    def _craft_prompt(self, prompt: str) -> tuple[str, str]:
        """Apply technique selection → template expansion → optional LLM refinement.

        Returns (crafted_prompt, technique_name).
        """
        try:
            intervention = Intervention(base_prompt=prompt)
            intervention.final_prompt = prompt

            technique = select_technique(
                goal=prompt,
                version_space=getattr(self._strategist, "_version_space", None) if self._strategist else None,
                used_techniques=self._used_techniques,
            )
            intervention = apply_technique_to_intervention(intervention, technique)
            self._used_techniques.append(technique)

            if self._red_team is not None:
                intervention = self._red_team.maybe_refine_intervention(intervention)

            crafted = intervention.final_prompt
            logger.info(
                "HarmonyASR: crafted attack for goal=%r (%d chars, technique=%s)",
                prompt[:50], len(crafted), technique,
            )
            return crafted, technique
        except Exception as e:
            logger.warning("HarmonyASR: _craft_prompt failed for goal=%r: %s", prompt[:50], e)
        return prompt, ""

    def _query_raw(self, crafted: str) -> str:
        try:
            if hasattr(self._victim, "query_raw"):
                return self._victim.query_raw(crafted)
            raw = self._victim.respond(crafted)
            return str(raw)
        except Exception as e:
            logger.warning("HarmonyASR: victim query failed: %s", e)
            return ""
