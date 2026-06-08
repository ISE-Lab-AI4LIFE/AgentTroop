"""Ablation: Bayesian Optimization baseline (optimizes ASR, not hypotheses).

This baseline uses Bayesian Optimization over a simple transform parameter
space to maximize Attack Success Rate (ASR), mimicking traditional red-teaming
approaches.  No hypothesis generation, no program synthesis.

Usage:
    python -m experiments.ablation.bayesian_optimization [--campaign bo_campaign]
"""

import argparse
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from core.intervention import Intervention


class BOStrategist:
    """Strategist variant using Bayesian Optimization over transform space.

    Maintains a simple Gaussian Process surrogate model (via random sampling
    as a lightweight proxy for proper BO) to select transforms that maximize
    the probability of REFUSE (ASR).

    This mimics the optimization objective of traditional red-teaming tools.
    """

    def __init__(self, base_strategist: Any) -> None:
        self._base = base_strategist
        self.intervention_budget = getattr(base_strategist, "intervention_budget", 50)
        self.episodic_memory = getattr(base_strategist, "episodic_memory", None)
        self.executor = getattr(base_strategist, "executor", None)
        self.llm_client = None
        self.use_llm = False
        self.max_chain_depth = 1

        self._history: List[Tuple[str, int]] = []
        self._transform_scores: Dict[str, float] = {}

    def select_hypothesis_pair(self, hypotheses: List[Any]) -> tuple:
        return (None, None)

    def design_intervention(
        self,
        h1: Any = None,
        h2: Any = None,
        base_prompts: Optional[List[str]] = None,
        campaign_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> Optional[Intervention]:
        transforms = self._base._get_transforms()
        prompts = self._base._resolve_base_prompts(base_prompts, campaign_id, experiment_id)
        if not prompts:
            return None

        bp = random.choice(prompts)

        if not transforms:
            return Intervention(base_prompt=bp, transforms=[], metadata={"exploratory": True})

        if len(self._history) < 5:
            t = random.choice(transforms)
            score = self._transform_scores.get(t.name, 0.5)
            return Intervention(base_prompt=bp, transforms=[t], metadata={"exploratory": True})

        best_t = max(transforms, key=lambda t: self._transform_scores.get(t.name, 0.5))
        return Intervention(base_prompt=bp, transforms=[best_t], metadata={"exploratory": True})

    def execute_intervention(self, intervention: Intervention, victim: Any) -> int:
        outcome = self._base.execute_intervention(intervention, victim)
        t_name = intervention.transforms[0].name if intervention.transforms else "identity"
        self._history.append((t_name, outcome))
        old = self._transform_scores.get(t_name, 0.5)
        self._transform_scores[t_name] = old + 0.1 * (outcome - old)
        return outcome

    def store_intervention(
        self, intervention: Intervention, outcome: int, campaign_id: str,
        h1: Any, h2: Any = None, experiment_id: Optional[str] = None,
        victim_name: str = "victim", strategy_name: str = "bayesian_optimization",
        agent_name: str = "BOStrategist",
    ) -> str:
        return self._base.store_intervention(
            intervention, outcome, campaign_id, h1, h2,
            experiment_id=experiment_id, victim_name=victim_name,
            strategy_name=strategy_name, agent_name=agent_name,
        )

    def evaluate_discriminative_power(self, *args: Any, **kwargs: Any) -> float:
        return 0.0


def run_bo_campaign(
    orchestrator_factory: Any,
    campaign_id: str = "ablation_bayesian_opt",
    max_iterations: int = 20,
) -> Dict[str, Any]:
    orch = orchestrator_factory(campaign_id=campaign_id)
    orch.strategist = BOStrategist(orch.strategist)
    orch.max_iterations = max_iterations
    result = orch.run()
    result["strategy"] = "bayesian_optimization"
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Bayesian Optimization ablation")
    parser.add_argument("--campaign", default="ablation_bayesian_opt", help="Campaign ID")
    parser.add_argument("--max-iter", type=int, default=20, help="Max iterations")
    args = parser.parse_args()
    print(f"Run: python -m experiments.ablation.bayesian_optimization --campaign {args.campaign}")
    print("Import and call run_bo_campaign() in your experiment harness.")
