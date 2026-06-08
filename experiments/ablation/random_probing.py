"""Ablation: random probing (no targeted interventions).

Replaces the StrategistAgent's targeted intervention design with random
transform selection.  Measures intervention efficiency vs. the full system.

Usage:
    python -m experiments.ablation.random_probing [--campaign toy_campaign] [--max_iter 20]
"""

import argparse
import logging
import random
from typing import Any, Dict, List, Optional

from core.intervention import Intervention
from core.primitive import Transform


class RandomProbingStrategist:
    """Strategist variant that probes randomly instead of using targeted interventions.

    Instead of selecting the hypothesis pair with highest uncertainty and
    designing an optimal intervention, this variant:
      - Picks a random hypothesis (or null)
      - Applies a random transform from the available catalog
    """

    def __init__(self, base_strategist: Any) -> None:
        self._base = base_strategist
        self.intervention_budget = getattr(base_strategist, "intervention_budget", 50)
        self.episodic_memory = getattr(base_strategist, "episodic_memory", None)
        self.executor = getattr(base_strategist, "executor", None)
        self.llm_client = None
        self.use_llm = False
        self.max_chain_depth = 1

    def select_hypothesis_pair(self, hypotheses: List[Any]) -> tuple:
        if not hypotheses:
            return None, None
        h = random.choice(hypotheses)
        return h, None

    def design_intervention(
        self,
        h1: Any,
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
        if transforms and random.random() > 0.3:
            t = random.choice(transforms)
            return Intervention(base_prompt=bp, transforms=[t], metadata={"exploratory": True})
        return Intervention(base_prompt=bp, transforms=[], metadata={"exploratory": True})

    def execute_intervention(self, intervention: Intervention, victim: Any) -> int:
        return self._base.execute_intervention(intervention, victim)

    def store_intervention(
        self, intervention: Intervention, outcome: int, campaign_id: str,
        h1: Any, h2: Any = None, experiment_id: Optional[str] = None,
        victim_name: str = "victim", strategy_name: str = "random_probing",
        agent_name: str = "RandomProbingStrategist",
    ) -> str:
        return self._base.store_intervention(
            intervention, outcome, campaign_id, h1, h2,
            experiment_id=experiment_id, victim_name=victim_name,
            strategy_name=strategy_name, agent_name=agent_name,
        )

    def evaluate_discriminative_power(self, *args: Any, **kwargs: Any) -> float:
        return 0.0


def run_random_probing_campaign(
    orchestrator_factory: Any,
    campaign_id: str = "ablation_random_probing",
    max_iterations: int = 20,
) -> Dict[str, Any]:
    """Run a full campaign with random probing strategy."""
    orch = orchestrator_factory(campaign_id=campaign_id)
    old_strategist = orch.strategist
    orch.strategist = RandomProbingStrategist(old_strategist)
    orch.max_iterations = max_iterations
    result = orch.run()
    result["strategy"] = "random_probing"
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Random probing ablation")
    parser.add_argument("--campaign", default="ablation_random_probing", help="Campaign ID")
    parser.add_argument("--max-iter", type=int, default=20, help="Max iterations")
    args = parser.parse_args()
    print(f"Run: python -m experiments.ablation.random_probing --campaign {args.campaign}")
    print("Import and call run_random_probing_campaign() in your experiment harness.")
