#!/usr/bin/env python3
"""Run all ablation experiments and produce a comparison table.

Usage:
    python -m experiments.ablation.run_all [--quick] [--toy-only]

Results are printed as a markdown table and optionally logged to MLflow
(if available).
"""

import argparse
import logging
import time
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_orchestrator(campaign_id: str, use_toy: bool = True) -> Any:
    """Build a minimal orchestrator for ablation testing.

    Uses the ToyVictim when use_toy=True, otherwise requires a real victim.
    """
    from agents.cognitive import CognitiveAgent
    from agents.researcher import ResearcherAgent
    from agents.strategist import StrategistAgent
    from knowledge.episodic import EpisodicMemory
    from knowledge.manager import KnowledgeManager
    from knowledge.session_memory import SessionMemory
    from orchestration import Orchestrator

    memory = EpisodicMemory()  # uses defaults
    km = KnowledgeManager(use_redis=False)
    session = SessionMemory(redis_url="redis://localhost:6379/0")
    cognitive = CognitiveAgent(episodic_memory=memory)
    strategist = StrategistAgent(episodic_memory=memory)
    researcher = ResearcherAgent(
        episodic_memory=memory,
        defense_store=km,
        scientific_memory=km,
    )

    if use_toy:
        from experiments.toy_model import ToyVictim

        victim = ToyVictim()
    else:
        raise ValueError("Non-toy victims are not supported in ablation mode.")

    return Orchestrator(
        cognitive_agent=cognitive,
        strategist_agent=strategist,
        researcher_agent=researcher,
        knowledge_manager=km,
        session_memory=session,
        victim=victim,
        campaign_id=campaign_id,
        max_iterations=10,
        max_interventions=100,
    )


def run_ablation(ablation_name: str, campaign_id: str) -> Dict[str, Any]:
    """Run a single ablation by name and return results."""
    from experiments.ablation.no_llm import wrap_orchestrator_no_llm
    from experiments.ablation.no_scientific_memory import (
        wrap_orchestrator_no_scientific_memory,
    )
    from experiments.ablation.no_synthesis import wrap_orchestrator_no_synthesis
    from experiments.ablation.random_probing import run_random_probing_campaign

    wrappers = {
        "no_llm": lambda o: wrap_orchestrator_no_llm(o),
        "no_scientific_memory": lambda o: wrap_orchestrator_no_scientific_memory(o),
        "no_synthesis": lambda o: wrap_orchestrator_no_synthesis(o),
    }

    if ablation_name == "random_probing":
        return run_random_probing_campaign(
            lambda cid=None: _build_orchestrator(cid or campaign_id),
            campaign_id=campaign_id,
            max_iterations=10,
        )

    orch = _build_orchestrator(campaign_id)
    wrapper = wrappers.get(ablation_name)
    if wrapper:
        orch = wrapper(orch)
    else:
        logger.warning("Unknown ablation '%s', running full pipeline", ablation_name)

    result = orch.run()
    result["ablation"] = ablation_name
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all HARMONY-X ablations")
    parser.add_argument("--quick", action="store_true", help="Minimal iterations")
    parser.add_argument("--toy-only", action="store_true", help="Only toy model")
    args = parser.parse_args()

    ablations = [
        "full_pipeline",
        "no_llm",
        "no_synthesis",
        "no_scientific_memory",
        "random_probing",
    ]

    results: List[Dict[str, Any]] = []
    for ablation in ablations:
        campaign_id = f"ablation_{ablation}_{int(time.time())}"
        logger.info("=" * 60)
        logger.info("Running ablation: %s (campaign=%s)", ablation, campaign_id)
        logger.info("=" * 60)
        try:
            result = run_ablation(ablation, campaign_id)
            result["ablation"] = ablation
            results.append(result)
            logger.info(
                "Result: success=%s accuracy=%.2f interventions=%d iterations=%d",
                result.get("success"), result.get("best_accuracy", 0.0),
                result.get("total_interventions", 0),
                result.get("total_iterations", 0),
            )
        except Exception as exc:
            logger.exception("Ablation '%s' failed: %s", ablation, exc)
            results.append({
                "ablation": ablation,
                "success": False,
                "error": str(exc),
            })

    print("\n")
    print("=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Ablation':<25} {'Success':<10} {'Accuracy':<10} {'Interventions':<15} {'Iterations':<12}")
    print("-" * 70)
    for r in results:
        name = r.get("ablation", "?")
        success = "YES" if r.get("success") else "NO"
        acc = f"{r.get('best_accuracy', 0.0):.3f}"
        intv = str(r.get("total_interventions", 0))
        iters = str(r.get("total_iterations", 0))
        print(f"{name:<25} {success:<10} {acc:<10} {intv:<15} {iters:<12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
