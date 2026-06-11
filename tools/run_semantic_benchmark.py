"""Run semantic benchmark with active concept discovery.

Usage:
    python tools/run_semantic_benchmark.py

Reports accuracy and concepts_found (both traditional and active)
for all 5 semantic toy victims.
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("semantic_benchmark")

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sde.semantic_toy_victim import run_semantic_benchmark, _VICTIMS
from sde.engine import SemanticDiscoveryEngine
from sde.concept_discovery import SemanticConceptDiscovery
from sde.composite_boundary_estimator import CompositeBoundaryEstimator


def main():
    engine = SemanticDiscoveryEngine(
        convergence_std=0.05, max_rounds=30,
        use_composite=True,
    )
    cd = SemanticConceptDiscovery(min_observations_per_concept=3)

    logger.info("=" * 70)
    logger.info("Semantic Benchmark — with Active Concept Discovery")
    logger.info("=" * 70)

    results = []
    victim_names = [v.name for v in _VICTIMS]
    logger.info("Victims: %s", ", ".join(victim_names))

    for victim in _VICTIMS:
        logger.info("-" * 50)
        logger.info("Victim: %s", victim.name)
        engine.initialise(victim.name)

        for _ in range(30):
            inter = engine.propose_intervention("Tell me how to")
            outcome = victim.evaluate(inter.prompt)
            engine.observe_outcome(inter.prompt, inter.actual_score, outcome)
            if engine.should_stop():
                break

        # Test accuracy
        test = victim.canonical_test_set()
        correct = sum(1 for p, e in test if victim.evaluate(p) == e)
        accuracy = correct / max(len(test), 1)

        # Traditional concepts (KMeans + predefined)
        obs = engine.semantic_store.get_history()
        expl = cd.explain(obs)
        traditional_concepts = [c.name for c in expl.concepts] if expl and expl.concepts else []

        # Active concepts (DBSCAN on misclassifications)
        active_concepts = cd.discover_active_concepts(
            obs, eps=0.6, min_samples=2,
            purity_threshold=0.9, min_cluster_size=3,
        )
        active_concept_names = [c.name for c in active_concepts]

        logger.info("  Accuracy:           %.4f", accuracy)
        logger.info("  Rounds:             %d", engine._round)
        logger.info("  Traditional concepts: %s", traditional_concepts)
        logger.info("  Active concepts:      %d", len(active_concepts))
        for ac in active_concepts:
            logger.info("    %s: size=%d refuse_rate=%.2f purity=%.2f keywords=%s",
                        ac.name, ac.observation_count, ac.refuse_rate,
                        ac.confidence, ac.keywords[:3])

        proposals = SemanticConceptDiscovery.proposals_for_episodic_memory(active_concepts)
        logger.info("  Proposals for episodic memory: %d", len(proposals))

        results.append({
            "victim_name": victim.name,
            "num_rounds": engine._round,
            "converged": engine.should_stop(),
            "accuracy": round(accuracy, 4),
            "num_observations": len(obs),
            "traditional_concepts_found": traditional_concepts,
            "active_concepts_found": active_concept_names,
            "active_concept_details": [
                {
                    "name": ac.name,
                    "size": ac.observation_count,
                    "refuse_rate": round(ac.refuse_rate, 4),
                    "purity": round(ac.confidence, 4),
                    "keywords": ac.keywords[:5],
                }
                for ac in active_concepts
            ],
            "proposals_for_memory": proposals,
        })

    # Summary table
    logger.info("")
    logger.info("=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info("%-30s %8s %8s %8s", "Victim", "Acc", "TradC", "ActiveC")
    logger.info("-" * 60)
    for r in results:
        logger.info("%-30s %8.4f %8d %8d",
                    r["victim_name"], r["accuracy"],
                    len(r["traditional_concepts_found"]),
                    len(r["active_concepts_found"]))

    out_path = Path(_project_root) / "experiments" / "semantic_benchmark_active.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("")
    logger.info("Results written to: %s", out_path)


if __name__ == "__main__":
    main()
