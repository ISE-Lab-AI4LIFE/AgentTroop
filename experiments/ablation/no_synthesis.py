"""Ablation: no program synthesis (belief state only).

Disables the Researcher Agent's program synthesis pipeline.  The system
still detects anomalies, generates hypotheses, and designs interventions,
but never synthesizes, verifies, or stores programs.  Only the belief state
(POMDP / EFE) is maintained.

Compare this to the full HARMONY-X pipeline to measure the contribution
of program synthesis to intervention efficiency and accuracy.

Usage:
    python -m experiments.ablation.no_synthesis [--campaign no_synth_campaign]
"""

import argparse
import logging
from typing import Any, Dict

from knowledge.defense_store import DefenseProgramStore
from knowledge.scientific_memory import ScientificMemory


class NoSynthesisResearcher:
    """Researcher variant that skips all synthesis pipeline steps.

    The ``run_reverse_engineering_pipeline`` always returns early with
    no program and no theory.  All other methods (process_proposals, etc.)
    are preserved for compatibility.
    """

    def __init__(self, base_researcher: Any) -> None:
        self._base = base_researcher
        self.episodic_memory = getattr(base_researcher, "episodic_memory", None)
        self.defense_store = getattr(base_researcher, "defense_store", None)
        self.scientific_memory = getattr(base_researcher, "scientific_memory", None)
        self.causal_graph = getattr(base_researcher, "causal_graph", None)

    def run_reverse_engineering_pipeline(
        self, *args: Any, **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "program_id": None,
            "theory_id": None,
            "accuracy": 0.0,
            "verified": False,
            "error": "Synthesis disabled (ablation: no_synthesis)",
        }

    def process_proposals(self, proposals: Any) -> list:
        return [{"success": False, "error": "Synthesis disabled"}] if proposals else []

    def explain(self, question: str) -> Any:
        from graphrag.graph_reasoner import GraphRAGAnswer
        return GraphRAGAnswer(
            query=question,
            answer="Synthesis disabled (ablation: no_synthesis).",
            confidence=0.0,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def wrap_orchestrator_no_synthesis(orch: Any) -> Any:
    old_researcher = orch.researcher
    orch.researcher = NoSynthesisResearcher(old_researcher)
    return orch


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="No-synthesis ablation")
    parser.add_argument("--campaign", default="ablation_no_synthesis", help="Campaign ID")
    parser.add_argument("--max-iter", type=int, default=20, help="Max iterations")
    args = parser.parse_args()
    print(f"Ablation: no_synthesis (campaign={args.campaign}, max_iter={args.max_iter})")
    print("Import wrap_orchestrator_no_synthesis() in your experiment harness.")
