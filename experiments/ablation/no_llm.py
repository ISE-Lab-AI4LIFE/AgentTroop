"""Ablation: no LLM hypothesis generation (fixed templates only).

Disables the LLM-based hypothesis generator in the Cognitive Agent.
Only the keyword-based fallback hypotheses (from `_fallback_hypotheses`)
are used.  Measures the contribution of LLM-guided structural hypotheses
to intervention efficiency and program accuracy.

Usage:
    python -m experiments.ablation.no_llm [--campaign no_llm_campaign]
"""

import argparse
import logging
from typing import Any, Dict, List, Optional

from agents.cognitive import Anomaly, CognitiveAgent, Hypothesis


class NoLLMCognitive(CognitiveAgent):
    """CognitiveAgent variant that skips LLM hypothesis generation.

    Only the keyword-based fallback hypotheses (content words, length
    heuristics, regex patterns) are generated.  The LLM prompt is never
    sent, saving API costs and providing a baseline for LLM contribution.
    """

    def generate_hypotheses(
        self,
        anomalies: List[Anomaly],
        prior_hypotheses: Optional[List[Hypothesis]] = None,
    ) -> List[Hypothesis]:
        if not anomalies:
            return []
        fallback = self._fallback_hypotheses(anomalies)
        seen_conds: set = set()
        merged: List[Hypothesis] = []
        for h in fallback:
            cond = getattr(h, "condition", "") or ""
            if cond not in seen_conds:
                seen_conds.add(cond)
                merged.append(h)
        for hyp in merged:
            self.estimate_confidence(hyp, anomalies)
        logger = logging.getLogger(__name__)
        logger.info(
            "NoLLM: generated %d template-only hypotheses from %d anomalies",
            len(merged), len(anomalies),
        )
        return merged


def wrap_orchestrator_no_llm(orch: Any) -> Any:
    old_cognitive = orch.cognitive
    if isinstance(old_cognitive, NoLLMCognitive):
        return orch
    orch.cognitive = NoLLMCognitive(
        episodic_memory=old_cognitive.episodic_memory,
        ontology_memory=old_cognitive.ontology_memory,
        llm_client=None,
        grammar_exporter=old_cognitive.grammar_exporter,
        anomaly_threshold=old_cognitive.anomaly_threshold,
        base_prompts=list(old_cognitive.base_prompts),
        persist_anomalies=False,
        hypothesis_store=getattr(old_cognitive, "hypothesis_store", None),
    )
    return orch


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="No LLM hypothesis ablation")
    parser.add_argument("--campaign", default="ablation_no_llm", help="Campaign ID")
    parser.add_argument("--max-iter", type=int, default=20, help="Max iterations")
    args = parser.parse_args()
    print(f"Ablation: no_llm (campaign={args.campaign}, max_iter={args.max_iter})")
    print("Import wrap_orchestrator_no_llm() in your experiment harness.")
