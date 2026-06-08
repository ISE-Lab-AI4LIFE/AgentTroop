"""PatternMiner — mines primitive candidates from anomalies, failures, and verifier gaps.

Strategies::

    AnomalyTraces
        → Group by transform name
        → Extract common prompt patterns from REFUSE vs ACCEPT
        → Propose keyword/pattern predicates

    FailedSynthesisStats
        → Identify examples with high error rate
        → Cluster by prompt length / encoding / domain
        → Propose specialized classifiers

    VerifierReports
        → Find failure prompts (boundary cases)
        → Analyse what distinguishes failures from successes
        → Propose boundary classifiers
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from .engine import CandidatePrimitive, DiscoverySource

logger = logging.getLogger(__name__)

_MIN_EXAMPLES_FOR_PATTERN = 3


class PatternMiner:
    """Mines patterns from traces to propose new primitives."""

    def mine_from_anomalies(
        self,
        anomalies: List[Any],
    ) -> List[CandidatePrimitive]:
        """Mine candidate predicates from anomaly data.

        Looks for:
        - Transforms that consistently change outcomes → suggest transform predicates
        - Keywords in base prompts that correlate with REFUSE → suggest keyword predicates
        """
        candidates: List[CandidatePrimitive] = []

        # Analyse transform effects
        transform_outcomes: Dict[str, List[int]] = {}
        keyword_outcomes: Dict[str, List[int]] = {}
        length_outcomes: List[int] = []

        for anomaly in anomalies:
            base = getattr(anomaly, "base_prompt", "")
            orig = getattr(anomaly, "outcome_original", 0)
            trans = getattr(anomaly, "outcome_transformed", 0)
            transforms = getattr(anomaly, "transform_names", []) or []

            # Track transform → outcome mapping
            for tname in transforms:
                if tname not in transform_outcomes:
                    transform_outcomes[tname] = []
                transform_outcomes[tname].append(trans)

            # Track keywords
            words = re.findall(r"[a-zA-Z]{4,}", base.lower())
            for w in words:
                if w not in keyword_outcomes:
                    keyword_outcomes[w] = []
                keyword_outcomes[w].append(orig)

            length_outcomes.append((len(base), orig))

        # Propose predicates from transforms
        for tname, outcomes in transform_outcomes.items():
            if len(outcomes) >= _MIN_EXAMPLES_FOR_PATTERN:
                refuse_rate = sum(outcomes) / len(outcomes)
                if refuse_rate > 0.7:
                    candidates.append(CandidatePrimitive(
                        name=f"bypasses_{tname}",
                        primitive_type="predicate",
                        signature="String → Bool",
                        description=f"Checks if the prompt bypasses the {tname} transform filter",
                        implementation_hint=f"Check if prompt contains patterns that survive {tname}",
                        positive_examples=[],
                        negative_examples=[],
                        discovery_source=DiscoverySource.ANOMALY_PATTERN,
                    ))

        # Propose keyword predicates
        for word, outcomes in keyword_outcomes.items():
            if len(outcomes) >= _MIN_EXAMPLES_FOR_PATTERN and word not in _STOPWORDS:
                refuse_rate = sum(outcomes) / len(outcomes)
                if refuse_rate > 0.6:
                    candidates.append(CandidatePrimitive(
                        name=f"contains_{word}",
                        primitive_type="predicate",
                        signature="String → Bool",
                        description=f"Checks if the prompt contains the word '{word}'",
                        implementation_hint=f"regex: \\b{word}\\b",
                        positive_examples=[],
                        negative_examples=[],
                        discovery_source=DiscoverySource.ANOMALY_PATTERN,
                    ))

        # Propose length classifier
        if length_outcomes:
            refuse_lens = [l for l, o in length_outcomes if o == 1]
            accept_lens = [l for l, o in length_outcomes if o == 0]
            if refuse_lens and accept_lens:
                avg_refuse = sum(refuse_lens) / len(refuse_lens)
                avg_accept = sum(accept_lens) / len(accept_lens)
                if abs(avg_refuse - avg_accept) > 30:
                    candidates.append(CandidatePrimitive(
                        name="length_based_refusal",
                        primitive_type="classifier",
                        signature="String → Real",
                        description="Predicts refusal likelihood based on prompt length",
                        implementation_hint=f"Score = min(1.0, len(prompt) / {avg_refuse:.0f})",
                        positive_examples=[],
                        negative_examples=[],
                        discovery_source=DiscoverySource.ANOMALY_PATTERN,
                    ))

        if candidates:
            logger.info("PatternMiner: mined %d candidates from %d anomalies",
                        len(candidates), len(anomalies))
        return candidates

    def mine_from_failures(
        self,
        synthesis_stats: Any,
    ) -> List[CandidatePrimitive]:
        """Mine candidates from failed synthesis attempts.

        When synthesis fails, the unsolved examples suggest missing primitives.
        """
        candidates: List[CandidatePrimitive] = []
        if not synthesis_stats:
            return candidates

        # Extract error information
        errors_actual = getattr(synthesis_stats, "errors_actual", 0)
        max_errors = getattr(synthesis_stats, "max_errors", 0)
        programs_tried = getattr(synthesis_stats, "programs_tried", 0)

        if errors_actual > max_errors and programs_tried > 100:
            # Synthesis failed because no program matched — suggest a generic
            # pattern-based predicate as a fallback
            candidates.append(CandidatePrimitive(
                name="synthesis_fallback_pattern",
                primitive_type="predicate",
                signature="String → Bool",
                description="Fallback predicate discovered from unsolved synthesis examples",
                implementation_hint="regex: (harmful|dangerous|illegal|attack|weapon)",
                positive_examples=[],
                negative_examples=[],
                discovery_source=DiscoverySource.FAILED_SYNTHESIS,
                metadata={
                    "errors_actual": errors_actual,
                    "programs_tried": programs_tried,
                },
            ))

        return candidates

    def mine_from_verifier(
        self,
        verifier_report: Any,
    ) -> List[CandidatePrimitive]:
        """Mine candidates from verifier failure reports.

        Failure prompts are boundary cases the current program can't handle.
        """
        candidates: List[CandidatePrimitive] = []
        if not verifier_report:
            return candidates

        failures = getattr(verifier_report, "failures", [])
        if not failures:
            return candidates

        # Extract failure prompts
        fail_prompts: List[str] = []
        for f in failures:
            if isinstance(f, tuple) and len(f) >= 1:
                fail_prompts.append(str(f[0]))
            elif isinstance(f, dict):
                fail_prompts.append(str(f.get("prompt", "")))

        if len(fail_prompts) < _MIN_EXAMPLES_FOR_PATTERN:
            return candidates

        # Find common patterns in failure prompts
        common_words: Counter = Counter()
        for p in fail_prompts:
            words = re.findall(r"[a-zA-Z]{4,}", p.lower())
            common_words.update(w for w in words if w not in _STOPWORDS)

        top_words = [w for w, _ in common_words.most_common(5)]
        if top_words:
            word_pattern = "|".join(top_words)
            candidates.append(CandidatePrimitive(
                name="verifier_boundary_pattern",
                primitive_type="predicate",
                signature="String → Bool",
                description=f"Matches verifier boundary patterns: {', '.join(top_words)}",
                implementation_hint=f"regex: \\b({word_pattern})\\b",
                positive_examples=fail_prompts[:5],
                negative_examples=[],
                discovery_source=DiscoverySource.VERIFIER_FAILURE,
                metadata={"num_failures": len(fail_prompts)},
            ))

        return candidates


_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "it", "its", "my", "your", "his",
    "her", "our", "their", "no", "nor", "not", "or", "and", "but",
    "if", "because", "so", "than", "too", "very", "just", "about",
    "also", "make", "get", "give", "tell", "show", "without",
    "please", "can", "could", "would", "should", "will", "may",
    "might", "shall", "need", "want", "like", "know", "think",
    "help", "find", "use", "need", "let", "ask", "say", "see",
    "come", "go", "take", "put", "try", "keep", "look", "allow",
    "allow", "seem", "wait", "call", "turn", "bring", "begin",
    "move", "leave", "reach", "set", "create", "provide", "cause",
    "allow", "allow", "result", "follow", "include", "involve",
    "require", "suggest", "support", "contain", "produce",
    "perform", "offer", "ensure", "enable", "prevent", "avoid",
}
