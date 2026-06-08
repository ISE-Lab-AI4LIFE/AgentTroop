"""PrimitiveDiscoveryEngine — automated discovery of new primitives.

Discovery sources::

    AnomalyTraces → Pattern Mining → Candidate Predicate
    FailedSynthesis → Failure Analysis → Candidate Transform
    VerifierFailures → Boundary Analysis → Candidate Classifier

Verification: generate positive/negative examples and test accuracy.
"""

from __future__ import annotations

import enum
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.primitive import (
    Classifier,
    Predicate,
    PrimitiveRegistry,
    Transform,
    default_registry,
)

logger = logging.getLogger(__name__)


class DiscoverySource(enum.Enum):
    ANOMALY_PATTERN = "anomaly_pattern"
    FAILED_SYNTHESIS = "failed_synthesis"
    VERIFIER_FAILURE = "verifier_failure"
    LLM_PROPOSAL = "llm_proposal"
    MANUAL = "manual"


@dataclass
class CandidatePrimitive:
    """A proposed new primitive discovered by the engine.

    Attributes
    ----------
    id : str
    name : str
        Suggested name (e.g. ``contains_custom_word``).
    primitive_type : str
        ``"predicate"``, ``"transform"``, or ``"classifier"``.
    signature : str
        Input → output description.
    description : str
        Natural-language description of what this primitive does.
    implementation_hint : str
        Hint for implementing (e.g. regex pattern, Python code sketch).
    positive_examples : list of str
        Prompts that should trigger this primitive.
    negative_examples : list of str
        Prompts that should NOT trigger this primitive.
    confidence : float
        Verified accuracy on test examples.
    discovery_source : DiscoverySource
        How this primitive was discovered.
    is_verified : bool
        Whether this primitive has passed verification.
    is_promoted : bool
        Whether this primitive has been promoted to the registry.
    created_at : float
    metadata : dict
    """

    id: str = ""
    name: str = ""
    primitive_type: str = "predicate"
    signature: str = "String → Bool"
    description: str = ""
    implementation_hint: str = ""
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    confidence: float = 0.0
    discovery_source: DiscoverySource = DiscoverySource.ANOMALY_PATTERN
    is_verified: bool = False
    is_promoted: bool = False
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"cdp_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "primitive_type": self.primitive_type,
            "signature": self.signature,
            "description": self.description,
            "implementation_hint": self.implementation_hint,
            "positive_examples": self.positive_examples[:10],
            "negative_examples": self.negative_examples[:10],
            "confidence": self.confidence,
            "discovery_source": self.discovery_source.value,
            "is_verified": self.is_verified,
            "is_promoted": self.is_promoted,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


PrimitiveVerifierFn = Callable[[str], float]
"""A verifier function returns a score in [0, 1] for a given prompt."""


class PrimitiveDiscoveryEngine:
    """Discovers new primitives from anomalies, synthesis failures, and verifier gaps.

    Pipeline::

        engine.scan(anomalies, synthesis_stats, verifier_reports)
            → engine.propose()
            → engine.verify_candidates()
            → engine.promote_to_registry()
            → engine.get_all_candidates()
    """

    def __init__(
        self,
        registry: Optional[PrimitiveRegistry] = None,
        min_confidence: float = 0.8,
        max_candidates: int = 20,
        pattern_miner: Optional[Any] = None,
        llm_proposer: Optional[Any] = None,
    ) -> None:
        self.registry = registry or default_registry
        self.min_confidence = min_confidence
        self.max_candidates = max_candidates
        self._candidates: Dict[str, CandidatePrimitive] = {}
        self._promoted_names: Set[str] = set()

        from .pattern_miner import PatternMiner
        from .llm_proposer import LLMPrimitiveProposer

        self.pattern_miner = pattern_miner or PatternMiner()
        self.llm_proposer = llm_proposer or LLMPrimitiveProposer()

        logger.info(
            "PrimitiveDiscoveryEngine initialised (min_conf=%.2f, max_candidates=%d)",
            min_confidence, max_candidates,
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def scan(
        self,
        anomalies: Optional[List[Any]] = None,
        synthesis_stats: Optional[Any] = None,
        verifier_reports: Optional[List[Any]] = None,
        additional_prompts: Optional[List[Tuple[str, int]]] = None,
    ) -> int:
        """Scan multiple sources for candidate primitives.

        Returns the number of new candidates found.
        """
        before = len(self._candidates)

        # 1. Pattern mining from anomalies
        if anomalies:
            anomaly_candidates = self.pattern_miner.mine_from_anomalies(anomalies)
            for c in anomaly_candidates:
                self._add_candidate(c)

        # 2. Synthesis failure analysis
        if synthesis_stats:
            failure_candidates = self.pattern_miner.mine_from_failures(synthesis_stats)
            for c in failure_candidates:
                self._add_candidate(c)

        # 3. Verifier failure analysis
        if verifier_reports:
            for report in verifier_reports:
                vf_candidates = self.pattern_miner.mine_from_verifier(report)
                for c in vf_candidates:
                    self._add_candidate(c)

        # 4. LLM-based proposals from examples
        if additional_prompts:
            llm_candidates = self.llm_proposer.propose(additional_prompts)
            for c in llm_candidates:
                self._add_candidate(c)

        new_count = len(self._candidates) - before
        if new_count > 0:
            logger.info("Scan: found %d new candidate primitives", new_count)
        return new_count

    def propose(self) -> List[CandidatePrimitive]:
        """Return all unverified candidates ready for verification."""
        return [c for c in self._candidates.values() if not c.is_verified]

    def verify_candidates(
        self,
        verifier_fn: Optional[PrimitiveVerifierFn] = None,
    ) -> int:
        """Verify all unverified candidates.

        Parameters
        ----------
        verifier_fn : callable, optional
            Function ``fn(prompt) → score [0,1]``.  When None, uses a
            placeholder that returns 0.5 for all prompts.

        Returns
        -------
        int
            Number of candidates that passed verification.
        """
        passed = 0
        for candidate in self._candidates.values():
            if candidate.is_verified:
                continue

            # Test on positive examples
            pos_score = 0.0
            if candidate.positive_examples:
                pos_matches = sum(
                    1 for p in candidate.positive_examples
                    if (verifier_fn or self._default_verifier)(p) > 0.5
                )
                pos_score = pos_matches / len(candidate.positive_examples)

            # Test on negative examples
            neg_score = 0.0
            if candidate.negative_examples:
                neg_matches = sum(
                    1 for p in candidate.negative_examples
                    if (verifier_fn or self._default_verifier)(p) <= 0.5
                )
                neg_score = neg_matches / len(candidate.negative_examples)

            # Combined accuracy
            total = len(candidate.positive_examples) + len(candidate.negative_examples)
            if total > 0:
                combined = (pos_score * len(candidate.positive_examples) +
                            neg_score * len(candidate.negative_examples)) / total
            else:
                combined = 0.0

            candidate.confidence = combined
            candidate.is_verified = total >= 2 and combined >= self.min_confidence

            if candidate.is_verified:
                passed += 1
                logger.info(
                    "Verified candidate '%s' (type=%s, confidence=%.2f, examples=%d)",
                    candidate.name, candidate.primitive_type,
                    candidate.confidence, total,
                )

        logger.info("Verification: %d/%d candidates passed",
                     passed, len(self._candidates))
        return passed

    def promote_to_registry(self) -> int:
        """Promote verified candidates to the PrimitiveRegistry.

        Only promotes primitives whose names don't already exist in the
        registry.  Once promoted, the primitive is available for synthesis.
        """
        promoted = 0
        for candidate in self._candidates.values():
            if not candidate.is_verified or candidate.is_promoted:
                continue
            if candidate.name in self._promoted_names:
                continue

            try:
                # Check if already in registry
                existing_names = set(self.registry.list_primitives())
                if candidate.name in existing_names:
                    logger.debug("Candidate '%s' already in registry", candidate.name)
                    candidate.is_promoted = True
                    self._promoted_names.add(candidate.name)
                    continue

                # Register the new primitive
                if candidate.primitive_type == "predicate":
                    impl = _make_predicate_from_hint(candidate)
                elif candidate.primitive_type == "transform":
                    impl = _make_transform_from_hint(candidate)
                elif candidate.primitive_type == "classifier":
                    impl = _make_classifier_from_hint(candidate)
                else:
                    logger.warning("Unknown primitive type '%s'", candidate.primitive_type)
                    continue

                if impl is not None:
                    self.registry.register(impl)
                    self._promoted_names.add(candidate.name)
                    candidate.is_promoted = True
                    promoted += 1
                    logger.info(
                        "Promoted '%s' (%s) to registry",
                        candidate.name, candidate.primitive_type,
                    )
            except Exception as exc:
                logger.warning("Failed to promote '%s': %s", candidate.name, exc)

        logger.info("Promotion: %d/%d candidates promoted to registry",
                     promoted, len(self._candidates))
        return promoted

    def get_all_candidates(self) -> List[CandidatePrimitive]:
        return list(self._candidates.values())

    def get_promoted(self) -> List[CandidatePrimitive]:
        return [c for c in self._candidates.values() if c.is_promoted]

    def retire(self, candidate_id: str) -> bool:
        """Remove a candidate (e.g. low accuracy after re-evaluation)."""
        if candidate_id in self._candidates:
            removed = self._candidates.pop(candidate_id)
            self._promoted_names.discard(removed.name)
            logger.info("Retired candidate '%s'", removed.name)
            return True
        return False

    def run_full_pipeline(
        self,
        anomalies: Optional[List[Any]] = None,
        synthesis_stats: Optional[Any] = None,
        verifier_reports: Optional[List[Any]] = None,
        additional_prompts: Optional[List[Tuple[str, int]]] = None,
        verifier_fn: Optional[PrimitiveVerifierFn] = None,
    ) -> Dict[str, int]:
        """Run the full discovery pipeline end-to-end.

        Returns counts::
            {"scanned": int, "verified": int, "promoted": int}
        """
        scanned = self.scan(anomalies, synthesis_stats, verifier_reports, additional_prompts)
        verified = self.verify_candidates(verifier_fn)
        promoted = self.promote_to_registry()
        return {"scanned": scanned, "verified": verified, "promoted": promoted}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_candidate(self, candidate: CandidatePrimitive) -> None:
        if candidate.name and candidate.name not in self._candidates:
            self._candidates[candidate.id] = candidate

    @staticmethod
    def _default_verifier(prompt: str) -> float:
        return 0.5


# ------------------------------------------------------------------
# Factory functions for primitive creation
# ------------------------------------------------------------------


def _make_predicate_from_hint(candidate: CandidatePrimitive) -> Optional[Predicate]:
    """Create a Predicate from a candidate's implementation hint."""
    hint = candidate.implementation_hint or ""
    name = candidate.name
    if "regex" in hint.lower() or "re." in hint:
        import re
        pattern = _extract_regex(hint)
        if pattern:
            return Predicate(
                name=name,
                parameters={"pattern": pattern},
                description=candidate.description,
            )
    # Generic predicate wrapper
    return Predicate(
        name=name,
        parameters={},
        description=candidate.description,
    )


def _make_transform_from_hint(candidate: CandidatePrimitive) -> Optional[Transform]:
    return Transform(
        name=candidate.name,
        parameters={},
        description=candidate.description,
    )


def _make_classifier_from_hint(candidate: CandidatePrimitive) -> Optional[Classifier]:
    return Classifier(
        name=candidate.name,
        parameters={},
        description=candidate.description,
    )


def _extract_regex(text: str) -> Optional[str]:
    """Extract a regex pattern from a hint string."""
    import re
    patterns = re.findall(r"[rR][\"']([^\"']+)[\"']", text)
    if patterns:
        return patterns[0]
    patterns = re.findall(r"re\.compile\([\"']([^\"']+)[\"']\)", text)
    return patterns[0] if patterns else None
