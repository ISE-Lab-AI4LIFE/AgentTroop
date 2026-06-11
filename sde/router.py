"""Stage 8: Router — decides which mode to use: symbolic, semantic, or hybrid.

Routing decisions:
- Primitive programs → symbolic mode (use existing PAIR/TAP/etc.)
- Boundary exploration → semantic mode (use SDE probes)
- High-entropy control panels → hybrid mode (co-trained)

The Router is a lightweight component that inspects the current state
and returns a routing decision.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .boundary_strategist import BoundaryAwareStrategist
from .semantic_verifier import SemanticVerifier, BoundaryConsistencyReport

logger = logging.getLogger(__name__)


class RoutingMode(Enum):
    SYMBOLIC = "symbolic"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

    def __str__(self) -> str:
        return self.value


@dataclass
class RoutingDecision:
    """Decision from the Router.

    Attributes
    ----------
    mode : RoutingMode
        Which mode to use.
    confidence : float
        Confidence in this decision (0-1).
    reason : str
        Explanation for the decision.
    recommended_primitives : List[str]
        Which semantic primitives to use (empty for symbolic).
    """
    mode: RoutingMode
    confidence: float
    reason: str
    recommended_primitives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": str(self.mode),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "recommended_primitives": self.recommended_primitives,
        }


class SemanticRouter:
    """Routes between symbolic, semantic, and hybrid modes.

    The Router is called at the start of each intervention round.
    It decides which probe generation strategy is most likely to succeed.

    Rules (simplified):
    1. If all primitives are inconsistent → hybrid mode
    2. If no primitives are active → symbolic mode
    3. If primitives are consistent but uncertain → semantic mode
    4. If primitives have collapsed → symbolic mode (fallback)

    Parameters
    ----------
    strategist : BoundaryAwareStrategist
    verifier : SemanticVerifier
    verbose : bool
    """

    def __init__(
        self,
        strategist: Optional[BoundaryAwareStrategist] = None,
        verifier: Optional[SemanticVerifier] = None,
        verbose: bool = False,
    ) -> None:
        self.strategist = strategist
        self.verifier = verifier or SemanticVerifier()
        self.verbose = verbose
        self._mode_history: List[RoutingDecision] = []

    def route(
        self,
        hypotheses: List[str],
        boundary_reports: Optional[Dict[str, BoundaryConsistencyReport]] = None,
        total_observations: int = 0,
    ) -> RoutingDecision:
        """Determine the best mode for the current round.

        Confidence is evidence-based:
            confidence = consistency × calibration × coverage × observation_factor

        Always prefers SEMANTIC or HYBRID mode when the semantic subsystem
        is active. The strategist's should_activate is no longer a gate —
        it's only used for diagnostics.

        Parameters
        ----------
        hypotheses : List[str]
            Current hypotheses from the hypothesis generator.
        boundary_reports : Dict[str, BoundaryConsistencyReport]
            Latest boundary consistency reports per primitive.
        total_observations : int
            Total observations across all primitives.

        Returns
        -------
        RoutingDecision
        """
        # If no boundary reports available
        if not boundary_reports:
            decision = RoutingDecision(
                mode=RoutingMode.SEMANTIC,
                confidence=0.5,
                reason="No boundary data yet; starting semantic exploration",
                recommended_primitives=["instruction_score"],
            )
            self._mode_history.append(decision)
            return decision

        # Analyse boundary reports
        reports = list(boundary_reports.values())
        all_inconsistent = all(not r.is_consistent for r in reports)
        any_collapsed = any(r.collapse_detected for r in reports)
        avg_calibration = float(np.mean([r.calibration_error for r in reports]))
        avg_monotonicity = float(np.mean([r.monotonicity_score for r in reports]))

        # Evidence-based confidence computation
        consistency = 0.0 if all_inconsistent else (1.0 - float(np.mean([0.0 if not r.is_consistent else 1.0 for r in reports])))
        consistency = 1.0 - float(np.mean([0.0 if not r.is_consistent else 1.0 for r in reports]))
        consistency = float(np.mean([1.0 if r.is_consistent else 0.0 for r in reports]))
        calibration = max(0.0, 1.0 - avg_calibration)
        coverage = max(0.0, float(np.mean([1.0 - r.collapse_score for r in reports])))
        obs_factor = min(1.0, total_observations / 20.0) if total_observations > 0 else 0.3
        monotonicity = avg_monotonicity

        evidence_confidence = consistency * calibration * coverage * monotonicity * (0.5 + 0.5 * obs_factor)

        if all_inconsistent and any_collapsed:
            decision = RoutingDecision(
                mode=RoutingMode.SYMBOLIC,
                confidence=max(0.3, min(0.7, evidence_confidence)),
                reason=f"All inconsistent + collapse; symbolic fallback (conf={evidence_confidence:.3f})",
                recommended_primitives=[],
            )
        elif all_inconsistent:
            decision = RoutingDecision(
                mode=RoutingMode.HYBRID,
                confidence=max(0.3, min(0.7, evidence_confidence)),
                reason=f"All ({len(reports)}) primitives inconsistent; hybrid (conf={evidence_confidence:.3f})",
                recommended_primitives=self._best_primitives(reports),
            )
        elif any_collapsed:
            decision = RoutingDecision(
                mode=RoutingMode.SYMBOLIC,
                confidence=max(0.3, min(0.7, evidence_confidence)),
                reason=f"Collapse in {self._collapsed_names(reports)}; symbolic (conf={evidence_confidence:.3f})",
                recommended_primitives=[],
            )
        elif evidence_confidence < 0.3:
            decision = RoutingDecision(
                mode=RoutingMode.HYBRID,
                confidence=max(0.3, evidence_confidence),
                reason=f"Low confidence ({evidence_confidence:.3f}); hybrid mode",
                recommended_primitives=self._best_primitives(reports),
            )
        else:
            decision = RoutingDecision(
                mode=RoutingMode.SEMANTIC,
                confidence=max(0.4, min(0.95, evidence_confidence)),
                reason=f"Confidence={evidence_confidence:.3f} (cal={avg_calibration:.3f}, mono={avg_monotonicity:.3f}); semantic",
                recommended_primitives=self._best_primitives(reports),
            )

        self._mode_history.append(decision)
        return decision

    def route_for_primitive(
        self,
        primitive_name: str,
        report: Optional[BoundaryConsistencyReport],
    ) -> RoutingDecision:
        """Route for a single primitive (fine-grained).

        Confidence is evidence-based using report metrics.
        """
        if report is None:
            return RoutingDecision(
                mode=RoutingMode.SEMANTIC,
                confidence=0.5,
                reason=f"No report for {primitive_name}; exploring",
                recommended_primitives=[primitive_name],
            )
        consistency = 1.0 if report.is_consistent else 0.0
        calibration = max(0.0, 1.0 - report.calibration_error)
        coverage = max(0.0, 1.0 - report.collapse_score)
        monotonicity = report.monotonicity_score
        evidence = consistency * calibration * coverage * monotonicity

        if report.collapse_detected:
            return RoutingDecision(
                mode=RoutingMode.SYMBOLIC,
                confidence=max(0.2, min(0.6, evidence)),
                reason=f"Primitive '{primitive_name}' collapsed (conf={evidence:.3f}); symbolic",
                recommended_primitives=[],
            )
        if not report.is_consistent:
            return RoutingDecision(
                mode=RoutingMode.HYBRID,
                confidence=max(0.3, min(0.7, evidence)),
                reason=f"Primitive '{primitive_name}' inconsistent (conf={evidence:.3f}); hybrid",
                recommended_primitives=[primitive_name],
            )
        return RoutingDecision(
            mode=RoutingMode.SEMANTIC,
            confidence=max(0.4, min(0.95, evidence)),
            reason=f"Primitive '{primitive_name}' consistent (conf={evidence:.3f}); semantic",
            recommended_primitives=[primitive_name],
        )

    def recent_mode_history(self, n: int = 5) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self._mode_history[-n:]]

    @staticmethod
    def _best_primitives(
        reports: List[BoundaryConsistencyReport],
    ) -> List[str]:
        sorted_reports = sorted(
            reports, key=lambda r: r.calibration_error
        )
        return [r.primitive_name for r in sorted_reports[:2]]

    @staticmethod
    def _collapsed_names(
        reports: List[BoundaryConsistencyReport],
    ) -> str:
        collapsed = [r.primitive_name for r in reports if r.collapse_detected]
        return ", ".join(collapsed) if collapsed else "none"
