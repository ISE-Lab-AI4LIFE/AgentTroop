"""Stage 5: Semantic memory — tracks exploration coverage and stores observations.

Independent of Version Space posterior.
Tracks which score regions are well-explored and which remain uncertain.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SemanticObservation:
    """A single semantic observation.

    Attributes
    ----------
    prompt : str
        The prompt that was probed.
    primitive_name : str
        Which semantic primitive was used.
    score : float
        The semantic score of the prompt (0-1).
    outcome : int
        1 = REFUSE, 0 = REFUSE_FAIL.
    round : int
        The intervention round.
    """
    prompt: str
    primitive_name: str
    score: float
    outcome: int
    round: int


@dataclass
class ScoreRegion:
    """A region of semantic score space with exploration statistics.

    Attributes
    ----------
    score_low : float
        Lower bound of the region.
    score_high : float
        Upper bound of the region.
    num_observations : int
        Number of interventions in this region.
    num_refuse : int
        Number of REFUSE outcomes in this region.
    num_accept : int
        Number of ACCEPT outcomes in this region.
    last_observation_time : float
        Timestamp of the most recent observation.
    """
    score_low: float
    score_high: float
    num_observations: int = 0
    num_refuse: int = 0
    num_accept: int = 0
    last_observation_time: float = 0.0

    @property
    def refusal_rate(self) -> float:
        if self.num_observations == 0:
            return 0.5
        return self.num_refuse / self.num_observations

    @property
    def is_explored(self) -> bool:
        return self.num_observations >= 3

    @property
    def uncertainty(self) -> float:
        if self.num_observations == 0:
            return 1.0
        p = self.refusal_rate
        n = self.num_observations
        return (p * (1 - p) / max(n, 1)) ** 0.5

    def record_observation(self, outcome: int) -> None:
        self.num_observations += 1
        if outcome == 1:
            self.num_refuse += 1
        else:
            self.num_accept += 1
        self.last_observation_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_low": round(self.score_low, 3),
            "score_high": round(self.score_high, 3),
            "num_observations": self.num_observations,
            "refusal_rate": round(self.refusal_rate, 4),
            "uncertainty": round(self.uncertainty, 4),
            "is_explored": self.is_explored,
        }


class SemanticStore:
    """Memory for semantic score observations.

    Stores prompt-level observations grouped by semantic primitive.
    Tracks coverage across score regions.
    Independent from Version Space posterior.
    """

    def __init__(self, num_regions: int = 10) -> None:
        self.num_regions = num_regions
        self.target_program: str = ""
        self._observations: Dict[str, List[Dict[str, Any]]] = {}
        self._regions: Dict[str, List[ScoreRegion]] = {}
        self._hypotheses: List[str] = []
        self._history: List[SemanticObservation] = []

    # ── Engine-compatible API ─────────────────────────────────────────────

    def initialise(self, target_program: str) -> None:
        """Initialise the store for a new target program."""
        self.target_program = target_program
        self._observations.clear()
        self._regions.clear()
        self._hypotheses.clear()
        self._history.clear()

    def store_observation(self, obs: SemanticObservation) -> None:
        """Store a SemanticObservation (engine API)."""
        self._history.append(obs)
        self.record(
            primitive_name=obs.primitive_name,
            prompt=obs.prompt,
            score=obs.score,
            outcome=obs.outcome,
        )

    def get_history(self) -> List[SemanticObservation]:
        """Return all stored observations in order."""
        return self._history.copy()

    def get_hypotheses(self) -> List[str]:
        return self._hypotheses.copy()

    def set_hypotheses(self, hypotheses: List[str]) -> None:
        self._hypotheses = list(hypotheses)

    def get_regions(self, primitive_name: Optional[str] = None) -> Dict[str, List[ScoreRegion]]:
        """Get regions grouped by primitive, or for a single primitive."""
        if primitive_name:
            return {primitive_name: self._regions.get(primitive_name, []).copy()}
        return {k: v.copy() for k, v in self._regions.items()}

    def get_observations(self, primitive_name: Optional[str] = None) -> List[Any]:
        """Get observations, optionally filtered by primitive name."""
        if primitive_name:
            return self._observations.get(primitive_name, []).copy()
        result: List[Dict[str, Any]] = []
        for obs_list in self._observations.values():
            result.extend(obs_list)
        return result

    # ── Original API ──────────────────────────────────────────────────────

    def _ensure_primitive(self, name: str) -> None:
        if name not in self._observations:
            self._observations[name] = []
            self._regions[name] = []
            bin_width = 1.0 / self.num_regions
            for i in range(self.num_regions):
                low = round(i * bin_width, 4)
                high = round((i + 1) * bin_width, 4)
                self._regions[name].append(ScoreRegion(score_low=low, score_high=high))

    def record(
        self,
        primitive_name: str,
        prompt: str,
        score: float,
        outcome: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a semantic observation."""
        self._ensure_primitive(primitive_name)
        self._observations[primitive_name].append({
            "prompt": prompt,
            "score": max(0.0, min(1.0, float(score))),
            "outcome": int(outcome),
            "timestamp": time.time(),
            "metadata": metadata or {},
        })
        score_clamped = max(0.0, min(1.0, float(score)))
        for region in self._regions[primitive_name]:
            if region.score_low <= score_clamped < region.score_high:
                region.record_observation(int(outcome))
                break

    def get_observations_by_primitive(self, primitive_name: str) -> List[Dict[str, Any]]:
        return self._observations.get(primitive_name, []).copy()

    def get_regions_for_primitive(self, primitive_name: str) -> List[ScoreRegion]:
        return self._regions.get(primitive_name, []).copy()

    def get_uncertain_regions(
        self, primitive_name: str, min_uncertainty: float = 0.1
    ) -> List[ScoreRegion]:
        """Return regions with high uncertainty (poorly explored)."""
        return [
            r for r in self.get_regions_for_primitive(primitive_name)
            if r.uncertainty >= min_uncertainty
        ]

    def get_num_observations(self, primitive_name: str) -> int:
        return len(self._observations.get(primitive_name, []))

    def get_coverage_summary(self, primitive_name: str) -> Dict[str, Any]:
        """Get a summary of exploration coverage for a primitive."""
        regions = self.get_regions_for_primitive(primitive_name)
        explored = sum(1 for r in regions if r.is_explored)
        total = len(regions)
        return {
            "primitive_name": primitive_name,
            "total_regions": total,
            "explored_regions": explored,
            "coverage_pct": round(explored / max(total, 1) * 100, 1),
            "total_observations": self.get_num_observations(primitive_name),
            "mean_uncertainty": round(
                sum(r.uncertainty for r in regions) / max(len(regions), 1), 4
            ),
            "regions": [r.to_dict() for r in regions],
        }

    def best_primitive(self) -> Optional[str]:
        """Return the primitive with the most coverage (best explored)."""
        best_name = None
        best_count = -1
        for name in self._observations:
            c = sum(1 for r in self._regions.get(name, []) if r.is_explored)
            if c > best_count:
                best_count = c
                best_name = name
        return best_name

    def primitive_names(self) -> List[str]:
        return list(self._observations.keys())

    def clear(self, primitive_name: Optional[str] = None) -> None:
        if primitive_name:
            self._observations.pop(primitive_name, None)
            self._regions.pop(primitive_name, None)
        else:
            self._observations.clear()
            self._regions.clear()
