"""Stage 3: SDE Engine — orchestrates the semantic discovery lifecycle.

The Engine:
1. Receives a target program (and optionally hypotheses)
2. Selects appropriate semantic primitives
3. Manages boundary estimation for each primitive
4. Designs interventions (via BoundaryAwareStrategist)
5. Verifies boundary quality (via SemanticVerifier)
6. Routes between symbolic/semantic/hybrid (via Router)

Lifecycle per round:
  1. Router decides mode (symbolic / semantic / hybrid)
  2. If semantic/hybrid: strategist proposes intervention(s)
  3. Intervention is executed (victim produces outcome)
  4. Boundary estimator updates posterior with (score, outcome)
  5. Verifier checks boundary consistency
  6. Repeat until convergence or max rounds
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .boundary_estimator import BayesianBoundaryEstimator, BoundaryEstimate
from .boundary_strategist import BoundaryAwareStrategist, SemanticIntervention
from .composite_boundary_estimator import (
    CompositeBoundaryEstimator,
    CompositeBoundaryEstimate,
)
from .concept_discovery import SemanticConceptDiscovery, ConceptExplanation
from .embedding_primitive import EmbeddingSemanticScorer, get_global_scorer
from .hybrid_synthesizer import HybridProbe, HybridSynthesiser
from .multi_dim_boundary import MultiDimensionalBoundaryEstimator, MultiBoundaryEstimate
from .prompt_embedding_store import PromptEmbeddingStore
from .prompt_generator import SemanticPromptGenerator
from .router import RoutingDecision, RoutingMode, SemanticRouter
from .score_primitives import (
    _ALL_SEMANTIC_PRIMITIVES,
    _compute_instruction_score,
    _compute_harmfulness_score,
    _compute_procedurality_score,
    _compute_jailbreak_score,
)
from .semantic_store import SemanticStore, SemanticObservation
from .semantic_verifier import BoundaryConsistencyReport, SemanticVerifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default score functions
# ---------------------------------------------------------------------------

_DEFAULT_SCORE_FUNCTIONS: Dict[str, Callable[[str], float]] = {
    "instruction_score": _compute_instruction_score,
    "harmfulness_score": _compute_harmfulness_score,
    "procedurality_score": _compute_procedurality_score,
    "jailbreak_score": _compute_jailbreak_score,
}


# ---------------------------------------------------------------------------
# Engine state
# ---------------------------------------------------------------------------

@dataclass
class SDEState:
    """Snapshot of the engine's internal state.

    Attributes
    ----------
    round : int
        Current round number.
    mode : str
        Current routing mode.
    active_primitive : Optional[str]
        Currently targeted primitive.
    num_observations : int
        Total observations across all primitives.
    mean_uncertainty : float
        Average posterior uncertainty across estimators.
    is_converged : bool
        Whether the engine believes it has found the boundary.
    best_theta : Optional[float]
        Best estimate of the boundary threshold.
    latest_probe_score : Optional[float]
        Score of the most recent probe.
    latest_outcome : Optional[int]
        Outcome of the most recent probe.
    """
    round: int = 0
    mode: str = "init"
    active_primitive: Optional[str] = None
    num_observations: int = 0
    mean_uncertainty: float = 1.0
    is_converged: bool = False
    best_theta: Optional[float] = None
    latest_probe_score: Optional[float] = None
    latest_outcome: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "mode": self.mode,
            "active_primitive": self.active_primitive,
            "num_observations": self.num_observations,
            "mean_uncertainty": round(self.mean_uncertainty, 4),
            "is_converged": self.is_converged,
            "best_theta": round(self.best_theta, 4) if self.best_theta is not None else None,
            "latest_probe_score": round(self.latest_probe_score, 4) if self.latest_probe_score is not None else None,
            "latest_outcome": self.latest_outcome,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SemanticDiscoveryEngine:
    """Orchestrates the semantic discovery lifecycle.

    Parameters
    ----------
    score_functions : Dict[str, Callable]
        Mapping from primitive name to score function.
    prompt_generator : SemanticPromptGenerator
    semantic_store : SemanticStore
    strategist : BoundaryAwareStrategist
    verifier : SemanticVerifier
    router : SemanticRouter
    hybrid_synthesiser : HybridSynthesiser
    convergence_std : float
        Stop when uncertainty < this (default 0.05).
    max_rounds : int
        Maximum intervention rounds (default 50).
    """

    def __init__(
        self,
        score_functions: Optional[Dict[str, Callable[[str], float]]] = None,
        prompt_generator: Optional[SemanticPromptGenerator] = None,
        semantic_store: Optional[SemanticStore] = None,
        strategist: Optional[BoundaryAwareStrategist] = None,
        verifier: Optional[SemanticVerifier] = None,
        router: Optional[SemanticRouter] = None,
        hybrid_synthesiser: Optional[HybridSynthesiser] = None,
        convergence_std: float = 0.05,
        max_rounds: int = 50,
        embedding_scorer: Optional[EmbeddingSemanticScorer] = None,
        prompt_embedding_store: Optional[PromptEmbeddingStore] = None,
        multi_dim_estimator: Optional[MultiDimensionalBoundaryEstimator] = None,
        concept_discovery: Optional[SemanticConceptDiscovery] = None,
        use_composite: bool = False,
        composite_estimator: Optional[CompositeBoundaryEstimator] = None,
    ) -> None:
        self.score_functions = score_functions or _DEFAULT_SCORE_FUNCTIONS
        self.prompt_generator = prompt_generator or SemanticPromptGenerator()
        self.semantic_store = semantic_store or SemanticStore()
        self.verifier = verifier or SemanticVerifier()
        self.hybrid_synthesiser = hybrid_synthesiser or HybridSynthesiser()
        self.convergence_std = convergence_std
        self.max_rounds = max_rounds
        # Use global scorer singleton to guarantee single-source-of-truth
        self.embedding_scorer = embedding_scorer or get_global_scorer()
        self.prompt_embedding_store = prompt_embedding_store or PromptEmbeddingStore()
        self.multi_dim_estimator = multi_dim_estimator or MultiDimensionalBoundaryEstimator()
        self.concept_discovery = concept_discovery or SemanticConceptDiscovery()
        self.use_composite = use_composite
        self.composite_estimator = composite_estimator or CompositeBoundaryEstimator(
            primitive_names=list(self.score_functions.keys()),
        )

        # Create boundary estimators for each primitive
        self.boundary_estimators: Dict[str, BayesianBoundaryEstimator] = {
            name: BayesianBoundaryEstimator(primitive_name=name)
            for name in self.score_functions
        }

        self.strategist = strategist or BoundaryAwareStrategist(
            prompt_generator=self.prompt_generator,
            score_functions=self.score_functions,
            prompt_embedding_store=self.prompt_embedding_store,
            embedding_scorer=self.embedding_scorer,
        )
        self.router = router or SemanticRouter(
            strategist=self.strategist,
            verifier=self.verifier,
        )

        # Internal state
        self._round: int = 0
        self._converged: bool = False
        self._last_decision: Optional[RoutingDecision] = None
        self._last_intervention_primitive: Optional[str] = None
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialise(
        self,
        target_program: str,
        hypotheses: Optional[List[str]] = None,
    ) -> SDEState:
        """Initialise the engine for a new target program.

        Resets all estimators and store, then returns initial state.

        Parameters
        ----------
        target_program : str
            The program being analysed.
        hypotheses : List[str], optional
            Initial hypotheses to guide routing.

        Returns
        -------
        SDEState
        """
        self._reset()
        self.semantic_store.initialise(target_program)
        if hypotheses:
            self.semantic_store.set_hypotheses(hypotheses)
        return self.get_state()

    _primitive_cycle_index: int = 0

    def propose_intervention(
        self,
        base_prompt: str,
        hypotheses: Optional[List[str]] = None,
    ) -> SemanticIntervention:
        """Propose the next intervention.

        Cycles through available primitives round-robin to ensure all
        semantic dimensions are explored (not just instruction_score).

        Parameters
        ----------
        base_prompt : str
            Original prompt to vary.
        hypotheses : List[str], optional
            Current hypotheses for routing.

        Returns
        -------
        SemanticIntervention
        """
        self._round += 1
        if hypotheses is None:
            hypotheses = self.semantic_store.get_hypotheses()
        boundary_reports = self._build_reports()
        total_obs = sum(e.num_observations for e in self.boundary_estimators.values())
        decision = self.router.route(hypotheses, boundary_reports, total_observations=total_obs)
        self._last_decision = decision

        # Round-robin through all primitives to ensure multi-dim exploration
        all_primitives = sorted(self.score_functions.keys())
        prim = all_primitives[self._primitive_cycle_index % len(all_primitives)]
        self._primitive_cycle_index = (self._primitive_cycle_index + 1) % len(all_primitives)

        est = self.boundary_estimators.get(prim)
        if est is None:
            est = BayesianBoundaryEstimator(primitive_name=prim)
            self.boundary_estimators[prim] = est

        if est.num_observations < 3:
            inter = self.strategist.design_gradient_probes(
                base_prompt, prim, est, n_probes=1
            )[0]
        else:
            inter = self.strategist.design_intervention(base_prompt, prim, est)

        # Store primitive for observe_outcome to use
        self._last_intervention_primitive = prim
        return inter

    def observe_outcome(
        self,
        prompt: str,
        score: float,
        outcome: int,
        primitive_name: Optional[str] = None,
    ) -> SDEState:
        """Record an observation from an intervention.

        Parameters
        ----------
        prompt : str
            The prompt that was sent.
        score : float
            The semantic score of the prompt.
        outcome : int
            1 = REFUSE, 0 = REFUSE_FAIL.
        primitive_name : str, optional
            The primitive used. If None, uses the last intervention's primitive.

        Returns
        -------
        SDEState
        """
        if primitive_name is None and self._last_intervention_primitive is not None:
            primitive_name = self._last_intervention_primitive
        if primitive_name is None:
            primitive_name = "instruction_score"

        estimator = self.boundary_estimators.get(primitive_name)
        if estimator is None:
            estimator = BayesianBoundaryEstimator(primitive_name=primitive_name)
            self.boundary_estimators[primitive_name] = estimator
        estimator.observe(score, outcome)

        # Feed multi-dimensional estimator using embedding scorer for consistency
        centroid_map = {
            "instruction_score": "instruction",
            "harmfulness_score": "harmful",
            "jailbreak_score": "jailbreak",
            "procedurality_score": "procedural",
        }
        scores_dict: Dict[str, float] = {
            name: self.embedding_scorer.score(prompt, centroid_map.get(name, "instruction")).final
            for name in self.score_functions
        }
        self.multi_dim_estimator.observe(scores_dict, outcome)

        # Feed composite boundary estimator
        self.composite_estimator.observe(scores_dict, outcome)

        # Store in embedding store for diversity tracking
        self.prompt_embedding_store.add(prompt, source="intervention", score_vector=scores_dict)

        self.semantic_store.store_observation(
            SemanticObservation(
                prompt=prompt,
                primitive_name=primitive_name,
                score=score,
                outcome=outcome,
                round=self._round,
            )
        )

        # Check convergence
        if estimator.estimate().posterior_std < self.convergence_std:
            self._converged = True

        # Check consistency
        report = self.verifier.verify_boundary(estimator)
        if not report.is_consistent and estimator.num_observations > 10:
            logger.warning(
                f"Boundary inconsistency for {primitive_name}: "
                f"pass_rate={report.details.get('pass_rate', '?'):.3f}"
            )

        state = self.get_state()
        state.latest_probe_score = score
        state.latest_outcome = outcome
        self._history.append(state.to_dict())
        return state

    def get_state(self) -> SDEState:
        """Get the current engine state."""
        obs_total = sum(
            e.num_observations for e in self.boundary_estimators.values()
        )
        uncertainties = [
            e.estimate().posterior_std
            for e in self.boundary_estimators.values()
            if e.num_observations > 0
        ]
        mean_unc = float(np.mean(uncertainties)) if uncertainties else 1.0
        best_estimator = max(
            self.boundary_estimators.values(),
            key=lambda e: e.num_observations,
            default=None,
        )
        best_theta = best_estimator.estimate().posterior_mean if best_estimator else None
        return SDEState(
            round=self._round,
            mode=str(self._last_decision.mode) if self._last_decision else "init",
            active_primitive=self._last_decision.recommended_primitives[0]
            if self._last_decision and self._last_decision.recommended_primitives
            else None,
            num_observations=obs_total,
            mean_uncertainty=round(float(mean_unc), 4),
            is_converged=self._converged,
            best_theta=round(float(best_theta), 4) if best_theta is not None else None,
        )

    def get_boundary_estimate(
        self,
        primitive_name: str,
    ) -> Optional[BoundaryEstimate]:
        """Get the boundary estimate for a primitive."""
        estimator = self.boundary_estimators.get(primitive_name)
        if estimator is None:
            return None
        return estimator.estimate()

    def get_consistency_report(
        self,
        primitive_name: str,
    ) -> Optional[BoundaryConsistencyReport]:
        """Get boundary consistency report for a primitive."""
        estimator = self.boundary_estimators.get(primitive_name)
        if estimator is None or estimator.num_observations < 3:
            return None
        return self.verifier.verify_boundary(estimator)

    def should_stop(self) -> bool:
        """Check if the engine should stop."""
        if self._round >= self.max_rounds:
            return True
        return self._converged

    def get_multi_dim_estimate(self) -> Optional[MultiBoundaryEstimate]:
        """Get the multi-dimensional boundary estimate."""
        return self.multi_dim_estimator.estimate() if self.multi_dim_estimator.num_observations > 0 else None

    def get_semantic_evidence(self) -> Optional[Dict[str, Any]]:
        """Package current semantic state as evidence for the strategist.

        Returns a dict compatible with ``SemanticEvidence`` construction,
        or ``None`` if the engine is not yet initialised with observations.

        The returned dict contains:
          - is_active: True when engine has observations
          - instruction_score, harmfulness_score, jailbreak_score:
            latest centroid proximity scores
          - boundary_uncertainty: average posterior std across estimators
          - concepts: discovered semantic concepts (list of str)
          - recommended_primitives: primitives with high uncertainty (>0.1)
        """
        obs = self.semantic_store.get_history()
        has_obs = len(obs) > 0
        if self._round == 0 and not has_obs:
            return None
        centroid_map = {
            "instruction_score": "instruction",
            "harmfulness_score": "harmful",
            "jailbreak_score": "jailbreak",
        }
        last_prompt = obs[-1].prompt if obs else ""
        scores = {}
        for key, centroid in centroid_map.items():
            try:
                sc = self.embedding_scorer.score(last_prompt, centroid)
                scores[key] = round(float(sc.final), 4)
            except Exception:
                scores[key] = 0.0

        # Boundary uncertainty: average posterior std
        uncertainties = []
        for e in self.boundary_estimators.values():
            if e.num_observations > 0:
                try:
                    uncertainties.append(e.estimate().posterior_std)
                except Exception:
                    pass
        mean_unc = float(np.mean(uncertainties)) if uncertainties else 1.0

        # Recommended primitives: those with high remaining uncertainty
        recommended = []
        for n, e in self.boundary_estimators.items():
            if e.num_observations > 0:
                try:
                    if e.estimate().posterior_std > 0.1:
                        recommended.append(n)
                except Exception:
                    pass
        recommended.sort(
            key=lambda n: (
                self.boundary_estimators[n].estimate().posterior_std
                if self.boundary_estimators[n].num_observations > 0
                else 0.0
            ),
            reverse=True,
        )

        # Concepts from concept discovery
        concept_expl = self.get_concept_explanation()
        concepts = []
        if concept_expl is not None:
            concepts = [
                c.to_dict() if hasattr(c, 'to_dict') else {
                    "name": getattr(c, 'name', str(c)),
                    "keywords": getattr(c, 'keywords', []),
                    "observation_count": getattr(c, 'observation_count', 0),
                    "refuse_rate": getattr(c, 'refuse_rate', 0.5),
                    "confidence": getattr(c, 'confidence', 0.0),
                }
                for c in getattr(concept_expl, 'concepts', [])
            ]

        return {
            "is_active": self._round > 0 or has_obs,
            "instruction_score": scores.get("instruction_score", 0.0),
            "harmfulness_score": scores.get("harmfulness_score", 0.0),
            "jailbreak_score": scores.get("jailbreak_score", 0.0),
            "boundary_uncertainty": round(float(mean_unc), 4),
            "concepts": concepts,
            "recommended_primitives": recommended,
            "composite_uncertainty": round(
                float(self.composite_estimator.estimate().uncertainty), 4
            ),
        }

    def get_concept_explanation(self) -> Optional[ConceptExplanation]:
        """Get an interpretable concept explanation from observations."""
        obs = self.semantic_store.get_history()
        if len(obs) < 3:
            return None
        return self.concept_discovery.explain(obs)

    def get_embedding_coverage(self) -> Dict[str, float]:
        """Get embedding diversity coverage statistics."""
        return self.prompt_embedding_store.embedding_coverage()

    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """Get the full history of engine states."""
        return self._history

    def get_routing_history(self) -> List[Dict[str, Any]]:
        """Get routing decision history."""
        return self.router.recent_mode_history(20)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._round = 0
        self._converged = False
        self._last_decision = None
        self._history = []
        self.boundary_estimators = {
            name: BayesianBoundaryEstimator(primitive_name=name)
            for name in self.score_functions
        }
        self.composite_estimator.reset()

    def _build_reports(
        self,
    ) -> Dict[str, BoundaryConsistencyReport]:
        reports: Dict[str, BoundaryConsistencyReport] = {}
        for name, estimator in self.boundary_estimators.items():
            if estimator.num_observations >= 3:
                reports[name] = self.verifier.verify_boundary(estimator)
        return reports
