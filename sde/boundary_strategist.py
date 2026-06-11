"""Stage 3 + Stage 4: Boundary-aware strategist.

Selects interventions to maximise information about decision boundaries
rather than hypothesis disagreement.

Key difference from existing StrategistAgent:
- Existing:   maximise |pred1 - pred2| (hypothesis disagreement)
- Semantic:   minimise uncertainty about P(θ) (boundary information)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .boundary_estimator import BayesianBoundaryEstimator, BoundaryEstimate
from .embedding_primitive import EmbeddingSemanticScorer, get_global_scorer
from .prompt_generator import SemanticPromptGenerator
from .score_primitives import (
    _ALL_SEMANTIC_PRIMITIVES,
    _compute_instruction_score,
    _compute_harmfulness_score,
    _compute_procedurality_score,
    _compute_jailbreak_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score function registry
# ---------------------------------------------------------------------------

_SCORE_FUNCTIONS: Dict[str, Callable[[str], float]] = {
    "instruction_score": _compute_instruction_score,
    "harmfulness_score": _compute_harmfulness_score,
    "procedurality_score": _compute_procedurality_score,
    "jailbreak_score": _compute_jailbreak_score,
}


@dataclass
class SemanticIntervention:
    """An intervention designed by the boundary-aware strategist.

    Attributes
    ----------
    prompt : str
        The prompt to send to the victim.
    primitive_name : str
        The semantic score primitive being probed.
    target_score : float
        The intended semantic score.
    actual_score : float
        The actual computed semantic score.
    expected_information_gain : float
        Estimated information gain from this intervention.
    is_boundary_probe : bool
        Whether this targets an uncertain boundary region.
    """
    prompt: str
    primitive_name: str
    target_score: float
    actual_score: float
    expected_information_gain: float
    is_boundary_probe: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt[:200],
            "primitive_name": self.primitive_name,
            "target_score": self.target_score,
            "actual_score": self.actual_score,
            "expected_information_gain": round(self.expected_information_gain, 4),
            "is_boundary_probe": self.is_boundary_probe,
        }


class BoundaryAwareStrategist:
    """Designs interventions to maximise boundary information.

    This is the semantic analogue of StrategistAgent.
    Where StrategistAgent maximises hypothesis disagreement,
    BoundaryAwareStrategist minimises boundary uncertainty.

    Parameters
    ----------
    prompt_generator : SemanticPromptGenerator
        Generator for semantic prompt variants.
    score_functions : Dict[str, Callable]
        Mapping from primitive name to score function.
    min_score_gap : float
        Minimum score gap between probes (default 0.05).
    max_probes_per_round : int
        Maximum probes per design round (default 5).
    """
    def __init__(
        self,
        prompt_generator: Optional[SemanticPromptGenerator] = None,
        score_functions: Optional[Dict[str, Callable[[str], float]]] = None,
        min_score_gap: float = 0.05,
        max_probes_per_round: int = 5,
        alpha_uncertainty: float = 0.5,
        beta_boundary: float = 0.3,
        gamma_diversity: float = 0.2,
        prompt_embedding_store: Optional[Any] = None,
        embedding_scorer: Optional[EmbeddingSemanticScorer] = None,
    ) -> None:
        self.prompt_generator = prompt_generator or SemanticPromptGenerator()
        self.score_functions = score_functions or _SCORE_FUNCTIONS
        self.min_score_gap = min_score_gap
        self.max_probes_per_round = max_probes_per_round
        self.alpha_uncertainty = alpha_uncertainty
        self.beta_boundary = beta_boundary
        self.gamma_diversity = gamma_diversity
        self.prompt_embedding_store = prompt_embedding_store
        # Use the global scorer singleton for consistent scoring
        self.embedding_scorer = embedding_scorer or get_global_scorer()

    def design_intervention(
        self,
        base_prompt: str,
        primitive_name: str,
        estimator: BayesianBoundaryEstimator,
    ) -> SemanticIntervention:
        """Design a single intervention to reduce boundary uncertainty.

        Selects the most informative score level to probe,
        then generates a prompt at that level.

        Uses the embedding scorer (hybrid score) for actual_score —
        this guarantees consistency with what the victim evaluates.

        Parameters
        ----------
        base_prompt : str
            Base prompt to vary.
        primitive_name : str
            Semantic primitive name.
        estimator : BayesianBoundaryEstimator
            Current boundary estimate.

        Returns
        -------
        SemanticIntervention
            The designed intervention.
        """
        # Map primitive_name to centroid name for embedding scorer
        centroid_map = {
            "instruction_score": "instruction",
            "harmfulness_score": "harmful",
            "jailbreak_score": "jailbreak",
            "procedurality_score": "procedural",
        }
        centroid_name = centroid_map.get(primitive_name, "instruction")

        targets = estimator.generate_target_scores(n=7)

        best_target = targets[len(targets) // 2]
        best_info_gain = -1.0

        for target in targets:
            info_gain = self._estimate_info_gain(target, estimator)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_target = target

        prompt = base_prompt
        try:
            candidates = self.prompt_generator.generate_at_target(
                base_prompt, primitive_name, best_target, n_variants=3
            )
            if candidates:
                prompt = candidates[0]
        except Exception:
            logger.warning("Prompt generation failed; using base prompt")

        # Use embedding scorer (hybrid score) for consistency with victims
        hy_score = self.embedding_scorer.score(prompt, centroid_name)
        actual_score = hy_score.final

        return SemanticIntervention(
            prompt=prompt,
            primitive_name=primitive_name,
            target_score=best_target,
            actual_score=actual_score,
            expected_information_gain=best_info_gain,
            is_boundary_probe=True,
        )

    def design_boundary_probes(
        self,
        base_prompt: str,
        primitive_name: str,
        estimator: BayesianBoundaryEstimator,
    ) -> List[SemanticIntervention]:
        """Design multiple probes across the uncertain region.

        Spreads probes near the current boundary estimate.

        Parameters
        ----------
        base_prompt : str
            Base prompt to vary.
        primitive_name : str
            Semantic primitive name.
        estimator : BayesianBoundaryEstimator
            Current boundary estimate.

        Returns
        -------
        List[SemanticIntervention]
            Ordered by expected information gain.
        """
        score_fn = self.score_functions.get(primitive_name, _compute_instruction_score)
        targets = estimator.generate_target_scores(n=min(self.max_probes_per_round + 2, 9))

        probes: List[SemanticIntervention] = []
        seen_prompts: set = set()

        for target in targets:
            candidates = self.prompt_generator.generate_at_target(
                base_prompt, primitive_name, target, n_variants=1
            )
            prompt = candidates[0] if candidates else base_prompt
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
            actual_score = score_fn(prompt)
            info_gain = self._estimate_info_gain(target, estimator)
            probes.append(SemanticIntervention(
                prompt=prompt,
                primitive_name=primitive_name,
                target_score=target,
                actual_score=actual_score,
                expected_information_gain=info_gain,
                is_boundary_probe=True,
            ))
            if len(probes) >= self.max_probes_per_round:
                break

        probes.sort(key=lambda p: p.expected_information_gain, reverse=True)
        return probes

    def propose_utility_probes(
        self,
        base_prompt: str,
        primitive_name: str,
        estimator: BayesianBoundaryEstimator,
        n_probes: int = 5,
    ) -> List[SemanticIntervention]:
        """Propose probes maximising utility = α·uncertainty + β·boundary + γ·diversity.

        Parameters
        ----------
        base_prompt : str
        primitive_name : str
        estimator : BayesianBoundaryEstimator
        n_probes : int

        Returns
        -------
        List[SemanticIntervention]
        """
        score_fn = self.score_functions.get(primitive_name)
        if score_fn is None:
            return []
        est = estimator.estimate()
        targets = estimator.generate_target_scores(n=n_probes + 2)

        candidates: List[Tuple[float, SemanticIntervention]] = []
        for target in targets:
            try:
                prompts = self.prompt_generator.generate_at_target(
                    base_prompt, primitive_name, target, n_variants=2
                )
            except Exception:
                prompts = [base_prompt]

            for prompt in prompts:
                actual_score = score_fn(prompt)
                # Uncertainty gain
                unc_gain = self._estimate_info_gain(target, estimator)

                # Boundary proximity
                dist = abs(target - est.posterior_mean)
                boundary_prox = max(0.0, 1.0 - dist / max(est.posterior_std * 3, 0.1))

                # Embedding diversity
                div_score = 1.0
                if self.prompt_embedding_store is not None:
                    try:
                        div_score = 1.0 if self.prompt_embedding_store.is_diverse(prompt) else 0.0
                    except Exception:
                        pass

                utility = (
                    self.alpha_uncertainty * unc_gain
                    + self.beta_boundary * boundary_prox
                    + self.gamma_diversity * div_score
                )
                candidates.append((utility, SemanticIntervention(
                    prompt=prompt,
                    primitive_name=primitive_name,
                    target_score=target,
                    actual_score=actual_score,
                    expected_information_gain=round(float(utility), 4),
                    is_boundary_probe=True,
                )))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:n_probes]]

    def design_gradient_probes(
        self,
        base_prompt: str,
        primitive_name: str,
        estimator: BayesianBoundaryEstimator,
        n_probes: int = 5,
    ) -> List[SemanticIntervention]:
        """Design a gradient of probes from low to high score.

        Useful for exploring a new primitive where the boundary is unknown.
        Uses the embedding scorer for consistent scoring with victims.
        """
        centroid_map = {
            "instruction_score": "instruction",
            "harmfulness_score": "harmful",
            "jailbreak_score": "jailbreak",
            "procedurality_score": "procedural",
        }
        centroid_name = centroid_map.get(primitive_name, "instruction")

        target_scores = list(round(t, 4) for t in (
            0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9
        ))[:n_probes]

        probes: List[SemanticIntervention] = []
        for target in target_scores:
            candidates = self.prompt_generator.generate_at_target(
                base_prompt, primitive_name, target, n_variants=1
            )
            prompt = candidates[0] if candidates else base_prompt
            hy_score = self.embedding_scorer.score(prompt, centroid_name)
            actual_score = hy_score.final
            info_gain = self._estimate_info_gain(target, estimator)
            probes.append(SemanticIntervention(
                prompt=prompt,
                primitive_name=primitive_name,
                target_score=target,
                actual_score=actual_score,
                expected_information_gain=info_gain,
                is_boundary_probe=True,
            ))
        return probes

    @staticmethod
    def should_activate(hypotheses: List[str]) -> bool:
        """Determine if semantic intervention is needed.

        Always returns True when the semantic subsystem is active,
        regardless of hypothesis text. This replaces the old keyword-gated
        logic that silently prevented semantic exploration.
        """
        return True

    @staticmethod
    def _estimate_info_gain(
        target_score: float,
        estimator: BayesianBoundaryEstimator,
    ) -> float:
        """Estimate expected information gain from probing at target_score.

        Uses expected reduction in posterior variance.
        Points near the current boundary estimate have highest value.
        """
        est = estimator.estimate()
        if est.evidence_weight < 1.0:
            return 0.5
        mean = est.posterior_mean
        std = est.posterior_std
        dist = abs(target_score - mean)
        if std > 0.01:
            gain = math.exp(-0.5 * (dist / std) ** 2) / (std * math.sqrt(2 * math.pi))
            gain = max(0.01, min(1.0, gain * std * 3))
        else:
            gain = max(0.01, 1.0 - dist * 3)
        return round(float(gain), 4)
