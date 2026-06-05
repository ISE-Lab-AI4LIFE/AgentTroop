from typing import List, Tuple

import numpy as np


class InformationGainEvaluator:
    """Measures how much information each intervention provides.
    
    Uses entropy reduction as the core metric, following the
    Expected Free Energy formulation from harmony_v5v.md.
    """

    @staticmethod
    def binary_entropy(p: float) -> float:
        """Compute binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))

    def compute_entropy_reduction(
        self, prior_belief: float, posterior_belief: float
    ) -> float:
        """Compute the reduction in binary entropy after an intervention.
        
        prior_belief and posterior_belief represent the probability
        that a given hypothesis is correct.
        """
        prior_entropy = self.binary_entropy(prior_belief)
        posterior_entropy = self.binary_entropy(posterior_belief)
        return prior_entropy - posterior_entropy

    def evaluate_intervention_sequence(
        self,
        belief_updates: List[Tuple[float, float]],
    ) -> List[float]:
        """Compute information gain for a sequence of belief updates.
        
        Each tuple is (prior_belief, posterior_belief) for one intervention.
        Returns a list of information gain values (one per intervention).
        """
        gains: List[float] = []
        for prior, posterior in belief_updates:
            gain = self.compute_entropy_reduction(prior, posterior)
            gains.append(gain)
        return gains

    def cumulative_information_gain(
        self, belief_updates: List[Tuple[float, float]]
    ) -> float:
        """Total information gained across a sequence of interventions."""
        return sum(self.evaluate_intervention_sequence(belief_updates))
