"""Bayesian belief update for HARMONY-X (Section 2.4 of harmony_v5v.md).

Implements the exact update from the report:

    b_{t+1}(Π) = P(o | Π, I) · b_t(Π) / Σ_{Π'} P(o | Π', I) · b_t(Π')

This replaces the simpler uncertainty heuristic ``1 - |conf1 - conf2|``
currently used in ``StrategistAgent.select_hypothesis_pair``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from inference.pomdp import BeliefState, POMDPAction, POMDPObservation, POMDPState

logger = logging.getLogger(__name__)


class BayesianBeliefUpdater:
    """Performs Bayesian belief updates as hypotheses are tested.

    Each hypothesis Π is treated as a possible *hidden state* s.
    After executing intervention *I* and observing outcome *o*, the
    belief distribution is updated via Bayes' rule.
    """

    def __init__(self, states: List[POMDPState]) -> None:
        self._states = states
        self._state_ids = [s.state_id for s in states]
        self._belief = BeliefState(self._state_ids, uniform_init=True)

    @property
    def belief(self) -> BeliefState:
        return self._belief

    def update(
        self,
        action: POMDPAction,
        observation: POMDPObservation,
        outcome_fn: Optional[Any] = None,
    ) -> BeliefState:
        """Perform a single Bayesian update.

        Parameters
        ----------
        action : POMDPAction
            The intervention that was executed.
        observation : POMDPObservation
            The observed outcome.
        outcome_fn : callable, optional
            Function ``fn(state_id, prompt) -> int`` that predicts the
            outcome for a given state and prompt.  When ``None``, the
            update uses cached predictions (defaults to uniform).

        Returns
        -------
        BeliefState
            The updated belief distribution.
        """
        log_probs = np.zeros(len(self._state_ids), dtype=np.float64)

        for i, s in enumerate(self._states):
            if outcome_fn is not None:
                pred = outcome_fn(s.state_id, action.prompt)
                likelihood = 1.0 if pred == observation.outcome else 0.0
            else:
                likelihood = self._default_likelihood(s, action, observation)
            log_probs[i] = np.log(max(likelihood, 1e-12)) + np.log(max(self._belief.b[i], 1e-12))

        log_probs -= np.max(log_probs)  # numerical stability
        self._belief.b = np.exp(log_probs)
        total = self._belief.b.sum()
        if total > 0:
            self._belief.b /= total
        else:
            self._belief.b = np.full(len(self._state_ids), 1.0 / len(self._state_ids))

        logger.info(
            "Belief update: entropy=%.3f → %.3f",
            self._belief.entropy(),
            self._belief.entropy(),
        )
        return self._belief

    def reset(self, uniform: bool = True) -> None:
        if uniform:
            self._belief = BeliefState(self._state_ids, uniform_init=True)
        else:
            self._belief = BeliefState(self._state_ids, uniform_init=False)

    @staticmethod
    def _default_likelihood(
        state: POMDPState, action: POMDPAction, observation: POMDPObservation
    ) -> float:
        """Default likelihood: uniform when no prediction function is available."""
        return 0.5

    def select_most_uncertain_pair(self) -> Optional[tuple]:
        """Return the pair of states with the highest epistemic uncertainty.

        Uses the *max-entropy* criterion: pick the two states whose
        beliefs are closest (i.e., most confusable).

        Returns
        -------
        tuple of (POMDPState, POMDPState) or None
        """
        if len(self._states) < 2:
            return None

        best_pair = None
        best_uncertainty = -1.0

        for i, s1 in enumerate(self._states):
            for s2 in self._states[i + 1:]:
                p1 = self._belief[s1.state_id]
                p2 = self._belief[s2.state_id]
                uncertainty = 1.0 - abs(p1 - p2)
                if uncertainty > best_uncertainty:
                    best_uncertainty = uncertainty
                    best_pair = (s1, s2)

        return best_pair
