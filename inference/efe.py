"""Expected Free Energy (EFE) for active inference (Section 2.4 of harmony_v5v.md).

Implements:

    G(I) = E_{o ~ P(o|I)}[ D_KL[ b_{t+1} || b_t ] ] + E_{o ~ P(o|I)}[ ln P_pref(o) ]

where:
  - The *epistemic* term values information gain (KL divergence from prior
    to posterior belief).
  - The *pragmatic* term values preferred outcomes (set to constant in
    HARMONY-X since the goal is pure information seeking).

This replaces the Δ heuristic currently used in ``design_intervention``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from inference.pomdp import BeliefState, POMDPAction, POMDPObservation, POMDPState
from inference.belief_updater import BayesianBeliefUpdater

logger = logging.getLogger(__name__)


class ExpectedFreeEnergy:
    """Computes G(I) for candidate interventions.

    To integrate with the existing StrategistAgent, the caller provides a
    *prediction function* ``predict(h_id, prompt) -> int`` that returns
    the expected outcome (0 or 1) for a given hypothesis and prompt.
    """

    def __init__(
        self,
        updater: BayesianBeliefUpdater,
        pragmatic_weight: float = 0.0,
    ) -> None:
        self._updater = updater
        self._pragmatic_weight = pragmatic_weight

    def compute(
        self,
        action: POMDPAction,
        predict_fn: Any,
    ) -> float:
        """Compute G(I) for a candidate intervention.

        Parameters
        ----------
        action : POMDPAction
            The candidate intervention to evaluate.
        predict_fn : callable
            ``fn(state_id, prompt) -> int`` that predicts the outcome
            for each state and prompt.

        Returns
        -------
        float
            The Expected Free Energy (lower is more informative).
        """
        belief = self._updater.belief
        state_ids = self._updater._state_ids
        num_states = len(state_ids)
        if num_states == 0:
            return 0.0

        # Estimate P(o=0|I) and P(o=1|I) under current belief
        prob_accept = 0.0
        for i, sid in enumerate(state_ids):
            pred = predict_fn(sid, action.prompt)
            prob_accept += belief[sid] * (1.0 - pred)
        prob_refuse = 1.0 - prob_accept

        # Epistemic value: expected KL divergence
        epistemic = 0.0
        for outcome_val, prob_o in [(0, prob_accept), (1, prob_refuse)]:
            if prob_o <= 1e-12:
                continue
            obs = POMDPObservation(outcome=outcome_val)
            hypo_action = POMDPAction(
                action_id=action.action_id,
                prompt=action.prompt,
            )
            posterior = self._updater.update(hypo_action, obs, predict_fn)
            kl = self._kl_divergence(posterior.b, belief.b)
            epistemic += prob_o * kl
            # Restore belief for the next outcome
            self._updater._belief = belief.copy()

        # Pragmatic value (constant in HARMONY-X)
        pragmatic = 0.0
        if self._pragmatic_weight > 0:
            for outcome_val, prob_o in [(0, prob_accept), (1, prob_refuse)]:
                if prob_o <= 1e-12:
                    continue
                pref = 0.0  # no outcome preference
                pragmatic += prob_o * pref

        efe = -epistemic + self._pragmatic_weight * pragmatic
        logger.debug(
            "EFE: G=%.4f (epistemic=%.4f, pragmatic=%.4f, P(accept)=%.3f)",
            efe, epistemic, pragmatic, prob_accept,
        )
        return efe

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """D_KL[p || q] with numerical stability."""
        eps = 1e-12
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        return float(np.sum(p * np.log(p / q)))
