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

from inference.pomdp import POMDPAction, POMDPObservation
from inference.version_space import VersionSpace

logger = logging.getLogger(__name__)


class ExpectedFreeEnergy:
    """Computes G(I) for candidate interventions.

    To integrate with the existing StrategistAgent, the caller provides a
    *prediction function* ``predict(h_id, prompt) -> int`` that returns
    the expected outcome (0 or 1) for a given hypothesis and prompt.
    """

    def __init__(
        self,
        version_space: VersionSpace,
        pragmatic_weight: float = 0.0,
    ) -> None:
        self._version_space = version_space
        self._pragmatic_weight = pragmatic_weight

    def compute(
        self,
        action: POMDPAction,
        predict_fn: Any,
    ) -> float:
        """Compute G(I) for a candidate intervention.

        EFE measures the expected information gain (KL divergence from
        prior to posterior belief) under each possible outcome.

        Parameters
        ----------
        action : POMDPAction
            The candidate intervention to evaluate.
        predict_fn : callable
            ``fn(program_id, prompt) -> int`` that predicts the outcome
            for each candidate program and prompt.

        Returns
        -------
        float
            The Expected Free Energy (lower = more informative).
        """
        vs = self._version_space
        n = vs.num_candidates
        if n < 2:
            return 0.0

        candidates = vs.candidates
        posterior = vs.posterior

        # Estimate P(o=0|I) and P(o=1|I) under current posterior
        prob_accept = 0.0
        for i, c in enumerate(candidates):
            pred = predict_fn(c.program_id, action.prompt)
            prob_accept += posterior[i] * (1.0 - pred)
        prob_refuse = 1.0 - prob_accept

        # Epistemic value: expected KL divergence
        epistemic = 0.0
        prior_b = posterior.copy()

        for outcome_val, prob_o in [(0, prob_accept), (1, prob_refuse)]:
            if prob_o <= 1e-12:
                continue

            obs = POMDPObservation(outcome=outcome_val)
            hypo_action = POMDPAction(
                action_id=action.action_id,
                prompt=action.prompt,
            )

            def _predict(program: Any, prompt: str) -> int:
                for c2 in candidates:
                    pid = getattr(program, "id", "") or getattr(program, "program_id", "")
                    if c2.program_id == pid:
                        return predict_fn(c2.program_id, prompt)
                return 0

            vs.update_belief(
                prompt=action.prompt,
                observed_outcome=outcome_val,
                predict_fn=_predict,
            )
            posterior_b = vs.posterior
            kl = self._kl_divergence(posterior_b, prior_b)
            epistemic += prob_o * kl
            # Restore posterior
            vs._posterior = prior_b.copy()
            vs._belief_dirty = False
            # Remove the info gain we just added
            if vs._info_gains:
                vs._info_gains.pop()

        # Pragmatic value (constant in HARMONY-X — no preference)
        pragmatic = 0.0
        if self._pragmatic_weight > 0:
            pragmatic = prob_accept * 0.0 + prob_refuse * 0.0

        efe = -epistemic + self._pragmatic_weight * pragmatic
        logger.debug(
            "EFE: G=%.4f (epistemic=%.4f, pragmatic=%.4f, P(accept)=%.3f, candidates=%d)",
            efe, epistemic, pragmatic, prob_accept, n,
        )
        return efe

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """D_KL[p || q] with numerical stability."""
        eps = 1e-12
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        return float(np.sum(p * np.log(p / q)))
