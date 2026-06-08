"""Bayesian belief update over candidate programs (via Version Space).

Replaces the previous ``BayesianBeliefUpdater(states=[])`` degenerate
implementation.  Belief is now maintained over candidate programs in the
version space rather than over empty POMDP states.

The update follows:

    P(program | o, I) ∝ P(o | program, I) * P(program)

where P(o | program, I) = 1 if program.predict(prompt) == o else 0.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

from core.program import Program
from core.types import Outcome
from inference.pomdp import POMDPAction, POMDPObservation
from inference.version_space import VersionSpace

logger = logging.getLogger(__name__)


class BayesianBeliefUpdater:
    """Bayesian belief update over candidate programs via Version Space.

    Wraps ``VersionSpace`` and provides backward-compatible API.
    The version space is the single source of truth for belief.

    Parameters
    ----------
    version_space : VersionSpace, optional
        If not provided, creates an empty one.  The orchestrator should
        share a single version space across all agents.
    """

    def __init__(
        self,
        version_space: Optional[VersionSpace] = None,
    ) -> None:
        self._version_space = version_space or VersionSpace(max_candidates=50)

    @property
    def belief(self) -> VersionSpace:
        """Return the version space (belief over programs)."""
        return self._version_space

    @property
    def version_space(self) -> VersionSpace:
        return self._version_space

    def set_version_space(self, vs: VersionSpace) -> None:
        """Replace the version space (called after synthesis)."""
        self._version_space = vs
        logger.info("BeliefUpdater: version space updated (%d candidates)", vs.num_candidates)

    def update(
        self,
        action: POMDPAction,
        observation: POMDPObservation,
        outcome_fn: Optional[Callable[[str, str], int]] = None,
    ) -> VersionSpace:
        """Bayesian update after executing an intervention.

        Delegates to ``VersionSpace.update_belief`` when a prediction
        function is provided.
        """
        vs = self._version_space
        if vs.is_empty or outcome_fn is None:
            return vs

        def _predict(program: Program, prompt: str) -> int:
            for c in vs.candidates:
                if c.program_id in [getattr(program, "id", "")]:
                    return outcome_fn(c.program_id, prompt)
            return 0

        vs.update_belief(
            prompt=action.prompt,
            observed_outcome=Outcome(observation.outcome),
            predict_fn=_predict,
        )
        logger.info(
            "Belief update: entropy=%.3f candidates=%d info_gain=%.4f",
            vs.entropy(), vs.num_candidates,
            vs.info_gains[-1] if vs.info_gains else 0.0,
        )
        return vs

    def reset(self, uniform: bool = True) -> None:
        """Reset belief to uniform (keeps candidates)."""
        self._version_space.reset_belief(uniform=uniform)
        logger.debug("BeliefUpdater: reset (uniform=%s)", uniform)
