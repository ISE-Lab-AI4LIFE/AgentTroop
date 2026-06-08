"""Active inference engine for HARMONY-X.

Implements the POMDP formalism (Section 2.1), Bayesian belief update
(Section 2.4), and Expected Free Energy (Section 2.4) described in the
harmony_v5v.md report.
"""

from inference.pomdp import (
    POMDPState,
    POMDPAction,
    POMDPObservation,
    TransitionFunction,
    ObservationFunction,
    BeliefState,
)
from inference.belief_updater import BayesianBeliefUpdater
from inference.efe import ExpectedFreeEnergy

__all__ = [
    "POMDPState",
    "POMDPAction",
    "POMDPObservation",
    "TransitionFunction",
    "ObservationFunction",
    "BeliefState",
    "BayesianBeliefUpdater",
    "ExpectedFreeEnergy",
]
