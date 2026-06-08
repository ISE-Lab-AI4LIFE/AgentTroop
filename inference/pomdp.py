"""POMDP formalism for HARMONY-X (Section 2.1 of harmony_v5v.md).

Models the interaction between HARMONY-X and the target LLM as a
Partially Observable Markov Decision Process:

    - Hidden state  s ∈ S:  internal safety configuration of the LLM
    - Action       a ∈ A:   prompt sent to the LLM
    - Observation  o ∈ O:   binary outcome (ACCEPT=0, REFUSE=1)
    - Transition   T(s'|s,a):  how the LLM's internal state evolves
    - Observation  Z(o|s,a):  probability of observing o given s,a
    - Belief       b(s):       distribution over hidden states
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class POMDPState:
    """A hidden state of the target LLM's safety configuration.

    Attributes
    ----------
    state_id : str
        Unique identifier for this state.
    label : str
        Human-readable label (e.g. "keyword_filter_on", "rot13_decoder_on").
    features : dict
        Arbitrary feature vector describing the configuration
        (e.g. {"keyword_filter": True, "rot13_decoder": False}).
    """

    state_id: str = ""
    label: str = ""
    features: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.state_id)


@dataclass
class POMDPAction:
    """A prompt sent to the victim LLM (wraps an Intervention-like structure).

    Attributes
    ----------
    action_id : str
    prompt : str
        The final prompt string sent to the LLM.
    metadata : dict
        Transforms applied, hypothesis pair, etc.
    """

    action_id: str = ""
    prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class POMDPObservation:
    """Observation returned by the victim LLM.

    Attributes
    ----------
    outcome : int
        0 = ACCEPT, 1 = REFUSE.
    raw_response : str
        Full response text from the LLM.
    latency : float
        Response time in seconds.
    """

    outcome: int = 0
    raw_response: str = ""
    latency: float = 0.0


# ---------------------------------------------------------------------------
# Transition function T(s' | s, a)
# ---------------------------------------------------------------------------


class TransitionFunction:
    """Transition function T(s' | s, a) — probability of moving to state
    *s_prime* from state *s* after executing action *a*.

    For safety reverse engineering, the LLM's internal state is typically
    *static* across interventions (the model weights don't change between
    API calls).  Therefore the default implementation is the identity:
    T(s'=s | s, a) = 1.0 for all a.

    Subclass and override ``__call__`` for dynamic environments where the
    model's safety configuration can change (e.g. A/B testing, router-based
    moderation).
    """

    def __init__(self, state_ids: List[str]) -> None:
        self._state_ids = state_ids
        self._num_states = len(state_ids)

    def __call__(self, s_prime: POMDPState, s: POMDPState, a: POMDPAction) -> float:
        """Return T(s' | s, a)."""
        return 1.0 if s_prime.state_id == s.state_id else 0.0

    def matrix(self, a: POMDPAction) -> np.ndarray:
        """Return the full |S| × |S| transition matrix for action *a*."""
        return np.eye(self._num_states, dtype=np.float64)

    @property
    def num_states(self) -> int:
        return self._num_states


# ---------------------------------------------------------------------------
# Observation function Z(o | s, a)
# ---------------------------------------------------------------------------


class ObservationFunction:
    """Observation function Z(o | s, a) — probability of observing *o*
    given that the system is in state *s* and action *a* was taken.

    This is the core *prediction* of a hypothesis or program: given a
    hypothesized state (which encodes the safety program), what outcome
    would we expect for a given prompt?

    Z(REFUSE | s, a) = Π(a)  where Π is the program associated with s
    Z(ACCEPT  | s, a) = 1 - Π(a)
    """

    def __init__(self, state_outcome_fn: Optional[Callable[[str, str], int]] = None) -> None:
        """If *state_outcome_fn* is None, predictions must be set manually
        via :meth:`set_outcome`."""
        self._fn = state_outcome_fn
        self._cache: Dict[tuple, int] = {}

    def set_outcome(self, state_id: str, action_id: str, outcome: int) -> None:
        """Manually set Z(o | s, a) = 1.0 for the given outcome."""
        self._cache[(state_id, action_id, outcome)] = 1
        self._cache[(state_id, action_id, 1 - outcome)] = 0

    def __call__(self, o: POMDPObservation, s: POMDPState, a: POMDPAction) -> float:
        key = (s.state_id, a.action_id, o.outcome)
        if key in self._cache:
            return float(self._cache[key])
        if self._fn is not None:
            pred = self._fn(s.state_id, a.prompt)
            return 1.0 if pred == o.outcome else 0.0
        return 0.5  # uniform when nothing is known

    def predict(self, s: POMDPState, a: POMDPAction) -> int:
        """Return the most likely outcome (0 or 1) for state *s* and action *a*."""
        key_refuse = (s.state_id, a.action_id, 1)
        if key_refuse in self._cache:
            return self._cache[key_refuse]
        if self._fn is not None:
            return self._fn(s.state_id, a.prompt)
        return 0  # default ACCEPT


# ---------------------------------------------------------------------------
# Belief state b(s)
# ---------------------------------------------------------------------------


class BeliefState:
    """A categorical distribution over a discrete set of hidden states.

    b : np.ndarray of shape (|S|,)
        The belief vector, where b[i] = P(state_i).
    """

    def __init__(self, state_ids: List[str], uniform_init: bool = True) -> None:
        self._state_ids = state_ids
        self._num_states = len(state_ids)
        if self._num_states == 0:
            self.b = np.array([], dtype=np.float64)
        elif uniform_init:
            self.b = np.full(self._num_states, 1.0 / self._num_states, dtype=np.float64)
        else:
            self.b = np.zeros(self._num_states, dtype=np.float64)

    def __getitem__(self, state_id: str) -> float:
        try:
            idx = self._state_ids.index(state_id)
        except ValueError:
            return 0.0
        return float(self.b[idx])

    def __setitem__(self, state_id: str, prob: float) -> None:
        idx = self._state_ids.index(state_id)
        self.b[idx] = prob

    def most_likely(self) -> Optional[str]:
        if self._num_states == 0:
            return None
        return self._state_ids[int(np.argmax(self.b))]

    def entropy(self) -> float:
        if self._num_states == 0:
            return 0.0
        eps = 1e-12
        p = np.clip(self.b, eps, 1.0)
        return float(max(0.0, -np.sum(p * np.log(p))))

    def copy(self) -> BeliefState:
        bs = BeliefState(self._state_ids, uniform_init=False)
        bs.b = self.b.copy()
        return bs

    def to_dict(self) -> Dict[str, float]:
        return {sid: float(self.b[i]) for i, sid in enumerate(self._state_ids)}

    def __repr__(self) -> str:
        top = self.most_likely()
        return f"BeliefState(entropy={self.entropy():.3f}, most_likely={top})"
