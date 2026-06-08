"""Version Space — maintains top-K candidate programs with posterior belief.

Replaces the previous single ``best_program_id`` storage with a true
version space that tracks multiple competing candidate programs, their
posterior beliefs, and computes disagreement-driven intervention targets.

Key design principles:
  1. Each candidate program is a *hypothesis* about the LLM's safety policy.
  2. Belief is maintained as a posterior over candidates via Bayesian update.
  3. Disagreement between candidates drives intervention design.
  4. Convergence is measured by entropy over the candidate posterior.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.program import Program
from core.types import Outcome, ProgramID

logger = logging.getLogger(__name__)


@dataclass
class CandidateProgram:
    """A candidate program with its accuracy, complexity, and posterior.

    Attributes
    ----------
    program : Program
        The synthesized defense program.
    program_id : str
        Unique identifier for this candidate.
    accuracy : float
        Observed accuracy on training episodes.
    complexity : int
        Program complexity (node count).
    posterior : float
        Current posterior belief P(program | data).
    source : str
        Origin: "cvc5", "enumeration", "verification", "manual".
    episodes_matched : int
        Number of training episodes this program correctly predicts.
    total_episodes : int
        Total training episodes evaluated.
    """

    program: Program
    program_id: str = ""
    accuracy: float = 0.0
    complexity: int = 0
    posterior: float = 0.0
    source: str = "unknown"
    episodes_matched: int = 0
    total_episodes: int = 0

    def __post_init__(self) -> None:
        if not self.program_id:
            self.program_id = self.program.id or f"candidate_{uuid.uuid4().hex[:12]}"
        if self.complexity == 0:
            self.complexity = self.program.complexity()

    def predict(self, prompt: str, executor: Any) -> int:
        """Predict outcome for a prompt: 0 (ACCEPT) or 1 (REFUSE)."""
        try:
            return int(executor.execute(self.program, prompt))
        except Exception:
            return 0


class VersionSpace:
    """Maintains a distribution over candidate programs.

    The version space stores top-K candidate programs and maintains a
    posterior distribution P(program | observed data).  This enables:

    - **Entropy-based convergence**: when posterior concentrates on one
      candidate, the system has converged.
    - **Disagreement-driven intervention**: find prompts where candidates
      disagree, target those for maximum information gain.
    - **Principled stopping**: stop when posterior entropy < threshold.

    Parameters
    ----------
    max_candidates : int
        Maximum number of candidates to retain (default 50).
    uniform_init : bool
        Initialize belief uniformly (default True).
    """

    def __init__(
        self,
        max_candidates: int = 50,
        uniform_init: bool = True,
    ) -> None:
        self._candidates: List[CandidateProgram] = []
        self._max_candidates = max(2, int(max_candidates))
        self._uniform_init = uniform_init
        self._belief_dirty = True
        self._posterior: np.ndarray = np.array([], dtype=np.float64)
        self._entropy_history: List[float] = []
        self._info_gains: List[float] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def candidates(self) -> List[CandidateProgram]:
        return list(self._candidates)

    @property
    def num_candidates(self) -> int:
        return len(self._candidates)

    @property
    def posterior(self) -> np.ndarray:
        """Return posterior array (lazy-normalised)."""
        if self._belief_dirty:
            self._normalise()
        return self._posterior.copy()

    @property
    def program_ids(self) -> List[str]:
        return [c.program_id for c in self._candidates]

    @property
    def is_empty(self) -> bool:
        return len(self._candidates) == 0

    # ------------------------------------------------------------------
    # Candidate management
    # ------------------------------------------------------------------

    def add_candidate(
        self,
        program: Program,
        accuracy: float = 0.0,
        source: str = "unknown",
        episodes_matched: int = 0,
        total_episodes: int = 0,
    ) -> str:
        """Add or update a candidate program.

        If a candidate with the same program ID already exists, its
        accuracy/posterior is updated.  Otherwise a new candidate is added.
        Trims to max_candidates by lowest posterior.

        Returns
        -------
        str
            The program_id of the added/updated candidate.
        """
        program_id = program.id or f"candidate_{uuid.uuid4().hex[:12]}"

        existing = self._find(program_id)
        if existing is not None:
            existing.accuracy = accuracy
            existing.episodes_matched = episodes_matched
            existing.total_episodes = total_episodes
            self._belief_dirty = True
            return program_id

        candidate = CandidateProgram(
            program=program,
            program_id=program_id,
            accuracy=accuracy,
            complexity=program.complexity(),
            posterior=0.0,
            source=source,
            episodes_matched=episodes_matched,
            total_episodes=total_episodes,
        )
        self._candidates.append(candidate)
        self._belief_dirty = True
        self._prune()
        return program_id

    def remove_candidate(self, program_id: str) -> bool:
        """Remove a candidate by ID.  Returns True if found."""
        for i, c in enumerate(self._candidates):
            if c.program_id == program_id:
                self._candidates.pop(i)
                self._belief_dirty = True
                return True
        return False

    def get_candidate(self, program_id: str) -> Optional[CandidateProgram]:
        """Look up a candidate by ID."""
        return self._find(program_id)

    def get_program(self, program_id: str) -> Optional[Program]:
        c = self._find(program_id)
        return c.program if c is not None else None

    def _find(self, program_id: str) -> Optional[CandidateProgram]:
        for c in self._candidates:
            if c.program_id == program_id:
                return c
        return None

    def _prune(self) -> None:
        """Trim to max_candidates, keeping those with highest posterior."""
        if len(self._candidates) <= self._max_candidates:
            return
        self._normalise()
        sorted_idx = np.argsort(self._posterior)[::-1]
        keep = sorted_idx[:self._max_candidates]
        self._candidates = [self._candidates[i] for i in keep]
        self._posterior = self._posterior[keep]
        self._belief_dirty = True

    # ------------------------------------------------------------------
    # Bayesian belief update
    # ------------------------------------------------------------------

    def reset_belief(self, uniform: bool = True) -> None:
        """Reset posterior to uniform or zero."""
        n = len(self._candidates)
        if n == 0:
            self._posterior = np.array([], dtype=np.float64)
        elif uniform:
            self._posterior = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            self._posterior = np.zeros(n, dtype=np.float64)
        self._belief_dirty = False

    def update_belief(
        self,
        prompt: str,
        observed_outcome: Outcome,
        predict_fn: Callable[[Program, str], int],
    ) -> np.ndarray:
        """Bayesian update of posterior given observed outcome.

        P(program | o, I) ∝ P(o | program, I) * P(program)

        where P(o | program, I) = 1.0 if program.predict(prompt) == o,
        else 0.0 (deterministic prediction).

        Parameters
        ----------
        prompt : str
            The intervention prompt.
        observed_outcome : Outcome
            0 (ACCEPT) or 1 (REFUSE).
        predict_fn : callable
            Function ``fn(program, prompt) -> int`` that predicts outcome.

        Returns
        -------
        np.ndarray
            Updated posterior array.
        """
        n = len(self._candidates)
        if n == 0:
            return self._posterior

        self._normalise()
        entropy_before = self.entropy()
        log_posterior = np.log(np.clip(self._posterior, 1e-12, 1.0))

        for i, c in enumerate(self._candidates):
            pred = predict_fn(c.program, prompt)
            likelihood = 1.0 if pred == observed_outcome else 1e-12
            log_posterior[i] += np.log(likelihood)

        log_posterior -= np.max(log_posterior)
        self._posterior = np.exp(log_posterior)
        total = self._posterior.sum()
        if total > 0:
            self._posterior /= total
        else:
            self._posterior = np.full(n, 1.0 / n)

        entropy_after = self.entropy()
        info_gain = entropy_before - entropy_after
        self._info_gains.append(info_gain)
        self._belief_dirty = False
        return self._posterior

    @property
    def info_gains(self) -> List[float]:
        return list(self._info_gains)

    @property
    def total_info_gain(self) -> float:
        return sum(self._info_gains)

    def _normalise(self) -> None:
        """Ensure posterior sums to 1 and matches candidate count."""
        n = len(self._candidates)
        if n == 0:
            self._posterior = np.array([], dtype=np.float64)
            self._belief_dirty = False
            return
        if len(self._posterior) != n:
            self.reset_belief(uniform=self._uniform_init)
        total = self._posterior.sum()
        if total > 0:
            self._posterior = self._posterior / total
        self._belief_dirty = False

    # ------------------------------------------------------------------
    # Disagreement analysis
    # ------------------------------------------------------------------

    def get_disagreement_pairs(
        self,
        prompts: List[str],
        executor: Any,
        top_k: int = 5,
    ) -> List[Tuple[CandidateProgram, CandidateProgram, str, float]]:
        """Find prompt regions where candidate programs disagree.

        For each prompt, computes the prediction variance across all
        candidates.  Returns the top-K (prompt, pair) combinations with
        highest disagreement.

        Parameters
        ----------
        prompts : list of str
            Base prompts to evaluate.
        executor : ProgramExecutor
            Executor to run program predictions.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list of (h1, h2, prompt, disagreement)
            Where disagreement = |pred1 - pred2|.
        """
        if len(self._candidates) < 2:
            return []

        results: List[Tuple[CandidateProgram, CandidateProgram, str, float]] = []

        for prompt in prompts:
            predictions = {}
            for c in self._candidates:
                try:
                    predictions[c.program_id] = int(executor.execute(c.program, prompt))
                except Exception:
                    predictions[c.program_id] = 0

            for i, c1 in enumerate(self._candidates):
                for c2 in self._candidates[i + 1:]:
                    p1 = predictions.get(c1.program_id, 0)
                    p2 = predictions.get(c2.program_id, 0)
                    disagreement = abs(p1 - p2)
                    if disagreement > 0:
                        results.append((c1, c2, prompt, disagreement))

        results.sort(key=lambda x: -x[3])
        return results[:top_k * 2]

    def get_most_uncertain_pair(
        self,
        prompts: List[str],
        executor: Any,
    ) -> Optional[Tuple[CandidateProgram, CandidateProgram, str, float]]:
        """Return the single most uncertain pair + prompt combination."""
        pairs = self.get_disagreement_pairs(prompts, executor, top_k=1)
        return pairs[0] if pairs else None

    def get_max_disagreement_pair(
        self,
        executor: Any,
    ) -> Optional[Tuple[CandidateProgram, CandidateProgram, str, float]]:
        """Find the pair of candidates with highest posterior-weighted
        expected disagreement across all base prompts."""
        from prompt_loader import load_prompts
        try:
            prompts = load_prompts()
        except Exception:
            prompts = []
        return self.get_most_uncertain_pair(prompts, executor)

    def get_highest_entropy_prompt(
        self,
        prompts: List[str],
        executor: Any,
    ) -> Optional[Tuple[str, float]]:
        """Find the prompt with highest predictive entropy across candidates.

        Returns (prompt, entropy) or None.
        """
        if len(self._candidates) < 2 or not prompts:
            return None

        best_prompt = prompts[0]
        best_entropy = -1.0

        for prompt in prompts:
            preds = []
            for c in self._candidates:
                try:
                    preds.append(int(executor.execute(c.program, prompt)))
                except Exception:
                    preds.append(0)

            p_refuse = sum(preds) / len(preds)
            p_accept = 1.0 - p_refuse
            eps = 1e-12
            entropy = -(p_refuse * np.log(max(p_refuse, eps)) +
                        p_accept * np.log(max(p_accept, eps)))
            if entropy > best_entropy:
                best_entropy = entropy
                best_prompt = prompt

        return best_prompt, best_entropy

    # ------------------------------------------------------------------
    # Entropy and convergence
    # ------------------------------------------------------------------

    def entropy(self) -> float:
        """Posterior entropy over candidate programs.

        Returns 0.0 when < 2 candidates (degenerate).
        """
        n = len(self._candidates)
        if n < 2:
            return 0.0
        self._normalise()
        eps = 1e-12
        p = np.clip(self._posterior, eps, 1.0)
        return float(max(0.0, -np.sum(p * np.log(p))))

    def is_converged(self, threshold: float = 0.1, min_cycles: int = 3) -> bool:
        """Check if posterior entropy indicates convergence.

        Returns True when all recent entropy values are below threshold
        (requires at least min_cycles data points).
        """
        if len(self._entropy_history) < min_cycles:
            return False
        recent = self._entropy_history[-min_cycles:]
        return all(e < threshold for e in recent)

    def most_likely(self) -> Optional[CandidateProgram]:
        """Return the candidate with highest posterior probability."""
        if not self._candidates:
            return None
        self._normalise()
        return self._candidates[int(np.argmax(self._posterior))]

    def posterior_for(self, program_id: str) -> float:
        """Return posterior probability for a specific candidate."""
        for i, c in enumerate(self._candidates):
            if c.program_id == program_id:
                self._normalise()
                return float(self._posterior[i])
        return 0.0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_candidates": len(self._candidates),
            "max_candidates": self._max_candidates,
            "entropy": self.entropy(),
            "total_info_gain": self.total_info_gain,
            "num_updates": len(self._info_gains),
            "candidates": [
                {
                    "program_id": c.program_id,
                    "accuracy": c.accuracy,
                    "complexity": c.complexity,
                    "posterior": float(self._posterior[i]) if i < len(self._posterior) else 0.0,
                    "source": c.source,
                }
                for i, c in enumerate(self._candidates)
            ],
        }

    def record_entropy(self) -> float:
        """Record current entropy and return it."""
        e = self.entropy()
        self._entropy_history.append(e)
        return e

    @property
    def entropy_history(self) -> List[float]:
        return list(self._entropy_history)
