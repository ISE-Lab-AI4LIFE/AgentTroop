"""Version Space — maintains top-K candidate programs with posterior belief.

Replaces the previous single ``best_program_id`` storage with a true
version space that tracks multiple competing candidate programs, their
posterior beliefs, and computes disagreement-driven intervention targets.

Key design principles:
  1. Each candidate program is a *hypothesis* about the LLM's safety policy.
  2. Belief is maintained as a posterior over candidates via Bayesian update.
  3. Disagreement between candidates drives intervention design.
  4. Convergence is measured by entropy over the candidate posterior.

Fixes applied (forensic audit 2026-06):
  - **P1/P8**: Soft likelihood with configurable noise_level; single misprediction
    no longer collapses posterior to zero.
  - **P1**: Incremental candidate addition no longer calls reset_belief();
    new candidates start at 1e-6 posterior, existing posteriors preserved.
  - **P4**: Family tracking for diversity analysis.
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

# Default noise level for soft likelihood:
# P(o | program, I) = (1 - noise) if pred == o else noise
# This prevents a single non-matching observation from collapsing posterior.
_DEFAULT_NOISE_LEVEL = 0.1

# Default posterior for newly added candidates (near-zero, not exactly zero)
_NEW_CANDIDATE_INITIAL_POSTERIOR = 0.01


def _classify_program(program: Program) -> str:
    """Classify a program into keyword/structural/jailbreak/semantic/discourse/composite/transform family."""
    from core.program import (
        PredicateNode, ThresholdNode, AndNode, OrNode, NotNode, ApplyTransformNode,
    )
    root = program.root
    node = root.condition if hasattr(root, "condition") else root
    if isinstance(node, ApplyTransformNode):
        return "transform"
    if isinstance(node, (AndNode, OrNode, NotNode)):
        return "composite"
    if isinstance(node, ThresholdNode):
        ret = "classifier"
        classifier = getattr(node, "classifier", None)
        if classifier is not None:
            cname = classifier.name if hasattr(classifier, "name") else ""
            if cname in {"instruction_score", "semantic_score"}:
                ret = "semantic_score"
        return ret
    if isinstance(node, PredicateNode):
        name = node.primitive.name if hasattr(node.primitive, "name") else ""
        keyword_preds = {"contains_word", "contains_any_word", "contains_all_words",
                         "starts_with", "ends_with", "matches_regex"}
        structural_preds = {"has_number", "is_empty", "has_emoji", "contains_url",
                            "is_repetitive", "char_count", "length_gt", "length_lt",
                            "contains_rot13", "contains_base64", "contains_hex",
                            "contains_code_block", "contains_delimiter"}
        jailbreak_preds = {"matches_jailbreak_pattern", "contains_system_override",
                           "contains_encoding_wrapper"}
        semantic_preds = {"starts_with_roleplay", "starts_with_imperative",
                          "is_grammatical_question", "sentiment", "intent",
                          "instruction_score"}
        discourse_preds = {"is_instruction_request"}
        if name in keyword_preds:
            return "keyword"
        if name in structural_preds:
            return "structural"
        if name in jailbreak_preds:
            return "jailbreak"
        if name in semantic_preds:
            return "semantic"
        if name in discourse_preds:
            return "discourse"
        return "unknown"
    return "unknown"


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
        Origin: "evolutionary", "neural", "fitness_guided", "enumeration", "verification", "manual".
    episodes_matched : int
        Number of training episodes this program correctly predicts.
    total_episodes : int
        Total training episodes evaluated.
    family : str
        Predicate family for diversity analysis.
    """

    program: Program
    program_id: str = ""
    accuracy: float = 0.0
    complexity: int = 0
    posterior: float = 0.0
    source: str = "unknown"
    episodes_matched: int = 0
    total_episodes: int = 0
    family: str = ""
    predicate_type: str = ""
    holdout_accuracy: Optional[float] = None
    train_accuracy: Optional[float] = None
    generalization_gap: Optional[float] = None
    generation_depth: int = 0

    def __post_init__(self) -> None:
        if not self.program_id:
            self.program_id = self.program.id or f"candidate_{uuid.uuid4().hex[:12]}"
        if self.complexity == 0:
            self.complexity = self.program.complexity()
        if not self.family:
            self.family = _classify_program(self.program)
        if not self.predicate_type:
            self.predicate_type = self.family

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
      disagree, target for maximum information gain.
    - **Principled stopping**: stop when posterior entropy < threshold.

    Parameters
    ----------
    max_candidates : int
        Maximum number of candidates to retain (default 50).
    uniform_init : bool
        Initialize belief uniformly (default True).
    noise_level : float
        Soft likelihood noise tolerance (default 0.1).
    complexity_prior_lambda : float
        Occam factor coefficient: posterior ∝ likelihood × exp(-λ · complexity).
        Set to 0 to disable (preserves legacy behaviour).  Default 0.01.
    """

    def __init__(
        self,
        max_candidates: int = 50,
        uniform_init: bool = True,
        noise_level: float = _DEFAULT_NOISE_LEVEL,
        complexity_prior_lambda: float = 0.01,
    ) -> None:
        self._candidates: List[CandidateProgram] = []
        self._max_candidates = max(2, int(max_candidates))
        self._uniform_init = uniform_init
        self._noise_level = max(0.0, min(0.49, float(noise_level)))
        self._complexity_prior_lambda = max(0.0, float(complexity_prior_lambda))
        self._belief_dirty = True
        self._posterior: np.ndarray = np.array([], dtype=np.float64)
        self._entropy_history: List[float] = []
        self._info_gains: List[float] = []
        self._update_count: int = 0
        self._prune_count: int = 0
        self._synthesis_count: int = 0
        self._holdout_accuracy_history: List[float] = []
        # Fix 4: posterior history for diagnostics
        self._posterior_history: List[np.ndarray] = []
        self._topk_posterior_traces: Dict[str, List[float]] = {}
        self._survival_by_source: Dict[str, int] = {}
        self._survival_by_predicate_type: Dict[str, int] = {}
        self._total_by_source: Dict[str, int] = {}
        self._total_by_predicate_type: Dict[str, int] = {}
        self._posterior_floor: float = 1e-5
        self._diversity_preservation = True

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

    @property
    def noise_level(self) -> float:
        return self._noise_level

    @noise_level.setter
    def noise_level(self, value: float) -> None:
        self._noise_level = max(0.0, min(0.49, float(value)))

    @property
    def diversity_preservation(self) -> bool:
        return self._diversity_preservation

    @diversity_preservation.setter
    def diversity_preservation(self, value: bool) -> None:
        self._diversity_preservation = bool(value)

    # ------------------------------------------------------------------
    # Family diversity analysis
    # ------------------------------------------------------------------

    def family_counts(self) -> Dict[str, int]:
        """Return count of candidates per predicate family."""
        counts: Dict[str, int] = {}
        for c in self._candidates:
            fam = getattr(c, "family", "") or _classify_program(c.program)
            counts[fam] = counts.get(fam, 0) + 1
        return counts

    def family_posterior_mass(self) -> Dict[str, float]:
        """Return total posterior mass per family."""
        if self._belief_dirty:
            self._normalise()
        masses: Dict[str, float] = {}
        for i, c in enumerate(self._candidates):
            fam = getattr(c, "family", "") or _classify_program(c.program)
            p = float(self._posterior[i]) if i < len(self._posterior) else 0.0
            masses[fam] = masses.get(fam, 0.0) + p
        return masses

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
        initial_posterior: Optional[float] = None,
    ) -> str:
        """Add or update a candidate program.

        If a candidate with the same program ID already exists, its
        accuracy/posterior is updated.  Otherwise a new candidate is added.
        Trims to max_candidates by lowest posterior.

        **Fix P1**: New candidates are added with ``initial_posterior``
        (default 1e-6) and the posterior array is **extended** rather than
        reset, preserving all accumulated belief from previous updates.

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

        # Track lifetime totals for survival rate calculation
        src = source or "unknown"
        self._total_by_source[src] = self._total_by_source.get(src, 0) + 1
        ptype = candidate.family or "unknown"
        self._total_by_predicate_type[ptype] = self._total_by_predicate_type.get(ptype, 0) + 1

        # FIX P1: Extend posterior array instead of resetting
        # Use accuracy-derived initial posterior when no explicit value given
        init_p = initial_posterior if initial_posterior is not None else self._initial_posterior(accuracy, candidate.complexity)
        if len(self._posterior) == len(self._candidates) - 1:
            # Normal case: one new candidate → append
            self._posterior = np.append(self._posterior, init_p)
        else:
            # Mismatch: rebuild posterior from scratch
            self._posterior = np.full(len(self._candidates), init_p)
            # Attempt to preserve existing candidate posteriors
            for i, c in enumerate(self._candidates[:-1]):
                existing_idx = self._find_index(c.program_id)
                if existing_idx is not None and existing_idx < len(self._posterior) - 1:
                    try:
                        self._posterior[i] = max(init_p, self._posterior[existing_idx])
                    except (IndexError, ValueError):
                        pass

        self._belief_dirty = True
        self._prune()
        return program_id

    def _find(self, program_id: str) -> Optional[CandidateProgram]:
        for c in self._candidates:
            if c.program_id == program_id:
                return c
        return None

    def _find_index(self, program_id: str) -> Optional[int]:
        for i, c in enumerate(self._candidates):
            if c.program_id == program_id:
                return i
        return None

    def _prune(self) -> None:
        """Trim to max_candidates, keeping highest posterior candidates."""
        if len(self._candidates) <= self._max_candidates:
            return
        self._normalise()
        sorted_idx = np.argsort(self._posterior)[::-1]
        keep = list(sorted_idx[:self._max_candidates])
        keep_set = set(keep)
        if self._diversity_preservation:
            families_represented = set()
            for i in keep:
                c = self._candidates[i]
                families_represented.add(c.family or _classify_program(c.program))
            all_families = set()
            for c in self._candidates:
                all_families.add(c.family or _classify_program(c.program))
            for fam in all_families:
                if fam not in families_represented:
                    best_idx = None
                    best_p = -1.0
                    for i, c in enumerate(self._candidates):
                        cfam = c.family or _classify_program(c.program)
                        if cfam == fam and float(self._posterior[i]) > best_p:
                            best_p = float(self._posterior[i])
                            best_idx = i
                    if best_idx is not None and best_idx not in keep_set:
                        lowest_idx = min(keep, key=lambda i: float(self._posterior[i]))
                        keep.remove(lowest_idx)
                        keep_set.remove(lowest_idx)
                        keep.append(best_idx)
                        keep_set.add(best_idx)
        removed = [self._candidates[i] for i in range(len(self._candidates)) if i not in keep_set]
        keep = sorted(keep)
        self._candidates = [self._candidates[i] for i in keep]
        self._posterior = self._posterior[keep]
        self._prune_count += 1
        self._belief_dirty = True
        for c in self._candidates:
            src = c.source or "unknown"
            self._survival_by_source[src] = self._survival_by_source.get(src, 0) + 1
            ptype = c.predicate_type or c.family or "unknown"
            self._survival_by_predicate_type[ptype] = self._survival_by_predicate_type.get(ptype, 0) + 1

    def keep_top_k(self, k: int) -> int:
        """Keep exactly k candidates with highest posterior.

        Ensures at least 1 candidate per family is retained if available.
        Returns the number of removed candidates.
        """
        if len(self._candidates) <= k:
            return 0
        self._normalise()
        n_before = len(self._candidates)
        sorted_idx = np.argsort(self._posterior)[::-1]
        keep = list(sorted_idx[:k])
        keep_set = set(keep)
        if self._diversity_preservation:
            families_represented = set()
            for i in keep:
                c = self._candidates[i]
                families_represented.add(c.family or _classify_program(c.program))
            all_families = set()
            for c in self._candidates:
                all_families.add(c.family or _classify_program(c.program))
            for fam in all_families:
                if fam not in families_represented:
                    best_idx = None
                    best_p = -1.0
                    for i, c in enumerate(self._candidates):
                        cfam = c.family or _classify_program(c.program)
                        if cfam == fam and float(self._posterior[i]) > best_p:
                            best_p = float(self._posterior[i])
                            best_idx = i
                    if best_idx is not None and best_idx not in keep_set:
                        lowest_idx = min(keep, key=lambda i: float(self._posterior[i]))
                        keep.remove(lowest_idx)
                        keep_set.remove(lowest_idx)
                        keep.append(best_idx)
                        keep_set.add(best_idx)
        keep = sorted(keep)
        n_removed = n_before - len(keep)
        self._candidates = [self._candidates[i] for i in keep]
        self._posterior = self._posterior[keep]
        self._prune_count += 1
        self._belief_dirty = True
        for c in self._candidates:
            src = c.source or "unknown"
            self._survival_by_source[src] = self._survival_by_source.get(src, 0) + 1
            ptype = c.predicate_type or c.family or "unknown"
            self._survival_by_predicate_type[ptype] = self._survival_by_predicate_type.get(ptype, 0) + 1
        return n_removed

    def absorb_candidates(
        self,
        new_programs: List[Tuple[Program, float, str, int, int]],
        alpha: float = 0.85,
        new_candidate_weight: float = 0.1,
        balance_types: bool = False,
    ) -> int:
        """Batch-add candidates preserving posterior mass of existing programs.

        Existing candidates (top-K) retain ``alpha`` fraction of the total
        posterior mass.  New candidates start with very low posterior
        (``new_candidate_weight * initial_posterior``), preventing entropy
        spikes when many candidates are added at once.

        After adding, the version space is trimmed back to ``max_candidates``,
        keeping only the highest-posterior candidates.

        When *balance_types* is True (typically only on first seed),
        reweights posterior so each predicate family has equal total
        mass, preventing count-heavy types from dominating early.
        """
        existing_ids = set(self.program_ids)
        added = 0

        if len(self._candidates) > 0:
            self._normalise()
            self._posterior *= alpha

        for prog, acc, source, matched, total in new_programs:
            pid = getattr(prog, "id", "") or ""
            if pid not in existing_ids:
                candidate = CandidateProgram(
                    program=prog,
                    accuracy=acc,
                    complexity=prog.complexity(),
                    posterior=0.0,
                    source=source,
                    episodes_matched=matched,
                    total_episodes=total,
                )
                self._candidates.append(candidate)

                src = source or "unknown"
                self._total_by_source[src] = self._total_by_source.get(src, 0) + 1
                ptype = candidate.family or "unknown"
                self._total_by_predicate_type[ptype] = self._total_by_predicate_type.get(ptype, 0) + 1

                init_p = self._initial_posterior(acc, candidate.complexity)
                weighted_init_p = new_candidate_weight * init_p

                if len(self._posterior) == len(self._candidates) - 1:
                    self._posterior = np.append(self._posterior, weighted_init_p)
                else:
                    self._posterior = np.full(len(self._candidates), weighted_init_p)
                    for i, c in enumerate(self._candidates[:-1]):
                        existing_idx = self._find_index(c.program_id)
                        if existing_idx is not None and existing_idx < len(self._posterior) - 1:
                            try:
                                self._posterior[i] = max(weighted_init_p, self._posterior[existing_idx])
                            except (IndexError, ValueError):
                                pass

                added += 1
                existing_ids.add(pid)

        if added > 0:
            self._belief_dirty = True
            self._normalise()
            self._prune()

        if balance_types and added > 0 and len(self._candidates) > 0:
            self._normalise()
            type_masses: Dict[str, float] = {}
            for i, c in enumerate(self._candidates):
                fam = getattr(c, "family", "") or _classify_program(c.program)
                type_masses[fam] = type_masses.get(fam, 0.0) + float(self._posterior[i])
            n_types = len(type_masses)
            if n_types > 1:
                target_per_type = 1.0 / n_types
                for i, c in enumerate(self._candidates):
                    fam = getattr(c, "family", "") or _classify_program(c.program)
                    current = type_masses.get(fam, 1e-10)
                    if current > 0:
                        self._posterior[i] *= target_per_type / current
                self._normalise()
                logger.info(
                    "Type-balanced posterior: %s (target=%.3f per type)",
                    {k: round(v, 4) for k, v in self.posterior_by_predicate_type().items()},
                    target_per_type,
                )
        return added

    @staticmethod
    def _initial_posterior(accuracy: float, complexity: int = 0) -> float:
        """All programs start with the same prior — complexity is NOT used to
        advantage simple programs.  Only evidence (accuracy) matters."""
        effective = max(accuracy, 0.0) if accuracy > 0 else 0.0
        return max(0.005, effective * 0.3)

    def holdout_adjusted_score(self, candidate: CandidateProgram) -> float:
        """Combined score that rewards high holdout accuracy.
        Complexity is NOT penalised here — only accuracy matters."""
        if candidate.holdout_accuracy is not None and candidate.holdout_accuracy > 0.0:
            score = candidate.holdout_accuracy
            if candidate.train_accuracy is not None:
                gap_penalty = max(0.0, abs(candidate.train_accuracy - candidate.holdout_accuracy) - 0.05)
                score -= gap_penalty * 0.5
            return max(0.01, score)
        return max(0.0, candidate.accuracy)

    def reweight_by_holdout(self) -> None:
        """Reweight posterior using holdout-adjusted scores so candidates
        with verified holdout accuracy dominate over unsubstantiated seeds.
        Scores are squared to amplify differentiation among candidates
        that make identical training predictions but differ on holdout."""
        if not self._candidates:
            return
        self._normalise()
        has_holdout = any(c.holdout_accuracy is not None and c.holdout_accuracy > 0.0
                          for c in self._candidates)
        if not has_holdout:
            return
        for i, c in enumerate(self._candidates):
            adj = self.holdout_adjusted_score(c)
            self._posterior[i] *= max(0.01, adj) ** 2
        self._posterior = np.maximum(self._posterior, self._posterior_floor)
        self._normalise()
        self._belief_dirty = False

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
        predict_fn: Callable[[Any, str], int],
        noise_level: Optional[float] = None,
    ) -> np.ndarray:
        """Bayesian update of posterior given observed outcome.

        **Fix P8**: Uses soft likelihood ``P(o | program, I)`` with
        configurable ``noise_level`` instead of deterministic 1.0/1e-12.

        **Fix Occam**: Applies a complexity-aware prior so that among
        programs with equal likelihood, the simpler one is preferred.

        Parameters
        ----------
        prompt : str
        observed_outcome : Outcome
        predict_fn : callable ``fn(program, prompt) -> int``
        noise_level : float, optional
            Soft likelihood noise (default ``self._noise_level``).

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
        nl = self._noise_level if noise_level is None else max(0.0, min(0.49, float(noise_level)))
        log_posterior = np.log(np.clip(self._posterior, 1e-12, 1.0))

        for i, c in enumerate(self._candidates):
            pred = predict_fn(c.program, prompt)
            likelihood = (1.0 - nl) if pred == observed_outcome else nl
            log_posterior[i] += np.log(max(likelihood, 1e-12))
            # NOTE: Occam penalty was removed — complexity is NOT penalised
            # during belief updates.  All programs compete on evidence alone.

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
        self._update_count += 1
        self._belief_dirty = False
        # Fix 4: record posterior snapshot for diagnostics
        self._posterior_history.append(self._posterior.copy())
        best = self.most_likely()
        if best is not None:
            pid = best.program_id
            if pid not in self._topk_posterior_traces:
                self._topk_posterior_traces[pid] = []
            self._topk_posterior_traces[pid].append(float(self._posterior[self._find_index(pid)]))
        return self._posterior

    @property
    def info_gains(self) -> List[float]:
        return list(self._info_gains)

    @property
    def total_info_gain(self) -> float:
        return sum(self._info_gains)

    def _normalise(self) -> None:
        """Ensure posterior sums to 1 and matches candidate count.

        **Fix P1**: When candidate count changes, this extends or truncates
        the posterior array instead of calling ``reset_belief()``, which
        would destroy all accumulated evidence.
        """
        n = len(self._candidates)
        if n == 0:
            self._posterior = np.array([], dtype=np.float64)
            self._belief_dirty = False
            return
        if len(self._posterior) != n:
            if len(self._posterior) < n:
                missing = n - len(self._posterior)
                self._posterior = np.append(
                    self._posterior, np.full(missing, _NEW_CANDIDATE_INITIAL_POSTERIOR)
                )
            else:
                self._posterior = self._posterior[:n]
        total = self._posterior.sum()
        if total > 0:
            self._posterior = self._posterior / total
        self._belief_dirty = False

    def set_holdout_accuracy(self, holdout_accuracy: float) -> None:
        """Set holdout accuracy on the most likely candidate."""
        best = self.most_likely()
        if best is not None:
            best.holdout_accuracy = holdout_accuracy
            self._holdout_accuracy_history.append(holdout_accuracy)

    def boost_candidate(self, program_id: str, posterior_value: float = 0.99) -> bool:
        idx = self._find_index(program_id)
        if idx is None or self.num_candidates == 0:
            return False
        n = len(self._posterior)
        remainder = 1.0 - posterior_value
        per_other = remainder / max(n - 1, 1)
        for i in range(n):
            self._posterior[i] = posterior_value if i == idx else per_other
        self._normalise()
        self._belief_dirty = False
        return True

    def boost_multiple(self, program_ids: List[str], total_mass: float = 0.99) -> int:
        """Boost several candidates equally, distributing ``total_mass``
        among them.  The remaining ``1 - total_mass`` is split among all
        other candidates.  Returns the number of candidates boosted."""
        idxs = [self._find_index(pid) for pid in program_ids]
        idxs = [i for i in idxs if i is not None]
        if not idxs or self.num_candidates == 0:
            return 0
        n = len(self._posterior)
        n_boosted = len(idxs)
        per_boosted = total_mass / n_boosted
        other_mass = 1.0 - total_mass
        n_other = max(n - n_boosted, 1)
        per_other = other_mass / n_other
        for i in range(n):
            self._posterior[i] = per_boosted if i in idxs else per_other
        self._normalise()
        self._belief_dirty = False
        return n_boosted

    # ------------------------------------------------------------------
    # Disagreement analysis

    def get_disagreement_pairs(
        self,
        prompts: List[str],
        executor: Any,
        top_k: int = 5,
        use_posterior_tie_breaker: bool = False,
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
        use_posterior_tie_breaker : bool
            If True, break ties at equal disagreement count by selecting
            the pair with the smallest posterior product p(h1)*p(h2).
            This favours exploring uncertain hypotheses.  Default False.

        Returns
        -------
        list of (h1, h2, prompt, disagreement)
            Where disagreement = |pred1 - pred2|.
        """
        if len(self._candidates) < 2 or not prompts:
            return []

        # Aggregate disagreement counts per pair across all prompts
        pair_disagreements: Dict[Tuple[str, str], Tuple[CandidateProgram, CandidateProgram, int]] = {}

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
                    if p1 != p2:
                        key = (c1.program_id, c2.program_id) if c1.program_id < c2.program_id else (c2.program_id, c1.program_id)
                        if key not in pair_disagreements:
                            pair_disagreements[key] = (c1, c2, 0)
                        _, _, count = pair_disagreements[key]
                        pair_disagreements[key] = (c1, c2, count + 1)

        if not pair_disagreements:
            return []

        # Sort by disagreement count (highest first), return top-k.
        # If use_posterior_tie_breaker is True, break ties at equal disagreement
        # by selecting the pair with the smallest posterior product (most uncertain).
        if use_posterior_tie_breaker:
            self._normalise()
            def _tie_break_key(item):
                c1, c2, count = item
                i1 = self._find_index(c1.program_id)
                i2 = self._find_index(c2.program_id)
                p1 = float(self._posterior[i1]) if i1 is not None else 0.0
                p2 = float(self._posterior[i2]) if i2 is not None else 0.0
                return (-count, p1 * p2)
            sorted_pairs = sorted(
                pair_disagreements.values(),
                key=_tie_break_key,
            )[:top_k]
        else:
            sorted_pairs = sorted(
                pair_disagreements.values(),
                key=lambda x: -x[2],
            )[:top_k]

        # For each top pair, return the first prompt where they disagree
        results: List[Tuple[CandidateProgram, CandidateProgram, str, float]] = []
        for c1, c2, _ in sorted_pairs:
            for prompt in prompts:
                try:
                    p1 = int(executor.execute(c1.program, prompt))
                    p2 = int(executor.execute(c2.program, prompt))
                except Exception:
                    continue
                if p1 != p2:
                    results.append((c1, c2, prompt, 1.0))
                    break

        return results

    def get_most_uncertain_pair(
        self,
        prompts: List[str],
        executor: Any,
        use_posterior_tie_breaker: bool = False,
    ) -> Optional[Tuple[CandidateProgram, CandidateProgram, str, float]]:
        """Return the single most uncertain pair + prompt combination.

        Parameters
        ----------
        prompts : list of str
            Base prompts to evaluate.
        executor : ProgramExecutor
            Executor to run program predictions.
        use_posterior_tie_breaker : bool
            If True, break ties at equal disagreement count by selecting
            the pair with the smallest posterior product p(h1)*p(h2).

        Returns
        -------
        (h1, h2, prompt, disagreement) or None if no uncertainty found.
        """
        pairs = self.get_disagreement_pairs(prompts, executor, top_k=1, use_posterior_tie_breaker=use_posterior_tie_breaker)
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

    def get_disagreement_pairs_posterior_weighted(
        self,
        prompts: List[str],
        executor: Any,
        top_k: int = 5,
    ) -> List[Tuple[CandidateProgram, CandidateProgram, str, float]]:
        """Find disagreement pairs weighted by posterior probability.

        Unlike ``get_disagreement_pairs`` which counts raw disagreement,
        this method weights each pair's disagreement by the product of
        their posterior probabilities, giving more weight to pairs where
        both candidates have high belief.

        Returns
        -------
        list of (h1, h2, prompt, posterior_weighted_disagreement)
        """
        if len(self._candidates) < 2 or not prompts:
            return []
        self._normalise()

        results: List[Tuple[CandidateProgram, CandidateProgram, str, float]] = []

        for prompt in prompts:
            predictions = {}
            for c in self._candidates:
                try:
                    predictions[c.program_id] = int(executor.execute(c.program, prompt))
                except Exception:
                    predictions[c.program_id] = 0

            for i, c1 in enumerate(self._candidates):
                for j, c2 in enumerate(self._candidates[i + 1:], i + 1):
                    if i >= len(self._posterior) or j >= len(self._posterior):
                        continue
                    p1 = predictions.get(c1.program_id, 0)
                    p2 = predictions.get(c2.program_id, 0)
                    disagreement = abs(p1 - p2)
                    if disagreement > 0:
                        posterior_weight = float(self._posterior[i] * self._posterior[j])
                        weighted = disagreement * posterior_weight
                        if weighted > 0:
                            results.append((c1, c2, prompt, weighted))

        results.sort(key=lambda x: -x[3])
        return results[:top_k * 2]

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

        Returns -1.0 when empty (no candidates → undefined),
        0.0 when exactly 1 candidate (certain by default),
        and the Shannon entropy H(p) = -Σ p_i log(p_i) for ≥2 candidates.
        """
        n = len(self._candidates)
        if n == 0:
            return -1.0
        if n < 2:
            return 0.0
        self._normalise()
        eps = 1e-12
        p = np.clip(self._posterior, eps, 1.0)
        return float(max(0.0, -np.sum(p * np.log(p))))

    def is_converged(self, threshold: float = 0.1, min_cycles: int = 3) -> bool:
        """Check if posterior entropy indicates convergence.

        Returns True only when:
          * At least *min_cycles* entropy values have been recorded.
          * There are ≥2 candidates in the version space.
          * All recent entropy values are below *threshold* and non-negative.

        Returns False for empty or single-candidate version spaces
        (entropy is degenerate in those cases).
        """
        if len(self._candidates) < 2:
            return False
        if len(self._entropy_history) < min_cycles:
            return False
        recent = self._entropy_history[-min_cycles:]
        return all(0.0 <= e < threshold for e in recent)

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
    # Source / predicate-type analysis
    # ------------------------------------------------------------------

    def count_by_source(self) -> Dict[str, int]:
        """Return count of candidates per source."""
        counts: Dict[str, int] = {}
        for c in self._candidates:
            src = c.source or "unknown"
            counts[src] = counts.get(src, 0) + 1
        return counts

    def count_by_predicate_type(self) -> Dict[str, int]:
        """Return count of candidates per predicate family."""
        return self.family_counts()

    def posterior_by_source(self) -> Dict[str, float]:
        """Return total posterior mass per source."""
        if self._belief_dirty:
            self._normalise()
        masses: Dict[str, float] = {}
        for i, c in enumerate(self._candidates):
            src = c.source or "unknown"
            p = float(self._posterior[i]) if i < len(self._posterior) else 0.0
            masses[src] = masses.get(src, 0.0) + p
        return masses

    def posterior_by_predicate_type(self) -> Dict[str, float]:
        """Return total posterior mass per predicate family."""
        return self.family_posterior_mass()

    def survival_rate_by_source(self) -> Dict[str, float]:
        """Return survival rate per source (survived / total appeared)."""
        rates: Dict[str, float] = {}
        for src, total in self._total_by_source.items():
            survived = self._survival_by_source.get(src, 0)
            rates[src] = survived / total if total > 0 else 0.0
        return rates

    def survival_rate_by_predicate_type(self) -> Dict[str, float]:
        """Return survival rate per predicate type."""
        rates: Dict[str, float] = {}
        for ptype, total in self._total_by_predicate_type.items():
            survived = self._survival_by_predicate_type.get(ptype, 0)
            rates[ptype] = survived / total if total > 0 else 0.0
        return rates

    def source_lifetime_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics per source."""
        return {
            "total_by_source": dict(self._total_by_source),
            "survived_by_source": dict(self._survival_by_source),
            "current_by_source": self.count_by_source(),
        }

    def predicate_type_lifetime_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics per predicate type."""
        return {
            "total_by_type": dict(self._total_by_predicate_type),
            "survived_by_type": dict(self._survival_by_predicate_type),
            "current_by_type": self.count_by_predicate_type(),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_candidates": len(self._candidates),
            "max_candidates": self._max_candidates,
            "entropy": self.entropy(),
            "total_info_gain": self.total_info_gain,
            "num_updates": self._update_count,
            "num_prunes": self._prune_count,
            "num_syntheses": self._synthesis_count,
            "posterior_by_source": self.posterior_by_source(),
            "survival_rate_by_source": self.survival_rate_by_source(),
            "source_lifetime_stats": self.source_lifetime_stats(),
            "posterior_by_predicate_type": self.posterior_by_predicate_type(),
            "survival_rate_by_predicate_type": self.survival_rate_by_predicate_type(),
            "predicate_type_lifetime_stats": self.predicate_type_lifetime_stats(),
            # Fix 4: posterior diagnostics
            "entropy_history": self.entropy_history,
            "holdout_accuracy_history": self.holdout_accuracy_history,
            "posterior_history": self.posterior_history,
            "topk_posterior_traces": self.topk_posterior_traces,
            "candidates": [
                {
                    "program_id": c.program_id,
                    "accuracy": c.accuracy,
                    "complexity": c.complexity,
                    "posterior": float(self._posterior[i]) if i < len(self._posterior) else 0.0,
                    "source": c.source,
                    "predicate_type": c.predicate_type or c.family,
                    "holdout_accuracy": c.holdout_accuracy,
                    "train_accuracy": c.train_accuracy,
                    "generalization_gap": c.generalization_gap,
                }
                for i, c in enumerate(self._candidates)
            ],
            "program_asts": {
                c.program_id: c.program.to_dict() if hasattr(c.program, "to_dict") else str(c.program)
                for c in self._candidates
            },
        }

    def record_entropy(self) -> float:
        """Record current entropy and return it."""
        e = self.entropy()
        self._entropy_history.append(e)
        return e

    @property
    def entropy_history(self) -> List[float]:
        return list(self._entropy_history)

    # Fix 4: posterior diagnostics
    @property
    def posterior_history(self) -> List[List[float]]:
        return [p.tolist() for p in self._posterior_history]

    @property
    def topk_posterior_traces(self) -> Dict[str, List[float]]:
        return dict(self._topk_posterior_traces)

    @property
    def holdout_accuracy_history(self) -> List[float]:
        return list(self._holdout_accuracy_history)
