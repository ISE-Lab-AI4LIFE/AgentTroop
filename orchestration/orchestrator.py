"""Orchestrator (V2) — coordinates the 6-phase HARMONY-X exploration loop
using KnowledgeManager and SessionMemory.

This is the primary orchestrator that manages the interaction between
Cognitive, Strategist, and Researcher agents through a 6-phase loop:

    1. Cognitive Agent — detect anomalies from Episodic Memory
    2. Cognitive Agent — generate structural hypotheses from anomalies
    3. Strategist Agent — design optimal intervention for a hypothesis pair
    4. Strategist Agent — execute intervention and record outcome
    5. Researcher Agent — synthesize program from all episodes
    6. Researcher Agent — verify, store program, abstract theory

It uses SessionMemory (L2, Redis) for checkpoint/state persistence and
KnowledgeManager for permission-aware read/write to all knowledge stores.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agents.red_team import RedTeamAgent
from agents.cognitive import Anomaly, CognitiveAgent, Hypothesis
from agents.researcher import ResearcherAgent
from agents.strategist import StrategistAgent
from core import ConditionRegistry
from core.condition import registry as _condition_registry
from core.intervention import Intervention
from core.executor import ProgramExecutor
from core.primitive import ContainsWordPredicate
from inference.belief_updater import BayesianBeliefUpdater
from inference.pomdp import POMDPAction, POMDPObservation
from inference.version_space import VersionSpace
from knowledge.manager import KnowledgeManager, Target
from knowledge.session_memory import SessionMemory
from orchestration.surrogate_policy_model import SurrogatePolicyModel
from orchestration.counterfactual_learner import CounterfactualLearner
from core.jailbreak import apply_technique_to_intervention

logger = logging.getLogger(__name__)

_SYNTHESIS_INTERVAL = 5

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "it", "its", "my", "your", "his",
    "her", "our", "their", "no", "nor", "not", "or", "and", "but",
    "if", "because", "so", "than", "too", "very", "just", "about",
    "also", "make", "get", "give", "tell", "show", "without",
}


class OrchestratorPhase(Enum):
    IDLE = 0
    ANOMALY_DETECTION = 1
    HYPOTHESIS_GENERATION = 2
    INTERVENTION_DESIGN = 3
    INTERVENTION_EXECUTION = 4
    PROGRAM_SYNTHESIS = 5
    VERIFICATION_AND_STORE = 6
    CONVERGED = 7


class Orchestrator:
    """Coordinates the 6-phase HARMONY-X loop with KnowledgeManager + SessionMemory.

    The POMDP belief state is the **central driving mechanism**:

        Observe → Update Belief → Select Action → Execute → Observe → ...

    Every intervention decision passes through the belief state.  The
    ``BayesianBeliefUpdater`` is mandatory (auto-created from the
    StrategistAgent's internal updater if not provided).

    Parameters
    ----------
    cognitive_agent : CognitiveAgent
    strategist_agent : StrategistAgent
    researcher_agent : ResearcherAgent
    knowledge_manager : KnowledgeManager
        Permission-aware read/write access to all knowledge stores.
    session_memory : SessionMemory
        Redis-backed campaign state cache (L2).
    victim : BaseVictim
        Target LLM to reverse-engineer.
    campaign_id : str
    experiment_id : str, optional
    max_iterations : int
        Maximum number of full 6-phase iterations (default 50).
    max_interventions : int
        Hard cap on total intervention count (default 500).
    accuracy_threshold : float
        Program accuracy needed to consider the target reverse-engineered (default 0.9).
    allow_error_rate : float
        Noise tolerance for program synthesis (default 0.0).
    synthesis_interval : int
        Run Researcher synthesis every N interventions (default 5).
    force_exploration_interval : int
        After this many consecutive iterations without a designed
        intervention (``_run_strategist_phase`` returns ``None``),
        force an exploration intervention to prevent pipeline stall
        (default 3).
    entropy_convergence_threshold : float
        When belief entropy drops below this value, consider converged.
    """

    def __init__(
        self,
        cognitive_agent: CognitiveAgent,
        strategist_agent: StrategistAgent,
        researcher_agent: ResearcherAgent,
        knowledge_manager: KnowledgeManager,
        session_memory: SessionMemory,
        victim: Any,
        campaign_id: str,
        experiment_id: Optional[str] = None,
        red_team_agent: Optional[RedTeamAgent] = None,
        max_iterations: int = 50,
        max_interventions: int = 500,
        accuracy_threshold: float = 0.9,
        allow_error_rate: float = 0.0,
        synthesis_interval: int = _SYNTHESIS_INTERVAL,
        force_exploration_interval: int = 3,
        entropy_convergence_threshold: float = 0.1,
        belief_updater: Any = None,
        version_space: Optional[VersionSpace] = None,
        top_k_candidates: int = 30,
        seed_telemetry: Optional[Dict[str, Any]] = None,
        # Fix 1: Occam factor — complexity-aware Bayesian prior
        complexity_prior_lambda: float = 0.01,
        # Fix 3: Hardened convergence criteria
        min_interventions_for_convergence: int = 10,
        min_holdout_size_for_convergence: int = 20,
        min_holdout_accuracy_for_convergence: float = 0.8,
        max_generalization_gap: float = 0.1,
        # Semantic fix 2: External holdout prompts file
        holdout_prompts_path: Optional[str] = None,
    ) -> None:
        self.cognitive = cognitive_agent
        self.strategist = strategist_agent
        self.researcher = researcher_agent
        self.red_team = red_team_agent
        self.knowledge_manager = knowledge_manager
        self.session_memory = session_memory
        self.victim = victim
        self.campaign_id = campaign_id
        self.experiment_id = experiment_id
        self.max_iterations = max(1, int(max_iterations))
        self.max_interventions = max(1, int(max_interventions))
        self.accuracy_threshold = accuracy_threshold
        self.allow_error_rate = allow_error_rate
        self.synthesis_interval = max(1, int(synthesis_interval))
        self.force_exploration_interval = max(1, int(force_exploration_interval))
        self.entropy_convergence_threshold = entropy_convergence_threshold
        self.complexity_prior_lambda = max(0.0, float(complexity_prior_lambda))
        self.min_interventions_for_convergence = max(1, int(min_interventions_for_convergence))
        self.min_holdout_size_for_convergence = max(1, int(min_holdout_size_for_convergence))
        self.min_holdout_accuracy_for_convergence = min(1.0, max(0.0, float(min_holdout_accuracy_for_convergence)))
        self.max_generalization_gap = min(1.0, max(0.0, float(max_generalization_gap)))

        # Semantic fix 2: External holdout prompts
        self.holdout_prompts_path = holdout_prompts_path
        self._holdout_prompts_cache: Optional[List[Tuple[str, int]]] = None

        self.phase = OrchestratorPhase.IDLE
        self.iteration = 0
        self.top_k_candidates = max(2, top_k_candidates)

        # ── Version Space: single source of truth (owned by Orchestrator) ──
        self.version_space = version_space or VersionSpace(
            max_candidates=self.top_k_candidates,
            complexity_prior_lambda=self.complexity_prior_lambda,
        )
        # All agents reference the same object
        self.strategist._version_space = self.version_space

        # Track belief history for convergence
        self._belief_history: List[float] = []
        self._entropy_history: List[float] = []
        self._efe_log: List[Dict[str, Any]] = []
        # Synthesis telemetry counters
        self._synthesis_attempts: int = 0
        self._synthesis_successes: int = 0
        self._synthesis_total_candidates: int = 0
        self._synthesis_fallbacks: int = 0

        # Surrogate Policy Model — trained on all episodes to provide signal
        # when victim outcome is uniform (≈100% REFUSE or ≈100% ACCEPT)
        self.surrogate = SurrogatePolicyModel()
        self._surrogate_training_stats: List[Dict[str, Any]] = []

        # Counterfactual Learning Layer — generates counterfactual prompt pairs
        # to extract invariant features from uniform outcomes
        self.counterfactual_learner = CounterfactualLearner()
        self._counterfactual_pairs: List[Any] = []

        # Anomaly-source telemetry
        self._seed_telemetry = seed_telemetry or {}
        self._anomaly_telemetry: Dict[str, Any] = {}
        self._intervention_telemetry: Dict[str, int] = {}

        # Diversity tracking
        self._diversity_history: List[Dict[str, Any]] = []
        self._used_techniques: List[str] = []

        self._session_created = self._ensure_session()
        logger.info(
            "Orchestrator V2 initialised: campaign=%s model=%s "
            "max_iter=%d max_intv=%d acc_thresh=%.2f "
            "version_space=%s entropy_thresh=%.2f top_k=%d",
            campaign_id, getattr(victim, "name", str(victim)),
            self.max_iterations, self.max_interventions,
            self.accuracy_threshold,
            self.version_space is not None,
            self.entropy_convergence_threshold,
            self.top_k_candidates,
        )

    def _ensure_session(self) -> bool:
        try:
            existing = self.session_memory.get_session(self.campaign_id)
            if existing is None:
                target_model = getattr(self.victim, "name", "unknown")
                return self.session_memory.create_session(
                    self.campaign_id, target_model,
                    metadata={
                        "allow_error_rate": self.allow_error_rate,
                        "accuracy_threshold": self.accuracy_threshold,
                        "experiment_id": self.experiment_id or "",
                    },
                )
            return True
        except Exception as exc:
            logger.warning("Session creation failed (Redis may be down): %s", exc)
            return False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _log_phase_time(self, phase_name: str, start: float) -> float:
        elapsed = time.time() - start
        logger.info("⏱ PHASE_TIMING: %s took %.3fs", phase_name, elapsed)
        return time.time()

    def run(self) -> Dict[str, Any]:
        """Execute the full 6-phase loop with POMDP as the central mechanism.

        POMDP cycle driving every intervention:

            Observe (anomalies / episode)
                ↓
            Update Belief (Bayesian)
                ↓
            Select Action (strategist via belief entropy)
                ↓
            Execute Intervention
                ↓
            Observe outcome
                ↓
            [loop]

        Returns
        -------
        dict with keys:
            - success (bool)
            - campaign_id (str)
            - best_program_id (str or None)
            - best_accuracy (float)
            - total_interventions (int)
            - total_iterations (int)
            - belief_entropy_history (list[float])
            - efe_log (list[dict])
            - error (str or None)
        """
        start_time = time.time()
        self.iteration = 0
        total_interventions = 0
        self._belief_history = []
        self._entropy_history = []
        self._efe_log = []
        self._synthesis_attempts = 0
        self._synthesis_successes = 0
        self._synthesis_total_candidates = 0
        self._synthesis_fallbacks = 0

        # Track timing per phase
        self._phase_timing: Dict[str, List[float]] = {
            "cognitive": [], "strategist_design": [], "strategist_execute": [],
            "belief_update": [], "synthesis": [], "holdout_eval": [], "full_cycle": [],
        }

        try:
            # Phase 1-2: Cognitive Agent → hypotheses
            t0 = time.time()
            logger.info("=== Campaign %s: Phase 1-2 (Cognitive) ===",
                        self.campaign_id)
            anomalies, hypotheses = self._run_cognitive_phase()
            t0 = self._log_phase_time("cognitive_phase_1_2", t0)
            self._phase_timing["cognitive"].append(time.time() - t0 + (time.time() - start_time))
            self._init_belief_states(hypotheses)

            # Compute anomaly-source telemetry
            self._anomaly_telemetry = self._compute_anomaly_telemetry(anomalies)

            if not hypotheses:
                return self._result(
                    success=False,
                    error="No hypotheses generated from anomalies",
                    total_interventions=total_interventions,
                )

            # Phase 3-6: Data → Programs → Version Space → Disagreement → Intervention
            stalled_iterations = 0
            converged_by_entropy = False
            converged_by_accuracy = False
            executor = ProgramExecutor(
                getattr(self.strategist, "primitive_registry", None),
            )

            for iteration in range(1, self.max_iterations + 1):
                self.iteration = iteration
                self._cycle_start = time.time()
                logger.info(
                    "=== Cycle %d/%d (campaign=%s, VS entropy=%.3f, candidates=%d) ===",
                    iteration, self.max_iterations, self.campaign_id,
                    self._get_current_entropy(),
                    self.version_space.num_candidates,
                )

                if total_interventions >= self.max_interventions:
                    logger.info("Intervention budget exhausted (%d)", total_interventions)
                    break

                self._record_belief_state()

                # ── Design intervention (disagreement-driven) ──
                t_design = time.time()
                intervention = self._run_strategist_phase(hypotheses)
                self._phase_timing["strategist_design"].append(time.time() - t_design)

                if intervention is None:
                    stalled_iterations += 1
                    logger.info(
                        "No intervention designed (%d consecutive stalls)",
                        stalled_iterations,
                    )
                    if stalled_iterations >= self.force_exploration_interval:
                        logger.warning(
                            "Force exploration after %d stalled iterations",
                            stalled_iterations,
                        )
                        # Try counterfactual prompts before force exploration
                        if self._counterfactual_pairs:
                            cf_pair = self._counterfactual_pairs.pop(0)
                            from core.intervention import Intervention
                            intervention = Intervention(
                                base_prompt=cf_pair.counterfactual_prompt,
                                transforms=[],
                                metadata={
                                    "counterfactual": True,
                                    "pair_id": cf_pair.pair_id,
                                    "feature_changed": cf_pair.feature_changed,
                                },
                            )
                            stalled_iterations = 0
                            logger.info(
                                "Using counterfactual prompt (feature=%s) as intervention",
                                cf_pair.feature_changed,
                            )
                        else:
                            intervention = self._force_exploration_intervention(hypotheses)
                            if intervention is None:
                                logger.warning("Force exploration also failed; stopping")
                                break
                            stalled_iterations = 0
                    else:
                        continue

                elif intervention.metadata.get("exploratory"):
                    stalled_iterations += 1
                else:
                    stalled_iterations = 0

                # ── Technique selection & application (Phase 3+) ──
                if self.phase.value > 2 and intervention is not None:
                    if not intervention.metadata.get("technique"):
                        tech = self.strategist.select_technique(
                            goal=intervention.base_prompt,
                            used_techniques=self._used_techniques,
                        )
                        intervention = apply_technique_to_intervention(intervention, tech)
                        self._used_techniques.append(tech)

                # ── Red Team LLM refinement (skips Phase 1-2 reconnaissance) ──
                if self.red_team is not None and self.phase.value > 2:
                    intervention = self.red_team.maybe_refine_intervention(intervention)

                # ── Execute intervention → Observe outcome ──
                t_exec = time.time()
                outcome = self.strategist.execute_intervention(intervention, self.victim)
                self._phase_timing["strategist_execute"].append(time.time() - t_exec)

                # ── Record technique outcome for adaptive selection ──
                technique = intervention.metadata.get("technique")
                if technique:
                    from core.jailbreak import record_technique_outcome
                    record_technique_outcome(technique, intervention.base_prompt, outcome)

                # ── Update version space belief ──
                t_belief = time.time()
                self._update_belief_from_observation(intervention, outcome, hypotheses)
                self._phase_timing["belief_update"].append(time.time() - t_belief)

                # ── Store episode ──
                episode_id = self.strategist.store_intervention(
                    intervention=intervention,
                    outcome=outcome,
                    campaign_id=self.campaign_id,
                    h1=intervention.metadata.get("h1"),
                    h2=intervention.metadata.get("h2"),
                    experiment_id=self.experiment_id,
                    victim_name=getattr(self.victim, "name", "victim"),
                )
                total_interventions += 1
                self.session_memory.increment_intervention_count(self.campaign_id)
                self.session_memory.increment_iteration(self.campaign_id)

                efe_record = self._record_efe(intervention, outcome, hypotheses)
                if efe_record:
                    self._efe_log.append(efe_record)

                self._update_causal_graph(intervention)
                self._poll_and_process_proposals()

                # ── Phase 5: Top-K Program Synthesis → Version Space ──
                if iteration == 1 or total_interventions % self.synthesis_interval == 0:
                    t_synth = time.time()
                    self._synthesize_and_update_version_space()
                    self._phase_timing["synthesis"].append(time.time() - t_synth)

                # ── Holdout evaluation → posterior reweight (every cycle) ──
                # Running every cycle differentiates near-identical candidates
                # (e.g. multiple keyword programs) that make the same training
                # predictions but diverge on unseen holdout prompts.
                if total_interventions >= 3:
                    holdout_result = self.evaluate_on_holdout(holdout_prompts=[])
                    self._last_holdout_size = holdout_result.get("holdout_size", 0) if holdout_result else 0
                    if holdout_result and holdout_result.get("holdout_size", 0) > 0:
                        best = self.version_space.most_likely()
                        if best is not None:
                            self.version_space.set_holdout_accuracy(
                                holdout_result.get("holdout_accuracy", 0.0)
                            )
                        # Reweight posterior using holdout-adjusted scores so
                        # candidates with verified holdout accuracy dominate
                        # over unsubstantiated seeds.
                        self.version_space.reweight_by_holdout()
                        # Feed holdout failures back into EpisodicMemory
                        # to amplify signal for candidates with large generalization gap.
                        self._feed_holdout_failures(holdout_result)

                # ── Diversity telemetry per cycle ──
                self._emit_diversity_telemetry()

                # ── Cycle timing summary ──
                cycle_time = time.time() - self._cycle_start
                self._phase_timing["full_cycle"].append(cycle_time)
                logger.info(
                    "⏱ CYCLE_TIMING: cycle=%d total=%.3fs design=%.3fs exec=%.3fs belief=%.3fs",
                    iteration, cycle_time,
                    self._phase_timing["strategist_design"][-1] if self._phase_timing["strategist_design"] else 0,
                    self._phase_timing["strategist_execute"][-1] if self._phase_timing["strategist_execute"] else 0,
                    self._phase_timing["belief_update"][-1] if self._phase_timing["belief_update"] else 0,
                )

                # ── VS-level entropy recording for convergence detection ──
                self.version_space.record_entropy()

                # ── Entropy-based convergence check (hardened: Fix 3) ──
                current_entropy = self._get_current_entropy()
                if len(self._entropy_history) >= 5 and self.version_space.num_candidates >= 2:
                    recent = self._entropy_history[-5:]
                    if all(e < self.entropy_convergence_threshold for e in recent):
                        best = self.version_space.most_likely()
                        real_holdout = getattr(best, "holdout_accuracy", 0.0) or 0.0
                        # Fix 3: require minimum intervention count
                        if total_interventions >= self.min_interventions_for_convergence:
                            # Fix 3: require minimum holdout size and accuracy
                            last_holdout = getattr(self, '_last_holdout_size', 0)
                            if (last_holdout >= self.min_holdout_size_for_convergence
                                    and real_holdout >= self.min_holdout_accuracy_for_convergence):
                                gap = abs(getattr(best, "accuracy", 0.0) - real_holdout)
                                if gap <= self.max_generalization_gap:
                                    converged_by_entropy = True
                                    logger.info(
                                        "Version space entropy below %.3f for 5 cycles "
                                        "(%d candidates, holdout=%.3f, interventions=%d, "
                                        "holdout_size=%d, gap=%.3f); converged by entropy",
                                        self.entropy_convergence_threshold,
                                        self.version_space.num_candidates,
                                        real_holdout, total_interventions,
                                        last_holdout, gap,
                                    )
                                    self._emit_vs_telemetry(
                                        "converged_by_entropy",
                                        threshold=self.entropy_convergence_threshold,
                                    )
                                    break

                # ── Holdout-based convergence check (hardened: Fix 3) ──
                # Convergence is purely evidence-driven: if the candidate
                # with the highest posterior also has high accuracy on both
                # train and holdout with a small generalization gap, it has
                # genuinely learned the policy — regardless of whether the
                # predicate is keyword, structural, or semantic.
                best = self.version_space.most_likely()
                if best is not None:
                    real_holdout = getattr(best, "holdout_accuracy", 0.0) or 0.0
                    last_holdout_size = getattr(self, '_last_holdout_size', 0)
                    if (total_interventions >= self.min_interventions_for_convergence
                            and last_holdout_size >= self.min_holdout_size_for_convergence
                            and real_holdout >= self.min_holdout_accuracy_for_convergence
                            and best.accuracy >= self.accuracy_threshold):
                        gap = abs(best.accuracy - real_holdout)
                        if gap <= self.max_generalization_gap:
                            converged_by_accuracy = True
                            logger.info(
                                "Best candidate train=%.3f holdout=%.3f gap=%.3f "
                                "type=%s interventions=%d holdout_size=%d; "
                                "converged by holdout",
                                best.accuracy, real_holdout, gap,
                                best.predicate_type,
                                total_interventions, last_holdout_size,
                            )
                            self.session_memory.set_best_program(
                                self.campaign_id, best.program_id, best.accuracy,
                            )
                            self._emit_vs_telemetry(
                                "converged_by_accuracy",
                                best_id=best.program_id,
                                best_accuracy=best.accuracy,
                                holdout_accuracy=real_holdout,
                                predicate_type=best.predicate_type,
                            )
                            break

            # ── Finalize ──
            self.session_memory.set_status(self.campaign_id, "completed")
            self._persist_belief_history()

            best = self.version_space.most_likely()
            best_id = best.program_id if best is not None else None
            best_acc = best.accuracy if best is not None else 0.0
            if best_id:
                self.session_memory.set_best_program(
                    self.campaign_id, best_id, best_acc,
                )

            return self._result(
                success=True,
                best_program_id=best_id,
                best_accuracy=best_acc,
                total_interventions=total_interventions,
                converged_by=f"{'accuracy' if converged_by_accuracy else ''}{' + ' if converged_by_accuracy and converged_by_entropy else ''}{'entropy' if converged_by_entropy else 'iteration_limit'}",
            )

        except Exception as exc:
            logger.exception("Pipeline failed: %s", exc)
            try:
                self.session_memory.set_status(self.campaign_id, "failed")
            except Exception:
                pass
            return self._result(
                success=False,
                error=str(exc),
                total_interventions=total_interventions,
            )
        finally:
            elapsed = time.time() - start_time
            # Timing summary
            timing_report = {
                "total_elapsed_s": round(elapsed, 2),
                "iterations": self.iteration,
                "total_interventions": total_interventions,
            }
            for phase, timings in self._phase_timing.items():
                if timings:
                    timing_report[f"{phase}_total_s"] = round(sum(timings), 3)
                    timing_report[f"{phase}_avg_s"] = round(sum(timings) / len(timings), 3)
                    timing_report[f"{phase}_count"] = len(timings)
            logger.info(
                "⏱ TIMING_SUMMARY: %s",
                json.dumps(timing_report, indent=2),
            )
            logger.info(
                "Campaign %s finished in %.1fs: "
                "iterations=%d interventions=%d entropy_history=%d",
                self.campaign_id, elapsed,
                self.iteration, total_interventions,
                len(self._entropy_history),
            )

    async def async_run(self) -> Dict[str, Any]:
        """Async variant of run() using the strategist's async intervention execution.

        The POMDP loop is identical to run() but calls
        ``strategist.async_execute_intervention`` to avoid blocking the
        event loop during I/O-bound victim API calls.
        """
        start_time = time.time()
        self.iteration = 0
        total_interventions = 0
        self._belief_history = []
        self._entropy_history = []
        self._efe_log = []

        # Track timing per phase
        self._phase_timing: Dict[str, List[float]] = {
            "cognitive": [], "strategist_design": [], "strategist_execute": [],
            "belief_update": [], "synthesis": [], "holdout_eval": [], "full_cycle": [],
        }

        try:
            t0 = time.time()
            anomalies, hypotheses = self._run_cognitive_phase()
            t0 = self._log_phase_time("cognitive_phase_1_2", t0)
            self._init_belief_states(hypotheses)

            # Compute anomaly-source telemetry
            self._anomaly_telemetry = self._compute_anomaly_telemetry(anomalies)

            if not hypotheses:
                return self._result(
                    success=False,
                    error="No hypotheses generated from anomalies",
                    total_interventions=total_interventions,
                )

            stalled_iterations = 0
            converged_by_entropy = False
            converged_by_accuracy = False

            for iteration in range(1, self.max_iterations + 1):
                self.iteration = iteration
                self._cycle_start = time.time()
                logger.info(
                    "=== Async cycle %d/%d (campaign=%s, VS entropy=%.3f, candidates=%d) ===",
                    iteration, self.max_iterations, self.campaign_id,
                    self._get_current_entropy(),
                    self.version_space.num_candidates,
                )

                if total_interventions >= self.max_interventions:
                    logger.info("Intervention budget exhausted (%d)", total_interventions)
                    break

                self._record_belief_state()

                t_design = time.time()
                intervention = self._run_strategist_phase(hypotheses)
                self._phase_timing["strategist_design"].append(time.time() - t_design)

                if intervention is None:
                    stalled_iterations += 1
                    logger.info(
                        "No intervention designed (%d consecutive stalls)",
                        stalled_iterations,
                    )
                    if stalled_iterations >= self.force_exploration_interval:
                        logger.warning(
                            "Force exploration after %d stalled iterations",
                            stalled_iterations,
                        )
                        # Try counterfactual prompts before force exploration
                        if self._counterfactual_pairs:
                            cf_pair = self._counterfactual_pairs.pop(0)
                            from core.intervention import Intervention
                            intervention = Intervention(
                                base_prompt=cf_pair.counterfactual_prompt,
                                transforms=[],
                                metadata={
                                    "counterfactual": True,
                                    "pair_id": cf_pair.pair_id,
                                    "feature_changed": cf_pair.feature_changed,
                                },
                            )
                            stalled_iterations = 0
                            logger.info(
                                "Using counterfactual prompt (feature=%s) as intervention",
                                cf_pair.feature_changed,
                            )
                        else:
                            intervention = self._force_exploration_intervention(hypotheses)
                            if intervention is None:
                                logger.warning("Force exploration also failed; stopping")
                                break
                            stalled_iterations = 0
                    else:
                        continue
                elif intervention.metadata.get("exploratory"):
                    stalled_iterations += 1
                else:
                    stalled_iterations = 0

                # ── Technique selection & application (Phase 3+) ──
                if self.phase.value > 2 and intervention is not None:
                    if not intervention.metadata.get("technique"):
                        tech = self.strategist.select_technique(
                            goal=intervention.base_prompt,
                            used_techniques=self._used_techniques,
                        )
                        intervention = apply_technique_to_intervention(intervention, tech)
                        self._used_techniques.append(tech)

                # ── Red Team LLM refinement (skips Phase 1-2 reconnaissance) ──
                if self.red_team is not None and self.phase.value > 2:
                    intervention = self.red_team.maybe_refine_intervention(intervention)

                t_exec = time.time()
                outcome = await self.strategist.async_execute_intervention(
                    intervention, self.victim
                )
                self._phase_timing["strategist_execute"].append(time.time() - t_exec)

                # ── Record technique outcome for adaptive selection ──
                technique = intervention.metadata.get("technique")
                if technique:
                    from core.jailbreak import record_technique_outcome
                    record_technique_outcome(technique, intervention.base_prompt, outcome)

                t_belief = time.time()
                self._update_belief_from_observation(intervention, outcome, hypotheses)
                self._phase_timing["belief_update"].append(time.time() - t_belief)

                episode_id = self.strategist.store_intervention(
                    intervention=intervention,
                    outcome=outcome,
                    campaign_id=self.campaign_id,
                    h1=intervention.metadata.get("h1"),
                    h2=intervention.metadata.get("h2"),
                    experiment_id=self.experiment_id,
                    victim_name=getattr(self.victim, "name", "victim"),
                )
                total_interventions += 1
                self.session_memory.increment_intervention_count(self.campaign_id)
                self.session_memory.increment_iteration(self.campaign_id)

                efe_record = self._record_efe(intervention, outcome, hypotheses)
                if efe_record:
                    self._efe_log.append(efe_record)

                self._update_causal_graph(intervention)
                self._poll_and_process_proposals()

                if iteration == 1 or total_interventions % self.synthesis_interval == 0:
                    self._synthesize_and_update_version_space()

                # ── Holdout evaluation (async) ──
                if total_interventions >= 3 and total_interventions % max(2, self.synthesis_interval) == 0:
                    holdout_result = self.evaluate_on_holdout(holdout_prompts=[])
                    if holdout_result and holdout_result.get("holdout_size", 0) > 0:
                        best = self.version_space.most_likely()
                        if best is not None:
                            self.version_space.set_holdout_accuracy(
                                holdout_result.get("holdout_accuracy", 0.0)
                            )

                # ── Diversity telemetry per cycle ──
                self._emit_diversity_telemetry()

                # ── Cycle timing summary (async) ──
                cycle_time = time.time() - self._cycle_start
                self._phase_timing["full_cycle"].append(cycle_time)
                logger.info(
                    "⏱ ASYNC_CYCLE_TIMING: cycle=%d total=%.3fs design=%.3fs exec=%.3fs belief=%.3fs",
                    iteration, cycle_time,
                    self._phase_timing["strategist_design"][-1] if self._phase_timing["strategist_design"] else 0,
                    self._phase_timing["strategist_execute"][-1] if self._phase_timing["strategist_execute"] else 0,
                    self._phase_timing["belief_update"][-1] if self._phase_timing["belief_update"] else 0,
                )

                # ── VS-level entropy recording for convergence detection ──
                self.version_space.record_entropy()

                current_entropy = self._get_current_entropy()
                if len(self._entropy_history) >= 5 and self.version_space.num_candidates >= 2:
                    recent = self._entropy_history[-5:]
                    if all(e < self.entropy_convergence_threshold for e in recent):
                        best = self.version_space.most_likely()
                        real_holdout = getattr(best, "holdout_accuracy", 0.0) or 0.0
                        if real_holdout >= 0.65:
                            converged_by_entropy = True
                            logger.info(
                                "Version space entropy below %.3f for 5 cycles "
                                "(%d candidates, holdout=%.3f); converged by entropy",
                                self.entropy_convergence_threshold,
                                self.version_space.num_candidates,
                                real_holdout,
                            )
                            break

                # ── Holdout-based convergence check (async) ──
                best = self.version_space.most_likely()
                if best is not None:
                    real_holdout = getattr(best, "holdout_accuracy", 0.0) or 0.0
                    if (real_holdout > 0.0
                            and best.accuracy >= self.accuracy_threshold):
                        gap = abs(best.accuracy - real_holdout)
                        if gap < 0.15:
                            converged_by_accuracy = True
                            logger.info(
                                "Best candidate train=%.3f holdout=%.3f gap=%.3f "
                                "type=%s >= threshold; converged by holdout",
                                best.accuracy, real_holdout, gap,
                                best.predicate_type,
                            )
                            self.session_memory.set_best_program(
                                self.campaign_id, best.program_id, best.accuracy,
                            )
                            break

            self.session_memory.set_status(self.campaign_id, "completed")
            self._persist_belief_history()

            best = self.version_space.most_likely()
            best_id = best.program_id if best is not None else None
            best_acc = best.accuracy if best is not None else 0.0
            if best_id:
                self.session_memory.set_best_program(
                    self.campaign_id, best_id, best_acc,
                )

            return self._result(
                success=True,
                best_program_id=best_id,
                best_accuracy=best_acc,
                total_interventions=total_interventions,
                converged_by=f"{'accuracy' if converged_by_accuracy else ''}{' + ' if converged_by_accuracy and converged_by_entropy else ''}{'entropy' if converged_by_entropy else 'iteration_limit'}",
            )

        except Exception as exc:
            logger.exception("Async pipeline failed: %s", exc)
            try:
                self.session_memory.set_status(self.campaign_id, "failed")
            except Exception:
                pass
            return self._result(
                success=False,
                error=str(exc),
                total_interventions=total_interventions,
            )
        finally:
            elapsed = time.time() - start_time
            timing_report = {
                "total_elapsed_s": round(elapsed, 2),
                "iterations": self.iteration,
                "total_interventions": total_interventions,
            }
            for phase, timings in self._phase_timing.items():
                if timings:
                    timing_report[f"{phase}_total_s"] = round(sum(timings), 3)
                    timing_report[f"{phase}_avg_s"] = round(sum(timings) / len(timings), 3)
                    timing_report[f"{phase}_count"] = len(timings)
            logger.info(
                "⏱ ASYNC_TIMING_SUMMARY: %s",
                json.dumps(timing_report, indent=2),
            )
            logger.info(
                "Async campaign %s finished in %.1fs: "
                "iterations=%d interventions=%d entropy_history=%d",
                self.campaign_id, elapsed,
                self.iteration, total_interventions,
                len(self._entropy_history),
            )

    # ------------------------------------------------------------------
    # Phase runners
    # ------------------------------------------------------------------

    def _run_cognitive_phase(
        self,
    ) -> Tuple[List[Anomaly], List[Hypothesis]]:
        """Phase 1-2: anomaly detection + hypothesis generation."""
        self.phase = OrchestratorPhase.ANOMALY_DETECTION
        logger.info("Phase 1/6: anomaly detection")
        anomalies = self.cognitive.detect_anomalies(
            campaign_id=self.campaign_id,
            experiment_id=self.experiment_id,
        )

        self.phase = OrchestratorPhase.HYPOTHESIS_GENERATION
        logger.info("Phase 2/6: hypothesis generation (%d anomalies)", len(anomalies))
        prior = self._load_prior_hypotheses()

        # Warm-start from scientific memory
        try:
            scientific_memory = self.knowledge_manager.get_store(Target.SCIENTIFIC.value)
            target_model = getattr(self.victim, "name", "unknown")
            if scientific_memory:
                warm_hyps = self.cognitive.warm_start_from_scientific_memory(
                    scientific_memory=scientific_memory,
                    target_model=target_model,
                )
                if warm_hyps:
                    if prior is None:
                        prior = []
                    prior.extend(warm_hyps)
        except Exception as exc:
            logger.warning("Failed to warm-start hypotheses: %s", exc)

        hypotheses = self.cognitive.generate_hypotheses(
            anomalies,
            prior_hypotheses=prior,
        )

        # Persist hypotheses in HypothesisStore for prior loading
        try:
            self.cognitive.store_hypotheses(hypotheses, campaign_id=self.campaign_id)
        except Exception as exc:
            logger.warning("Failed to persist hypotheses: %s", exc)

        for h in hypotheses:
            try:
                self.session_memory.add_hypothesis(self.campaign_id, h.id)
            except Exception:
                pass

        # Hypotheses are persisted directly through cognitive.store_hypotheses()
        # and loaded as priors via _load_prior_hypotheses → cognitive.get_stored_hypotheses().
        # No separate proposal to defense_program_store is needed (the store type
        # is for synthesized programs, not hypotheses).
        return anomalies, hypotheses

    def _load_prior_hypotheses(self) -> Optional[List[Hypothesis]]:
        try:
            hyp_ids = self.session_memory.list_hypotheses(self.campaign_id)
            if not hyp_ids:
                return None
        except Exception:
            return None
        try:
            prior = self.cognitive.get_stored_hypotheses(
                hypothesis_ids=hyp_ids,
            )
            if prior:
                logger.info(
                    "Loaded %d prior hypotheses from store",
                    len(prior),
                )
            return prior or None
        except Exception as exc:
            logger.warning("Failed to load prior hypotheses: %s", exc)
            return None

    def _run_strategist_phase(
        self,
        hypotheses: List[Hypothesis],
    ) -> Optional[Intervention]:
        """Phase 3-4: design and execute intervention for hypothesis pair."""
        self.phase = OrchestratorPhase.INTERVENTION_DESIGN
        logger.info("Phase 3/6: intervention design (%d hypotheses)", len(hypotheses))

        h1, h2 = self.strategist.select_hypothesis_pair(
            hypotheses,
            campaign_id=self.campaign_id,
            experiment_id=self.experiment_id,
        )
        if h1 is None or h2 is None:
            return None

        self.phase = OrchestratorPhase.INTERVENTION_EXECUTION
        intervention = self.strategist.design_intervention(
            h1, h2,
            campaign_id=self.campaign_id,
            experiment_id=self.experiment_id,
        )
        if intervention is not None:
            intervention.metadata.setdefault("h1", getattr(h1, "id", str(h1)))
            intervention.metadata.setdefault("h2", getattr(h2, "id", str(h2)))

        return intervention

    # ------------------------------------------------------------------
    # Anomaly-source telemetry
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_anomaly_telemetry(anomalies: List[Any]) -> Dict[str, Any]:
        """Aggregate anomaly metadata into per-family counts and rates."""
        from collections import Counter
        families: Counter = Counter()
        sources: Counter = Counter()
        categories: Counter = Counter()
        tags: Counter = Counter()

        family_total = 0
        for a in anomalies:
            tf = getattr(a, "transform_family", "") or ""
            src = getattr(a, "anomaly_source", "") or ""
            cat = getattr(a, "semantic_category", "") or ""
            tag = getattr(a, "source_tag", "") or ""
            if tf:
                families[tf] += 1
                family_total += 1
            if src:
                sources[src] += 1
            if cat:
                categories[cat] += 1
            if tag:
                tags[tag] += 1

        total = family_total or 1
        return {
            "anomaly_count": len(anomalies),
            "anomaly_count_by_family": dict(families),
            "anomaly_rate_by_family": {
                k: round(v / total, 4) for k, v in families.items()
            },
            "anomaly_count_by_source": dict(sources),
            "anomaly_count_by_category": dict(categories),
            "anomaly_count_by_tag": dict(tags),
        }

    def _force_exploration_intervention(
        self,
        hypotheses: List[Hypothesis],
    ) -> Optional[Intervention]:
        """Create a forced exploration intervention when the pipeline is stalled.

        Picks a random hypothesis pair (or single hypothesis + null) and
        generates a random intervention with a random transform to maximise
        diversity when the heuristic search finds no discriminating candidate.
        """
        if not hypotheses:
            return None

        h1 = random.choice(hypotheses)
        h2 = hypotheses[0] if len(hypotheses) < 2 else random.choice(
            [h for h in hypotheses if h is not h1]
        )
        if h1 is h2:
            return None

        intervention = self.strategist.design_intervention(
            h1, h2,
            campaign_id=self.campaign_id,
            experiment_id=self.experiment_id,
        )

        if intervention is None:
            props = self.strategist._resolve_base_prompts(
                None, self.campaign_id, self.experiment_id,
            )
            if props:
                bp = random.choice(props)
                transforms = self.strategist._get_transforms()
                if transforms:
                    t = random.choice(transforms)
                    intervention = Intervention(base_prompt=bp, transforms=[t])
                    intervention.metadata.setdefault("h1", getattr(h1, "id", str(h1)))
                    intervention.metadata.setdefault("h2", getattr(h2, "id", str(h2)))

        if intervention is not None:
            intervention.metadata.setdefault("h1", getattr(h1, "id", str(h1)))
            intervention.metadata.setdefault("h2", getattr(h2, "id", str(h2)))
            intervention.metadata["exploratory"] = True
            logger.info("Created force-exploration intervention (prompt=%s, transforms=%s)",
                         intervention.base_prompt[:80],
                         [t.name for t in intervention.transforms])
        return intervention

    def _poll_and_process_proposals(self) -> None:
        """Poll proposals for each owner agent and process them.

        Follows the sequence diagram (Section 3.3): proposals submitted
        by non-owner agents are polled by the owner and processed.
        """
        targets_to_poll = [
            ("defense_program_store", self.researcher),
            ("scientific_memory", self.researcher),
            ("causal_graph", self.researcher),
        ]
        for target, agent in targets_to_poll:
            try:
                proposals = self.knowledge_manager.poll_proposals(
                    agent_id="ResearcherAgent",
                    target=target,
                    timeout=0.0,
                )
                if proposals:
                    processed = self.researcher.process_proposals(proposals)
                    for prop, result in zip(proposals, processed):
                        pid = prop.get("proposal_id", "")
                        if pid:
                            self.knowledge_manager.resolve_proposal(
                                proposal_id=pid,
                                accepted=result.get("success", False),
                                result=result.get("result"),
                                error=result.get("error"),
                            )
            except Exception as exc:
                logger.debug("Proposal polling failed for %s: %s", target, exc)

    def _update_causal_graph(self, intervention: Intervention) -> None:
        """Update causal graph using historical episodes for the same transform."""
        if not intervention.transforms:
            return

        causal_store = self.knowledge_manager.get_store(Target.CAUSAL.value)
        if not causal_store or not hasattr(causal_store, "do_intervention"):
            return

        episodic = self.knowledge_manager.get_store(Target.EPISODIC.value)
        if not episodic:
            return

        # Use the first transform as the source node, REFUSAL as target
        t0 = intervention.transforms[0]
        source_name = t0.name if hasattr(t0, "name") else ""
        if not source_name:
            return
        target_name = "REFUSAL"

        try:
            from knowledge.episodic import EpisodeFilter
            episodes = episodic.filter_episodes(EpisodeFilter(campaign_id=self.campaign_id))
        except Exception:
            return

        x1_outcomes = []
        base_prompts = set()

        for ep in episodes:
            inv = ep.intervention
            if inv and inv.transforms and inv.transforms[0].get("name") == source_name:
                x1_outcomes.append(int(ep.outcome))
                base_prompts.add(inv.prompt)

        x0_outcomes = []
        for ep in episodes:
            inv = ep.intervention
            if inv and not inv.transforms and inv.prompt in base_prompts:
                x0_outcomes.append(int(ep.outcome))

        if x0_outcomes and x1_outcomes:
            try:
                causal_store.do_intervention(
                    intervention_id=intervention.id,
                    source_name=source_name,
                    target_name=target_name,
                    outcomes_do_x0=x0_outcomes,
                    outcomes_do_x1=x1_outcomes,
                )
            except ValueError as e:
                # Expected when observations < threshold, ignore
                logger.debug("Causal graph do_intervention skipped: %s", e)
            except Exception as e:
                logger.warning("Causal graph update failed: %s", e)

    def _synthesize_and_update_version_space(self) -> None:
        """Run top-K synthesis and add candidates to version space.

        Invariant: after this method, VersionSpace.num_candidates >= 10.
        If synthesis returns 0 programs, generates heuristic fallback
        candidates from keyword-based predicates to ensure the VS is
        never empty.

        **Fix P1**: No longer calls ``reset_belief()``.  New candidates
        are added incrementally with near-zero initial posterior, and
        existing posteriors are preserved.  When synthesis returns
        duplicate programs (same program_id as existing candidates),
        only accuracy is updated — posterior mass remains untouched.

        **Fix P4**: ``_generate_heuristic_candidates`` now includes
        all predicate families (keyword, structural, jailbreak,
        semantic, discourse) for balanced coverage.
        """
        try:
            from synthesis.cvc5_synthesizer import CVC5Synthesizer
            from inference.version_space import _classify_program
            synthesizer = getattr(self.researcher, "synthesizer", None)
            if synthesizer is None:
                logger.warning("No synthesizer available on researcher agent")
                return

            episodes = self._fetch_episodes()
            if not episodes:
                logger.info("No episodes for synthesis yet")
                return

            executor = ProgramExecutor(
                getattr(self.strategist, "primitive_registry", None),
            )

            # FIX P3: Adaptive error tolerance — auto-increase when
            # consecutive synthesis rounds return 0 programs.
            consecutive = getattr(self, '_consecutive_synthesis_failures', 0)
            if consecutive >= 2:
                new_rate = min(0.45, 0.15 + consecutive * 0.05)
                old_rate = synthesizer.allow_error_rate
                synthesizer.allow_error_rate = new_rate
                logger.info(
                    "Adaptive error tolerance: allow_error_rate=%.2f -> %.2f "
                    "(%d consecutive failures)",
                    old_rate, new_rate, consecutive,
                )

            self._synthesis_attempts += 1
            programs = synthesizer.synthesize_top_k(
                episodes, k=self.top_k_candidates,
            )

            # ── Track synthesis success/failure ──
            if not programs:
                self._consecutive_synthesis_failures = getattr(self, '_consecutive_synthesis_failures', 0) + 1
                self._synthesis_fallbacks += 1
            else:
                self._consecutive_synthesis_failures = 0
                self._synthesis_successes += 1

            # ── Heuristic fallback: if synthesis found 0 programs ──
            if not programs:
                logger.warning(
                    "Synthesis found 0 matching programs; "
                    "generating full-family heuristic fallback candidates"
                )
                programs = self._generate_heuristic_candidates(episodes, executor)

            # FIX P1: Incremental addition — preserve existing posterior
            existing_ids = set(self.version_space.program_ids)
            entropy_before = self.version_space.entropy()
            added = 0
            updated = 0
            for prog in programs:
                accuracy = self._compute_accuracy(prog, episodes, executor)
                source = getattr(prog, "source", "enumeration")
                prog_id = getattr(prog, "id", "") or ""
                if prog_id in existing_ids:
                    # Update accuracy only — posterior preserved
                    self.version_space.add_candidate(
                        program=prog, accuracy=accuracy, source=source,
                        episodes_matched=int(accuracy * len(episodes)),
                        total_episodes=len(episodes),
                    )
                    updated += 1
                else:
                    # New candidate — initial posterior scales with accuracy
                    # so substantiated programs (>0.0) start higher than seeds
                    self.version_space.add_candidate(
                        program=prog, accuracy=accuracy, source=source,
                        episodes_matched=int(accuracy * len(episodes)),
                        total_episodes=len(episodes),
                    )
                    added += 1

            # FIX P1: DO NOT reset_belief — posterior preserved
            entropy_after = self.version_space.entropy()
            logger.info(
                "Version space: added=%d updated=%d (total=%d, "
                "entropy_before=%.3f entropy_after=%.3f)",
                added, updated, self.version_space.num_candidates,
                entropy_before, entropy_after,
            )

            # FIX: CVC5 exact-match winner override — if CVC5 found
            # programs with zero errors on the training examples, boost
            # ALL of them equally to dominate the version space.  This
            # prevents the pipeline from selecting a spurious program
            # (e.g. contains_word('instructions')) over the correct one
            # (contains_word('bomb')) when the Bayesian posterior alone
            # cannot differentiate equally-accurate programs on small
            # data.
            #
            # We boost ALL exact-match programs collectively (splitting
            # 0.99 among them) so that the correct keyword program gets
            # the same boost as any spurious semantic program.  The
            # holdout set's reweighting then breaks the tie among them.
            vs = self.version_space
            exact_match_ids: List[str] = []
            exact_match_acc = 0.0
            for prog in (programs or []):
                prog_id = getattr(prog, "id", "")
                if not prog_id:
                    continue
                metadata = getattr(prog, "metadata", {}) or {}
                if not metadata.get("exact_match"):
                    continue
                cand_idx = vs._find_index(prog_id)
                if cand_idx is not None and cand_idx < len(vs._posterior):
                    acc = float(vs._candidates[cand_idx].accuracy or 0.0)
                    if acc >= 0.999:
                        exact_match_ids.append(prog_id)
                        exact_match_acc = max(exact_match_acc, acc)
            if exact_match_ids:
                n_boosted = vs.boost_multiple(exact_match_ids, 0.99)
                logger.info(
                    "CVC5 exact-match override: boosted %d / %d candidate(s) "
                    "posterior to 0.99 (exact_match=True, accuracy=%.3f)",
                    n_boosted, len(exact_match_ids), exact_match_acc,
                )

            # FIX 2: Type diversity survival guarantee — ensure at least
            # 3 candidates of each core predicate type survive pruning.
            vs = self.version_space
            core_types = {"keyword", "structural", "semantic", "composite", "classifier"}
            type_counts = vs.count_by_predicate_type()
            for ptype in core_types:
                count = type_counts.get(ptype, 0)
                if count >= 3 or vs.num_candidates == 0:
                    continue

                # Boost existing candidates of this type so they survive pruning
                for i, c in enumerate(vs.candidates):
                    if (c.predicate_type or c.family or "") == ptype:
                        if i < len(vs.posterior):
                            vs.posterior[i] = max(vs.posterior[i], 0.5)

                # If type has 0 candidates, rescue from synthesis/heuristic pool
                if count == 0:
                    rescue_candidates = [
                        p for p in (programs or [])
                        if p not in vs.program_ids
                        and hasattr(p, 'root')
                        and _classify_program(p) == ptype
                    ]
                    # If synthesis pool has none of this type, generate fresh
                    # heuristic candidates for the missing type(s).
                    if not rescue_candidates:
                        try:
                            heur = self._generate_heuristic_candidates(
                                episodes, executor,
                            )
                            rescue_candidates = [
                                p for p in heur
                                if p not in vs.program_ids
                                and hasattr(p, 'root')
                                and _classify_program(p) == ptype
                            ]
                        except Exception as exc:
                            logger.debug(
                                "FIX2: Heuristic generation failed for type '%s': %s",
                                ptype, exc,
                            )
                    rescue_candidates.sort(
                        key=lambda p: self._compute_accuracy(p, episodes, executor)
                        if episodes else 0.0,
                        reverse=True,
                    )
                    for p in rescue_candidates[:3 - count]:
                        acc = self._compute_accuracy(p, episodes, executor) if episodes else 0.0
                        vs.add_candidate(
                            program=p, accuracy=acc, source="type_rescue",
                            episodes_matched=int(acc * len(episodes)) if episodes else 0,
                            total_episodes=len(episodes) if episodes else 0,
                        )

                logger.info(
                    "FIX2: Protected/boosted type '%s' (count=%d -> target=3)",
                    ptype, count,
                )

            # Cap: no single predicate type exceeds 40% of max_candidates
            max_per_type = max(3, int(self.top_k_candidates * 0.4))
            type_counts = vs.count_by_predicate_type()
            for ptype, count in type_counts.items():
                if count > max_per_type:
                    excess = count - max_per_type
                    candidates_of_type = [
                        c for c in vs.candidates
                        if (c.predicate_type or c.family or "") == ptype
                    ]
                    candidates_of_type.sort(
                        key=lambda c: vs.posterior[vs.candidates.index(c)]
                        if vs.candidates.index(c) < len(vs.posterior) else 0.0
                    )
                    for c in candidates_of_type[:excess]:
                        vs.remove_candidate(c.program_id)
                    logger.info(
                        "Capped type '%s' from %d to %d (removed %d lowest-posterior)",
                        ptype, count, max_per_type, excess,
                    )

            # Persist to SessionMemory for resume support
            try:
                vs_candidates = [
                    {
                        "program_id": c.program_id,
                        "accuracy": c.accuracy,
                        "posterior": float(
                            self.version_space.posterior[i]
                        ) if i < len(self.version_space.posterior) else 0.0,
                        "source": c.source,
                        "family": getattr(c, "family", "unknown"),
                    }
                    for i, c in enumerate(self.version_space.candidates)
                ]
                self.session_memory.set_version_space(
                    self.campaign_id, vs_candidates,
                )
            except Exception as exc:
                logger.debug("Failed to persist version space: %s", exc)

            # Trigger verification of the best candidate
            best = self.version_space.most_likely()
            if best is not None:
                verified = self.researcher.verify_and_store(
                    program=best.program,
                    campaign_id=self.campaign_id,
                    victim=self.victim,
                    program_id=best.program_id,
                )
                # Semantic fix 1: Boost verified program's posterior immediately
                if verified:
                    vs = self.version_space
                    boosted = vs.boost_candidate(best.program_id, 0.99)
                    if boosted:
                        logger.info(
                            "BOOST_VERIFIED: program_id=%s accuracy=%.3f "
                            "new_posterior=0.99",
                            best.program_id,
                            getattr(best, "accuracy", 0.0),
                        )
                    else:
                        logger.warning(
                            "BOOST_VERIFIED: failed to boost %s",
                            best.program_id,
                        )
        except Exception as exc:
            logger.warning("Synthesis → version space update failed: %s", exc)

    def _generate_heuristic_candidates(
        self,
        episodes: List[Tuple[str, int]],
        executor: ProgramExecutor,
    ) -> List[Any]:
        """Generate diverse heuristic fallback candidates from all predicate
        families when synthesis finds nothing.

        Includes keyword, length, roleplay, jailbreak, encoding, override,
        and structural predicates to ensure the VS has diverse candidates.
        """
        import re
        from collections import Counter
        from core.program import Program, IfThenElseNode, PredicateNode
        from core.primitive import (
            ContainsWordPredicate, ContainsAnyWordPredicate,
            LengthGtPredicate, LengthLtPredicate,
            StartsWithRoleplayPredicate, ContainsSystemOverridePredicate,
            MatchesJailbreakPatternPredicate, ContainsEncodingWrapperPredicate,
            ContainsCodeBlockPredicate, ContainsDelimiterPredicate,
            ContainsLeetPredicate,
            HasNumberPredicate, HasSpecialCharPredicate,
            IsAllCapsPredicate, IsEmptyPredicate,
            HasEmojiPredicate, ContainsURLPredicate,
            IsGrammaticalQuestionPredicate, StartsWithImperativePredicate,
            IsRepetitivePredicate,
            SentimentPredicate, IntentPredicate,
        )
        from core.condition import registry as _cond_registry

        if not episodes:
            return []

        # Extract keywords from REFUSE examples
        refuse_words: Counter = Counter()
        for prompt, outcome in episodes:
            if outcome == 1:
                words = re.findall(r"[a-zA-Z]{3,}", prompt.lower())
                refuse_words.update(w for w in words if w not in _STOPWORDS)

        top_kw = [w for w, _ in refuse_words.most_common(10)]
        if not top_kw:
            top_kw = ["the", "a", "to", "is", "of", "in", "it", "that", "this", "you"]

        programs: List[Program] = []

        def _make_prog(primitive, then_out=1, else_out=0, source="heuristic"):
            prog = Program(
                root=IfThenElseNode(
                    condition=PredicateNode(primitive=primitive),
                    then_outcome=then_out,
                    else_outcome=else_out,
                )
            )
            prog.source = source
            return prog

        # 1. Keyword predicates
        for kw in top_kw[:5]:
            for to, eo in [(1, 0), (0, 1)]:
                programs.append(_make_prog(ContainsWordPredicate(word=kw), to, eo))

        # 2. ContainsAnyWord
        if len(top_kw) >= 2:
            programs.append(_make_prog(ContainsAnyWordPredicate(words=top_kw[:3]), 1, 0))
            programs.append(_make_prog(ContainsAnyWordPredicate(words=top_kw[:3]), 0, 1))

        # 3. Length predicates
        for threshold in [30, 50, 100, 200]:
            for to, eo in [(1, 0), (0, 1)]:
                programs.append(_make_prog(LengthGtPredicate(threshold=threshold), to, eo))
                programs.append(_make_prog(LengthLtPredicate(threshold=threshold), to, eo))

        # 4. Structural predicates (no-arg, one instance each)
        structural_preds = [
            StartsWithRoleplayPredicate(), ContainsSystemOverridePredicate(),
            MatchesJailbreakPatternPredicate(), ContainsEncodingWrapperPredicate(),
            ContainsCodeBlockPredicate(), ContainsDelimiterPredicate(),
            ContainsLeetPredicate(),
            HasNumberPredicate(), HasSpecialCharPredicate(),
            IsAllCapsPredicate(), IsEmptyPredicate(),
            HasEmojiPredicate(), ContainsURLPredicate(),
            IsGrammaticalQuestionPredicate(), StartsWithImperativePredicate(),
            IsRepetitivePredicate(),
        ]
        for sp in structural_preds:
            for to, eo in [(1, 0), (0, 1)]:
                programs.append(_make_prog(sp, to, eo))

        # 5. Semantic predicates
        programs.append(_make_prog(SentimentPredicate(threshold=0.55), 1, 0))
        programs.append(_make_prog(SentimentPredicate(threshold=0.55), 0, 1))
        programs.append(_make_prog(IntentPredicate(intent_type="harmful"), 1, 0))
        programs.append(_make_prog(IntentPredicate(intent_type="harmful"), 0, 1))
        programs.append(_make_prog(IntentPredicate(intent_type="innocuous"), 1, 0))
        programs.append(_make_prog(IntentPredicate(intent_type="innocuous"), 0, 1))

        # FIX 5: Check which core types are extinct in version space and
        # lower the threshold for reintroducing them.
        extinct_types: set = set()
        vs = getattr(self, "version_space", None)
        if vs is not None and vs.num_candidates > 0:
            type_counts = vs.count_by_predicate_type()
            core_types = {"keyword", "structural", "semantic"}
            for ptype in core_types:
                if type_counts.get(ptype, 0) == 0:
                    extinct_types.add(ptype)
                    logger.info("FIX5: Type '%s' is extinct — lowering reintroduction threshold", ptype)

        # Filter: keep diverse candidates; keyword candidates pass through
        # even with low accuracy since they provide a useful baseline posterior
        # signal.  The Bayesian update will naturally demote low-accuracy
        # programs over time.
        valid: List[Any] = []
        from inference.version_space import _classify_program
        for prog in programs:
            acc = self._compute_accuracy(prog, episodes, executor)
            ptype = ""
            cond_ok = hasattr(prog, 'root') and hasattr(prog.root, 'condition')
            if cond_ok:
                ptype = _classify_program(prog)
            if acc >= 0.6:
                valid.append(prog)
            elif acc >= 0.4:
                valid.append(prog)
            elif ptype in extinct_types and cond_ok:
                existing_of_type = sum(1 for p in valid if _classify_program(p) == ptype)
                if existing_of_type < 3:
                    valid.append(prog)
            elif ptype in ("keyword", "structural", "semantic") and cond_ok:
                existing_of_type = sum(1 for p in valid if _classify_program(p) == ptype)
                if existing_of_type < 3:
                    valid.append(prog)

        # Ensure diversity: no single predicate family >50% of valid set
        if len(valid) > 2:
            from collections import Counter as _Counter
            fam_count: _Counter = _Counter()
            for p in valid:
                root = p.root
                if hasattr(root, "condition") and hasattr(root.condition, "primitive"):
                    fam_count[type(root.condition.primitive).__name__] += 1
            total = len(valid)
            for fam, cnt in fam_count.most_common():
                if cnt / total > 0.5:
                    excess = cnt - int(total * 0.5)
                    valid = [p for p in valid if not (
                        hasattr(p.root, "condition") and
                        hasattr(p.root.condition, "primitive") and
                        type(p.root.condition.primitive).__name__ == fam and
                        (excess := excess - 1) >= -1
                    )]

        if not valid:
            valid = programs[:self.top_k_candidates]

        logger.info(
            "Generated %d diverse heuristic candidates from %d raw "
            "(keywords=%d, structural=%d, semantic=%d, length=%d)",
            len(valid), len(programs), len(top_kw), len(structural_preds), 6, 4,
        )
        return valid

    def _fetch_episodes(self) -> List[Tuple[str, int]]:
        """Fetch all episodes for the current campaign as (prompt, outcome) pairs."""
        try:
            from knowledge.episodic import EpisodeFilter
            from knowledge.manager import Target
            ep_mem = self.knowledge_manager.get_store(Target.EPISODIC.value)
            if ep_mem is None:
                return []
            episodes = ep_mem.filter_episodes(EpisodeFilter(
                campaign_id=self.campaign_id,
                experiment_id=self.experiment_id,
            ))
            return [
                (ep.intervention.final_prompt, int(ep.outcome))
                for ep in episodes
            ]
        except Exception as exc:
            logger.debug("Failed to fetch episodes: %s", exc)
            return []

    @staticmethod
    def _compute_accuracy(
        program: Any,
        examples: List[Tuple[str, int]],
        executor: ProgramExecutor,
    ) -> float:
        if not examples:
            return 0.0
        correct = 0
        for prompt, expected in examples:
            try:
                if int(executor.execute(program, prompt)) == expected:
                    correct += 1
            except Exception:
                pass
        return correct / len(examples)

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def resume(self) -> Dict[str, Any]:
        """Resume a previously interrupted campaign from its saved state.

        Loads the campaign session from SessionMemory and continues
        the pipeline from where it left off.
        """
        session = self.session_memory.get_session(self.campaign_id)
        if session is None:
            return self._result(
                success=False,
                error=f"No saved session for campaign '{self.campaign_id}'",
            )
        logger.info(
            "Resuming campaign %s (status=%s, iterations=%d)",
            self.campaign_id, session.get("status"), session.get("iteration"),
        )
        self.iteration = int(session.get("iteration", 0))
        return self.run()

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # POMDP belief helpers
    # ------------------------------------------------------------------

    def _init_belief_states(self, hypotheses: List[Any]) -> None:
        """Initialize version space from hypotheses (if no programs yet).

        When no candidate programs exist yet, seeds the version space by
        converting hypotheses into heuristic candidate programs.
        Every hypothesis with a ``program`` attribute is added directly.
        For keyword-only hypotheses, uses the Strategist's compilation
        bridge to convert condition strings into DSL Programs (supports
        all 27 predicate types), so the VS has semantically rich
        candidates from cycle 1.

        Uses ``absorb_candidates`` so posterior is uniform (no prior data),
        and the same entry-point is used as the synthesis path for
        consistent provenance tracking.
        """
        if self.version_space.num_candidates > 0:
            return
        if not hypotheses:
            return
        import re
        from core.program import Program, IfThenElseNode, PredicateNode
        from core.primitive import ContainsWordPredicate

        condition_registry = getattr(self.strategist, "condition_registry", _condition_registry)
        new_entries: List[Tuple[Program, float, str, int, int]] = []
        seen_conditions: set = set()
        compile_stats = {"compiled": 0, "keyword_only": 0, "failed": 0}
        for h in hypotheses:
            prog = getattr(h, "program", None)
            if prog is not None:
                accuracy = getattr(h, "confidence", 0.0)
                new_entries.append((prog, accuracy, "hypothesis", 0, 0))
                continue

            # Try compilation bridge first (handles all 27 predicate types)
            cond = getattr(h, "condition", "") or getattr(h, "description", "")
            if not cond:
                continue

            # Primary: ConditionRegistry lookup when hypothesis has condition_name
            cond_name = getattr(h, "condition_name", None)
            if cond_name is not None and cond_name in condition_registry:
                compiled = StrategistAgent.compile_condition_to_program(cond)
                if compiled is not None:
                    compiled.source = "compiled_from_condition"
                    new_entries.append((compiled, 0.0, "compiled_from_condition", 0, 0))
                    compile_stats["compiled"] += 1
                    continue

            # Secondary: Strategist's compilation bridge
            compiled = StrategistAgent.compile_condition_to_program(cond)
            if compiled is not None:
                compiled.source = "compiled_from_condition"
                new_entries.append((compiled, 0.0, "compiled_from_condition", 0, 0))
                compile_stats["compiled"] += 1
                continue

            # Legacy fallback: keyword-only programs from single-quoted tokens
            keywords = re.findall(r"'([^']*)'", cond)
            keywords = [kw for kw in keywords if kw not in seen_conditions]
            if not keywords:
                compile_stats["failed"] += 1
                continue
            for kw in keywords[:3]:  # limit per hypothesis
                seen_conditions.add(kw)
                heuristic_prog = Program(
                    root=IfThenElseNode(
                        condition=PredicateNode(
                            primitive=ContainsWordPredicate(word=kw)
                        ),
                        then_outcome=1,
                        else_outcome=0,
                    )
                )
                heuristic_prog.source = "hypothesis_seed"
                new_entries.append((heuristic_prog, 0.0, "hypothesis_seed", 0, 0))
                compile_stats["keyword_only"] += 1

        added = self.version_space.absorb_candidates(new_entries, balance_types=True)

        # ── Always generate heuristic candidates for diverse starting pool ──
        # Even if hypotheses produced programs, we supplement with heuristic
        # candidates (keyword, structural, length, semantic) so the VS has
        # broad coverage from cycle 1.  This prevents the common failure
        # mode where all hypotheses point to the same predicate type.
        executor = ProgramExecutor(
            getattr(self.strategist, "primitive_registry", None),
        )
        episodes = self._fetch_episodes()
        if episodes:
            heuristic = self._generate_heuristic_candidates(episodes, executor)
            if heuristic:
                heur_entries = []
                for p in heuristic:
                    acc = self._compute_accuracy(p, episodes, executor)
                    heur_entries.append((p, acc, "init_heuristic", int(acc * len(episodes)), len(episodes)))
                added += self.version_space.absorb_candidates(heur_entries, balance_types=True)
                logger.info(
                    "Pre-seeded %d heuristic candidates for diverse starting pool "
                    "(total candidates=%d)", len(heur_entries),
                    self.version_space.num_candidates,
                )

        if added > 0:
            logger.info(
                "Version space: seeded %d candidate programs from %d hypotheses "
                "(candidates=%d, sources=%s, compiled=%d, keyword=%d, failed=%d)",
                added, len(hypotheses), self.version_space.num_candidates,
                self.version_space.count_by_source(),
                compile_stats["compiled"], compile_stats["keyword_only"],
                compile_stats["failed"],
            )
            self._emit_vs_telemetry(
                "init_belief", seeded=added, hypotheses=len(hypotheses),
                compile_stats=compile_stats,
            )
        else:
            logger.info(
                "Version space: no candidate programs could be created from %d hypotheses "
                "(compiled=%d, keyword=%d, failed=%d)",
                len(hypotheses),
                compile_stats["compiled"], compile_stats["keyword_only"],
                compile_stats["failed"],
            )

    def _emit_vs_telemetry(self, event: str, **extra: Any) -> None:
        """Emit structured version-space telemetry without affecting runtime."""
        vs = self.version_space
        telemetry = {
            "event": event,
            "campaign_id": self.campaign_id,
            "iteration": self.iteration,
            "candidate_count": vs.num_candidates,
            "entropy": self._get_current_entropy(),
            "posterior": vs.posterior.tolist() if vs.num_candidates > 0 else [],
            "total_info_gain": vs.total_info_gain,
            "posterior_by_source": vs.posterior_by_source(),
            "source_lifetime_stats": vs.source_lifetime_stats(),
            "survival_rate_by_source": vs.survival_rate_by_source(),
            "synthesis_attempts": self._synthesis_attempts,
            "synthesis_successes": self._synthesis_successes,
            "synthesis_fallbacks": self._synthesis_fallbacks,
        }
        telemetry.update(extra)
        logger.info("VS_TELEMETRY %s", json.dumps(telemetry))

    def _emit_diversity_telemetry(self) -> None:
        """Emit family diversity metrics every cycle."""
        vs = self.version_space
        if vs.num_candidates == 0:
            return
        by_type = vs.count_by_predicate_type()
        by_posterior = vs.posterior_by_predicate_type()
        total = sum(by_type.values())
        diversity = {
            "iteration": self.iteration,
            "entropy": self._get_current_entropy(),
            "total_candidates": total,
            "count_by_type": by_type,
            "posterior_by_type": {k: round(v, 4) for k, v in by_posterior.items()},
            "dominant_type": max(by_type, key=by_type.get) if by_type else "none",
            "dominant_pct": round(max(by_type.values()) / max(total, 1) * 100, 1) if by_type else 0.0,
        }
        if total > 0:
            diversity["posterior_entropy"] = round(
                -sum(p * __import__("math").log(p + 1e-10) for p in by_posterior.values() if p > 0)
                / __import__("math").log(max(len(by_posterior), 2)),
                4,
            )
        self._diversity_history.append(diversity)
        logger.info("DIVERSITY %s", json.dumps(diversity))

    def _feed_holdout_failures(self, holdout_result: Dict[str, Any]) -> None:
        """Feed holdout evaluation failures back into EpisodicMemory.

        **Fix P5**: When a candidate with high posterior has a large
        generalization gap on holdout, its failures are re-inserted
        as new episodes with lower weight (noise_level=0.2) so the
        belief update can incorporate holdout signal without
        overfitting to artifacts.

        If no specific failures are available, uses the best candidate's
        mismatched holdout prompts (weighted by gap magnitude).
        """
        try:
            vs = self.version_space
            best = vs.most_likely()
            if best is None:
                return
            gap = getattr(best, "generalization_gap", 0.0)
            if gap < 0.05:
                return  # Generalization gap too small to warrant correction

            holdout_size = holdout_result.get("holdout_size", 0)
            episodes = self._fetch_episodes()
            if not episodes:
                return

            executor = ProgramExecutor(
                getattr(self.strategist, "primitive_registry", None),
            )

            # Find prompts in holdout (last ~20% of episodes by default)
            split = max(1, len(episodes) - holdout_size)
            holdout_episodes = episodes[split:]

            failures_fed = 0
            for prompt, true_outcome in holdout_episodes:
                try:
                    pred = int(executor.execute(best.program, prompt))
                    if pred != true_outcome:
                        self.strategist.store_intervention(
                            intervention=Intervention(
                                base_prompt=prompt,
                                transforms=[],
                                metadata={
                                    "source": "holdout_feedback",
                                    "program_id": best.program_id,
                                    "original_outcome": true_outcome,
                                    "predicted_outcome": pred,
                                    "generalization_gap": gap,
                                },
                            ),
                            outcome=true_outcome,
                            campaign_id=self.campaign_id,
                            h1=best,
                            h2=None,
                            experiment_id=self.experiment_id,
                            victim_name=getattr(self.victim, "name", "victim"),
                        )
                        failures_fed += 1
                except Exception:
                    continue

            if failures_fed > 0:
                logger.info(
                    "Holdout feedback: fed %d failures (gap=%.3f) into episodic memory",
                    failures_fed, gap,
                )
        except Exception as exc:
            logger.debug("_feed_holdout_failures failed: %s", exc)

    def _record_belief_state(self) -> None:
        """Record current version space entropy to history."""
        entropy = self._get_current_entropy()
        self._entropy_history.append(entropy)
        try:
            self._belief_history.append(self.version_space.to_dict())
        except Exception:
            pass

    def _get_current_entropy(self) -> float:
        try:
            e = self.version_space.entropy()
            return e if e >= 0.0 else 0.0
        except Exception:
            return 0.0

    def _update_belief_from_observation(
        self,
        intervention: Any,
        outcome: int,
        hypotheses: List[Any],
    ) -> None:
        """Update version space posterior after observing intervention outcome.

        Delegates to VersionSpace.update_belief to reweight candidates.
        Also trains the Surrogate Policy Model and generates counterfactual
        prompt pairs for additional signal extraction.
        """
        vs = self.version_space
        if vs.is_empty:
            logger.debug("Version space is empty; skipping belief update")
            return
        try:
            prompt = intervention.final_prompt
            executor = ProgramExecutor(
                getattr(self.strategist, "primitive_registry", None),
            )

            def _predict(program: Any, p: str) -> int:
                try:
                    return int(executor.execute(program, p))
                except Exception as exc:
                    logger.debug("Executor failure for %s: %s",
                                 getattr(program, "id", "?"), exc)
                    if not hasattr(self, "_execution_failures"):
                        self._execution_failures = 0
                    self._execution_failures += 1
                    return 0

            entropy_before = vs.entropy()
            posterior_before = vs.posterior.tolist() if vs.num_candidates > 0 else []
            vs.update_belief(prompt, int(outcome), _predict)
            entropy_after = vs.entropy()
            posterior_after = vs.posterior.tolist() if vs.num_candidates > 0 else []
            ig = entropy_before - entropy_after
            logger.info(
                "Belief update: H=%.3f→%.3f IG=%.4f candidates=%d",
                entropy_before, entropy_after, ig, vs.num_candidates,
            )

            # ── Train Surrogate Policy Model on all episodes ──
            episodes = self._fetch_episodes()
            if len(episodes) >= 3:
                stats = self.surrogate.train(episodes)
                self._surrogate_training_stats.append({
                    "iteration": self.iteration,
                    "n_episodes": stats.n_episodes,
                    "train_accuracy": stats.train_accuracy,
                    "n_refuse": stats.n_refuse,
                    "n_accept": stats.n_accept,
                    "duration_ms": stats.duration_ms,
                })
                # Log surrogate-predicted outcome for this intervention
                spred = self.surrogate.predict(prompt)
                logger.info(
                    "Surrogate: predicted=%d (conf=%.3f, unc=%.3f) vs actual=%d",
                    spred.predicted_outcome, spred.confidence,
                    spred.uncertainty, outcome,
                )

            # ── Generate counterfactual pairs for uniform-outcome cases ──
            if episodes and len(set(o for _, o in episodes)) == 1:
                # All outcomes are the same → need counterfactual signal
                from orchestration.counterfactual_learner import CounterfactualLearner
                prompts = [p for p, _ in episodes[-10:]]
                outcomes = [o for _, o in episodes[-10:]]
                pairs = self.counterfactual_learner.generate_counterfactual_pairs(
                    prompts, outcomes, max_pairs=3,
                )
                for pair in pairs:
                    pair.original_outcome = outcome
                    self._counterfactual_pairs.append(pair)
                if pairs:
                    logger.info(
                        "Counterfactual: generated %d pairs for uniform-outcome learning",
                        len(pairs),
                    )

            self._emit_vs_telemetry(
                "belief_update",
                entropy_before=round(entropy_before, 4),
                entropy_after=round(entropy_after, 4),
                info_gain=round(ig, 4),
                posterior_before=posterior_before[:10],
                posterior_after=posterior_after[:10],
            )
        except Exception as exc:
            logger.warning("Version space belief update failed: %s", exc)

    def _record_efe(
        self,
        intervention: Any,
        outcome: int,
        hypotheses: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Record EFE / epistemic gain for this intervention."""
        if not hasattr(self.strategist, "record_efe_outcome"):
            return None
        try:
            h1_id = intervention.metadata.get("h1", "")
            h2_id = intervention.metadata.get("h2", "")
            h1 = next((h for h in hypotheses if getattr(h, "id", "") == h1_id), None)
            h2 = next((h for h in hypotheses if getattr(h, "id", "") == h2_id), None)
            if h1 is None or h2 is None:
                return None
            return self.strategist.record_efe_outcome(intervention, outcome, h1, h2)
        except Exception as exc:
            logger.debug("EFE recording failed: %s", exc)
            return None

    def _persist_belief_history(self) -> None:
        """Store belief history in session memory for analysis."""
        try:
            vs = self.version_space
            self.session_memory.set_metadata(
                self.campaign_id,
                {
                    "entropy_history": self._entropy_history,
                    "efe_log": self._efe_log,
                    "total_iterations": self.iteration,
                    # Fix 4: posterior diagnostics
                    "belief_history": self._belief_history,
                    "posterior_history": vs.posterior_history,
                    "topk_posterior_traces": vs.topk_posterior_traces,
                    "holdout_accuracy_history": vs.holdout_accuracy_history,
                    "vs_state": vs.to_dict(),
                },
            )
        except Exception as exc:
            logger.debug("Failed to persist belief history: %s", exc)

    # ------------------------------------------------------------------
    # Holdout evaluation
    # ------------------------------------------------------------------

    def _stratified_holdout_split(
        self,
        episodes: List[Tuple[str, int]],
        test_size: float = 0.2,
        min_holdout: int = 3,
    ) -> Optional[Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]]:
        """Stratified train/holdout split preserving outcome distribution.

        Ensures both train and holdout contain at least one example of each
        outcome (ACCEPT=0, REFUSE=1) when possible, preventing degenerate
        splits where holdout is all REFUSE or all ACCEPT.

        Returns (train, holdout) or None when impossible.
        """
        import random as _random
        refuse = [(p, o) for p, o in episodes if o == 1]
        accept = [(p, o) for p, o in episodes if o == 0]
        _random.shuffle(refuse)
        _random.shuffle(accept)

        n_refuse = len(refuse)
        n_accept = len(accept)

        # When both outcomes exist, preserve ratio in both splits
        if n_refuse > 0 and n_accept > 0:
            def _split_by_outcome(items, n_total):
                n_test = max(1, int(len(items) * test_size))
                return items[:n_test], items[n_test:]

            r_test, r_train = _split_by_outcome(refuse, n_refuse)
            a_test, a_train = _split_by_outcome(accept, n_accept)
            holdout = r_test + a_test
            train = r_train + a_train
            _random.shuffle(holdout)
            _random.shuffle(train)
            if len(holdout) < min_holdout:
                return None
            return train, holdout

        # Single outcome: random split with coverage warning
        if len(episodes) >= min_holdout * 3:
            _random.shuffle(episodes)
            split = int(len(episodes) * (1.0 - test_size))
            return episodes[:split], episodes[split:]

        return None

    def _load_holdout_prompts(self) -> List[Tuple[str, int]]:
        """Load holdout prompts from external CSV if configured."""
        if not self.holdout_prompts_path:
            return []
        if self._holdout_prompts_cache is not None:
            return self._holdout_prompts_cache
        try:
            import csv
            prompts: List[Tuple[str, int]] = []
            with open(self.holdout_prompts_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    p = row.get("prompt", "").strip()
                    label_str = row.get("label", "0").strip()
                    if p:
                        prompts.append((p, int(label_str)))
            if prompts:
                logger.info(
                    "Loaded %d external holdout prompts from %s",
                    len(prompts), self.holdout_prompts_path,
                )
            self._holdout_prompts_cache = prompts
            return prompts
        except Exception as exc:
            logger.warning("Failed to load holdout prompts from %s: %s",
                           self.holdout_prompts_path, exc)
            return []

    def evaluate_on_holdout(
        self,
        holdout_prompts: List[Tuple[str, int]],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate ALL candidates on a holdout set to detect overfitting.

        **Fix 2 (Stratified Split)**: Uses ``_stratified_holdout_split``
        to preserve outcome distribution in train/holdout.

        **Fix 2 (Per-Candidate Logging)**: Logs train accuracy, holdout
        accuracy, and generalization gap for *every* candidate.

        Every candidate in the version space gets ``holdout_accuracy``,
        ``train_accuracy`` and ``generalization_gap`` stored directly on
        the object so that ``holdout_adjusted_score()`` and
        ``update_belief()`` always have real holdout data to work with.

        If no holdout set is provided, splits available episodes into
        train/holdout using stratified sampling and recomputes both
        accuracies.

        Returns a dict with aggregated results, or None if insufficient
        data.
        """
        if self.version_space.is_empty:
            return None

        executor = ProgramExecutor(
            getattr(self.strategist, "primitive_registry", None),
        )

        # If surrogate has been trained, use it to add EIG-prioritized
        # holdout evaluation — evaluate prompts with highest disagreement first
        if hasattr(self, "surrogate") and self.surrogate._is_trained:
            vs = self.version_space
            if vs.num_candidates >= 2:
                exec_preds = {
                    c.program_id: int(executor.execute(c.program, ""))
                    for c in vs.candidates
                }
                # Compute disagreement for each holdout prompt
                holdout_disagreement = {}
                for p, _ in holdout_prompts:
                    preds = [int(executor.execute(c.program, p)) for c in vs.candidates]
                    n_refuse = sum(1 for pr in preds if pr == 1)
                    n_tot = len(preds)
                    p_r = n_refuse / max(n_tot, 1)
                    if 0 < p_r < 1:
                        disc = -p_r * math.log(p_r) - (1 - p_r) * math.log(1 - p_r)
                        holdout_disagreement[p] = disc / math.log(2)
                    else:
                        holdout_disagreement[p] = 0.0
                # Sort holdout by disagreement (descending) to prioritize
                holdout_prompts = sorted(holdout_prompts, key=lambda x: -holdout_disagreement.get(x[0], 0.0))

        # Semantic fix 2: Use external holdout prompts if configured
        if not holdout_prompts:
            external = self._load_holdout_prompts()
            if external:
                holdout_prompts = external

        if not holdout_prompts:
            episodes = self._fetch_episodes()
            if len(episodes) >= 5:
                split_result = self._stratified_holdout_split(episodes, test_size=0.2, min_holdout=3)
                if split_result is not None:
                    train, holdout_prompts = split_result
                else:
                    import random as _random
                    _random.shuffle(episodes)
                    split = int(len(episodes) * 0.8)
                    train = episodes[:split]
                    holdout_prompts = episodes[split:]
            else:
                logger.info("Too few episodes (%d) for holdout split", len(episodes))
                return None
        else:
            train = []

        best_id = None
        best_holdout = 0.0
        best_train = 0.0
        best_gap = 0.0
        evaluated_count = 0
        per_candidate_log: List[Dict[str, Any]] = []

        for candidate in self.version_space.candidates:
            try:
                if train:
                    train_acc = self._compute_accuracy(candidate.program, train, executor)
                else:
                    train_acc = candidate.accuracy
                hold_acc = self._compute_accuracy(candidate.program, holdout_prompts, executor)
                gap = train_acc - hold_acc

                candidate.holdout_accuracy = hold_acc
                candidate.train_accuracy = train_acc
                candidate.generalization_gap = gap

                per_candidate_log.append({
                    "program_id": candidate.program_id,
                    "predicate_type": candidate.predicate_type,
                    "complexity": candidate.complexity,
                    "train_acc": round(train_acc, 4),
                    "holdout_acc": round(hold_acc, 4),
                    "gap": round(gap, 4),
                })

                if hold_acc > best_holdout:
                    best_holdout = hold_acc
                    best_train = train_acc
                    best_gap = gap
                    best_id = candidate.program_id

                evaluated_count += 1
            except Exception as exc:
                logger.debug("Holdout eval failed for %s: %s", candidate.program_id, exc)
                continue

        if best_id is None:
            return None

        # Log per-candidate details
        log_lines = [
            f"  {c['program_id'][:12]} type={c['predicate_type']} "
            f"train={c['train_acc']:.3f} hold={c['holdout_acc']:.3f} "
            f"gap={c['gap']:.3f} complex={c['complexity']}"
            for c in per_candidate_log[:20]
        ]
        logger.info(
            "Holdout evaluation: evaluated %d/%d candidates, "
            "best=%s train=%.3f holdout=%.3f gap=%.3f (holdout_n=%d)\n%s",
            evaluated_count, len(self.version_space.candidates),
            best_id, best_train, best_holdout, best_gap,
            len(holdout_prompts),
            "\n".join(log_lines),
        )

        return {
            "best_program_id": best_id,
            "train_accuracy": best_train,
            "holdout_accuracy": best_holdout,
            "generalization_gap": best_gap,
            "holdout_size": len(holdout_prompts),
            "evaluated_count": evaluated_count,
            "per_candidate": per_candidate_log,
        }

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _result(
        self,
        success: bool = False,
        best_program_id: Optional[str] = None,
        best_accuracy: float = 0.0,
        total_interventions: int = 0,
        error: Optional[str] = None,
        converged_by: str = "unknown",
    ) -> Dict[str, Any]:
        best = self.version_space.most_likely()
        return {
            "success": success,
            "campaign_id": self.campaign_id,
            "best_program_id": best_program_id or (best.program_id if best else None),
            "best_predicate_type": best.predicate_type if best else None,
            "best_accuracy": best_accuracy or (best.accuracy if best else 0.0),
            "total_interventions": total_interventions,
            "total_iterations": self.iteration,
            "belief_entropy_history": self._entropy_history,
            "efe_log": self._efe_log,
            "version_space": self.version_space.to_dict(),
            "num_candidates": self.version_space.num_candidates,
            "converged_by": converged_by,
            "error": error,
            "telemetry": {
                "seed": self._seed_telemetry,
                "anomaly": self._anomaly_telemetry,
            },
        }
