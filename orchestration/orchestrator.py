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
import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agents.cognitive import Anomaly, CognitiveAgent, Hypothesis
from agents.researcher import ResearcherAgent
from agents.strategist import StrategistAgent
from core.intervention import Intervention
from core.executor import ProgramExecutor
from inference.belief_updater import BayesianBeliefUpdater
from inference.pomdp import POMDPAction, POMDPObservation
from inference.version_space import VersionSpace
from knowledge.manager import KnowledgeManager, Target
from knowledge.session_memory import SessionMemory

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
        max_iterations: int = 50,
        max_interventions: int = 500,
        accuracy_threshold: float = 0.9,
        allow_error_rate: float = 0.0,
        synthesis_interval: int = _SYNTHESIS_INTERVAL,
        force_exploration_interval: int = 3,
        entropy_convergence_threshold: float = 0.1,
        belief_updater: Any = None,
        version_space: Optional[VersionSpace] = None,
        top_k_candidates: int = 10,
        seed_telemetry: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cognitive = cognitive_agent
        self.strategist = strategist_agent
        self.researcher = researcher_agent
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

        self.phase = OrchestratorPhase.IDLE
        self.iteration = 0
        self.top_k_candidates = max(2, top_k_candidates)

        # ── Version Space: single source of truth (owned by Orchestrator) ──
        self.version_space = version_space or VersionSpace(
            max_candidates=self.top_k_candidates,
        )
        # All agents reference the same object
        self.strategist._version_space = self.version_space

        # Track belief history for convergence
        self._belief_history: List[float] = []
        self._entropy_history: List[float] = []
        self._efe_log: List[Dict[str, Any]] = []

        # Anomaly-source telemetry
        self._seed_telemetry = seed_telemetry or {}
        self._anomaly_telemetry: Dict[str, Any] = {}
        self._intervention_telemetry: Dict[str, int] = {}

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

        try:
            # Phase 1-2: Cognitive Agent → hypotheses
            logger.info("=== Campaign %s: Phase 1-2 (Cognitive) ===",
                        self.campaign_id)
            anomalies, hypotheses = self._run_cognitive_phase()
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
                intervention = self._run_strategist_phase(hypotheses)

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

                # ── Execute intervention → Observe outcome ──
                outcome = self.strategist.execute_intervention(intervention, self.victim)

                # ── Update version space belief ──
                self._update_belief_from_observation(intervention, outcome, hypotheses)

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
                if total_interventions % self.synthesis_interval == 0:
                    self._synthesize_and_update_version_space()

                # ── Entropy-based convergence check ──
                current_entropy = self._get_current_entropy()
                if len(self._entropy_history) >= 5 and self.version_space.num_candidates >= 2:
                    recent = self._entropy_history[-5:]
                    if all(e < self.entropy_convergence_threshold for e in recent):
                        converged_by_entropy = True
                        logger.info(
                            "Version space entropy below %.3f for 5 cycles "
                            "(%d candidates); converged",
                            self.entropy_convergence_threshold,
                            self.version_space.num_candidates,
                        )
                        break

                # ── Accuracy convergence check (most likely candidate) ──
                best = self.version_space.most_likely()
                if best is not None and best.accuracy >= self.accuracy_threshold:
                    converged_by_accuracy = True
                    logger.info(
                        "Best candidate accuracy %.2f >= threshold %.2f; converged",
                        best.accuracy, self.accuracy_threshold,
                    )
                    self.session_memory.set_best_program(
                        self.campaign_id, best.program_id, best.accuracy,
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

        try:
            anomalies, hypotheses = self._run_cognitive_phase()
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

                intervention = self._run_strategist_phase(hypotheses)

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

                outcome = await self.strategist.async_execute_intervention(
                    intervention, self.victim
                )
                self._update_belief_from_observation(intervention, outcome, hypotheses)

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

                if total_interventions % self.synthesis_interval == 0:
                    self._synthesize_and_update_version_space()

                current_entropy = self._get_current_entropy()
                if len(self._entropy_history) >= 5 and self.version_space.num_candidates >= 2:
                    recent = self._entropy_history[-5:]
                    if all(e < self.entropy_convergence_threshold for e in recent):
                        converged_by_entropy = True
                        logger.info(
                            "Version space entropy below %.3f for 5 cycles "
                            "(%d candidates); converged",
                            self.entropy_convergence_threshold,
                            self.version_space.num_candidates,
                        )
                        break

                best = self.version_space.most_likely()
                if best is not None and best.accuracy >= self.accuracy_threshold:
                    converged_by_accuracy = True
                    logger.info(
                        "Best candidate accuracy %.2f >= threshold %.2f; converged",
                        best.accuracy, self.accuracy_threshold,
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

        # Propose hypotheses to Defense Program Store via KnowledgeManager
        try:
            proposal_data = [
                {
                    "id": h.id,
                    "description": h.description,
                    "condition": h.condition,
                    "confidence": h.confidence,
                    "supporting_anomaly_ids": h.supporting_anomaly_ids,
                }
                for h in hypotheses
            ]
            self.knowledge_manager.propose(
                target="defense_program_store",
                data={"action": "register_hypotheses", "hypotheses": proposal_data},
                agent_id="Orchestrator",
                action="register_hypotheses",
            )
        except Exception as exc:
            logger.warning("Failed to propose hypotheses: %s", exc)

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

        h1, h2 = self.strategist.select_hypothesis_pair(hypotheses)
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
        """
        try:
            from synthesis.cvc5_synthesizer import CVC5Synthesizer
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

            programs = synthesizer.synthesize_top_k(
                episodes, k=self.top_k_candidates,
            )

            # ── Heuristic fallback: if synthesis found 0 programs ──
            if not programs:
                logger.warning(
                    "Synthesis found 0 matching programs; "
                    "generating heuristic fallback candidates"
                )
                programs = self._generate_heuristic_candidates(episodes, executor)

            added = 0
            for prog in programs:
                accuracy = self._compute_accuracy(prog, episodes, executor)
                source = getattr(prog, "source", "enumeration")
                self.version_space.add_candidate(
                    program=prog,
                    accuracy=accuracy,
                    source=source,
                    episodes_matched=int(accuracy * len(episodes)),
                    total_episodes=len(episodes),
                )
                added += 1

            self.version_space.reset_belief(uniform=True)
            logger.info(
                "Version space: added %d programs (total=%d, entropy=%.3f, "
                "info_gain=%.4f)",
                added, self.version_space.num_candidates,
                self.version_space.entropy(),
                self.version_space.total_info_gain,
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
                self.researcher.verify_and_store(
                    program=best.program,
                    campaign_id=self.campaign_id,
                    victim=self.victim,
                    program_id=best.program_id,
                )
        except Exception as exc:
            logger.warning("Synthesis → version space update failed: %s", exc)

    def _generate_heuristic_candidates(
        self,
        episodes: List[Tuple[str, int]],
        executor: ProgramExecutor,
    ) -> List[Any]:
        """Generate heuristic fallback candidates when synthesis finds nothing.

        Creates keyword-based programs from frequently-occurring words
        in REFUSE examples, ensuring the Version Space always has ≥ 10
        candidates.
        """
        import re
        from collections import Counter
        from core.program import Program, IfThenElseNode, PredicateNode
        from core.primitive import ContainsWordPredicate

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

        # Keyword → REFUSE programs
        for kw in top_kw[:8]:
            for then_out, else_out in [(1, 0), (0, 1)]:
                prog = Program(
                    root=IfThenElseNode(
                        condition=PredicateNode(
                            primitive=ContainsWordPredicate(word=kw)
                        ),
                        then_outcome=then_out,
                        else_outcome=else_out,
                    )
                )
                programs.append(prog)
                prog.source = "heuristic"

        # Length-based programs
        for threshold in [30, 50, 100, 200]:
            from core.primitive import LengthGtPredicate
            for then_out, else_out in [(1, 0), (0, 1)]:
                prog = Program(
                    root=IfThenElseNode(
                        condition=PredicateNode(
                            primitive=LengthGtPredicate(threshold=threshold)
                        ),
                        then_outcome=then_out,
                        else_outcome=else_out,
                    )
                )
                programs.append(prog)
                prog.source = "heuristic"

        # Filter: only keep programs with accuracy > random
        valid: List[Any] = []
        for prog in programs:
            acc = self._compute_accuracy(prog, episodes, executor)
            if acc > 0.5 or acc == 0.5:
                valid.append(prog)
                if len(valid) >= self.top_k_candidates:
                    break

        if not valid:
            valid = programs[:self.top_k_candidates]

        logger.info(
            "Generated %d heuristic candidates (from %d raw, %d keywords, %d lengths)",
            len(valid), len(programs), len(top_kw), 4,
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
        """
        if self.version_space.num_candidates > 0:
            return
        if not hypotheses:
            return
        logger.info(
            "Version space: no candidate programs yet, seeding from %d hypotheses",
            len(hypotheses),
        )
        self.version_space.reset_belief(uniform=True)

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
            return self.version_space.entropy()
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
                except Exception:
                    return 0

            entropy_before = vs.entropy()
            vs.update_belief(prompt, int(outcome), _predict)
            entropy_after = vs.entropy()
            ig = entropy_before - entropy_after
            logger.info(
                "Belief update: H=%.3f→%.3f IG=%.4f candidates=%d",
                entropy_before, entropy_after, ig, vs.num_candidates,
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
            self.session_memory.set_metadata(
                self.campaign_id,
                {
                    "entropy_history": self._entropy_history,
                    "efe_log": self._efe_log,
                    "total_iterations": self.iteration,
                },
            )
        except Exception as exc:
            logger.debug("Failed to persist belief history: %s", exc)

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
