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

import json
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agents.cognitive import Anomaly, CognitiveAgent, Hypothesis
from agents.researcher import ResearcherAgent
from agents.strategist import StrategistAgent
from core.intervention import Intervention
from knowledge.manager import KnowledgeManager, Target
from knowledge.session_memory import SessionMemory

logger = logging.getLogger(__name__)

_SYNTHESIS_INTERVAL = 5


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

        self.phase = OrchestratorPhase.IDLE
        self.iteration = 0

        self._session_created = self._ensure_session()
        logger.info(
            "Orchestrator V2 initialised: campaign=%s model=%s "
            "max_iter=%d max_intv=%d acc_thresh=%.2f",
            campaign_id, getattr(victim, "name", str(victim)),
            self.max_iterations, self.max_interventions,
            self.accuracy_threshold,
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
        """Execute the full 6-phase loop and return results.

        Returns
        -------
        dict with keys:
            - success (bool)
            - campaign_id (str)
            - best_program_id (str or None)
            - best_accuracy (float)
            - total_interventions (int)
            - total_iterations (int)
            - error (str or None)
        """
        start_time = time.time()
        self.iteration = 0
        total_interventions = 0

        try:
            # Phase 1-2: Cognitive Agent
            logger.info("=== Campaign %s: Phase 1-2 (Cognitive) ===", self.campaign_id)
            anomalies, hypotheses = self._run_cognitive_phase()
            if not hypotheses:
                return self._result(
                    success=False,
                    error="No hypotheses generated from anomalies",
                    total_interventions=total_interventions,
                )

            # Phase 3-6: Strategist + Researcher loop
            for iteration in range(1, self.max_iterations + 1):
                self.iteration = iteration
                logger.info(
                    "=== Iteration %d/%d (campaign=%s) ===",
                    iteration, self.max_iterations, self.campaign_id,
                )

                if total_interventions >= self.max_interventions:
                    logger.info("Intervention budget exhausted (%d)", total_interventions)
                    break

                # Phase 3-4: Strategist Agent
                intervention = self._run_strategist_phase(hypotheses)
                if intervention is None:
                    logger.info("No discriminating intervention found; stopping")
                    break

                # Execute and store
                outcome = self.strategist.execute_intervention(intervention, self.victim)
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

                # Phase 5-6: Researcher Agent (periodic)
                if total_interventions % self.synthesis_interval == 0:
                    result = self.researcher.run_reverse_engineering_pipeline(
                        campaign_id=self.campaign_id,
                        victim=self.victim,
                        allow_error_rate=self.allow_error_rate,
                        accuracy_threshold=self.accuracy_threshold,
                        experiment_id=self.experiment_id,
                    )
                    if result.get("success") and result.get("program_id"):
                        accuracy = result.get("accuracy", 0.0)
                        self.session_memory.set_best_program(
                            self.campaign_id,
                            result["program_id"],
                            accuracy,
                        )
                        if accuracy >= self.accuracy_threshold:
                            self.session_memory.set_status(
                                self.campaign_id, "completed",
                            )
                            logger.info(
                                "Accuracy %.2f >= threshold %.2f; converged",
                                accuracy, self.accuracy_threshold,
                            )
                            return self._result(
                                success=True,
                                best_program_id=result["program_id"],
                                best_accuracy=accuracy,
                                total_interventions=total_interventions,
                            )

            # Max iterations reached without convergence
            self.session_memory.set_status(self.campaign_id, "completed")
            final = self.session_memory.get_session(self.campaign_id) or {}
            return self._result(
                success=True,
                best_program_id=final.get("current_best_program_id"),
                best_accuracy=float(final.get("current_best_accuracy", 0.0)),
                total_interventions=total_interventions,
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
                "iterations=%d interventions=%d",
                self.campaign_id, elapsed,
                self.iteration, total_interventions,
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
        hypotheses = self.cognitive.generate_hypotheses(
            anomalies,
            prior_hypotheses=prior,
        )

        for h in hypotheses:
            try:
                self.session_memory.add_hypothesis(self.campaign_id, h.id)
            except Exception:
                pass

        return anomalies, hypotheses

    def _load_prior_hypotheses(self) -> Optional[List[Hypothesis]]:
        try:
            hyp_ids = self.session_memory.list_hypotheses(self.campaign_id)
            if not hyp_ids:
                return None
        except Exception:
            return None
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
            intervention.metadata.setdefault("h1", h1)
            intervention.metadata.setdefault("h2", h2)

        return intervention

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

    def _result(
        self,
        success: bool = False,
        best_program_id: Optional[str] = None,
        best_accuracy: float = 0.0,
        total_interventions: int = 0,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "success": success,
            "campaign_id": self.campaign_id,
            "best_program_id": best_program_id,
            "best_accuracy": best_accuracy,
            "total_interventions": total_interventions,
            "total_iterations": self.iteration,
            "error": error,
        }
