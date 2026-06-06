"""Orchestrator — coordinates the 6-phase HARMONY-X exploration loop.

Phases
-------
1. Cognitive Agent — detect anomalies from Episodic Memory
2. Cognitive Agent — generate structural hypotheses from anomalies
3. Strategist Agent — design optimal intervention for hypothesis pair
4. Strategist Agent — execute intervention, record outcome in Episodic Memory
5. Researcher Agent — synthesise program from all episodes
6. Researcher Agent — verify program, store in Defense Program Store + theory

Design (HARMONY-X §5.4)
------------------------
The Orchestrator manages the interaction between Cognitive, Strategist,
and Researcher agents, maintaining a state machine with checkpoint support.
It iterates the 6-phase loop until convergence or max_iterations.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from agents.cognitive import Anomaly, CognitiveAgent, Hypothesis
from agents.researcher import ResearcherAgent
from agents.strategist import StrategistAgent
from core.intervention import Intervention
from knowledge.episodic import EpisodicMemory

logger = logging.getLogger(__name__)


class OrchestratorPhase(Enum):
    """Current phase of the Orchestrator's state machine."""
    IDLE = 0
    ANOMALY_DETECTION = 1
    HYPOTHESIS_GENERATION = 2
    INTERVENTION_DESIGN = 3
    INTERVENTION_EXECUTION = 4
    PROGRAM_SYNTHESIS = 5
    VERIFICATION_AND_STORE = 6
    CONVERGED = 7


class Orchestrator:
    """Coordinates the 6-phase HARMONY-X exploration loop.

    Parameters
    ----------
    cognitive_agent : CognitiveAgent
        Agent for anomaly detection and hypothesis generation.
    strategist_agent : StrategistAgent
        Agent for intervention design and execution.
    researcher_agent : ResearcherAgent
        Agent for program synthesis, verification, and theory extraction.
    episodic_memory : EpisodicMemory
        Central episode store (L1).
    max_iterations : int
        Maximum number of full 6-phase iterations (default 10).
    convergence_threshold : float
        Minimum improvement in hypothesis confidence to continue iterating
        (default 0.05).
    """

    def __init__(
        self,
        cognitive_agent: CognitiveAgent,
        strategist_agent: StrategistAgent,
        researcher_agent: ResearcherAgent,
        episodic_memory: EpisodicMemory,
        max_iterations: int = 10,
        convergence_threshold: float = 0.05,
    ) -> None:
        self.cognitive = cognitive_agent
        self.strategist = strategist_agent
        self.researcher = researcher_agent
        self.episodic_memory = episodic_memory
        self.max_iterations = max(1, max_iterations)
        self.convergence_threshold = convergence_threshold

        self.phase = OrchestratorPhase.IDLE
        self.iteration = 0

        logger.info(
            "Orchestrator initialised (max_iterations=%d, convergence=%.2f)",
            self.max_iterations, self.convergence_threshold,
        )

    def run_pipeline(
        self,
        campaign_id: str,
        victim: Any,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full 6-phase HARMONY-X pipeline.

        Parameters
        ----------
        campaign_id : str
            Campaign identifier for this run.
        victim : Any
            Victim LLM instance (must have ``respond(prompt) -> Outcome``).
        experiment_id : str, optional
            Specific experiment to focus on.

        Returns
        -------
        dict with keys:
            - success (bool)
            - campaign_id (str)
            - iterations (int)
            - hypotheses (list of Hypothesis dicts)
            - programs (list of program IDs)
            - theories (list of theory IDs)
            - anomalies (list of Anomaly dicts)
            - interventions (list of Intervention IDs)
            - error (str or None)
        """
        start_time = time.time()
        self.iteration = 0

        result: Dict[str, Any] = {
            "success": False,
            "campaign_id": campaign_id,
            "iterations": 0,
            "hypotheses": [],
            "programs": [],
            "theories": [],
            "anomalies": [],
            "interventions": [],
            "error": None,
        }

        try:
            for iteration in range(1, self.max_iterations + 1):
                self.iteration = iteration
                logger.info(
                    "=== Pipeline iteration %d/%d (campaign=%s) ===",
                    iteration, self.max_iterations, campaign_id,
                )

                # Phase 1-2: Cognitive Agent
                anomalies = self._phase_anomaly_detection(
                    campaign_id, experiment_id,
                )
                result["anomalies"].extend(anomalies)

                if not anomalies:
                    logger.info("No anomalies detected; moving to next iteration")
                    continue

                hypotheses = self._phase_hypothesis_generation(
                    anomalies, result["hypotheses"],
                )
                result["hypotheses"].extend([h.to_dict() for h in hypotheses])

                if not hypotheses:
                    continue

                # Phase 3-4: Strategist Agent
                interventions = self._phase_strategist(
                    hypotheses, campaign_id, experiment_id, victim,
                )
                result["interventions"].extend(
                    inv.id for inv in interventions
                )

                # Phase 5-6: Researcher Agent
                pipeline_result = self._phase_researcher(
                    campaign_id, victim, experiment_id,
                )
                if pipeline_result.get("program_id"):
                    result["programs"].append(pipeline_result["program_id"])
                if pipeline_result.get("theory_id"):
                    result["theories"].append(pipeline_result["theory_id"])

                # Convergence check
                if self._check_convergence(result, anomalies):
                    logger.info(
                        "Pipeline converged after %d iterations", iteration,
                    )
                    break

            result["success"] = True

        except Exception as exc:
            logger.exception(
                "Pipeline failed at iteration %d, phase %s",
                self.iteration, self.phase.name,
            )
            result["error"] = str(exc)

        result["iterations"] = self.iteration
        self.phase = OrchestratorPhase.CONVERGED

        elapsed = time.time() - start_time
        logger.info(
            "Pipeline completed in %.1fs: campaign=%s iterations=%d "
            "hypotheses=%d programs=%d theories=%d success=%s",
            elapsed, campaign_id, self.iteration,
            len(result["hypotheses"]),
            len(result["programs"]),
            len(result["theories"]),
            result["success"],
        )

        return result

    def _phase_anomaly_detection(
        self,
        campaign_id: str,
        experiment_id: Optional[str],
    ) -> List[Anomaly]:
        """Phase 1: detect anomalies via Cognitive Agent."""
        self.phase = OrchestratorPhase.ANOMALY_DETECTION
        logger.info("Phase 1/6: anomaly detection")
        return self.cognitive.detect_anomalies(
            campaign_id=campaign_id,
            experiment_id=experiment_id,
        )

    def _phase_hypothesis_generation(
        self,
        anomalies: List[Anomaly],
        prior_dicts: List[Dict[str, Any]],
    ) -> List[Hypothesis]:
        """Phase 2: generate hypotheses from anomalies."""
        self.phase = OrchestratorPhase.HYPOTHESIS_GENERATION
        logger.info("Phase 2/6: hypothesis generation (%d anomalies)", len(anomalies))

        prior = None
        if prior_dicts:
            prior = [
                Hypothesis(
                    id=d.get("id", ""),
                    description=d.get("description", ""),
                    condition=d.get("condition", ""),
                    confidence=d.get("confidence", 0.0),
                    supporting_anomaly_ids=d.get("supporting_anomaly_ids", []),
                    created_at=d.get("created_at", 0.0),
                )
                for d in prior_dicts[-5:]
            ]

        return self.cognitive.generate_hypotheses(
            anomalies,
            prior_hypotheses=prior,
        )

    def _phase_strategist(
        self,
        hypotheses: List[Hypothesis],
        campaign_id: str,
        experiment_id: Optional[str],
        victim: Any,
    ) -> List[Intervention]:
        """Phase 3-4: design and execute interventions for hypothesis pairs."""
        self.phase = OrchestratorPhase.INTERVENTION_DESIGN
        logger.info(
            "Phase 3-4/6: strategist (%d hypotheses)", len(hypotheses),
        )

        interventions: List[Intervention] = []

        if len(hypotheses) < 2:
            bare_hyp = Hypothesis(
                description="Default: always ACCEPT",
                condition="",
            )
            pairs = [(hypotheses[0], bare_hyp)]
        else:
            pairs = []
            for i in range(min(3, len(hypotheses))):
                for j in range(i + 1, min(3, len(hypotheses))):
                    pairs.append((hypotheses[i], hypotheses[j]))

        self.phase = OrchestratorPhase.INTERVENTION_EXECUTION

        for h1, h2 in pairs:
            intervention = self.strategist.design_intervention(
                hypothesis_a=h1, hypothesis_b=h2,
            )
            if intervention is not None:
                exp_id = experiment_id or f"exp_strategist_{self.iteration}"
                self.strategist.execute_intervention(
                    intervention=intervention,
                    victim=victim,
                    campaign_id=campaign_id,
                    experiment_id=exp_id,
                    hypothesis_id=getattr(h1, "id", None),
                )
                interventions.append(intervention)

        return interventions

    def _phase_researcher(
        self,
        campaign_id: str,
        victim: Any,
        experiment_id: Optional[str],
    ) -> Dict[str, Any]:
        """Phase 5-6: synthesise and verify programs via Researcher Agent."""
        self.phase = OrchestratorPhase.PROGRAM_SYNTHESIS
        logger.info("Phase 5-6/6: researcher pipeline")

        self.phase = OrchestratorPhase.VERIFICATION_AND_STORE
        return self.researcher.run_reverse_engineering_pipeline(
            campaign_id=campaign_id,
            victim=victim,
            experiment_id=experiment_id,
        )

    def _check_convergence(
        self,
        result: Dict[str, Any],
        anomalies: List[Anomaly],
    ) -> bool:
        """Check if the pipeline has converged.

        Convergence: a verified program exists AND no new anomalies were
        detected in this iteration.
        """
        if self.iteration <= 1:
            return False
        if result["programs"] and not anomalies:
            return True
        return False
