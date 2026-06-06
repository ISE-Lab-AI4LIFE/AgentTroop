"""Tests for Orchestrator."""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agents.cognitive import Anomaly, CognitiveAgent, Hypothesis
from agents.researcher import ResearcherAgent
from agents.strategist import StrategistAgent
from core.intervention import Intervention
from knowledge.episodic import EpisodicMemory
from orchestration import Orchestrator, OrchestratorPhase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_anomaly(
    base_prompt: str = "bomb",
    diff: float = 1.0,
) -> Anomaly:
    return Anomaly(
        id=f"anom_{base_prompt}",
        base_prompt=base_prompt,
        transform_names=["rot13"],
        outcome_original=1,
        outcome_transformed=0,
        difference=diff,
        episode_id_original="ep_1",
        episode_id_transformed="ep_2",
    )


def _make_mock_hypothesis(
    description: str = "keyword filter",
    condition: str = "IF contains_word('bomb') THEN REFUSE",
    confidence: float = 0.6,
) -> Hypothesis:
    return Hypothesis(
        description=description,
        condition=condition,
        confidence=confidence,
        supporting_anomaly_ids=["anom_bomb"],
    )


def _make_mock_intervention() -> Intervention:
    return Intervention(base_prompt="bomb test", transforms=[])


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def cognitive() -> MagicMock:
    return MagicMock(spec=CognitiveAgent)


@pytest.fixture
def strategist() -> MagicMock:
    return MagicMock(spec=StrategistAgent)


@pytest.fixture
def researcher() -> MagicMock:
    return MagicMock(spec=ResearcherAgent)


@pytest.fixture
def memory() -> MagicMock:
    return MagicMock(spec=EpisodicMemory)


@pytest.fixture
def orchestrator(
    cognitive: MagicMock,
    strategist: MagicMock,
    researcher: MagicMock,
    memory: MagicMock,
) -> Orchestrator:
    return Orchestrator(
        cognitive_agent=cognitive,
        strategist_agent=strategist,
        researcher_agent=researcher,
        episodic_memory=memory,
        max_iterations=3,
        convergence_threshold=0.05,
    )


# ===================================================================
# Constructor
# ===================================================================


class TestConstructor:
    def test_default_params(self, memory: MagicMock) -> None:
        cog = MagicMock(spec=CognitiveAgent)
        strat = MagicMock(spec=StrategistAgent)
        res = MagicMock(spec=ResearcherAgent)
        orch = Orchestrator(
            cognitive_agent=cog,
            strategist_agent=strat,
            researcher_agent=res,
            episodic_memory=memory,
        )
        assert orch.max_iterations == 10
        assert orch.convergence_threshold == 0.05
        assert orch.phase == OrchestratorPhase.IDLE

    def test_custom_params(self, memory: MagicMock) -> None:
        cog = MagicMock(spec=CognitiveAgent)
        strat = MagicMock(spec=StrategistAgent)
        res = MagicMock(spec=ResearcherAgent)
        orch = Orchestrator(
            cognitive_agent=cog,
            strategist_agent=strat,
            researcher_agent=res,
            episodic_memory=memory,
            max_iterations=5,
            convergence_threshold=0.1,
        )
        assert orch.max_iterations == 5
        assert orch.convergence_threshold == 0.1

    def test_max_iterations_at_least_one(self, memory: MagicMock) -> None:
        cog = MagicMock(spec=CognitiveAgent)
        strat = MagicMock(spec=StrategistAgent)
        res = MagicMock(spec=ResearcherAgent)
        orch = Orchestrator(
            cognitive_agent=cog,
            strategist_agent=strat,
            researcher_agent=res,
            episodic_memory=memory,
            max_iterations=0,
        )
        assert orch.max_iterations == 1


# ===================================================================
# run_pipeline
# ===================================================================


class TestRunPipeline:
    def test_basic_flow(self, orchestrator: Orchestrator) -> None:
        anomalies = [_make_mock_anomaly()]
        hypotheses = [_make_mock_hypothesis()]

        orchestrator.cognitive.detect_anomalies.return_value = anomalies
        orchestrator.cognitive.generate_hypotheses.return_value = hypotheses
        orchestrator.strategist.design_intervention.return_value = (
            _make_mock_intervention()
        )
        orchestrator.researcher.run_reverse_engineering_pipeline.return_value = {
            "success": True,
            "program_id": "prog_001",
            "theory_id": "thr_001",
        }

        result = orchestrator.run_pipeline(
            campaign_id="camp_test",
            victim=MagicMock(),
        )

        assert result["success"] is True
        assert result["campaign_id"] == "camp_test"
        assert result["iterations"] >= 1
        assert "prog_001" in result["programs"]
        assert "thr_001" in result["theories"]
        assert len(result["anomalies"]) >= 1

    def test_no_anomalies(self, orchestrator: Orchestrator) -> None:
        orchestrator.cognitive.detect_anomalies.return_value = []

        result = orchestrator.run_pipeline(
            campaign_id="camp_empty",
            victim=MagicMock(),
        )

        assert result["success"] is True
        assert result["anomalies"] == []
        assert result["hypotheses"] == []

    def test_strategist_produces_interventions(
        self, orchestrator: Orchestrator,
    ) -> None:
        anomalies = [_make_mock_anomaly()]
        hypotheses = [_make_mock_hypothesis()]

        orchestrator.cognitive.detect_anomalies.return_value = anomalies
        orchestrator.cognitive.generate_hypotheses.return_value = hypotheses
        orchestrator.strategist.design_intervention.return_value = (
            _make_mock_intervention()
        )
        orchestrator.researcher.run_reverse_engineering_pipeline.return_value = {
            "success": True,
        }

        result = orchestrator.run_pipeline(
            campaign_id="camp_int",
            victim=MagicMock(),
        )

        assert result["success"] is True
        orchestrator.strategist.design_intervention.assert_called()
        orchestrator.strategist.execute_intervention.assert_called()

    def test_convergence_stops_early(self, orchestrator: Orchestrator) -> None:
        anomalies = [_make_mock_anomaly()]
        hypotheses = [_make_mock_hypothesis()]

        orchestrator.cognitive.detect_anomalies.return_value = anomalies
        orchestrator.cognitive.generate_hypotheses.return_value = hypotheses
        orchestrator.strategist.design_intervention.return_value = (
            _make_mock_intervention()
        )
        orchestrator.researcher.run_reverse_engineering_pipeline.return_value = {
            "success": True,
            "program_id": "prog_conv",
            "theory_id": "thr_conv",
        }

        with patch.object(orchestrator, "_check_convergence", return_value=True):
            result = orchestrator.run_pipeline(
                campaign_id="camp_conv",
                victim=MagicMock(),
            )

        assert result["success"] is True
        assert result["programs"] == ["prog_conv"]

    def test_exception_handling(self, orchestrator: Orchestrator) -> None:
        orchestrator.cognitive.detect_anomalies.side_effect = RuntimeError(
            "cognitive failure",
        )

        result = orchestrator.run_pipeline(
            campaign_id="camp_err",
            victim=MagicMock(),
        )

        assert result["success"] is False
        assert result["error"] is not None
        assert "cognitive failure" in result["error"]

    def test_anomalies_detected_but_no_hypotheses(
        self, orchestrator: Orchestrator,
    ) -> None:
        orchestrator.cognitive.detect_anomalies.return_value = [
            _make_mock_anomaly(),
        ]
        orchestrator.cognitive.generate_hypotheses.return_value = []

        result = orchestrator.run_pipeline(
            campaign_id="camp_no_hyp",
            victim=MagicMock(),
        )

        assert result["success"] is True
        assert len(result["anomalies"]) >= 1
        assert result["hypotheses"] == []


# ===================================================================
# _check_convergence
# ===================================================================


class TestCheckConvergence:
    def test_not_converged_first_iteration(self, orchestrator: Orchestrator) -> None:
        orchestrator.iteration = 1
        assert orchestrator._check_convergence({}, []) is False

    def test_converged_with_program_no_anomalies(
        self, orchestrator: Orchestrator,
    ) -> None:
        orchestrator.iteration = 3
        result = {"programs": ["prog_1"]}
        assert orchestrator._check_convergence(result, []) is True

    def test_not_converged_with_anomalies(
        self, orchestrator: Orchestrator,
    ) -> None:
        orchestrator.iteration = 3
        result = {"programs": ["prog_1"]}
        anomalies = [_make_mock_anomaly()]
        assert orchestrator._check_convergence(result, anomalies) is False

    def test_not_converged_no_program(self, orchestrator: Orchestrator) -> None:
        orchestrator.iteration = 3
        result = {"programs": []}
        assert orchestrator._check_convergence(result, []) is False


# ===================================================================
# _phase_strategist
# ===================================================================


class TestPhaseStrategist:
    def test_multiple_hypotheses_paired(self, orchestrator: Orchestrator) -> None:
        hypotheses = [
            _make_mock_hypothesis(description=f"hyp_{i}")
            for i in range(3)
        ]
        orchestrator.strategist.design_intervention.return_value = (
            _make_mock_intervention()
        )

        interventions = orchestrator._phase_strategist(
            hypotheses=hypotheses,
            campaign_id="camp_pairs",
            experiment_id=None,
            victim=MagicMock(),
        )

        assert len(interventions) >= 1
        assert orchestrator.strategist.design_intervention.call_count >= 1

    def test_single_hypothesis_uses_default(
        self, orchestrator: Orchestrator,
    ) -> None:
        hypotheses = [_make_mock_hypothesis()]
        orchestrator.strategist.design_intervention.return_value = (
            _make_mock_intervention()
        )

        interventions = orchestrator._phase_strategist(
            hypotheses=hypotheses,
            campaign_id="camp_single",
            experiment_id="exp_single",
            victim=MagicMock(),
        )

        assert len(interventions) >= 1

    def no_hypotheses(self, orchestrator: Orchestrator) -> None:
        interventions = orchestrator._phase_strategist(
            hypotheses=[],
            campaign_id="camp_empty",
            experiment_id=None,
            victim=MagicMock(),
        )
        assert interventions == []
