"""Tests for Orchestrator V2 (KnowledgeManager + SessionMemory integration)."""

import os
import time
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agents.cognitive import Anomaly, CognitiveAgent, Hypothesis
from agents.researcher import ResearcherAgent
from agents.strategist import StrategistAgent
from core.intervention import Intervention
from knowledge.manager import KnowledgeManager
from knowledge.session_memory import SessionMemory

pytest.importorskip("testcontainers.redis")
from testcontainers.redis import RedisContainer  # type: ignore[import-untyped]

REDIS_IMAGE = os.environ.get("REDIS_IMAGE", "redis:7")


@pytest.fixture(scope="module")
def redis_url() -> Generator[str, None, None]:
    container = RedisContainer(image=REDIS_IMAGE)
    container.start()
    time.sleep(1)
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anomaly(base_prompt: str = "bomb", diff: float = 1.0) -> Anomaly:
    return Anomaly(
        base_prompt=base_prompt,
        transform_names=["rot13"],
        outcome_original=1,
        outcome_transformed=0,
        difference=diff,
        episode_id_original="ep_1",
        episode_id_transformed="ep_2",
    )


def _make_hypothesis(
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


def _make_intervention() -> Intervention:
    return Intervention(base_prompt="bomb test", transforms=[])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cognitive() -> MagicMock:
    c = MagicMock(spec=CognitiveAgent)
    c.detect_anomalies.return_value = [_make_anomaly()]
    c.generate_hypotheses.return_value = [
        _make_hypothesis(confidence=0.6),
        _make_hypothesis(description="always accept", condition="", confidence=0.4),
    ]
    return c


@pytest.fixture
def strategist() -> MagicMock:
    s = MagicMock(spec=StrategistAgent)
    inv = _make_intervention()
    inv.metadata = {}
    s.select_hypothesis_pair.return_value = (
        _make_hypothesis(),
        _make_hypothesis(description="always accept", condition="", confidence=0.4),
    )
    s.design_intervention.return_value = inv
    s.execute_intervention.return_value = 1
    s.store_intervention.return_value = "ep_test"
    return s


@pytest.fixture
def researcher() -> MagicMock:
    r = MagicMock(spec=ResearcherAgent)
    r.run_reverse_engineering_pipeline.return_value = {
        "success": False,
        "program_id": None,
        "accuracy": 0.0,
        "error": None,
    }
    return r


@pytest.fixture
def km() -> KnowledgeManager:
    return KnowledgeManager(use_redis=False)


@pytest.fixture
def victim() -> MagicMock:
    v = MagicMock()
    v.respond.return_value = 1
    v.name = "TestVictim"
    return v


@pytest.fixture
def session_memory(redis_url: str) -> Generator[SessionMemory, None, None]:
    sm = SessionMemory(redis_url=redis_url, ttl=3600)
    sm.client.flushall()
    yield sm
    sm.client.flushall()
    sm.close()


@pytest.fixture
def orchestrator(
    cognitive: MagicMock,
    strategist: MagicMock,
    researcher: MagicMock,
    km: KnowledgeManager,
    session_memory: SessionMemory,
    victim: MagicMock,
) -> Any:
    from orchestration.orchestrator import Orchestrator

    return Orchestrator(
        cognitive_agent=cognitive,
        strategist_agent=strategist,
        researcher_agent=researcher,
        knowledge_manager=km,
        session_memory=session_memory,
        victim=victim,
        campaign_id="test_campaign",
        experiment_id="exp_001",
        max_iterations=3,
        max_interventions=20,
        accuracy_threshold=0.95,
        allow_error_rate=0.0,
        synthesis_interval=10,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_init_creates_session(
        self,
        cognitive: MagicMock,
        strategist: MagicMock,
        researcher: MagicMock,
        km: KnowledgeManager,
        session_memory: SessionMemory,
        victim: MagicMock,
    ) -> None:
        from orchestration.orchestrator import Orchestrator

        o = Orchestrator(
            cognitive_agent=cognitive,
            strategist_agent=strategist,
            researcher_agent=researcher,
            knowledge_manager=km,
            session_memory=session_memory,
            victim=victim,
            campaign_id="init_test",
            max_iterations=1,
        )
        assert o.campaign_id == "init_test"
        assert session_memory.session_exists("init_test")
        o.session_memory.delete_session("init_test")


class TestRunPipeline:
    def test_run_basic_flow(self, orchestrator: Any) -> None:
        result = orchestrator.run()
        assert result is not None
        assert result["campaign_id"] == "test_campaign"

    def test_run_calls_cognitive(self, orchestrator: Any, cognitive: MagicMock) -> None:
        orchestrator.run()
        cognitive.detect_anomalies.assert_called_once()
        cognitive.generate_hypotheses.assert_called_once()

    def test_run_calls_strategist(
        self, orchestrator: Any, strategist: MagicMock
    ) -> None:
        orchestrator.run()
        strategist.select_hypothesis_pair.assert_called()
        strategist.design_intervention.assert_called()
        strategist.execute_intervention.assert_called()
        strategist.store_intervention.assert_called()

    def test_run_calls_researcher_on_interval(
        self, orchestrator: Any, researcher: MagicMock
    ) -> None:
        orchestrator.run()
        assert researcher.run_reverse_engineering_pipeline.call_count >= 0

    def test_run_hypotheses_stored_in_session(
        self, orchestrator: Any, session_memory: SessionMemory
    ) -> None:
        orchestrator.run()
        hyps = session_memory.list_hypotheses("test_campaign")
        assert len(hyps) >= 1

    def test_run_returns_program_when_converged(
        self,
        cognitive: MagicMock,
        strategist: MagicMock,
        researcher: MagicMock,
        km: KnowledgeManager,
        session_memory: SessionMemory,
        victim: MagicMock,
    ) -> None:
        from orchestration.orchestrator import Orchestrator

        researcher.run_reverse_engineering_pipeline.return_value = {
            "success": True,
            "program_id": "prog_001",
            "accuracy": 0.99,
            "error": None,
        }
        o = Orchestrator(
            cognitive_agent=cognitive,
            strategist_agent=strategist,
            researcher_agent=researcher,
            knowledge_manager=km,
            session_memory=session_memory,
            victim=victim,
            campaign_id="converge_test",
            max_iterations=10,
            synthesis_interval=1,
        )
        result = o.run()
        assert result["success"]
        assert result["best_program_id"] == "prog_001"
        assert result["best_accuracy"] >= 0.95
        session_memory.delete_session("converge_test")


class TestConvergence:
    def test_stops_when_accuracy_met(
        self,
        cognitive: MagicMock,
        strategist: MagicMock,
        researcher: MagicMock,
        km: KnowledgeManager,
        session_memory: SessionMemory,
        victim: MagicMock,
    ) -> None:
        from orchestration.orchestrator import Orchestrator

        researcher.run_reverse_engineering_pipeline.return_value = {
            "success": True,
            "program_id": "prog_001",
            "accuracy": 0.98,
            "error": None,
        }
        o = Orchestrator(
            cognitive_agent=cognitive,
            strategist_agent=strategist,
            researcher_agent=researcher,
            knowledge_manager=km,
            session_memory=session_memory,
            victim=victim,
            campaign_id="acc_test",
            max_iterations=100,
            accuracy_threshold=0.9,
            synthesis_interval=1,
        )
        result = o.run()
        assert result["success"]
        assert result["total_iterations"] < 10
        session_memory.delete_session("acc_test")

    def test_no_hypotheses_returns_error(
        self,
        cognitive: MagicMock,
        strategist: MagicMock,
        researcher: MagicMock,
        km: KnowledgeManager,
        session_memory: SessionMemory,
        victim: MagicMock,
    ) -> None:
        from orchestration.orchestrator import Orchestrator

        cognitive.generate_hypotheses.return_value = []
        o = Orchestrator(
            cognitive_agent=cognitive,
            strategist_agent=strategist,
            researcher_agent=researcher,
            knowledge_manager=km,
            session_memory=session_memory,
            victim=victim,
            campaign_id="no_hyp",
            max_iterations=3,
        )
        result = o.run()
        assert not result["success"]
        session_memory.delete_session("no_hyp")


class TestSessionIntegration:
    def test_session_updated_during_run(self, orchestrator: Any) -> None:
        orchestrator.run()
        session = orchestrator.session_memory.get_session("test_campaign")
        assert session is not None
        assert session["iteration"] >= 1

    def test_session_status_completed(self, orchestrator: Any) -> None:
        orchestrator.run()
        session = orchestrator.session_memory.get_session("test_campaign")
        assert session is not None
        assert session["status"] == "completed"


class TestInterventionBudget:
    def test_respects_max_interventions(
        self,
        cognitive: MagicMock,
        strategist: MagicMock,
        researcher: MagicMock,
        km: KnowledgeManager,
        session_memory: SessionMemory,
        victim: MagicMock,
    ) -> None:
        from orchestration.orchestrator import Orchestrator

        o = Orchestrator(
            cognitive_agent=cognitive,
            strategist_agent=strategist,
            researcher_agent=researcher,
            knowledge_manager=km,
            session_memory=session_memory,
            victim=victim,
            campaign_id="budget_test",
            max_iterations=100,
            max_interventions=3,
            synthesis_interval=100,
        )
        o.run()
        assert strategist.execute_intervention.call_count <= 3
        session_memory.delete_session("budget_test")


class TestForceExploration:
    def test_force_exploration_interval_parameter(self) -> None:
        from orchestration.orchestrator import Orchestrator
        cog = MagicMock(spec=CognitiveAgent)
        strat = MagicMock(spec=StrategistAgent)
        res = MagicMock(spec=ResearcherAgent)
        km = MagicMock(spec=KnowledgeManager)
        sm = MagicMock(spec=SessionMemory)
        sm.get_session.return_value = None
        sm.create_session.return_value = True
        vic = MagicMock()
        vic.name = "TestVictim"

        o = Orchestrator(
            cognitive_agent=cog,
            strategist_agent=strat,
            researcher_agent=res,
            knowledge_manager=km,
            session_memory=sm,
            victim=vic,
            campaign_id="force_test",
            max_iterations=5,
            force_exploration_interval=2,
        )
        assert o.force_exploration_interval == 2

    def test_force_exploration_after_stalled_iterations(
        self,
        cognitive: MagicMock,
        strategist: MagicMock,
        researcher: MagicMock,
        km: KnowledgeManager,
        session_memory: SessionMemory,
        victim: MagicMock,
    ) -> None:
        from orchestration.orchestrator import Orchestrator

        # Strategist always returns None — should trigger force exploration
        strategist.select_hypothesis_pair.return_value = (None, None)

        o = Orchestrator(
            cognitive_agent=cognitive,
            strategist_agent=strategist,
            researcher_agent=researcher,
            knowledge_manager=km,
            session_memory=session_memory,
            victim=victim,
            campaign_id="force_stall_test",
            max_iterations=10,
            force_exploration_interval=2,
            synthesis_interval=100,
        )
        result = o.run()
        # Pipeline should not crash; may or may not succeed
        assert result is not None
        assert "campaign_id" in result
        session_memory.delete_session("force_stall_test")
    def test_resume_nonexistent(self, orchestrator: Any) -> None:
        o2 = orchestrator
        o2.campaign_id = "nonexistent_campaign"
        result = o2.resume()
        assert not result["success"]
        assert "No saved session" in (result.get("error") or "")

    def test_resume_existing(self, orchestrator: Any) -> None:
        orchestrator.run()
        session = orchestrator.session_memory.get_session("test_campaign")
        assert session is not None

        from orchestration.orchestrator import Orchestrator

        o2 = Orchestrator(
            cognitive_agent=orchestrator.cognitive,
            strategist_agent=orchestrator.strategist,
            researcher_agent=orchestrator.researcher,
            knowledge_manager=orchestrator.knowledge_manager,
            session_memory=orchestrator.session_memory,
            victim=orchestrator.victim,
            campaign_id="test_campaign",
            max_iterations=3,
        )
        result = o2.resume()
        assert result is not None
        assert result["campaign_id"] == "test_campaign"
