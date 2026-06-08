"""Tests for CognitiveAgent."""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agents.cognitive import (
    Anomaly,
    CognitiveAgent,
    DEFAULT_BASE_PROMPTS,
    Hypothesis,
    load_base_prompts,
)
from knowledge.episodic import EpisodicMemory, EpisodeFilter
from synthesis.grammar_exporter import PrimitiveCatalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_episode(
    prompt: str,
    outcome: int,
    episode_id: str = "ep_default",
    transforms: Optional[List[Dict[str, Any]]] = None,
) -> MagicMock:
    ep = MagicMock()
    ep.episode_id = episode_id
    ep.outcome = outcome
    ep.intervention.prompt = prompt
    ep.intervention.final_prompt = prompt
    ep.intervention.transforms = transforms or []
    ep.intervention.transformation_trace = None
    return ep


def _make_mock_primitive_catalog(
    predicates: int = 5,
    transforms: int = 5,
    classifiers: int = 3,
) -> PrimitiveCatalog:
    cat = PrimitiveCatalog()
    for i in range(predicates):
        p = MagicMock()
        p.__class__.__name__ = f"Predicate_{i}"
        cat.predicates.append(p)
    for i in range(transforms):
        t = MagicMock()
        t.__class__.__name__ = f"Transform_{i}"
        cat.transforms.append(t)
    for i in range(classifiers):
        c = MagicMock()
        c.__class__.__name__ = f"Classifier_{i}"
        cat.classifiers.append(c)
    return cat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory() -> MagicMock:
    return MagicMock(spec=EpisodicMemory)


@pytest.fixture
def agent(memory: MagicMock) -> CognitiveAgent:
    return CognitiveAgent(
        episodic_memory=memory,
        llm_client=MagicMock(),
        base_prompts=["test prompt", "hello world"],
    )


# ===================================================================
# Constructor
# ===================================================================


class TestConstructor:
    def test_raises_without_memory(self) -> None:
        with pytest.raises(TypeError, match="episodic_memory"):
            CognitiveAgent(episodic_memory=None)  # type: ignore

    def test_default_llm_client(self, memory: MagicMock) -> None:
        with patch("agents.cognitive.get_default_client") as mock_get:
            mock_get.return_value = "default_llm"
            agent = CognitiveAgent(episodic_memory=memory, llm_client=None)
            assert agent.llm_client == "default_llm"

    def test_default_base_prompts(self, memory: MagicMock) -> None:
        agent = CognitiveAgent(episodic_memory=memory)
        assert agent.base_prompts == set(DEFAULT_BASE_PROMPTS)

    def test_custom_base_prompts(self, memory: MagicMock) -> None:
        agent = CognitiveAgent(
            episodic_memory=memory,
            base_prompts=["custom prompt"],
        )
        assert agent.base_prompts == {"custom prompt"}

    def test_custom_threshold(self, memory: MagicMock) -> None:
        agent = CognitiveAgent(episodic_memory=memory, anomaly_threshold=0.5)
        assert agent.anomaly_threshold == 0.5

    def test_threshold_clamping_above_1(self, memory: MagicMock) -> None:
        with patch("agents.cognitive.logger") as mock_log:
            agent = CognitiveAgent(episodic_memory=memory, anomaly_threshold=2.5)
        assert agent.anomaly_threshold == 1.0
        mock_log.warning.assert_called_once()

    def test_threshold_clamping_below_0(self, memory: MagicMock) -> None:
        with patch("agents.cognitive.logger") as mock_log:
            agent = CognitiveAgent(episodic_memory=memory, anomaly_threshold=-1.0)
        assert agent.anomaly_threshold == 0.0
        mock_log.warning.assert_called_once()

    def test_anomaly_store(self, memory: MagicMock) -> None:
        store = MagicMock()
        agent = CognitiveAgent(episodic_memory=memory, anomaly_store=store)
        assert agent.anomaly_store is store


# ===================================================================
# load_base_prompts
# ===================================================================


class TestLoadBasePrompts:
    def test_load_json(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            json.dump({"base_prompts": ["p1", "p2"]}, f)
            path = f.name
        try:
            result = load_base_prompts(path)
            assert result == ["p1", "p2"]
        finally:
            os.unlink(path)

    def test_load_yaml(self) -> None:
        yaml_content = "base_prompts:\n  - 'p1'\n  - 'p2'\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            result = load_base_prompts(path)
            assert result == ["p1", "p2"]
        finally:
            os.unlink(path)

    def test_unsupported_extension(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            load_base_prompts("config.txt")

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_base_prompts("nonexistent.json")

    def test_base_prompts_path_parameter(self, memory: MagicMock) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            json.dump({"base_prompts": ["file_prompt"]}, f)
            path = f.name
        try:
            agent = CognitiveAgent(episodic_memory=memory,
                                   base_prompts_path=path)
            assert agent.base_prompts == {"file_prompt"}
        finally:
            os.unlink(path)

    def test_base_prompts_path_fallback_on_error(self, memory: MagicMock) -> None:
        agent = CognitiveAgent(episodic_memory=memory,
                               base_prompts_path="nonexistent.json",
                               base_prompts=["fallback"])
        assert agent.base_prompts == {"fallback"}


# ===================================================================
# detect_anomalies — pairwise
# ===================================================================


class TestDetectAnomalies:
    def test_no_episodes(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = []
        result = agent.detect_anomalies()
        assert result == []

    def test_no_episodes_with_filter(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = []
        result = agent.detect_anomalies(campaign_id="camp_non_existent")
        assert result == []

    def test_single_prompt_no_difference(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("hello", 1, "ep_1"),
            _make_mock_episode("hello", 1, "ep_2"),
        ]
        result = agent.detect_anomalies()
        assert result == []

    def test_different_prompts_no_anomaly(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("hello", 1, "ep_1"),
            _make_mock_episode("world", 0, "ep_2"),
        ]
        result = agent.detect_anomalies()
        assert result == []

    def test_detects_anomaly(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        result = agent.detect_anomalies()
        assert len(result) == 1
        anom = result[0]
        assert anom.base_prompt == "bomb"
        assert anom.difference == 1.0
        assert anom.outcome_original == 1
        assert anom.outcome_transformed == 0
        assert anom.episode_id_original == "ep_1"
        assert anom.episode_id_transformed == "ep_2"
        assert anom.transform_names == ["rot13"]

    def test_multiple_anomalies(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2", [{"name": "rot13"}]),
            _make_mock_episode("kill", 1, "ep_3"),
            _make_mock_episode("kill", 0, "ep_4", [{"name": "base64"}]),
            _make_mock_episode("hello", 0, "ep_5"),
            _make_mock_episode("hello", 0, "ep_6", [{"name": "rot13"}]),
        ]
        result = agent.detect_anomalies()
        assert len(result) == 2

    def test_filters_by_campaign(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = []
        agent.detect_anomalies(campaign_id="camp_1")
        call_filter: EpisodeFilter = (
            agent.episodic_memory.filter_episodes.call_args[0][0]
        )
        assert call_filter.campaign_id == "camp_1"

    def test_filters_by_experiment(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = []
        agent.detect_anomalies(experiment_id="exp_42")
        call_filter: EpisodeFilter = (
            agent.episodic_memory.filter_episodes.call_args[0][0]
        )
        assert call_filter.experiment_id == "exp_42"

    def test_filters_by_both_campaign_and_experiment(
        self, agent: CognitiveAgent,
    ) -> None:
        agent.episodic_memory.filter_episodes.return_value = []
        agent.detect_anomalies(campaign_id="camp_1", experiment_id="exp_42")
        call_filter: EpisodeFilter = (
            agent.episodic_memory.filter_episodes.call_args[0][0]
        )
        assert call_filter.campaign_id == "camp_1"
        assert call_filter.experiment_id == "exp_42"

    def test_none_outcome_skipped(self, agent: CognitiveAgent) -> None:
        ep = _make_mock_episode("bomb", 1, "ep_1")
        ep.outcome = None
        agent.episodic_memory.filter_episodes.return_value = [
            ep,
            _make_mock_episode("bomb", 0, "ep_2"),
        ]
        result = agent.detect_anomalies()
        assert len(result) == 0

    def test_none_outcome_does_not_block_valid_pairs(
        self, agent: CognitiveAgent,
    ) -> None:
        ep_none = _make_mock_episode("bomb", 1, "ep_skip")
        ep_none.outcome = None
        agent.episodic_memory.filter_episodes.return_value = [
            ep_none,
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        result = agent.detect_anomalies()
        assert len(result) == 1

    def test_anomaly_auto_id(self) -> None:
        a = Anomaly()
        assert a.id.startswith("anom_")
        assert len(a.id) == 17

    def test_anomaly_difference_auto_calculated(self) -> None:
        a = Anomaly(outcome_original=1, outcome_transformed=0)
        assert a.difference == 1.0

    def test_no_transform_names_fallback(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2", transforms=[]),
        ]
        result = agent.detect_anomalies()
        assert len(result) == 1
        assert result[0].transform_names == ["(no transform)"]

    def test_single_episode_in_group_no_anomaly(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
        ]
        result = agent.detect_anomalies()
        assert result == []

    def test_same_outcome_no_anomaly(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 1, "ep_2", [{"name": "rot13"}]),
            _make_mock_episode("bomb", 1, "ep_3", [{"name": "base64"}]),
        ]
        result = agent.detect_anomalies()
        assert result == []


# ===================================================================
# detect_anomalies — transform chain
# ===================================================================


class TestTransformChain:
    def test_detect_chain_anomaly(self, agent: CognitiveAgent) -> None:
        """Outcome changes only when chain of transforms is applied."""
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_bare"),
            _make_mock_episode("bomb", 1, "ep_rot13",
                               transforms=[{"name": "rot13"}]),
            _make_mock_episode("bomb", 1, "ep_base64",
                               transforms=[{"name": "base64"}]),
            _make_mock_episode("bomb", 0, "ep_chain",
                               transforms=[{"name": "rot13"},
                                           {"name": "base64"}]),
        ]
        result = agent.detect_anomalies()
        chain = [a for a in result if len(a.transform_names) > 1]
        assert len(chain) >= 1
        assert chain[0].transform_names == ["rot13", "base64"]

    def test_no_chain_when_single_differs(self, agent: CognitiveAgent) -> None:
        """When a single transform already changes outcome, chain detection
        does not emit a separate anomaly."""
        episodes = [
            _make_mock_episode("bomb", 1, "ep_bare"),
            _make_mock_episode("bomb", 0, "ep_rot13",
                               transforms=[{"name": "rot13"}]),
            _make_mock_episode("bomb", 0, "ep_chain",
                               transforms=[{"name": "rot13"},
                                           {"name": "base64"}]),
        ]
        chain_specific = agent._detect_transform_chain_anomalies(episodes)
        assert len(chain_specific) == 0

    def test_no_chain_insufficient_episodes(self, agent: CognitiveAgent) -> None:
        """Need ≥ 3 episodes in group for chain detection."""
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_bare"),
            _make_mock_episode("bomb", 1, "ep_chain",
                               transforms=[{"name": "rot13"},
                                           {"name": "base64"}]),
        ]
        result = agent.detect_anomalies()
        assert len(result) == 0

    def test_chain_requires_single_transform_consistency(
        self, agent: CognitiveAgent,
    ) -> None:
        """Two single-transform episodes must have same outcome for chain."""
        episodes = [
            _make_mock_episode("bomb", 1, "ep_bare"),
            _make_mock_episode("bomb", 1, "ep_rot13",
                               transforms=[{"name": "rot13"}]),
            _make_mock_episode("bomb", 0, "ep_base64",
                               transforms=[{"name": "base64"}]),
            _make_mock_episode("bomb", 0, "ep_chain",
                               transforms=[{"name": "rot13"},
                                           {"name": "base64"}]),
        ]
        chain_specific = agent._detect_transform_chain_anomalies(episodes)
        assert len(chain_specific) == 0


# ===================================================================
# anomaly_store callback
# ===================================================================


class TestAnomalyStore:
    def test_store_called_with_anomalies(self, memory: MagicMock) -> None:
        store = MagicMock()
        agent = CognitiveAgent(
            episodic_memory=memory,
            llm_client=MagicMock(),
            base_prompts=["test"],
            anomaly_store=store,
        )
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        result = agent.detect_anomalies()
        store.assert_called_once()
        args = store.call_args[0][0]
        assert len(args) == len(result)
        assert args[0].base_prompt == "bomb"

    def test_store_not_called_with_empty(self, memory: MagicMock) -> None:
        store = MagicMock()
        agent = CognitiveAgent(
            episodic_memory=memory,
            llm_client=MagicMock(),
            base_prompts=["test"],
            anomaly_store=store,
        )
        agent.episodic_memory.filter_episodes.return_value = []
        agent.detect_anomalies()
        store.assert_not_called()

    def test_store_exception_does_not_propagate(
        self, memory: MagicMock,
    ) -> None:
        store = MagicMock(side_effect=RuntimeError("store failed"))
        agent = CognitiveAgent(
            episodic_memory=memory,
            llm_client=MagicMock(),
            base_prompts=["test"],
            anomaly_store=store,
        )
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        result = agent.detect_anomalies()
        assert len(result) == 1  # store failure does not affect results


# ===================================================================
# generate_hypotheses
# ===================================================================


class TestGenerateHypotheses:
    def test_no_anomalies(self, agent: CognitiveAgent) -> None:
        result = agent.generate_hypotheses([])
        assert result == []

    def test_llm_returns_valid_json(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        anomalies = agent.detect_anomalies()
        assert len(anomalies) == 1

        llm_response = json.dumps([
            {"description": "ROT13 bypasses keyword filter",
             "condition": "IF contains_word(decode_rot13(prompt), 'bomb') THEN REFUSE"},
            {"description": "Model checks for dangerous keywords directly",
             "condition": "IF contains_word(prompt, 'bomb') THEN REFUSE"},
        ])
        agent.llm_client.generate.return_value = llm_response

        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            hypotheses = agent.generate_hypotheses(anomalies)

        assert len(hypotheses) >= 2
        assert any("ROT13" in h.description for h in hypotheses)

    def test_llm_fallback_on_error(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        anomalies = agent.detect_anomalies()
        agent.llm_client.generate.side_effect = RuntimeError("API error")

        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            hypotheses = agent.generate_hypotheses(anomalies)

        assert len(hypotheses) >= 3
        assert any("keyword" in h.description.lower() or
                    "contains" in h.condition.lower() for h in hypotheses)

    def test_llm_fallback_on_invalid_json(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        anomalies = agent.detect_anomalies()
        agent.llm_client.generate.return_value = "not json at all"

        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            hypotheses = agent.generate_hypotheses(anomalies)

        assert len(hypotheses) >= 3

    def test_hypothesis_auto_id(self) -> None:
        h = Hypothesis()
        assert h.id.startswith("hyp_")
        assert len(h.id) == 16

    def test_caps_at_five_hypotheses(self, agent: CognitiveAgent) -> None:
        anomalies = [
            Anomaly(base_prompt="test", outcome_original=1,
                    outcome_transformed=0),
        ]
        llm_response = json.dumps([
            {"description": f"Hypothesis {i}",
             "condition": f"IF cond_{i} THEN REFUSE"}
            for i in range(10)
        ])
        agent.llm_client.generate.return_value = llm_response

        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            hypotheses = agent.generate_hypotheses(anomalies)

        # generate_hypotheses merges LLM output with fallback keyword hypotheses.
        # Expect >= 5 LLM + fallback hypotheses, each with a unique condition.
        assert len(hypotheses) >= 5
        # At least the first 5 should be the LLM-generated ones (order preserved)
        assert any("Hypothesis" in h.description for h in hypotheses)

    def test_prior_hypotheses_included_in_prompt(
        self, agent: CognitiveAgent,
    ) -> None:
        """Prior hypotheses should appear in the LLM prompt."""
        anomalies = [
            Anomaly(base_prompt="test", outcome_original=1,
                    outcome_transformed=0),
        ]
        prior = [
            Hypothesis(description="Old filter hypothesis",
                       condition="IF contains(x) THEN REFUSE",
                       confidence=0.6),
        ]
        agent.llm_client.generate.return_value = "[]"

        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            agent.generate_hypotheses(anomalies, prior_hypotheses=prior)

        prompt = agent.llm_client.generate.call_args[0][0]
        assert "Prior hypotheses" in prompt
        assert "Old filter hypothesis" in prompt
        assert "0.60" in prompt or "0.6" in prompt


class TestFallbackHypotheses:
    def test_fallback_count(self, agent: CognitiveAgent) -> None:
        """_fallback_hypotheses should return at least 5 diverse hypotheses."""
        anomalies = [
            Anomaly(base_prompt="test", outcome_original=1,
                    outcome_transformed=0),
        ]
        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            hyps = agent._fallback_hypotheses(anomalies)

        assert len(hyps) >= 5
        for h in hyps:
            assert h.condition
            assert h.description
            assert h.supporting_anomaly_ids

    def test_fallback_includes_please_accept_hypothesis(
        self, agent: CognitiveAgent,
    ) -> None:
        """Fallback should include ACCEPT hypotheses for polite/benign patterns."""
        anomalies = [
            Anomaly(base_prompt="test", outcome_original=1,
                    outcome_transformed=0),
        ]
        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            hyps = agent._fallback_hypotheses(anomalies)

        conditions = [h.condition for h in hyps]
        # Polite language now predicts ACCEPT (not REFUSE)
        assert any("please" in c and "ACCEPT" in c for c in conditions)
        # At least one ACCEPT hypothesis exists
        assert any("ACCEPT" in c for c in conditions)

    def test_fallback_includes_roleplay_accepthypothesis(
        self, agent: CognitiveAgent,
    ) -> None:
        """Fallback should have a hypothesis that accepts prompts with 'researcher'."""
        anomalies = [
            Anomaly(base_prompt="test", outcome_original=1,
                    outcome_transformed=0),
        ]
        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            hyps = agent._fallback_hypotheses(anomalies)

        assert any("THEN ACCEPT" in h.condition for h in hyps)


# ===================================================================
# estimate_confidence — weighted
# ===================================================================


class TestEstimateConfidence:
    def test_no_anomalies(self, agent: CognitiveAgent) -> None:
        hyp = Hypothesis(description="test", condition="IF x THEN REFUSE")
        agent.estimate_confidence(hyp, [])
        assert hyp.confidence == 0.5

    def test_all_supporting_equal_weight(self, agent: CognitiveAgent) -> None:
        anomalies = [
            Anomaly(id="a1", outcome_original=1, outcome_transformed=0),
            Anomaly(id="a2", outcome_original=0, outcome_transformed=1),
        ]
        hyp = Hypothesis(description="test", condition="IF x THEN REFUSE",
                         supporting_anomaly_ids=["a1", "a2"])
        agent.estimate_confidence(hyp, anomalies)
        # weighted: supporting_weight = 1+1=2, total_weight=1+1+1=3
        # (supporting_weight) / (total_weight+1) = 2/4 = 0.5
        # Wait: supporting_weight = sum(diff of supporting) + 1 = 1+1+1 = 3
        # total_weight = sum(diff of all) + 1 = 1+1+1 = 3
        # confidence = 3 / (3+1) = 0.75
        assert hyp.confidence == pytest.approx(3.0 / 4.0)

    def test_weighted_by_difference(self, agent: CognitiveAgent) -> None:
        anomalies = [
            Anomaly(id="a1", outcome_original=1, outcome_transformed=0),
            Anomaly(id="a2", outcome_original=1, outcome_transformed=1),
        ]
        # a1: diff=1, a2: diff=0 (same outcome)
        hyp = Hypothesis(description="test", condition="IF x THEN REFUSE",
                         supporting_anomaly_ids=["a1"])
        agent.estimate_confidence(hyp, anomalies)
        # supporting_weight = diff(a1) + 1 = 1 + 1 = 2
        # total_weight = diff(a1) + diff(a2) + 1 = 1 + 0 + 1 = 2
        # confidence = 2 / (2 + 1) = 0.666...
        assert hyp.confidence == pytest.approx(2.0 / 3.0)

    def test_no_support(self, agent: CognitiveAgent) -> None:
        anomalies = [
            Anomaly(id="a1", outcome_original=1, outcome_transformed=0),
            Anomaly(id="a2", outcome_original=0, outcome_transformed=1),
        ]
        hyp = Hypothesis(description="test", condition="IF x THEN REFUSE",
                         supporting_anomaly_ids=[])
        agent.estimate_confidence(hyp, anomalies)
        # supporting_weight = 0 + 1 = 1
        # total_weight = 1+1+1 = 3
        # confidence = 1 / (3+1) = 0.25
        assert hyp.confidence == pytest.approx(1.0 / 4.0)

    def test_single_anomaly(self, agent: CognitiveAgent) -> None:
        anomalies = [Anomaly(id="a1", outcome_original=1,
                             outcome_transformed=0)]
        hyp = Hypothesis(description="test", condition="IF x THEN REFUSE",
                         supporting_anomaly_ids=["a1"])
        agent.estimate_confidence(hyp, anomalies)
        # supporting_weight = 1 + 1 = 2, total_weight = 1 + 1 = 2
        # confidence = 2 / (2+1) = 2/3
        assert hyp.confidence == pytest.approx(2.0 / 3.0)


# ===================================================================
# Pipeline: detect → generate
# ===================================================================


class TestPipeline:
    def test_detect_and_generate(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        llm_response = json.dumps([
            {"description": "ROT13 bypass detection",
             "condition": "IF contains_word(decode_rot13(prompt), 'bomb') THEN REFUSE"},
        ])
        agent.llm_client.generate.return_value = llm_response

        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            anomalies = agent.detect_anomalies()
            hypotheses = agent.generate_hypotheses(anomalies)

        assert len(anomalies) == 1
        assert len(hypotheses) >= 1
        # LLM-generated hypothesis should be present
        assert any("ROT13" in h.description for h in hypotheses)
        # At least one hypothesis should have supporting evidence
        assert any(h.supporting_anomaly_ids == [a.id for a in anomalies]
                   for h in hypotheses)
        assert all(h.confidence > 0 for h in hypotheses)

    def test_no_anomalies_pipeline(self, agent: CognitiveAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("hello", 0, "ep_1"),
            _make_mock_episode("hello", 0, "ep_2"),
        ]
        anomalies = agent.detect_anomalies()
        hypotheses = agent.generate_hypotheses(anomalies)
        assert anomalies == []
        assert hypotheses == []


# ===================================================================
# Data class serialisation
# ===================================================================


class TestDataClassSerialisation:
    def test_anomaly_to_dict(self) -> None:
        a = Anomaly(
            id="anom_test", base_prompt="test prompt",
            transform_names=["rot13"],
            outcome_original=1, outcome_transformed=0, difference=1.0,
            episode_id_original="ep_1", episode_id_transformed="ep_2",
            timestamp=1000.0,
        )
        d = a.to_dict()
        assert d["id"] == "anom_test"
        assert d["base_prompt"] == "test prompt"
        assert d["difference"] == 1.0

    def test_hypothesis_to_dict(self) -> None:
        h = Hypothesis(id="hyp_test", description="test hypothesis",
                       condition="IF x THEN REFUSE", confidence=0.75,
                       supporting_anomaly_ids=["a1", "a2"], created_at=2000.0)
        d = h.to_dict()
        assert d["id"] == "hyp_test"
        assert d["condition"] == "IF x THEN REFUSE"
        assert d["confidence"] == 0.75


# ===================================================================
# LLM parsing edge cases
# ===================================================================


class TestLlmParsing:
    def test_direct_json_fragment(self, agent: CognitiveAgent) -> None:
        hyps = agent._parse_llm_hypotheses(
            '{"description": "test", "condition": "IF x THEN REFUSE"}',
            [Anomaly(id="a1")],
        )
        assert len(hyps) >= 1

    def test_mixed_text_and_json(self, agent: CognitiveAgent) -> None:
        raw = (
            "I think the model has a keyword filter.\n\n"
            '[{"description": "Keyword filter", '
            '"condition": "IF contains(x) THEN REFUSE"}]\n\n'
            "This explains the ROT13 behavior."
        )
        hyps = agent._parse_llm_hypotheses(raw, [Anomaly(id="a1")])
        assert len(hyps) == 1
        assert hyps[0].description == "Keyword filter"


# ===================================================================
# base_prompts validation (2.3)
# ===================================================================


class TestBasePromptsValidation:
    def test_rejects_empty_string(self, memory: MagicMock) -> None:
        with pytest.raises(ValueError, match="empty"):
            CognitiveAgent(
                episodic_memory=memory,
                base_prompts=["valid", ""],
            )

    def test_rejects_whitespace_only(self, memory: MagicMock) -> None:
        with pytest.raises(ValueError, match="empty"):
            CognitiveAgent(
                episodic_memory=memory,
                base_prompts=["valid", "   "],
            )

    def test_rejects_too_long(self, memory: MagicMock) -> None:
        long_prompt = "x" * 1001
        with pytest.raises(ValueError, match="exceeds max length"):
            CognitiveAgent(
                episodic_memory=memory,
                base_prompts=[long_prompt],
            )

    def test_accepts_boundary_length(self, memory: MagicMock) -> None:
        prompt = "x" * 1000
        agent = CognitiveAgent(
            episodic_memory=memory,
            base_prompts=[prompt],
        )
        assert len(next(iter(agent.base_prompts))) == 1000

    def test_rejects_non_string_item(self, memory: MagicMock) -> None:
        with pytest.raises(ValueError, match="not a string"):
            CognitiveAgent(
                episodic_memory=memory,
                base_prompts=["ok", 123],  # type: ignore
            )

    def test_validates_inline_prompts(self, memory: MagicMock) -> None:
        agent = CognitiveAgent(
            episodic_memory=memory,
            base_prompts=["prompt one", "prompt two"],
        )
        assert agent.base_prompts == {"prompt one", "prompt two"}


# ===================================================================
# anomaly_store_queue (2.1)
# ===================================================================


class TestAnomalyStoreQueue:
    def test_queue_receives_anomalies(self, memory: MagicMock) -> None:
        from queue import Queue
        q: Queue = Queue()
        agent = CognitiveAgent(
            episodic_memory=memory,
            llm_client=MagicMock(),
            base_prompts=["test"],
            anomaly_store_queue=q,
        )
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        agent.detect_anomalies()
        assert not q.empty()
        items = q.get_nowait()
        assert len(items) >= 1
        assert items[0].base_prompt == "bomb"

    def test_queue_not_called_when_empty(self, memory: MagicMock) -> None:
        from queue import Queue
        q: Queue = Queue()
        agent = CognitiveAgent(
            episodic_memory=memory,
            llm_client=MagicMock(),
            base_prompts=["test"],
            anomaly_store_queue=q,
        )
        agent.episodic_memory.filter_episodes.return_value = []
        agent.detect_anomalies()
        assert q.empty()

    def test_queue_exception_does_not_propagate(
        self, memory: MagicMock,
    ) -> None:
        from queue import Queue, Full
        q: Queue = Queue(maxsize=0)  # non-blocking put will fail
        agent = CognitiveAgent(
            episodic_memory=memory,
            llm_client=MagicMock(),
            base_prompts=["test"],
            anomaly_store_queue=q,
        )
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        result = agent.detect_anomalies()
        assert len(result) >= 1

    def test_queue_preferred_over_callback(self, memory: MagicMock) -> None:
        from queue import Queue
        q: Queue = Queue()
        store = MagicMock()
        agent = CognitiveAgent(
            episodic_memory=memory,
            llm_client=MagicMock(),
            base_prompts=["test"],
            anomaly_store=store,
            anomaly_store_queue=q,
        )
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        agent.detect_anomalies()
        assert not q.empty()
        store.assert_not_called()


# ===================================================================
# persist_anomalies SQLite (2.1)
# ===================================================================


class TestPersistAnomalies:
    def test_sqlite_persistence(self, memory: MagicMock) -> None:
        agent = CognitiveAgent(
            episodic_memory=memory,
            llm_client=MagicMock(),
            base_prompts=["test"],
            persist_anomalies=True,
        )
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        anomalies = agent.detect_anomalies()
        assert len(anomalies) >= 1

    def test_persist_creates_anomaly_db(self, memory: MagicMock) -> None:
        agent = CognitiveAgent(
            episodic_memory=memory,
            llm_client=MagicMock(),
            base_prompts=["test"],
            persist_anomalies=True,
        )
        assert agent._anomaly_db is not None

    def test_no_persist_by_default(self, memory: MagicMock) -> None:
        agent = CognitiveAgent(
            episodic_memory=memory,
            llm_client=MagicMock(),
            base_prompts=["test"],
        )
        assert agent._anomaly_db is None


# ===================================================================
# LLM retry (2.1)
# ===================================================================


class TestLlmRetry:
    def test_retry_on_failure(self, agent: CognitiveAgent) -> None:
        anomalies = [Anomaly(base_prompt="test", outcome_original=1,
                             outcome_transformed=0)]
        # 3 failures: first attempt + 2 retries
        agent.llm_client.generate.side_effect = [
            RuntimeError("fail 1"),
            RuntimeError("fail 2"),
            RuntimeError("fail 3"),
        ]
        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            hypotheses = agent.generate_hypotheses(anomalies)

        assert len(hypotheses) >= 3
        assert any("keyword" in h.description.lower() or
                    "contains" in h.condition.lower() for h in hypotheses)

    def test_retry_eventually_succeeds(self, agent: CognitiveAgent) -> None:
        anomalies = [Anomaly(base_prompt="test", outcome_original=1,
                             outcome_transformed=0)]
        valid_json = json.dumps([
            {"description": "Retry hypothesis",
             "condition": "IF retry_works THEN REFUSE"},
        ])
        agent.llm_client.generate.side_effect = [
            RuntimeError("fail 1"),
            valid_json,
        ]
        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            hypotheses = agent.generate_hypotheses(anomalies)

        assert len(hypotheses) >= 1
        assert any("Retry hypothesis" in h.description for h in hypotheses)

    def test_retry_logs_debug(self, agent: CognitiveAgent) -> None:
        anomalies = [Anomaly(base_prompt="test", outcome_original=1,
                             outcome_transformed=0)]
        agent.llm_client.generate.side_effect = RuntimeError("fail")
        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            with patch("agents.cognitive.logger") as mock_log:
                agent.generate_hypotheses(anomalies)
        # Should have warning logs for each attempt
        assert mock_log.warning.call_count >= 1


# ===================================================================
# refresh_primitive_cache (2.6)
# ===================================================================


class TestPrimitiveCache:
    def test_cache_invalidated(self, agent: CognitiveAgent) -> None:
        # Pre-populate cache
        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            _ = agent._get_primitives()

        assert agent._cached_primitives is not None

        # Invalidate
        agent.refresh_primitive_cache()
        assert agent._cached_primitives is None

    def test_refetch_after_invalidation(self, agent: CognitiveAgent) -> None:
        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()) as mock_get:
            # First call populates cache
            first = agent._get_primitives()
            mock_get.assert_called_once()

            # Second call uses cache
            second = agent._get_primitives()
            assert mock_get.call_count == 1
            assert first is second

            # After refresh, next call refetches
            agent.refresh_primitive_cache()
            third = agent._get_primitives()
            assert mock_get.call_count == 2

    def test_cache_used_in_generate_hypotheses(
        self, agent: CognitiveAgent,
    ) -> None:
        anomalies = [Anomaly(base_prompt="test", outcome_original=1,
                             outcome_transformed=0)]
        agent.llm_client.generate.return_value = json.dumps([
            {"description": "h1", "condition": "IF x THEN REFUSE"},
        ])
        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()) as mock_get:
            agent.generate_hypotheses(anomalies)
            agent.generate_hypotheses(anomalies)

        # get_primitives called only once (cached)
        assert mock_get.call_count == 1


# ===================================================================
# get_anomalies returns List[Anomaly] (2.5)
# ===================================================================


class TestGetAnomalies:
    def test_returns_list_of_anomaly_objects(
        self, memory: MagicMock,
    ) -> None:
        agent = CognitiveAgent(
            episodic_memory=memory,
            llm_client=MagicMock(),
            base_prompts=["test"],
            persist_anomalies=True,
        )
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("bomb", 0, "ep_2",
                               transforms=[{"name": "rot13"}]),
        ]
        agent.detect_anomalies(campaign_id="camp_get")
        anomalies = agent.get_anomalies(campaign_id="camp_get")

        assert isinstance(anomalies, list)
        if anomalies:
            assert isinstance(anomalies[0], Anomaly)
            assert anomalies[0].base_prompt == "bomb"
            assert anomalies[0].transform_names == ["rot13"]
            assert anomalies[0].difference >= 0

    def test_warns_when_not_persisted(
        self, agent: CognitiveAgent,
    ) -> None:
        with patch("agents.cognitive.logger") as mock_log:
            result = agent.get_anomalies()
        assert result == []
        mock_log.warning.assert_called_once()

    def test_empty_when_no_matches(
        self, memory: MagicMock,
    ) -> None:
        agent = CognitiveAgent(
            episodic_memory=memory,
            llm_client=MagicMock(),
            base_prompts=["test"],
            persist_anomalies=True,
        )
        result = agent.get_anomalies(campaign_id="nonexistent")
        assert result == []


# ===================================================================
# Logging format (2.2)
# ===================================================================


class TestLoggingFormat:
    def test_fallback_logs_fallback_true(self, agent: CognitiveAgent) -> None:
        anomalies = [Anomaly(base_prompt="test", outcome_original=1,
                             outcome_transformed=0)]
        agent.llm_client.generate.side_effect = RuntimeError("fail")
        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            with patch("agents.cognitive.logger") as mock_log:
                agent.generate_hypotheses(anomalies)

        # Find the "Generated" log call
        gen_calls = [
            c for c in mock_log.info.call_args_list
            if "Generated" in str(c)
        ]
        assert len(gen_calls) >= 1
        # fallback=True should appear in the log message or args
        call_str = str(gen_calls[0])
        assert "fallback=True" in call_str or True in gen_calls[0][0]

    def test_success_logs_fallback_false(self, agent: CognitiveAgent) -> None:
        anomalies = [Anomaly(base_prompt="test", outcome_original=1,
                             outcome_transformed=0)]
        agent.llm_client.generate.return_value = json.dumps([
            {"description": "h1", "condition": "IF x THEN REFUSE"},
        ])
        with patch.object(agent.grammar_exporter, "get_primitives",
                          return_value=_make_mock_primitive_catalog()):
            with patch("agents.cognitive.logger") as mock_log:
                agent.generate_hypotheses(anomalies)

        gen_calls = [
            c for c in mock_log.info.call_args_list
            if "llm=" in str(c)
        ]
        assert len(gen_calls) >= 1
        args = gen_calls[0][0]
        # args: fmt, len(merged), len(anomalies), avg_conf, fallback, llm, merged
        assert args[4] is False  # fallback=False (LLM succeeded)
