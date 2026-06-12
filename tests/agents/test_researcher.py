"""Tests for ResearcherAgent."""

from unittest.mock import MagicMock, patch

import pytest

from agents.researcher import ResearcherAgent
from core.executor import ProgramExecutor
from core.primitive import default_registry
from core.program import Program
from knowledge.scientific_memory import Theory
from harmony.synthesis import SynthesisStats
from synthesis.verifier import VerificationReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXECUTOR = ProgramExecutor(default_registry)


def _make_mock_episode(prompt: str, outcome: int, episode_id: str = "ep_1") -> MagicMock:
    ep = MagicMock()
    ep.episode_id = episode_id
    ep.outcome = outcome
    ep.intervention.final_prompt = prompt
    ep.intervention.prompt = prompt
    return ep


def _make_mock_program(prog_id: str = "prog_test") -> MagicMock:
    prog = MagicMock(spec=Program)
    prog.id = prog_id
    prog.complexity.return_value = 1
    prog.depth.return_value = 1
    prog.canonical_form.return_value = "ITE(contains_word('bomb'), 1, 0)"
    return prog


def _make_mock_report(
    accuracy: float = 0.95,
    verified: bool = True,
) -> VerificationReport:
    return VerificationReport(
        program=_make_mock_program(),
        accuracy=accuracy,
        verified=verified,
        num_tested=10,
        num_correct=int(accuracy * 10),
        failures=[],
        suggestions=[],
    )


def _make_mock_theory(theory_id: str = "thr_abc") -> MagicMock:
    t = MagicMock()
    t.id = theory_id
    t.pattern = "contains_word('bomb')"
    return t


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def agent() -> ResearcherAgent:
    episodic = MagicMock()
    defense_store = MagicMock()
    sci_mem = MagicMock()
    synth = MagicMock()
    executor = MagicMock(spec=ProgramExecutor)

    agent = ResearcherAgent(
        episodic_memory=episodic,
        defense_store=defense_store,
        scientific_memory=sci_mem,
        synthesizer=synth,
        executor=executor,
        default_model_family="test_family",
    )
    return agent


@pytest.fixture
def agent_no_synth() -> ResearcherAgent:
    episodic = MagicMock()
    defense_store = MagicMock()
    sci_mem = MagicMock()
    executor = MagicMock(spec=ProgramExecutor)

    agent = ResearcherAgent(
        episodic_memory=episodic,
        defense_store=defense_store,
        scientific_memory=sci_mem,
        executor=executor,
        default_model_family="test_family",
    )
    return agent


# ===================================================================
# synthesize_from_campaign
# ===================================================================


class TestSynthesizeFromCampaign:
    def test_success(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("hello", 0, "ep_2"),
        ]
        agent.synthesizer.synthesize.return_value = [mock_program]

        program, stats = agent.synthesize_from_campaign("camp_1")

        assert program is mock_program
        assert stats is not None
        agent.episodic_memory.filter_episodes.assert_called_once()
        agent.synthesizer.synthesize.assert_called_once()

    def test_no_episodes(self, agent: ResearcherAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = []

        program, stats = agent.synthesize_from_campaign("camp_empty")

        assert program is None
        assert stats.programs_tried == 0

    def test_with_experiment_id(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("kill", 1, "ep_3"),
        ]
        agent.synthesizer.synthesize.return_value = [mock_program]

        program, stats = agent.synthesize_from_campaign(
            "camp_1", experiment_id="exp_42", error_tolerance=0.1,
        )

        assert program is mock_program
        assert stats is not None
        call_filter = agent.episodic_memory.filter_episodes.call_args[0][0]
        assert call_filter.experiment_id == "exp_42"

    def test_all_none_outcomes_skipped(self, agent: ResearcherAgent) -> None:
        ep = _make_mock_episode("test", 1, "ep_1")
        ep.outcome = None
        agent.episodic_memory.filter_episodes.return_value = [ep]

        program, stats = agent.synthesize_from_campaign("camp_1")
        assert program is None

    def test_exclude_prompts(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("hello", 0, "ep_2"),
        ]
        agent.synthesizer.synthesize.return_value = [mock_program]

        program, stats = agent.synthesize_from_campaign(
            "camp_1", exclude_prompts={"hello"},
        )

        assert program is mock_program
        examples = agent.synthesizer.synthesize.call_args[0][0]
        assert len(examples) == 1
        assert examples[0] == ("bomb", 1)

    def test_exclude_episode_ids(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("hello", 0, "ep_2"),
        ]
        agent.synthesizer.synthesize.return_value = [mock_program]

        program, stats = agent.synthesize_from_campaign(
            "camp_1", exclude_episode_ids={"ep_2"},
        )

        assert program is mock_program
        examples = agent.synthesizer.synthesize.call_args[0][0]
        assert len(examples) == 1
        assert examples[0] == ("bomb", 1)

    def test_with_nonexistent_experiment_id(self, agent: ResearcherAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = []

        program, stats = agent.synthesize_from_campaign(
            "camp_1", experiment_id="exp_nonexistent",
        )

        assert program is None
        assert stats.programs_tried == 0
        call_filter = agent.episodic_memory.filter_episodes.call_args[0][0]
        assert call_filter.experiment_id == "exp_nonexistent"

    def test_all_excluded(self, agent: ResearcherAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
        ]

        program, stats = agent.synthesize_from_campaign(
            "camp_1", exclude_prompts={"bomb"},
        )

        assert program is None
        assert stats.programs_tried == 0


# ===================================================================
# verify_program
# ===================================================================


class TestVerifyProgram:
    def test_verify_success(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        mock_victim = MagicMock()
        mock_report = _make_mock_report()

        with patch("agents.researcher.ProgramVerifier", autospec=True) as MockVerifier:
            verifier_instance = MagicMock()
            verifier_instance.verify.return_value = mock_report
            MockVerifier.return_value = verifier_instance

            report = agent.verify_program(
                mock_program, mock_victim,
                num_test_interventions=10,
                accuracy_threshold=0.9,
                verbose=True,
            )

        assert report.verified is True
        assert report.accuracy == 0.95
        verifier_instance.verify.assert_called_once_with(
            mock_program,
            num_test_interventions=10,
            accuracy_threshold=0.9,
            exclude_prompts=None,
            verbose=True,
        )

    def test_verify_fails_below_threshold(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        mock_victim = MagicMock()
        mock_report = _make_mock_report(accuracy=0.5, verified=False)

        with patch("agents.researcher.ProgramVerifier", autospec=True) as MockVerifier:
            verifier_instance = MagicMock()
            verifier_instance.verify.return_value = mock_report
            MockVerifier.return_value = verifier_instance

            report = agent.verify_program(
                mock_program, mock_victim, accuracy_threshold=0.9,
            )

        assert report.verified is False

    def test_verify_caches_verifier_per_victim(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        mock_victim = MagicMock()
        mock_victim.victim_id = "vic_1"
        mock_report = _make_mock_report()

        with patch("agents.researcher.ProgramVerifier", autospec=True) as MockVerifier:
            MockVerifier.return_value.verify.return_value = mock_report

            agent.verify_program(mock_program, mock_victim)
            agent.verify_program(mock_program, mock_victim)

        # Only one ProgramVerifier should be created (cached)
        assert MockVerifier.call_count == 1

    def test_verify_with_shared_verifier(self, agent: ResearcherAgent) -> None:
        shared_verifier = MagicMock()
        shared_verifier.verify.return_value = _make_mock_report()
        agent._shared_verifier = shared_verifier

        report = agent.verify_program(_make_mock_program(), MagicMock())

        assert report.verified is True


# ===================================================================
# store_program
# ===================================================================


class TestStoreProgram:
    def test_store_default_name(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        agent.defense_store.save.return_value = "prog_abc123"

        pid = agent.store_program(mock_program, confidence=0.95)

        assert pid == "prog_abc123"
        agent.defense_store.save.assert_called_once()
        saved_record = agent.defense_store.save.call_args[0][0]
        assert saved_record.confidence == 0.95
        assert saved_record.metadata["status"] == "confirmed"

    def test_store_custom_name_and_provenance(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        agent.defense_store.save.return_value = "prog_xyz"

        pid = agent.store_program(
            mock_program,
            name="my_program",
            confidence=0.8,
            provenance=["exp_1", "exp_2"],
            status="draft",
        )

        assert pid == "prog_xyz"
        saved = agent.defense_store.save.call_args[0][0]
        assert saved.name == "my_program"
        assert saved.provenance == ["exp_1", "exp_2"]
        assert saved.metadata["status"] == "draft"


# ===================================================================
# abstract_theory
# ===================================================================


class TestAbstractTheory:
    def test_abstract_with_default_family(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        mock_theory = _make_mock_theory()
        agent.synthesizer.abstract_theory.return_value = mock_theory

        theory = agent.abstract_theory(mock_program)

        assert theory is mock_theory
        agent.synthesizer.abstract_theory.assert_called_once_with(
            mock_program,
            model_family="test_family",
            conditions=None,
            provenance=None,
        )

    def test_abstract_with_custom_family(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        mock_theory = _make_mock_theory()
        agent.synthesizer.abstract_theory.return_value = mock_theory

        theory = agent.abstract_theory(
            mock_program,
            model_family="GPT-4o",
            conditions={"key": "val"},
            provenance=["ep_1"],
        )

        agent.synthesizer.abstract_theory.assert_called_once_with(
            mock_program,
            model_family="GPT-4o",
            conditions={"key": "val"},
            provenance=["ep_1"],
        )

    def test_abstract_fallback_when_synthesizer_lacks_method(
        self, agent: ResearcherAgent,
    ) -> None:
        mock_program = _make_mock_program()
        del agent.synthesizer.abstract_theory

        theory = agent.abstract_theory(mock_program, model_family="fallback_family")

        assert isinstance(theory, Theory)
        assert theory.conditions.get("model_family") == "fallback_family"
        assert theory.pattern == str(mock_program)

    def test_abstract_fallback_on_exception(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        agent.synthesizer.abstract_theory.side_effect = RuntimeError("synthesis error")

        theory = agent.abstract_theory(mock_program, model_family="fallback_family")

        assert isinstance(theory, Theory)
        assert theory.conditions.get("model_family") == "fallback_family"


# ===================================================================
# store_theory
# ===================================================================


class TestStoreTheory:
    def test_store_theory(self, agent: ResearcherAgent) -> None:
        mock_theory = _make_mock_theory()
        agent.scientific_memory.save_theory.return_value = "thr_abc"

        tid = agent.store_theory(mock_theory)

        assert tid == "thr_abc"
        agent.scientific_memory.save_theory.assert_called_once_with(mock_theory)


# ===================================================================
# run_reverse_engineering_pipeline
# ===================================================================


class TestPipeline:
    def test_pipeline_success(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        mock_report = _make_mock_report()
        mock_theory = _make_mock_theory(theory_id="thr_pipe")

        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
        ]
        agent.synthesizer.synthesize.return_value = [mock_program]
        agent.defense_store.save.return_value = "prog_pipe"
        agent.scientific_memory.save_theory.return_value = "thr_pipe"

        with patch.object(agent, "verify_program") as mock_vp:
            mock_vp.return_value = mock_report
            result = agent.run_reverse_engineering_pipeline(
                campaign_id="camp_pipe",
                victim=MagicMock(),
                program_name="test_prog",
                num_test_interventions=10,
                accuracy_threshold=0.9,
            )

        assert result["success"] is True
        assert result["program_id"] == "prog_pipe"
        assert result["theory_id"] == "thr_pipe"
        assert result["accuracy"] == 0.95
        assert result["verified"] is True

    def test_pipeline_no_program(self, agent: ResearcherAgent) -> None:
        agent.episodic_memory.filter_episodes.return_value = []

        result = agent.run_reverse_engineering_pipeline(
            campaign_id="camp_empty",
            victim=MagicMock(),
        )

        assert result["success"] is False
        assert result["program_id"] is None
        assert result["error"] is not None

    def test_pipeline_verification_below_threshold(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        mock_report = _make_mock_report(accuracy=0.4, verified=False)
        mock_theory = _make_mock_theory(theory_id="thr_low")

        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
        ]
        agent.synthesizer.synthesize.return_value = [mock_program]
        agent.defense_store.save.return_value = "prog_low"
        agent.scientific_memory.save_theory.return_value = "thr_low"

        with patch.object(agent, "verify_program") as mock_vp:
            mock_vp.return_value = mock_report
            result = agent.run_reverse_engineering_pipeline(
                campaign_id="camp_low",
                victim=MagicMock(),
            )

        assert result["success"] is True
        assert result["verified"] is False
        assert result["accuracy"] == 0.4

    def test_pipeline_with_experiment_id(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        mock_report = _make_mock_report(accuracy=1.0)
        mock_theory = _make_mock_theory(theory_id="thr_exp")

        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("attack", 1, "ep_exp"),
        ]
        agent.synthesizer.synthesize.return_value = [mock_program]
        agent.defense_store.save.return_value = "prog_exp"
        agent.scientific_memory.save_theory.return_value = "thr_exp"

        with patch.object(agent, "verify_program") as mock_vp:
            mock_vp.return_value = mock_report
            result = agent.run_reverse_engineering_pipeline(
                campaign_id="camp_exp",
                victim=MagicMock(),
                experiment_id="exp_42",
            )

        assert result["success"] is True
        first_call_filter = agent.episodic_memory.filter_episodes.call_args_list[0].args[0]
        assert first_call_filter.experiment_id == "exp_42"

    def test_pipeline_exception_handling(self, agent: ResearcherAgent) -> None:
        agent.episodic_memory.filter_episodes.side_effect = RuntimeError("DB crash")

        result = agent.run_reverse_engineering_pipeline(
            campaign_id="camp_crash",
            victim=MagicMock(),
        )

        assert result["success"] is False
        assert "DB crash" in (result["error"] or "")

    def test_pipeline_with_exclude_prompts(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        mock_report = _make_mock_report()
        mock_theory = _make_mock_theory(theory_id="thr_excl")

        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("bomb", 1, "ep_1"),
            _make_mock_episode("hello", 0, "ep_2"),
        ]
        agent.synthesizer.synthesize.return_value = [mock_program]
        agent.defense_store.save.return_value = "prog_excl"
        agent.scientific_memory.save_theory.return_value = "thr_excl"

        with patch.object(agent, "verify_program") as mock_vp:
            mock_vp.return_value = mock_report
            result = agent.run_reverse_engineering_pipeline(
                campaign_id="camp_excl",
                victim=MagicMock(),
                exclude_prompts={"hello"},
            )

        assert result["success"] is True

    def test_pipeline_checkpoint_resume(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        mock_report = _make_mock_report()
        mock_theory = _make_mock_theory(theory_id="thr_chk")

        checkpoint = {
            "step": 2,
            "program": mock_program,
        }

        agent.episodic_memory.filter_episodes.side_effect = RuntimeError("should not be called")
        agent.defense_store.save.return_value = "prog_chk"
        agent.scientific_memory.save_theory.return_value = "thr_chk"

        with patch.object(agent, "verify_program") as mock_vp:
            mock_vp.return_value = mock_report
            result = agent.run_reverse_engineering_pipeline(
                campaign_id="camp_chk",
                victim=MagicMock(),
                checkpoint=checkpoint,
            )

        assert result["success"] is True
        assert result["program_id"] == "prog_chk"
        assert result["theory_id"] == "thr_chk"


# ===================================================================
# process_proposals
# ===================================================================


class TestProcessProposals:
    def test_synthesize_proposal(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("test", 1),
        ]
        agent.synthesizer.synthesize.return_value = [mock_program]

        results = agent.process_proposals([
            {"type": "synthesize", "campaign_id": "camp_1"},
        ])

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["proposal_type"] == "synthesize"
        assert results[0]["result"][0] is mock_program

    def test_verify_proposal(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        mock_report = _make_mock_report()

        with patch.object(agent, "verify_program", return_value=mock_report):
            results = agent.process_proposals([
                {"type": "verify", "program": mock_program, "victim": MagicMock()},
            ])

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["result"] is mock_report

    def test_store_program_proposal(self, agent: ResearcherAgent) -> None:
        agent.defense_store.save.return_value = "prog_123"

        results = agent.process_proposals([
            {
                "type": "store_program",
                "program": _make_mock_program(),
                "name": "test_prog",
                "confidence": 0.9,
                "status": "draft",
            },
        ])

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["result"] == "prog_123"

    def test_store_theory_proposal(self, agent: ResearcherAgent) -> None:
        agent.scientific_memory.save_theory.return_value = "thr_123"

        results = agent.process_proposals([
            {"type": "store_theory", "theory": _make_mock_theory()},
        ])

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["result"] == "thr_123"

    def test_unknown_proposal_type(self, agent: ResearcherAgent) -> None:
        results = agent.process_proposals([
            {"type": "unknown_type"},
        ])

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "unknown" in (results[0]["error"] or "").lower()

    def test_proposal_exception_handling(self, agent: ResearcherAgent) -> None:
        agent.episodic_memory.filter_episodes.side_effect = RuntimeError("fail")

        results = agent.process_proposals([
            {"type": "synthesize", "campaign_id": "camp_1"},
        ])

        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["error"] is not None


# ===================================================================
# error_tolerance validation
# ===================================================================


class TestErrorToleranceValidation:
    def test_valid_tolerance_passed_through(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("test", 1),
        ]
        agent.synthesizer.synthesize.return_value = [mock_program]

        program, stats = agent.synthesize_from_campaign("camp_1", error_tolerance=0.3)

        assert program is mock_program

    def test_tolerance_clamped_to_zero(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("test", 1),
        ]
        agent.synthesizer.synthesize.return_value = [mock_program]

        program, stats = agent.synthesize_from_campaign("camp_1", error_tolerance=-0.5)

        assert program is mock_program

    def test_tolerance_clamped_to_one(self, agent: ResearcherAgent) -> None:
        mock_program = _make_mock_program()
        agent.episodic_memory.filter_episodes.return_value = [
            _make_mock_episode("test", 1),
        ]
        agent.synthesizer.synthesize.return_value = [mock_program]

        program, stats = agent.synthesize_from_campaign("camp_1", error_tolerance=1.5)

        assert program is mock_program


# ===================================================================
# Default synthesizer creation
# ===================================================================


class TestDefaultSynthesizer:
    def test_creates_default_synthesizer(self, agent_no_synth: ResearcherAgent) -> None:
        assert agent_no_synth.synthesizer is not None
        assert agent_no_synth.synthesizer.max_depth == 3
        assert agent_no_synth.synthesizer.beam_width == 200
        assert agent_no_synth.synthesizer.timeout == 30
        assert agent_no_synth.synthesizer.use_cache is True


# ===================================================================
# Verifier cache
# ===================================================================


class TestVerifierCache:
    def test_get_verifier_cache_info(self, agent: ResearcherAgent) -> None:
        info = agent.get_verifier_cache_info()
        assert "cached_victims" in info
        assert "has_shared_verifier" in info
        assert info["has_shared_verifier"] is False

    def test_clear_verifier_cache(self, agent: ResearcherAgent) -> None:
        agent._verifier_cache["vic_1"] = MagicMock()
        assert len(agent._verifier_cache) == 1

        agent.clear_verifier_cache()

        assert len(agent._verifier_cache) == 0


# ===================================================================
# Causal Graph integration
# ===================================================================


class TestCausalGraph:
    def test_causal_graph_stored(self) -> None:
        episodic = MagicMock()
        defense_store = MagicMock()
        sci_mem = MagicMock()
        causal_graph = MagicMock()
        causal_graph.name = "test_graph"

        agent = ResearcherAgent(
            episodic_memory=episodic,
            defense_store=defense_store,
            scientific_memory=sci_mem,
            causal_graph=causal_graph,
        )

        assert agent.causal_graph is causal_graph
        assert agent.causal_graph.name == "test_graph"

    def test_causal_graph_none_by_default(self, agent: ResearcherAgent) -> None:
        assert agent.causal_graph is None
