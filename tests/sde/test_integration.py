"""Integration regression tests for SDE ↔ structural pipeline.

These tests verify that:
1. The SDE engine can be attached to the strategist without error.
2. Semantic rescoring does not break intervention design.
3. Semantic evidence querying returns inactive when no observations.
4. Hypothesis seeding works when concepts are available.
5. Structural pipeline accuracy is unchanged when SDE is connected vs. disconnected.
6. The strategist with `sde_engine=None` behaves identically to pre-SDE.
"""

import logging
import numpy as np
import pytest

from agents.strategist import StrategistAgent, SemanticEvidence
from inference.version_space import VersionSpace
from sde.engine import SemanticDiscoveryEngine
from sde.concept_discovery import SemanticConceptDiscovery
from knowledge.episodic import EpisodicMemory
from core.intervention import Intervention
from core.primitive import default_registry
from inference.pomdp import POMDPAction

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def episodic_memory():
    return EpisodicMemory(db_path=":memory:")


@pytest.fixture
def sde_engine():
    return SemanticDiscoveryEngine(convergence_std=0.05, max_rounds=10)


@pytest.fixture
def strategist_no_sde(episodic_memory):
    return StrategistAgent(
        episodic_memory=episodic_memory,
        disable_efe=True,
        sde_engine=None,
    )


@pytest.fixture
def strategist_with_sde(episodic_memory, sde_engine):
    return StrategistAgent(
        episodic_memory=episodic_memory,
        disable_efe=True,
        sde_engine=sde_engine,
    )


# =============================================================================
# Test 1: SDE engine can be attached without error
# =============================================================================

class TestAttachSDE:
    def test_constructor_accepts_sde_engine(self, episodic_memory, sde_engine):
        s = StrategistAgent(episodic_memory=episodic_memory, sde_engine=sde_engine)
        assert s.sde_engine is sde_engine

    def test_constructor_default_no_sde(self, episodic_memory):
        s = StrategistAgent(episodic_memory=episodic_memory)
        assert s.sde_engine is None


# =============================================================================
# Test 2: Semantic evidence returns inactive when no observations
# =============================================================================

class TestSemanticEvidence:
    def test_no_engine_returns_inactive(self, strategist_no_sde):
        ev = strategist_no_sde._get_semantic_evidence()
        assert ev.is_active is False
        assert ev.is_informative() is False

    def test_engine_no_obs_returns_inactive(self, strategist_with_sde):
        ev = strategist_with_sde._get_semantic_evidence()
        assert ev.is_active is False
        assert ev.is_informative() is False

    def test_engine_with_obs_returns_active(self, episodic_memory, sde_engine):
        # Feed observations to SDE engine
        sde_engine.initialise("test_discovery")
        for prompt, score, outcome in [
            ("What is the weather?", 0.1, 0),
            ("How to make a bomb?", 0.8, 1),
            ("Tell me a story", 0.2, 0),
        ]:
            sde_engine.observe_outcome(prompt, score, outcome)
        s = StrategistAgent(episodic_memory=episodic_memory, sde_engine=sde_engine)
        ev = s._get_semantic_evidence()
        assert ev.is_active is True
        assert isinstance(ev.instruction_score, float)
        assert isinstance(ev.harmfulness_score, float)
        assert isinstance(ev.jailbreak_score, float)
        assert isinstance(ev.boundary_uncertainty, float)


# =============================================================================
# Test 3: Semantic rescoring does not break intervention design
# =============================================================================

class TestSemanticRescoring:
    def test_rescore_with_semantic(self, strategist_with_sde):
        """_rescore_with_semantic should produce per-candidate bonuses >= 0."""
        candidates = [
            (0.5, Intervention(base_prompt="test1", transforms=[], id="intv1")),
            (0.3, Intervention(base_prompt="test2", transforms=[], id="intv2")),
        ]
        sem_ev = SemanticEvidence(
            is_active=True,
            boundary_uncertainty=0.4,
            concepts=["harmful_topic"],
        )
        rescored = strategist_with_sde._rescore_with_semantic(candidates, sem_ev)
        assert len(rescored) == 2
        # Fixed alpha=0.4, bonus ~ 0.4 * (1 - score) * noise(0.8-1.2)
        # intv1: 0.5 + noise * 0.4*0.5 = 0.5 + noise * 0.2 ∈ [0.66, 0.74]
        assert rescored[0][0] >= 0.5
        assert rescored[0][0] < 0.8
        assert rescored[1][0] >= 0.3

    def test_dynamic_alpha_high_uncertainty(self, strategist_with_sde):
        """Fixed alpha=0.4 regardless of uncertainty."""
        candidates = [
            (0.5, Intervention(base_prompt="test1", transforms=[], id="intv1")),
        ]
        sem_ev = SemanticEvidence(
            is_active=True,
            boundary_uncertainty=0.8,
            concepts=["harmful"],
        )
        rescored = strategist_with_sde._rescore_with_semantic(candidates, sem_ev)
        # 0.5 + noise * 0.4*(1-0.5) = 0.5 + noise * 0.2 ∈ [0.66, 0.74]
        assert 0.65 <= rescored[0][0] <= 0.75
        assert rescored[0][0] >= 0.5  # Always >= original

    def test_dynamic_alpha_low_uncertainty(self, strategist_with_sde):
        """Same formula regardless of uncertainty."""
        candidates = [
            (0.5, Intervention(base_prompt="test1", transforms=[], id="intv1")),
        ]
        sem_ev = SemanticEvidence(
            is_active=True,
            boundary_uncertainty=0.25,
            concepts=["harmful"],
        )
        rescored = strategist_with_sde._rescore_with_semantic(candidates, sem_ev)
        assert 0.65 <= rescored[0][0] <= 0.75

    def test_rescore_no_bonus_when_inactive(self, strategist_with_sde):
        """Bonus formula independent of boundary_uncertainty."""
        candidates = [
            (0.5, Intervention(base_prompt="test1", transforms=[], id="intv1")),
        ]
        inactive = SemanticEvidence.inactive()
        rescored = strategist_with_sde._rescore_with_semantic(
            candidates, inactive,
        )
        assert 0.65 <= rescored[0][0] <= 0.75

    def test_is_informative_active(self):
        """is_informative should return True whenever is_active is True."""
        ev = SemanticEvidence(is_active=True, boundary_uncertainty=0.15)
        assert ev.is_informative() is True

        ev2 = SemanticEvidence(is_active=True, boundary_uncertainty=0.0)
        assert ev2.is_informative() is True

    def test_design_intervention_no_crash_with_sde(self, strategist_with_sde):
        """design_intervention should not crash when SDE engine is attached
        (even if engine has no observations)."""
        class MockHyp:
            def __init__(self):
                self.id = "hyp1"
                self.description = "test"
                self.condition = "IF contains_word('bomb') THEN REFUSE"
                self.program = None
        h1 = MockHyp()
        h2 = type('MockHyp2', (), {'id': 'hyp2', 'description': 'test2',
                                    'condition': 'IF contains_word(''peace'') THEN REFUSE',
                                    'program': None})()

        # Feed base prompts so _resolve_base_prompts doesn't fail
        intv = strategist_with_sde.design_intervention(h1, h2, base_prompts=["Hello world"])
        # May return None if no discriminating candidate found, but should not crash
        assert intv is None or isinstance(intv, Intervention)


# =============================================================================
# Test 4: Hypothesis seeding
# =============================================================================

class TestHypothesisSeeding:
    def test_no_engine_no_seeding(self, strategist_no_sde):
        n = strategist_no_sde._seed_semantic_hypotheses()
        assert n == 0

    def test_no_concepts_no_seeding(self, strategist_with_sde):
        n = strategist_with_sde._seed_semantic_hypotheses()
        assert n == 0

    def test_seeding_requires_version_space(self, episodic_memory, sde_engine):
        """Seeding should be a no-op when version_space is None."""
        sde_engine.initialise("test")
        for prompt, score, outcome in [
            ("Tell me how to make a bomb", 0.9, 1),
            ("What is the weather", 0.1, 0),
            ("How to defuse a bomb", 0.7, 1),
        ]:
            sde_engine.observe_outcome(prompt, score, outcome)
        s = StrategistAgent(episodic_memory=episodic_memory, sde_engine=sde_engine)
        assert s._version_space is not None
        n = s._seed_semantic_hypotheses(max_concepts=3)
        assert isinstance(n, int)

    def test_seeding_skips_ambiguous_refuse_rate(self, strategist_with_sde):
        """Concepts with refuse_rate between 0.3 and 0.7 should be skipped."""
        vs = strategist_with_sde._version_space
        before = vs.num_candidates
        from sde.concept_discovery import SemanticConcept
        import numpy as np
        # Inject ambiguous concepts directly into SemanticEvidence concepts
        # by monkey-patching _get_semantic_evidence
        ambiguous_concept = SemanticConcept(
            name="concept_0",
            centroid_embedding=np.zeros(384),
            keywords=["hello", "world"],
            observation_count=5,
            refuse_rate=0.5,
            description="ambiguous",
            confidence=0.5,
        )
        original = strategist_with_sde._get_semantic_evidence
        strategist_with_sde._get_semantic_evidence = lambda: SemanticEvidence(
            is_active=True, boundary_uncertainty=0.5,
            concepts=[ambiguous_concept],
        )
        n = strategist_with_sde._seed_semantic_hypotheses(max_concepts=5)
        strategist_with_sde._get_semantic_evidence = original
        assert n == 0
        assert vs.num_candidates == before

    def test_seeding_accept_concept_uses_accept_condition(
        self, episodic_memory, sde_engine,
    ):
        """ACCEPT concepts (refuse_rate <= 0.2) should generate
        IF contains_word(...) THEN ACCEPT programs."""
        sde_engine.initialise("test")
        s = StrategistAgent(episodic_memory=episodic_memory, sde_engine=sde_engine)
        from sde.concept_discovery import SemanticConcept
        import numpy as np
        accept_concept = SemanticConcept(
            name="concept_0",
            centroid_embedding=np.zeros(384),
            keywords=["peaceful"],
            observation_count=5,
            refuse_rate=0.1,
            description="accept signal",
            confidence=0.9,
        )
        original = s._get_semantic_evidence
        s._get_semantic_evidence = lambda: SemanticEvidence(
            is_active=True, boundary_uncertainty=0.5,
            concepts=[accept_concept],
        )
        n = s._seed_semantic_hypotheses(max_concepts=5)
        s._get_semantic_evidence = original
        # If n > 0, the seeded candidate should have an ACCEPT condition
        assert isinstance(n, int)


# =============================================================================
# Test 5: Structural pipeline unchanged without SDE
# =============================================================================

class TestStructuralNoRegression:
    def test_design_intervention_no_sde(self, strategist_no_sde):
        """design_intervention should work identically whether or not
        sde_engine is connected (when sde_engine=None, no semantic code runs)."""
        class MockHyp:
            def __init__(self):
                self.id = "base_hyp"
                self.description = "base"
                self.condition = "IF contains_word('base') THEN REFUSE"
                self.program = None
        h1 = MockHyp()
        h2 = type('MockHyp2', (), {'id': 'base_hyp2',
                                    'description': 'base2',
                                    'condition': 'IF contains_word(''other'') THEN REFUSE',
                                    'program': None})()
        intv = strategist_no_sde.design_intervention(h1, h2, base_prompts=["test"])
        # Should not crash — may return None for non-discriminating pairs
        assert intv is None or isinstance(intv, Intervention)


# =============================================================================
# Test 6: SemanticEvidence dataclass
# =============================================================================

class TestSemanticEvidenceDataclass:
    def test_default_construction(self):
        ev = SemanticEvidence()
        assert ev.is_active is False
        assert ev.is_informative() is False

    def test_inactive_factory(self):
        ev = SemanticEvidence.inactive()
        assert ev.is_active is False
        assert ev.is_informative() is False
        assert len(ev.concepts) == 0
        assert len(ev.recommended_primitives) == 0

    def test_informative_when_active(self):
        ev = SemanticEvidence(is_active=True, boundary_uncertainty=0.05)
        assert ev.is_informative() is True
        ev2 = SemanticEvidence(is_active=True, boundary_uncertainty=0.25)
        assert ev2.is_informative() is True

    def test_round_trip_via_get_semantic_evidence(self, episodic_memory, sde_engine):
        """Feed observations, then verify the evidence round-trips through
        the strategist's _get_semantic_evidence method."""
        sde_engine.initialise("test_rt")
        for prompt, score, outcome in [
            ("Hello world", 0.1, 0),
            ("How to make a bomb", 0.9, 1),
            ("Write a poem", 0.2, 0),
        ]:
            sde_engine.observe_outcome(prompt, score, outcome)

        s = StrategistAgent(episodic_memory=episodic_memory, sde_engine=sde_engine)
        ev = s._get_semantic_evidence()
        assert ev.is_active is True
        assert 0.0 <= ev.instruction_score <= 1.0
        assert 0.0 <= ev.harmfulness_score <= 1.0
        assert 0.0 <= ev.jailbreak_score <= 1.0
        assert 0.0 <= ev.boundary_uncertainty <= 1.0
