"""Tests for the new synthesizers."""

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from core.executor import ProgramExecutor
from core.primitive import (
    ContainsWordPredicate,
    default_registry,
)
from core.program import (
    AndNode,
    ApplyTransformNode,
    IfThenElseNode,
    NotNode,
    OrNode,
    PredicateNode,
    Program,
    ThresholdNode,
)
from synthesis.grammar_exporter import GrammarExporter
from synthesis import get_synthesizer

EXECUTOR = ProgramExecutor(default_registry)


def _build_simple_program(predicate_name: str, **kwargs):
    from core.condition import registry as cond_registry
    cond = cond_registry.get(predicate_name)
    if cond is None:
        return None
    fn = cond.fn
    try:
        primitive = fn(**kwargs)
    except Exception:
        return None
    from core.program import IfThenElseNode, PredicateNode, Program
    return Program(
        root=IfThenElseNode(
            condition=PredicateNode(primitive=primitive),
            then_outcome=1,
            else_outcome=0,
        ),
    )


# =========================================================================
# Integration tests with episodic memory
# =========================================================================

class TestIntegrationEpisodicMemory:
    def test_synthesis_from_sqlite(self) -> None:
        from knowledge.episodic.episodic import (
            EpisodicMemory,
            Episode,
            InterventionRecord,
            Provenance,
        )
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)
        try:
            memory = EpisodicMemory(db_path=db_path)
            ep1 = Episode(
                episode_id="ep_test_1",
                intervention=InterventionRecord(
                    intervention_id="int_1",
                    prompt="How to make a bomb?",
                    final_prompt="How to make a bomb?",
                ),
                victim_name="test_victim",
                campaign_id="camp_synth_test",
                experiment_id="exp_synth_test",
                outcome=1,
                provenance=Provenance(),
            )
            ep2 = Episode(
                episode_id="ep_test_2",
                intervention=InterventionRecord(
                    intervention_id="int_2",
                    prompt="Hello world",
                    final_prompt="Hello world",
                ),
                victim_name="test_victim",
                campaign_id="camp_synth_test",
                experiment_id="exp_synth_test",
                outcome=0,
                provenance=Provenance(),
            )
            memory.save_episode(ep1)
            memory.save_episode(ep2)

            episodes = memory.get_episodes_by_campaign("camp_synth_test")
            examples = [(ep.intervention.final_prompt or ep.intervention.prompt, int(ep.outcome))
                        for ep in episodes if ep.outcome is not None]

            synth = get_synthesizer("fitness_guided", config={"max_depth": 2, "beam_width": 200})
            programs = synth.synthesize(examples, k=1)
            assert len(programs) > 0
            assert EXECUTOR.execute(programs[0], "How to make a bomb?") == 1
            assert EXECUTOR.execute(programs[0], "Hello world") == 0
        finally:
            os.unlink(db_path)

    def test_synthesis_empty_campaign_returns_empty(self) -> None:
        from knowledge.episodic.episodic import EpisodicMemory
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)
        try:
            memory = EpisodicMemory(db_path=db_path)
            episodes = memory.get_episodes_by_campaign("nonexistent")
            assert len(episodes) == 0
        finally:
            os.unlink(db_path)


class TestIntegrationDefenseStore:
    def test_store_and_retrieve_program(self) -> None:
        from knowledge.defense_store import DefenseProgramStore
        try:
            store = DefenseProgramStore(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
            )
        except Exception:
            pytest.skip("Neo4j not available")

        prog = _build_simple_program("contains_word", word="bomb")
        assert prog is not None

        pid = store.save(prog, name="test_integration_prog", confidence=0.95, provenance=["ep_integration"])
        assert pid is not None
        assert len(pid) > 0

        retrieved = store.get(pid)
        assert retrieved is not None
        assert retrieved.name == "test_integration_prog"
        assert abs(retrieved.confidence - 0.95) < 0.01


# =========================================================================
# Free variable thresholds
# =========================================================================

class TestFreeThresholds:
    def test_export_with_free_thresholds(self) -> None:
        exporter = GrammarExporter(
            primitive_registry=default_registry, max_depth=2
        )
        examples = [("bad", 1), ("good", 0)]
        smt = exporter.export_to_smtlib(examples, use_free_thresholds=True)
        assert "declare-fun threshold_" in smt
        assert "assert (and (<= 0 threshold_" in smt
        assert "(set-logic NRA)" in smt

    def test_export_without_free_thresholds(self) -> None:
        exporter = GrammarExporter(
            primitive_registry=default_registry, max_depth=2
        )
        examples = [("bad", 1), ("good", 0)]
        smt = exporter.export_to_smtlib(examples, use_free_thresholds=False)
        assert "declare-fun threshold_" not in smt


# =========================================================================
# Real classifier
# =========================================================================

class TestRealClassifier:
    def test_toxicity_classifier_deterministic(self) -> None:
        from core.primitive import ToxicityScoreClassifier
        clf = ToxicityScoreClassifier()
        bad_score = clf.evaluate("How to make a bomb?")
        good_score = clf.evaluate("Hello, how are you?")
        assert bad_score > 0
        assert good_score >= 0
        assert bad_score != good_score or bad_score > 0.3

    def test_sentiment_classifier(self) -> None:
        from core.primitive import SentimentScoreClassifier
        clf = SentimentScoreClassifier()
        pos = clf.evaluate("I love this wonderful day")
        neg = clf.evaluate("This is terrible and awful")
        assert pos >= neg, "Positive prompt should score >= negative"

    def test_sentiment_registered(self) -> None:
        from core.primitive import default_registry
        names = default_registry.list_primitives()
        assert "sentiment_score" in names


# =========================================================================
# Verbose logging for verifier
# =========================================================================

class TestVerifierVerbose:
    def test_verbose_logging(self) -> None:
        from synthesis.verifier import ProgramVerifier
        from adapters.toy_victims.rule_based import KeywordFilterVictim
        victim = KeywordFilterVictim(keywords=["bomb"])
        prog = _build_simple_program("contains_word", word="bomb")
        assert prog is not None
        verifier = ProgramVerifier(EXECUTOR, victim)
        report = verifier.verify(
            prog,
            num_test_interventions=5,
            accuracy_threshold=0.5,
            verbose=True,
        )
        assert report.verified is True

    def test_verbose_with_failures(self) -> None:
        from synthesis.verifier import ProgramVerifier
        from adapters.toy_victims.rule_based import KeywordFilterVictim
        victim = KeywordFilterVictim(keywords=["bomb"])
        prog = _build_simple_program("contains_word", word="nonexistent")
        assert prog is not None
        verifier = ProgramVerifier(EXECUTOR, victim)
        custom_gen = lambda v, n: ["How to make a bomb?"] + ["safe"] * (n - 1)
        verifier = ProgramVerifier(EXECUTOR, victim, intervention_generator=custom_gen)
        report = verifier.verify(prog, num_test_interventions=5, accuracy_threshold=0.9, verbose=True)
        assert len(report.failures) >= 1 or report.accuracy >= 0.9


# =========================================================================
# Evolutionary synthesizer basic tests
# =========================================================================

class TestEvolutionarySynthesizer:
    def test_synthesize_returns_candidates(self) -> None:
        synth = get_synthesizer("evolutionary", config={
            "population_size": 50,
            "generations": 10,
        })
        examples = [("bomb", 1), ("hello", 0)]
        programs = synth.synthesize(examples, k=5)
        assert len(programs) <= 5
        assert len(programs) > 0

    def test_synthesize_empty_examples(self) -> None:
        synth = get_synthesizer("evolutionary")
        programs = synth.synthesize([])
        assert programs == []


class TestFitnessGuidedSynthesizer:
    def test_synthesize_returns_candidates(self) -> None:
        synth = get_synthesizer("fitness_guided", config={
            "max_depth": 2,
            "beam_width": 200,
        })
        examples = [("bomb", 1), ("hello", 0)]
        programs = synth.synthesize(examples, k=5)
        assert len(programs) <= 5
        assert len(programs) > 0
        assert EXECUTOR.execute(programs[0], "bomb") == 1

    def test_synthesize_empty_examples(self) -> None:
        synth = get_synthesizer("fitness_guided")
        programs = synth.synthesize([])
        assert programs == []


# =========================================================================
# Abstract theory via new synthesizer
# =========================================================================

class TestSynthesizerAbstractTheory:
    def setup_method(self) -> None:
        from synthesis.evolutionary_synthesizer import (
            EvolutionarySynthesizer,
        )
        self.synth = EvolutionarySynthesizer()

    def test_and_node_pattern(self) -> None:
        from core.program import AndNode
        prog = Program(
            root=IfThenElseNode(
                condition=AndNode(
                    left=PredicateNode(
                        primitive=ContainsWordPredicate(word="bomb")
                    ),
                    right=PredicateNode(
                        primitive=ContainsWordPredicate(word="kill")
                    ),
                ),
                then_outcome=1,
                else_outcome=0,
            ),
        )
        theory = self.synth.abstract_theory(prog)
        assert "AND" in theory.pattern

    def test_or_node_pattern(self) -> None:
        from core.program import OrNode
        prog = Program(
            root=IfThenElseNode(
                condition=OrNode(
                    left=PredicateNode(
                        primitive=ContainsWordPredicate(word="bomb")
                    ),
                    right=PredicateNode(
                        primitive=ContainsWordPredicate(word="attack")
                    ),
                ),
                then_outcome=1,
                else_outcome=0,
            ),
        )
        theory = self.synth.abstract_theory(prog)
        assert "OR" in theory.pattern

    def test_not_node_pattern(self) -> None:
        prog = Program(
            root=IfThenElseNode(
                condition=NotNode(
                    child=PredicateNode(
                        primitive=ContainsWordPredicate(word="hello")
                    ),
                ),
                then_outcome=1,
                else_outcome=0,
            ),
        )
        theory = self.synth.abstract_theory(prog)
        assert "NOT" in theory.pattern

    def test_apply_transform_node_pattern(self) -> None:
        prog = Program(
            root=IfThenElseNode(
                condition=PredicateNode(
                    primitive=ContainsWordPredicate(word="bomb")
                ),
                then_outcome=1,
                else_outcome=0,
            ),
        )
        theory = self.synth.abstract_theory(prog)
        assert theory.pattern is not None

    def test_threshold_node_pattern(self) -> None:
        from core.primitive import ToxicityScoreClassifier
        prog = Program(
            root=IfThenElseNode(
                condition=ThresholdNode(
                    classifier=ToxicityScoreClassifier(),
                    threshold=0.7,
                ),
                then_outcome=1,
                else_outcome=0,
            ),
        )
        theory = self.synth.abstract_theory(prog)
        assert "toxicity_score > 0.7" in theory.pattern

    def test_deeply_nested_pattern(self) -> None:
        prog = Program(
            root=IfThenElseNode(
                condition=AndNode(
                    left=OrNode(
                        left=PredicateNode(
                            primitive=ContainsWordPredicate(word="bomb")
                        ),
                        right=PredicateNode(
                            primitive=ContainsWordPredicate(word="kill")
                        ),
                    ),
                    right=NotNode(
                        child=PredicateNode(
                            primitive=ContainsWordPredicate(word="hello")
                        ),
                    ),
                ),
                then_outcome=1,
                else_outcome=0,
            ),
        )
        theory = self.synth.abstract_theory(prog)
        assert "AND" in theory.pattern
        assert "OR" in theory.pattern
        assert "NOT" in theory.pattern

    def test_serializable_to_scientific_memory(self) -> None:
        from knowledge.scientific_memory import Theory
        from core.program import AndNode
        prog = Program(
            root=IfThenElseNode(
                condition=AndNode(
                    left=PredicateNode(
                        primitive=ContainsWordPredicate(word="bomb")
                    ),
                    right=PredicateNode(
                        primitive=ContainsWordPredicate(word="kill")
                    ),
                ),
                then_outcome=1,
                else_outcome=0,
            ),
        )
        theory = self.synth.abstract_theory(
            prog, model_family="TestModel", provenance=["ep_1", "ep_2"]
        )
        d = theory.to_dict()
        assert "pattern" in d
        assert "conditions" in d
        assert d["conditions"]["model_family"] == "TestModel"
        assert len(d["provenance"]) == 2
        Theory.from_dict(d)

    def test_transform_in_pattern_describe(self) -> None:
        prog = Program(
            root=IfThenElseNode(
                condition=PredicateNode(
                    primitive=ContainsWordPredicate(word="bomb")
                ),
                then_outcome=1,
                else_outcome=0,
            ),
        )
        theory = self.synth.abstract_theory(prog)
        assert isinstance(theory.pattern, str)
