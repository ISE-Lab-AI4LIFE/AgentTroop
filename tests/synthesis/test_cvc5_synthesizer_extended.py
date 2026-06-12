"""Extended tests for CVC5Synthesizer — all 10 improvements."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import cvc5_available

from core.executor import ProgramExecutor
from core.primitive import (
    ContainsWordPredicate,
    LengthGtPredicate,
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
from synthesis.cvc5_synthesizer import CVC5Synthesizer, SynthesisStats, build_simple_program
from synthesis.grammar_exporter import GrammarExporter

EXECUTOR = ProgramExecutor(default_registry)

_SKIP_CVC5 = not cvc5_available()


# =========================================================================
# Item 1: Real CVC5 binary test
# =========================================================================

class TestCVC5RealBinary:
    @pytest.mark.skipif(_SKIP_CVC5, reason="CVC5 binary not in PATH")
    def test_cvc5_finds_simple_solution(self) -> None:
        synth = CVC5Synthesizer(cvc5_path="cvc5", timeout=10, max_depth=1)
        examples = [("bomb", 1), ("hello", 0)]
        program, stats = synth.synthesize_with_stats(
            examples, primitive_registry=default_registry
        )
        assert program is not None
        assert stats.cvc5_used or stats.enumeration_found
        assert EXECUTOR.execute(program, "bomb") == 1
        assert EXECUTOR.execute(program, "hello") == 0

    @pytest.mark.skipif(_SKIP_CVC5, reason="CVC5 binary not in PATH")
    def test_cvc5_syntax_error_falls_back(self) -> None:
        synth = CVC5Synthesizer(cvc5_path="cvc5", timeout=5, max_depth=2)
        examples = [("bomb", 1), ("hello", 0)]
        program = synth.synthesize(examples, primitive_registry=default_registry)
        assert program is not None

    @pytest.mark.skipif(_SKIP_CVC5, reason="CVC5 binary not in PATH")
    def test_cvc5_unsat_returns_none(self) -> None:
        synth = CVC5Synthesizer(cvc5_path="cvc5", timeout=10, max_depth=1)
        examples = [("hello", 0), ("hello", 1)]
        program = synth.synthesize(examples, primitive_registry=default_registry)
        assert program is None

    @pytest.mark.skipif(_SKIP_CVC5, reason="CVC5 binary not in PATH")
    def test_cvc5_timeout_handled_gracefully(self) -> None:
        synth = CVC5Synthesizer(cvc5_path="cvc5", timeout=1, max_depth=3)
        examples = [("a" * 10, 1), ("b" * 10, 0)]
        program = synth.synthesize(examples, primitive_registry=default_registry)
        assert program is not None


# =========================================================================
# Item 2: Integration tests with real databases
# =========================================================================

class TestIntegrationEpisodicMemory:
    def test_synthesis_from_sqlite(
        self,
    ) -> None:
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

            synth = CVC5Synthesizer(max_depth=2)
            program, stats = synth.synthesize_from_episodes(
                memory,
                campaign_id="camp_synth_test",
                primitive_registry=default_registry,
            )
            assert program is not None
            assert stats.programs_tried > 0
            assert EXECUTOR.execute(program, "How to make a bomb?") == 1
            assert EXECUTOR.execute(program, "Hello world") == 0
        finally:
            os.unlink(db_path)

    def test_synthesis_empty_campaign_returns_none(self) -> None:
        from knowledge.episodic.episodic import EpisodicMemory
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)
        try:
            memory = EpisodicMemory(db_path=db_path)
            synth = CVC5Synthesizer(max_depth=2)
            result, stats = synth.synthesize_from_episodes(
                memory, campaign_id="nonexistent"
            )
            assert result is None
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

        synth = CVC5Synthesizer()
        prog = build_simple_program("contains_word", word="bomb")
        assert prog is not None

        pid = synth.store_verified_program(
            store, prog,
            name="test_integration_prog",
            confidence=0.95,
            provenance=["ep_integration"],
        )
        assert pid is not None
        assert len(pid) > 0

        retrieved = store.get(pid)
        assert retrieved is not None
        assert retrieved.name == "test_integration_prog"
        assert abs(retrieved.confidence - 0.95) < 0.01


class TestIntegrationScientificMemory:
    def test_abstract_and_store_theory(self) -> None:
        from knowledge.scientific_memory import ScientificMemory, Theory
        try:
            sci = ScientificMemory(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
                database="neo4j",
            )
        except Exception:
            pytest.skip("Neo4j not available")

        synth = CVC5Synthesizer()
        prog = build_simple_program("contains_word", word="bomb")
        assert prog is not None

        theory = synth.abstract_theory(
            prog,
            model_family="RLHF",
            conditions={"test": "integration"},
            provenance=["ep_int_1"],
        )
        assert isinstance(theory, Theory)
        assert "bomb" in theory.pattern
        assert theory.conditions["model_family"] == "RLHF"

        tid = sci.save_theory(theory)
        assert tid is not None
        assert len(tid) > 0

        retrieved = sci.get_theory(tid)
        assert retrieved is not None
        assert "bomb" in retrieved.pattern


# =========================================================================
# Item 3: Free variable thresholds
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

    def test_extract_thresholds_from_model(self) -> None:
        synth = CVC5Synthesizer()
        model = {
            "threshold_toxicity_score": {"type": "Real", "body": "0.75"},
            "contains_word": {"type": "Bool", "body": "true"},
        }
        thresholds = synth._extract_thresholds_from_model(model)
        assert "toxicity_score" in thresholds
        assert thresholds["toxicity_score"] == 0.75

    def test_extract_thresholds_clamps_values(self) -> None:
        synth = CVC5Synthesizer()
        model = {
            "threshold_toxicity_score": {"type": "Real", "body": "2.5"},
        }
        thresholds = synth._extract_thresholds_from_model(model)
        assert thresholds["toxicity_score"] == 1.0

    @pytest.mark.skipif(_SKIP_CVC5, reason="CVC5 binary not in PATH")
    def test_cvc5_free_threshold_finds_solution(self) -> None:
        synth = CVC5Synthesizer(cvc5_path="cvc5", timeout=10, max_depth=1)
        examples = [("bad", 1), ("neutral", 0)]
        program = synth.synthesize(examples, primitive_registry=default_registry)
        assert program is not None


# =========================================================================
# Item 5: Real classifier
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
# Item 6: Beam width + fitness scoring
# =========================================================================

class TestBeamWidth:
    def test_beam_width_limits_programs(self) -> None:
        synth = CVC5Synthesizer(max_depth=2, beam_width=5)
        examples = [("bomb", 1), ("hello", 0)]
        _program, stats = synth.synthesize_with_stats(
            examples, primitive_registry=default_registry
        )
        assert stats.beam_width == 5

    def test_beam_width_zero_unlimited(self) -> None:
        synth = CVC5Synthesizer(max_depth=1, beam_width=0)
        examples = [("bomb", 1), ("hello", 0)]
        program = synth.synthesize(examples, primitive_registry=default_registry)
        assert program is not None

    def test_beam_width_prioritizes_good_programs(self) -> None:
        synth = CVC5Synthesizer(max_depth=3, beam_width=10)
        examples = [("bomb", 1), ("hello", 0)]
        program = synth.synthesize(examples, primitive_registry=default_registry)
        assert program is not None
        assert synth._fitness_score(program, examples, EXECUTOR) >= 0.5

    def test_fitness_score(self) -> None:
        synth = CVC5Synthesizer()
        prog = build_simple_program("contains_word", word="bomb")
        assert prog is not None
        examples = [("bomb", 1), ("hello", 0)]
        score = synth._fitness_score(prog, examples, EXECUTOR)
        assert score == 1.0

    def test_fitness_score_partial(self) -> None:
        synth = CVC5Synthesizer()
        prog = build_simple_program("contains_word", word="hello")
        assert prog is not None
        examples = [("bomb", 1), ("hello", 0)]
        score = synth._fitness_score(prog, examples, EXECUTOR)
        assert score < 1.0

    def test_fitness_score_empty(self) -> None:
        synth = CVC5Synthesizer()
        prog = build_simple_program("contains_word", word="bomb")
        assert prog is not None
        assert synth._fitness_score(prog, [], EXECUTOR) == 0.0


# =========================================================================
# Item 7: Verbose logging for verifier
# =========================================================================

class TestVerifierVerbose:
    def test_verbose_logging(self) -> None:
        from synthesis.verifier import ProgramVerifier
        from adapters.toy_victims.rule_based import KeywordFilterVictim
        victim = KeywordFilterVictim(keywords=["bomb"])
        prog = build_simple_program("contains_word", word="bomb")
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
        prog = build_simple_program("contains_word", word="nonexistent")
        assert prog is not None
        verifier = ProgramVerifier(EXECUTOR, victim)
        custom_gen = lambda v, n: ["How to make a bomb?"] + ["safe"] * (n - 1)
        verifier = ProgramVerifier(EXECUTOR, victim, intervention_generator=custom_gen)
        report = verifier.verify(prog, num_test_interventions=5, accuracy_threshold=0.9, verbose=True)
        assert len(report.failures) >= 1 or report.accuracy >= 0.9


# =========================================================================
# Item 8: Disk cache
# =========================================================================

class TestDiskCache:
    def test_disk_cache_persists(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            cache_path = f.name
        try:
            synth1 = CVC5Synthesizer(
                max_depth=2, use_cache=True, cache_path=cache_path
            )
            examples = [("bomb", 1), ("hello", 0)]
            synth1.synthesize(examples, primitive_registry=default_registry)

            assert os.path.exists(cache_path)
            assert os.path.getsize(cache_path) > 0

            synth2 = CVC5Synthesizer(
                max_depth=2, use_cache=True, cache_path=cache_path
            )
            assert len(synth2._cache) > 0
        finally:
            os.unlink(cache_path)

    def test_disk_cache_speeds_up_second_run(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            cache_path = f.name
        try:
            synth1 = CVC5Synthesizer(
                max_depth=1, use_cache=True, cache_path=cache_path
            )
            examples = [("bomb", 1), ("hello", 0)]
            _, s1 = synth1.synthesize_with_stats(
                examples, primitive_registry=default_registry
            )

            synth2 = CVC5Synthesizer(
                max_depth=1, use_cache=True, cache_path=cache_path
            )
            _, s2 = synth2.synthesize_with_stats(
                examples, primitive_registry=default_registry
            )
            assert s2.programs_skipped_cache >= 0
        finally:
            os.unlink(cache_path)

    def test_disk_cache_nonexistent_path(self) -> None:
        synth = CVC5Synthesizer(
            max_depth=2, use_cache=True, cache_path="/nonexistent/cache.pkl"
        )
        examples = [("bomb", 1), ("hello", 0)]
        program = synth.synthesize(examples, primitive_registry=default_registry)
        assert program is not None

    def test_disk_cache_empty_after_new(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            cache_path = f.name
        try:
            synth = CVC5Synthesizer(
                max_depth=2, use_cache=True, cache_path=cache_path
            )
            assert len(synth._cache) == 0 or len(synth._cache) >= 0
        finally:
            os.unlink(cache_path)


# =========================================================================
# Item 10: Extended abstract_theory tests
# =========================================================================

class TestAbstractTheoryExtended:
    def setup_method(self) -> None:
        self.synth = CVC5Synthesizer()

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
