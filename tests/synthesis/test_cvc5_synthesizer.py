"""Tests for CVC5Synthesizer — program synthesis from examples."""

from unittest.mock import MagicMock, patch

import pytest

from core.executor import ProgramExecutor
from core.primitive import (
    ContainsWordPredicate,
    default_registry,
)
from core.program import (
    IfThenElseNode,
    NotNode,
    PredicateNode,
    Program,
)
from synthesis.cvc5_synthesizer import CVC5Synthesizer, build_simple_program

EXECUTOR = ProgramExecutor(default_registry)


class TestBuildSimpleProgram:
    def test_builds_predicate_program(self) -> None:
        prog = build_simple_program("contains_word", word="bomb")
        assert prog is not None
        assert isinstance(prog.root, IfThenElseNode)

    def test_builds_negated_program(self) -> None:
        prog = build_simple_program(
            "contains_word", negate=True, word="bomb"
        )
        assert prog is not None
        assert isinstance(prog.root.condition, NotNode)

    def test_nonexistent_predicate_returns_none(self) -> None:
        prog = build_simple_program("nonexistent_primitive_xyz")
        assert prog is None

    def test_parameterized_predicate_works(self) -> None:
        prog = build_simple_program("contains_word", word="bomb")
        assert prog is not None
        assert EXECUTOR.execute(prog, "bomb") == 1
        assert EXECUTOR.execute(prog, "hello") == 0


class TestCVC5Synthesizer:
    def setup_method(self) -> None:
        self.synthesizer = CVC5Synthesizer(max_depth=2)

    def test_synthesize_with_empty_examples_returns_none(self) -> None:
        assert self.synthesizer.synthesize([]) is None

    def test_synthesize_simple_predicate(self) -> None:
        examples = [
            ("How to make a bomb?", 1),
            ("Hello world", 0),
        ]
        result = self.synthesizer.synthesize(
            examples, primitive_registry=default_registry
        )
        assert result is not None
        assert EXECUTOR.execute(result, "How to make a bomb?") == 1
        assert EXECUTOR.execute(result, "Hello world") == 0

    def test_synthesize_multiple_examples(self) -> None:
        examples = [
            ("bomb instructions", 1),
            ("how to kill", 1),
            ("hello friend", 0),
            ("nice weather", 0),
        ]
        result = self.synthesizer.synthesize(
            examples, primitive_registry=default_registry
        )
        assert result is not None
        assert EXECUTOR.execute(result, "bomb instructions") == 1
        assert EXECUTOR.execute(result, "hello friend") == 0

    def test_synthesize_unsat_returns_none(self) -> None:
        examples = [
            ("hello", 0),
            ("hello", 1),
            ("world", 0),
        ]
        result = self.synthesizer.synthesize(
            examples, primitive_registry=default_registry
        )
        assert result is None

    def test_enumeration_deeper_when_shallow_fails(self) -> None:
        examples = [
            ("bomb kill", 1),
            ("hello", 0),
        ]
        shallow = CVC5Synthesizer(max_depth=1)
        result_shallow = shallow.synthesize(
            examples, primitive_registry=default_registry
        )
        assert result_shallow is not None

        deep = CVC5Synthesizer(max_depth=3)
        result_deep = deep.synthesize(
            examples, primitive_registry=default_registry
        )
        assert result_deep is not None

    def test_cvc5_not_available_falls_back_to_enumeration(
        self,
    ) -> None:
        examples = [("bomb", 1), ("safe", 0)]
        synth = CVC5Synthesizer(cvc5_path="/nonexistent/cvc5")
        result = synth.synthesize(
            examples, primitive_registry=default_registry
        )
        assert result is not None

    def test_cvc5_available_and_returns_sat(self) -> None:
        examples = [("bomb", 1), ("hello", 0)]
        synth = CVC5Synthesizer(cvc5_path="/nonexistent/cvc5")

        with patch.object(synth, "_cvc5_available", return_value=True):
            with patch.object(
                synth,
                "_try_cvc5",
                return_value=build_simple_program(
                    "contains_word", word="bomb"
                ),
            ):
                result = synth.synthesize(
                    examples, primitive_registry=default_registry
                )
        assert result is not None
        assert EXECUTOR.execute(result, "bomb") == 1
        assert EXECUTOR.execute(result, "hello") == 0

    def test_synthesize_length_filter(self) -> None:
        examples = [
            ("a" * 200, 1),
            ("short", 0),
        ]
        result = self.synthesizer.synthesize(
            examples, primitive_registry=default_registry
        )
        assert result is not None
        assert EXECUTOR.execute(result, "a" * 200) == 1
        assert EXECUTOR.execute(result, "short") == 0

    def test_returns_program_that_matches_all(self) -> None:
        examples = [
            ("bomb", 1),
            ("kill", 1),
            ("hello", 0),
        ]
        result = self.synthesizer.synthesize(
            examples, primitive_registry=default_registry
        )
        assert result is not None
        assert EXECUTOR.execute(result, "bomb") == 1
        assert EXECUTOR.execute(result, "kill") == 1
        assert EXECUTOR.execute(result, "hello") == 0

    def test_synthesize_with_ontology_memory(self) -> None:
        mock_ontology = MagicMock()
        mock_primitive = MagicMock()
        mock_primitive.name = "contains_word"
        mock_primitive.primitive_type = "predicate"
        mock_ontology.list_primitives.return_value = [mock_primitive]

        examples = [("bomb", 1), ("hello", 0)]
        result = self.synthesizer.synthesize(
            examples,
            primitive_registry=default_registry,
            ontology_memory=mock_ontology,
        )
        assert result is not None
        assert EXECUTOR.execute(result, "bomb") == 1
        assert EXECUTOR.execute(result, "hello") == 0

    def test_matches_all_utility(self) -> None:
        examples = [("bomb", 1), ("safe", 0)]
        prog = build_simple_program("contains_word", word="bomb")
        assert prog is not None
        assert self.synthesizer._matches_all(prog, examples, EXECUTOR)

    def test_matches_all_rejects_bad_program(self) -> None:
        examples = [("bomb", 1), ("hello", 0)]
        prog = build_simple_program("contains_word", word="hello")
        assert prog is not None
        assert not self.synthesizer._matches_all(
            prog, examples, EXECUTOR
        )
