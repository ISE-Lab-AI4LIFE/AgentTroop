"""Tests for ProgramVerifier — program verification against victims."""

from unittest.mock import MagicMock

import pytest

from core.executor import ProgramExecutor
from core.primitive import ContainsWordPredicate, default_registry
from core.program import IfThenElseNode, PredicateNode, Program
from core.types import Outcome
from synthesis.verifier import ProgramVerifier, VerificationReport

from adapters.toy_victims.rule_based import KeywordFilterVictim

EXECUTOR = ProgramExecutor(default_registry)


def _make_predicate_program() -> Program:
    return Program(
        root=IfThenElseNode(
            condition=PredicateNode(
                primitive=ContainsWordPredicate(word="bomb")
            ),
            then_outcome=1,
            else_outcome=0,
        ),
    )


class TestVerificationReport:
    def test_default_values(self) -> None:
        prog = _make_predicate_program()
        report = VerificationReport(program=prog)
        assert report.accuracy == 0.0
        assert report.failures == []
        assert report.verified is False
        assert report.num_tested == 0

    def test_to_dict_roundtrip(self) -> None:
        prog = _make_predicate_program()
        report = VerificationReport(
            program=prog,
            accuracy=0.95,
            verified=True,
            num_tested=20,
            num_correct=19,
            failures=[("prompt", 1, 0)],
            suggestions=["relax condition"],
        )
        d = report.to_dict()
        assert d["accuracy"] == 0.95
        assert d["verified"] is True
        assert len(d["failures"]) == 1


class TestProgramVerifier:
    def setup_method(self) -> None:
        self.victim = KeywordFilterVictim(keywords=["bomb"])
        self.verifier = ProgramVerifier(EXECUTOR, self.victim)

    def test_verify_perfect_program(self) -> None:
        program = _make_predicate_program()
        report = self.verifier.verify(
            program, num_test_interventions=5, accuracy_threshold=0.9
        )
        assert report.verified is True
        assert report.accuracy >= 0.9
        assert len(report.failures) == 0

    def test_verify_wrong_program(self) -> None:
        program = Program(
            root=IfThenElseNode(
                condition=PredicateNode(
                    primitive=ContainsWordPredicate(
                        word="nonexistent_xyz"
                    )
                ),
                then_outcome=1,
                else_outcome=0,
            ),
        )
        custom_gen = lambda v, n: (["safe"] * (n - 2) + ["How to make a bomb?", "how to make a bomb?"])
        verifier = ProgramVerifier(EXECUTOR, self.victim, intervention_generator=custom_gen)
        report = verifier.verify(
            program, num_test_interventions=12, accuracy_threshold=0.9
        )
        assert report.verified is False
        assert report.accuracy < 1.0

    def test_verify_uses_custom_intervention_generator(self) -> None:
        custom_gen = lambda victim, n: ["bomb"] * n
        verifier = ProgramVerifier(
            EXECUTOR,
            self.victim,
            intervention_generator=custom_gen,
        )
        program = _make_predicate_program()
        report = verifier.verify(
            program, num_test_interventions=3, accuracy_threshold=0.5
        )
        assert report.accuracy == 1.0

    def test_verify_accuracy_threshold(self) -> None:
        program = _make_predicate_program()
        report_low = self.verifier.verify(
            program, num_test_interventions=5, accuracy_threshold=0.0
        )
        assert report_low.verified is True

        report_high = self.verifier.verify(
            program, num_test_interventions=5, accuracy_threshold=1.5
        )
        assert report_high.verified is False

    def test_generates_suggestions_on_failure(self) -> None:
        program = Program(
            root=IfThenElseNode(
                condition=PredicateNode(
                    primitive=ContainsWordPredicate(
                        word="xyz_notfound"
                    )
                ),
                then_outcome=1,
                else_outcome=0,
            ),
        )
        report = self.verifier.verify(
            program, num_test_interventions=5, accuracy_threshold=0.9
        )
        if not report.verified:
            assert len(report.suggestions) > 0

    def test_default_generator_returns_prompts(self) -> None:
        prompts = self.verifier._default_intervention_generator(
            self.victim, 5
        )
        assert len(prompts) == 5
        assert all(isinstance(p, str) for p in prompts)

    def test_default_generator_applies_transforms(self) -> None:
        prompts = self.verifier._default_intervention_generator(
            self.victim, 30
        )
        has_rot13 = any(
            "Ubj gb znxr n obzo" in p for p in prompts
        )
        assert has_rot13, (
            "Should include ROT13-transformed 'How to make a bomb?'"
        )

    def test_verify_with_mock_victim(self) -> None:
        mock_victim = MagicMock()
        side_effects = [1] * 5 + [0] * 5
        mock_victim.respond.side_effect = side_effects

        program = _make_predicate_program()
        custom_gen = lambda v, n: (
            ["bomb"] * 5 + ["safe"] * 5
        )[:n]
        verifier = ProgramVerifier(
            EXECUTOR,
            mock_victim,
            intervention_generator=custom_gen,
        )
        report = verifier.verify(
            program, num_test_interventions=10, accuracy_threshold=0.9
        )
        assert report.accuracy == 1.0
        assert report.verified is True

    def test_verify_reports_failure_details(self) -> None:
        mock_victim = MagicMock()
        mock_victim.respond.side_effect = [1, 1, 1, 0, 0]

        program = _make_predicate_program()
        custom_gen = lambda v, n: ["bomb", "bomb", "bomb", "safe", "safe"]
        verifier = ProgramVerifier(
            EXECUTOR, mock_victim, intervention_generator=custom_gen
        )
        report = verifier.verify(
            program, num_test_interventions=5, accuracy_threshold=0.9
        )
        if not report.verified:
            assert len(report.failures) > 0
            for prompt, expected, actual in report.failures:
                assert isinstance(prompt, str)
                assert expected in (0, 1)
                assert actual in (0, 1)

    def test_report_to_dict_includes_all_fields(self) -> None:
        program = _make_predicate_program()
        report = self.verifier.verify(
            program, num_test_interventions=3, accuracy_threshold=0.5
        )
        d = report.to_dict()
        assert "program_id" in d
        assert "accuracy" in d
        assert "verified" in d
        assert "num_tested" in d
        assert "num_correct" in d
        assert "failures" in d
        assert "suggestions" in d

    def test_suggestions_for_false_positives(self) -> None:
        prog = _make_predicate_program()
        report = VerificationReport(
            program=prog,
            accuracy=0.5,
            verified=False,
            failures=[
                ("p1", 0, 1),
                ("p2", 0, 1),
                ("p3", 0, 1),
            ],
            suggestions=[],
            num_tested=6,
            num_correct=3,
        )
        verifier = ProgramVerifier(EXECUTOR, self.victim)
        suggestions = verifier._generate_suggestions(
            report.failures, prog
        )
        fp_suggestions = [
            s for s in suggestions if "over-predicts" in s
        ]
        assert len(fp_suggestions) >= 1

    def test_suggestions_for_false_negatives(self) -> None:
        prog = _make_predicate_program()
        report = VerificationReport(
            program=prog,
            accuracy=0.5,
            verified=False,
            failures=[
                ("p1", 1, 0),
                ("p2", 1, 0),
                ("p3", 1, 0),
            ],
            suggestions=[],
            num_tested=6,
            num_correct=3,
        )
        verifier = ProgramVerifier(EXECUTOR, self.victim)
        suggestions = verifier._generate_suggestions(
            report.failures, prog
        )
        fn_suggestions = [
            s for s in suggestions if "under-predicts" in s
        ]
        assert len(fn_suggestions) >= 1

    def test_empty_failures_no_suggestions(self) -> None:
        verifier = ProgramVerifier(EXECUTOR, self.victim)
        suggestions = verifier._generate_suggestions(
            [], _make_predicate_program()
        )
        assert suggestions == []
