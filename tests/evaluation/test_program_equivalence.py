from core.primitive import ContainsWordPredicate, default_registry
from core.program import IfThenElseNode, PredicateNode, Program

from evaluation.program_equivalence import ProgramEquivalenceChecker


class TestProgramEquivalenceChecker:
    def test_identical_programs_are_equivalent(self):
        predicate = ContainsWordPredicate(word="bomb")
        program = Program(
            root=IfThenElseNode(
                condition=PredicateNode(primitive=predicate),
                then_outcome=1,
                else_outcome=0,
            )
        )
        checker = ProgramEquivalenceChecker()
        assert checker.are_equivalent(program, program, ["test1", "test2"])

    def test_different_programs_are_not_equivalent(self):
        p1 = Program(
            root=IfThenElseNode(
                condition=PredicateNode(primitive=ContainsWordPredicate(word="bomb")),
                then_outcome=1,
                else_outcome=0,
            )
        )
        p2 = Program(
            root=IfThenElseNode(
                condition=PredicateNode(primitive=ContainsWordPredicate(word="kill")),
                then_outcome=1,
                else_outcome=0,
            )
        )
        checker = ProgramEquivalenceChecker()
        assert not checker.are_equivalent(p1, p2, ["bomb here", "kill here"])

    def test_equivalence_with_tolerance(self):
        p1 = Program(
            root=IfThenElseNode(
                condition=PredicateNode(primitive=ContainsWordPredicate(word="bomb")),
                then_outcome=1,
                else_outcome=0,
            )
        )
        p2 = Program(
            root=IfThenElseNode(
                condition=PredicateNode(primitive=ContainsWordPredicate(word="bomb")),
                then_outcome=1,
                else_outcome=0,
            )
        )
        checker = ProgramEquivalenceChecker()
        agreement = checker.equivalence_with_tolerance(p1, p2, ["bomb", "safe"])
        assert agreement == 1.0

    def test_empty_inputs(self):
        p1 = Program(
            root=IfThenElseNode(
                condition=PredicateNode(primitive=ContainsWordPredicate(word="bomb")),
                then_outcome=1,
                else_outcome=0,
            )
        )
        checker = ProgramEquivalenceChecker()
        assert checker.equivalence_with_tolerance(p1, p1, []) == 1.0
