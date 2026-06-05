from core.primitive import ContainsWordPredicate
from core.program import IfThenElseNode, PredicateNode, Program
from core.utils import canonicalize_program, complexity, hash_program, program_equivalence


def test_utility_functions_operate_on_programs():
    node_a = PredicateNode(primitive=ContainsWordPredicate(word="bomb"))
    program_a = Program(root=IfThenElseNode(condition=node_a, then_outcome=1, else_outcome=0))
    program_b = Program(root=IfThenElseNode(condition=node_a, then_outcome=1, else_outcome=0))

    assert program_equivalence(program_a, program_b)
    assert complexity(program_a) == 2
    assert isinstance(hash_program(program_a), str)
    assert hash_program(program_a) == hash_program(program_b)

    canonical = canonicalize_program(program_a)
    assert canonical == program_a
