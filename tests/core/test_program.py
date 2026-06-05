from core.primitive import ContainsWordPredicate
from core.program import AndNode, IfThenElseNode, PredicateNode, Program


def test_program_serialization_and_equality():
    predicate = ContainsWordPredicate(word="bomb")
    node = PredicateNode(primitive=predicate)
    program = Program(root=IfThenElseNode(condition=node, then_outcome=1, else_outcome=0))

    serialized = program.to_dict()
    deserialized = Program.from_dict(serialized)

    assert isinstance(deserialized, Program)
    assert program == deserialized
    assert program.complexity() == deserialized.complexity()


def test_program_canonicalization_sorts_binary_nodes():
    predicate_a = PredicateNode(primitive=ContainsWordPredicate(word="a"))
    predicate_b = PredicateNode(primitive=ContainsWordPredicate(word="b"))
    root = IfThenElseNode(condition=AndNode(left=predicate_b, right=predicate_a), then_outcome=1, else_outcome=0)
    program = Program(root=root)

    canonical = program.canonicalize()
    assert str(canonical.root.condition.left) < str(canonical.root.condition.right)
