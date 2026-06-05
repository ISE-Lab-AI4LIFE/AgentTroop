from core.hypothesis import Hypothesis
from core.primitive import ContainsWordPredicate
from core.program import IfThenElseNode, PredicateNode, Program


def test_hypothesis_serialization_roundtrip():
    node = PredicateNode(primitive=ContainsWordPredicate(word="bomb"))
    program = Program(root=IfThenElseNode(condition=node, then_outcome=1, else_outcome=0))
    hypothesis = Hypothesis(
        id="hyp_1",
        statement="Prompt contains bomb implies refusal",
        program=program,
        confidence=0.75,
        supporting_observations=["obs_1", "obs_2"],
        provenance=[{"source": "intervention", "details": {"id": "int_1"}}],
        status="CONFIRMED",
    )

    data = hypothesis.to_dict()
    restored = Hypothesis.from_dict(data)

    assert restored.id == hypothesis.id
    assert restored.confidence == hypothesis.confidence
    assert restored.supporting_observations == hypothesis.supporting_observations
    assert restored.provenance == hypothesis.provenance
    assert restored.status == hypothesis.status
    assert restored.program == hypothesis.program
