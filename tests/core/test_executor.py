from core.executor import ProgramExecutor
from core.primitive import ContainsWordPredicate, ToxicityScoreClassifier
from core.program import ApplyTransformNode, IfThenElseNode, PredicateNode, Program, ThresholdNode
from core.primitive import default_registry


def test_executor_returns_expected_outcome():
    predicate = ContainsWordPredicate(word="bomb")
    root = IfThenElseNode(condition=PredicateNode(primitive=predicate), then_outcome=1, else_outcome=0)
    program = Program(root=root)
    executor = ProgramExecutor(default_registry)

    assert executor.execute(program, "there is a bomb") == 1
    assert executor.execute(program, "harmless text") == 0


def test_executor_trace_contains_nodes_and_values():
    classifier = ToxicityScoreClassifier()
    threshold_node = ThresholdNode(classifier=classifier, threshold=0.5)
    root = IfThenElseNode(condition=threshold_node, then_outcome=1, else_outcome=0)
    program = Program(root=root)
    executor = ProgramExecutor(default_registry)

    outcome, trace = executor.execute_with_trace(program, "prompt")
    assert outcome in (0, 1)
    assert any("ThresholdNode" in step.get("node", "") for step in trace)
