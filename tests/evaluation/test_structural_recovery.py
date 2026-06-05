import networkx as nx

from core.primitive import ContainsWordPredicate
from core.program import (
    AndNode,
    IfThenElseNode,
    OrNode,
    PredicateNode,
    Program,
)

from adapters.toy_victims.hybrid_logic import AndVictim, OrVictim
from adapters.toy_victims.rule_based import KeywordFilterVictim
from evaluation.structural_recovery import StructuralRecoveryEvaluator


class TestStructuralRecoveryEvaluator:
    def test_dependency_graph_is_dag(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        evaluator = StructuralRecoveryEvaluator()
        graph = evaluator.compute_dependency_graph(
            victim.get_ground_truth_program()
        )
        assert isinstance(graph, nx.DiGraph)
        assert nx.is_directed_acyclic_graph(graph)

    def test_identical_programs_have_zero_ged(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        gt = victim.get_ground_truth_program()
        evaluator = StructuralRecoveryEvaluator()
        distance = evaluator.graph_edit_distance(gt, gt)
        assert distance == 0.0

    def test_different_programs_have_positive_ged(self):
        v1 = KeywordFilterVictim(keywords=["bomb"])
        v2 = KeywordFilterVictim(keywords=["kill"])
        evaluator = StructuralRecoveryEvaluator()
        distance = evaluator.graph_edit_distance(
            v1.get_ground_truth_program(),
            v2.get_ground_truth_program(),
        )
        assert distance > 0

    def test_edge_precision_recall_perfect(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        gt = victim.get_ground_truth_program()
        evaluator = StructuralRecoveryEvaluator()
        metrics = evaluator.edge_precision_recall(gt, gt)
        assert metrics["edge_precision"] == 1.0
        assert metrics["edge_recall"] == 1.0

    def test_recovery_score_with_ground_truth(self):
        v1 = KeywordFilterVictim(keywords=["bomb"])
        v2 = v1  # same victim
        evaluator = StructuralRecoveryEvaluator()
        score = evaluator.recovery_score(v1, v2.get_ground_truth_program())
        assert score["recovery_score"] > 0.9
