from core.primitive import ContainsWordPredicate
from core.program import IfThenElseNode, PredicateNode, Program

from adapters.toy_victims.rule_based import KeywordFilterVictim
from evaluation.primitive_discovery import PrimitiveDiscoveryEvaluator


class TestPrimitiveDiscoveryEvaluator:
    def test_perfect_discovery(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        discovered = victim.get_ground_truth_program()
        evaluator = PrimitiveDiscoveryEvaluator()
        result = evaluator.evaluate(victim, discovered)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_discovery(self):
        victim = KeywordFilterVictim(keywords=["bomb", "kill"])
        # Discover only the "bomb" filter
        partial = Program(
            root=IfThenElseNode(
                condition=PredicateNode(
                    primitive=ContainsWordPredicate(word="bomb")
                ),
                then_outcome=1,
                else_outcome=0,
            )
        )
        evaluator = PrimitiveDiscoveryEvaluator()
        result = evaluator.evaluate(victim, partial)
        assert result["recall"] < 1.0
        assert result["gt_primitive_count"] == 2

    def test_no_ground_truth(self):
        from adapters.toy_victims.neural import SKLearnVictim
        victim = SKLearnVictim(random_state=42, training_size=500)
        dummy = Program(
            root=IfThenElseNode(
                condition=PredicateNode(
                    primitive=ContainsWordPredicate(word="bomb")
                ),
                then_outcome=1,
                else_outcome=0,
            )
        )
        evaluator = PrimitiveDiscoveryEvaluator()
        result = evaluator.evaluate(victim, dummy)
        assert result["f1"] == 0.0

    def test_primitive_set_extraction(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        evaluator = PrimitiveDiscoveryEvaluator()
        primitives = evaluator.primitive_set(victim.get_ground_truth_program())
        assert len(primitives) >= 1
