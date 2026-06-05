from dataclasses import dataclass
from typing import Optional

from core.primitive import ContainsWordPredicate
from core.program import IfThenElseNode, PredicateNode, Program

from adapters.toy_victims.rule_based import KeywordFilterVictim
from evaluation.hypothesis_quality import HypothesisQualityEvaluator


@dataclass
class MockHypothesis:
    program: Optional[Program] = None
    belief: float = 0.5


class TestHypothesisQualityEvaluator:
    def test_mrr_perfect_ranking(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        gt = victim.get_ground_truth_program()
        correct_hyp = MockHypothesis(program=gt, belief=0.9)
        incorrect_hyp = MockHypothesis(
            program=Program(
                root=IfThenElseNode(
                    condition=PredicateNode(
                        primitive=ContainsWordPredicate(word="kill")
                    ),
                    then_outcome=1,
                    else_outcome=0,
                )
            ),
            belief=0.1,
        )
        evaluator = HypothesisQualityEvaluator()
        result = evaluator.rank_quality(
            hypotheses=[correct_hyp, incorrect_hyp],
            victim=victim,
            test_inputs=["bomb", "hello"],
        )
        assert result["mrr"] == 1.0

    def test_mrr_not_found(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        incorrect_hyp = MockHypothesis(
            program=Program(
                root=IfThenElseNode(
                    condition=PredicateNode(
                        primitive=ContainsWordPredicate(word="kill")
                    ),
                    then_outcome=1,
                    else_outcome=0,
                )
            ),
            belief=0.5,
        )
        evaluator = HypothesisQualityEvaluator()
        result = evaluator.rank_quality(
            hypotheses=[incorrect_hyp],
            victim=victim,
            test_inputs=["bomb", "hello"],
        )
        assert result["mrr"] == 0.0

    def test_precision_at_k(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        gt = victim.get_ground_truth_program()
        h1 = MockHypothesis(program=gt, belief=0.9)
        h2 = MockHypothesis(
            program=Program(
                root=IfThenElseNode(
                    condition=PredicateNode(
                        primitive=ContainsWordPredicate(word="kill")
                    ),
                    then_outcome=1,
                    else_outcome=0,
                )
            ),
            belief=0.5,
        )
        evaluator = HypothesisQualityEvaluator()
        result = evaluator.rank_quality(
            hypotheses=[h1, h2],
            victim=victim,
            test_inputs=["bomb", "hello"],
            k=2,
        )
        assert result["precision@2"] == 0.5

    def test_empty_hypotheses(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        evaluator = HypothesisQualityEvaluator()
        result = evaluator.rank_quality(
            hypotheses=[], victim=victim, test_inputs=["bomb"]
        )
        assert result["mrr"] == 0.0
