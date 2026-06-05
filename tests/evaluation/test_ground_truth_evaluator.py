from core.primitive import ContainsWordPredicate
from core.program import IfThenElseNode, PredicateNode, Program

from adapters.toy_victims.rule_based import KeywordFilterVictim
from evaluation.ground_truth_evaluator import GroundTruthEvaluator


class TestGroundTruthEvaluator:
    def test_compute_accuracy_perfect_match(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        checker = GroundTruthEvaluator(victim, victim.get_ground_truth_program())
        prompts = ["this is a bomb", "hello world", "bomb here", "safe"]
        acc = checker.compute_accuracy(prompts)
        assert acc == 1.0

    def test_compute_accuracy_partial_match(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        wrong_program = Program(
            root=IfThenElseNode(
                condition=PredicateNode(
                    primitive=ContainsWordPredicate(word="kill")
                ),
                then_outcome=1,
                else_outcome=0,
            )
        )
        checker = GroundTruthEvaluator(victim, wrong_program)
        prompts = ["bomb here", "kill here", "hello"]
        acc = checker.compute_accuracy(prompts)
        # "bomb here" → victim REFUSES but wrong program ACCEPTS → mismatch
        # "kill here" → victim ACCEPTS but wrong program REFUSES → mismatch
        # "hello" → both ACCEPT → match
        assert acc == 1.0 / 3.0

    def test_compute_program_similarity(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        exact = victim.get_ground_truth_program()
        checker = GroundTruthEvaluator(victim, exact)
        similarity = checker.compute_program_similarity()
        assert similarity > 0.9

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
        checker = GroundTruthEvaluator(victim, dummy)
        assert checker.compute_program_similarity() == 0.0

    def test_empty_prompts(self):
        victim = KeywordFilterVictim(keywords=["bomb"])
        checker = GroundTruthEvaluator(victim, victim.get_ground_truth_program())
        assert checker.compute_accuracy([]) == 0.0
