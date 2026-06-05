from evaluation.information_gain import InformationGainEvaluator


class TestInformationGainEvaluator:
    def test_binary_entropy_extremes(self):
        assert InformationGainEvaluator.binary_entropy(0.0) == 0.0
        assert InformationGainEvaluator.binary_entropy(1.0) == 0.0

    def test_binary_entropy_maximum(self):
        h = InformationGainEvaluator.binary_entropy(0.5)
        assert h == 1.0

    def test_entropy_reduction_positive(self):
        evaluator = InformationGainEvaluator()
        reduction = evaluator.compute_entropy_reduction(0.5, 0.1)
        assert reduction > 0

    def test_entropy_reduction_no_change(self):
        evaluator = InformationGainEvaluator()
        reduction = evaluator.compute_entropy_reduction(0.5, 0.5)
        assert reduction == 0.0

    def test_entropy_reduction_negative(self):
        evaluator = InformationGainEvaluator()
        # If posterior is more uncertain, reduction is negative
        reduction = evaluator.compute_entropy_reduction(0.1, 0.5)
        assert reduction < 0

    def test_intervention_sequence(self):
        evaluator = InformationGainEvaluator()
        updates = [(0.5, 0.3), (0.3, 0.1), (0.1, 0.05)]
        gains = evaluator.evaluate_intervention_sequence(updates)
        assert len(gains) == 3
        assert all(g > 0 for g in gains)

    def test_cumulative_information_gain(self):
        evaluator = InformationGainEvaluator()
        updates = [(0.5, 0.4), (0.4, 0.2), (0.2, 0.05)]
        total = evaluator.cumulative_information_gain(updates)
        assert total > 0
        assert total <= 1.0  # max possible H(0.5) - H(0) = 1.0
