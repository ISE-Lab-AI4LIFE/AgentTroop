from typing import List, Tuple

from evaluation.sample_efficiency import SampleEfficiencyEvaluator


def dummy_strategy(victim, budget: int) -> List[Tuple[int, float]]:
    """A dummy strategy that returns a simple learning curve."""
    return [(i, min(1.0, i / budget)) for i in range(0, budget + 1, budget // 5)]


class TestSampleEfficiencyEvaluator:
    def test_learning_curve_structure(self):
        evaluator = SampleEfficiencyEvaluator()
        result = evaluator.compute_learning_curve(
            victim=None, strategy=dummy_strategy, max_interventions=100, n_runs=1
        )
        assert "x" in result
        assert "mean" in result
        assert "std" in result
        assert len(result["x"]) > 0
        assert len(result["mean"]) == len(result["x"])

    def test_auc_perfect_curve(self):
        evaluator = SampleEfficiencyEvaluator()
        perfect_curve = {
            "x": [0, 10, 20, 30],
            "mean": [0.0, 0.5, 0.8, 1.0],
        }
        auc = evaluator.area_under_curve(perfect_curve)
        assert 0.0 < auc <= 1.0

    def test_auc_zero_curve(self):
        evaluator = SampleEfficiencyEvaluator()
        zero_curve = {
            "x": [0, 10, 20, 30],
            "mean": [0.0, 0.0, 0.0, 0.0],
        }
        auc = evaluator.area_under_curve(zero_curve)
        assert auc == 0.0

    def test_auc_empty_curve(self):
        evaluator = SampleEfficiencyEvaluator()
        assert evaluator.area_under_curve({"x": [], "mean": []}) == 0.0
