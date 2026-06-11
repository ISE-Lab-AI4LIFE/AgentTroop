"""Tests for MultiDimensionalBoundaryEstimator."""

import numpy as np
import pytest

from sde.multi_dim_boundary import (
    MultiBoundaryEstimate,
    MultiDimensionalBoundaryEstimator,
)


class TestMultiBoundaryEstimate:
    def test_feature_importance(self):
        est = MultiBoundaryEstimate(
            coefficients={"a": 2.0, "b": 0.5, "c": 0.0},
            intercept=0.0,
            feature_names=["a", "b", "c"],
            num_observations=10,
            confidence_intervals={"a": (1.0, 3.0), "b": (0.0, 1.0), "c": (-0.5, 0.5)},
            accuracy=0.8,
        )
        imp = est.feature_importance
        assert abs(imp["a"] - 0.8) < 0.01
        assert abs(imp["b"] - 0.2) < 0.01
        assert imp["c"] == 0.0

    def test_to_dict(self):
        est = MultiBoundaryEstimate(
            coefficients={"a": 1.0}, intercept=-0.5,
            feature_names=["a"], num_observations=5,
            confidence_intervals={"a": (0.5, 1.5)},
            accuracy=0.9,
        )
        d = est.to_dict()
        assert "coefficients" in d
        assert "intercept" in d
        assert d["num_observations"] == 5

    def test_decision_boundary_format(self):
        est = MultiBoundaryEstimate(
            coefficients={"x": 2.0, "y": -1.0},
            intercept=0.5,
            feature_names=["x", "y"],
            num_observations=20,
            confidence_intervals={"x": (1.0, 3.0), "y": (-2.0, 0.0)},
            accuracy=0.85,
        )
        assert "P(refuse)" in est.decision_boundary
        assert "2.000" in est.decision_boundary
        assert "-1.000" in est.decision_boundary


class TestMultiDimensionalBoundaryEstimator:
    def test_initial_state(self):
        est = MultiDimensionalBoundaryEstimator()
        assert est.num_observations == 0
        assert est.feature_names == ["instruction_score", "harmfulness_score",
                                     "procedurality_score", "jailbreak_score"]

    def test_observe(self):
        est = MultiDimensionalBoundaryEstimator()
        est.observe({"instruction_score": 0.8, "harmfulness_score": 0.1}, 1)
        assert est.num_observations == 1

    def test_estimate_with_few_observations(self):
        est = MultiDimensionalBoundaryEstimator(min_observations=10)
        for _ in range(3):
            est.observe({"instruction_score": 0.8}, 1)
        result = est.estimate()
        assert result.num_observations == 3
        assert result.decision_boundary == "Insufficient data"

    def test_estimate_with_enough_observations(self):
        est = MultiDimensionalBoundaryEstimator(min_observations=3)
        for _ in range(10):
            est.observe({"instruction_score": 0.9, "harmfulness_score": 0.8}, 1)
            est.observe({"instruction_score": 0.1, "harmfulness_score": 0.1}, 0)
        result = est.estimate()
        assert result.num_observations == 20
        assert result.accuracy > 0.0
        assert "instruction_score" in result.coefficients
        assert "harmfulness_score" in result.coefficients
        assert isinstance(result.confidence_intervals, dict)

    def test_predict(self):
        est = MultiDimensionalBoundaryEstimator(min_observations=3)
        for _ in range(5):
            est.observe({"instruction_score": 0.9, "harmfulness_score": 0.8}, 1)
            est.observe({"instruction_score": 0.1, "harmfulness_score": 0.1}, 0)
        prob = est.predict({"instruction_score": 0.9, "harmfulness_score": 0.9})
        assert 0.0 <= prob <= 1.0
        prob_low = est.predict({"instruction_score": 0.1, "harmfulness_score": 0.1})
        assert prob_low < prob

    def test_observe_vector(self):
        est = MultiDimensionalBoundaryEstimator(min_observations=3)
        est.observe_vector([0.8, 0.2, 0.5, 0.1], 1)
        assert est.num_observations == 1

    def test_reset(self):
        est = MultiDimensionalBoundaryEstimator()
        est.observe({"instruction_score": 0.8}, 1)
        assert est.num_observations == 1
        est.reset()
        assert est.num_observations == 0

    def test_heuristic_fit_fallback(self):
        est = MultiDimensionalBoundaryEstimator(min_observations=3)
        # Force heuristic by making sklearn unavailable - we just verify it works
        # with few observations
        for _ in range(3):
            est.observe({"instruction_score": 0.7, "harmfulness_score": 0.6}, 1)
            est.observe({"instruction_score": 0.3, "harmfulness_score": 0.2}, 0)
        result = est.estimate()
        assert result.num_observations == 6
        assert result.accuracy > 0.0
