#!/usr/bin/env python3
"""Comprehensive audit of intervention coverage mechanisms.

Tests:
1. _check_convergence — accuracy, improvement, entropy criteria
2. _maybe_force_exploration — deep-dive prompt injection
3. _uncertainty_sampling_intervention — active learning fallback
4. _dynamic_intervention_budget — budget scaling
5. Integration: easy victim (quick convergence)
6. Integration: hard victim (force exploration triggers)
7. Integration: stalled pipeline (uncertainty sampling fallback)
"""

import json
import logging
import math
import random
import sys
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("test_intervention_coverage")

# ===========================================================================
# Stubs
# ===========================================================================

class _StubIntervention:
    def __init__(self, base_prompt: str = "",
                 transforms: Optional[List[Dict[str, str]]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.base_prompt = base_prompt
        self.transforms = transforms or []
        self.metadata = metadata or {}


class _StubBest:
    def __init__(self, holdout_accuracy: float = 0.0,
                 accuracy: float = 0.0):
        self.holdout_accuracy = holdout_accuracy
        self.accuracy = accuracy


class _StubVersionSpace:
    def __init__(self, current_entropy: float = 0.5):
        self._entropy = current_entropy
        self._best = _StubBest()
        self.num_candidates = 10

    def entropy(self) -> float:
        return self._entropy

    def most_likely(self) -> Optional[_StubBest]:
        return self._best

    def get_highest_entropy_prompt(
        self, prompts: List[str], executor: Any,
    ) -> Tuple[Optional[str], float]:
        return ("high_entropy_prompt", 0.85)


class _StubExecutor:
    pass


class _StubStrategist:
    def __init__(self):
        self.executor = _StubExecutor()

    def _default_base_prompts(self) -> List[str]:
        return [f"prompt_{i}" for i in range(100)]


# ===========================================================================
# Minimal Orchestrator stub
# ===========================================================================

class _MinimalOrchestrator:
    CONVERGENCE_DEFAULTS: Dict[str, Any] = {
        "validation_accuracy_threshold": 0.85,
        "min_improvement": 0.02,
        "min_iterations_for_convergence": 5,
        "entropy_threshold": 0.1,
    }
    FORCE_EXPLORATION_DEFAULTS: Dict[str, Any] = {
        "enabled": True,
        "consecutive_iterations_without_intervention": 3,
        "num_forced_prompts": 3,
    }

    def __init__(
        self,
        convergence_config: Optional[Dict[str, Any]] = None,
        force_exploration_config: Optional[Dict[str, Any]] = None,
        uncertainty_sampling_fallback: bool = True,
        min_interventions_per_iteration: int = 3,
        absolute_max_iterations: int = 200,
        max_interventions: int = 500,
    ):
        raw_cc = convergence_config or {}
        self.convergence: Dict[str, Any] = {
            **self.CONVERGENCE_DEFAULTS,
            **{k: v for k, v in raw_cc.items() if v is not None},
        }
        raw_fe = force_exploration_config or {}
        self.force_exploration: Dict[str, Any] = {
            **self.FORCE_EXPLORATION_DEFAULTS,
            **{k: v for k, v in raw_fe.items() if v is not None},
        }
        self.uncertainty_sampling_fallback = bool(uncertainty_sampling_fallback)
        self.min_interventions_per_iteration = max(1, int(min_interventions_per_iteration))
        self.absolute_max_iterations = max(1, int(absolute_max_iterations))
        self.max_interventions = max_interventions

        self._accuracy_history: List[float] = []
        self._deep_dive_prompts: List[str] = []
        self._deep_dive_used: List[str] = []

        self.strategist = _StubStrategist()
        self.version_space = _StubVersionSpace()
        self._entropy_history: List[float] = []

    def _get_current_entropy(self) -> float:
        return self.version_space.entropy()

    # ==== _check_convergence (mirrors orchestrator.py) ====

    def _check_convergence(self) -> Optional[str]:
        cv = self.convergence
        val_thresh = float(cv.get("validation_accuracy_threshold", 0.85))
        min_impr = float(cv.get("min_improvement", 0.02))
        min_iters = int(cv.get("min_iterations_for_convergence", 5))
        ent_thresh = float(cv.get("entropy_threshold", 0.1))

        if len(self._accuracy_history) < min_iters:
            return None

        recent_acc = self._accuracy_history[-min_iters:]
        if all(a < val_thresh for a in recent_acc):
            return None

        window = min(10, len(self._accuracy_history))
        if window >= 5:
            acc_window = self._accuracy_history[-window:]
            improvement = max(acc_window) - acc_window[0]
            if improvement > min_impr:
                return None
        else:
            improvement = 0.0

        current_entropy = self._get_current_entropy()
        if current_entropy > ent_thresh:
            return None

        return (
            f"converged: acc={recent_acc[-1]:.3f} >= {val_thresh}, "
            f"entropy={current_entropy:.3f} < {ent_thresh}, "
            f"improvement={improvement:.3f} < {min_impr}"
        )

    def _update_accuracy_history(self, acc: float) -> None:
        self._accuracy_history.append(acc)

    # ==== _maybe_force_exploration (mirrors orchestrator.py) ====

    def _maybe_force_exploration(
        self, stalled_iterations: int,
    ) -> Optional[_StubIntervention]:
        fe = self.force_exploration
        if not fe.get("enabled", True):
            return None

        threshold = int(fe.get("consecutive_iterations_without_intervention", 3))
        if stalled_iterations < threshold:
            return None

        n_forced = int(fe.get("num_forced_prompts", 3))
        if not self._deep_dive_prompts:
            return None

        available = [p for p in self._deep_dive_prompts if p not in self._deep_dive_used]
        if not available:
            return None

        chosen = available[:n_forced]
        self._deep_dive_used.extend(chosen)

        return _StubIntervention(
            base_prompt=chosen[0],
            metadata={
                "forced_exploration": True,
                "deep_dive": True,
                "deep_dive_pool": chosen,
            },
        )

    # ==== _uncertainty_sampling_intervention (mirrors orchestrator.py) ====

    def _uncertainty_sampling_intervention(
        self, hypotheses: List[Any],
    ) -> Optional[_StubIntervention]:
        if not self.uncertainty_sampling_fallback:
            return None
        try:
            prompts = self.strategist._default_base_prompts()
            prompt, entropy = self.version_space.get_highest_entropy_prompt(
                prompts, self.strategist.executor,
            )
            if prompt is None or entropy <= 0.0:
                return None

            return _StubIntervention(
                base_prompt=prompt,
                metadata={
                    "uncertainty_sampling": True,
                    "entropy": round(entropy, 4),
                },
            )
        except Exception:
            return None

    # ==== _dynamic_intervention_budget (mirrors orchestrator.py) ====

    def _dynamic_intervention_budget(self) -> int:
        n_available = len(self._deep_dive_prompts) if self._deep_dive_prompts else 500
        return min(self.max_interventions, max(50, n_available))

    def _load_deep_dive_prompts(self) -> None:
        try:
            prompts = self.strategist._default_base_prompts()
            self._deep_dive_prompts = list(prompts)
        except Exception:
            self._deep_dive_prompts = []


# ===========================================================================
# Test: _check_convergence
# ===========================================================================

class TestCheckConvergence(unittest.TestCase):
    """Convergence detection criteria."""

    def setUp(self):
        self.orchestrator = _MinimalOrchestrator(convergence_config={
            "validation_accuracy_threshold": 0.85,
            "min_improvement": 0.02,
            "min_iterations_for_convergence": 5,
            "entropy_threshold": 0.1,
        })
        self.orchestrator.version_space = _StubVersionSpace(current_entropy=0.05)

    def test_converges_when_accuracy_met_and_entropy_low(self):
        """All three criteria met -> convergence string returned."""
        for _ in range(5):
            self.orchestrator._update_accuracy_history(0.90)
        result = self.orchestrator._check_convergence()
        self.assertIsNotNone(result)
        self.assertIn("converged", result or "")

    def test_not_converged_accuracy_too_low(self):
        """Accuracy below threshold -> no convergence."""
        for _ in range(5):
            self.orchestrator._update_accuracy_history(0.50)
        result = self.orchestrator._check_convergence()
        self.assertIsNone(result)

    def test_not_converged_entropy_too_high(self):
        """Entropy above threshold -> no convergence."""
        self.orchestrator.version_space = _StubVersionSpace(current_entropy=0.5)
        for _ in range(5):
            self.orchestrator._update_accuracy_history(0.90)
        result = self.orchestrator._check_convergence()
        self.assertIsNone(result)

    def test_not_converged_improving_too_fast(self):
        """Improvement > min_improvement -> no convergence."""
        self.orchestrator.version_space = _StubVersionSpace(current_entropy=0.05)
        # Rapidly increasing accuracy over 10 iterations
        for i in range(10):
            self.orchestrator._update_accuracy_history(0.50 + i * 0.05)
        result = self.orchestrator._check_convergence()
        self.assertIsNone(result)

    def test_not_converged_insufficient_history(self):
        """Fewer than min_iterations_for_convergence -> no convergence."""
        self.orchestrator._update_accuracy_history(0.90)
        result = self.orchestrator._check_convergence()
        self.assertIsNone(result)

    def test_convergence_returns_reason_string(self):
        """Convergence string contains accuracy, entropy, improvement."""
        self.orchestrator.version_space = _StubVersionSpace(current_entropy=0.05)
        for _ in range(5):
            self.orchestrator._update_accuracy_history(0.90)
        result = self.orchestrator._check_convergence()
        self.assertIsNotNone(result)
        self.assertIn("acc=", result or "")
        self.assertIn("entropy=", result or "")
        self.assertIn("improvement=", result or "")

    def test_not_converged_when_accuracy_fluctuates(self):
        """Fluctuating accuracy (high then low) -> no convergence."""
        self.orchestrator.version_space = _StubVersionSpace(current_entropy=0.05)
        vals = [0.90, 0.95, 0.80, 0.92, 0.88]
        for v in vals:
            self.orchestrator._update_accuracy_history(v)
        result = self.orchestrator._check_convergence()
        self.assertIsNone(result)

    def test_convergence_thresholds_configurable(self):
        """Config thresholds are respected."""
        strict = _MinimalOrchestrator(convergence_config={
            "validation_accuracy_threshold": 0.95,
            "min_improvement": 0.01,
            "min_iterations_for_convergence": 3,
            "entropy_threshold": 0.05,
        })
        strict.version_space = _StubVersionSpace(current_entropy=0.04)
        # Accuracy 0.90 < 0.95 -> no converge
        for _ in range(3):
            strict._update_accuracy_history(0.90)
        self.assertIsNone(strict._check_convergence())

        # Accuracy 0.96 >= 0.95 -> converge
        strict._accuracy_history.clear()
        for _ in range(3):
            strict._update_accuracy_history(0.96)
        self.assertIsNotNone(strict._check_convergence())

    def test_entropy_threshold_zero_disables_entropy_check(self):
        """entropy_threshold=1.0 means entropy can never block convergence."""
        relaxed = _MinimalOrchestrator(convergence_config={
            "entropy_threshold": 1.0,
            "min_iterations_for_convergence": 3,
        })
        relaxed.version_space = _StubVersionSpace(current_entropy=0.99)
        for _ in range(3):
            relaxed._update_accuracy_history(0.90)
        result = relaxed._check_convergence()
        self.assertIsNotNone(result)

    def test_default_config_is_sensible(self):
        """Default convergence config should be usable."""
        default = _MinimalOrchestrator()
        default.version_space = _StubVersionSpace(current_entropy=0.05)
        for _ in range(5):
            default._update_accuracy_history(0.90)
        self.assertIsNotNone(default._check_convergence())


# ===========================================================================
# Test: _maybe_force_exploration
# ===========================================================================

class TestMaybeForceExploration(unittest.TestCase):
    """Deep-dive prompt injection when stalled."""

    def setUp(self):
        self.orchestrator = _MinimalOrchestrator()
        self.orchestrator._load_deep_dive_prompts()
        self.initial_count = len(self.orchestrator._deep_dive_prompts)

    def test_triggers_after_stalled_threshold(self):
        """Force exploration returns intervention when stalled >= threshold."""
        result = self.orchestrator._maybe_force_exploration(stalled_iterations=3)
        self.assertIsNotNone(result)
        self.assertTrue(result.metadata.get("forced_exploration"))

    def test_not_triggers_below_threshold(self):
        """No intervention when stalled < threshold."""
        result = self.orchestrator._maybe_force_exploration(stalled_iterations=1)
        self.assertIsNone(result)
        result = self.orchestrator._maybe_force_exploration(stalled_iterations=2)
        self.assertIsNone(result)

    def test_consumes_deep_dive_prompts(self):
        """Each call consumes deep-dive prompts."""
        before = len(self.orchestrator._deep_dive_used)
        self.orchestrator._maybe_force_exploration(stalled_iterations=3)
        after = len(self.orchestrator._deep_dive_used)
        self.assertEqual(after - before, 3)

    def test_exhausted_prompts_return_none(self):
        """After all deep-dive prompts used, returns None."""
        n = len(self.orchestrator._deep_dive_prompts)
        # Exhaust all prompts, then one more call to confirm exhaustion
        max_triggers = n // 3 + 1  # some calls consume < 3 on the boundary
        for i in range(max_triggers + 1):
            result = self.orchestrator._maybe_force_exploration(stalled_iterations=3)
        self.assertIsNone(result)

    def test_disabled_config_returns_none(self):
        """When force_exploration.enabled=False, always returns None."""
        disabled = _MinimalOrchestrator(force_exploration_config={"enabled": False})
        disabled._load_deep_dive_prompts()
        result = disabled._maybe_force_exploration(stalled_iterations=100)
        self.assertIsNone(result)

    def test_no_prompts_loaded_returns_none(self):
        """Without loaded prompts, always returns None."""
        empty = _MinimalOrchestrator()
        result = empty._maybe_force_exploration(stalled_iterations=3)
        self.assertIsNone(result)

    def test_configurable_threshold(self):
        """Threshold can be customized via config."""
        fast = _MinimalOrchestrator(force_exploration_config={
            "consecutive_iterations_without_intervention": 1,
        })
        fast._load_deep_dive_prompts()

        result = fast._maybe_force_exploration(stalled_iterations=0)
        self.assertIsNone(result)

        result = fast._maybe_force_exploration(stalled_iterations=1)
        self.assertIsNotNone(result)

    def test_configurable_num_forced(self):
        """Number of prompts consumed per trigger is configurable."""
        configurable = _MinimalOrchestrator(force_exploration_config={
            "num_forced_prompts": 5,
        })
        configurable._load_deep_dive_prompts()
        before = len(configurable._deep_dive_used)
        configurable._maybe_force_exploration(stalled_iterations=3)
        after = len(configurable._deep_dive_used)
        self.assertEqual(after - before, 5)

    def test_intervention_contains_deep_dive_pool(self):
        """Intervention metadata includes the full chosen pool."""
        result = self.orchestrator._maybe_force_exploration(stalled_iterations=3)
        pool = result.metadata.get("deep_dive_pool", [])
        self.assertEqual(len(pool), 3)
        for p in pool:
            self.assertIn(p, self.orchestrator._deep_dive_used)


# ===========================================================================
# Test: _uncertainty_sampling_intervention
# ===========================================================================

class TestUncertaintySampling(unittest.TestCase):
    """Active learning fallback when no intervention designed."""

    def setUp(self):
        self.orchestrator = _MinimalOrchestrator()
        self.hypotheses: List[Any] = ["hyp1", "hyp2"]

    def test_returns_intervention_when_enabled(self):
        """Returns an intervention with uncertainty_sampling metadata."""
        result = self.orchestrator._uncertainty_sampling_intervention(self.hypotheses)
        self.assertIsNotNone(result)
        self.assertTrue(result.metadata.get("uncertainty_sampling"))
        self.assertIn("entropy", result.metadata)

    def test_returns_none_when_disabled(self):
        """When uncertainty_sampling_fallback=False, returns None."""
        disabled = _MinimalOrchestrator(uncertainty_sampling_fallback=False)
        result = disabled._uncertainty_sampling_intervention(self.hypotheses)
        self.assertIsNone(result)

    def test_entropy_value_in_metadata(self):
        """Entropy value is recorded in metadata."""
        result = self.orchestrator._uncertainty_sampling_intervention(self.hypotheses)
        self.assertGreater(result.metadata.get("entropy", 0), 0.0)

    def test_uses_high_entropy_prompt(self):
        """Uses the prompt returned by get_highest_entropy_prompt."""
        result = self.orchestrator._uncertainty_sampling_intervention(self.hypotheses)
        self.assertEqual(result.base_prompt, "high_entropy_prompt")

    def test_handles_exception_gracefully(self):
        """Exception in uncertainty sampling returns None, doesn't crash."""
        broken = _MinimalOrchestrator()
        broken.version_space = None  # will cause AttributeError
        result = broken._uncertainty_sampling_intervention(self.hypotheses)
        self.assertIsNone(result)

    def test_empty_hypotheses_still_works(self):
        """Works even with empty hypotheses list."""
        result = self.orchestrator._uncertainty_sampling_intervention([])
        self.assertIsNotNone(result)


# ===========================================================================
# Test: _dynamic_intervention_budget
# ===========================================================================

class TestDynamicInterventionBudget(unittest.TestCase):
    """Budget scaling with available prompts."""

    def test_scales_with_prompt_count(self):
        """Budget = min(max_interventions, max(50, n_prompts))."""
        orch = _MinimalOrchestrator(max_interventions=500)
        orch._deep_dive_prompts = [f"p{i}" for i in range(100)]
        self.assertEqual(orch._dynamic_intervention_budget(), 100)

    def test_never_below_minimum(self):
        """Budget never below 50 even with few prompts."""
        orch = _MinimalOrchestrator(max_interventions=500)
        orch._deep_dive_prompts = [f"p{i}" for i in range(10)]
        self.assertEqual(orch._dynamic_intervention_budget(), 50)

    def test_capped_by_max_interventions(self):
        """Budget never exceeds max_interventions."""
        orch = _MinimalOrchestrator(max_interventions=100)
        orch._deep_dive_prompts = [f"p{i}" for i in range(1000)]
        self.assertEqual(orch._dynamic_intervention_budget(), 100)

    def test_fallback_without_deep_dive_prompts(self):
        """Without deep dive prompts, fallback = 500."""
        orch = _MinimalOrchestrator(max_interventions=1000)
        self.assertEqual(orch._dynamic_intervention_budget(), 500)

    def test_fallback_honors_max_interventions(self):
        """Fallback 500 is capped by max_interventions if lower."""
        orch = _MinimalOrchestrator(max_interventions=100)
        self.assertEqual(orch._dynamic_intervention_budget(), 100)

    def test_empty_prompts_list_fallback(self):
        """Empty prompts list -> fallback 500."""
        orch = _MinimalOrchestrator(max_interventions=500)
        orch._deep_dive_prompts = []
        self.assertEqual(orch._dynamic_intervention_budget(), 500)

    def test_small_max_interventions(self):
        """Small max_interventions caps budget."""
        orch = _MinimalOrchestrator(max_interventions=30)
        orch._deep_dive_prompts = [f"p{i}" for i in range(100)]
        # max(50, 100) = 100, min(100, 30) = 30
        self.assertEqual(orch._dynamic_intervention_budget(), 30)

    def test_minimum_max_interventions_above_50(self):
        """When max_interventions >= 50 and prompts >= 50, budget = prompts."""
        orch = _MinimalOrchestrator(max_interventions=500)
        orch._deep_dive_prompts = [f"p{i}" for i in range(75)]
        self.assertEqual(orch._dynamic_intervention_budget(), 75)


# ===========================================================================
# Integration scenarios
# ===========================================================================

class TestIntegrationEasyVictim(unittest.TestCase):
    """Easy victim: high accuracy early, entropy drops -> convergence."""

    def test_converges_on_easy_victim(self):
        """Easy victim produces convergence within 15 iterations."""
        orch = _MinimalOrchestrator(convergence_config={
            "validation_accuracy_threshold": 0.85,
            "min_improvement": 0.02,
            "min_iterations_for_convergence": 5,
            "entropy_threshold": 0.1,
        })
        orch.version_space = _StubVersionSpace(current_entropy=0.05)
        orch._load_deep_dive_prompts()

        stalled = 0
        converged_reason = None
        for i in range(15):
            orch._update_accuracy_history(0.90)
            orch._entropy_history.append(0.05)

            reason = orch._check_convergence()
            if reason is not None:
                converged_reason = reason
                break

            # Not stalled -> force exploration not needed
            result = orch._maybe_force_exploration(stalled)
            if result is not None:
                stalled = 0
            else:
                stalled += 1

        self.assertIsNotNone(converged_reason)
        self.assertIn("converged", converged_reason)
        self.assertLessEqual(i + 1, 10,
                             "Easy victim should converge quickly")


class TestIntegrationHardVictim(unittest.TestCase):
    """Hard victim: low accuracy initially -> force exploration triggers."""

    def test_force_exploration_triggers_on_hard_victim(self):
        """Force exploration triggers after repeated stalls on hard victim."""
        orch = _MinimalOrchestrator(
            convergence_config={
                "validation_accuracy_threshold": 0.85,
                "min_improvement": 0.02,
                "min_iterations_for_convergence": 5,
                "entropy_threshold": 0.1,
            },
            force_exploration_config={
                "enabled": True,
                "consecutive_iterations_without_intervention": 3,
                "num_forced_prompts": 3,
            },
        )
        orch.version_space = _StubVersionSpace(current_entropy=0.5)
        orch._load_deep_dive_prompts()

        force_count = 0
        stalled = 0
        forced_used = 0
        for i in range(20):
            # Low accuracy (stuck at 0.60)
            orch._update_accuracy_history(0.60)

            reason = orch._check_convergence()
            if reason is not None:
                break

            # Simulate strategist producing no intervention
            stalled += 1
            result = orch._maybe_force_exploration(stalled)
            if result is not None:
                force_count += 1
                stalled = 0
                forced_used += len(result.metadata.get("deep_dive_pool", []))

        self.assertGreaterEqual(
            force_count, 1,
            "Force exploration should trigger at least once on hard victim",
        )
        self.assertGreaterEqual(
            forced_used, 3,
            "At least 3 deep-dive prompts should be consumed",
        )


class TestIntegrationStalledPipeline(unittest.TestCase):
    """Fully stalled pipeline -> uncertainty sampling creates interventions."""

    def test_uncertainty_sampling_creates_interventions_when_stalled(self):
        """Uncertainty sampling provides fallback interventions."""
        orch = _MinimalOrchestrator(
            uncertainty_sampling_fallback=True,
            convergence_config={
                "validation_accuracy_threshold": 0.85,
                "min_improvement": 0.02,
                "min_iterations_for_convergence": 10,
                "entropy_threshold": 0.05,
            },
        )
        orch.version_space = _StubVersionSpace(current_entropy=0.4)
        orch._load_deep_dive_prompts()

        stalled = 0
        interventions_created = 0
        for i in range(25):
            orch._update_accuracy_history(0.50)

            reason = orch._check_convergence()
            if reason is not None:
                break

            # Strategist fails every time
            stalled += 1

            # Try force exploration first
            intervention = orch._maybe_force_exploration(stalled)
            if intervention is not None:
                stalled = 0
                interventions_created += 1
                continue

            # Fall back to uncertainty sampling
            intervention = orch._uncertainty_sampling_intervention(["h1"])
            if intervention is not None:
                stalled = 0
                interventions_created += 1

        self.assertGreaterEqual(
            interventions_created, 1,
            "At least one intervention should be created via fallback chain",
        )


# ===========================================================================
# Benchmark scenarios
# ===========================================================================

def benchmark_convergence_scenario(
    name: str,
    accuracy_pattern: List[float],
    entropy: float,
    convergence_config: Dict[str, Any],
    max_iterations: int = 30,
) -> Dict[str, Any]:
    """Run a convergence benchmark and return detailed metrics."""
    orch = _MinimalOrchestrator(convergence_config=convergence_config)
    orch.version_space = _StubVersionSpace(current_entropy=entropy)

    start = time.time()
    iters: List[Dict[str, Any]] = []
    converged_iteration: Optional[int] = None
    converged_reason: Optional[str] = None

    for i in range(max_iterations):
        t0 = time.time()
        acc = accuracy_pattern[i] if i < len(accuracy_pattern) else accuracy_pattern[-1]
        orch._update_accuracy_history(acc)

        reason = orch._check_convergence()
        elapsed = (time.time() - t0) * 1000

        iters.append({
            "iteration": i + 1,
            "accuracy": acc,
            "history_size": len(orch._accuracy_history),
            "checked_ms": round(elapsed, 2),
        })

        if reason is not None:
            converged_iteration = i + 1
            converged_reason = reason
            break

    duration = time.time() - start

    return {
        "scenario": name,
        "config": dict(convergence_config),
        "data": {
            "accuracy_pattern_len": len(accuracy_pattern),
            "entropy": entropy,
            "max_iterations": max_iterations,
        },
        "result": {
            "converged": converged_iteration is not None,
            "converged_at_iteration": converged_iteration,
            "converged_reason": converged_reason,
        },
        "iterations": iters,
        "timing_ms": round(duration * 1000, 2),
    }


def benchmark_force_exploration_scenario(
    name: str,
    n_prompts: int,
    num_forced: int,
    n_calls: int = 30,
) -> Dict[str, Any]:
    """Run a force exploration benchmark."""
    orch = _MinimalOrchestrator(force_exploration_config={
        "enabled": True,
        "consecutive_iterations_without_intervention": 3,
        "num_forced_prompts": num_forced,
    })
    orch._deep_dive_prompts = [f"p{i}" for i in range(n_prompts)]

    start = time.time()
    triggers = 0
    prompts_consumed = 0
    failures = 0
    for i in range(n_calls):
        stalled = 3 if i % 5 == 0 else 0
        t0 = time.time()
        result = orch._maybe_force_exploration(stalled)
        elapsed = (time.time() - t0) * 1000
        if result is not None:
            triggers += 1
            pool = result.metadata.get("deep_dive_pool", [])
            prompts_consumed += len(pool)
        else:
            failures += 1
    duration = time.time() - start

    return {
        "scenario": name,
        "config": {"n_prompts": n_prompts, "num_forced": num_forced, "n_calls": n_calls},
        "result": {
            "triggers": triggers,
            "failures": failures,
            "prompts_consumed": prompts_consumed,
            "prompts_remaining": len(orch._deep_dive_prompts) - len(orch._deep_dive_used),
        },
        "timing_ms": round(duration * 1000, 2),
    }


def run_benchmarks() -> Dict[str, Any]:
    """Run all benchmark scenarios."""
    results: Dict[str, Any] = {}

    # Convergence scenarios
    results["convergence_easy"] = benchmark_convergence_scenario(
        "Easy: high accuracy, low entropy",
        [0.90] * 30, 0.05,
        {"validation_accuracy_threshold": 0.85, "entropy_threshold": 0.1,
         "min_iterations_for_convergence": 5, "min_improvement": 0.02},
    )

    results["convergence_slow_improvement"] = benchmark_convergence_scenario(
        "Slow: gradual accuracy climb",
        [0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88,
         0.89, 0.89, 0.89, 0.90, 0.90], 0.08,
        {"validation_accuracy_threshold": 0.85, "entropy_threshold": 0.1,
         "min_iterations_for_convergence": 5, "min_improvement": 0.02},
    )

    results["convergence_never"] = benchmark_convergence_scenario(
        "Never: accuracy stuck at 0.50",
        [0.50] * 30, 0.5,
        {"validation_accuracy_threshold": 0.85, "entropy_threshold": 0.1,
         "min_iterations_for_convergence": 5, "min_improvement": 0.02},
    )

    results["convergence_fluctuating"] = benchmark_convergence_scenario(
        "Fluctuating: oscillating accuracy",
        [0.85, 0.90, 0.75, 0.88, 0.82, 0.91, 0.78, 0.86, 0.80, 0.89], 0.08,
        {"validation_accuracy_threshold": 0.85, "entropy_threshold": 0.1,
         "min_iterations_for_convergence": 5, "min_improvement": 0.02},
    )

    results["convergence_fast_entropy_drop"] = benchmark_convergence_scenario(
        "Fast convergence: high accuracy + low entropy",
        [0.92, 0.93, 0.91, 0.94, 0.92], 0.03,
        {"validation_accuracy_threshold": 0.85, "entropy_threshold": 0.1,
         "min_iterations_for_convergence": 3, "min_improvement": 0.02},
    )

    results["convergence_strict_thresholds"] = benchmark_convergence_scenario(
        "Strict: threshold 0.95 never reached",
        [0.90, 0.91, 0.92, 0.91, 0.93] * 6, 0.05,
        {"validation_accuracy_threshold": 0.95, "entropy_threshold": 0.1,
         "min_iterations_for_convergence": 5, "min_improvement": 0.01},
    )

    # Force exploration scenarios
    results["force_exploration_large_pool"] = benchmark_force_exploration_scenario(
        "Large pool (1000 prompts, 3 per trigger)",
        n_prompts=1000, num_forced=3, n_calls=30,
    )

    results["force_exploration_small_pool_exhausted"] = benchmark_force_exploration_scenario(
        "Small pool exhausted (10 prompts, 3 per trigger)",
        n_prompts=10, num_forced=3, n_calls=30,
    )

    results["force_exploration_aggressive"] = benchmark_force_exploration_scenario(
        "Aggressive (5 prompts per trigger, 50 pool)",
        n_prompts=50, num_forced=5, n_calls=20,
    )

    return results


# ===========================================================================
# Main
# ===========================================================================

def run_tests() -> Dict[str, Any]:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.wasSuccessful(),
    }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("INTERVENTION COVERAGE -- COMPREHENSIVE TEST")
    logger.info("=" * 60)

    t1 = time.time()
    test_result = run_tests()
    t1 = time.time() - t1

    logger.info("\n>>> Benchmarks")
    t2 = time.time()
    benchmarks = run_benchmarks()
    t2 = time.time() - t2

    report = {
        "test_results": test_result,
        "benchmarks": benchmarks,
    }

    report_path = "/tmp/intervention_coverage_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Tests: %d passed, %d failed, %d errors, %d skipped",
                test_result["tests_run"] - test_result["failures"] - test_result["errors"],
                test_result["failures"], test_result["errors"], test_result["skipped"])
    logger.info("  Test time: %.2fs", t1)
    logger.info("  Benchmark time: %.2fs", t2)

    logger.info("\n  -- Benchmark scenarios --")
    for name, bm in benchmarks.items():
        res = bm.get("result", {})
        cfg = bm.get("config", {})
        logger.info("  %s:", bm.get("scenario", name))
        logger.info("    Config: %s", cfg)
        logger.info("    Converged: %s at iter %s%s",
                    res.get("converged", "N/A"),
                    res.get("converged_at_iteration", "-"),
                    f" ({res['converged_reason'][:60]})"
                    if res.get("converged_reason") else "")
        logger.info("    Triggers: %d  Failures: %d  Consumed: %d  Remaining: %d",
                    res.get("triggers", 0), res.get("failures", 0),
                    res.get("prompts_consumed", 0), res.get("prompts_remaining", 0))

    logger.info("\n  Report: %s", report_path)
    logger.info("=" * 60)
    logger.info("OVERALL: %s", "PASSED" if test_result["passed"] else "FAILED")
